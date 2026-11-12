"""
Source-aware PDF ingestion and semantic retrieval for QuizForge.

Page numbers are preserved from the original PDF (1-based). The vector store
is intentionally ephemeral and is rebuilt when a new document is ingested.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import uuid
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from rag_helpers import (
    DEFAULT_MAX_COSINE_DISTANCE,
    MAX_PDF_PAGES,
    build_page_chunks,
    format_retrieved_context,
    validate_pdf_bytes,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "student_notes"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBED_BATCH_SIZE = 16
TOP_K_DEFAULT = 6
MAX_COSINE_DISTANCE = float(
    os.getenv("QUIZFORGE_MAX_COSINE_DISTANCE", str(DEFAULT_MAX_COSINE_DISTANCE))
)

_embedding_model: SentenceTransformer | None = None
_chroma_client: chromadb.EphemeralClient | None = None
_collection: chromadb.Collection | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        log.info("Loading embedding model '%s' on CPU", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _embedding_model


def _get_chroma_client() -> chromadb.EphemeralClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_client


def _reset_collection() -> chromadb.Collection:
    global _collection
    client = _get_chroma_client()
    if COLLECTION_NAME in [item.name for item in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    _collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    return _collection


def _get_collection() -> chromadb.Collection:
    if _collection is None:
        raise RuntimeError("No document is indexed. Call ingest_pdf() first.")
    return _collection


def parse_pdf_pages(pdf_source: str | Path | bytes) -> list[tuple[int, str]]:
    """Extract non-empty page text as (original 1-based page number, text)."""
    if isinstance(pdf_source, (str, Path)):
        pdf_bytes = Path(pdf_source).read_bytes()
    elif isinstance(pdf_source, bytes):
        pdf_bytes = pdf_source
    else:
        raise TypeError("PDF source must be a file path or bytes.")

    validate_pdf_bytes(pdf_bytes)
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:
                raise ValueError("This PDF is password-protected and cannot be read.") from exc
            if not decrypted:
                raise ValueError("This PDF is password-protected and cannot be read.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ValueError(f"PDF has too many pages. Maximum supported is {MAX_PDF_PAGES}.")
        pages: list[tuple[int, str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").encode(
                "utf-8", errors="ignore"
            ).decode("utf-8").strip()
            if text:
                pages.append((page_number, text))
        return pages
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Could not read this PDF. It may be damaged or unsupported.") from exc


def parse_pdf(pdf_source: str | Path | bytes) -> str:
    """Backward-compatible convenience function returning all extracted text."""
    return "\n\n".join(text for _, text in parse_pdf_pages(pdf_source))


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def split_pages(pages: list[tuple[int, str]]) -> tuple[list[str], list[dict[str, int]]]:
    chunks, metadatas = build_page_chunks(pages, _make_splitter())
    log.info("Split %d page(s) into %d source-aware chunks.", len(pages), len(chunks))
    return chunks, metadatas


def split_text(text: str) -> List[str]:
    """Legacy helper retained for callers that only need plain text chunks."""
    return _make_splitter().split_text(text)


def _embed_in_batches(texts: List[str]) -> List[List[float]]:
    model = _get_embedding_model()
    embeddings: List[List[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors = model.encode(
            texts[start : start + EMBED_BATCH_SIZE],
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings.extend(vectors.tolist())
        gc.collect()
    return embeddings


def store_chunks(
    chunks: List[str], metadatas: list[dict[str, int]] | None = None
) -> int:
    if not chunks:
        raise ValueError("No readable text chunks were found in this PDF.")
    if metadatas is None:
        metadatas = [
            {"chunk_index": index, "page_number": 1}
            for index in range(len(chunks))
        ]
    if len(chunks) != len(metadatas):
        raise ValueError("Each chunk must have exactly one metadata record.")

    collection = _reset_collection()
    embeddings = _embed_in_batches(chunks)
    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    log.info("Stored %d chunks in ChromaDB.", len(chunks))
    return len(chunks)


def ingest_pdf(pdf_source: str | Path | bytes) -> int:
    """Validate, extract page-by-page, chunk, embed, and index a PDF."""
    pages = parse_pdf_pages(pdf_source)
    if not pages:
        raise ValueError(
            "The PDF contains no extractable text. Scanned/image-only PDFs need OCR, "
            "which is not supported yet."
        )
    chunks, metadatas = split_pages(pages)
    return store_chunks(chunks, metadatas)


def retrieve_context(
    query: str, top_k: int = TOP_K_DEFAULT, return_sources: bool = False
) -> str | tuple[str, list[int]]:
    """Return only sufficiently relevant chunks, each labelled with its PDF page."""
    if not query or not query.strip():
        raise ValueError("Retrieval query cannot be empty.")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return ("", []) if return_sources else ""

    model = _get_embedding_model()
    query_vector = model.encode(
        [query.strip()],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    context, source_pages = format_retrieved_context(
        documents, metadatas, distances, MAX_COSINE_DISTANCE
    )
    log.info(
        "Accepted %d relevant chunks from pages %s for query '%s'",
        len(source_pages), source_pages, query[:60]
    )
    if not context:
        log.info("No chunks passed cosine-distance threshold %.3f.", MAX_COSINE_DISTANCE)
    return (context, source_pages) if return_sources else context


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_engine.py <path/to/test.pdf>")
        raise SystemExit(1)
    total = ingest_pdf(sys.argv[1])
    print(f"Ingested {total} source-aware chunks.")
    print(retrieve_context("Summarise the main concepts.", top_k=3))
