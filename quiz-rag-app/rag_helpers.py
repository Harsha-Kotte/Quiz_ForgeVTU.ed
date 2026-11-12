"""Small, dependency-free helpers for source-aware PDF retrieval."""

from __future__ import annotations

from typing import Iterable, Sequence

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 500
DEFAULT_MAX_COSINE_DISTANCE = 0.65


def validate_pdf_bytes(data: bytes, max_bytes: int = MAX_PDF_BYTES) -> None:
    """Reject obviously invalid, empty, or oversized PDF uploads."""
    if not isinstance(data, bytes) or not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > max_bytes:
        limit_mb = max_bytes // (1024 * 1024)
        raise ValueError(f"PDF is too large. Maximum supported size is {limit_mb} MB.")
    if not data.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF (missing PDF header).")


def build_page_chunks(
    pages: Sequence[tuple[int, str]], splitter
) -> tuple[list[str], list[dict[str, int]]]:
    """Split each page independently so every chunk retains its original page number."""
    chunks: list[str] = []
    metadatas: list[dict[str, int]] = []

    for page_number, page_text in pages:
        for chunk in splitter.split_text(page_text):
            if chunk.strip():
                chunks.append(chunk)
                metadatas.append(
                    {"chunk_index": len(chunks) - 1, "page_number": int(page_number)}
                )

    return chunks, metadatas


def format_retrieved_context(
    documents: Iterable[str],
    metadatas: Iterable[dict],
    distances: Iterable[float],
    max_distance: float = DEFAULT_MAX_COSINE_DISTANCE,
) -> tuple[str, list[int]]:
    """Filter weak cosine matches and annotate accepted chunks with PDF page sources."""
    accepted: list[str] = []
    source_pages: set[int] = set()

    for document, metadata, distance in zip(documents, metadatas, distances):
        if distance is None or float(distance) > max_distance:
            continue
        page_number = metadata.get("page_number")
        if page_number is None:
            continue
        page_number = int(page_number)
        source_pages.add(page_number)
        chunk_index = int(metadata.get("chunk_index", 0))
        accepted.append(
            f"[SOURCE: PDF page {page_number} | Chunk {chunk_index + 1}]\n{document}"
        )

    if not accepted:
        return "", []

    return "\n\n---\n\n".join(accepted), sorted(source_pages)
