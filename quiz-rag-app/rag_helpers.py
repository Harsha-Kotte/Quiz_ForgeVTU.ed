"""Small, dependency-free helpers for source-aware PDF retrieval."""

from __future__ import annotations

import re
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


_QUESTION_HEADER_RE = re.compile(
    r"(?im)^\\s*(?:[-*]\\s*)?(?:\\*{0,2})(Q\\s*\\d+\\b|\\[CHALLENGE[_ -]\\d+\\])"
)
_SOURCE_CITATION_RE = re.compile(
    r"\\bSource:\\s*PDF\\s+page(?:\\(s\\)|s)?\\s*([0-9]+(?:\\s*,\\s*[0-9]+)*)\\s*\\.?",
    re.IGNORECASE,
)


def validate_source_citations(
    generated_text: str, allowed_pages: Iterable[int]
) -> dict[str, object]:
    """Check that detected questions cite pages included in retrieved evidence.

    This validates citation presence and page membership only. It cannot prove
    that a cited passage semantically supports the generated question.
    """
    if not isinstance(generated_text, str):
        raise TypeError("generated_text must be a string.")

    allowed = {int(page) for page in allowed_pages}
    question_matches = list(_QUESTION_HEADER_RE.finditer(generated_text))
    uncited_questions: list[str] = []
    cited_pages: set[int] = set()
    unknown_pages: set[int] = set()

    for index, question in enumerate(question_matches):
        block_end = (
            question_matches[index + 1].start()
            if index + 1 < len(question_matches)
            else len(generated_text)
        )
        question_block = generated_text[question.start() : block_end]
        citations = list(_SOURCE_CITATION_RE.finditer(question_block))
        label = question.group(1).strip()

        if not citations:
            uncited_questions.append(label)

        for citation in citations:
            pages = [int(value) for value in re.findall(r"\\d+", citation.group(1))]
            cited_pages.update(pages)
            unknown_pages.update(page for page in pages if page not in allowed)

    # Still collect page membership when the output uses an unrecognised
    # question heading, but report that per-question coverage was not checked.
    if not question_matches:
        for citation in _SOURCE_CITATION_RE.finditer(generated_text):
            pages = [int(value) for value in re.findall(r"\\d+", citation.group(1))]
            cited_pages.update(pages)
            unknown_pages.update(page for page in pages if page not in allowed)

    return {
        "question_count": len(question_matches),
        "uncited_questions": uncited_questions,
        "cited_pages": sorted(cited_pages),
        "unknown_pages": sorted(unknown_pages),
        "question_format_detected": bool(question_matches),
    }
