from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1] / "quiz-rag-app"
sys.path.insert(0, str(APP_DIR))

import pytest

from rag_helpers import (
    build_page_chunks,
    format_retrieved_context,
    validate_pdf_bytes,
)


class FakeSplitter:
    def split_text(self, text):
        return [text[:4], text[4:]] if len(text) > 4 else [text]


def test_pdf_validation_rejects_empty_and_non_pdf_bytes():
    with pytest.raises(ValueError, match="empty"):
        validate_pdf_bytes(b"")
    with pytest.raises(ValueError, match="not a valid PDF"):
        validate_pdf_bytes(b"not a pdf")


def test_pdf_validation_rejects_oversized_upload():
    with pytest.raises(ValueError, match="too large"):
        validate_pdf_bytes(b"%PDF-123", max_bytes=5)


def test_pdf_validation_accepts_pdf_header():
    validate_pdf_bytes(b"%PDF-1.7 sample")


def test_page_chunking_preserves_original_page_numbers():
    chunks, metadata = build_page_chunks(
        [(2, "abcdefgh"), (7, "xyz")], FakeSplitter()
    )
    assert chunks == ["abcd", "efgh", "xyz"]
    assert [item["page_number"] for item in metadata] == [2, 2, 7]
    assert [item["chunk_index"] for item in metadata] == [0, 1, 2]


def test_context_contains_source_page_labels_and_sorted_sources():
    context, pages = format_retrieved_context(
        ["first evidence", "second evidence"],
        [{"page_number": 9, "chunk_index": 0}, {"page_number": 3, "chunk_index": 1}],
        [0.2, 0.4],
        max_distance=0.65,
    )
    assert "[SOURCE: PDF page 9 | Chunk 1]" in context
    assert "[SOURCE: PDF page 3 | Chunk 2]" in context
    assert pages == [3, 9]


def test_context_filters_weak_matches_and_missing_page_metadata():
    context, pages = format_retrieved_context(
        ["weak", "missing page"],
        [{"page_number": 1, "chunk_index": 0}, {"chunk_index": 1}],
        [0.9, 0.1],
        max_distance=0.65,
    )
    assert context == ""
    assert pages == []
