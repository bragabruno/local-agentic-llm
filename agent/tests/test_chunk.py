import pytest

from agent.rag.chunk import chunk_text


def test_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_single_short_paragraph() -> None:
    chunks = chunk_text("hello world")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].index == 0


def test_respects_max_chars() -> None:
    text = "\n\n".join(f"paragraph number {i} " * 10 for i in range(20))
    chunks = chunk_text(text, max_chars=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_hard_splits_oversized_paragraph() -> None:
    chunks = chunk_text("x" * 5000, max_chars=1000, overlap=100)
    assert len(chunks) >= 5
    assert all(len(c.text) <= 1000 for c in chunks)


def test_invalid_params() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("abc", max_chars=100, overlap=100)
