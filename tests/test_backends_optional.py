"""Tests for optional-dependency backends (zstd, brotli, lz4, snappy, lzo)."""

import pytest

from compression_lm import backends

CONTEXT = b"the quality of mercy is not strained; it droppeth as the gentle rain " * 4
CANDIDATE_REPEAT = b"the quality of mercy is not strained"
CANDIDATE_NOVEL = b"\xde\xad\xbe\xef totally alien bytes \xf0\x9f\x98\x80"

OPTIONAL_NAMES = ["zstd", "brotli", "lz4", "snappy", "lzo"]


def _available(name: str) -> bool:
    return backends.info(name).available


def test_registry_lists_optional_backends():
    assert set(OPTIONAL_NAMES) <= set(backends.names())


def test_unavailable_backend_raises_runtime_error():
    for name in OPTIONAL_NAMES:
        if not _available(name):
            with pytest.raises(RuntimeError, match="unavailable"):
                backends.get(name)
            assert backends.info(name).unavailable_reason
            return
    pytest.skip("all optional backends installed")


@pytest.mark.parametrize("name", OPTIONAL_NAMES)
def test_optional_scorer_basics(name):
    if not _available(name):
        pytest.skip(f"{name} not installed")
    scorer = backends.get(name).prepare(CONTEXT)
    assert scorer.context == CONTEXT
    # A corpus substring must cost fewer bytes than alien input.
    assert scorer.measure(CANDIDATE_REPEAT) < scorer.measure(CANDIDATE_NOVEL)


@pytest.mark.parametrize("name", OPTIONAL_NAMES)
def test_optional_empty_context(name):
    if not _available(name):
        pytest.skip(f"{name} not installed")
    scorer = backends.get(name).prepare(b"")
    assert isinstance(scorer.measure(b"hello"), int)
