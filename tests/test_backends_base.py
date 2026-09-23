"""Tests for the backend abstraction, registry, and the zlib scorer."""

import zlib

import pytest

from compression_lm import backends
from compression_lm.backends.base import OneShotScorer

CONTEXT = b"to be or not to be, that is the question. " * 8
CANDIDATE_REPEAT = b"to be or not to be"
CANDIDATE_NOVEL = b"\x00\xfe\xdc\xba\x98randomnovelbytes"

_WBITS = {"deflate": -15, "zlib": 15, "gzip": 31}


def _full_len(name: str, data: bytes) -> int:
    enc = zlib.compressobj(9, zlib.DEFLATED, _WBITS[name])
    return len(enc.compress(data) + enc.flush(zlib.Z_FINISH))


def test_registry_lists_zlib_backends():
    assert {"deflate", "zlib", "gzip"} <= set(backends.names())


def test_get_unknown_backend_raises():
    with pytest.raises(KeyError, match="unknown backend"):
        backends.get("nope")


def test_backend_info_shape():
    i = backends.info("deflate")
    assert i.available
    assert i.package == "stdlib"
    assert i.context_limit == 32768
    assert i.unavailable_reason is None


@pytest.mark.parametrize("name", ["deflate", "zlib", "gzip"])
def test_zlib_scorer_matches_full_recompression(name):
    """Cloned-encoder deltas must equal full-recompression deltas."""
    scorer = backends.get(name).prepare(CONTEXT)
    empty = scorer.measure(b"")
    for cand in (CANDIDATE_REPEAT, CANDIDATE_NOVEL, b"x"):
        true_delta = _full_len(name, CONTEXT + cand) - _full_len(name, CONTEXT)
        assert scorer.measure(cand) - empty == true_delta


@pytest.mark.parametrize("name", ["deflate", "zlib", "gzip"])
def test_repeated_substring_scores_below_novel(name):
    scorer = backends.get(name).prepare(CONTEXT)
    assert scorer.measure(CANDIDATE_REPEAT) < scorer.measure(CANDIDATE_NOVEL)


def test_one_shot_scorer_delta_semantics():
    scorer = OneShotScorer(lambda b: zlib.compress(b, 9), CONTEXT)
    assert scorer.measure(b"") == 0
    for cand in (CANDIDATE_REPEAT, CANDIDATE_NOVEL):
        expected = len(zlib.compress(CONTEXT + cand, 9)) - len(zlib.compress(CONTEXT, 9))
        assert scorer.measure(cand) == expected


def test_scorer_measure_many_matches_measure():
    scorer = backends.get("deflate").prepare(CONTEXT)
    cands = [CANDIDATE_REPEAT, CANDIDATE_NOVEL, b"x"]
    assert scorer.measure_many(cands) == [scorer.measure(c) for c in cands]
