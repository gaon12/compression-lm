"""Tests for stdlib one-shot backends: bzip2 and the lzma family."""

import bz2
import lzma

import pytest

from compression_lm import backends

CONTEXT = b"all the world's a stage, and all the men and women merely players. " * 4
CANDIDATE_REPEAT = b"all the world's a stage"
CANDIDATE_NOVEL = b"\x89\xab\xcd\xef wholly unseen bytes \x01\x02\x03"

STDLIB_NAMES = ["bzip2", "xz", "lzma", "lzma2"]


def test_registry_lists_stdlib_backends():
    assert set(STDLIB_NAMES) <= set(backends.names())


@pytest.mark.parametrize("name", STDLIB_NAMES)
def test_stdlib_scorer_delta_semantics(name):
    """measure() must equal len(comp(ctx+c)) - len(comp(ctx)) exactly."""
    scorer = backends.get(name).prepare(CONTEXT)
    comp = {
        "bzip2": lambda d: bz2.compress(d, compresslevel=9),
        "xz": lambda d: lzma.compress(d, format=lzma.FORMAT_XZ, preset=6),
        "lzma": lambda d: lzma.compress(d, format=lzma.FORMAT_ALONE, preset=6),
        "lzma2": lambda d: lzma.compress(
            d, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 6}]
        ),
    }[name]
    for cand in (CANDIDATE_REPEAT, CANDIDATE_NOVEL, b""):
        assert scorer.measure(cand) == len(comp(CONTEXT + cand)) - len(comp(CONTEXT))


@pytest.mark.parametrize("name", STDLIB_NAMES)
def test_repeated_substring_scores_below_novel(name):
    scorer = backends.get(name).prepare(CONTEXT)
    assert scorer.measure(CANDIDATE_REPEAT) < scorer.measure(CANDIDATE_NOVEL)


@pytest.mark.parametrize("name", STDLIB_NAMES)
def test_stream_roundtrip(name):
    """The compress function inside each scorer must produce valid streams."""
    scorer = backends.get(name).prepare(CONTEXT)
    assert scorer.context == CONTEXT
    payload = scorer._compress(CONTEXT + CANDIDATE_REPEAT)  # OneShotScorer internals
    if name == "bzip2":
        assert bz2.decompress(payload) == CONTEXT + CANDIDATE_REPEAT
    elif name == "lzma2":
        out = lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2}])
        assert out == CONTEXT + CANDIDATE_REPEAT
    else:
        assert lzma.decompress(payload) == CONTEXT + CANDIDATE_REPEAT
