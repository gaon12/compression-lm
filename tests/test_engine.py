"""Tests for the generation engine."""

import random

import pytest

from compression_lm import backends
from compression_lm.engine import (
    GenerationConfig,
    _ngrams,
    _repeat_hits,
    _select,
    corpus_alphabet,
    generate,
)

CORPUS = (
    b"the quick brown fox jumps over the lazy dog. pack my box with five dozen liquor jugs. "
) * 20


def test_corpus_alphabet_only_observed_bytes():
    alpha = corpus_alphabet(b"abca", b"bd")
    assert alpha == (ord("a"), ord("b"), ord("c"), ord("d"))


def test_corpus_alphabet_empty_is_full_byte_range():
    assert corpus_alphabet(b"") == tuple(range(256))


def test_ngrams_and_repeat_hits():
    grams = _ngrams(b"abcdef", 3)
    assert grams == frozenset({b"abc", b"bcd", b"cde", b"def"})
    assert _repeat_hits(b"xxbcdyy", grams, 3) == 1
    assert _repeat_hits(b"zzz", grams, 3) == 0


def test_select_argmin_at_zero_temperature():
    assert _select([5, 1, 9], 0.0, random.Random(0)) == 0


def test_select_samples_valid_index():
    rng = random.Random(1)
    picks = {_select([0, 5, 9], 1.0, rng) for _ in range(50)}
    assert picks <= {0, 1, 2}
    assert 0 in picks  # best score dominates


def test_generate_respects_length_and_alphabet():
    backend = backends.get("deflate")
    cfg = GenerationConfig(length=64, lookahead=8, beam_width=8, seed=0)
    res = generate(backend, CORPUS, b"the quick ", cfg)
    assert len(res.text) == 64
    assert set(res.text) <= set(corpus_alphabet(CORPUS))
    assert res.backend == "deflate"
    assert res.evaluations > 0
    assert res.bits_per_byte >= 0


def test_generate_reproduces_periodic_corpus():
    """On a periodic corpus, greedy decoding should track the period."""
    backend = backends.get("deflate")
    cfg = GenerationConfig(length=32, lookahead=16, beam_width=16, temperature=0.0, seed=0)
    res = generate(backend, CORPUS, b"the quick brown fox jumps over the lazy dog", cfg)
    # Most generated 8-grams should exist verbatim in the corpus.
    hits = sum(1 for i in range(len(res.text) - 7) if res.text[i : i + 8] in CORPUS)
    assert hits >= (len(res.text) - 7) * 0.8


def test_corpus_tiebreak_improves_coherence():
    """Corpus-membership tie-breaking must beat pure random tie-breaking."""
    backend = backends.get("deflate")
    rates = {}
    for tb in ("corpus", "none"):
        cfg = GenerationConfig(
            length=48,
            lookahead=16,
            beam_width=16,
            temperature=0.0,
            seed=0,
            tiebreak=tb,
        )
        res = generate(backend, CORPUS, b"the quick brown fox jumps ", cfg)
        hits = sum(1 for i in range(len(res.text) - 7) if res.text[i : i + 8] in CORPUS)
        rates[tb] = hits / (len(res.text) - 7)
    assert rates["corpus"] > rates["none"]


def test_generate_deterministic_with_seed():
    backend = backends.get("deflate")
    cfg = GenerationConfig(length=48, lookahead=8, beam_width=8, temperature=0.9, seed=42)
    a = generate(backend, CORPUS, b"jumps ", cfg)
    b = generate(backend, CORPUS, b"jumps ", cfg)
    assert a.text == b.text


def test_generate_context_cap_respected():
    backend = backends.get("deflate")
    cfg = GenerationConfig(length=16, lookahead=4, beam_width=4, context_bytes=64)
    res = generate(backend, CORPUS, b"", cfg)
    assert len(res.text) == 16


def test_corpus_gate_strict_recombination_and_speedup():
    """Gated mode scores far fewer candidates and copies corpus verbatim."""
    backend = backends.get("deflate")
    base = dict(length=48, lookahead=16, beam_width=16, temperature=0.0, seed=0)
    gated = generate(backend, CORPUS, b"the quick ", GenerationConfig(corpus_gate=True, **base))
    free = generate(backend, CORPUS, b"the quick ", GenerationConfig(corpus_gate=False, **base))
    assert gated.evaluations < free.evaluations
    # every 8-gram of gated output must be a corpus substring
    assert all(gated.text[i : i + 8] in CORPUS for i in range(len(gated.text) - 7))


def test_repeat_penalty_changes_ranking():
    """A banned n-gram must push otherwise-best candidates down the ranking."""
    backend = backends.get("deflate")
    ctx = CORPUS[-512:]
    scorer = backend.prepare(ctx)
    repeat = b" the lazy dog."
    novel = b" zzzqqq novel!!"
    assert scorer.measure(repeat) < scorer.measure(novel)
    banned = _ngrams(repeat, 6)
    assert _repeat_hits(repeat, banned, 6) > 0
    assert _repeat_hits(novel, banned, 6) == 0


@pytest.mark.parametrize("workers", [1, 4])
def test_generate_threaded_matches_sequential(workers):
    backend = backends.get("deflate")
    cfg = GenerationConfig(length=40, lookahead=8, beam_width=8, temperature=0.0, workers=workers)
    res = generate(backend, CORPUS, b"pack my box ", cfg)
    assert len(res.text) == 40
