"""Tests for the teacher-forced benchmark math."""

import math

import pytest

from compression_lm import backends
from compression_lm.bench.teacher import (
    _expected_topk,
    aggregate_records,
    evaluate_positions,
    rank_and_nll,
)


def test_expected_topk_boundary_cases():
    # strictly worse than top-k
    assert _expected_topk(better=10, tie=5, k=5) == 0.0
    # tie group entirely inside top-k
    assert _expected_topk(better=0, tie=4, k=5) == 1.0
    # straddling: true byte's score class is ranks 1..100, k=1 -> 1/100
    assert _expected_topk(better=0, tie=100, k=1) == pytest.approx(0.01)
    # ranks 3..12 straddle k=10 -> 8 of 10 members covered
    assert _expected_topk(better=2, tie=10, k=10) == pytest.approx(0.8)


def test_rank_and_nll_unique_best():
    # true byte strictly best
    rb, ra, _ = rank_and_nll([10, 3, 7, 9], 1)
    assert rb == 1 and ra == 1.0


def test_rank_average_convention():
    # scores: 1,1,1,5 — three tied at best; true byte is one of them
    rb, ra, _ = rank_and_nll([1, 1, 1, 5], 2)
    assert rb == 1
    assert ra == (1 + 2 + 3) / 3  # average of spanned ranks = 2.0


def test_rank_average_with_better_candidates():
    # scores: 0, 3, 3, 9 — true byte 3 tied at ranks 2-3 -> avg 2.5
    rb, ra, _ = rank_and_nll([0, 3, 3, 9], 1)
    assert rb == 2 and ra == 2.5


def test_nll_matches_pseudo_probability():
    scores = [0, 1, 2, 3]
    _, _, nll = rank_and_nll(scores, 0)
    # p_y = 2^0 / sum(2^-s) — compute expected directly
    weights = [2.0 ** (-8 * s) for s in scores]
    p = weights[0] / sum(weights)
    assert nll == pytest.approx(-math.log2(p), abs=1e-9)


def test_nll_nonnegative_and_ordered():
    scores = [5, 5, 0, 9]
    _, _, nll_best = rank_and_nll(scores, 2)
    _, _, nll_worst = rank_and_nll(scores, 3)
    assert 0 <= nll_best < nll_worst


def test_evaluate_positions_end_to_end():
    corpus = b"to be or not to be, that is the question. " * 40
    backend = backends.get("deflate")
    positions = [len(corpus) - 100 + i for i in range(5)]
    recs = evaluate_positions(backend, corpus, positions, 512)
    assert len(recs) == 5
    for r in recs:
        assert r.n_candidates == 256
        assert 1 <= r.rank_best <= 256
        assert r.rank_best <= r.rank_avg
        assert r.unique_scores >= 1
        assert r.max_tie >= 1


def test_evaluate_positions_predicts_periodic_text():
    """On periodic text the true next byte should usually be top-ranked."""
    corpus = b"abcd" * 400
    backend = backends.get("deflate")
    positions = list(range(800, 900))
    recs = evaluate_positions(backend, corpus, positions, 512)
    row = aggregate_records(
        recs,
        backend="deflate",
        family="deflate",
        mode="quick",
        context_provided=512,
        context_effective=512,
    )
    assert row["top1"] > 0.9
    assert row["teacher_forced_pseudo_bpb"] < 1.0
    assert row["n_positions"] == 100


def test_aggregate_empty():
    row = aggregate_records(
        [],
        backend="x",
        family="f",
        mode="m",
        context_provided=1,
        context_effective=1,
    )
    assert row["n_positions"] == 0


def test_quantization_fields_present():
    corpus = b"some varied text here " * 50
    backend = backends.get("deflate")
    recs = evaluate_positions(backend, corpus, [700, 701, 702], 256)
    row = aggregate_records(
        recs,
        backend="deflate",
        family="deflate",
        mode="quick",
        context_provided=256,
        context_effective=256,
    )
    for key in (
        "tie_candidate_frac",
        "pct_true_tied",
        "pct_true_in_best_tie",
        "mean_unique_scores",
        "mean_max_tie",
        "top1_exp",
        "top5_exp",
        "top10_exp",
    ):
        assert key in row
    assert 0.0 <= row["tie_candidate_frac"] <= 1.0
    # expected top-k must be at least as large as deterministic avg-rank top-k
    assert row["top1_exp"] >= row["top1"]
    assert row["top1_exp"] <= row["top5_exp"] <= row["top10_exp"]
