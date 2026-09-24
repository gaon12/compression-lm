"""Teacher-forced next-byte benchmark.

The compressor never generates here. For each held-out test position we
score all 256 possible next bytes::

    raw_score(b) = len(compress(context + b)) - len(compress(context))
    delta_bits(b) = 8 * raw_score(b)

and ask where the *true* next byte ranks. Because compressed sizes are
byte-granular, scores tie constantly; ranks use the **average-rank**
convention: if k candidates share the same score, each is assigned the
mean of the ranks they span (ten candidates tied at rank 1 -> all get
rank 5.5). Top-k uses the same average rank (hit iff avg_rank <= k), so
the convention is consistent across metrics.

The score-to-probability mapping is a *pseudo* probability::

    p(b | ctx) = 2**(-delta_bits(b)) / sum(2**(-delta_bits(all)))

computed with a min-shift for numerical stability. Its mean NLL is
reported as ``teacher_forced_pseudo_bpb`` — explicitly not a true LM
perplexity and never confused with ``generated_compression_bpb``.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..backends.base import Backend, Scorer

ALL_BYTES: tuple[bytes, ...] = tuple(bytes([b]) for b in range(256))


@dataclass
class PositionRecord:
    """Per-position detail kept for quantization analysis and debugging."""

    position: int
    true_byte: int
    true_score: int  # raw_score in compressed bytes
    rank_best: int  # optimistic rank: better-scoring candidates + 1
    rank_avg: float  # average-rank under ties (primary convention)
    nll_bits: float
    unique_scores: int
    max_tie: int  # size of the largest same-score group
    in_best_tie: bool  # true byte shares the best score with others/alone
    n_candidates: int
    prep_seconds: float
    score_seconds: float


def rank_and_nll(scores: Sequence[int], true_index: int) -> tuple[int, float, float]:
    """Return (rank_best, rank_avg, nll_bits) of the true candidate.

    rank_avg implements the average-rank tie convention; nll_bits is the
    pseudo-probability NLL in bits with min-shift stabilization.
    """
    s_true = scores[true_index]
    better = sum(1 for s in scores if s < s_true)
    tied = sum(1 for s in scores if s == s_true)
    rank_best = better + 1
    rank_avg = better + (tied + 1) / 2
    m = min(scores)
    # scores are in bytes; weights are 2^(-8*s). Min-shifted:
    # p_y = 2^(8(m - s_y)) / sum(2^(8(m - s_i)))
    log_z = math.log2(sum(2.0 ** (8.0 * (m - s)) for s in scores))
    nll_bits = 8.0 * (s_true - m) + log_z
    return rank_best, rank_avg, nll_bits


def score_position(
    scorer: Scorer,
    candidates: Sequence[bytes] = ALL_BYTES,
    pool: ThreadPoolExecutor | None = None,
) -> list[int]:
    """Score every candidate byte against one prepared context."""
    if pool is not None:
        return list(pool.map(scorer.measure, candidates))
    return scorer.measure_many(candidates)


def evaluate_positions(
    backend: Backend,
    source: bytes,
    positions: Sequence[int],
    context_bytes: int,
    *,
    workers: int = 1,
    candidates: Sequence[bytes] = ALL_BYTES,
    backend_options: dict | None = None,
    on_record=None,
) -> list[PositionRecord]:
    """Run the teacher-forced evaluation over ``positions``.

    ``context_bytes`` is the *provided* context; the scorer is fed at most
    ``backend.context_limit`` bytes (the effective context — recorded by
    the caller). ``source`` must contain every position; the byte at
    ``source[p]`` is the held-out truth.
    """
    opts = dict(backend_options or {})
    eff = min(context_bytes, backend.context_limit)
    pool = ThreadPoolExecutor(workers) if workers > 1 else None
    records: list[PositionRecord] = []
    try:
        for p in positions:
            context = source[max(0, p - eff) : p]
            t0 = time.perf_counter()
            scorer = backend.prepare(context, **opts)
            t1 = time.perf_counter()
            scores = score_position(scorer, candidates, pool)
            t2 = time.perf_counter()
            y = source[p]
            s_true = scores[y]
            rank_best, rank_avg, nll = rank_and_nll(scores, y)
            counts: dict[int, int] = {}
            for s in scores:
                counts[s] = counts.get(s, 0) + 1
            rec = PositionRecord(
                position=p,
                true_byte=y,
                true_score=s_true,
                rank_best=rank_best,
                rank_avg=rank_avg,
                nll_bits=nll,
                unique_scores=len(counts),
                max_tie=max(counts.values()),
                in_best_tie=s_true == min(scores),
                n_candidates=len(candidates),
                prep_seconds=t1 - t0,
                score_seconds=t2 - t1,
            )
            records.append(rec)
            if on_record is not None:
                on_record(rec)
    finally:
        if pool is not None:
            pool.shutdown(wait=False)
    return records


def aggregate_records(
    records: Sequence[PositionRecord],
    *,
    backend: str,
    family: str,
    mode: str,
    context_provided: int,
    context_effective: int,
    note: str = "",
) -> dict:
    """Aggregate position records into one teacher_forced.csv row."""
    n = len(records)
    if n == 0:
        return {
            "backend": backend,
            "family": family,
            "mode": mode,
            "context_provided": context_provided,
            "context_effective": context_effective,
            "n_positions": 0,
            "note": note or "no positions evaluated",
        }
    ranks = [r.rank_avg for r in records]
    evals = sum(r.n_candidates for r in records)
    score_s = sum(r.score_seconds for r in records)
    prep_s = sum(r.prep_seconds for r in records)
    total_s = score_s + prep_s
    nll = [r.nll_bits for r in records]
    tied_counts = [r.n_candidates - r.unique_scores for r in records]
    row = {
        "backend": backend,
        "family": family,
        "mode": mode,
        "context_provided": context_provided,
        "context_effective": context_effective,
        "n_positions": n,
        "top1": sum(1 for r in ranks if r <= 1.0) / n,
        "top5": sum(1 for r in ranks if r <= 5.0) / n,
        "top10": sum(1 for r in ranks if r <= 10.0) / n,
        "mean_rank": statistics.fmean(ranks),
        "median_rank": statistics.median(ranks),
        "mrr": statistics.fmean(1.0 / r for r in ranks),
        "mean_true_delta_bits": 8.0 * statistics.fmean(r.true_score for r in records),
        "teacher_forced_pseudo_bpb": statistics.fmean(nll),
        "pseudo_ppl": 2.0 ** statistics.fmean(nll),
        # quantization / tie analysis
        "tie_candidate_frac": statistics.fmean(tied_counts) / records[0].n_candidates,
        # true byte shares its score with >=1 other candidate
        "pct_true_tied": sum(1 for r in records if r.rank_best != r.rank_avg) / n,
        "pct_true_in_best_tie": sum(1 for r in records if r.in_best_tie) / n,
        "mean_unique_scores": statistics.fmean(r.unique_scores for r in records),
        "mean_max_tie": statistics.fmean(r.max_tie for r in records),
        # throughput
        "evals": evals,
        "evals_per_sec": evals / score_s if score_s else 0.0,
        "positions_per_sec": n / total_s if total_s else 0.0,
        "prep_seconds": prep_s,
        "score_seconds": score_s,
        "seconds": total_s,
        "note": note,
    }
    return row
