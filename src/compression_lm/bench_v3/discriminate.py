"""Multi-byte continuation discrimination — the v3 core experiment.

For each held-out position the scorer ranks the *true* N-byte continuation
against 31 negatives of the same length (32-way). This answers whether a
compressor whose 1-byte scores are fully tied still provides sequence-level
signal at 4/8/16/32 bytes.

Negative types:

- ``random``: N-byte spans drawn uniformly from the same dataset source
  (any split — the scorer never sees the answer's identity, only bytes).
- ``hard``: spans sharing the true continuation's first 1-2 bytes but
  differing afterwards. If fewer than 31 exist the remainder falls back to
  random negatives and the row records the mix.

Scores use the same ``scorer.measure(candidate)`` conditional code length
as the 1-byte benchmark — the discrimination task only changes candidate
length and count.
"""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..backends.base import Backend
from ..bench.teacher import _expected_topk, score_position

CONTINUATION_LENGTHS: tuple[int, ...] = (1, 4, 8, 16, 32)
SNAPPY_LENGTHS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
N_CANDIDATES = 32  # 1 true + 31 negatives


@dataclass
class DiscRecord:
    """Per-position discrimination outcome."""

    position: int
    true_rank_avg: float
    better_count: int
    true_tie_size: int
    unique_scores: int
    max_tie: int
    all_tied: bool
    n_candidates: int
    n_hard_negatives: int
    score_margin: float  # min(neg scores) - true score; >0 means true wins outright
    hard_neg_win: float | None  # fraction of hard negatives outscored (ties win)
    rand_neg_win: float | None  # fraction of random negatives outscored
    prep_seconds: float
    score_seconds: float


def _random_negatives(
    source: bytes, n: int, count: int, rng: random.Random, forbid: tuple[int, int]
) -> list[bytes]:
    """Uniform random N-byte spans, excluding the true span's start."""
    hi = len(source) - n
    if hi <= 0:
        return []
    out: list[bytes] = []
    seen = {forbid[0]}
    tries = 0
    while len(out) < count and tries < count * 40:
        tries += 1
        i = rng.randrange(0, hi + 1)
        if i in seen:
            continue
        seen.add(i)
        out.append(source[i : i + n])
    return out


def _occurrences(source: bytes, needle: bytes, cap: int = 20000) -> list[int]:
    """Start offsets of ``needle`` inside ``source`` (bounded scan)."""
    out: list[int] = []
    start = 0
    while len(out) < cap:
        i = source.find(needle, start)
        if i < 0:
            break
        out.append(i)
        start = i + 1
    return out


def _hard_negatives(
    source: bytes, true_cont: bytes, count: int, rng: random.Random, forbid: int
) -> list[bytes]:
    """Same first 1-2 bytes as the truth, different remainder."""
    n = len(true_cont)
    prefix = true_cont[: min(2, n)]
    starts = [i for i in _occurrences(source, prefix) if i != forbid and i + n <= len(source)]
    cands = [source[i : i + n] for i in starts]
    cands = [c for c in cands if c != true_cont]
    rng.shuffle(cands)
    return cands[:count]


def negative_candidates(
    source: bytes,
    position: int,
    n: int,
    count: int,
    neg_type: str,
    seed: int,
) -> tuple[list[bytes], int]:
    """Return ``count`` negative spans + how many are hard negatives."""
    rng = random.Random(f"{seed}:{position}:{n}:{neg_type}")
    if neg_type == "hard":
        hard = _hard_negatives(source, source[position : position + n], count, rng, position)
        rest = _random_negatives(source, n, count - len(hard), rng, (position, position + n))
        return hard + rest, len(hard)
    return _random_negatives(source, n, count, rng, (position, position + n)), 0


def evaluate_discrimination(
    backend: Backend,
    source: bytes,
    positions: Sequence[int],
    context_bytes: int,
    n: int,
    *,
    neg_type: str = "random",
    n_neg: int = N_CANDIDATES - 1,
    seed: int = 42,
    backend_options: dict | None = None,
    workers: int = 1,
) -> list[DiscRecord]:
    """Rank the true N-byte continuation against negatives per position."""
    from concurrent.futures import ThreadPoolExecutor

    opts = dict(backend_options or {})
    eff = min(context_bytes, backend.context_limit)
    pool = ThreadPoolExecutor(workers) if workers > 1 else None
    records: list[DiscRecord] = []
    try:
        for p in positions:
            if p + n > len(source):
                continue
            context = source[max(0, p - eff) : p]
            true_cont = source[p : p + n]
            negs, n_hard = negative_candidates(source, p, n, n_neg, neg_type, seed)
            cands = [true_cont, *negs]
            t0 = time.perf_counter()
            scorer = backend.prepare(context, **opts)
            t1 = time.perf_counter()
            scores = score_position(scorer, cands, pool)
            t2 = time.perf_counter()
            s_true = scores[0]
            better = sum(1 for s in scores if s < s_true)
            tied = sum(1 for s in scores if s == s_true)
            counts: dict[int, int] = {}
            for s in scores:
                counts[s] = counts.get(s, 0) + 1
            neg_scores = scores[1:]
            hard_scores = neg_scores[:n_hard]
            rand_scores = neg_scores[n_hard:]
            records.append(
                DiscRecord(
                    position=p,
                    true_rank_avg=better + (tied + 1) / 2,
                    better_count=better,
                    true_tie_size=tied,
                    unique_scores=len(counts),
                    max_tie=max(counts.values()),
                    all_tied=len(counts) == 1,
                    n_candidates=len(cands),
                    n_hard_negatives=n_hard,
                    score_margin=(min(neg_scores) - s_true) if neg_scores else 0.0,
                    hard_neg_win=(
                        sum(1 for s in hard_scores if s_true <= s) / len(hard_scores)
                        if hard_scores
                        else None
                    ),
                    rand_neg_win=(
                        sum(1 for s in rand_scores if s_true <= s) / len(rand_scores)
                        if rand_scores
                        else None
                    ),
                    prep_seconds=t1 - t0,
                    score_seconds=t2 - t1,
                )
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=False)
    return records


def _mean_opt(vals) -> float | None:
    """Mean over non-None values; None if the whole column is empty."""
    vs = [v for v in vals if v is not None]
    return statistics.fmean(vs) if vs else None


def aggregate_disc(
    records: Sequence[DiscRecord],
    *,
    dataset: str,
    backend: str,
    scoring_mode: str,
    context_bytes: int,
    continuation_bytes: int,
    neg_type: str,
    note: str = "",
) -> dict:
    """One continuation_discrimination.csv row."""
    n = len(records)
    base = {
        "dataset": dataset,
        "backend": backend,
        "scoring_mode": scoring_mode,
        "context_bytes": context_bytes,
        "continuation_bytes": continuation_bytes,
        "negative_type": neg_type,
        "num_positions": n,
        "note": note,
    }
    if n == 0:
        return base
    evals = sum(r.n_candidates for r in records)
    score_s = sum(r.score_seconds for r in records)
    base.update(
        {
            "candidate_count": statistics.fmean(r.n_candidates for r in records),
            "hard_negatives_mean": statistics.fmean(r.n_hard_negatives for r in records),
            "top1": sum(_expected_topk(r.better_count, r.true_tie_size, 1) for r in records) / n,
            "top5": sum(_expected_topk(r.better_count, r.true_tie_size, 5) for r in records) / n,
            "mrr": statistics.fmean(1.0 / r.true_rank_avg for r in records),
            "mean_rank": statistics.fmean(r.true_rank_avg for r in records),
            "median_rank": statistics.median(r.true_rank_avg for r in records),
            "unique_scores_mean": statistics.fmean(r.unique_scores for r in records),
            "largest_tie_mean": statistics.fmean(r.max_tie for r in records),
            "all_tied_fraction": sum(1 for r in records if r.all_tied) / n,
            "score_margin_mean": statistics.fmean(r.score_margin for r in records),
            "hard_neg_win_rate": _mean_opt(r.hard_neg_win for r in records),
            "rand_neg_win_rate": _mean_opt(r.rand_neg_win for r in records),
            "evals": evals,
            "evals_per_sec": evals / score_s if score_s else 0.0,
            "contexts_per_sec": n / (score_s + sum(r.prep_seconds for r in records))
            if score_s
            else 0.0,
        }
    )
    return base
