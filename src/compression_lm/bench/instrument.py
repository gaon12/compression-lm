"""Instrumentation: framing overhead, speed microbenchmarks, correlations.

Two things can quietly distort the benchmark and are measured explicitly:

- **Container/header overhead**: compressing a 1-byte candidate inside a
  container (gzip CRC, xz index, snappy header) adds framing bytes that
  pollute the delta score. probe_overhead() measures how the emitted size
  grows with candidate size for bytes that can't match the context, plus
  the empty-payload cost, and records which scoring path the backend uses
  (incremental state, dictionary, or full recompression).
- **LM-scorer speed**: not bulk-file MB/s but per-candidate latency,
  contexts/sec, evals/sec, and memory — what actually gates generation.

Also here: pearson/spearman implementations for the retrieval-vs-quality
correlation analysis (no scipy dependency).
"""

from __future__ import annotations

import importlib.metadata
import math
import os
import statistics
import time
from collections.abc import Sequence

from ..backends.base import Backend

try:
    import psutil
except ImportError:  # pragma: no cover - optional
    psutil = None


def scorer_method(scorer: object) -> str:
    """Classify how the scorer computes its delta."""
    name = type(scorer).__name__
    if "OneShot" in name:
        return "full_recompress"
    if "Dict" in name:
        return "dictionary"
    return "incremental_state"


def probe_overhead(
    backend: Backend,
    context: bytes,
    *,
    sizes: Sequence[int] = (0, 1, 2, 4, 8, 16, 32, 64),
    backend_options: dict | None = None,
) -> dict:
    """Measure per-candidate framing overhead for novel (unmatchable) bytes.

    Uses deterministic pseudo-random bytes outside the corpus alphabet so
    the delta reflects framing+literal coding cost, not match savings.
    """
    scorer = backend.prepare(context, **dict(backend_options or {}))
    row: dict = {
        "backend": backend.name,
        "scoring_method": scorer_method(scorer),
        "context_bytes": len(context),
    }
    for k in sizes:
        # deterministic bytes that cannot extend any context match
        cand = bytes(((k * 31 + i * 17) % 256) for i in range(k))
        # avoid accidentally-corpus bytes by xor-ing into the high range
        cand = bytes(b ^ 0x80 if b < 128 else b for b in cand)
        row[f"delta_novel_{k}B"] = scorer.measure(cand)
    return row


def rss_bytes() -> int | None:
    """Current process RSS in bytes, or None if psutil is unavailable."""
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss


def peak_rss_bytes() -> int | None:
    """Peak working set (Windows peak_wset) or max RSS where available."""
    if psutil is None:
        return None
    info = psutil.Process(os.getpid()).memory_info()
    return getattr(info, "peak_wset", None) or info.rss


def speed_microbench(
    backend: Backend,
    context: bytes,
    *,
    n_prepare: int = 5,
    n_evals: int = 2048,
    backend_options: dict | None = None,
) -> dict:
    """Time prepare() and measure() — the LM-scorer cost profile.

    ``prepare_seconds`` is the compressor-init + context-consumption cost;
    ``eval_seconds`` is per-candidate latency over ``n_evals`` single-byte
    calls. Python wrapper/object costs are included on purpose: they are
    part of the real system cost.
    """
    opts = dict(backend_options or {})
    rss0 = rss_bytes()

    # warm-up: one full prepare+measure cycle, untimed
    scorer = backend.prepare(context, **opts)
    scorer.measure(b"x")

    t0 = time.perf_counter()
    for _ in range(n_prepare):
        backend.prepare(context, **opts)
    prep_s = (time.perf_counter() - t0) / n_prepare

    scorer = backend.prepare(context, **opts)
    cands = [bytes([i % 256]) for i in range(n_evals)]
    t0 = time.perf_counter()
    scorer.measure_many(cands)
    eval_s = (time.perf_counter() - t0) / n_evals

    return {
        "backend": backend.name,
        "context_bytes": len(context),
        "prepare_seconds": prep_s,
        "contexts_per_sec": 1.0 / prep_s if prep_s else 0.0,
        "eval_seconds_mean": eval_s,
        "evals_per_sec": 1.0 / eval_s if eval_s else 0.0,
        "rss_before_bytes": rss0,
        "rss_after_bytes": rss_bytes(),
        "peak_rss_bytes": peak_rss_bytes(),
    }


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson r over paired samples; nan for degenerate inputs."""
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def _ranks(xs: Sequence[float]) -> list[float]:
    """Average ranks (1-based) with tie-sharing."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # ranks i+1..j+1 averaged
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho via Pearson on average ranks."""
    if len(x) < 3:
        return float("nan")
    return pearson(_ranks(x), _ranks(y))


def package_versions() -> dict[str, str]:
    """Versions of every optional backend package actually installed."""
    out: dict[str, str] = {}
    for dist in (
        "zstandard",
        "brotli",
        "lz4",
        "python-snappy",
        "python-lzo",
        "psutil",
        "matplotlib",
    ):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = "not-installed"
    return out
