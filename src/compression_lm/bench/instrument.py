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


def _plain_compressor(name: str, options: dict):
    """Method-A compressor: a backend's plain one-shot compress, used to
    compare concat-delta scores against each backend's actual scoring
    method (spec §14 experimental comparison). Returns None if the
    backend has no meaningful plain-compress path."""
    import bz2
    import gzip
    import lzma
    import zlib

    # mirror each backend's own default so A/B compare the same settings
    default_level = {
        "deflate": 9,
        "zlib": 9,
        "gzip": 9,
        "bzip2": 9,
        "lzma": 6,
        "lzma2": 6,
        "xz": 6,
        "zstd": 3,
        "zstd_dict": 3,
        "brotli": 5,
        "lz4": 1,
        "lzo": 9,
    }.get(name, 3)
    level = int(
        options.get(
            "level",
            options.get(
                "compresslevel",
                options.get("quality", options.get("preset", default_level)),
            ),
        )
    )
    if name == "deflate":
        return lambda d: zlib.compress(d, level)[2:-4]  # strip zlib wrapper
    if name == "zlib":
        return lambda d: zlib.compress(d, level)
    if name == "gzip":
        return lambda d: gzip.compress(d, compresslevel=level)
    if name == "bzip2":
        return lambda d: bz2.compress(d, compresslevel=level)
    if name in ("lzma", "lzma2", "xz"):
        fmt = {"lzma": lzma.FORMAT_ALONE, "xz": lzma.FORMAT_XZ}.get(name, lzma.FORMAT_RAW)
        filt = [{"id": lzma.FILTER_LZMA2, "preset": level}]
        return lambda d: lzma.compress(
            d,
            format=fmt,
            filters=filt if fmt == lzma.FORMAT_RAW else None,
            preset=level if fmt != lzma.FORMAT_RAW else None,
        )
    if name == "zstd":
        import zstandard

        c = zstandard.ZstdCompressor(level=level)
        return c.compress
    if name == "brotli":
        import brotli

        return lambda d: brotli.compress(d, quality=level)
    if name == "lz4":
        import lz4.block

        return lambda d: lz4.block.compress(d, store_size=False)
    if name == "snappy":
        import snappy

        return snappy.compress
    return None


def compare_scoring_methods(
    backend: Backend,
    context: bytes,
    *,
    candidates: Sequence[bytes] | None = None,
    backend_options: dict | None = None,
) -> dict:
    """Compare concat-delta (A) vs the backend's scorer delta (B).

    Method A: ``len(compress(ctx + c)) - len(compress(ctx))`` using the
    backend's plain compressor — the universally implementable score.
    Method B: whatever ``scorer.measure`` actually does (state clone /
    dictionary / one-shot). For one-shot backends A and B coincide by
    construction; for clone/dict scorers the agreement (or drift) is the
    interesting measurement. Experimental — never used for ranking.
    """
    opts = dict(backend_options or {})
    row: dict = {"backend": backend.name, "scoring_method": None}
    plain = _plain_compressor(backend.name, opts)
    scorer = backend.prepare(context, **opts)
    row["scoring_method"] = scorer_method(scorer)
    if plain is None:
        row["note"] = "no plain-compress path; comparison skipped"
        return row
    cands = candidates or [bytes([i % 256]) for i in range(64)]
    base = len(plain(context))
    a_vals, b_vals = [], []
    for c in cands:
        a_vals.append(len(plain(context + c)) - base)
        b_vals.append(scorer.measure(c))
    row["n_candidates"] = len(cands)
    row["methodA_mean"] = statistics.fmean(a_vals)
    row["methodB_mean"] = statistics.fmean(b_vals)
    diffs = [b - a for a, b in zip(a_vals, b_vals, strict=True)]
    row["B_minus_A_mean"] = statistics.fmean(diffs)
    row["B_minus_A_max"] = max(diffs)
    row["B_minus_A_min"] = min(diffs)
    row["pearson_A_B"] = pearson(a_vals, b_vals)
    row["spearman_A_B"] = spearman(a_vals, b_vals)
    row["identical_frac"] = sum(1 for d in diffs if d == 0) / len(diffs)
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
