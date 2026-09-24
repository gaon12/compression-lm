"""Container-vs-raw comparison (spec §10).

Does the stream wrapper distort next-byte scoring? For each backend pair
(e.g. ``lzma2`` raw vs ``xz`` container) we score the same candidate set on
the same contexts and compare:

- per-context Spearman correlation of the candidate score vectors,
- each side's expected top-1 on the true next byte,
- unique score counts and score variance (framing quantizes deltas),
- fixed overhead: ``len(compress(b""))`` per backend.

Pairs default to all 256 candidates; slow backends may pass a smaller
deterministic candidate subset (recorded in the row).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence

from ..backends.base import Backend
from ..bench.instrument import _plain_compressor, spearman
from ..bench.teacher import ALL_BYTES, _expected_topk, score_position

CONTAINER_PAIRS: tuple[tuple[str, str], ...] = (
    ("deflate", "zlib"),
    ("deflate", "gzip"),
    ("lzma2", "xz"),
    ("lzma", "xz"),
    ("zstd_raw", "zstd_full"),
)


def empty_frame_sizes(names: Sequence[str], options: dict[str, dict]) -> dict[str, int]:
    """``len(compress(b""))`` per backend — the fixed framing overhead.

    ``_plain_compressor`` only knows v2 names, so zstd variants share the
    "zstd" entry — recorded as approximate framing sizes.
    """
    out: dict[str, int] = {}
    for name in names:
        try:
            key = "zstd" if name.startswith("zstd") else name
            fn = _plain_compressor(key, options.get(name, {}))
            out[name] = len(fn(b"")) if fn is not None else -1
        except Exception:
            out[name] = -1
    return out


def compare_pair(
    be_a: Backend,
    be_b: Backend,
    source: bytes,
    positions: Sequence[int],
    context_bytes: int,
    *,
    candidates: Sequence[bytes] = ALL_BYTES,
    opts_a: dict | None = None,
    opts_b: dict | None = None,
) -> tuple[list[dict], dict]:
    """Score ``candidates`` under both backends on the same contexts.

    Returns ``(per_context_rows, aggregate_row)``. Per-context rows keep the
    Spearman coefficient and each side's score stats; the aggregate adds
    expected top-1 of the true byte on both backends.
    """
    oa, ob = dict(opts_a or {}), dict(opts_b or {})
    eff_a = min(context_bytes, be_a.context_limit)
    eff_b = min(context_bytes, be_b.context_limit)
    rows: list[dict] = []
    top1_a = top1_b = 0.0
    uniq_a: list[float] = []
    uniq_b: list[float] = []
    var_a: list[float] = []
    var_b: list[float] = []
    rhos: list[float] = []
    t0 = time.perf_counter()
    for p in positions:
        ctx_a = source[max(0, p - eff_a) : p]
        ctx_b = source[max(0, p - eff_b) : p]
        sa = score_position(be_a.prepare(ctx_a, **oa), candidates)
        sb = score_position(be_b.prepare(ctx_b, **ob), candidates)
        y = source[p]
        rho = spearman(sa, sb)
        rhos.append(rho)  # nan when a side is all-tied — filtered in the agg
        for s_list, uni, var in ((sa, uniq_a, var_a), (sb, uniq_b, var_b)):
            counts: dict[int, int] = {}
            for s in s_list:
                counts[s] = counts.get(s, 0) + 1
            uni.append(len(counts))
            var.append(statistics.pvariance(s_list) if len(s_list) > 1 else 0.0)
        for scores, acc in ((sa, "a"), (sb, "b")):
            s_true = scores[y] if y < len(scores) else None
            if s_true is None:
                continue
            better = sum(1 for s in scores if s < s_true)
            tied = sum(1 for s in scores if s == s_true)
            if acc == "a":
                top1_a += _expected_topk(better, tied, 1)
            else:
                top1_b += _expected_topk(better, tied, 1)
        rows.append({"position": p, "spearman": rho})
    n = len(rows)
    finite_rhos = [r for r in rhos if r == r]  # nan: one side all-tied
    agg = {
        "n_contexts": n,
        "n_candidates": len(candidates),
        "spearman_mean": statistics.fmean(finite_rhos) if finite_rhos else 0.0,
        "spearman_median": statistics.median(finite_rhos) if finite_rhos else 0.0,
        "spearman_undefined": n - len(finite_rhos),
        "top1_exp_a": top1_a / n if n else 0.0,
        "top1_exp_b": top1_b / n if n else 0.0,
        "unique_a": statistics.fmean(uniq_a) if uniq_a else 0.0,
        "unique_b": statistics.fmean(uniq_b) if uniq_b else 0.0,
        "score_var_a": statistics.fmean(var_a) if var_a else 0.0,
        "score_var_b": statistics.fmean(var_b) if var_b else 0.0,
        "seconds": time.perf_counter() - t0,
    }
    return rows, agg
