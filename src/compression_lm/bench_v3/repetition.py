"""Repetition-collapse metrics (spec §17).

A compressor can drive generated-BPB down by emitting degenerate periodic
text (``uyuyuyuy...``) — low BPB, zero language content. These metrics
catch that failure mode for every backend's generated output:

- ``rep_2/4/8gram``: fraction of positions where an n-gram is immediately
  followed by an identical n-gram (adjacent periodic repeat).
- ``unique_byte_ratio``: distinct byte values / 256 — near 0 on collapse.
- ``longest_periodic_run``: longest span that is a pure repetition of a
  unit of <=16 bytes.
- ``byte_entropy``: Shannon entropy of the byte histogram (bits/byte).
"""

from __future__ import annotations

import math


def adjacent_repeat_ratio(text: bytes, n: int) -> float:
    """Fraction of n-gram slots followed by the identical n-gram."""
    if len(text) < 2 * n:
        return 0.0
    total = len(text) - 2 * n + 1
    hits = sum(1 for i in range(total) if text[i : i + n] == text[i + n : i + 2 * n])
    return hits / total


def longest_periodic_run(text: bytes, max_unit: int = 16) -> tuple[int, int]:
    """``(span, period)`` of the longest strict repetition of a <=``max_unit`` unit.

    A period-p run is a maximal stretch of positions j with
    ``text[j] == text[j-p]``; the repeated span has length ``run + p`` and
    counts only when it covers at least two copies (``run >= p``).
    """
    n = len(text)
    best = best_p = 0
    for p in range(1, max_unit + 1):
        run = 0
        for j in range(p, n):
            if text[j] == text[j - p]:
                run += 1
                if run >= p and run + p > best:
                    best, best_p = run + p, p
            else:
                run = 0
    return best, best_p


def byte_entropy(text: bytes) -> float:
    """Shannon entropy of the byte histogram, bits per byte (max 8)."""
    if not text:
        return 0.0
    counts = [0] * 256
    for b in text:
        counts[b] += 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts if c)


def repetition_metrics(text: bytes) -> dict:
    """repetition_metrics.csv columns for one generated output."""
    span, period = longest_periodic_run(text)
    return {
        "rep_2gram": adjacent_repeat_ratio(text, 2),
        "rep_4gram": adjacent_repeat_ratio(text, 4),
        "rep_8gram": adjacent_repeat_ratio(text, 8),
        "unique_byte_ratio": len(set(text)) / 256.0,
        "longest_periodic_run": span,
        "periodic_run_period": period,
        "byte_entropy": byte_entropy(text),
    }
