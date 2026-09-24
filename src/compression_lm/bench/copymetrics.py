"""Retrieval/copy metrics — separating prediction from memorization.

A compressor-LM that simply recites corpus substrings looks great on
naive metrics. These functions measure how much of an output is copied,
at several granularities:

- per-position longest corpus match (and its max/mean/median),
- exact n-gram overlap for n in {4, 8, 16, 32, 64},
- novelty = 1 - overlap at the sizes that matter,
- copy-run: the longest single span appearing verbatim in the corpus.

Plus UTF-8 validation stats, kept byte-oriented internally so multibyte
corpora (e.g. Korean) work unchanged.
"""

from __future__ import annotations

import statistics


def longest_match_lengths(text: bytes, corpus: bytes) -> list[int]:
    """Per-position length of the longest corpus substring starting there.

    ``L[i] = max L such that text[i:i+L] occurs in corpus``. Computed in
    O(len(text) + total growth): a match of length K ending at position i
    guarantees a match of length K-1 at i+1 (it's a substring of the same
    occurrence), so each position only extends beyond the previous best.
    """
    n = len(text)
    out = [0] * n
    prev = 0
    for i in range(n):
        length = max(0, prev - 1)
        while i + length + 1 <= n and text[i : i + length + 1] in corpus:
            length += 1
        out[i] = length
        prev = length
    return out


def ngram_overlap(text: bytes, corpus: bytes, n: int) -> float:
    """Fraction of ``text``'s n-grams occurring verbatim in ``corpus``."""
    if len(text) < n or not corpus:
        return 0.0
    total = len(text) - n + 1
    hits = sum(1 for i in range(total) if text[i : i + n] in corpus)
    return hits / total


def copy_metrics(
    text: bytes,
    corpus: bytes,
    *,
    ngram_sizes: tuple[int, ...] = (4, 8, 16, 32, 64),
    novelty_sizes: tuple[int, ...] = (8, 16, 32),
) -> dict:
    """Full retrieval/copy profile of one generated output."""
    lm = longest_match_lengths(text, corpus)
    row: dict = {
        "copy_span_max": max(lm) if lm else 0,
        "copy_span_mean": statistics.fmean(lm) if lm else 0.0,
        "copy_span_median": statistics.median(lm) if lm else 0.0,
        "copy_run": max(lm) if lm else 0,
    }
    for n in ngram_sizes:
        row[f"overlap_{n}gram"] = ngram_overlap(text, corpus, n)
    for n in novelty_sizes:
        row[f"novel_{n}gram"] = 1.0 - row[f"overlap_{n}gram"]
    return row


def utf8_stats(text: bytes) -> dict:
    """UTF-8 validity profile of a byte string.

    ``valid`` is the strict-decode verdict; ``invalid_sequences`` counts
    replacement chars produced under errors='replace' (one per invalid
    sequence); ``chars``/``repl`` describe the decoded text.
    """
    decoded = text.decode("utf-8", errors="replace")
    try:
        text.decode("utf-8", errors="strict")
        valid = 1
    except UnicodeDecodeError:
        valid = 0
    repl = decoded.count("\ufffd")
    return {
        "utf8_valid": valid,
        "utf8_invalid_sequences": repl if not valid else 0,
        "utf8_chars": len(decoded),
        "utf8_repl_chars": repl,
    }


def self_repetition(text: bytes, n: int = 8) -> float:
    """Fraction of n-grams occurring more than once — loop detector."""
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    seen: set[bytes] = set()
    repeated = 0
    for g in grams:
        if g in seen:
            repeated += 1
        seen.add(g)
    return repeated / len(grams)


def distinct_ratio(text: bytes, n: int = 4) -> float:
    """Unique n-grams / total n-grams — vocabulary diversity proxy."""
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0
