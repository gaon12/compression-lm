"""Quality metrics for generated text.

Compression-based generators live on a spectrum between verbatim copying
and noise, so the interesting numbers are: how much of the output is
traceable to the corpus (copy ratio), how much the output repeats itself
(self-repetition), and the mean conditional code length the compressor
reported while generating (bits/byte — the model's own confidence).
"""

from __future__ import annotations


def corpus_overlap(text: bytes, corpus: bytes, n: int = 8) -> float:
    """Fraction of ``text``'s n-grams that occur verbatim in ``corpus``.

    1.0 means every n-gram is a corpus substring (pure recombination);
    near 0 means the output is mostly novel bytes.
    """
    if len(text) < n or not corpus:
        return 0.0
    hits = sum(1 for i in range(len(text) - n + 1) if text[i : i + n] in corpus)
    return hits / (len(text) - n + 1)


def self_repetition(text: bytes, n: int = 8) -> float:
    """Fraction of ``text``'s n-grams that occur more than once in ``text``.

    High values indicate degenerate looping — the model falling into a
    cycle it can compress trivially against itself.
    """
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
    """Unique n-grams / total n-grams — a vocabulary-diversity proxy."""
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0
