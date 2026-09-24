"""Tests for copy/retrieval metrics and UTF-8 validation."""

from compression_lm.bench.copymetrics import (
    copy_metrics,
    distinct_ratio,
    longest_match_lengths,
    ngram_overlap,
    self_repetition,
    utf8_stats,
)

CORPUS = b"the quick brown fox jumps over the lazy dog"


def test_longest_match_basic():
    text = b"the quick"
    lm = longest_match_lengths(text, CORPUS)
    assert lm[0] == 9  # whole text is a corpus substring
    assert lm[1] == 8
    assert lm[-1] == 1


def test_longest_match_novel_bytes():
    text = b"the\x00\xff"
    lm = longest_match_lengths(text, CORPUS)
    assert lm[0] == 3  # "the"
    assert lm[3] == 0  # \x00 not in corpus
    assert lm[4] == 0


def test_longest_match_incremental_correctness():
    """Incremental algorithm must equal naive per-position search."""
    text = b"ox jumps lazy"
    lm = longest_match_lengths(text, CORPUS)

    def naive(i):
        best = 0
        for L in range(1, len(text) - i + 1):
            if text[i : i + L] in CORPUS:
                best = L
        return best

    assert lm == [naive(i) for i in range(len(text))]


def test_ngram_overlap():
    assert ngram_overlap(b"the quick brown", CORPUS, 4) == 1.0
    assert ngram_overlap(b"\x00\x01\x02\x03", CORPUS, 4) == 0.0
    assert ngram_overlap(b"ab", CORPUS, 8) == 0.0  # text shorter than n


def test_copy_metrics_shape():
    row = copy_metrics(b"the quick brown fox", CORPUS)
    for n in (4, 8, 16, 32, 64):
        assert f"overlap_{n}gram" in row
    for n in (8, 16, 32):
        assert f"novel_{n}gram" in row
    assert row["copy_run"] == row["copy_span_max"] == 19
    assert row["copy_span_mean"] > 0


def test_copy_run_catches_recitation():
    copied = CORPUS[:40] + b"\x00"
    row = copy_metrics(copied, CORPUS)
    assert row["copy_run"] == 40


def test_utf8_stats_valid_ascii():
    s = utf8_stats(b"hello world")
    assert s["utf8_valid"] == 1
    assert s["utf8_invalid_sequences"] == 0
    assert s["utf8_chars"] == 11


def test_utf8_stats_multibyte():
    s = utf8_stats("안녕 세계".encode())
    assert s["utf8_valid"] == 1
    assert s["utf8_chars"] == 5  # chars, not bytes


def test_utf8_stats_invalid():
    s = utf8_stats(b"abc\xff\xfe\x00def")
    assert s["utf8_valid"] == 0
    assert s["utf8_invalid_sequences"] >= 1
    assert s["utf8_repl_chars"] >= 1


def test_self_repetition_and_distinct():
    assert self_repetition(b"abcabcabcabc", 3) > 0.5
    assert distinct_ratio(b"abcdefghij", 4) == 1.0
    assert distinct_ratio(b"abababab", 2) < 1.0
