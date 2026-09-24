"""Tests for instrumentation helpers."""

import math

from compression_lm import backends
from compression_lm.bench.instrument import (
    package_versions,
    pearson,
    probe_overhead,
    scorer_method,
    spearman,
    speed_microbench,
)

CTX = b"instrumentation context bytes " * 40


def test_probe_overhead_shape():
    row = probe_overhead(backends.get("deflate"), CTX)
    assert row["scoring_method"] == "incremental_state"
    assert row["context_bytes"] == len(CTX)
    for k in (0, 1, 32):
        assert f"delta_novel_{k}B" in row


def test_scorer_method_classification():
    assert scorer_method(backends.get("deflate").prepare(CTX)) == "incremental_state"
    assert scorer_method(backends.get("bzip2").prepare(CTX)) == "full_recompress"
    zstd = backends.info("zstd")
    if zstd.available:
        assert scorer_method(backends.get("zstd").prepare(CTX)) == "dictionary"


def test_speed_microbench():
    row = speed_microbench(backends.get("deflate"), CTX, n_prepare=2, n_evals=64)
    assert row["prepare_seconds"] > 0
    assert row["evals_per_sec"] > 0
    assert row["contexts_per_sec"] > 0


def test_pearson_basic():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) > 0.99
    assert pearson([1, 2, 3, 4], [8, 6, 4, 2]) < -0.99
    assert math.isnan(pearson([1, 1, 1], [1, 2, 3]))


def test_spearman_monotonic():
    # tied y-values get shared average ranks; exact rho = 0.8944
    assert spearman([1, 2, 3, 4, 5], [0, 0, 0, 1, 9]) > 0.85
    # ties get shared average ranks, deterministic
    r = spearman([1, 1, 2, 2], [3, 4, 5, 6])
    assert r == spearman([1, 1, 2, 2], [3, 4, 5, 6])


def test_package_versions():
    v = package_versions()
    assert "zstandard" in v and "matplotlib" in v
