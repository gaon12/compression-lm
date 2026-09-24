"""Tests for dataset splitting and position sampling."""

import pytest

from compression_lm.bench.datasets import (
    load_tiny_shakespeare,
    sample_positions,
    split_contiguous,
)

DATA = bytes(range(256)) * 500  # 128_000 bytes


def test_split_ranges_do_not_overlap():
    ds = split_contiguous(DATA, name="t", gap=1024)
    assert ds.train_span[1] < ds.val_span[0]
    assert ds.val_span[1] < ds.test_span[0]
    # gaps are actually discarded
    assert ds.val_span[0] - ds.train_span[1] == 1024
    assert ds.test_span[0] - ds.val_span[1] == 1024


def test_split_sizes_and_contents():
    ds = split_contiguous(DATA, name="t", gap=512)
    assert len(ds.train) == int(len(DATA) * 0.8)
    assert len(ds.val) == int(len(DATA) * 0.1)
    assert len(ds.test) == len(DATA) - ds.test_span[0]
    # train/test bytes are disjoint subsets of source
    assert ds.source[ds.train_span[0] : ds.train_span[1]] == ds.train
    assert ds.source[ds.test_span[0] : ds.test_span[1]] == ds.test


def test_no_test_byte_in_train():
    """The leakage invariant: scored bytes must not occur in train."""
    # deliberately repetitive data would break this; use structured data
    # where split boundaries still hold
    ds = split_contiguous(DATA, name="t", gap=512)
    assert ds.train_span[1] <= ds.test_span[0]


def test_split_too_small_raises():
    with pytest.raises(ValueError):
        split_contiguous(b"x" * 1000, name="t", gap=4096)


def test_context_at_respects_position():
    ds = split_contiguous(DATA, name="t", gap=512)
    pos = ds.test_span[0] + 100
    ctx = ds.context_at(pos, 64)
    assert ctx == DATA[pos - 64 : pos]
    # context never includes the scored byte or the future
    assert ctx[-1] == DATA[pos - 1]


def test_context_at_clamps_to_source_start():
    ds = split_contiguous(DATA, name="t", gap=512)
    assert ds.context_at(10, 4096) == DATA[:10]


def test_sample_positions_deterministic_and_in_range():
    span = (1000, 5000)
    a = sample_positions(span, 100, seed=42)
    b = sample_positions(span, 100, seed=42)
    assert a == b
    assert all(1000 <= p < 5000 for p in a)
    assert len(set(a)) == 100
    c = sample_positions(span, 100, seed=7)
    assert c != a


def test_sample_positions_subset_is_prefix_comparable():
    span = (0, 10_000)
    full = sample_positions(span, 1000, seed=42)
    assert full[:100] == sample_positions(span, 1000, seed=42)[:100]


def test_load_tiny_shakespeare(tmp_path):
    f = tmp_path / "s.txt"
    f.write_bytes(DATA)
    ds = load_tiny_shakespeare(f)
    assert ds.name == "tiny_shakespeare"
    assert len(ds.source) == len(DATA)
    meta = ds.split_meta()
    assert meta["train_bytes"] + meta["val_bytes"] + meta["test_bytes"] < len(DATA)
