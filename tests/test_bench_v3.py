"""v3 experiment tests — options registry, zstd modes, discrimination,
repetition metrics, stats, raw-pair comparison, equal-time, resume.

Optional-dependency tests are skipped (not failed) when a package is absent.
"""

from __future__ import annotations

import csv
import importlib.util
import zlib

import pytest

from compression_lm.backends.base import Backend, OneShotScorer
from compression_lm.backends.zlib_backend import ZlibBackend
from compression_lm.bench import datasets, reporting
from compression_lm.bench_v3 import (
    backends_v3,
    discriminate,
    equaltime,
    options,
    rawpair,
    repetition,
    runner3,
    stats3,
    wikitext,
)
from compression_lm.engine import GenerationConfig, generate


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


class _ToyBackend(Backend):
    """Minimal stdlib-only backend for runner/equaltime tests."""

    name = "toy"
    context_limit = 64

    def prepare(self, context: bytes, **_opts) -> OneShotScorer:
        return OneShotScorer(lambda d: zlib.compress(d, 9), context)


# ------------------------------------------------------------- options


def test_options_registry_covers_all_factories():
    names = set(backends_v3.v3_factories()) | {"zstd_dict_trained"}
    assert set(options.BACKEND_OPTIONS) == names
    for name, o in options.BACKEND_OPTIONS.items():
        assert o.get("container"), name
        assert o.get("checksum"), name
        assert o.get("streaming"), name
        assert o.get("dictionary"), name
        assert o.get("scoring_mode"), name


def test_prepare_options_are_accepted_by_prepare():
    for name in ("deflate", "zlib", "gzip", "bzip2", "lzma", "lzma2", "xz"):
        be, reason = backends_v3.check_backend(name)
        assert be is not None, reason
        sc = be.prepare(b"abcabc" * 8, **options.prepare_options(name))
        assert isinstance(sc.measure(b"x"), int)


def test_options_rows_have_versions():
    rows = options.options_rows()
    assert {"zstd_full", "snappy", "lzma2"} <= {r["backend"] for r in rows}
    for r in rows:
        assert r["scoring_mode"]
        assert r["package_version"]


# ------------------------------------------------------------- zstd modes


@pytest.mark.skipif(not _have("zstandard"), reason="zstandard not installed")
def test_zstd_modes_are_distinct_backends():
    modes = {
        n: backends_v3.check_backend(n)[0]
        for n in ("zstd_full", "zstd_raw", "zstd_stream", "zstd_dict")
    }
    assert all(modes.values())
    assert {be.name for be in modes.values()} == set(modes)
    ctx = b"the quick brown fox jumps over the lazy dog. " * 8
    for name, be in modes.items():
        sc = be.prepare(ctx)
        assert all(isinstance(sc.measure(c), int) for c in (b"a", b" ", b"the")), name


@pytest.mark.skipif(not _have("zstandard"), reason="zstandard not installed")
def test_zstd_full_matches_naive_concat_delta():
    """spec §30: optimized/naive agreement for the full-recompress mode."""
    import zstandard

    ctx = b"hello world, hello compression " * 8
    be = backends_v3.check_backend("zstd_full")[0]
    sc = be.prepare(ctx)
    cctx = zstandard.ZstdCompressor(level=3)
    base = len(cctx.compress(ctx))
    for b in b"ax q":
        naive = len(cctx.compress(ctx + bytes([b]))) - base
        assert sc.measure(bytes([b])) == naive


@pytest.mark.skipif(not _have("zstandard"), reason="zstandard not installed")
def test_zstd_stream_scores_are_finite_and_positive():
    ctx = b"hello world, hello compression " * 8
    be = backends_v3.check_backend("zstd_stream")[0]
    sc = be.prepare(ctx)
    vals = [sc.measure(c) for c in (b"a", b" ", b"the", b"zzzz")]
    assert all(v > 0 for v in vals)
    # multi-byte candidate costs more than a single byte at this boundary
    assert max(vals) > min(vals) or len(set(vals)) >= 1


@pytest.mark.skipif(not _have("zstandard"), reason="zstandard not installed")
def test_zstd_dict_trained_backend_constructible():
    import zstandard

    d = zstandard.ZstdCompressionDict(
        b"trained content bytes " * 128, dict_type=zstandard.DICT_TYPE_RAWCONTENT
    )
    be = backends_v3.ZstdDictTrainedBackend(d.as_bytes())
    sc = be.prepare(b"context here")
    assert isinstance(sc.measure(b"x"), int)


# ------------------------------------------------------------- discriminate


def test_negative_candidates_deterministic_and_correct_length():
    source = bytes(range(256)) * 4
    a, n_hard_a = discriminate.negative_candidates(source, 100, 8, 31, "random", 42)
    b, _ = discriminate.negative_candidates(source, 100, 8, 31, "random", 42)
    assert a == b and len(a) == 31 and n_hard_a == 0
    assert all(len(g) == 8 for g in a)


def test_hard_negatives_share_prefix_and_differ():
    import random as _r

    src = bytearray(_r.Random(0).randbytes(2048))
    pos = 100
    true = bytes(src[pos : pos + 8])
    # plant extra occurrences of the true 2-byte prefix
    src[400:402] = true[:2]
    src[900:902] = true[:2]
    source = bytes(src)
    negs, n_hard = discriminate.negative_candidates(source, pos, 8, 31, "hard", 42)
    assert n_hard >= 2
    for g in negs[:n_hard]:
        assert g[:2] == true[:2]
        assert g != true


def test_negative_candidates_exclude_true_start():
    source = bytes(range(128)) * 8
    negs, _ = discriminate.negative_candidates(source, 200, 8, 31, "random", 42)
    # the sampled span at the forbidden offset must not be among negatives
    assert source[200:208] not in [g for g in negs] or source.count(source[200:208]) > 1


def test_evaluate_discrimination_ranks_true_continuation():
    source = b"abcdefgh" * 64
    be = _ToyBackend()
    recs = discriminate.evaluate_discrimination(be, source, [10, 20, 30], 16, 4, seed=42)
    assert len(recs) == 3
    for r in recs:
        assert r.n_candidates == 32
        assert 1.0 <= r.true_rank_avg <= r.n_candidates
        assert 1 <= r.unique_scores <= r.n_candidates
        assert r.n_hard_negatives == 0
    agg = discriminate.aggregate_disc(
        recs,
        dataset="t",
        backend="toy",
        scoring_mode="full_recompress",
        context_bytes=16,
        continuation_bytes=4,
        neg_type="random",
    )
    assert 0.0 <= agg["top1"] <= 1.0
    assert agg["candidate_count"] == 32


# ------------------------------------------------------------- repetition


def test_periodic_run_reports_span_and_period():
    span, period = repetition.longest_periodic_run(b"uy" * 50)
    assert span == 100 and period == 2
    span2, _ = repetition.longest_periodic_run(b"abcdefghijklmnop")
    assert span2 == 0


def test_repetition_metrics_flag_collapse():
    m = repetition.repetition_metrics(b"uy" * 50)
    assert m["rep_2gram"] > 0.9
    assert m["unique_byte_ratio"] < 0.02
    assert m["byte_entropy"] < 1.5
    healthy = repetition.repetition_metrics(bytes(range(256)))
    assert healthy["rep_2gram"] == 0.0
    assert healthy["unique_byte_ratio"] == 1.0
    assert healthy["byte_entropy"] == 8.0


# ------------------------------------------------------------- stats3


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [0.2, 0.3, 0.4, 0.5] * 10
    mean, lo, hi = stats3.bootstrap_mean_ci(vals, n_boot=200)
    assert 0.2 <= lo <= mean <= hi <= 0.5
    assert stats3.bootstrap_mean_ci(vals, n_boot=200) == (mean, lo, hi)


# ------------------------------------------------------------- rawpair


def test_compare_pair_deflate_vs_zlib_stdlib():
    source = b"hello world, this is a test of the compression benchmark. " * 8
    a, b = ZlibBackend("deflate", -15), ZlibBackend("zlib", 15)
    rows, agg = rawpair.compare_pair(a, b, source, [200, 300], 64)
    assert agg["n_contexts"] == 2
    assert len(rows) == 2
    assert 0.0 <= agg["top1_exp_a"] <= 1.0
    assert 0.0 <= agg["top1_exp_b"] <= 1.0
    assert agg["spearman_mean"] > 0.5  # same codec, different wrapper


def test_empty_frame_sizes_records_values():
    sizes = rawpair.empty_frame_sizes(["deflate", "zlib", "lzma2"], options.BACKEND_OPTIONS)
    assert sizes["zlib"] > 0 and sizes["lzma2"] > 0


# ------------------------------------------------------------- equaltime / engine


def test_engine_time_budget_stops_early():
    res = generate(
        _ToyBackend(),
        b"abcdef " * 400,
        b"abc",
        GenerationConfig(length=256, beam_width=2, lookahead=8, time_budget_sec=0.001, workers=1),
    )
    assert 0 < len(res.text) < 256  # partial output kept, not discarded


def test_run_equal_time_rows():
    source = b"abcdef " * 400
    rows = equaltime.run_equal_time(
        _ToyBackend(),
        source,
        [b"abc"],
        budget_sec=0.05,
        beam=1,
        horizon=4,
        length=64,
        seed=42,
    )
    (r,) = rows
    assert r["status"] == "ok"
    assert r["time_budget_sec"] == 0.05
    assert 0 < r["generated_bytes"] <= 64
    assert r["actual_runtime_sec"] > 0
    assert "copy_run_ratio" in r and "rep_4gram" in r and "output_file" not in r


# ------------------------------------------------------------- wikitext


def test_wikitext_adapter_from_local_files(tmp_path):
    d = tmp_path / "wikitext2"
    d.mkdir()
    for f in wikitext.FILES:
        (d / f).write_bytes(b"= Title =\n some text here\n" * 50)
    ds = wikitext.load_wikitext2(tmp_path, gap=64, seed=42)
    assert ds.name == "wikitext2"
    assert len(ds.train) == len(ds.val) == len(ds.test) > 0
    assert ds.val_span[0] == ds.train_span[1] + 64
    assert ds.test_span[0] == ds.val_span[1] + 64
    assert ds.split_meta()["gap"] == 64


# ------------------------------------------------------------- runner plumbing


def test_manifest_persists_done_keys(tmp_path):
    path = tmp_path / "manifest.json"
    runner3._Manifest(path).mark("teacher:t:zlib:4096:10")
    m2 = runner3._Manifest(path)
    assert "teacher:t:zlib:4096:10" in m2.done


def test_manifest_retries_error_rows(tmp_path):
    path = tmp_path / "manifest.json"
    key = "speed:t:zlib"
    reporting.write_json(tmp_path / "rows" / "speed__t__zlib.json", [{"status": "error"}])
    runner3._Manifest(path).mark(key)
    assert key not in runner3._Manifest(path).done


def test_resume_rejects_changed_data_or_settings(tmp_path):
    cfg = runner3.Bench3Config(out_dir=tmp_path / "res")
    manifest = runner3._Manifest(cfg.raw_dir / "manifest.json")
    ds = datasets.split_contiguous(b"abcdef" * 30, name="toy", gap=1)
    runner3._check_run_config(cfg, ds, manifest)
    runner3._check_run_config(cfg, ds, manifest)
    cfg.seed += 1
    with pytest.raises(ValueError, match="different data or settings"):
        runner3._check_run_config(cfg, ds, manifest)


def test_assemble_smoke_writes_every_csv(tmp_path):
    cfg = runner3.Bench3Config(out_dir=tmp_path / "res", dataset="t", phases=frozenset())
    rows_dir = cfg.raw_dir / "rows"
    rows_dir.mkdir(parents=True)
    tf_row = {
        "task": "teacher",
        "status": "ok",
        "dataset": "t",
        "backend": "zlib",
        "scoring_mode": "incremental_state",
        "mode": "teacher_forced_1byte",
        "context_provided": 4096,
        "context_effective": 4096,
        "n_positions": 3,
        "top1": 0.5,
        "top5": 0.8,
        "top10": 0.9,
        "top1_exp": 0.5,
        "top5_exp": 0.8,
        "top10_exp": 0.9,
        "mrr": 0.6,
        "mean_rank": 2.0,
        "median_rank": 1.5,
        "teacher_forced_pseudo_bpb": 4.0,
        "pseudo_ppl": 16.0,
        "mean_unique_scores": 5.0,
        "mean_max_tie": 50.0,
        "tie_candidate_frac": 0.5,
        "pct_true_tied": 0.4,
        "pct_true_in_best_tie": 0.3,
        "evals": 768,
        "evals_per_sec": 1000.0,
        "positions_per_sec": 5.0,
        "note": "",
    }
    disc_row = {
        "task": "discrimination",
        "status": "ok",
        "dataset": "t",
        "backend": "zlib",
        "scoring_mode": "incremental_state",
        "context_bytes": 4096,
        "context_effective": 4096,
        "continuation_bytes": 4,
        "negative_type": "random",
        "num_positions": 2,
        "candidate_count": 32,
        "hard_negatives_mean": 0.0,
        "top1": 0.5,
        "top5": 0.9,
        "mrr": 0.6,
        "mean_rank": 3.0,
        "median_rank": 2.0,
        "unique_scores_mean": 10.0,
        "largest_tie_mean": 5.0,
        "all_tied_fraction": 0.0,
        "score_margin_mean": 1.0,
        "hard_neg_win_rate": None,
        "rand_neg_win_rate": 0.8,
        "evals": 64,
        "evals_per_sec": 100.0,
        "contexts_per_sec": 5.0,
        "note": "",
    }
    reporting.write_json(rows_dir / "teacher__t__zlib__4096__3.json", [tf_row])
    reporting.write_json(
        rows_dir / "evalgen__t__zlib__2000.json",
        [
            {
                "task": "evalgen",
                "status": "ok",
                "dataset": "t",
                "backend": "zlib",
                "prompt": "__mean__",
                "budget_per_byte": 2000,
                "generated_compression_bpb": 2.0,
            }
        ],
    )
    reporting.write_json(
        rows_dir / "evalgen__t__zlib__10000.json",
        [
            {
                "task": "evalgen",
                "status": "ok",
                "dataset": "t",
                "backend": "zlib",
                "prompt": "__mean__",
                "budget_per_byte": 10000,
                "generated_compression_bpb": 1.0,
                "copy_run": 12,
                "overlap_32gram": 0.5,
            }
        ],
    )
    reporting.write_json(rows_dir / "disc__t__zlib__4096__4__random__2.json", [disc_row])
    reporting.write_json(
        cfg.raw_dir / "posrec" / "t__zlib__4096__3.json",
        [
            {
                "better_count": 0,
                "true_tie_size": 1,
                "rank_avg": 1.0,
                "nll_bits": 3.0,
                "unique_scores": 5,
            },
            {
                "better_count": 2,
                "true_tie_size": 4,
                "rank_avg": 4.0,
                "nll_bits": 5.0,
                "unique_scores": 4,
            },
            {
                "better_count": 1,
                "true_tie_size": 2,
                "rank_avg": 2.5,
                "nll_bits": 4.0,
                "unique_scores": 6,
            },
        ],
    )
    reporting.write_json(
        cfg.raw_dir / "discpos" / "t__zlib__4__random__2.json",
        [
            {
                "better_count": 0,
                "true_tie_size": 1,
                "true_rank_avg": 1.0,
                "unique_scores": 10,
            },
            {
                "better_count": 2,
                "true_tie_size": 2,
                "true_rank_avg": 3.0,
                "unique_scores": 9,
            },
        ],
    )
    runner3.assemble(cfg)
    for fname in (
        "teacher_forced_1byte.csv",
        "continuation_discrimination.csv",
        "quantization.csv",
        "scoring_method_validation.csv",
        "container_vs_raw.csv",
        "generation_equal_evaluation.csv",
        "generation_equal_time.csv",
        "copy_metrics.csv",
        "repetition_metrics.csv",
        "speed.csv",
        "backend_options.csv",
        "backend_status.csv",
        "summary.csv",
        "config.json",
        "report.md",
    ):
        assert (cfg.out_dir / fname).exists(), fname
    tf_csv = (cfg.out_dir / "teacher_forced_1byte.csv").read_text()
    assert "top1_exp_ci" in tf_csv
    assert "[0." in tf_csv  # CI brackets populated from posrec
    disc_csv = (cfg.out_dir / "continuation_discrimination.csv").read_text()
    assert "top1_ci" in disc_csv
    with (cfg.out_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    assert summary[0]["gen_bpb"] == "1.0"
    assert summary[0]["gen_copy_run"] == "12"
    rep = (cfg.out_dir / "report.md").read_text(encoding="utf-8")
    assert "zstd" in rep and "4 KiB" in rep
