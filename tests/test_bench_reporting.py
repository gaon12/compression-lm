"""Smoke tests for result writers, plots, and the report builder."""

from pathlib import Path

from compression_lm.bench import report, reporting
from compression_lm.bench.runner import BenchConfig, assemble


def _fake_rows(tmp: Path) -> None:
    """Minimal synthetic raw rows exercising every assembled CSV."""
    rows_dir = tmp / "raw" / "rows"
    rows_dir.mkdir(parents=True)
    teacher_row = {
        "backend": "deflate",
        "family": "deflate",
        "mode": "quick",
        "context_provided": 32768,
        "context_effective": 32768,
        "n_positions": 10,
        "top1": 0.1,
        "top5": 0.3,
        "top10": 0.4,
        "top1_exp": 0.2,
        "top5_exp": 0.4,
        "top10_exp": 0.5,
        "mean_rank": 40.0,
        "median_rank": 30.0,
        "mrr": 0.2,
        "mean_true_delta_bits": 5.0,
        "teacher_forced_pseudo_bpb": 6.5,
        "pseudo_ppl": 90.0,
        "tie_candidate_frac": 0.9,
        "pct_true_tied": 0.9,
        "pct_true_in_best_tie": 0.5,
        "mean_unique_scores": 5.0,
        "mean_max_tie": 120.0,
        "evals": 2560,
        "evals_per_sec": 10000.0,
        "positions_per_sec": 39.0,
        "prep_seconds": 0.1,
        "score_seconds": 0.25,
        "seconds": 0.26,
        "dedup": False,
        "note": "",
    }
    gen_row = {
        "backend": "deflate",
        "mode": "equal_search",
        "prompt": "__mean__",
        "status": "ok",
        "generated_compression_bpb": 1.5,
        "copy_run": 80.0,
        "copy_span_max": 80.0,
        "copy_span_mean": 40.0,
        "copy_span_median": 35.0,
        "overlap_8gram": 0.8,
        "overlap_16gram": 0.6,
        "overlap_32gram": 0.4,
        "overlap_64gram": 0.2,
        "novel_8gram": 0.2,
        "novel_16gram": 0.4,
        "novel_32gram": 0.6,
        "self_rep_8": 0.05,
        "distinct_4": 0.5,
        "utf8_valid": True,
        "evals_per_sec": 9000.0,
        "bytes_per_sec": 10.0,
        "context_effective": 32768,
        "out_bytes": 256,
        "n_prompts": 2,
    }
    speed_row = {
        "backend": "deflate",
        "source": "microbench",
        "context_bytes": 32768,
        "prepare_seconds": 0.01,
        "contexts_per_sec": 100.0,
        "eval_seconds_mean": 0.00005,
        "evals_per_sec": 20000.0,
        "peak_rss_bytes": 50_000_000,
    }
    overhead_row = {
        "backend": "deflate",
        "scoring_method": "incremental_state",
        "context_bytes": 8192,
        "delta_novel_0B": 0,
        "delta_novel_1B": 1,
        "delta_novel_8B": 6,
        "delta_novel_32B": 24,
        "delta_novel_64B": 47,
    }
    for name, rows in {
        "t": [teacher_row],
        "g": [gen_row],
        "s": [speed_row],
        "o": [overhead_row],
    }.items():
        reporting.write_json(rows_dir / f"{name}.json", rows)


def test_write_csv_roundtrip(tmp_path):
    path = tmp_path / "x.csv"
    reporting.write_csv(path, [{"a": 1, "b": "x"}, {"a": 2, "c": 3}])
    text = path.read_text()
    assert "a,b" in text and "2,,3" in text


def test_plots_and_report_on_synthetic(tmp_path):
    _fake_rows(tmp_path)
    cfg = BenchConfig(dataset=Path("unused"), out_dir=tmp_path)

    class _DS:
        def split_meta(self):
            return {"name": "fake"}

    assemble(cfg, _DS())
    for f in (
        "teacher_forced.csv",
        "quantization.csv",
        "generation_equal_search.csv",
        "copy_metrics.csv",
        "speed.csv",
        "overhead.csv",
        "summary.csv",
    ):
        assert (tmp_path / f).exists(), f
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "deflate" in md and "Executive summary" in md
    assert (tmp_path / "plots" / "01_top1_by_backend.png").exists()


def test_report_handles_empty(tmp_path):
    out = report.build_report(tmp_path)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "no rows" in text or "insufficient" in text
