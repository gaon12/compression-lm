"""Tests for generation benchmark plumbing."""

from compression_lm import backends
from compression_lm.bench.genbench import (
    aggregate_generation,
    eval_budget_to_beam,
    measure_output,
    run_generation,
    validate_prompts,
)
from compression_lm.engine import GenerationConfig

CORPUS = b"to be or not to be, that is the question. " * 40


def test_validate_prompts_filters_absent():
    got = validate_prompts([b"to be ", b"NOTPRESENTXYZ", b"the "], CORPUS)
    assert got == [b"to be ", b"the "]


def test_eval_budget_to_beam():
    assert eval_budget_to_beam(2000, 100) == 20
    assert eval_budget_to_beam(1, 100) == 1


def test_measure_output_keys():
    row = measure_output(b"to be or not", CORPUS, 1.5)
    for k in (
        "copy_run",
        "overlap_8gram",
        "novel_16gram",
        "utf8_valid",
        "self_rep_8",
        "distinct_4",
        "generated_compression_bpb",
    ):
        assert k in row
    assert row["generated_compression_bpb"] == 1.5


def test_run_generation_rows(tmp_path):
    backend = backends.get("deflate")
    cfg = GenerationConfig(length=32, lookahead=8, beam_width=8, seed=0)
    rows = run_generation(
        backend,
        CORPUS,
        [b"to be ", b"or not"],
        cfg,
        mode="equal_search",
        out_dir=tmp_path,
    )
    assert len(rows) == 2
    for r in rows:
        assert r["status"] == "ok"
        assert r["out_bytes"] == 32
        assert r["evals"] > 0
        assert "output_file" in r
    assert (tmp_path / "equal_search_deflate_00.txt").exists()


def test_aggregate_generation():
    backend = backends.get("deflate")
    cfg = GenerationConfig(length=24, lookahead=8, beam_width=8, seed=0)
    rows = run_generation(backend, CORPUS, [b"to be ", b"the "], cfg, mode="equal_search")
    agg = aggregate_generation(rows, mode="equal_search")
    assert agg["status"] == "ok"
    assert agg["n_prompts"] == 2
    assert agg["backend"] == "deflate"
    assert agg["evals"] > 0
    assert "copy_span_max" in agg
