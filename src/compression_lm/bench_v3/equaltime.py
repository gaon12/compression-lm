"""Equal-time generation (spec §14/§15) — real wall-clock budgets.

Every backend gets the same seconds per prompt; whatever was produced in
budget is what gets measured. Backends run sequentially — no cross-backend
contention — and each row records the actual runtime so overshoot is
visible instead of hidden.

Search-parameter tuning happens on VALIDATION only, per spec §15: for
each (beam, horizon) grid point we generate a short continuation and score
it with a *discrimination proxy* — how many hard negatives the generated
span outscores under the same backend. Generated-BPB is never the target.
Degenerate periodic output is penalized by the adjacent-repeat gate.
Backends too slow to emit >=16 bytes inside the tuning slice are marked
``not tunable`` and fall back to default params (recorded).
"""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from ..backends.base import Backend
from ..bench.genbench import measure_output
from ..engine import GenerationConfig, generate
from .discriminate import _hard_negatives, _random_negatives
from .repetition import repetition_metrics

TIME_BUDGETS: tuple[float, ...] = (10.0, 30.0)


def _disc_proxy(
    backend: Backend,
    corpus: bytes,
    prompt: bytes,
    span: bytes,
    seed: int,
    backend_options: dict | None = None,
) -> tuple[float, int]:
    """Fraction of hard negatives the generated span outscores.

    Scores ``span`` against same-length negatives sharing its first bytes;
    fluent continuations should beat corpus near-misses, degenerate ones
    should not. Returns (win_fraction, n_negatives). Needs >=2 bytes.
    """
    n = len(span)
    if n < 2:
        return 0.0, 0
    context = corpus + prompt
    scorer = backend.prepare(
        context[-min(len(context), backend.context_limit) :],
        **dict(backend_options or {}),
    )
    # hard negatives share the GENERATED span's own first 1-2 bytes —
    # near-misses the span must beat to count as fluent (spec §15).
    rng = random.Random(f"{seed}:{n}:{len(span)}")
    negs = _hard_negatives(corpus, span, 15, rng, forbid=-1)
    if len(negs) < 15:
        negs += _random_negatives(corpus, n, 15 - len(negs), rng, (-1, -1))
    negs = [g for g in negs if g != span][:15]
    if not negs:
        return 0.0, 0
    s_span = scorer.measure(span)
    s_negs = scorer.measure_many(negs)
    wins = sum(1 for s in s_negs if s_span <= s)
    return wins / len(negs), len(negs)


def tune_search_params(
    backend: Backend,
    corpus: bytes,
    val_prompts: Sequence[bytes],
    *,
    tune_budget_sec: float,
    grid: Sequence[tuple[int, int]],
    seed: int = 42,
    target_len: int = 48,
    workers: int = 1,
    backend_options: dict | None = None,
) -> tuple[tuple[int, int], list[dict]]:
    """Pick (beam, horizon) on validation — never on test results.

    ``corpus`` must be the VALIDATION split and ``val_prompts`` prompts that
    occur inside it; test data plays no role here (spec §15).
    Returns ``((beam, horizon), tuning_rows)``. Metric = discrimination
    proxy wins minus periodic-repeat penalty; ties prefer smaller beam.
    """
    rows: list[dict] = []
    for beam, horizon in grid:
        cfg = GenerationConfig(
            length=target_len,
            beam_width=beam,
            lookahead=horizon,
            temperature=0.0,
            time_budget_sec=tune_budget_sec,
            context_bytes=4096,  # parity with the benchmarked conditions
            seed=seed,
            workers=workers,
            backend_options=dict(backend_options or {}),
        )
        wins: list[float] = []
        reps: list[float] = []
        t0 = time.perf_counter()
        for prompt in val_prompts:
            res = generate(backend, corpus, prompt, cfg)
            if len(res.text) >= 16:
                w, _ = _disc_proxy(
                    backend, corpus, prompt, res.text, seed, backend_options
                )
                wins.append(w)
                reps.append(repetition_metrics(res.text)["rep_4gram"])
        row = {
            "backend": backend.name,
            "beam": beam,
            "horizon": horizon,
            "disc_proxy": statistics.fmean(wins) if wins else 0.0,
            "rep_4gram": statistics.fmean(reps) if reps else 0.0,
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
    if not rows:
        return (4, 8), rows
    best = max(rows, key=lambda r: (r["disc_proxy"] - r["rep_4gram"], -r["beam"]))
    return (int(best["beam"]), int(best["horizon"])), rows


def default_grid(name: str, slow: bool, very_slow: bool) -> list[tuple[int, int]]:
    """(beam, horizon) candidates sized to the backend's speed class."""
    if very_slow:
        return [(1, 4), (4, 4)]
    if slow:
        return [(1, 4), (4, 4), (4, 8)]
    return [(1, 4), (2, 8), (4, 8), (8, 16), (16, 16)]


def run_equal_time(
    backend: Backend,
    corpus: bytes,
    prompts: Sequence[bytes],
    *,
    budget_sec: float,
    beam: int,
    horizon: int,
    length: int = 256,
    workers: int = 1,
    seed: int = 42,
    dataset: str = "",
    out_dir: Path | None = None,
    backend_options: dict | None = None,
) -> list[dict]:
    """One row per prompt: real runtime, partial output allowed."""
    rows: list[dict] = []
    for idx, prompt in enumerate(prompts):
        cfg = GenerationConfig(
            length=length,
            beam_width=beam,
            lookahead=horizon,
            temperature=0.8,
            time_budget_sec=budget_sec,
            context_bytes=4096,  # same effective context for all backends
            workers=workers,
            seed=seed,
            backend_options=dict(backend_options or {}),
        )
        row = {
            "dataset": dataset,
            "backend": backend.name,
            "prompt_id": idx,
            "time_budget_sec": budget_sec,
            "beam_width": beam,
            "horizon": horizon,
            "status": "ok",
            "note": "",
        }
        try:
            res = generate(backend, corpus, prompt, cfg)
        except Exception as exc:
            row["status"] = "error"
            row["note"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        text = res.text
        row.update(
            {
                "actual_runtime_sec": res.elapsed,
                "generated_bytes": len(text),
                "candidate_evaluations": res.evaluations,
                "evals_per_sec": res.evaluations / res.elapsed if res.elapsed else 0.0,
                "bytes_per_sec": res.bytes_per_second,
                "completed": len(text) >= length,
            }
        )
        row.update(measure_output(text, corpus, res.bits_per_byte))
        row.update(repetition_metrics(text))
        row["copy_run_ratio"] = row["copy_run"] / max(1, len(text))
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"equal_time_{int(budget_sec)}s_{backend.name}_{idx:02d}.txt"
            (out_dir / fname).write_bytes(text)
            row["output_file"] = str(out_dir / fname)
        rows.append(row)
    return rows
