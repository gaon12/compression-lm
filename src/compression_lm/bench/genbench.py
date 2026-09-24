"""Generation benchmarks: equal-search and equal-compute.

Equal-search gives every backend the same search budget (beam 16 x
lookahead 16, 32 KiB provided context, 256-byte outputs, same prompts,
seed, temperature). Equal-compute instead fixes the *candidate
evaluation* budget per output byte — evals/byte ~ beam x |alphabet|, so
the budget maps onto beam width — and lets each backend pick its
lookahead from a small grid, tuned on validation data only.

Every generated output is measured against the train corpus for copy
metrics (longest match, n-gram overlap, novelty, copy-run), UTF-8
validity, self-repetition, and the compressor's own
``generated_compression_bpb`` — explicitly distinct from the
teacher-forced pseudo BPB.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from pathlib import Path

from ..backends.base import Backend
from ..engine import GenerationConfig, generate
from .copymetrics import copy_metrics, distinct_ratio, self_repetition, utf8_stats

#: Verified to occur in the tiny_shakespeare *train* split (checked against
#: the corpus — speaker names like ANTONIO/PROSPERO only appear in the test
#: region and would leak held-out content if used as prompts).
DEFAULT_PROMPTS: tuple[bytes, ...] = (
    b"ROMEO:\n",
    b"JULIET:\n",
    b"MENENIUS:\n",
    b"CORIOLANUS:\n",
    b"BRUTUS:\n",
    b"LEONTES:\n",
    b"CAPULET:\n",
    b"MERCUTIO:\n",
    b"SICINIUS:\n",
    b"PAULINA:\n",
    b"My lord, ",
    b"I am ",
)

#: Lookahead choices each backend may pick under an eval budget.
LOOKAHEAD_GRID: tuple[int, ...] = (8, 16, 32)


def validate_prompts(prompts: Sequence[bytes], corpus: bytes) -> list[bytes]:
    """Keep only prompts actually occurring in ``corpus`` (with a warning)."""
    ok = [p for p in prompts if p in corpus]
    return ok


def eval_budget_to_beam(budget_per_byte: int, alphabet_size: int) -> int:
    """Beam width that spends ~``budget_per_byte`` evaluations per byte.

    Per output byte the engine evaluates ~beam x |alphabet| candidates
    (the lookahead factor cancels against the committed span length), so
    beam = budget / alphabet.
    """
    return max(1, int(budget_per_byte) // max(1, alphabet_size))


def tune_lookahead(
    backend: Backend,
    corpus: bytes,
    prompts: Sequence[bytes],
    *,
    beam_width: int,
    out_len: int = 128,
    workers: int = 1,
    seed: int = 42,
    backend_options: dict | None = None,
    lookahead_grid: Sequence[int] = LOOKAHEAD_GRID,
) -> tuple[int, dict]:
    """Pick a lookahead on *validation* prompts under a fixed eval budget.

    Score per lookahead = mean(corpus_overlap_16) - self_repetition_8:
    rewards coherent corpus-grounded output, penalizes loops. Returns the
    chosen lookahead plus an audit dict of scores used for the decision.
    """
    scores: dict[int, float] = {}
    for la in lookahead_grid:
        cfg = GenerationConfig(
            length=out_len,
            beam_width=beam_width,
            lookahead=la,
            temperature=0.0,
            seed=seed,
            workers=workers,
            backend_options=dict(backend_options or {}),
        )
        vals = []
        for prompt in prompts:
            res = generate(backend, corpus, prompt, cfg)
            ov = copy_metrics(res.text, corpus)["overlap_16gram"]
            vals.append(ov - self_repetition(res.text))
        scores[la] = statistics.fmean(vals)
    best = max(scores, key=lambda k: scores[k])
    return best, {"grid": list(lookahead_grid), "val_scores": scores}


def measure_output(text: bytes, corpus: bytes, bpb: float) -> dict:
    """Copy/UTF-8/repetition metrics for one generated output."""
    row = copy_metrics(text, corpus)
    row.update(utf8_stats(text))
    row["self_rep_8"] = self_repetition(text)
    row["distinct_4"] = distinct_ratio(text)
    row["generated_compression_bpb"] = bpb
    return row


def run_generation(
    backend: Backend,
    corpus: bytes,
    prompts: Sequence[bytes],
    cfg: GenerationConfig,
    *,
    mode: str,
    out_dir: Path | None = None,
    context_provided: int = 0,
    file_tag: str = "",
) -> list[dict]:
    """Generate one output per prompt and collect per-output rows.

    Returns one dict per (backend, prompt). Exceptions become status=error
    rows so one backend failure never aborts the sweep.
    """
    rows: list[dict] = []
    eff_ctx = min(cfg.context_bytes or backend.context_limit, backend.context_limit)
    for idx, prompt in enumerate(prompts):
        row = {
            "backend": backend.name,
            "mode": mode,
            "prompt": prompt.decode("utf-8", errors="replace"),
            "beam_width": cfg.beam_width,
            "lookahead": cfg.lookahead,
            "context_provided": context_provided or eff_ctx,
            "context_effective": eff_ctx,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
            "status": "ok",
            "note": "",
        }
        try:
            res = generate(backend, corpus, prompt, cfg)
        except Exception as exc:  # record, don't abort the sweep
            row["status"] = "error"
            row["note"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        row["seconds"] = res.elapsed
        row["out_bytes"] = len(res.text)
        row["evals"] = res.evaluations
        row["evals_per_byte"] = res.evaluations / max(1, len(res.text))
        row["bytes_per_sec"] = res.bytes_per_second
        row["evals_per_sec"] = res.evaluations / res.elapsed if res.elapsed else 0.0
        row.update(measure_output(res.text, corpus, res.bits_per_byte))
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{file_tag}_" if file_tag else ""
            fname = f"{mode}_{tag}{backend.name}_{idx:02d}.txt"
            (out_dir / fname).write_bytes(res.text)
            row["output_file"] = str(out_dir / fname)
        rows.append(row)
    return rows


def aggregate_generation(rows: Sequence[dict], *, mode: str) -> dict:
    """Per-backend aggregate over a mode's prompt rows (ok rows only)."""
    ok = [r for r in rows if r.get("status") == "ok" and r.get("mode") == mode]
    if not ok:
        return {
            "backend": rows[0]["backend"] if rows else "",
            "mode": mode,
            "status": "error",
            "n_prompts": 0,
        }
    keys = [
        "out_bytes",
        "evals",
        "evals_per_byte",
        "bytes_per_sec",
        "evals_per_sec",
        "generated_compression_bpb",
        "copy_span_max",
        "copy_span_mean",
        "copy_span_median",
        "copy_run",
        "self_rep_8",
        "distinct_4",
        "utf8_valid",
        "utf8_invalid_sequences",
        "utf8_chars",
        "utf8_repl_chars",
    ]
    keys += [k for k in ok[0] if k.startswith(("overlap_", "novel_"))]
    agg = {
        "backend": ok[0]["backend"],
        "mode": mode,
        "status": "ok",
        "n_prompts": len(ok),
        "beam_width": ok[0]["beam_width"],
        "lookahead": ok[0]["lookahead"],
        "context_provided": ok[0]["context_provided"],
        "context_effective": ok[0]["context_effective"],
        "temperature": ok[0]["temperature"],
        "seed": ok[0]["seed"],
        "seconds": sum(r["seconds"] for r in ok),
    }
    for k in keys:
        vals = [r[k] for r in ok if k in r]
        if vals:
            agg[k] = statistics.fmean(vals)
    return agg
