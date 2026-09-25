"""v3 benchmark orchestrator — checkpointed, resumable, dataset-aware.

Every task writes its rows to ``results_v3/raw/rows/{key}.json`` and marks
``raw/manifest.json``; a re-run skips finished keys (``--resume`` is the
default behavior, kept explicit on the CLI). Task keys carry the dataset,
backend, workload size, and experiment parameters so quick/debug runs can
never be mistaken for full ones.

Task inventory (spec §32 phases):

- ``options``                -> backend_options.csv
- ``zstdval:{ds}:{ctx}``     -> zstd mode comparison @1k (Phase 1)
- ``scorerval:{ds}:{be}``    -> §7 scorer validation @100+ contexts
- ``teacher:{ds}:{be}:{ctx}:{n}``   -> teacher_forced_1byte.csv (+ posrec)
- ``disc:{ds}:{be}:{ctx}:{n}:{L}:{neg}`` -> continuation_discrimination.csv
- ``rawpair:{ds}:{a}:{b}:{ctx}``   -> container_vs_raw.csv
- ``evalgen:{ds}:{be}:{budget}``   -> generation_equal_evaluation.csv
- ``eetime:{ds}:{be}:{sec}``       -> generation_equal_time.csv
- ``snappysweep:{ds}:{budget}``    -> snappy budget curve rows
- ``speed:{ds}:{be}``              -> speed.csv
"""

from __future__ import annotations

import contextlib
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from ..backends.base import Backend
from ..bench import datasets as bench_ds
from ..bench import genbench, instrument, reporting, teacher
from ..engine import GenerationConfig
from . import (
    backends_v3,
    discriminate,
    equaltime,
    rawpair,
    repetition,
    wikitext,
)
from . import (
    options as v3opts,
)

SLOW_BACKENDS = frozenset(
    {"brotli", "bzip2", "zstd_full", "zstd_raw", "zstd_stream", "zstd_dict_trained", "lzo"}
)
VERY_SLOW_BACKENDS = frozenset({"lzma", "lzma2", "xz"})

ALL_PHASES = frozenset(
    {"zstdval", "scorerval", "teacher", "disc", "rawpair", "evalgen", "eetime", "snappy", "speed"}
)

TEACHER_CTXS: tuple[int, ...] = (4096, 32768, 262144)
DISC_CTX = 4096
EVAL_BUDGETS: tuple[int, ...] = (2000, 10000)
SNAPPY_SWEEP_BUDGETS: tuple[int, ...] = (500, 1000, 2000, 5000, 10000, 20000)
SNAPPY_LENGTHS = discriminate.SNAPPY_LENGTHS


@dataclass
class Bench3Config:
    out_dir: Path = Path("results_v3")
    data_dir: Path = Path("data")
    ts_path: Path = Path("data/tiny_shakespeare.txt")
    dataset: str = "tiny_shakespeare"  # tiny_shakespeare | wikitext2 | all
    seed: int = 42
    gap: int = 4096
    n_positions: int | None = None  # None -> tiered full (10k/2k/1k)
    n_prompts: int = 10
    gen_length: int = 256
    beam: int = 16
    lookahead: int = 16
    temperature: float = 0.8
    workers: int = 8
    time_budgets: tuple[float, ...] = equaltime.TIME_BUDGETS
    eval_budgets: tuple[int, ...] = EVAL_BUDGETS
    disc_lengths: tuple[int, ...] = discriminate.CONTINUATION_LENGTHS
    teacher_ctxs: tuple[int, ...] = TEACHER_CTXS
    backend_names: tuple[str, ...] = ()
    phases: frozenset[str] | tuple[str, ...] = ALL_PHASES
    resume: bool = True  # manifest always resumes; flag documents intent
    skip: frozenset[str] = frozenset()
    assemble_only: bool = False
    zip_after: bool = False
    command: str = ""  # verbatim CLI invocation, recorded for reproducibility
    raw_dir: Path = field(init=False)

    def __post_init__(self):
        self.raw_dir = self.out_dir / "raw"


class _Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.done: set[str] = set()
        if path.exists():
            try:
                self.done = set(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                self.done = set()

    def mark(self, key: str) -> None:
        self.done.add(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(self.done), indent=1), encoding="utf-8")


def _row_path(cfg: Bench3Config, task_key: str) -> Path:
    safe = task_key.replace(":", "__").replace("/", "_")
    return cfg.raw_dir / "rows" / f"{safe}.json"


def _save_rows(cfg: Bench3Config, task_key: str, rows: list[dict]) -> None:
    reporting.write_json(_row_path(cfg, task_key), rows)


def _n_positions(cfg: Bench3Config, backend: str) -> int:
    """Tiered position counts; all are prefixes of one canonical list."""
    base = cfg.n_positions if cfg.n_positions is not None else 10000
    if backend in VERY_SLOW_BACKENDS:
        return max(1, base // 10)
    if backend in SLOW_BACKENDS:
        return max(1, base // 5)
    return base


def _n_disc_positions(cfg: Bench3Config, backend: str) -> int:
    if backend in VERY_SLOW_BACKENDS:
        return min(200, _n_positions(cfg, backend))
    if backend in SLOW_BACKENDS:
        return min(500, _n_positions(cfg, backend))
    return min(1000, _n_positions(cfg, backend))


def _n_prompts(cfg: Bench3Config, backend: str) -> int:
    if backend in VERY_SLOW_BACKENDS:
        return min(cfg.n_prompts, 2)
    if backend in SLOW_BACKENDS:
        return min(cfg.n_prompts, 4)
    return min(cfg.n_prompts, 10)


def _eff_ctx(backend: Backend, provided: int) -> int:
    """Effective context == provided, capped by the backend's own limit."""
    return min(provided, backend.context_limit)


def _prompts_from_train(train: bytes, n: int) -> list[bytes]:
    """Corpus-validated prompts: character-name lines sampled from TRAIN."""
    prompts = genbench.validate_prompts(
        [
            b"JULIET:\n",
            b"ROMEO:\n",
            b"KING:\n",
            b"To be, or not to be",
            b"MACBETH:\n",
            b"CORIOLANUS:\n",
            b"ANTONIO:\n",
            b"PROSPERO:\n",
            b"FALSTAFF:\n",
            b"HENRY:\n",
        ],
        train,
    )
    if len(prompts) < n:
        tail = train[-4096:].split(b"\n")
        for line in tail:
            cand = line.strip()[:24]
            if len(cand) >= 6:
                prompts.append(cand + b"\n")
            if len(prompts) >= n:
                break
    return prompts[: max(1, n)]


# ---------------------------------------------------------------- tasks


def _teacher_task(
    cfg: Bench3Config,
    ds: bench_ds.Dataset,
    name: str,
    backend: Backend,
    ctx: int,
    positions: list[int],
    manifest: _Manifest,
) -> None:
    n = _n_positions(cfg, name)
    key = f"teacher:{ds.name}:{name}:{ctx}:{n}"
    if key in manifest.done:
        return
    try:
        recs = teacher.evaluate_positions(
            backend,
            ds.source,
            positions[:n],
            ctx,
            workers=cfg.workers,
            backend_options=v3opts.prepare_options(name),
        )
        row = teacher.aggregate_records(
            recs,
            backend=name,
            family=backend.family,
            mode="teacher_forced_1byte",
            context_provided=ctx,
            context_effective=_eff_ctx(backend, ctx),
        )
        row.update(
            {
                "dataset": ds.name,
                "scoring_mode": v3opts.BACKEND_OPTIONS[name]["scoring_mode"],
                "task": "teacher",
                "status": "ok",
                "context_bytes": _eff_ctx(backend, ctx),
            }
        )
        # per-position records -> bootstrap CIs + quantization detail
        reporting.write_json(
            cfg.raw_dir / "posrec" / f"{ds.name}__{name}__{ctx}__{n}.json",
            [r.__dict__ for r in recs],
        )
        rows = [row]
    except Exception as exc:
        rows = [
            {
                "dataset": ds.name,
                "backend": name,
                "scoring_mode": v3opts.BACKEND_OPTIONS[name]["scoring_mode"],
                "task": "teacher",
                "status": "error",
                "context_provided": ctx,
                "context_bytes": _eff_ctx(backend, ctx),
                "n_positions": n,
                "note": f"{type(exc).__name__}: {exc}",
            }
        ]
    _save_rows(cfg, key, rows)
    manifest.mark(key)


def _scoreval_task(
    cfg: Bench3Config,
    ds: bench_ds.Dataset,
    name: str,
    backend: Backend,
    positions: list[int],
    manifest: _Manifest,
) -> None:
    """§7 scorer validation: score spread + all-tied rate on >=100 ctx."""
    key = f"scorerval:{ds.name}:{name}"
    if key in manifest.done:
        return
    ctx = 4096
    subset = positions[: max(100, min(200, len(positions)))]
    try:
        recs = teacher.evaluate_positions(
            backend,
            ds.source,
            subset,
            ctx,
            workers=cfg.workers,
            backend_options=v3opts.prepare_options(name),
        )
        # candidate-score spread: re-score 16 contexts keeping all 256 scores
        opts = v3opts.prepare_options(name)
        eff = _eff_ctx(backend, ctx)
        all_scores: list[int] = []
        hist_counts: dict[int, int] = {}
        for p in subset[:16]:
            sc = backend.prepare(ds.source[max(0, p - eff) : p], **opts)
            for s in sc.measure_many(teacher.ALL_BYTES):
                all_scores.append(s)
                hist_counts[s] = hist_counts.get(s, 0) + 1
        uniq = [r.unique_scores for r in recs]
        all_tied = sum(1 for u in uniq if u == 1) / len(uniq)
        warn = all_tied >= 0.9
        rows = [
            {
                "dataset": ds.name,
                "backend": name,
                "scoring_mode": v3opts.BACKEND_OPTIONS[name]["scoring_mode"],
                "task": "scoring_validate",
                "status": "ok",
                "context_bytes": eff,
                "n_contexts": len(recs),
                "n_candidates": 256,
                "unique_scores_mean": statistics.fmean(uniq),
                "unique_scores_min": min(uniq),
                "unique_scores_max": max(uniq),
                "score_min": min(all_scores) if all_scores else "",
                "score_max": max(all_scores) if all_scores else "",
                "score_std": statistics.pstdev(all_scores) if len(all_scores) > 1 else 0.0,
                "score_hist_top": json.dumps(
                    sorted(hist_counts.items(), key=lambda kv: -kv[1])[:10]
                ),
                "all_tied_fraction": all_tied,
                "true_score_mean": statistics.fmean(r.true_score for r in recs),
                "warning": "WARNING: backend scoring is effectively "
                "non-discriminative for 1-byte candidates"
                if warn
                else "",
                "note": "",
            }
        ]
    except Exception as exc:
        rows = [
            {
                "dataset": ds.name,
                "backend": name,
                "task": "scoring_validate",
                "status": "error",
                "note": f"{type(exc).__name__}: {exc}",
            }
        ]
    _save_rows(cfg, key, rows)
    manifest.mark(key)


def _zstdval_task(
    cfg: Bench3Config,
    ds: bench_ds.Dataset,
    zstd_backends: dict[str, Backend],
    positions: list[int],
    manifest: _Manifest,
) -> None:
    """Phase-1: compare the zstd scoring modes on the same 1k positions."""
    n = min(1000, len(positions))
    for ctx in (4096, 32768):
        key = f"zstdval:{ds.name}:{ctx}:{n}"
        if key in manifest.done:
            continue
        rows = []
        for name, backend in zstd_backends.items():
            try:
                recs = teacher.evaluate_positions(
                    backend,
                    ds.source,
                    positions[:n],
                    ctx,
                    workers=cfg.workers,
                    backend_options=v3opts.prepare_options(name),
                )
                row = teacher.aggregate_records(
                    recs,
                    backend=name,
                    family=backend.family,
                    mode="zstd_mode_validation",
                    context_provided=ctx,
                    context_effective=_eff_ctx(backend, ctx),
                )
                row.update(
                    {
                        "dataset": ds.name,
                        "scoring_mode": v3opts.BACKEND_OPTIONS[name]["scoring_mode"],
                        "task": "zstd_validation",
                        "status": "ok",
                    }
                )
                reporting.write_json(
                    cfg.raw_dir / "posrec" / f"{ds.name}__{name}__{ctx}__{n}__val.json",
                    [r.__dict__ for r in recs],
                )
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "dataset": ds.name,
                        "backend": name,
                        "task": "zstd_validation",
                        "status": "error",
                        "context_provided": ctx,
                        "note": f"{type(exc).__name__}: {exc}",
                    }
                )
        _save_rows(cfg, key, rows)
        manifest.mark(key)


def _disc_task(
    cfg: Bench3Config,
    ds: bench_ds.Dataset,
    name: str,
    backend: Backend,
    n_len: int,
    neg: str,
    positions: list[int],
    manifest: _Manifest,
) -> None:
    n = _n_disc_positions(cfg, name)
    key = f"disc:{ds.name}:{name}:{DISC_CTX}:{n}:{n_len}:{neg}"
    if key in manifest.done:
        return
    try:
        recs = discriminate.evaluate_discrimination(
            backend,
            ds.source,
            positions[:n],
            DISC_CTX,
            n_len,
            neg_type=neg,
            seed=cfg.seed,
            backend_options=v3opts.prepare_options(name),
            workers=cfg.workers,
        )
        row = discriminate.aggregate_disc(
            recs,
            dataset=ds.name,
            backend=name,
            scoring_mode=v3opts.BACKEND_OPTIONS[name]["scoring_mode"],
            context_bytes=DISC_CTX,
            continuation_bytes=n_len,
            neg_type=neg,
        )
        row["context_effective"] = _eff_ctx(backend, DISC_CTX)
        row["task"] = "discrimination"
        row["status"] = "ok"
        reporting.write_json(
            cfg.raw_dir / "discpos" / f"{ds.name}__{name}__{n_len}__{neg}__{len(recs)}.json",
            [r.__dict__ for r in recs],
        )
        rows = [row]
    except Exception as exc:
        rows = [
            {
                "dataset": ds.name,
                "backend": name,
                "task": "discrimination",
                "status": "error",
                "context_bytes": DISC_CTX,
                "continuation_bytes": n_len,
                "negative_type": neg,
                "note": f"{type(exc).__name__}: {exc}",
            }
        ]
    _save_rows(cfg, key, rows)
    manifest.mark(key)


def _rawpair_task(
    cfg: Bench3Config,
    ds: bench_ds.Dataset,
    backends: dict[str, Backend],
    pair: tuple[str, str],
    positions: list[int],
    manifest: _Manifest,
) -> None:
    a, b = pair
    ctx = 4096
    n = 1000 if a not in VERY_SLOW_BACKENDS and b not in VERY_SLOW_BACKENDS else 200
    key = f"rawpair:{ds.name}:{a}:{b}:{ctx}:{n}"
    if key in manifest.done:
        return
    be_a, be_b = backends[a], backends[b]
    # xz-tier backends get a deterministic 64-candidate subset (cost cap).
    if n == 200:
        subset = sorted(range(256), key=lambda i: (i * 2654435761) % 256)[:64]
        candidates = [bytes([i]) for i in subset]
    else:
        candidates = teacher.ALL_BYTES
    pos = positions[:n]
    try:
        _, agg = rawpair.compare_pair(
            be_a,
            be_b,
            ds.source,
            pos,
            ctx,
            candidates=candidates,
            opts_a=v3opts.prepare_options(a),
            opts_b=v3opts.prepare_options(b),
        )
        sizes = rawpair.empty_frame_sizes([a, b], v3opts.BACKEND_OPTIONS)
        row = {
            "dataset": ds.name,
            "backend_a": a,
            "backend_b": b,
            "context_bytes": ctx,
            "task": "rawpair",
            "status": "ok",
            "empty_bytes_a": sizes.get(a),
            "empty_bytes_b": sizes.get(b),
            "note": "64-candidate subset" if len(candidates) != 256 else "",
        }
        row.update(agg)
        rows = [row]
    except Exception as exc:
        rows = [
            {
                "dataset": ds.name,
                "backend_a": a,
                "backend_b": b,
                "task": "rawpair",
                "status": "error",
                "note": f"{type(exc).__name__}: {exc}",
            }
        ]
    _save_rows(cfg, key, rows)
    manifest.mark(key)


def _evalgen_task(
    cfg: Bench3Config,
    ds: bench_ds.Dataset,
    name: str,
    backend: Backend,
    prompts: list[bytes],
    budget: int,
    manifest: _Manifest,
) -> None:
    """Equal-evaluation generation (v2 semantics, renamed per spec §13)."""
    np_ = _n_prompts(cfg, name)
    out_len = cfg.gen_length if name not in VERY_SLOW_BACKENDS else min(64, cfg.gen_length)
    key = f"evalgen:{ds.name}:{name}:{budget}:{np_}x{out_len}"
    if key in manifest.done:
        return
    try:
        alphabet = len(set(ds.train))
        beam = genbench.eval_budget_to_beam(budget, alphabet)
        # tune lookahead on VALIDATION only — never on test outcomes
        val_prompts = [p for p in genbench.DEFAULT_PROMPTS if p in ds.val][:1] or [b"the "]
        tune_len = 16 if name in VERY_SLOW_BACKENDS else 32 if name in SLOW_BACKENDS else 64
        lookahead, tune = genbench.tune_lookahead(
            backend,
            ds.val,
            val_prompts,
            beam_width=beam,
            out_len=tune_len,
            workers=cfg.workers,
            seed=cfg.seed,
            backend_options=v3opts.prepare_options(name),
        )
        rows = genbench.run_generation(
            backend,
            ds.train,
            prompts[:np_],
            GenerationConfig(
                length=out_len,
                beam_width=beam,
                lookahead=lookahead,
                temperature=cfg.temperature,
                context_bytes=DISC_CTX,  # same effective context for all backends
                workers=cfg.workers,
                seed=cfg.seed,
                backend_options=v3opts.prepare_options(name),
            ),
            mode="equal_evaluation_budget",
            out_dir=cfg.out_dir / "generations",
            context_provided=DISC_CTX,
            file_tag=str(budget),
        )
        for r in rows:
            r["dataset"] = ds.name
            r["budget_per_byte"] = budget
            r["copy_run_ratio"] = r.get("copy_run", 0) / max(1, r.get("out_bytes", 1))
            r["tuning"] = json.dumps(tune)
            r["task"] = "evalgen"
        agg = genbench.aggregate_generation(rows, mode="equal_evaluation_budget")
        agg.update(
            {
                "prompt": "__mean__",
                "dataset": ds.name,
                "budget_per_byte": budget,
                "task": "evalgen",
                "tuning": json.dumps(tune),
            }
        )
        if agg.get("out_bytes"):
            agg["copy_run_ratio"] = agg.get("copy_run", 0) / max(1, agg["out_bytes"])
        rows.append(agg)
    except Exception as exc:
        rows = [
            {
                "dataset": ds.name,
                "backend": name,
                "task": "evalgen",
                "status": "error",
                "budget_per_byte": budget,
                "note": f"{type(exc).__name__}: {exc}",
            }
        ]
    _save_rows(cfg, key, rows)
    manifest.mark(key)


def _eetime_task(
    cfg: Bench3Config,
    ds: bench_ds.Dataset,
    name: str,
    backend: Backend,
    prompts: list[bytes],
    budget_sec: float,
    manifest: _Manifest,
) -> None:
    np_ = _n_prompts(cfg, name)
    key = f"eetime:{ds.name}:{name}:{int(budget_sec)}:{np_}"
    if key in manifest.done:
        return
    try:
        very_slow = name in VERY_SLOW_BACKENDS
        slow = name in SLOW_BACKENDS
        grid = equaltime.default_grid(name, slow, very_slow)
        tune_budget = max(2.0, budget_sec / 5.0)
        # tuning happens on VALIDATION only (spec §15) — corpus and prompts
        # both come from ds.val; test outputs never influence the choice.
        val_prompts = _prompts_from_train(ds.val, 1 if very_slow else 2)
        (beam, horizon), tune_rows = equaltime.tune_search_params(
            backend,
            ds.val,
            val_prompts,
            tune_budget_sec=tune_budget,
            grid=grid,
            seed=cfg.seed,
            workers=cfg.workers,
            backend_options=v3opts.prepare_options(name),
        )
        rows = equaltime.run_equal_time(
            backend,
            ds.train,
            prompts[:np_],
            budget_sec=budget_sec,
            beam=beam,
            horizon=horizon,
            length=cfg.gen_length,
            workers=cfg.workers,
            seed=cfg.seed,
            dataset=ds.name,
            out_dir=cfg.out_dir / "generations",
            backend_options=v3opts.prepare_options(name),
        )
        for r in rows:
            r["task"] = "eetime"
        reporting.write_json(
            cfg.raw_dir / "tuning" / f"{ds.name}__{name}__{int(budget_sec)}.json",
            tune_rows,
        )
    except Exception as exc:
        rows = [
            {
                "dataset": ds.name,
                "backend": name,
                "task": "eetime",
                "status": "error",
                "time_budget_sec": budget_sec,
                "note": f"{type(exc).__name__}: {exc}",
            }
        ]
    _save_rows(cfg, key, rows)
    manifest.mark(key)


def _snappysweep_task(
    cfg: Bench3Config,
    ds: bench_ds.Dataset,
    backend: Backend,
    prompts: list[bytes],
    budget: int,
    manifest: _Manifest,
) -> None:
    """§16 snappy retrieval-vs-budget curve."""
    name = "snappy"
    np_ = min(cfg.n_prompts, 4)
    key = f"snappysweep:{ds.name}:{budget}:{np_}x{cfg.gen_length}"
    if key in manifest.done:
        return
    try:
        alphabet = len(set(ds.train))
        beam = genbench.eval_budget_to_beam(budget, alphabet)
        rows = genbench.run_generation(
            backend,
            ds.train,
            prompts[:np_],
            GenerationConfig(
                length=cfg.gen_length,
                beam_width=beam,
                lookahead=16,
                temperature=cfg.temperature,
                context_bytes=DISC_CTX,
                workers=cfg.workers,
                seed=cfg.seed,
                backend_options=v3opts.prepare_options(name),
            ),
            mode="snappy_budget_sweep",
            out_dir=cfg.out_dir / "generations",
            context_provided=DISC_CTX,
            file_tag=f"sweep{budget}",
        )
        for r in rows:
            r["dataset"] = ds.name
            r["budget_per_byte"] = budget
            r["copy_run_ratio"] = r.get("copy_run", 0) / max(1, r.get("out_bytes", 1))
            r["task"] = "snappysweep"
        agg = genbench.aggregate_generation(rows, mode="snappy_budget_sweep")
        agg.update(
            {
                "prompt": "__mean__",
                "dataset": ds.name,
                "budget_per_byte": budget,
                "task": "snappysweep",
            }
        )
        if agg.get("out_bytes"):
            agg["copy_run_ratio"] = agg.get("copy_run", 0) / max(1, agg["out_bytes"])
        rows.append(agg)
    except Exception as exc:
        rows = [
            {
                "dataset": ds.name,
                "backend": "snappy",
                "task": "snappysweep",
                "status": "error",
                "budget_per_byte": budget,
                "note": f"{type(exc).__name__}: {exc}",
            }
        ]
    _save_rows(cfg, key, rows)
    manifest.mark(key)


def _speed_task(
    cfg: Bench3Config,
    ds: bench_ds.Dataset,
    name: str,
    backend: Backend,
    manifest: _Manifest,
) -> None:
    key = f"speed:{ds.name}:{name}"
    if key in manifest.done:
        return
    eff = _eff_ctx(backend, 32768)
    try:
        row = instrument.speed_microbench(
            backend,
            ds.train[-eff:],
            n_evals=512 if name in VERY_SLOW_BACKENDS else 2048,
            backend_options=v3opts.prepare_options(name),
        )
        row.update({"dataset": ds.name, "backend": name, "task": "speed", "status": "ok"})
    except Exception as exc:
        row = {
            "dataset": ds.name,
            "backend": name,
            "task": "speed",
            "status": "error",
            "note": f"{type(exc).__name__}: {exc}",
        }
    _save_rows(cfg, key, [row])
    manifest.mark(key)


# ---------------------------------------------------------------- driver


def _load_dataset(cfg: Bench3Config, name: str) -> bench_ds.Dataset:
    if name == "wikitext2":
        return wikitext.load_wikitext2(cfg.data_dir, gap=cfg.gap, seed=cfg.seed)
    if name == "tiny_shakespeare":
        return bench_ds.load_tiny_shakespeare(cfg.ts_path, gap=cfg.gap, seed=cfg.seed)
    # treat --dataset as a path to a raw text file
    p = Path(name)
    return bench_ds.split_contiguous(
        p.read_bytes(), name=p.stem, gap=cfg.gap, seed=cfg.seed
    )


def _backend_map(cfg: Bench3Config, ds: bench_ds.Dataset) -> dict[str, Backend]:
    """Instantiate selected backends; unavailable ones are recorded."""
    names = list(cfg.backend_names) or [
        *sorted(backends_v3.v3_factories()),
        "zstd_dict_trained",
    ]
    out: dict[str, Backend] = {}
    for name in names:
        reason = None
        if name == "zstd_dict_trained":
            data = _train_zstd_dict(ds.train, cfg.raw_dir / f"zstd_dict_{ds.name}.bin")
            if data is None:
                reason = "zstandard dictionary training unavailable"
            else:
                out[name] = backends_v3.ZstdDictTrainedBackend(data)
                continue
        elif name in backends_v3.v3_factories():
            be, reason = backends_v3.check_backend(name)
            if be is None:
                reason = reason or "unavailable"
            else:
                out[name] = be
                continue
        else:
            reason = "unknown backend name"
        key = f"backend:{ds.name}:{name}"
        _save_rows(
            cfg,
            key,
            [
                {
                    "dataset": ds.name,
                    "backend": name,
                    "task": "backend",
                    "status": "skipped",
                    "note": reason,
                }
            ],
        )
    return out


def _train_zstd_dict(train: bytes, out_path: Path, size: int = 112 * 1024) -> bytes | None:
    if out_path.exists():
        return out_path.read_bytes()
    try:
        import zstandard

        chunk = 4096
        samples = [train[i : i + chunk] for i in range(0, len(train) - chunk, chunk)]
        d = zstandard.train_dictionary(size, samples)
        data = d.as_bytes()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return data
    except Exception:
        return None


def run(cfg: Bench3Config) -> Path:
    """Execute enabled phases for each selected dataset."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    if cfg.assemble_only:
        assemble(cfg)
        return cfg.out_dir
    manifest = _Manifest(cfg.raw_dir / "manifest.json")
    ds_names = (
        ["tiny_shakespeare", "wikitext2"] if cfg.dataset == "all" else [cfg.dataset]
    )
    for ds_name in ds_names:
        try:
            ds = _load_dataset(cfg, ds_name)
        except Exception as exc:
            _save_rows(
                cfg,
                f"dataset:{ds_name}",
                [
                    {
                        "dataset": ds_name,
                        "task": "dataset",
                        "status": "skipped",
                        "note": f"{type(exc).__name__}: {exc}",
                    }
                ],
            )
            manifest.mark(f"dataset:{ds_name}")
            continue
        reporting.write_json(cfg.raw_dir / f"dataset_{ds.name}.json", ds.split_meta())
        reporting.write_json(cfg.out_dir / f"config_{ds.name}.json", _cfg_meta(cfg, ds.name))
        # canonical shared position list (same seed -> same list as v2)
        list_n = max(10000, cfg.n_positions or 0)
        positions = bench_ds.sample_positions(ds.test_span, list_n, seed=cfg.seed)
        reporting.write_json(
            cfg.raw_dir / f"positions_{ds.name}.json",
            {"n": len(positions), "seed": cfg.seed, "positions": positions},
        )
        backends = _backend_map(cfg, ds)
        phases = set(cfg.phases) or ALL_PHASES
        zstd_modes = {k: v for k, v in backends.items() if k.startswith("zstd")}
        prompts = _prompts_from_train(ds.train, cfg.n_prompts)
        if "zstdval" in phases and zstd_modes:
            _zstdval_task(cfg, ds, zstd_modes, positions, manifest)
        for name, backend in backends.items():
            if "scorerval" in phases:
                _scoreval_task(cfg, ds, name, backend, positions, manifest)
            if "speed" in phases:
                _speed_task(cfg, ds, name, backend, manifest)
            if "teacher" in phases:
                # dedup by effective context: provided > backend window feeds
                # the scorer the identical bytes — run the smallest provided
                # ctx per distinct effective value instead of recomputing.
                by_eff: dict[int, int] = {}
                for ctx in sorted(cfg.teacher_ctxs):
                    by_eff.setdefault(_eff_ctx(backend, ctx), ctx)
                for ctx in sorted(by_eff.values()):
                    _teacher_task(cfg, ds, name, backend, ctx, positions, manifest)
            if "disc" in phases:
                lens = SNAPPY_LENGTHS if name == "snappy" else cfg.disc_lengths
                for n_len in lens:
                    for neg in ("random", "hard"):
                        _disc_task(cfg, ds, name, backend, n_len, neg, positions, manifest)
            if "evalgen" in phases:
                budgets = (
                    cfg.eval_budgets
                    if name not in SLOW_BACKENDS | VERY_SLOW_BACKENDS
                    else (min(cfg.eval_budgets),)
                )
                for budget in budgets:
                    _evalgen_task(cfg, ds, name, backend, prompts, budget, manifest)
            if "eetime" in phases:
                for sec in cfg.time_budgets:
                    _eetime_task(cfg, ds, name, backend, prompts, sec, manifest)
            if name == "snappy" and "snappy" in phases:
                for budget in SNAPPY_SWEEP_BUDGETS:
                    _snappysweep_task(cfg, ds, backend, prompts, budget, manifest)
        if "rawpair" in phases:
            for pair in rawpair.CONTAINER_PAIRS:
                if pair[0] in backends and pair[1] in backends:
                    _rawpair_task(cfg, ds, backends, pair, positions, manifest)
    assemble(cfg)
    return cfg.out_dir


# ---------------------------------------------------------------- assembly


def _posrec_path(cfg: Bench3Config, ds: str, be: str, ctx: int, n: int) -> Path:
    return cfg.raw_dir / "posrec" / f"{ds}__{be}__{ctx}__{n}.json"


def _posrec_find(cfg: Bench3Config, ds: str, be: str, ctx: int, n: int) -> Path | None:
    """Standard posrec name, else the zstd-mode-validation ``__val`` variant."""
    p = _posrec_path(cfg, ds, be, ctx, n)
    if p.exists():
        return p
    alt = cfg.raw_dir / "posrec" / f"{ds}__{be}__{ctx}__{n}__val.json"
    return alt if alt.exists() else None


def _teacher_cis(cfg: Bench3Config, row: dict) -> dict:
    """Bootstrap 95% CIs for top1_exp / mrr / pseudo_bpb from posrec."""
    from ..bench.teacher import _expected_topk
    from .stats3 import bootstrap_mean_ci

    path = _posrec_find(
        cfg,
        str(row.get("dataset")),
        str(row.get("backend")),
        int(row.get("context_provided") or 0),
        int(row.get("n_positions") or 0),
    )
    if path is None:
        return {}
    recs = json.loads(path.read_text(encoding="utf-8"))
    top1 = [_expected_topk(r["better_count"], r["true_tie_size"], 1) for r in recs]
    mrr = [1.0 / r["rank_avg"] for r in recs]
    bpb = [r["nll_bits"] for r in recs]
    _, lo1, hi1 = bootstrap_mean_ci(top1)
    _, lom, him = bootstrap_mean_ci(mrr)
    _, lob, hib = bootstrap_mean_ci(bpb)
    return {
        "top1_exp_ci": f"[{lo1:.3f},{hi1:.3f}]",
        "mrr_ci": f"[{lom:.3f},{him:.3f}]",
        "pseudo_bpb_ci": f"[{lob:.3f},{hib:.3f}]",
    }


def _disc_cis(cfg: Bench3Config, row: dict) -> dict:
    from ..bench.teacher import _expected_topk
    from .stats3 import bootstrap_mean_ci

    path = (
        cfg.raw_dir
        / "discpos"
        / f"{row.get('dataset')}__{row.get('backend')}__{row.get('continuation_bytes')}"
        f"__{row.get('negative_type')}__{row.get('num_positions')}.json"
    )
    if not path.exists():
        return {}
    recs = json.loads(path.read_text(encoding="utf-8"))
    top1 = [_expected_topk(r["better_count"], r["true_tie_size"], 1) for r in recs]
    mrr = [1.0 / r["true_rank_avg"] for r in recs]
    _, lo1, hi1 = bootstrap_mean_ci(top1)
    _, lom, him = bootstrap_mean_ci(mrr)
    return {"top1_ci": f"[{lo1:.3f},{hi1:.3f}]", "mrr_ci": f"[{lom:.3f},{him:.3f}]"}


def _discpos_all_tied(cfg: Bench3Config, row: dict) -> float | None:
    """Fraction of contexts whose 256 1-byte scores are all tied."""
    path = _posrec_find(
        cfg,
        str(row.get("dataset")),
        str(row.get("backend")),
        int(row.get("context_provided") or 0),
        int(row.get("n_positions") or 0),
    )
    if path is None:
        return None
    recs = json.loads(path.read_text(encoding="utf-8"))
    return sum(1 for r in recs if r.get("unique_scores") == 1) / max(1, len(recs))


def _cfg_meta(cfg: Bench3Config, dataset: str) -> dict:
    """Explicit settings for one dataset run — reproducibility record."""
    return {
        "dataset": dataset,
        "command": cfg.command,
        "seed": cfg.seed,
        "gap": cfg.gap,
        "teacher_ctxs": list(cfg.teacher_ctxs),
        "disc_ctx": DISC_CTX,
        "disc_lengths": list(cfg.disc_lengths),
        "eval_budgets": list(cfg.eval_budgets),
        "time_budgets": list(cfg.time_budgets),
        "n_positions": cfg.n_positions,
        "n_prompts": cfg.n_prompts,
        "gen_length": cfg.gen_length,
        "workers": cfg.workers,
        "phases": sorted(cfg.phases),
        "backend_names": list(cfg.backend_names) or "all",
        "snappy_sweep_budgets": list(SNAPPY_SWEEP_BUDGETS),
        "wikitext2_source": wikitext.URL,
        "package_versions": instrument.package_versions(),
        "env": reporting.environment_info(),
    }


def assemble(cfg: Bench3Config) -> None:
    """Write every CSV/JSON artifact from raw checkpoint rows."""
    rows = reporting.read_jsonl_rows(cfg.raw_dir)

    tf = [r for r in rows if r.get("task") == "teacher"]
    zval = [r for r in rows if r.get("task") == "zstd_validation"]
    for r in tf + zval:
        if r.get("status") == "ok":
            r.update(_teacher_cis(cfg, r))
            # spec §22 column names alongside the internal metric names
            r.setdefault("unique_scores_mean", r.get("mean_unique_scores"))
            r.setdefault("largest_tie_mean", r.get("mean_max_tie"))
            r.setdefault("contexts_per_sec", r.get("positions_per_sec"))
            r.setdefault("pseudo_bpb", r.get("teacher_forced_pseudo_bpb"))
            r.setdefault("pseudo_perplexity", r.get("pseudo_ppl"))
            if r.get("mean_unique_scores"):
                r.setdefault(
                    "avg_tie_group_size",
                    r.get("evals", 0) / max(1, r.get("n_positions", 1))
                    / r["mean_unique_scores"],
                )
            at = _discpos_all_tied(cfg, r)
            if at is not None:
                r["all_tied_fraction"] = at
                if at >= 0.9:
                    r["warning"] = (
                        "WARNING: backend scoring is effectively "
                        "non-discriminative for 1-byte candidates"
                    )
    reporting.write_csv(cfg.out_dir / "teacher_forced_1byte.csv", tf + zval)

    disc = [r for r in rows if r.get("task") == "discrimination"]
    for r in disc:
        if r.get("status") == "ok":
            r.update(_disc_cis(cfg, r))
    reporting.write_csv(cfg.out_dir / "continuation_discrimination.csv", disc)

    reporting.write_csv(
        cfg.out_dir / "quantization.csv",
        [
            {k: r.get(k) for k in (
                "dataset", "backend", "scoring_mode", "context_provided",
                "context_effective", "n_positions", "mean_unique_scores",
                "mean_max_tie", "tie_candidate_frac", "pct_true_tied",
                "pct_true_in_best_tie", "all_tied_fraction", "warning", "note",
            )}
            for r in tf + zval
            if r.get("status") == "ok"
        ],
    )
    reporting.write_csv(
        cfg.out_dir / "scoring_method_validation.csv",
        [r for r in rows if r.get("task") == "scoring_validate"],
    )
    reporting.write_csv(
        cfg.out_dir / "container_vs_raw.csv",
        [r for r in rows if r.get("task") == "rawpair"],
    )
    reporting.write_csv(
        cfg.out_dir / "generation_equal_evaluation.csv",
        [r for r in rows if r.get("task") in ("evalgen", "snappysweep")],
    )
    # spec §24 column names alongside the v2 metric names
    et_rows = [r for r in rows if r.get("task") == "eetime"]
    for r in et_rows:
        r.setdefault("longest_copy_run", r.get("copy_run"))
        r.setdefault("valid_utf8", r.get("utf8_valid"))
        for short, long_ in (
            ("overlap_8", "overlap_8gram"),
            ("overlap_16", "overlap_16gram"),
            ("overlap_32", "overlap_32gram"),
            ("novel_8", "novel_8gram"),
            ("novel_16", "novel_16gram"),
            ("novel_32", "novel_32gram"),
        ):
            r.setdefault(short, r.get(long_))
    reporting.write_csv(cfg.out_dir / "generation_equal_time.csv", et_rows)

    gen_rows = [
        r
        for r in rows
        if r.get("task") in ("evalgen", "snappysweep", "eetime")
        and r.get("status") == "ok"
        and r.get("prompt") != "__mean__"
    ]
    reporting.write_csv(
        cfg.out_dir / "copy_metrics.csv",
        [
            {k: r.get(k) for k in (
                "dataset", "backend", "task", "mode", "prompt", "prompt_id",
                "generated_compression_bpb", "copy_span_max", "copy_span_mean",
                "copy_span_median", "copy_run", "copy_run_ratio",
                "overlap_4gram", "overlap_8gram", "overlap_16gram",
                "overlap_32gram", "overlap_64gram", "novel_8gram",
                "novel_16gram", "novel_32gram", "utf8_valid",
            )}
            for r in gen_rows
        ],
    )
    # repetition on every generated artifact (recomputed from output files
    # for evalgen rows, carried inline for eetime rows)
    rep_rows = []
    for r in gen_rows:
        rep = {
            k: r.get(k)
            for k in (
                "rep_2gram", "rep_4gram", "rep_8gram", "unique_byte_ratio",
                "longest_periodic_run", "periodic_run_period", "byte_entropy",
            )
        }
        if rep.get("rep_4gram") is None and r.get("output_file"):
            with contextlib.suppress(OSError):
                rep = repetition.repetition_metrics(Path(r["output_file"]).read_bytes())
        rep_rows.append(
            {
                "dataset": r.get("dataset"),
                "backend": r.get("backend"),
                "task": r.get("task"),
                "mode": r.get("mode"),
                "prompt": r.get("prompt", r.get("prompt_id")),
                **rep,
            }
        )
    reporting.write_csv(cfg.out_dir / "repetition_metrics.csv", rep_rows)

    reporting.write_csv(
        cfg.out_dir / "speed.csv", [r for r in rows if r.get("task") == "speed"]
    )
    reporting.write_csv(cfg.out_dir / "backend_options.csv", v3opts.options_rows())
    reporting.write_csv(
        cfg.out_dir / "backend_status.csv",
        [
            r
            for r in rows
            if r.get("status") in ("skipped", "error")
        ],
    )

    cfg_meta = _cfg_meta(cfg, cfg.dataset)
    # Per-dataset run configs live in config_<ds>.json (written by run()).
    # --assemble-only never fabricates one — a missing file just means that
    # dataset's run predates config logging.
    cfg_path = cfg.out_dir / f"config_{cfg.dataset}.json"
    if not cfg.assemble_only and not cfg_path.exists():
        reporting.write_json(cfg_path, cfg_meta)
    runs = {}
    for p in sorted(cfg.out_dir.glob("config_*.json")):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ds_key = meta.get("dataset") or p.stem.removeprefix("config_")
        runs[ds_key] = meta
    reporting.write_json(
        cfg.out_dir / "config.json",
        {
            "env": cfg_meta["env"],
            "package_versions": cfg_meta["package_versions"],
            "runs": runs,
        },
    )

    # summary.csv: one headline row per backend (4KiB teacher + gen stats)
    tf4 = [
        r
        for r in tf
        if r.get("status") == "ok" and str(r.get("context_provided")) == "4096"
    ]
    gen_mean = [r for r in gen_rows if r.get("prompt") == "__mean__"]
    summ = []
    for r in tf4:
        g = next((x for x in gen_mean if x.get("backend") == r["backend"]), {})
        summ.append(
            {
                "dataset": r.get("dataset"),
                "backend": r["backend"],
                "scoring_mode": r.get("scoring_mode"),
                "tf_top1_exp": r.get("top1_exp"),
                "tf_top1_exp_ci": r.get("top1_exp_ci"),
                "tf_pseudo_bpb": r.get("teacher_forced_pseudo_bpb"),
                "tf_mrr": r.get("mrr"),
                "tf_mean_unique": r.get("mean_unique_scores"),
                "gen_bpb": g.get("generated_compression_bpb"),
                "gen_copy_run": g.get("copy_run"),
                "gen_overlap32": g.get("overlap_32gram"),
            }
        )
    reporting.write_csv(cfg.out_dir / "summary.csv", summ)

    try:
        from . import plots3

        plots3.make_plots(cfg.out_dir)
    except Exception as exc:
        print(f"plots failed: {exc}")
    try:
        from . import report3

        report3.build_report(cfg.out_dir)
    except Exception as exc:
        print(f"report failed: {exc}")
    if cfg.zip_after:
        _make_zip(cfg.out_dir)


def _make_zip(results_dir: Path) -> Path:
    import zipfile

    out = results_dir.parent / (results_dir.name + ".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(results_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(results_dir.parent))
    print(f"zip -> {out}")
    return out
