"""Benchmark orchestration with per-task checkpointing.

Task granularity: each (experiment, backend, params) triple is one task
writing a JSON row file under results/raw/rows/ plus a manifest entry.
A crash loses at most the task in flight; re-running with --resume skips
completed tasks. The assemble step merges row files into the final CSVs.

Feasibility tiering: one-shot scorers (bzip2, lzma family, zstd_dict,
snappy, lzo) pay O(context) per candidate, so they evaluate a *prefix
subset* of the shared position list and fewer prompts — always recorded
in n_positions / n_prompts, never hidden.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import backends
from ..backends.zstd_backend import ZstdDictBackend
from ..engine import GenerationConfig, corpus_alphabet
from . import genbench, instrument, plots, report, reporting, teacher
from .datasets import Dataset, load_tiny_shakespeare, sample_positions

#: Workload tiers by per-eval cost (measured, not guessed):
#: fast   — clone/dict scorers <0.1 ms/eval: full positions + all prompts
#: slow   — one-shot ~0.2-0.7 ms/eval: 1/5 positions, fewer prompts
#: vslow  — one-shot ~1-4 ms/eval: 1/10 positions, minimal prompts,
#:          64-byte outputs for equal-compute. Reductions are recorded in
#:          every row (n_positions / n_prompts), never silent.
SLOW_BACKENDS = frozenset({"brotli", "bzip2", "zstd_dict", "lzo"})
VERY_SLOW_BACKENDS = frozenset({"lzma", "lzma2", "xz"})

#: Effective-context caps for the context sweep: (cap_bytes, reason).
#: "window" = algorithmic limit; "compute" = benchmark-imposed cap.
CTX_CAPS: dict[str, tuple[int, str]] = {
    "deflate": (32768, "window: deflate 32 KiB match window"),
    "zlib": (32768, "window: deflate 32 KiB match window"),
    "gzip": (32768, "window: deflate 32 KiB match window"),
    "zstd": (1 << 20, "window: raw-content dict, sweep cap 1 MiB"),
    "zstd_dict": (32768, "compute: one-shot payload recompression"),
    "brotli": (32768, "compute: per-call dictionary attach cost"),
    "lz4": (65536, "window: LZ4 dictionary max 64 KiB"),
    "snappy": (32768, "compute: one-shot recompression"),
    "lzo": (16384, "compute: one-shot recompression"),
    "bzip2": (16384, "compute: one-shot recompression"),
    "lzma": (8192, "compute: one-shot recompression"),
    "lzma2": (8192, "compute: one-shot recompression"),
    "xz": (8192, "compute: one-shot recompression"),
}

ZSTD_DICT_SIZE = 110 * 1024


@dataclass
class BenchConfig:
    dataset: Path
    out_dir: Path = Path("results")
    mode: str = "quick"  # "quick" | "full"
    seed: int = 42
    workers: int = 8
    temperature: float = 0.8
    gen_length: int = 256
    beam: int = 16
    lookahead: int = 16
    context_sizes: tuple[int, ...] = (4096, 32768, 262144, 1048576)
    budgets: tuple[int, ...] = (2000, 10000)
    backend_names: tuple[str, ...] | None = None
    n_positions: int | None = None
    n_prompts: int = 10
    skip: frozenset[str] = frozenset()
    backend_options: dict[str, dict] = field(default_factory=dict)

    @property
    def raw_dir(self) -> Path:
        return self.out_dir / "raw"


def _n_positions(cfg: BenchConfig, backend: str) -> int:
    """Position count per backend tier; same prefix subset for all."""
    if cfg.n_positions is not None:
        base = cfg.n_positions
    elif cfg.mode == "full":
        base = 10000
    else:
        base = 1000
    if backend in VERY_SLOW_BACKENDS:
        return max(1, base // 10)  # 100 quick / 1000 full
    if backend in SLOW_BACKENDS:
        return max(1, base // 5)  # 200 quick / 2000 full
    return base


def _n_prompts(cfg: BenchConfig, backend: str) -> int:
    """Equal-search prompt counts (spec asks for >=10 in the final run)."""
    if backend in VERY_SLOW_BACKENDS:
        return min(cfg.n_prompts, 1 if cfg.mode == "quick" else 2)
    if backend in SLOW_BACKENDS:
        return min(cfg.n_prompts, 2 if cfg.mode == "quick" else 4)
    return min(cfg.n_prompts, 4 if cfg.mode == "quick" else 10)


def _n_compute_prompts(cfg: BenchConfig, backend: str, budget: int) -> int:
    """Equal-compute prompt counts: fewer than equal-search since each
    prompt at budget 2000 is already ~0.5M candidate evaluations."""
    if backend in VERY_SLOW_BACKENDS:
        return 1
    if backend in SLOW_BACKENDS:
        return min(cfg.n_prompts, 2)
    base = _n_prompts(cfg, backend)
    if budget > 2000:
        return max(1, base // 2)
    return base


def _compute_out_len(cfg: BenchConfig, backend: str) -> int:
    """Equal-compute keeps the same *per-byte* eval budget; the slowest
    tier emits 64 bytes so actual total evals stay feasible (recorded)."""
    if backend in VERY_SLOW_BACKENDS:
        return min(64, cfg.gen_length)
    return cfg.gen_length


def _eff_ctx(name: str, provided: int) -> tuple[int, str]:
    cap, reason = CTX_CAPS.get(name, (32768, "default cap"))
    return min(provided, cap), reason


class _Manifest:
    """Append-only set of completed task keys, persisted as JSON."""

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
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.done)), encoding="utf-8")
        tmp.replace(self.path)


def _row_path(cfg: BenchConfig, task_key: str) -> Path:
    safe = task_key.replace(":", "__").replace("/", "_")
    return cfg.raw_dir / "rows" / f"{safe}.json"


def _save_rows(cfg: BenchConfig, task_key: str, rows: list[dict]) -> None:
    reporting.write_json(_row_path(cfg, task_key), rows)


def _train_zstd_dict(train: bytes, out_path: Path, size: int = ZSTD_DICT_SIZE) -> bytes:
    """Train a zstd dictionary from TRAIN bytes only (cached on disk)."""
    if out_path.exists():
        return out_path.read_bytes()
    import zstandard

    chunk = 4096
    samples = [train[i : i + chunk] for i in range(0, len(train) - chunk, chunk)]
    d = zstandard.train_dictionary(size, samples)
    data = d.as_bytes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return data


def register_zstd_dict(cfg: BenchConfig, ds: Dataset) -> bool:
    """Build the train-only dictionary and register the zstd_dict backend."""
    if "zstd_dict" in backends.names():
        return True
    if not backends.info("zstd").available:
        return False
    try:
        data = _train_zstd_dict(ds.train, cfg.raw_dir / "zstd_dict.bin")
    except Exception:
        return False
    backends.register("zstd_dict", lambda: ZstdDictBackend(data))
    return True


def _selected_backends(cfg: BenchConfig) -> list[str]:
    names = list(cfg.backend_names) if cfg.backend_names else backends.names()
    return names


def _teacher_task(
    cfg: BenchConfig,
    ds: Dataset,
    name: str,
    provided_ctx: int,
    positions: list[int],
    manifest: _Manifest,
    eff_cache: dict,
) -> None:
    n = _n_positions(cfg, name)
    key = f"teacher:{name}:{provided_ctx}:{n}"
    if key in manifest.done:
        return
    eff, reason = _eff_ctx(name, provided_ctx)
    if eff in eff_cache and eff_cache[eff] != provided_ctx:
        # same effective context already measured — copy stats, mark dedup
        src = _row_path(cfg, f"teacher:{name}:{eff_cache[eff]}:{n}")
        if src.exists():
            rows = json.loads(src.read_text(encoding="utf-8"))
            for r in rows:
                r["context_provided"] = provided_ctx
                r["dedup"] = True
                r["note"] = (r.get("note") or "") + f" [dedup: effective context == {eff}]"
            _save_rows(cfg, key, rows)
            manifest.mark(key)
            return
    backend = backends.get(name)
    # instance-level sweep cap: zstd's dict mode can digest megabytes while
    # one-shot backends are compute-capped; generation keeps the class
    # default for 32 KiB parity with deflate.
    backend.context_limit = _eff_ctx(name, provided_ctx)[0]
    pos = positions[:n]
    backend_options = cfg.backend_options.get(name, {})
    recs = teacher.evaluate_positions(
        backend,
        ds.source,
        pos,
        eff,
        workers=cfg.workers,
        backend_options=backend_options,
    )
    info = backends.info(name)
    row = teacher.aggregate_records(
        recs,
        backend=name,
        family=info.family,
        mode=cfg.mode,
        context_provided=provided_ctx,
        context_effective=eff,
        note=f"cap: {reason}" if eff < provided_ctx else "",
    )
    row["dedup"] = False
    _save_rows(cfg, key, [row])
    # per-position detail for downstream quantization analysis
    reporting.write_json(
        cfg.raw_dir / "positions" / f"{name}_{provided_ctx}_{n}.json",
        [r.__dict__ for r in recs],
    )
    eff_cache[eff] = provided_ctx
    manifest.mark(key)


def _overhead_task(cfg, ds, name, manifest):
    key = f"overhead:{name}"
    if key in manifest.done:
        return
    backend = backends.get(name)
    ctx_size = min(8192, len(ds.train))
    try:
        row = instrument.probe_overhead(
            backend,
            ds.train[-ctx_size:],
            backend_options=cfg.backend_options.get(name, {}),
        )
    except Exception as exc:
        row = {"backend": name, "status": "error", "note": f"{type(exc).__name__}: {exc}"}
    _save_rows(cfg, key, [row])
    manifest.mark(key)


def _speed_task(cfg, ds, name, manifest):
    key = f"speed:{name}"
    if key in manifest.done:
        return
    backend = backends.get(name)
    eff = min(32768, _eff_ctx(name, 32768)[0])
    try:
        row = instrument.speed_microbench(
            backend,
            ds.train[-eff:],
            n_evals=1024 if name in SLOW_BACKENDS else 4096,
            backend_options=cfg.backend_options.get(name, {}),
        )
    except Exception as exc:
        row = {"backend": name, "status": "error", "note": f"{type(exc).__name__}: {exc}"}
    row["source"] = "microbench"
    _save_rows(cfg, key, [row])
    manifest.mark(key)


def _gen_search_task(cfg, ds, name, prompts, manifest, gen_dir):
    np_ = _n_prompts(cfg, name)
    key = f"gensearch:{name}:{np_}"
    if key in manifest.done:
        return
    backend = backends.get(name)
    prompts = prompts[:np_]
    cfg_gen = GenerationConfig(
        length=cfg.gen_length,
        beam_width=cfg.beam,
        lookahead=cfg.lookahead,
        temperature=cfg.temperature,
        context_bytes=32768,
        workers=cfg.workers,
        seed=cfg.seed,
        backend_options=cfg.backend_options.get(name, {}),
    )
    rows = genbench.run_generation(
        backend,
        ds.train,
        prompts,
        cfg_gen,
        mode="equal_search",
        out_dir=gen_dir,
        context_provided=32768,
    )
    rows.append(genbench.aggregate_generation(rows, mode="equal_search") | {"prompt": "__mean__"})
    _save_rows(cfg, key, rows)
    manifest.mark(key)


def _gen_compute_task(cfg, ds, name, budget, prompts, alphabet_size, manifest, gen_dir):
    np_ = _n_compute_prompts(cfg, name, budget)
    out_len = _compute_out_len(cfg, name)
    key = f"gencompute:{name}:{budget}:{np_}x{out_len}"
    if key in manifest.done:
        return
    backend = backends.get(name)
    beam = genbench.eval_budget_to_beam(budget, alphabet_size)
    # tune lookahead on VALIDATION only — never on test outcomes
    val_prompts = [p for p in genbench.DEFAULT_PROMPTS if p in ds.val][:1] or [b"the "]
    tune_len = 16 if name in VERY_SLOW_BACKENDS else 32 if name in SLOW_BACKENDS else 64
    la, tune = genbench.tune_lookahead(
        backend,
        ds.val,
        val_prompts,
        beam_width=beam,
        out_len=tune_len,
        workers=cfg.workers,
        seed=cfg.seed,
        backend_options=cfg.backend_options.get(name, {}),
    )
    prompts = prompts[:np_]
    cfg_gen = GenerationConfig(
        length=out_len,
        beam_width=beam,
        lookahead=la,
        temperature=cfg.temperature,
        context_bytes=32768,
        workers=cfg.workers,
        seed=cfg.seed,
        backend_options=cfg.backend_options.get(name, {}),
    )
    rows = genbench.run_generation(
        backend,
        ds.train,
        prompts,
        cfg_gen,
        mode="equal_compute",
        out_dir=gen_dir,
        context_provided=32768,
        file_tag=str(budget),
    )
    for r in rows:
        r["budget_per_byte"] = budget
        r["tuning"] = json.dumps(tune)
    agg = genbench.aggregate_generation(rows, mode="equal_compute")
    agg.update({"prompt": "__mean__", "budget_per_byte": budget, "tuning": json.dumps(tune)})
    rows.append(agg)
    _save_rows(cfg, key, rows)
    manifest.mark(key)


def run(cfg: BenchConfig) -> None:
    """Execute the benchmark suite with checkpointing."""
    ds = load_tiny_shakespeare(cfg.dataset, seed=cfg.seed)
    gen_dir = cfg.out_dir / "generations"
    manifest = _Manifest(cfg.raw_dir / "manifest.json")

    # train-only dictionary experiment; registered before selection so it
    # participates like any other backend (unless explicitly excluded)
    if (cfg.backend_names is None or "zstd_dict" in cfg.backend_names) and (
        "zstd_dict" not in cfg.skip
    ):
        register_zstd_dict(cfg, ds)

    names = _selected_backends(cfg)
    prompts = genbench.validate_prompts(genbench.DEFAULT_PROMPTS, ds.train)
    alphabet = corpus_alphabet(ds.train)

    # one canonical position list: always the 10k-sample so quick-mode
    # subsets are exact prefixes of the full run (comparable point-for-point)
    list_n = max(10000, cfg.n_positions or 0)
    all_positions = sample_positions(ds.test_span, list_n, seed=cfg.seed)
    positions = all_positions
    reporting.write_json(
        cfg.raw_dir / "positions.json",
        {"n_sampled": list_n, "positions": all_positions},
    )

    meta = {
        "dataset": ds.split_meta(),
        "mode": cfg.mode,
        "seed": cfg.seed,
        "workers": cfg.workers,
        "temperature": cfg.temperature,
        "gen_length": cfg.gen_length,
        "beam": cfg.beam,
        "lookahead": cfg.lookahead,
        "context_sizes": list(cfg.context_sizes),
        "budgets": list(cfg.budgets),
        "n_prompts": cfg.n_prompts,
        "prompts": [p.decode() for p in prompts],
        "alphabet_size": len(alphabet),
        "ctx_caps": {k: {"cap": v[0], "reason": v[1]} for k, v in CTX_CAPS.items()},
        "slow_backends": sorted(SLOW_BACKENDS),
        "environment": reporting.environment_info(),
        "package_versions": instrument.package_versions(),
        "backend_options": cfg.backend_options,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    reporting.write_json(cfg.out_dir / "config.json", meta)

    for name in names:
        if not backends.info(name).available:
            _save_rows(
                cfg,
                f"unavailable:{name}",
                [
                    {
                        "backend": name,
                        "status": "skipped",
                        "note": backends.info(name).unavailable_reason or "unavailable",
                    }
                ],
            )
            continue
        try:
            if "overhead" not in cfg.skip:
                _overhead_task(cfg, ds, name, manifest)
            if "speed" not in cfg.skip:
                _speed_task(cfg, ds, name, manifest)
            if "teacher" not in cfg.skip:
                eff_cache: dict[int, int] = {}
                for ctx in cfg.context_sizes:
                    _teacher_task(cfg, ds, name, ctx, positions, manifest, eff_cache)
            if "generation" not in cfg.skip:
                _gen_search_task(cfg, ds, name, prompts, manifest, gen_dir)
                for budget in cfg.budgets:
                    if name in SLOW_BACKENDS | VERY_SLOW_BACKENDS and budget > 2000:
                        continue  # slow tiers only get the smaller budget
                    _gen_compute_task(
                        cfg, ds, name, budget, prompts, len(alphabet), manifest, gen_dir
                    )
        except Exception as exc:
            _save_rows(
                cfg,
                f"error:{name}",
                [
                    {
                        "backend": name,
                        "status": "error",
                        "note": f"{type(exc).__name__}: {exc}",
                    }
                ],
            )

    assemble(cfg, ds)


def assemble(cfg: BenchConfig, ds: Dataset) -> dict:
    """Merge raw rows into the final CSVs + summary."""
    rows = reporting.read_jsonl_rows(cfg.raw_dir)

    tf = [r for r in rows if r.get("top1") is not None]
    reporting.write_csv(cfg.out_dir / "teacher_forced.csv", tf)
    reporting.write_csv(
        cfg.out_dir / "quantization.csv",
        [
            {
                k: r.get(k)
                for k in (
                    "backend",
                    "family",
                    "mode",
                    "context_provided",
                    "context_effective",
                    "n_positions",
                    "mean_unique_scores",
                    "mean_max_tie",
                    "tie_candidate_frac",
                    "pct_true_tied",
                    "pct_true_in_best_tie",
                    "dedup",
                    "note",
                )
            }
            for r in tf
        ],
    )
    reporting.write_csv(
        cfg.out_dir / "generation_equal_search.csv",
        [r for r in rows if r.get("mode") == "equal_search"],
    )
    reporting.write_csv(
        cfg.out_dir / "generation_equal_compute.csv",
        [r for r in rows if r.get("mode") == "equal_compute"],
    )
    copy_rows = [
        {
            k: r.get(k)
            for k in (
                "backend",
                "mode",
                "prompt",
                "generated_compression_bpb",
                "copy_span_max",
                "copy_span_mean",
                "copy_span_median",
                "copy_run",
                "overlap_4gram",
                "overlap_8gram",
                "overlap_16gram",
                "overlap_32gram",
                "overlap_64gram",
                "novel_8gram",
                "novel_16gram",
                "novel_32gram",
                "self_rep_8",
                "distinct_4",
                "utf8_valid",
                "utf8_invalid_sequences",
                "utf8_chars",
                "utf8_repl_chars",
                "status",
                "note",
            )
        }
        for r in rows
        if r.get("mode") in ("equal_search", "equal_compute") and r.get("status") == "ok"
    ]
    reporting.write_csv(cfg.out_dir / "copy_metrics.csv", copy_rows)
    speed_rows = [r for r in rows if r.get("source") == "microbench"]
    speed_rows += [
        {
            "backend": r["backend"],
            "source": "teacher_forced",
            "context_bytes": r.get("context_effective"),
            "evals_per_sec": r.get("evals_per_sec"),
            "positions_per_sec": r.get("positions_per_sec"),
            "prepare_seconds": r.get("prep_seconds"),
            "eval_seconds_mean": None,
        }
        for r in tf
    ]
    reporting.write_csv(cfg.out_dir / "speed.csv", speed_rows)
    reporting.write_csv(
        cfg.out_dir / "overhead.csv",
        [r for r in rows if r.get("scoring_method") or r.get("task") == "overhead"],
    )
    status = [r for r in rows if r.get("status") in ("skipped", "error") and "mode" not in r]
    reporting.write_csv(cfg.out_dir / "backend_status.csv", status)

    # summary.csv — one row per backend, the headline metrics merged
    tf32 = {r["backend"]: r for r in tf if str(r.get("context_provided")) == "32768"}
    gen_s = {
        r["backend"]: r
        for r in rows
        if r.get("mode") == "equal_search" and r.get("prompt") == "__mean__"
    }
    mb = {r["backend"]: r for r in speed_rows if r.get("source") == "microbench"}
    all_names = sorted(set(tf32) | set(gen_s) | set(mb) | {r["backend"] for r in status})
    summary = []
    for name in all_names:
        t, g, s = tf32.get(name, {}), gen_s.get(name, {}), mb.get(name, {})
        summary.append(
            {
                "backend": name,
                "tf_top1": t.get("top1"),
                "tf_top1_exp": t.get("top1_exp"),
                "tf_top5_exp": t.get("top5_exp"),
                "tf_mrr": t.get("mrr"),
                "tf_pseudo_bpb": t.get("teacher_forced_pseudo_bpb"),
                "tf_evals_per_sec": t.get("evals_per_sec"),
                "tf_ctx_effective": t.get("context_effective"),
                "tf_n_positions": t.get("n_positions"),
                "gen_bpb": g.get("generated_compression_bpb"),
                "gen_copy_run": g.get("copy_run"),
                "gen_overlap32": g.get("overlap_32gram"),
                "gen_self_rep8": g.get("self_rep_8"),
                "gen_utf8_valid": g.get("utf8_valid"),
                "microbench_evals_per_sec": s.get("evals_per_sec"),
            }
        )
    reporting.write_csv(cfg.out_dir / "summary.csv", summary)

    # plots + final report — a rendering failure must not lose results
    try:
        plots.generate_plots(cfg.out_dir)
    except Exception as exc:
        print(f"[warn] plots failed: {type(exc).__name__}: {exc}")
    try:
        report.build_report(cfg.out_dir)
    except Exception as exc:
        print(f"[warn] report build failed: {type(exc).__name__}: {exc}")

    return {"teacher_rows": tf, "all_rows": rows}
