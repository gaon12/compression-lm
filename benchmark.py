"""Benchmark entry point.

    python benchmark.py --dataset data/tiny_shakespeare.txt --mode quick
    python benchmark.py --dataset data/tiny_shakespeare.txt --mode full

Resumable: each completed task is checkpointed under results/raw/, so a
re-run picks up where the last one stopped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compression_lm.bench.runner import BenchConfig, run


def _csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="benchmark.py",
        description="Teacher-forced + generation benchmark of compressors as language models.",
    )
    p.add_argument("--dataset", type=Path, default=Path("data/tiny_shakespeare.txt"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick = 1,000 positions (200 slow); full = 10,000 (2,000 slow)",
    )
    p.add_argument(
        "--backends", default=None, help="comma-separated backend subset (default: all registered)"
    )
    p.add_argument(
        "--positions", type=int, default=None, help="override position count for every backend"
    )
    p.add_argument(
        "--contexts",
        type=_csv_ints,
        default=None,
        help="comma-separated context sizes for the sweep",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--lookahead", type=int, default=16)
    p.add_argument(
        "--budgets",
        type=_csv_ints,
        default=None,
        help="equal-compute eval budgets per output byte (default 2000,10000)",
    )
    p.add_argument("--prompts", type=int, default=10)
    p.add_argument(
        "--skip", default="", help="comma-separated: teacher,generation,overhead,speed,zstd_dict"
    )
    p.add_argument(
        "--assemble-only",
        action="store_true",
        help="rebuild CSVs/plots/report from existing raw rows",
    )
    p.add_argument(
        "--zip",
        action="store_true",
        help="package results/ into results.zip after the run",
    )
    args = p.parse_args(argv)

    names = tuple(s.strip() for s in args.backends.split(",")) if args.backends else None
    cfg = BenchConfig(
        dataset=args.dataset,
        out_dir=args.out,
        mode=args.mode,
        seed=args.seed,
        workers=args.workers,
        temperature=args.temperature,
        gen_length=args.gen_length,
        beam=args.beam,
        lookahead=args.lookahead,
        context_sizes=args.contexts or (4096, 32768, 262144, 1048576),
        budgets=args.budgets or (2000, 10000),
        backend_names=names,
        n_positions=args.positions,
        n_prompts=args.prompts,
        skip=frozenset(s.strip() for s in args.skip.split(",") if s.strip()),
    )
    if args.assemble_only:
        from compression_lm.bench.datasets import load_tiny_shakespeare
        from compression_lm.bench.runner import assemble

        assemble(cfg, load_tiny_shakespeare(cfg.dataset, seed=cfg.seed))
    else:
        run(cfg)
    if args.zip:
        _make_zip(cfg.out_dir)
    print(f"results written under {cfg.out_dir}")
    return 0


def _make_zip(results_dir: Path) -> Path:
    """Package every result artifact into results.zip (raw/ included —
    it is the checkpoint + per-position evidence)."""
    import zipfile

    out = results_dir.parent / "results.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(results_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(results_dir.parent))
    print(f"zip -> {out}")
    return out


if __name__ == "__main__":
    sys.exit(main())
