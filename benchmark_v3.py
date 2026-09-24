#!/usr/bin/env python3
"""Third benchmark entry point — v3 verification experiment.

Usage:
    python benchmark_v3.py --dataset tiny_shakespeare --resume
    python benchmark_v3.py --dataset wikitext2 --resume
    python benchmark_v3.py --dataset tiny_shakespeare --phases scoreval,teacher
    python benchmark_v3.py --assemble-only --out results_v3

Everything lands in results_v3/ (or --out); v2 outputs are untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from compression_lm.bench_v3 import runner3


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default="tiny_shakespeare",
        help="tiny_shakespeare | wikitext2 | all | path to a raw text file",
    )
    p.add_argument(
        "--backends",
        default="",
        help="comma list to restrict backends (default: all available)",
    )
    p.add_argument("--out", default="results_v3", help="output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gap", type=int, default=4096)
    p.add_argument("--n-positions", type=int, default=10000)
    p.add_argument("--teacher-ctxs", default="4096,32768")
    p.add_argument("--disc-lengths", default="1,4,8,16,32")
    p.add_argument("--eval-budgets", default="2000,10000")
    p.add_argument("--time-budgets", default="10.0,30.0")
    p.add_argument("--n-prompts", type=int, default=4)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--phases",
        default="all",
        help="comma list of: zstdval,scorerval,teacher,disc,rawpair,evalgen,eetime,snappy,speed",
    )
    p.add_argument("--resume", action="store_true", help="continue from checkpoint")
    p.add_argument("--assemble-only", action="store_true")
    p.add_argument("--zip", action="store_true", help="write <out>.zip at the end")
    a = p.parse_args()

    cfg = runner3.Bench3Config(
        dataset=a.dataset,
        out_dir=Path(a.out),
        seed=a.seed,
        gap=a.gap,
        n_positions=a.n_positions,
        teacher_ctxs=tuple(int(x) for x in a.teacher_ctxs.split(",") if x),
        disc_lengths=tuple(int(x) for x in a.disc_lengths.split(",") if x),
        eval_budgets=tuple(int(x) for x in a.eval_budgets.split(",") if x),
        time_budgets=tuple(float(x) for x in a.time_budgets.split(",") if x),
        n_prompts=a.n_prompts,
        gen_length=a.gen_length,
        workers=a.workers,
        phases=frozenset(
            runner3.ALL_PHASES
            if a.phases == "all"
            else [s.strip() for s in a.phases.split(",") if s.strip()]
        ),
        backend_names=tuple(s.strip() for s in a.backends.split(",") if s.strip()),
        resume=a.resume,
        assemble_only=a.assemble_only,
        zip_after=a.zip,
    )
    runner3.run(cfg)


if __name__ == "__main__":
    main()
