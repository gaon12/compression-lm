"""Matplotlib report figures — spec section 19.

Reads the assembled CSVs under results/ and writes one PNG per figure to
results/plots/. Missing data (e.g. a backend that failed) is skipped, so
plots still render from partial runs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rows_at(rows: list[dict], ctx: int, mode: str | None = None) -> dict[str, dict]:
    """Latest row per backend at a given provided context size."""
    out: dict[str, dict] = {}
    for r in rows:
        if str(r.get("context_provided")) != str(ctx):
            continue
        if mode is not None and r.get("mode") != mode:
            continue
        out[r["backend"]] = r
    return out


def _bar(ax, names, vals, ylabel, title):
    ax.bar(range(len(names)), vals)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def generate_plots(results_dir: Path) -> list[Path]:
    out = results_dir / "plots"
    made: list[Path] = []
    tf = _read_csv(results_dir / "teacher_forced.csv")
    gen_s = [
        r
        for r in _read_csv(results_dir / "generation_equal_search.csv")
        if r.get("prompt") == "__mean__"
    ]
    gen_c = [
        r
        for r in _read_csv(results_dir / "generation_equal_compute.csv")
        if r.get("prompt") == "__mean__"
    ]
    canonical = _rows_at(tf, 32768)
    if not canonical and tf:
        canonical = _rows_at(tf, int(tf[0].get("context_provided") or 0))
    names = sorted(canonical)

    def col(key, rows=None):
        rows = rows or canonical
        return [_f(rows[n].get(key)) or 0.0 for n in names]

    if canonical:
        # 1. Top-1 (deterministic avg-rank and expected under random ties)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        w, x = 0.4, range(len(names))
        ax.bar([i - w / 2 for i in x], col("top1"), w, label="top1 (avg-rank)")
        ax.bar([i + w / 2 for i in x], col("top1_exp"), w, label="top1 expected")
        ax.set_xticks(list(x), names, rotation=45, ha="right")
        ax.set_ylabel("accuracy")
        ax.set_title("Top-1 next-byte accuracy @32KiB provided ctx")
        ax.legend()
        p = out / "01_top1_by_backend.png"
        _save(fig, p)
        made.append(p)

        # 2. pseudo BPB
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _bar(
            ax,
            names,
            col("teacher_forced_pseudo_bpb"),
            "bits/byte",
            "Teacher-forced pseudo BPB @32KiB",
        )
        p = out / "02_pseudo_bpb_by_backend.png"
        _save(fig, p)
        made.append(p)

        # 3. MRR
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _bar(ax, names, col("mrr"), "MRR", "Mean reciprocal rank @32KiB")
        p = out / "03_mrr_by_backend.png"
        _save(fig, p)
        made.append(p)

        # 4. eval/s (log)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _bar(
            ax,
            names,
            col("evals_per_sec"),
            "evals/s (log)",
            "Candidate evaluations per second @32KiB",
        )
        ax.set_yscale("log")
        p = out / "04_evals_per_sec.png"
        _save(fig, p)
        made.append(p)

        # 5. pseudo BPB vs eval/s scatter
        fig, ax = plt.subplots(figsize=(7, 5))
        xs, ys = col("evals_per_sec"), col("teacher_forced_pseudo_bpb")
        ax.scatter(xs, ys)
        for n, x, y in zip(names, xs, ys, strict=True):
            ax.annotate(n, (x, y), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("evals/s (log)")
        ax.set_ylabel("pseudo BPB")
        ax.set_title("Prediction quality vs scorer speed")
        p = out / "05_bpb_vs_evals.png"
        _save(fig, p)
        made.append(p)

    # 6/7. generated/copy overlap relationships (equal-search means)
    for pname, xlabel, title, xs_fn in (
        (
            "06_genbpb_vs_overlap32",
            "generated compression BPB",
            "Generated compression BPB vs corpus copy overlap",
            lambda r: _f(r.get("generated_compression_bpb")),
        ),
        (
            "07_pseudobpb_vs_overlap32",
            "teacher-forced pseudo BPB",
            "Teacher pseudo BPB vs copy overlap",
            lambda r: _f(canonical.get(r["backend"], {}).get("teacher_forced_pseudo_bpb")),
        ),
    ):
        if not gen_s:
            break
        xs = [xs_fn(r) or 0.0 for r in gen_s]
        ys = [_f(r.get("overlap_32gram")) or 0.0 for r in gen_s]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(xs, ys)
        for r, x, y in zip(gen_s, xs, ys, strict=True):
            ax.annotate(r["backend"], (x, y), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("32-byte n-gram overlap")
        ax.set_title(title)
        p = out / f"{pname}.png"
        _save(fig, p)
        made.append(p)

    # 8/9. context sweep lines
    if tf:
        per_be: dict[str, list[dict]] = {}
        for r in tf:
            per_be.setdefault(r["backend"], []).append(r)
        for key, ylabel, pname in (
            ("top1_exp", "Top-1 expected", "08_ctx_vs_top1"),
            ("teacher_forced_pseudo_bpb", "pseudo BPB", "09_ctx_vs_bpb"),
        ):
            fig, ax = plt.subplots(figsize=(8, 5))
            for be, rows in per_be.items():
                pts = sorted(
                    ((int(r["context_effective"]), _f(r.get(key)) or 0.0) for r in rows),
                    key=lambda t: t[0],
                )
                seen: dict[int, float] = {}
                for cx, v in pts:
                    seen[cx] = v
                ax.plot(sorted(seen), [seen[c] for c in sorted(seen)], marker="o", label=be)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("effective context bytes")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs context size")
            ax.legend(fontsize=7)
            p = out / f"{pname}.png"
            _save(fig, p)
            made.append(p)

    # 10. unique candidate scores
    if canonical:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _bar(ax, names, col("mean_unique_scores"), "unique scores / 256", "Score resolution @32KiB")
        p = out / "10_unique_scores.png"
        _save(fig, p)
        made.append(p)

    # extra: equal-compute vs equal-search BPB comparison if present
    if gen_c:
        fig, ax = plt.subplots(figsize=(7, 5))
        for rows, label in ((gen_s, "equal_search"), (gen_c, "equal_compute")):
            xs = [_f(r.get("evals_per_sec")) or 0.0 for r in rows]
            ys = [_f(r.get("generated_compression_bpb")) or 0.0 for r in rows]
            ax.scatter(xs, ys, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("evals/s (log)")
        ax.set_ylabel("generated compression BPB")
        ax.set_title("Generation quality vs scoring speed")
        ax.legend()
        p = out / "11_genbpb_vs_speed.png"
        _save(fig, p)
        made.append(p)

    return made
