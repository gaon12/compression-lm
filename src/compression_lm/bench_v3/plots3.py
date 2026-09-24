"""v3 plots — the 12 figures from spec §27, one PNG each.

All values come from the assembled CSVs; nothing here recomputes scores.
Figures that need multi-dataset rows draw one series per dataset.
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


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _bar(ax, names, vals, ylabel, title, log=False):
    ax.bar(range(len(names)), vals)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")


def _lines(ax, series: dict[str, list[tuple[float, float]]], xlabel, ylabel, title, logx=False):
    for name, pts in series.items():
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=name)
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)


def _at_ctx(rows: list[dict], ctx: int) -> list[dict]:
    return [r for r in rows if str(r.get("context_provided") or r.get("context_bytes")) == str(ctx)]


def make_plots(results_dir: Path) -> list[Path]:
    out = results_dir / "plots"
    made: list[Path] = []
    tf = _read_csv(results_dir / "teacher_forced_1byte.csv")
    disc = _read_csv(results_dir / "continuation_discrimination.csv")
    eetime = _read_csv(results_dir / "generation_equal_time.csv")
    evalgen = _read_csv(results_dir / "generation_equal_evaluation.csv")
    rawpairs = _read_csv(results_dir / "container_vs_raw.csv")
    tf_main = [r for r in tf if r.get("mode") != "zstd_mode_validation" and r.get("status") == "ok"]

    c4 = _at_ctx(tf_main, 4096)
    names = [r["backend"] for r in sorted(c4, key=lambda r: _f(r.get("top1_exp")), reverse=True)]
    by_name = {r["backend"]: r for r in c4}

    def col(key, default=0.0):
        return [_f(by_name.get(n, {}).get(key), default) for n in names]

    if c4:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _bar(ax, names, col("top1_exp"), "top-1 expected", "1-byte top-1 accuracy @4KiB")
        p = out / "01_top1_1byte.png"
        _save(fig, p)
        made.append(p)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        _bar(ax, names, col("teacher_forced_pseudo_bpb"), "bits/byte", "1-byte pseudo BPB @4KiB")
        p = out / "02_pseudo_bpb_1byte.png"
        _save(fig, p)
        made.append(p)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(col("evals_per_sec"), col("top1_exp"))
        for n in names:
            ax.annotate(
                n,
                (_f(by_name[n].get("evals_per_sec")), _f(by_name[n].get("top1_exp"))),
                fontsize=7,
            )
        ax.set_xscale("log")
        ax.set_xlabel("evals/s (log)")
        ax.set_ylabel("top-1 expected")
        ax.set_title("Prediction vs throughput @4KiB")
        p = out / "03_top1_vs_evals.png"
        _save(fig, p)
        made.append(p)

    # 4/5: continuation-length curves (random negatives, per dataset+backend)
    disc_r = [
        r
        for r in disc
        if r.get("negative_type") == "random" and r.get("status", "ok") == "ok"
    ]
    for key, fname, ylabel, title in (
        ("top1", "04_contlen_vs_top1.png", "top-1", "Continuation length vs top-1"),
        (
            "unique_scores_mean",
            "05_contlen_vs_unique.png",
            "unique scores",
            "Continuation length vs score resolution",
        ),
    ):
        series: dict[str, list[tuple[float, float]]] = {}
        for r in disc_r:
            series.setdefault(r["backend"], []).append(
                (_f(r.get("continuation_bytes")), _f(r.get(key)))
            )
        if series:
            fig, ax = plt.subplots(figsize=(8, 5))
            _lines(ax, series, "continuation bytes", ylabel, title, logx=True)
            p = out / fname
            _save(fig, p)
            made.append(p)

    gen_rows = [
        r
        for r in evalgen
        if r.get("status") == "ok" and r.get("prompt") != "__mean__"
    ] + [r for r in eetime if r.get("status") == "ok"]
    if gen_rows:
        agg: dict[str, dict[str, list[float]]] = {}
        for r in gen_rows:
            b = r.get("backend", "?")
            agg.setdefault(b, {}).setdefault("bpb", []).append(
                _f(r.get("generated_compression_bpb"))
            )
            agg.setdefault(b, {}).setdefault("crr", []).append(_f(r.get("copy_run_ratio")))
        xs = [statistics_mean(v["bpb"]) for v in agg.values()]
        ys = [statistics_mean(v["crr"]) for v in agg.values()]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(xs, ys)
        for b, x, y in zip(agg, xs, ys, strict=True):
            ax.annotate(b, (x, y), fontsize=7)
        ax.set_xlabel("generated compression BPB")
        ax.set_ylabel("copy-run ratio")
        ax.set_title("gen-BPB vs copy-run ratio")
        p = out / "06_genbpb_vs_copyratio.png"
        _save(fig, p)
        made.append(p)

    if c4:
        fig, ax = plt.subplots(figsize=(8, 5))
        crr_map = {}
        for r in gen_rows:
            crr_map.setdefault(r.get("backend", "?"), []).append(_f(r.get("copy_run_ratio")))
        xs = [statistics_mean(crr_map.get(n, [0.0])) for n in names]
        ys = col("top1_exp")
        ax.scatter(xs, ys)
        for n, x, y in zip(names, xs, ys, strict=True):
            ax.annotate(n, (x, y), fontsize=7)
        ax.set_xlabel("copy-run ratio (generated)")
        ax.set_ylabel("teacher top-1 @4KiB")
        ax.set_title("Prediction vs retrieval tendency")
        p = out / "07_top1_vs_copyratio.png"
        _save(fig, p)
        made.append(p)

    # 8/9: equal-time bytes + copy ratio per backend
    et = [r for r in eetime if r.get("status") == "ok"]
    if et:
        for i, (key, ylabel, title) in enumerate(
            (
                ("generated_bytes", "bytes generated", "Equal-time output size"),
                ("copy_run_ratio", "copy-run ratio", "Equal-time copy tendency"),
            ),
            start=8,
        ):
            per: dict[str, list[float]] = {}
            for r in et:
                per.setdefault(r["backend"], []).append(_f(r.get(key)))
            names_et = sorted(per, key=lambda b: statistics_mean(per[b]), reverse=True)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            _bar(ax, names_et, [statistics_mean(per[b]) for b in names_et], ylabel, title)
            tag = "bytes" if key == "generated_bytes" else "copyratio"
            p = out / f"{i:02d}_{tag}_equaltime.png"
            _save(fig, p)
            made.append(p)

    # 10: snappy budget sweep
    sw = [
        r
        for r in evalgen
        if r.get("mode") == "snappy_budget_sweep" and r.get("prompt") == "__mean__"
    ]
    if sw:
        pts = sorted((_f(r.get("budget_per_byte")), _f(r.get("copy_run"))) for r in sw)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("search budget (evals/output-byte)")
        ax.set_ylabel("copy-run (bytes)")
        ax.set_title("snappy: search budget vs corpus copying")
        p = out / "10_snappy_budget_vs_copyrun.png"
        _save(fig, p)
        made.append(p)

    # 11: raw lzma/lzma2 vs xz
    pair_rows = [r for r in rawpairs if r.get("status") == "ok"]
    if pair_rows:
        labels = [f"{r['backend_a']} vs {r['backend_b']}" for r in pair_rows]
        a_vals = [_f(r.get("top1_exp_a")) for r in pair_rows]
        b_vals = [_f(r.get("top1_exp_b")) for r in pair_rows]
        x = range(len(labels))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([i - 0.2 for i in x], a_vals, width=0.4, label="raw/alone")
        ax.bar([i + 0.2 for i in x], b_vals, width=0.4, label="container")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("top-1 expected")
        ax.set_title("raw vs container framing @4KiB")
        ax.legend()
        p = out / "11_raw_vs_container.png"
        _save(fig, p)
        made.append(p)

    # 12: zstd scoring modes
    zrows = [r for r in tf if r.get("backend", "").startswith("zstd") and r.get("status") == "ok"]
    if zrows:
        fig, ax = plt.subplots(figsize=(8, 5))
        series = {}
        for r in zrows:
            series.setdefault(r["backend"], []).append(
                (_f(r.get("context_provided")), _f(r.get("top1_exp")))
            )
        _lines(ax, series, "context bytes", "top-1 expected", "zstd scoring modes", logx=True)
        p = out / "12_zstd_modes.png"
        _save(fig, p)
        made.append(p)

    return made


def statistics_mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0
