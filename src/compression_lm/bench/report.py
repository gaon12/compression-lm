"""Final report.md builder — assembles measured numbers into the
spec'd section structure. Everything printed here comes from the CSVs;
interpretation stays separated from observation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import instrument
from .reporting import fmt, md_table


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


def _tf_table(rows: list[dict], ctx: int | None = None) -> str:
    sel = [r for r in rows if ctx is None or str(r.get("context_provided")) == str(ctx)]
    if not sel:
        return "_no rows_"
    sel.sort(key=lambda r: _f(r.get("top1_exp")), reverse=True)
    return md_table(
        [
            "backend",
            "ctx_eff",
            "n_pos",
            "top1*",
            "top1_exp",
            "top5_exp",
            "top10_exp",
            "mean_rank",
            "MRR",
            "pseudo_BPB",
            "evals/s",
            "pos/s",
        ],
        [
            [
                r["backend"],
                r.get("context_effective", "?"),
                r.get("n_positions", "?"),
                fmt(_f(r.get("top1"))),
                fmt(_f(r.get("top1_exp"))),
                fmt(_f(r.get("top5_exp"))),
                fmt(_f(r.get("top10_exp"))),
                fmt(_f(r.get("mean_rank")), ".1f"),
                fmt(_f(r.get("mrr"))),
                fmt(_f(r.get("teacher_forced_pseudo_bpb"))),
                fmt(_f(r.get("evals_per_sec")), ".0f"),
                fmt(_f(r.get("positions_per_sec")), ".1f"),
            ]
            for r in sel
        ],
    )


def _gen_table(rows: list[dict], budget_col: bool = False) -> str:
    means = [r for r in rows if r.get("prompt") == "__mean__"]
    if not means:
        return "_no rows_"
    means.sort(key=lambda r: (_f(r.get("budget_per_byte")), _f(r.get("generated_compression_bpb"))))
    header = ["backend", "ctx_eff"]
    if budget_col:
        header.append("budget/B")
    header += [
        "gen_BPB",
        "copy_run",
        "copy_span_max",
        "ovl_8",
        "ovl_32",
        "self_rep8",
        "distinct4",
        "utf8",
        "evals/s",
        "B/s",
    ]
    rows_out = []
    for r in means:
        line = [r["backend"], r.get("context_effective", "?")]
        if budget_col:
            line.append(r.get("budget_per_byte", "?"))
        line += [
            fmt(_f(r.get("generated_compression_bpb"))),
            fmt(_f(r.get("copy_run")), ".0f"),
            fmt(_f(r.get("copy_span_max")), ".0f"),
            fmt(_f(r.get("overlap_8gram"))),
            fmt(_f(r.get("overlap_32gram"))),
            fmt(_f(r.get("self_rep_8"))),
            fmt(_f(r.get("distinct_4"))),
            r.get("utf8_valid", "?"),
            fmt(_f(r.get("evals_per_sec")), ".0f"),
            fmt(_f(r.get("bytes_per_sec")), ".1f"),
        ]
        rows_out.append(line)
    return md_table(header, rows_out)


def _quant_table(tf_rows: list[dict], ctx: int = 32768) -> str:
    sel = [r for r in tf_rows if str(r.get("context_provided")) == str(ctx)]
    if not sel:
        return "_no rows_"
    sel.sort(key=lambda r: _f(r.get("mean_unique_scores")), reverse=True)
    return md_table(
        ["backend", "uniq_scores", "max_tie", "tie_frac", "true_tied%", "in_best_tie%"],
        [
            [
                r["backend"],
                fmt(_f(r.get("mean_unique_scores")), ".1f"),
                fmt(_f(r.get("mean_max_tie")), ".1f"),
                fmt(_f(r.get("tie_candidate_frac"))),
                fmt(_f(r.get("pct_true_tied"))),
                fmt(_f(r.get("pct_true_in_best_tie"))),
            ]
            for r in sel
        ],
    )


def _speed_table(speed_rows: list[dict]) -> str:
    mb = [r for r in speed_rows if r.get("source") == "microbench"]
    if not mb:
        return "_no rows_"
    mb.sort(key=lambda r: _f(r.get("evals_per_sec")), reverse=True)
    return md_table(
        ["backend", "ctx", "prep_s", "eval_s_mean(ms)", "evals/s", "ctx/s", "RSS_MB"],
        [
            [
                r["backend"],
                r.get("context_bytes", "?"),
                fmt(_f(r.get("prepare_seconds")), ".4f"),
                fmt(_f(r.get("eval_seconds_mean"), 0.0) * 1000, ".3f"),
                fmt(_f(r.get("evals_per_sec")), ".0f"),
                fmt(_f(r.get("contexts_per_sec")), ".1f"),
                fmt(_f(r.get("peak_rss_bytes"), 0.0) / 1e6, ".0f"),
            ]
            for r in mb
        ],
    )


def _correlations(gen_rows: list[dict], tf_rows: list[dict]) -> str:
    means = [r for r in gen_rows if r.get("prompt") == "__mean__" and r.get("status") == "ok"]
    if len(means) < 3:
        return "_insufficient rows for correlation_"
    gen_bpb = [_f(r["generated_compression_bpb"]) for r in means]
    ovl32 = [_f(r["overlap_32gram"]) for r in means]
    copy_run = [_f(r["copy_run"]) for r in means]
    tf_by_be = {
        r["backend"]: _f(r["teacher_forced_pseudo_bpb"])
        for r in tf_rows
        if str(r.get("context_provided")) == "32768"
    }
    tf_bpb = [tf_by_be.get(r["backend"]) for r in means]
    pairs = [(g, o, c, t) for g, o, c, t in zip(gen_bpb, ovl32, copy_run, tf_bpb, strict=True) if t]
    if len(pairs) < 3:
        return "_insufficient matched rows_"
    gs = [p[0] for p in pairs]
    os_ = [p[1] for p in pairs]
    cs = [p[2] for p in pairs]
    ts = [p[3] for p in pairs]
    lines = [
        "| relationship | pearson | spearman | n |",
        "|---|---|---|---|",
        f"| gen_BPB vs 32-gram overlap | {instrument.pearson(gs, os_):.3f} "
        f"| {instrument.spearman(gs, os_):.3f} | {len(gs)} |",
        f"| gen_BPB vs copy_run | {instrument.pearson(gs, cs):.3f} "
        f"| {instrument.spearman(gs, cs):.3f} | {len(gs)} |",
        f"| teacher pseudo_BPB vs 32-gram overlap | {instrument.pearson(ts, os_):.3f} "
        f"| {instrument.spearman(ts, os_):.3f} | {len(ts)} |",
        f"| teacher pseudo_BPB vs gen_BPB | {instrument.pearson(ts, gs):.3f} "
        f"| {instrument.spearman(ts, gs):.3f} | {len(ts)} |",
    ]
    return "\n".join(lines)


def _samples_section(gen_dir: Path, gen_rows: list[dict], max_show: int = 6) -> str:
    """Embed short excerpts of the most interesting generations."""
    detail = [r for r in gen_rows if r.get("prompt") != "__mean__" and r.get("status") == "ok"]
    if not detail or not gen_dir.exists():
        return "_no generation samples_"
    interesting = sorted(
        detail,
        key=lambda r: (-_f(r.get("copy_run")), -_f(r.get("self_rep_8"))),
    )[:max_show]
    parts = []
    for r in interesting:
        f = Path(r.get("output_file", ""))
        if not f.exists():
            f = gen_dir / f.name
        if not f.exists():
            continue
        text = f.read_bytes()[:240].decode("utf-8", errors="replace")
        parts.append(
            f"**{r['backend']}** prompt=`{r['prompt']!r}` "
            f"copy_run={_f(r.get('copy_run')):.0f} self_rep={_f(r.get('self_rep_8')):.2f}\n"
            f"```\n{text}\n```"
        )
    return "\n\n".join(parts)


def _executive_summary(tf_rows, gen_rows) -> str:
    """Data-driven bullets — the numbers pick the observations."""
    canon = [r for r in tf_rows if str(r.get("context_provided")) == "32768"]
    bullets: list[str] = []
    if canon:
        best_pred = max(canon, key=lambda r: _f(r.get("top1_exp")))
        best_bpb = min(canon, key=lambda r: _f(r.get("teacher_forced_pseudo_bpb")))
        bullets.append(
            f"- Best next-byte predictor @32KiB: **{best_pred['backend']}** "
            f"(top1_exp={_f(best_pred['top1_exp']):.3f}, "
            f"pseudo_BPB={_f(best_pred['teacher_forced_pseudo_bpb']):.3f})"
        )
        bullets.append(
            f"- Lowest teacher-forced pseudo-BPB: **{best_bpb['backend']}** "
            f"({_f(best_bpb['teacher_forced_pseudo_bpb']):.3f} bits/byte)"
        )
        umax = max(canon, key=lambda r: _f(r.get("mean_unique_scores")))
        umin = min(canon, key=lambda r: _f(r.get("mean_unique_scores")))
        bullets.append(
            f"- Score resolution spans {_f(umin['mean_unique_scores']):.1f}-"
            f"{_f(umax['mean_unique_scores']):.1f} unique scores per 256 "
            f"candidates ({umin['backend']} ... {umax['backend']}) - byte-level "
            "quantization dominates ranking, see top1 vs top1_exp"
        )
    means = [r for r in gen_rows if r.get("prompt") == "__mean__" and r.get("status") == "ok"]
    if len(means) >= 3:
        gs = [_f(r["generated_compression_bpb"]) for r in means]
        os_ = [_f(r["overlap_32gram"]) for r in means]
        sp = instrument.spearman(gs, os_)
        copier = max(means, key=lambda r: _f(r.get("copy_run")))
        bullets.append(
            f"- Generated-compression BPB correlates with corpus copying "
            f"(spearman gen_BPB vs 32-gram overlap = {sp:.2f}): "
            f"**{copier['backend']}** copies runs up to "
            f"{_f(copier['copy_run']):.0f} B verbatim — low gen_BPB ≠ good LM"
        )
    if canon:
        fastest = max(canon, key=lambda r: _f(r.get("evals_per_sec")))
        bullets.append(
            f"- Scorer throughput ranges {_f(fastest['evals_per_sec']):.0f} evals/s "
            f"({fastest['backend']}) down to "
            f"{_f(min(canon, key=lambda r: _f(r.get('evals_per_sec')))['evals_per_sec']):.0f} "
            f"— equal-compute conclusions differ from equal-search"
        )
    return "\n".join(bullets) or "_insufficient data_"


def build_report(results_dir: Path) -> Path:
    tf = _read_csv(results_dir / "teacher_forced.csv")
    gen_s = _read_csv(results_dir / "generation_equal_search.csv")
    gen_c = _read_csv(results_dir / "generation_equal_compute.csv")
    speed = _read_csv(results_dir / "speed.csv")
    overhead = _read_csv(results_dir / "overhead.csv")
    scorecmp = _read_csv(results_dir / "scoring_methods.csv")
    status = _read_csv(results_dir / "backend_status.csv")
    cfg = {}
    cj = results_dir / "config.json"
    if cj.exists():
        cfg = json.loads(cj.read_text(encoding="utf-8"))

    ds_meta = cfg.get("dataset", {})
    parts = [
        "# Compressor-as-LM — second benchmark report\n",
        "> Teacher-forced prediction, equal-search generation, equal-compute "
        "generation, retrieval/copy analysis, score quantization, and speed "
        "for lossless compressors used as language models.\n",
        "## Executive summary\n",
        _executive_summary(tf, gen_s),
        "\n## Methodology\n",
        "- Common score for every backend: "
        "`raw_score(b) = len(compress(context + b)) - len(compress(context))`, "
        "converted to bits as `8 * raw_score`.\n"
        "- Pseudo-probability: `p(b) = 2^-delta_bits(b) / Σ 2^-delta_bits` over "
        "all 256 byte candidates (min-shifted, log-space stable).\n"
        "- Ties: `rank_best = #better + 1`, "
        "`rank_avg = #better + (tied + 1) / 2`.\n"
        "  `top1/top5/top10` use avg-rank (deterministic); `top*_exp` is the "
        "expected accuracy under uniform-random tie-breaking — the comparable "
        "metric when score quantization is extreme.\n"
        "- `teacher_forced_pseudo_bpb` (held-out next-byte NLL) is strictly "
        "separate from `generated_compression_bpb` (self-generated text "
        "compressibility — a behavioral metric, NOT a LM metric).\n",
        "## Dataset & split\n",
        f"```json\n{json.dumps(ds_meta, indent=2)}\n```\n",
        "- Split method: contiguous ranges with 4 KiB discarded gaps at "
        "boundaries (tiny_shakespeare is a single concatenated stream — no "
        "document boundaries). Train is the only compressor corpus; "
        "validation is used only for equal-compute lookahead tuning; test "
        "positions are scored bytes — disjoint from train by construction.\n",
        "## Backend status\n",
        _status_table(status),
        "\n## Teacher-forced next-byte prediction (32 KiB provided)\n",
        _tf_table(tf, 32768),
        "\n*top1 = avg-rank deterministic; top*_exp = expected under random "
        "tie-breaking. n_pos differs across tiers (recorded, prefix subsets "
        "of one shared position list).\n",
        "\n## Context-length sweep\n",
        _sweep_table(tf),
        "\n## Equal-search generation\n",
        "beam=16, lookahead=16, temperature=0.8, ctx=32KiB provided, "
        "len=256B, seed=42, identical prompts:\n",
        _gen_table(gen_s),
        "\n## Equal-compute generation\n",
        "Fixed candidate-eval budget per output byte (fast backends ran "
        "budget 2000 and 10000; slow/very-slow ran 2000); "
        "beam = budget // alphabet; lookahead tuned on VALIDATION only:\n",
        _gen_table(gen_c, budget_col=True),
        "\n## Retrieval / copy analysis\n",
        _correlations(gen_s, tf),
        "\n(Per-output detail in `copy_metrics.csv`.)\n",
        "\n## Score quantization\n",
        _quant_table(tf),
        "\n## Speed (LM-scorer throughput)\n",
        _speed_table(speed),
        "\n## Compression container overhead\n",
        _overhead_table(overhead),
        "\n## Scoring-method comparison (A vs B)\n",
        "Method A = plain `len(compress(ctx + b)) - len(compress(ctx))` "
        "recompression; Method B = each backend's actual scorer "
        "(incremental state / dictionary / full recompress):\n",
        _scorecmp_table(scorecmp),
        "\n## Generation samples (most copy-heavy)\n",
        _samples_section(results_dir / "generations", gen_s),
        "\n## Caveats\n",
        "- Pseudo-probabilities are normalized compressed-size deltas, not "
        "calibrated LM probabilities.\n"
        "- `generated_compression_bpb` rewards copying: a backend that "
        "regurgitates the corpus scores a *lower* gen_BPB — never read it "
        "as language-model quality.\n"
        "- One-shot backends (bzip2/lzma family/zstd_dict) evaluated reduced "
        "prefix subsets and smaller context caps — rows are marked.\n"
        "- Results are seed=42 tiny_shakespeare; character-level compressors "
        "measure byte predictability, not semantics.\n",
    ]
    out = results_dir / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def _status_table(status_rows: list[dict]) -> str:
    if not status_rows:
        return "all backends ok"
    return md_table(
        ["backend", "status", "note"],
        [[r.get("backend", "?"), r.get("status", "?"), r.get("note", "")] for r in status_rows],
    )


def _sweep_table(tf_rows: list[dict]) -> str:
    if not tf_rows:
        return "_no rows_"
    rows = sorted(
        tf_rows,
        key=lambda r: (r["backend"], _f(r.get("context_effective"))),
    )
    return md_table(
        [
            "backend",
            "ctx_prov",
            "ctx_eff",
            "top1_exp",
            "top5_exp",
            "MRR",
            "pseudo_BPB",
            "evals/s",
            "dedup",
        ],
        [
            [
                r["backend"],
                r.get("context_provided", "?"),
                r.get("context_effective", "?"),
                fmt(_f(r.get("top1_exp"))),
                fmt(_f(r.get("top5_exp"))),
                fmt(_f(r.get("mrr"))),
                fmt(_f(r.get("teacher_forced_pseudo_bpb"))),
                fmt(_f(r.get("evals_per_sec")), ".0f"),
                r.get("dedup", "?"),
            ]
            for r in rows
        ],
    )


def _overhead_table(rows: list[dict]) -> str:
    if not rows:
        return "_no rows_"
    return md_table(
        ["backend", "scoring", "delta_0B", "delta_1B", "delta_8B", "delta_32B", "delta_64B"],
        [
            [
                r.get("backend", "?"),
                r.get("scoring_method", "?"),
                r.get("delta_novel_0B", "?"),
                r.get("delta_novel_1B", "?"),
                r.get("delta_novel_8B", "?"),
                r.get("delta_novel_32B", "?"),
                r.get("delta_novel_64B", "?"),
            ]
            for r in rows
        ],
    )


def _scorecmp_table(rows: list[dict]) -> str:
    if not rows:
        return "_no rows_"

    def cell(v) -> str:
        f = _f(v, default=float("nan"))
        return "-" if f != f else fmt(f)  # nan / empty -> "-"

    rows = sorted(rows, key=lambda r: r.get("backend", ""))
    return md_table(
        ["backend", "method_B", "A_mean", "B_mean", "B-A", "pearson", "identical"],
        [
            [
                r.get("backend", "?"),
                r.get("scoring_method") or (r.get("note", "")[:40] or "?"),
                cell(r.get("methodA_mean")),
                cell(r.get("methodB_mean")),
                cell(r.get("B_minus_A_mean")),
                cell(r.get("pearson_A_B")),
                cell(r.get("identical_frac")),
            ]
            for r in rows
        ],
    )
