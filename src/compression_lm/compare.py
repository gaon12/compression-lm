"""Side-by-side comparison of compressor backends on one prompt.

Runs the same generation task across backends and records output quality
metrics plus throughput, then renders a Markdown report. Search budgets
scale with each backend's speed class so slow algorithms (lzma family)
stay usable — pass explicit overrides via :func:`run_comparison` args to
level the field.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import backends
from .engine import GenerationConfig, generate
from .metrics import corpus_overlap, distinct_ratio, self_repetition

#: (beam_width, lookahead) defaults per backend speed class.
SPEED_BUDGET: dict[str, tuple[int, int]] = {
    "fast": (16, 16),
    "medium": (8, 8),
    "slow": (4, 6),
}


@dataclass
class ComparisonRow:
    """One backend's result line in a comparison run."""

    backend: str
    family: str
    status: str  # "ok" | "skipped" | "error"
    text: str = ""
    elapsed: float = 0.0
    bytes_per_second: float = 0.0
    bits_per_byte: float = 0.0
    corpus_overlap: float = 0.0
    self_repetition: float = 0.0
    distinct_4gram: float = 0.0
    evaluations: int = 0
    beam_width: int = 0
    lookahead: int = 0
    context_bytes: int = 0
    note: str = ""
    extra: dict = field(default_factory=dict)


def run_comparison(
    corpus: bytes,
    prompt: bytes,
    *,
    length: int = 120,
    names: list[str] | None = None,
    temperature: float = 0.0,
    workers: int = 1,
    seed: int | None = 0,
    beam_width: int | None = None,
    lookahead: int | None = None,
    context_bytes: int | None = None,
    memory: int = 128,
    tiebreak: str = "corpus",
    corpus_gate: bool = False,
    repeat_penalty: float = 0.0,
    backend_options: dict | None = None,
) -> list[ComparisonRow]:
    """Generate with every requested backend and collect metrics.

    ``names=None`` selects every registered backend; unavailable ones are
    reported as skipped rather than aborting the run.
    """
    selected = names if names is not None else backends.names()
    rows: list[ComparisonRow] = []
    for name in selected:
        info = backends.info(name)
        row = ComparisonRow(backend=name, family=info.family, status="ok")
        if not info.available:
            row.status = "skipped"
            row.note = info.unavailable_reason or "unavailable"
            rows.append(row)
            continue
        bw, la = SPEED_BUDGET[info.speed]
        cfg = GenerationConfig(
            length=length,
            beam_width=beam_width or bw,
            lookahead=lookahead or la,
            temperature=temperature,
            context_bytes=context_bytes or 0,
            memory=memory,
            tiebreak=tiebreak,
            corpus_gate=corpus_gate,
            repeat_penalty=repeat_penalty,
            workers=workers,
            seed=seed,
            backend_options=dict(backend_options or {}),
        )
        try:
            res = generate(backends.get(name), corpus, prompt, cfg)
        except Exception as exc:  # report per-row; don't abort the whole run
            row.status = "error"
            row.note = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        row.text = res.text.decode("utf-8", errors="replace")
        row.elapsed = res.elapsed
        row.bytes_per_second = res.bytes_per_second
        row.bits_per_byte = res.bits_per_byte
        row.corpus_overlap = corpus_overlap(res.text, corpus)
        row.self_repetition = self_repetition(res.text)
        row.distinct_4gram = distinct_ratio(res.text)
        row.evaluations = res.evaluations
        row.beam_width = cfg.beam_width
        row.lookahead = cfg.lookahead
        row.context_bytes = min(cfg.context_bytes or info.context_limit, info.context_limit)
        rows.append(row)
    return rows


def render_markdown(rows: list[ComparisonRow], *, prompt: str = "") -> str:
    """Render a comparison as a Markdown report."""
    lines = [
        "# compression-lm comparison",
        "",
        f"prompt: `{prompt}`" if prompt else "prompt: *(none)*",
        f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| backend | family | status | time (s) | B/s | bits/byte | copy% | "
        "self-rep% | beam x look | ctx |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.status == "ok":
            lines.append(
                f"| {r.backend} | {r.family} | ok | {r.elapsed:.2f} | "
                f"{r.bytes_per_second:.1f} | {r.bits_per_byte:.2f} | "
                f"{r.corpus_overlap * 100:.0f}% | {r.self_repetition * 100:.0f}% | "
                f"{r.beam_width}x{r.lookahead} | {r.context_bytes} |"
            )
        else:
            lines.append(
                f"| {r.backend} | {r.family} | {r.status}: {r.note} | - | - | - | - | - | - | - |"
            )
    lines += ["", "## outputs", ""]
    for r in rows:
        if r.status == "ok":
            lines += [f"### {r.backend}", "", "```", r.text, "```", ""]
    return "\n".join(lines)


def write_report(rows: list[ComparisonRow], out_dir: Path, *, prompt: str = "") -> Path:
    """Write report.md, report.json, and per-backend output files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(rows, prompt=prompt)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    outputs = out_dir / "outputs"
    outputs.mkdir(exist_ok=True)
    for r in rows:
        if r.status == "ok":
            (outputs / f"{r.backend}.txt").write_text(r.text, encoding="utf-8")
    return out_dir / "report.md"
