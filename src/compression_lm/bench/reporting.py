"""Result persistence: CSV/JSON writers and the final report builder.

Every benchmark task appends its rows as a JSON file under
``results/raw/rows/`` keyed by task id, so a crash only loses the task in
flight and ``--resume`` skips everything already done. The final assembly
step merges row files into the spec'd CSVs.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    """Write row dicts to CSV with the union of keys as columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_jsonl_rows(raw_dir: Path) -> list[dict]:
    """Concatenate every row file under results/raw/rows/."""
    rows_dir = raw_dir / "rows"
    out: list[dict] = []
    if not rows_dir.exists():
        return out
    for f in sorted(rows_dir.glob("*.json")):
        try:
            out.extend(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue  # partial write from a crash — skip
    return out


def rows_where(rows: Iterable[dict], **eq) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def environment_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os": platform.system(),
    }


def fmt(x, spec=".3f") -> str:
    if isinstance(x, float):
        return format(x, spec)
    return str(x)


def md_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)
