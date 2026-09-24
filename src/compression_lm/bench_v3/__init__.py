"""Third benchmark phase (v3) — hypothesis verification, not new features.

v3 re-runs the core measurements under stricter methodology:

- identical *effective* context bytes for every backend (4 KiB / 32 KiB),
- explicit, recorded backend options (level/preset/format/dictionary),
- three zstd scoring modes (full / stream / dict) instead of one "zstd",
- multi-byte continuation discrimination (1/4/8/16/32 B, 32-way),
- wall-clock equal-time generation alongside equal-evaluation budgets,
- container-vs-raw framing comparison,
- WikiText-2 replication.

v2 modules under ``compression_lm.bench`` are reused unchanged; everything
here is additive. Results land in ``results_v3/`` and never overwrite v2.
"""
