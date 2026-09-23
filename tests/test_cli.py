"""CLI smoke tests + compare/report plumbing."""

import json

import pytest

from compression_lm.cli import _parse_opt, main
from compression_lm.metrics import corpus_overlap, distinct_ratio, self_repetition

CORPUS_TEXT = b"First Citizen:\nBefore we proceed any further, hear me speak.\n" * 10


@pytest.fixture()
def corpus_file(tmp_path):
    f = tmp_path / "corpus.txt"
    f.write_bytes(CORPUS_TEXT)
    return f


def test_parse_opt_casts_numbers():
    assert _parse_opt(["level=9", "quality=4.5", "mode=text"]) == {
        "level": 9,
        "quality": 4.5,
        "mode": "text",
    }


def test_cli_backends(capsys):
    assert main(["backends"]) == 0
    out = capsys.readouterr().out
    assert "deflate" in out and "zstd" in out and "available" in out


def test_cli_generate(capsys, corpus_file):
    rc = main(
        [
            "generate",
            "--corpus",
            str(corpus_file),
            "--backend",
            "deflate",
            "--prompt",
            "First Citizen:",
            "--length",
            "32",
            "--lookahead",
            "8",
            "--beam-width",
            "8",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("First Citizen:")
    assert len(out.strip()) > len("First Citizen:")


def test_cli_generate_stats(capsys, corpus_file):
    rc = main(
        [
            "generate",
            "--corpus",
            str(corpus_file),
            "--length",
            "16",
            "--lookahead",
            "4",
            "--beam-width",
            "4",
            "--stats",
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "bits/B" in err and "copy=" in err


def test_cli_compare_writes_report(capsys, corpus_file, tmp_path):
    out_dir = tmp_path / "report"
    rc = main(
        [
            "compare",
            "--corpus",
            str(corpus_file),
            "--backends",
            "deflate,zstd",
            "--prompt",
            "First Citizen:",
            "--length",
            "24",
            "--lookahead",
            "6",
            "--beam-width",
            "6",
            "--out",
            str(out_dir),
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "deflate" in md and "zstd" in md
    rows = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert {r["backend"] for r in rows} == {"deflate", "zstd"}
    assert (out_dir / "outputs" / "deflate.txt").exists()


def test_metrics_functions():
    assert corpus_overlap(b"hear me speak", CORPUS_TEXT) == 1.0
    assert corpus_overlap(b"\x00\x01\x02\x03zzz", CORPUS_TEXT) == 0.0
    assert self_repetition(b"abcabcabcabc", 3) > 0.5
    assert distinct_ratio(b"abcdefghij", 4) == 1.0
