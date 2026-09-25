# data/

`tiny_shakespeare.txt` — the ~1.1 MB Shakespeare subset from
[karpathy/char-rnn](https://github.com/karpathy/char-rnn)
(`data/tinyshakespeare/input.txt`), concatenated Shakespeare plays used as
the default corpus for examples and comparisons.

Re-fetch with `python scripts/fetch_data.py`.

`wikitext2/` (not vendored) — WikiText-2 (`wiki.{train,valid,test}.tokens`
extracted from `wikitext-2-v1.zip`, MD5 `542ccefacc6c27f945fb54453812b3cd`).
Fetched automatically by the v3 benchmark
(`compression_lm.bench_v3.wikitext`); the canonical metamind S3 URL is dead,
so the fetcher falls back to a Wayback Machine snapshot of the same file.
