# compression-lm

Text generation and comparative analysis using general-purpose compressors as
language models. A compressor that has seen a corpus encodes continuations that
"look like" the corpus in fewer bytes — so compressed length doubles as an
implicit conditional probability, no training required.

Inspired by the idea behind `gzipt` (gzip as a language model), implemented
independently and extended to compare many compression algorithms (deflate,
zstd, brotli, lzma/xz, bzip2, lz4, snappy, lzo) side by side.

Work in progress — see `clm --help` for usage.
