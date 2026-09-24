"""Benchmark suite: rigorous evaluation of compressors as language models.

Three questions, measured separately:

1. Teacher-forced prediction — how well does a compressor rank the true next
   byte of held-out text among all 256 candidates?
2. Equal-search generation — same beam/lookahead for every backend.
3. Equal-compute generation — same candidate-evaluation budget.

Plus retrieval/copy analysis so that "compresses well" is never confused
with "predicts well".
"""
