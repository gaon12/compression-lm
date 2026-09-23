"""Fetch the tiny_shakespeare corpus into data/.

Source: karpathy/char-rnn (the standard ~1.1 MB Shakespeare subset used by
char-rnn and nanoGPT). Kept out of the package proper; run once before
using the CLI examples:

    python scripts/fetch_data.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = Path(__file__).resolve().parent.parent / "data" / "tiny_shakespeare.txt"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"already present: {DEST} ({DEST.stat().st_size} bytes)")
        return 0
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()[:16]
    print(f"wrote {DEST} ({DEST.stat().st_size} bytes, sha256:{digest}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
