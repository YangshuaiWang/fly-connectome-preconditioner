#!/usr/bin/env python3
"""Fetch the official FlyVis optic-lobe connectome specification."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import ssl
import urllib.request

import certifi


URL = (
    "https://raw.githubusercontent.com/TuragaLab/flyvis/"
    "92b3845cc426dd309a1a0e1b3890156c42e14021/"
    "flyvis/connectome/fib25-fib19_v2.2.json"
)
EXPECTED_SHA256 = "bfbb0766251ff09e22723d0ebbf7b14793e70b3ae8ad0eea64eac9d28223351a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fib25-fib19_v2.2.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URL, timeout=60, context=context) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Connectome digest mismatch: {digest}")
    args.output.write_bytes(payload)
    print(f"Downloaded {len(payload):,} bytes to {args.output}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
