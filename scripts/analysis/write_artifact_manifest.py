#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for a publication artifact folder."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", default="SHA256SUMS")
    args = parser.parse_args()

    root = args.directory.resolve()
    output = root / args.output
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.resolve() != output.resolve()
    )
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output} ({len(files)} files)")


if __name__ == "__main__":
    main()
