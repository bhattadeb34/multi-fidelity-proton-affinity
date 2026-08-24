#!/usr/bin/env python3
"""Create a complete SHA-256 manifest for a prepared release directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.release_root.expanduser().resolve()
    output = (args.output or root / "SHA256SUMS").expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to replace {output}; pass --force")
    links = [path for path in root.rglob("*") if path.is_symlink()]
    if links:
        raise ValueError(f"Release contains symlinks, first match: {links[0]}")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.resolve() != output
    )
    if not files:
        raise ValueError(f"No files found under {root}")
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} checksums to {output}")


if __name__ == "__main__":
    main()
