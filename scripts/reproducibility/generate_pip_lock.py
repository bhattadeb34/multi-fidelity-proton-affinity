#!/usr/bin/env python3
"""Convert a pip installation report into a hash-locked requirements file.

Generate the report for a specific Python version and platform with pip's
``--dry-run --report`` options. This script records the exact resolved versions
and downloaded archive hashes. It does not resolve packages or use the network.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalized_name(name: str) -> str:
    """Return the canonical spelling used for deterministic sorting."""
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_lines(report: dict) -> list[str]:
    """Build sorted PEP 508 requirement lines with SHA-256 hashes."""
    resolved: list[tuple[str, str, str]] = []
    for item in report.get("install", []):
        metadata = item["metadata"]
        hashes = item.get("download_info", {}).get("archive_info", {}).get("hashes", {})
        digest = hashes.get("sha256")
        if not digest:
            raise ValueError(f"No SHA-256 archive hash for {metadata['name']}")
        resolved.append((metadata["name"], metadata["version"], digest))
    if not resolved:
        raise ValueError("The pip report contains no resolved installations")
    resolved.sort(key=lambda item: normalized_name(item[0]))
    return [
        f"{name}=={version} --hash=sha256:{digest}"
        for name, version, digest in resolved
    ]


def minimal_resolution(report: dict) -> dict:
    """Keep only fields needed to audit or regenerate the package lock."""
    packages = []
    for item in report.get("install", []):
        metadata = item["metadata"]
        download = item.get("download_info", {})
        digest = download.get("archive_info", {}).get("hashes", {}).get("sha256")
        if not digest:
            raise ValueError(f"No SHA-256 archive hash for {metadata['name']}")
        packages.append(
            {
                "name": metadata["name"],
                "version": metadata["version"],
                "url": download.get("url"),
                "sha256": digest,
            }
        )
    packages.sort(key=lambda item: normalized_name(item["name"]))
    return {
        "pip_version": report.get("pip_version"),
        "environment": report.get("environment", {}),
        "packages": packages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON file produced by pip --report")
    parser.add_argument("output", type=Path, help="Hash-locked requirements output")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional minimal JSON resolution record without package descriptions",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to replace {args.output}; pass --force")
    report = json.loads(args.report.read_text())
    header = [
        "# Exact pip package set resolved for CPython 3.10 on Linux x86_64.",
        "# Install with: pip install --require-hashes -r requirements-lock-py310-linux-x86_64.txt",
        "# Regenerate from the archived pip report with scripts/reproducibility/generate_pip_lock.py.",
    ]
    args.output.write_text("\n".join(header + lock_lines(report)) + "\n")
    if args.metadata_output:
        if args.metadata_output.exists() and not args.force:
            raise FileExistsError(
                f"Refusing to replace {args.metadata_output}; pass --force"
            )
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(minimal_resolution(report), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
