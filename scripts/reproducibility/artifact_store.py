#!/usr/bin/env python3
"""Shared paths, hashes, and manifest helpers for reloadable artifacts.

Scientific source data remain under ``data/`` and analysis outputs remain
under ``code/results/``.  Reloadable convenience artifacts are written to a
separate top-level ``artifacts/`` directory in a complete release.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "1.0"


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReleaseLayout:
    """Resolved locations in either a source checkout or complete release."""

    code: Path
    release: Path
    data: Path
    artifacts: Path

    @classmethod
    def discover(
        cls,
        code_root: Path | None = None,
        artifact_root: Path | None = None,
        data_root: Path | None = None,
    ) -> "ReleaseLayout":
        code = (code_root or Path(__file__).resolve().parents[2]).resolve()
        release = code.parent
        data_candidates = (
            data_root,
            Path(os.environ["MFPA_DATA_ROOT"]) if "MFPA_DATA_ROOT" in os.environ else None,
            release / "data",
            code / "data",
        )
        data = next((Path(p).expanduser().resolve() for p in data_candidates if p and Path(p).expanduser().exists()),
                    (release / "data").resolve())
        artifacts = Path(
            artifact_root
            or os.environ.get("MFPA_ARTIFACT_ROOT", release / "artifacts")
        ).expanduser().resolve()
        return cls(code=code, release=release, data=data, artifacts=artifacts)

    def describe(self) -> dict[str, str]:
        def portable(path: Path) -> str:
            try:
                return path.relative_to(self.release).as_posix() or "."
            except ValueError:
                return "external"

        return {
            "release": ".",
            "code": portable(self.code),
            "data": portable(self.data),
            "artifacts": portable(self.artifacts),
        }


def relative_to_release(path: Path, layout: ReleaseLayout) -> str:
    """Return a portable release-relative path, or an explicit external path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(layout.release).as_posix()
    except ValueError:
        return f"external:{resolved}"


def file_record(
    path: Path,
    layout: ReleaseLayout,
    role: str,
    source: Path | None = None,
) -> dict[str, object]:
    """Build one manifest record for an artifact or indexed source file."""
    record: dict[str, object] = {
        "path": relative_to_release(path, layout),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if source is not None:
        record["source_path"] = relative_to_release(source, layout)
        record["source_sha256"] = sha256(source)
    return record


def runtime_record(packages: Iterable[str] = ()) -> dict[str, object]:
    """Record the runtime and selected installed package versions."""
    versions: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in packages:
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "not-installed"
    except ImportError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def write_manifest(
    output: Path,
    layout: ReleaseLayout,
    files: list[dict[str, object]],
    command: str,
) -> None:
    """Write a deterministic file list with non-deterministic run metadata isolated."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "layout": layout.describe(),
        "runtime": runtime_record(
            ["numpy", "pandas", "pyarrow", "scikit-learn", "joblib", "matplotlib"]
        ),
        "files": sorted(files, key=lambda item: str(item["path"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
