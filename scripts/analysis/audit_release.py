#!/usr/bin/env python3
"""Audit a checkout before a GitHub or Zenodo release.

The audit is read-only. It checks tracked filenames and text for internal
workflow traces, credentials, and personal absolute paths. It also parses all
tracked Python files and can verify every tracked SHA-256 manifest.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".log",
    ".txt",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ARCHIVED_PROVENANCE_PREFIXES = (
    "docs/publication_run_logs/",
)


@dataclass
class RepositoryAudit:
    """Collect publication-release failures for one repository checkout."""

    root: Path
    verify_checksums: bool = False
    failures: list[str] = field(default_factory=list)

    def release_files(self) -> list[Path]:
        """Return tracked and non-ignored untracked files in the checkout."""
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=self.root,
            check=True,
            capture_output=True,
        )
        paths = [
            self.root / name.decode()
            for name in result.stdout.split(b"\0")
            if name
        ]
        return [path for path in paths if path.is_file()]

    def run(self) -> list[str]:
        files = self.release_files()
        self._check_filenames(files)
        self._check_text(files)
        self._check_python(files)
        self._check_manifests(files)
        return self.failures

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _check_filenames(self, files: list[Path]) -> None:
        forbidden_names = re.compile(
            r"(workflow_prompt|verification_prompt|handoff)", re.I
        )
        for path in files:
            relative = self._relative(path)
            if relative == "data" or relative.startswith("data/"):
                self.failures.append(f"tracked external data path: {relative}")
            if forbidden_names.search(relative):
                self.failures.append(f"internal workflow filename: {relative}")

    def _check_text(self, files: list[Path]) -> None:
        secret_patterns = {
            "GitHub token": re.compile(r"ghp_[A-Za-z0-9]{20,}"),
            "Google API key": re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
            "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
        }
        personal_path = re.compile(r"/(?:Users|home|noether/s[01])/[A-Za-z0-9._-]+/")

        for path in files:
            if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
                continue
            relative = self._relative(path)
            if relative == "scripts/analysis/audit_release.py":
                continue
            try:
                text = path.read_text(errors="strict")
            except UnicodeDecodeError:
                continue
            for label, pattern in secret_patterns.items():
                if pattern.search(text):
                    self.failures.append(f"{label} pattern: {relative}")
            if (
                personal_path.search(text)
                and not relative.startswith(ARCHIVED_PROVENANCE_PREFIXES)
            ):
                self.failures.append(f"personal absolute path: {relative}")

    def _check_python(self, files: list[Path]) -> None:
        for path in files:
            if path.suffix != ".py" or not path.is_file():
                continue
            try:
                ast.parse(path.read_text(), filename=self._relative(path))
            except SyntaxError as exc:
                self.failures.append(
                    f"Python syntax error: {self._relative(path)}:{exc.lineno}"
                )

    def _check_manifests(self, files: list[Path]) -> None:
        manifests = [path for path in files if path.name == "SHA256SUMS"]
        for manifest in manifests:
            for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
                if not match:
                    self.failures.append(
                        f"invalid checksum line: {self._relative(manifest)}:{line_number}"
                    )
                    continue
                expected, name = match.groups()
                target = manifest.parent / name
                if not target.is_file():
                    self.failures.append(
                        f"manifest target missing: {self._relative(target)}"
                    )
                    continue
                if self.verify_checksums:
                    actual = hashlib.sha256(target.read_bytes()).hexdigest()
                    if actual != expected:
                        self.failures.append(
                            f"checksum mismatch: {self._relative(target)}"
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Hash every file listed by a tracked SHA256SUMS manifest",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    audit = RepositoryAudit(root=root, verify_checksums=args.verify_checksums)
    failures = audit.run()
    if failures:
        print("Release audit failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("Release audit passed.")


if __name__ == "__main__":
    main()
