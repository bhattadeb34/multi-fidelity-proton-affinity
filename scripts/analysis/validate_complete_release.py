#!/usr/bin/env python3
"""Run read-only checks on a complete code-and-data release.

Expected values are loaded from manifests, configuration, and generated
summaries. The validator does not repeat manuscript counts in source code. It
does not refit models, run PM7 or DFT, query an LLM, or contact a database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    """Hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_layout(code_dir: Path) -> tuple[Path, Path, Path]:
    """Return release root, code root, and scientific data root."""
    code_dir = code_dir.resolve()
    sibling_data = code_dir.parent / "data"
    nested_data = code_dir / "data"
    if sibling_data.is_dir():
        return code_dir.parent, code_dir, sibling_data
    if nested_data.is_dir():
        return code_dir, code_dir, nested_data
    raise FileNotFoundError(
        "No data directory found beside the code directory or inside it"
    )


def discover_unique_directory(
    explicit: Path | None,
    root: Path,
    pattern: str,
    required_files: tuple[str, ...],
) -> Path:
    """Resolve one artifact directory and fail on ambiguous discovery."""
    candidates = (
        [explicit.expanduser().resolve()]
        if explicit is not None
        else sorted(path for path in root.glob(pattern) if path.is_dir())
    )
    matches = [
        path for path in candidates
        if all((path / name).is_file() for name in required_files)
    ]
    if len(matches) != 1:
        found = ", ".join(str(path) for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected one directory under {root} matching {pattern!r} and "
            f"containing {list(required_files)}, found: {found}"
        )
    return matches[0]


def close(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0)


def documentation_file(directory: Path, stem: str = "README") -> Path:
    """Resolve Markdown or plain-text documentation in packaged releases."""
    for suffix in (".md", ".txt"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return directory / f"{stem}.md"


@dataclass
class CompleteReleaseValidator:
    """Collect failures and publication blockers from non-mutating checks."""

    release_root: Path
    code: Path
    data: Path
    screening_dir: Path | None = None
    dft_dir: Path | None = None
    verify_checksums: bool = False
    publication_ready: bool = False
    failures: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    checks: dict[str, object] = field(default_factory=dict)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)

    def run_check(self, name: str, function) -> None:
        """Record an unexpected check error instead of aborting the audit."""
        try:
            function()
        except Exception as exc:
            self.failures.append(f"{name} check could not complete: {exc}")

    def run(self) -> dict[str, object]:
        checks = [
            ("layout", self.check_layout),
            ("manifest", self.check_manifest),
            ("reloadable artifacts", self.check_reloadable_artifacts),
            ("input fingerprints", self.check_input_fingerprints),
            ("CV bundles", self.check_cv_bundles),
            ("group isolation", self.check_group_isolation),
            ("downstream analyses", self.check_downstream_analyses),
            ("screening", self.check_screening),
            ("publication metadata", self.check_publication_metadata),
        ]
        for name, function in checks:
            self.run_check(name, function)
        valid = not self.failures and (not self.publication_ready or not self.blockers)
        return {
            "release_root": str(self.release_root),
            "checks": self.checks,
            "failures": self.failures,
            "publication_blockers": self.blockers,
            "publication_ready_requested": self.publication_ready,
            "valid": valid,
        }

    def check_layout(self) -> None:
        required = [
            documentation_file(self.code),
            self.code / "CITATION.cff",
            self.code / "requirements.txt",
            self.code / "environment.yml",
            self.code / "results" / "publication_summary.json",
            self.code / "docs" / "input_data_fingerprints.json",
            self.code / "screening" / "config" / "pipeline_config.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        self.require(not missing, f"required files missing: {missing}")
        self.require(
            not (self.release_root / ".git").exists()
            and not (self.code / ".git").exists(),
            "complete release contains Git metadata",
        )
        links = [path for path in self.release_root.rglob("*") if path.is_symlink()]
        self.require(not links, f"complete release contains symlinks: {links[:10]}")
        generated = [
            str(path.relative_to(self.release_root))
            for path in self.release_root.rglob("*")
            if path.name == ".DS_Store"
            or path.name.startswith("._")
            or path.suffix == ".pyc"
            or "__pycache__" in path.parts
        ]
        self.require(
            not generated,
            f"complete release contains generated platform files: {generated[:20]}",
        )
        forbidden = re.compile(
            r"(verification_prompt|workflow_prompt|handoff|manuscript_verification)",
            re.I,
        )
        internal = [
            str(path.relative_to(self.release_root))
            for path in self.release_root.rglob("*")
            if forbidden.search(path.name)
        ]
        self.require(not internal, f"internal workflow files are present: {internal}")
        self.checks["required_files"] = len(required) - len(missing)
        self.checks["symlinks"] = len(links)
        self.checks["generated_platform_files"] = len(generated)

    def check_manifest(self) -> None:
        manifest = self.release_root / "SHA256SUMS"
        if not manifest.is_file():
            self.failures.append("release-level SHA256SUMS is missing")
            return
        pattern = re.compile(r"([0-9a-f]{64})  (.+)")
        entries: dict[str, str] = {}
        for number, line in enumerate(manifest.read_text().splitlines(), 1):
            if not line:
                continue
            match = pattern.fullmatch(line)
            if match is None:
                self.failures.append(f"invalid manifest line {number}")
                continue
            digest, name = match.groups()
            if name in entries:
                self.failures.append(f"duplicate manifest path: {name}")
                continue
            entries[name] = digest
            path = self.release_root / name
            if not path.is_file():
                self.failures.append(f"manifest file missing: {name}")
            elif self.verify_checksums and sha256(path) != digest:
                self.failures.append(f"checksum mismatch: {name}")
        actual = {
            path.relative_to(self.release_root).as_posix()
            for path in self.release_root.rglob("*")
            if path.is_file() and path != manifest
        }
        listed = set(entries)
        self.require(not (actual - listed), f"files absent from manifest: {sorted(actual - listed)[:20]}")
        self.require(not (listed - actual), f"manifest-only paths: {sorted(listed - actual)[:20]}")
        self.checks["manifest_entries"] = len(entries)
        self.checks["manifest_content_hashed"] = self.verify_checksums
        self.checks["manifest_covers_all_files"] = actual == listed

    def check_input_fingerprints(self) -> None:
        record_path = self.code / "docs" / "input_data_fingerprints.json"
        record = json.loads(record_path.read_text())
        files = record.get("files", {})
        self.require(bool(files), "input fingerprint record is empty")
        checked = []
        for relative, metadata in files.items():
            path = self.release_root / relative
            if not path.is_file():
                self.failures.append(f"fingerprinted input missing: {relative}")
                continue
            self.require(path.stat().st_size == int(metadata["bytes"]), f"input size mismatch: {relative}")
            self.require(sha256(path) == metadata["sha256"], f"input hash mismatch: {relative}")
            checked.append(relative)
        self.checks["fingerprinted_inputs"] = checked

    def check_reloadable_artifacts(self) -> None:
        """Validate the optional restart layer when it is included in a release."""
        artifacts = self.release_root / "artifacts"
        if not artifacts.exists():
            self.checks["reloadable_artifacts"] = "not included"
            return
        required = [documentation_file(artifacts), artifacts / "artifact_manifest.json"]
        missing = [str(path) for path in required if not path.is_file()]
        self.require(not missing, f"reloadable artifact files missing: {missing}")
        if missing:
            return
        validator = (
            self.code / "scripts" / "reproducibility" / "validate_quick_artifacts.py"
        )
        process = subprocess.run(
            [
                sys.executable,
                str(validator),
                "--code-root",
                str(self.code),
                "--data-root",
                str(self.data),
                "--artifact-root",
                str(artifacts),
            ],
            cwd=self.code,
            capture_output=True,
            text=True,
        )
        self.require(
            process.returncode == 0,
            "reloadable artifact validation failed: "
            + (process.stderr or process.stdout),
        )
        manifest = json.loads((artifacts / "artifact_manifest.json").read_text())
        self.checks["reloadable_artifacts"] = {
            "files": len(manifest.get("files", [])),
            "model_checkpoint_manifest": (
                artifacts / "model_checkpoints" / "manifest.json"
            ).is_file(),
            "validated": process.returncode == 0,
        }

    def result_bundles(self) -> list[Path]:
        bundles = []
        for cv_path in sorted((self.code / "results").rglob("cv_results.json")):
            directory = cv_path.parent
            companions = [directory / "predictions.csv", directory / "mae_summary.csv"]
            if all(path.is_file() for path in companions):
                bundles.append(directory)
            else:
                self.failures.append(f"incomplete CV bundle: {directory}")
        return bundles

    def check_cv_bundles(self) -> None:
        validator = self.code / "scripts" / "analysis" / "validate_result_bundle.py"
        bundles = self.result_bundles()
        self.require(bool(bundles), "no complete CV bundles were found")
        for result_dir in bundles:
            process = subprocess.run(
                [sys.executable, str(validator), str(result_dir)],
                cwd=self.code,
                capture_output=True,
                text=True,
            )
            if process.returncode:
                self.failures.append(
                    f"CV bundle failed reconciliation at {result_dir}: "
                    f"{process.stderr or process.stdout}"
                )
        self.checks["cv_bundles_checked"] = [
            str(path.relative_to(self.code)) for path in bundles
        ]

    def check_group_isolation(self) -> None:
        grouped = {}
        for directory in self.result_bundles():
            cv = json.loads((directory / "cv_results.json").read_text())
            group_col = cv.get("group_col")
            if not group_col:
                continue
            predictions = pd.read_csv(directory / "predictions.csv")
            if group_col not in predictions:
                self.failures.append(f"{directory}: predictions lack {group_col}")
                continue
            maxima = {
                model: int(frame.groupby(group_col)["fold"].nunique().max())
                for model, frame in predictions.groupby("model")
            }
            maximum = max(maxima.values())
            self.require(
                maximum == 1,
                f"{directory}: a group spans multiple held-out folds",
            )
            grouped[str(directory.relative_to(self.code))] = {
                "group_col": group_col,
                "groups": int(predictions[group_col].nunique()),
                "max_folds_per_group": maximum,
            }
        self.checks["grouped_cv"] = grouped

    def check_downstream_analyses(self) -> None:
        """Check saved rerun audits and reconcile the publication summary."""
        preprocessing = json.loads(
            (self.code / "results" / "preprocessing_equivalence_audit.json").read_text()
        )
        postprocessing = json.loads(
            (self.code / "results" / "postprocessing_validation_report.json").read_text()
        )
        self.require(bool(preprocessing.get("valid")), "preprocessing audit is not valid")
        self.require(bool(postprocessing.get("valid")), "postprocessing audit is not valid")
        self.require(not postprocessing.get("failures"), "postprocessing audit reports failures")

        publication = json.loads(
            (self.code / "results" / "publication_summary.json").read_text()
        )
        compared = {}
        sections = ("production", "dft_ablations")
        for section in sections:
            for dataset, expected in publication.get(section, {}).items():
                cv_path = self.code / "results" / dataset / "cv_results.json"
                cv = json.loads(cv_path.read_text())
                model = expected["model"]
                metrics = cv["models"][model]
                self.require(
                    close(metrics["mae_pa_mean"], expected["mae_mean"]),
                    f"publication MAE mismatch: {dataset}",
                )
                self.require(
                    close(metrics["mae_pa_std"], expected["mae_std"]),
                    f"publication MAE spread mismatch: {dataset}",
                )
                record = {"model": model, "mae_mean": metrics["mae_pa_mean"]}
                if "fold_feature_counts" in expected:
                    counts = [len(features) for features in cv["selected_features_per_fold"]]
                    self.require(
                        counts == expected["fold_feature_counts"],
                        f"publication feature counts mismatch: {dataset}",
                    )
                    frequencies: dict[str, int] = {}
                    for features in cv["selected_features_per_fold"]:
                        for feature in features:
                            frequencies[feature] = frequencies.get(feature, 0) + 1
                    consensus = sum(count >= 3 for count in frequencies.values())
                    self.require(
                        consensus == expected["consensus_features"],
                        f"publication consensus count mismatch: {dataset}",
                    )
                    record["consensus_features"] = consensus
                compared[dataset] = record
        self.checks["downstream_reports"] = {
            "preprocessing_valid": bool(preprocessing.get("valid")),
            "postprocessing_valid": bool(postprocessing.get("valid")),
            "publication_results": compared,
        }

    def check_screening(self) -> None:
        screening_root = self.data / "screening"
        stage = discover_unique_directory(
            self.screening_dir,
            screening_root,
            "iter*",
            (
                "pipeline_summary.json",
                "candidates.parquet",
                "pm7_results.parquet",
                "pm7_failed.csv",
                "features.parquet",
                "predictions.parquet",
                "molecular_pa.parquet",
                "llm_verdicts.parquet",
                "pareto_selected.csv",
            ),
        )
        dft = discover_unique_directory(
            self.dft_dir,
            screening_root / "dft_validation",
            "iter*",
            ("lead_selection_analysis.csv", "top5_leads.csv"),
        )
        archived = json.loads((stage / "pipeline_summary.json").read_text())
        summary = archived["summary"]
        config = json.loads(
            (self.code / "screening" / "config" / "pipeline_config.json").read_text()
        )
        candidates = pd.read_parquet(stage / "candidates.parquet")
        pm7 = pd.read_parquet(stage / "pm7_results.parquet")
        failed = pd.read_csv(stage / "pm7_failed.csv")
        features = pd.read_parquet(stage / "features.parquet")
        predictions = pd.read_parquet(stage / "predictions.parquet")
        molecular = pd.read_parquet(stage / "molecular_pa.parquet")
        verdicts = pd.read_parquet(stage / "llm_verdicts.parquet")
        pareto = pd.read_csv(stage / "pareto_selected.csv")
        analysis = pd.read_csv(dft / "lead_selection_analysis.csv")
        leads = pd.read_csv(dft / "top5_leads.csv")
        low = float(config["pa_window_kcalmol"]["low"])
        high = float(config["pa_window_kcalmol"]["high"])
        eligible_values = config["pareto"]["eligible_verdicts"]
        eligible = verdicts["final_verdict"].isin(eligible_values)
        in_window = verdicts["pa_pred_kcalmol"].between(low, high)
        derived = {
            "retrieved_candidates": len(candidates),
            "pm7_successful_molecules": pm7["smiles"].nunique(),
            "pm7_failed_molecules": failed["smiles"].nunique(),
            "pm7_successful_sites": len(pm7),
            "featurized_molecules": features["smiles"].nunique(),
            "featurized_sites": len(features),
            "predicted_molecules": molecular["smiles"].nunique(),
            "predicted_sites": len(predictions),
            "predicted_pa_window_molecules": molecular["pa_pred_kcalmol"].between(low, high).sum(),
            "rule_accept": (verdicts["rule_verdict"] == "accept").sum(),
            "rule_flag": (verdicts["rule_verdict"] == "flag").sum(),
            "rule_reject": (verdicts["rule_verdict"] == "reject").sum(),
            "llm_accept": (verdicts["final_verdict"] == "accept").sum(),
            "llm_flag": (verdicts["final_verdict"] == "flag").sum(),
            "llm_reject": (verdicts["final_verdict"] == "reject").sum(),
            "llm_eligible_total": eligible.sum(),
            "llm_eligible_in_pa_window": (eligible & in_window).sum(),
            "pareto_selected": len(pareto),
            "dft_completed": len(analysis),
            "dft_in_pa_window": analysis["dft_pa_in_window"].sum(),
            "stability_eligible": analysis["stability_eligible"].sum(),
            "lead_eligible": analysis["lead_eligible"].sum(),
            "priority_leads": len(leads),
        }
        for key, actual in derived.items():
            self.require(int(actual) == int(summary[key]), f"screening summary mismatch: {key}")
        self.require(
            len(pareto) == int(config["pareto"]["n_select"]),
            "Pareto count differs from pipeline configuration",
        )
        self.require(
            len(leads) == int(config["lead_selection"]["n_leads"]),
            "priority-lead count differs from pipeline configuration",
        )
        self.require(all(archived["checks"].values()), "archived screening checks include a failure")
        ml_error = analysis["ml_error_kcalmol"]
        pm7_error = analysis["pm7_error_kcalmol"]
        self.require(close(ml_error.abs().mean(), summary["prospective_ml_mae_kcalmol"]), "prospective ML MAE mismatch")
        self.require(close(ml_error.mean(), summary["prospective_ml_bias_kcalmol"]), "prospective ML bias mismatch")
        self.require(close(pm7_error.abs().mean(), summary["prospective_pm7_mae_kcalmol"]), "prospective PM7 MAE mismatch")
        publication = json.loads(
            (self.code / "results" / "publication_summary.json").read_text()
        )["screening"]
        publication_map = {
            "retrieved_candidates": "retrieved_candidates",
            "pm7_successful_molecules": "pm7_successful_molecules",
            "pm7_failed_molecules": "pm7_failed_molecules",
            "pm7_successful_sites": "pm7_successful_sites",
            "llm_eligible_in_pa_window": "llm_eligible_in_pa_window",
            "pareto_selected": "pareto_selected",
            "dft_completed": "dft_completed",
            "dft_in_pa_window": "dft_in_pa_window",
            "priority_leads": "priority_leads",
        }
        for public_key, stage_key in publication_map.items():
            self.require(publication[public_key] == summary[stage_key], f"publication summary mismatch: {public_key}")
        self.require(close(publication["prospective_ml_mae"], summary["prospective_ml_mae_kcalmol"]), "publication prospective MAE mismatch")
        self.require(close(publication["prospective_ml_bias"], summary["prospective_ml_bias_kcalmol"]), "publication prospective bias mismatch")
        self.checks["screening_directory"] = str(stage.relative_to(self.release_root))
        self.checks["dft_directory"] = str(dft.relative_to(self.release_root))
        self.checks["screening_counts"] = {key: int(value) for key, value in derived.items()}

    def check_publication_metadata(self) -> None:
        license_files = [
            path for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING")
            if (path := self.code / name).is_file()
        ]
        if not license_files:
            self.blockers.append("No software license file is present")
        citation = (self.code / "CITATION.cff").read_text()
        readme = documentation_file(self.code).read_text()
        if "corrected Zenodo DOI will be" in readme:
            self.blockers.append("Corrected Zenodo DOI is still a README placeholder")
        if re.search(r"(?m)^version:\s*\S+", citation) is None:
            self.blockers.append("CITATION.cff has no release version")
        if re.search(r"(?m)^date-released:\s*\S+", citation) is None:
            self.blockers.append("CITATION.cff has no release date")
        self.checks["license_files"] = [path.name for path in license_files]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Code directory in the complete release",
    )
    parser.add_argument("--screening-dir", type=Path)
    parser.add_argument("--dft-dir", type=Path)
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Hash the content of every file in the release-level manifest",
    )
    parser.add_argument(
        "--publication-ready",
        action="store_true",
        help="Treat missing final release metadata as validation failures",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path. No files are written by default.",
    )
    args = parser.parse_args()
    release_root, code, data = infer_layout(args.code_dir)
    validator = CompleteReleaseValidator(
        release_root=release_root,
        code=code,
        data=data,
        screening_dir=args.screening_dir,
        dft_dir=args.dft_dir,
        verify_checksums=args.verify_checksums,
        publication_ready=args.publication_ready,
    )
    report = validator.run()
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
