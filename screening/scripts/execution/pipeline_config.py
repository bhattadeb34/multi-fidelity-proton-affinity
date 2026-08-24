"""Load and validate portable configuration for the screening pipeline.

The public functions retain the dictionary interface used by the stage
scripts.  The dataclasses in this module keep path resolution and validation
in one place so that individual stages do not need machine-specific paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


KJMOL_TO_KCAL = 1 / 4.184


@dataclass(frozen=True)
class RepositoryPaths:
    """Repository paths shared by every screening stage."""

    project: Path
    screening: Path
    data: Path
    results: Path
    config: Path

    @classmethod
    def from_module(cls, module_path: str | Path = __file__) -> "RepositoryPaths":
        screening = Path(module_path).resolve().parents[2]
        project = screening.parent
        data_root = next(
            (path for path in (project / "data", project.parent / "data") if path.exists()),
            project / "data",
        )
        return cls(
            project=project,
            screening=screening,
            data=data_root / "screening",
            results=project / "results",
            config=screening / "config" / "pipeline_config.json",
        )

    def iteration_dir(self, iteration: int) -> Path:
        """Return the data directory for a positive screening iteration."""
        if iteration < 1:
            raise ValueError("Screening iteration must be a positive integer")
        return self.data / f"iter{iteration}"

    def dft_validation_dir(
        self,
        iteration: int,
        explicit: str | Path | None = None,
        required_files: tuple[str, ...] = (),
    ) -> Path:
        """Resolve one DFT-validation archive without assuming a date suffix.

        An explicit directory takes precedence. Otherwise, the method searches
        for directories named ``iterN`` or ``iterN_*`` and fails when none or
        more than one satisfy the requested file requirements.
        """
        if iteration < 1:
            raise ValueError("Screening iteration must be a positive integer")
        if explicit is not None:
            candidates = [Path(explicit).expanduser().resolve()]
        else:
            root = self.data / "dft_validation"
            candidates = sorted(
                path for path in root.glob(f"iter{iteration}*") if path.is_dir()
            )
        matches = [
            path for path in candidates
            if all((path / name).is_file() for name in required_files)
        ]
        if len(matches) != 1:
            details = ", ".join(str(path) for path in matches) or "none"
            raise FileNotFoundError(
                f"Expected one DFT-validation directory for iteration "
                f"{iteration} containing {list(required_files)}, found: {details}. "
                "Pass an explicit directory when multiple archives are present."
            )
        return matches[0]


@dataclass(frozen=True)
class ScreeningConfiguration:
    """Validated screening settings loaded from JSON."""

    values: dict[str, Any]
    source: Path

    @classmethod
    def load(cls, path: str | Path) -> "ScreeningConfiguration":
        source = Path(path).expanduser().resolve()
        with source.open() as handle:
            values = json.load(handle)
        config = cls(values=values, source=source)
        config.validate()
        return config

    def validate(self) -> None:
        """Fail early when required sections or bounded values are invalid."""
        required = {
            "consensus_min_folds",
            "featurization",
            "random_forest",
            "pa_window_kcalmol",
            "rule_filter",
            "llm",
            "pareto",
            "lead_selection",
            "retrieval",
        }
        missing = sorted(required - self.values.keys())
        if missing:
            raise ValueError(f"Missing screening configuration sections: {missing}")

        low = float(self.values["pa_window_kcalmol"]["low"])
        high = float(self.values["pa_window_kcalmol"]["high"])
        if low >= high:
            raise ValueError("PA window lower bound must be below its upper bound")

        folds = int(self.values["consensus_min_folds"])
        if not 1 <= folds <= 5:
            raise ValueError("consensus_min_folds must be between 1 and 5")

        weights = self.values["pareto"]["objective_weights"]
        if any(float(value) < 0 for value in weights.values()):
            raise ValueError("Pareto objective weights cannot be negative")
        if abs(sum(map(float, weights.values())) - 1.0) > 1e-9:
            raise ValueError("Pareto objective weights must sum to 1")

        if int(self.values["pareto"]["n_select"]) < 1:
            raise ValueError("pareto.n_select must be positive")
        if int(self.values["lead_selection"]["n_leads"]) < 1:
            raise ValueError("lead_selection.n_leads must be positive")

        retrieval = self.values["retrieval"]
        tanimoto_min = float(retrieval["tanimoto_min"])
        tanimoto_max = float(retrieval["tanimoto_max"])
        if not 0 <= tanimoto_min < tanimoto_max <= 1:
            raise ValueError(
                "retrieval Tanimoto bounds must satisfy 0 <= min < max <= 1"
            )


PATHS = RepositoryPaths.from_module()
SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING = PATHS.screening
PROJECT = PATHS.project
DEFAULT_CONFIG_PATH = PATHS.config


def load_pipeline_config(config_path: str | Path | None = None) -> dict:
    """Load the shared screening configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return ScreeningConfiguration.load(path).values


def load_training_reference_stats(
    target_path: str | Path | None = None,
) -> dict[str, float]:
    """Derive rule-filter reference values from the corrected training targets.

    Values are rounded to the precision used by the previous constants so that
    the default configuration reproduces the current screening decisions.
    """
    path = (
        Path(target_path)
        if target_path
        else PATHS.data.parent / "targets" / "kmeans251_ml.parquet"
    )
    targets = pd.read_parquet(
        path, columns=["neutral_smiles", "dft_pa_kjmol", "pm7_pa_kjmol"]
    )
    varying_site_targets = (
        targets.groupby("neutral_smiles")["dft_pa_kjmol"].nunique() > 1
    )
    if not varying_site_targets.any():
        raise ValueError(
            "Training targets appear to be the stale broadcast-target version: "
            "no multi-site molecule has distinct per-site DFT targets. Rebuild "
            f"the corrected target parquet before screening: {path}"
        )
    dft_pa = targets["dft_pa_kjmol"] * KJMOL_TO_KCAL
    correction = (
        targets["dft_pa_kjmol"] - targets["pm7_pa_kjmol"]
    ) * KJMOL_TO_KCAL
    return {
        "pa_train_max_kcalmol": round(float(dft_pa.max()), 1),
        "correction_mean_kcalmol": round(float(correction.mean()), 2),
        "correction_std_kcalmol": round(float(correction.std()), 2),
    }
