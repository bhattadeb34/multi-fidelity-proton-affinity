#!/usr/bin/env python3
"""Build compact plot inputs and selected-feature tables for quick validation.

The command never edits ``data/`` or ``results/``.  By default it writes to a
sibling ``artifacts/`` directory beside ``code/`` and ``data/``.  Existing
files are not replaced unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from artifact_store import ReleaseLayout, file_record, write_manifest


RESULT_FILE_NAMES = {
    "baseline_data.csv",
    "cv_results.json",
    "feature_importance.csv",
    "learning_curve_data.csv",
    "learning_curve_detail.json",
    "mae_summary.csv",
    "predictions.csv",
    "provenance.json",
    "shap_feature_values.csv",
    "shap_importance.csv",
    "shap_values.csv",
    "summary.json",
    "tanimoto_bias_results.csv",
}

SCREENING_PLOT_INPUTS = {
    "candidates.parquet",
    "molecular_pa.parquet",
    "predictions.parquet",
    "llm_verdicts.parquet",
    "pareto_selected.csv",
    "pareto_report.json",
    "pipeline_summary.json",
    "dft_files_summary.csv",
}

DATASET_SPECS = {
    "nist1155": {
        "target": "nist1155_ml.parquet",
        "metadata": ["record_id", "neutral_smiles", "exp_pa_kjmol", "pm7_best_pa_kjmol", "delta_pm7_exp"],
    },
    "kmeans251": {
        "target": "kmeans251_ml.parquet",
        "metadata": ["record_id", "neutral_smiles", "site_idx", "dft_pa_kjmol", "pm7_pa_kjmol", "delta_dft_pm7"],
    },
}


def copy_one(source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to replace {destination}. Pass --force.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_result_inputs(
    layout: ReleaseLayout,
    force: bool,
    records: list[dict[str, object]],
) -> None:
    source_root = layout.code / "results"
    output_root = layout.artifacts / "plot_inputs" / "results"
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or source.name not in RESULT_FILE_NAMES:
            continue
        destination = output_root / source.relative_to(source_root)
        copy_one(source, destination, force)
        records.append(file_record(destination, layout, "plot_input", source))


def consensus_features(cv_path: Path, minimum_folds: int = 3) -> list[str]:
    cv = json.loads(cv_path.read_text())
    counts: Counter[str] = Counter()
    for names in cv["selected_features_per_fold"]:
        counts.update(names)
    return sorted(name for name, count in counts.items() if count >= minimum_folds)


def export_selected_feature_tables(
    layout: ReleaseLayout,
    force: bool,
    records: list[dict[str, object]],
) -> None:
    """Store only features selected in at least one outer fold plus metadata."""
    target_root = layout.data / "targets"
    output_root = layout.artifacts / "features"
    for dataset, spec in DATASET_SPECS.items():
        source = target_root / str(spec["target"])
        cv_path = layout.code / "results" / dataset / "cv_results.json"
        if not source.exists() or not cv_path.exists():
            print(f"SKIP selected features for {dataset}: required source is absent")
            continue
        cv = json.loads(cv_path.read_text())
        selected_union = sorted({name for fold in cv["selected_features_per_fold"] for name in fold})
        available_columns = set(pq.ParquetFile(source).schema.names)
        missing = sorted(set(selected_union) - available_columns)
        if missing:
            raise ValueError(f"{dataset} selected features missing from target table: {missing[:10]}")
        metadata = [name for name in spec["metadata"] if name in available_columns]
        columns = list(dict.fromkeys(metadata + selected_union))
        table = pd.read_parquet(source, columns=columns)
        destination = output_root / f"{dataset}_selected_union.parquet"
        if destination.exists() and not force:
            raise FileExistsError(f"Refusing to replace {destination}. Pass --force.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(destination, index=False)
        records.append(file_record(destination, layout, "selected_feature_table", source))

        description = {
            "dataset": dataset,
            "rows": len(table),
            "metadata_columns": metadata,
            "selected_feature_union": selected_union,
            "consensus_features_minimum_3_folds": consensus_features(cv_path),
            "source_target_table": source.name,
            "source_cv_results": cv_path.relative_to(layout.code).as_posix(),
        }
        metadata_path = output_root / f"{dataset}_selected_union.json"
        if metadata_path.exists() and not force:
            raise FileExistsError(f"Refusing to replace {metadata_path}. Pass --force.")
        metadata_path.write_text(json.dumps(description, indent=2) + "\n")
        records.append(file_record(metadata_path, layout, "selected_feature_metadata"))


def copy_screening_inputs(
    layout: ReleaseLayout,
    iteration: int,
    force: bool,
    records: list[dict[str, object]],
) -> None:
    source_root = layout.data / "screening" / f"iter{iteration}"
    output_root = layout.artifacts / "plot_inputs" / "screening" / f"iter{iteration}"
    if not source_root.exists():
        print(f"SKIP screening plot inputs: {source_root} is absent")
        return
    for name in sorted(SCREENING_PLOT_INPUTS):
        source = source_root / name
        if not source.exists():
            continue
        destination = output_root / name
        copy_one(source, destination, force)
        records.append(file_record(destination, layout, "screening_plot_input", source))


def export_screening_feature_table(
    layout: ReleaseLayout,
    iteration: int,
    force: bool,
    records: list[dict[str, object]],
) -> None:
    """Export the consensus screening columns without duplicating all descriptors."""
    source = layout.data / "screening" / f"iter{iteration}" / "features.parquet"
    cv_path = layout.code / "results" / "kmeans251" / "cv_results.json"
    if not source.exists() or not cv_path.exists():
        print("SKIP screening feature checkpoint: required source is absent")
        return
    features = consensus_features(cv_path)
    available = set(pq.ParquetFile(source).schema.names)
    missing = sorted(set(features) - available)
    if missing:
        raise ValueError(f"Screening consensus features are absent: {missing[:10]}")
    metadata_candidates = [
        "mol_id",
        "smiles",
        "protonated_smiles",
        "site_idx",
        "site_element",
        "site_index",
        "site_normalized_index",
        "site_n_sites",
        "pa_pm7_kcalmol",
    ]
    metadata = [name for name in metadata_candidates if name in available]
    table = pd.read_parquet(source, columns=metadata + features)
    destination = layout.artifacts / "features" / f"screening_iter{iteration}_consensus.parquet"
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to replace {destination}. Pass --force.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(destination, index=False)
    records.append(file_record(destination, layout, "screening_feature_table", source))
    description = {
        "iteration": iteration,
        "rows": len(table),
        "metadata_columns": metadata,
        "consensus_features_minimum_3_folds": features,
        "source_cv_results": cv_path.relative_to(layout.code).as_posix(),
    }
    metadata_path = destination.with_suffix(".json")
    if metadata_path.exists() and not force:
        raise FileExistsError(f"Refusing to replace {metadata_path}. Pass --force.")
    metadata_path.write_text(json.dumps(description, indent=2) + "\n")
    records.append(file_record(metadata_path, layout, "screening_feature_metadata"))


def write_readme(layout: ReleaseLayout, force: bool) -> Path:
    output = layout.artifacts / "README.txt"
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to replace {output}. Pass --force.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        """# Reloadable validation artifacts

This directory contains convenience checkpoints derived from the authoritative
scientific data and result bundles. It is deliberately separate from `data/`.

- `plot_inputs/` contains compact copies of saved numerical results used by the
  quick plotting path. Plotting from these files does not rerun feature
  selection or train a model.
- `features/` contains the union of features selected in the archived outer
  folds, plus identifiers and targets. These tables avoid recomputing molecular
  descriptors or feature selection.
- `model_checkpoints/` is created by `export_model_checkpoints.py`. Each fold
  stores its fitted estimator, feature order, training-fold medians, and a
  validation report against the archived predictions.
- `artifact_manifest.json` records source paths, SHA-256 hashes, sizes, and the
  software environment used to construct this directory.

Source data remain authoritative. A checkpoint is considered verified only
when its validation report shows that it reproduced the archived held-out
predictions within the stated numerical tolerance.

Build or refresh this directory from the release root with:

```bash
python code/scripts/reproducibility/build_quick_artifacts.py
python code/scripts/reproducibility/export_model_checkpoints.py --dataset all
python code/scripts/reproducibility/validate_quick_artifacts.py
```

Use `code/scripts/reproducibility/load_checkpoint.py` to apply one saved model
to a Parquet or CSV feature table. The loader verifies hashes and restores the
saved feature order and training medians before prediction.

Use `--force` only when intentionally rebuilding a previously generated
artifact directory.
"""
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-screening", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    layout = ReleaseLayout.discover(args.code_root, args.artifact_root, args.data_root)
    records: list[dict[str, object]] = []
    copy_result_inputs(layout, args.force, records)
    if not args.skip_features:
        export_selected_feature_tables(layout, args.force, records)
    if not args.skip_screening:
        copy_screening_inputs(layout, args.iteration, args.force, records)
        export_screening_feature_table(layout, args.iteration, args.force, records)
    readme = write_readme(layout, args.force)
    records.append(file_record(readme, layout, "documentation"))
    manifest = layout.artifacts / "artifact_manifest.json"
    if manifest.exists() and not args.force:
        raise FileExistsError(f"Refusing to replace {manifest}. Pass --force.")
    write_manifest(manifest, layout, records, " ".join(sys.argv))
    print(f"Wrote {len(records)} reloadable artifacts under {layout.artifacts}")


if __name__ == "__main__":
    main()
