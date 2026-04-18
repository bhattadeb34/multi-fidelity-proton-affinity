"""
build_features.py
=================
Orchestrates all feature modules and builds the final feature matrices.

Feature set summary
-------------------
  MACCS fingerprints         :   167   binary structural keys
  Morgan fingerprints        :  1024   count vectors, radius=2
  RDKit descriptors          :   630   210 × 3 states (neutral/protonated/delta)
  PM7 quantum descriptors    :    39   13 × 3 states
  Mordred 2D descriptors     :  4839   1613 × 3 states
  Mordred 3D-only descriptors:   639   213 × 3 states  (when --mordred-states all_states)
                           or    213   213 × 1 state   (when --mordred-states neutral_full_delta_2d)
  Site descriptors           :     6
  ──────────────────────────────────
  Total (3D, all_states)     :  7344
  Total (3D, neu_full)       :  6918   (saves 426 cols vs all_states)
  Total (2D only)            :  6705

Mordred state strategies
------------------------
  all_states            : 3D for neutral, protonated, delta  [default]
  neutral_full_delta_2d : 3D for neutral only; protonated/delta use
                          2D padded to full column layout (3D cols = NaN).
                          Rationale: protonation geometry changes are already
                          captured by RDKit/PM7 delta features.

3D geometry sources
-------------------
  DFT folder records  → B3LYP/def2-TZVP optimized coordinates (highest quality)
  PM7/JSON records    → ETKDG conformer generated from SMILES

Outputs  (../data/features/)
-------
  {name}_features.parquet   — ML-ready feature matrix
  {name}_features.csv
  feature_manifest.json     — column names, counts, run configuration

Usage
-----
  python featurize/build_features.py --dataset all
  python featurize/build_features.py --dataset nist --mordred-states neutral_full_delta_2d
  python featurize/build_features.py --dataset dft  --no-3d
"""

import sys
import json
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running as a script from the scripts/ root or from within featurize/
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fp_maccs     import maccs_from_smiles,        MACCS_COLS
from fp_morgan    import morgan_from_smiles,        MORGAN_COLS
from desc_rdkit   import (rdkit_descs_three_states,
                            NEUTRAL_RDKIT_COLS,
                            PROTONATED_RDKIT_COLS,
                            DELTA_RDKIT_COLS)
from desc_pm7     import (pm7_features_from_record,
                           pm7_features_from_dft_record,
                           NEUTRAL_PM7_COLS,
                           PROTONATED_PM7_COLS,
                           DELTA_PM7_COLS)
from desc_mordred import (mordred_three_states,
                           NEUTRAL_MORDRED_COLS,
                           PROTONATED_MORDRED_COLS,
                           DELTA_MORDRED_COLS,
                           N_MORDRED_2D, N_MORDRED_3D)
from desc_site    import site_features_from_record, SITE_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = _HERE.parent          # scripts/
DATA_DIR   = SCRIPT_DIR.parent.parent / "data"
FEAT_DIR   = DATA_DIR / "features"


# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------

ID_COLS = [
    "record_id", "mol_id", "source", "dataset",
    "neutral_smiles", "protonated_smiles",
    "site_idx", "site_name",
    "mordred_geom_source",
]

LABEL_COLS = [
    "exp_pa_kjmol",  "exp_pa_kcalmol",
    "dft_pa_kjmol",  "dft_pa_kcalmol",
    "pm7_pa_kjmol",  "pm7_pa_kcalmol",
    "delta_dft_exp", "delta_pm7_exp", "dft_correction",
]

FEATURE_COLS = (
    MACCS_COLS                                              #  167
    + MORGAN_COLS                                           # 1024
    + NEUTRAL_RDKIT_COLS                                    #  210
    + PROTONATED_RDKIT_COLS                                 #  210
    + DELTA_RDKIT_COLS                                      #  210
    + NEUTRAL_PM7_COLS                                      #   13
    + PROTONATED_PM7_COLS                                   #   13
    + DELTA_PM7_COLS                                        #   13
    + NEUTRAL_MORDRED_COLS                                  # 1826
    + PROTONATED_MORDRED_COLS                               # 1826
    + DELTA_MORDRED_COLS                                    # 1826
    + SITE_COLS                                             #    6
)                                                           # 7344 total (with 3D)

ALL_COLS = ID_COLS + LABEL_COLS + FEATURE_COLS


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def build_row(
    record: dict,
    site: dict,
    dataset_tag: str,
    use_dft_pm7: bool   = False,
    compute_3d: bool    = True,
    mordred_strategy: str = "all_states",
) -> dict:
    """
    Compute all features for one (molecule, site) pair.

    Parameters
    ----------
    record           : molecule record from dataset.json / pm7_dataset.json
    site             : one element of record['all_sites']
    dataset_tag      : string written to the 'dataset' column
    use_dft_pm7      : True for DFT folder records (maps DFT→PM7 schema,
                       passes DFT coords to Mordred)
    compute_3d       : compute 3D Mordred descriptors
    mordred_strategy : 'all_states' | 'neutral_full_delta_2d'
    """
    neutral  = record["neutral"]
    labels   = record["labels"]
    n_sites  = record["metadata"].get("n_sites", len(record["all_sites"]))
    neu_smi  = neutral.get("smiles", "")
    prot_smi = site.get("protonated_smiles", "")

    # ---- identifiers ----
    row = {
        "record_id":           record["record_id"],
        "mol_id":              record.get("mol_id", record["record_id"]),
        "source":              record["metadata"].get("source", ""),
        "dataset":             dataset_tag,
        "neutral_smiles":      neu_smi,
        "protonated_smiles":   prot_smi,
        "site_idx":            site.get("site_idx"),
        "site_name":           site.get("site_name", ""),
        "mordred_geom_source": None,
    }

    # ---- labels ----
    for col in LABEL_COLS:
        row[col] = labels.get(col)
    if site.get("pa_kjmol") is not None:
        if dataset_tag.startswith("dft"):
            row["dft_pa_kjmol"]   = site["pa_kjmol"]
            row["dft_pa_kcalmol"] = site.get("pa_kcalmol")
        else:
            row["pm7_pa_kjmol"]   = site["pa_kjmol"]
            row["pm7_pa_kcalmol"] = site.get("pa_kcalmol")

    # ---- MACCS (neutral) ----
    maccs = maccs_from_smiles(neu_smi)
    arr   = maccs.astype(float) if maccs is not None else np.full(len(MACCS_COLS), np.nan)
    for col, val in zip(MACCS_COLS, arr):
        row[col] = val

    # ---- Morgan count (neutral) ----
    morgan = morgan_from_smiles(neu_smi)
    arr    = morgan.astype(float) if morgan is not None else np.full(len(MORGAN_COLS), np.nan)
    for col, val in zip(MORGAN_COLS, arr):
        row[col] = val

    # ---- RDKit descriptors — 3 states ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = rdkit_descs_three_states(neu_smi, prot_smi)
    for col, val in zip(NEUTRAL_RDKIT_COLS,    rd["neutral"]):    row[col] = val
    for col, val in zip(PROTONATED_RDKIT_COLS, rd["protonated"]): row[col] = val
    for col, val in zip(DELTA_RDKIT_COLS,      rd["delta"]):      row[col] = val

    # ---- PM7 quantum descriptors — 3 states ----
    pm7_fn = pm7_features_from_dft_record if use_dft_pm7 else pm7_features_from_record
    pm7    = pm7_fn(neutral, site)
    for col, val in zip(NEUTRAL_PM7_COLS,    pm7["neutral"]):    row[col] = val
    for col, val in zip(PROTONATED_PM7_COLS, pm7["protonated"]): row[col] = val
    for col, val in zip(DELTA_PM7_COLS,      pm7["delta"]):      row[col] = val

    # ---- Mordred descriptors ----
    # Pass DFT coords when available (folder records only)
    neu_sym  = neutral.get("opt_coords_symbols")  if use_dft_pm7 else None
    neu_xyz  = neutral.get("opt_coords_angstrom") if use_dft_pm7 else None
    prot_sym = site.get("opt_coords_symbols")     if use_dft_pm7 else None
    prot_xyz = site.get("opt_coords_angstrom")    if use_dft_pm7 else None

    md = mordred_three_states(
        neutral_smiles      = neu_smi,
        protonated_smiles   = prot_smi,
        neutral_dft_symbols = neu_sym,
        neutral_dft_coords  = neu_xyz,
        prot_dft_symbols    = prot_sym,
        prot_dft_coords     = prot_xyz,
        compute_3d          = compute_3d,
        state_strategy      = mordred_strategy,
    )
    row["mordred_geom_source"] = md["geom_source"]
    for col, val in zip(NEUTRAL_MORDRED_COLS,    md["neutral"]):    row[col] = val
    for col, val in zip(PROTONATED_MORDRED_COLS, md["protonated"]): row[col] = val
    for col, val in zip(DELTA_MORDRED_COLS,      md["delta"]):      row[col] = val

    # ---- Site descriptors ----
    sf = site_features_from_record(site, neutral, n_sites)
    for col, val in zip(SITE_COLS, sf):
        row[col] = val

    return row


# ---------------------------------------------------------------------------
# Dataset processor
# ---------------------------------------------------------------------------

def process_dataset(
    records: dict,
    dataset_tag: str,
    use_dft_pm7: bool     = False,
    compute_3d: bool      = True,
    mordred_strategy: str = "all_states",
) -> pd.DataFrame:
    rows, skipped = [], 0

    for record in tqdm(records.values(), desc=dataset_tag, unit="mol"):
        sites = record.get("all_sites", [])
        if not sites:
            skipped += 1
            continue
        for site in sites:
            if site.get("status") == "FAILED":
                skipped += 1
                continue
            try:
                rows.append(build_row(
                    record, site, dataset_tag,
                    use_dft_pm7, compute_3d, mordred_strategy,
                ))
            except Exception as e:
                log.warning(f"  Row failed {record['record_id']}: {e}")
                skipped += 1

    if skipped:
        log.info(f"  {skipped} rows skipped")

    return pd.DataFrame(rows, columns=ALL_COLS)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_dataset(df: pd.DataFrame, name: str):
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    pq  = FEAT_DIR / f"{name}_features.parquet"
    csv = FEAT_DIR / f"{name}_features.csv"
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)
    log.info(f"  {pq.name}  ({len(df)} rows × {len(df.columns)} cols)")
    log.info(f"  {csv.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build ML feature matrices for proton affinity datasets."
    )
    parser.add_argument(
        "--dataset", default="all",
        choices=["all", "nist", "kmeans", "dft"],
        help="Which dataset(s) to process.",
    )
    parser.add_argument(
        "--no-3d", action="store_true",
        help="Skip 3D Mordred descriptors. Faster; 213 3D-only cols become NaN.",
    )
    parser.add_argument(
        "--mordred-states", default="all_states",
        choices=["all_states", "neutral_full_delta_2d"],
        help=(
            "all_states: 3D Mordred for neutral+protonated+delta (default). "
            "neutral_full_delta_2d: 3D for neutral only; protonated/delta use 2D."
        ),
    )
    args = parser.parse_args()
    compute_3d       = not args.no_3d
    mordred_strategy = args.mordred_states

    # Feature count summary for manifest
    n_mordred_3d_active = {
        "all_states":             N_MORDRED_3D * 3,
        "neutral_full_delta_2d":  N_MORDRED_2D * 2 + N_MORDRED_3D,
    }[mordred_strategy] if compute_3d else N_MORDRED_2D * 3

    manifest = {
        "run_config": {
            "compute_3d":        compute_3d,
            "mordred_strategy":  mordred_strategy,
        },
        "n_features": len(FEATURE_COLS),
        "breakdown": {
            "maccs":                 len(MACCS_COLS),
            "morgan":                len(MORGAN_COLS),
            "rdkit_per_state":       210,
            "rdkit_total":           210 * 3,
            "pm7_per_state":         13,
            "pm7_total":             13 * 3,
            "mordred_total_columns": len(NEUTRAL_MORDRED_COLS) * 3,
            "mordred_3d_computed":   n_mordred_3d_active,
            "site":                  len(SITE_COLS),
        },
        "datasets": {},
    }

    # Lazy-load datasets
    _cache: dict = {}

    def load(key, path):
        if key not in _cache:
            _cache[key] = json.loads(Path(path).read_text())
        return _cache[key]

    pm7_path = DATA_DIR / "processed" / "pm7_dataset.json"
    dft_path = DATA_DIR / "processed" / "dataset.json"

    def run(tag, source_filter, json_path, use_dft, out_name):
        log.info(f"Processing {tag} ...")
        data    = load(json_path, json_path)
        records = {k: v for k, v in data.items()
                   if v["metadata"]["source"] == source_filter}
        df = process_dataset(records, tag, use_dft, compute_3d, mordred_strategy)
        save_dataset(df, out_name)
        geom_counts = df["mordred_geom_source"].value_counts().to_dict()
        manifest["datasets"][out_name] = {
            "rows":      len(df),
            "molecules": df["record_id"].nunique(),
            "geom":      geom_counts,
        }
        log.info(f"  Done: {len(df)} rows, geom={geom_counts}")

    if args.dataset in ("all", "nist"):
        run("pm7_nist",    "pm7_nist",    pm7_path, False, "nist1185")

    if args.dataset in ("all", "kmeans"):
        run("pm7_kmeans",  "pm7_kmeans",  pm7_path, False, "kmeans251")

    if args.dataset in ("all", "dft"):
        run("dft_folder",  "folder",      dft_path, True,  "dft251")

    manifest_path = FEAT_DIR / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest → {manifest_path.relative_to(SCRIPT_DIR.parent.parent)}")

    # Summary
    print("\n" + "=" * 64)
    print(f"  Total feature columns : {len(FEATURE_COLS)}")
    print(f"  3D Mordred            : {'yes' if compute_3d else 'no'}"
          f"  strategy={mordred_strategy}")
    print(f"\n  Breakdown:")
    for k, v in manifest["breakdown"].items():
        print(f"    {k:<32} {v:>6}")
    print(f"  {'─' * 40}")
    print(f"  {'Total feature columns':<32} {len(FEATURE_COLS):>6}")
    print(f"\n  Datasets written:")
    for name, info in manifest["datasets"].items():
        print(f"    {name:<16} {info['rows']:>5} rows  "
              f"{info['molecules']:>4} mols  geom={info['geom']}")
    print(f"\n  Output dir: {FEAT_DIR.relative_to(SCRIPT_DIR.parent.parent)}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
