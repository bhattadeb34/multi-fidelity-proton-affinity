"""
Select representative molecules from a chemical space by clustering in latent space.

This script was originally a Colab notebook; it has been converted into a normal
Python CLI module (no notebook magics, no plotting).

Expected input: a CSV with at least:
  - a SMILES column (default: "smiles")
  - 3 latent coordinate columns (default: "latent_1,latent_2,latent_3")
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.cluster import KMeans
from tqdm import tqdm


@dataclass(frozen=True)
class FilterConfig:
    heavy_atoms_min: int = 5
    heavy_atoms_max: int = 25
    mw_min: float = 50.0
    mw_max: float = 250.0
    min_hbd: int = 1
    min_hba: int = 1
    min_protonatable_sites: int = 2
    allow_halogens: bool = False
    allow_charged: bool = False


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def count_protonatable_sites(smiles: str) -> int:
    """Heuristic count of "protonatable sites" from RDKit TPSA contributions."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0

    # RDKit-internal function; kept to preserve notebook behavior.
    tpsa_contribs = Chem.rdMolDescriptors._CalcTPSAContribs(mol)  # pylint: disable=protected-access
    sites = np.argwhere(np.array(tpsa_contribs) > 0).flatten().tolist()

    valid = 0
    for atom_idx in sites:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        if atom.GetSymbol() in {"N", "O", "S", "P"} and atom.GetFormalCharge() == 0:
            valid += 1
    return valid


def _molecule_properties(smiles: str) -> tuple[int, int, int, float, bool, bool, bool, int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (0, 0, 0, 0.0, False, True, True, 0)

    try:
        hbd = int(rdMolDescriptors.CalcNumHBD(mol))
        hba = int(rdMolDescriptors.CalcNumHBA(mol))
        heavy_atoms = int(mol.GetNumHeavyAtoms())
        mw = float(Descriptors.MolWt(mol))

        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        has_heteroatoms = any(x in symbols for x in ("N", "O", "S", "P"))
        has_halogens = any(x in symbols for x in ("F", "Cl", "Br", "I"))
        has_charges = any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())

        protonatable_sites = int(count_protonatable_sites(smiles))
        return (hbd, hba, heavy_atoms, mw, has_heteroatoms, has_halogens, has_charges, protonatable_sites)
    except Exception:
        return (0, 0, 0, 0.0, False, True, True, 0)


def enhanced_molecular_screen(
    df: pd.DataFrame,
    smiles_col: str,
    cfg: FilterConfig,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """Screen molecules with chunked RDKit property calculations for speed."""
    if smiles_col not in df.columns:
        raise KeyError(f"Missing SMILES column {smiles_col!r}. Columns: {list(df.columns)}")

    property_columns = [
        "HBD",
        "HBA",
        "heavy_atoms",
        "MW",
        "has_heteroatoms",
        "has_halogens",
        "has_charges",
        "protonatable_sites",
    ]

    results: list[tuple[int, int, int, float, bool, bool, bool, int]] = []
    total_chunks = (len(df) + chunk_size - 1) // chunk_size

    for i in tqdm(
        range(0, len(df), chunk_size),
        desc=f"RDKit properties (chunks of {chunk_size})",
        total=total_chunks,
    ):
        chunk_end = min(i + chunk_size, len(df))
        smiles_chunk = df[smiles_col].iloc[i:chunk_end].astype(str).tolist()
        results.extend(_molecule_properties(s) for s in smiles_chunk)

    props_df = pd.DataFrame(results, columns=property_columns, index=df.index)
    df = df.copy()
    for col in property_columns:
        df[col] = props_df[col]

    original_len = len(df)

    df = df[df["has_heteroatoms"] == True]  # noqa: E712

    if not cfg.allow_halogens:
        df = df[df["has_halogens"] == False]  # noqa: E712
    if not cfg.allow_charged:
        df = df[df["has_charges"] == False]  # noqa: E712

    df = df[
        (df["heavy_atoms"] >= cfg.heavy_atoms_min)
        & (df["heavy_atoms"] <= cfg.heavy_atoms_max)
        & (df["MW"] >= cfg.mw_min)
        & (df["MW"] <= cfg.mw_max)
    ]

    df = df[(df["HBD"] >= cfg.min_hbd) & (df["HBA"] >= cfg.min_hba)]
    df = df[df["protonatable_sites"] >= cfg.min_protonatable_sites]

    print(
        f"Enhanced filtering: {original_len:,} → {len(df):,} molecules "
        f"({(100.0 * len(df) / original_len) if original_len else 0.0:.2f}%)"
    )
    return df


def simple_molecular_screen(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Legacy simple filter from the notebook."""
    if smiles_col not in df.columns:
        raise KeyError(f"Missing SMILES column {smiles_col!r}. Columns: {list(df.columns)}")
    return df[df[smiles_col].apply(lambda x: "N" in x if pd.notna(x) else False)]


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Present: {list(df.columns)}")


def kmeans_molecular_selection_corrected(
    df: pd.DataFrame,
    *,
    n_dft: int,
    n_pm7_per_dft: int,
    smiles_col: str,
    latent_cols: Sequence[str],
    seed: int,
    kmeans_n_init: int,
    kmeans_max_iter: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    K-means selection:
      1) cluster into `n_dft` clusters in latent space
      2) pick 1 representative per cluster (closest to centroid) => DFT set
      3) within each cluster, sub-cluster to pick up to `n_pm7_per_dft` evenly distributed points => PM7 set
    """
    _require_columns(df, [smiles_col, *latent_cols])

    if n_dft <= 0:
        raise ValueError("--n-dft must be > 0")
    if n_pm7_per_dft <= 0:
        raise ValueError("--n-pm7-per-dft must be > 0")
    if len(df) < n_dft:
        raise ValueError(f"Not enough molecules ({len(df)}) to form {n_dft} clusters after filtering.")

    print("K-means molecular selection")
    print(f"  - Input molecules: {len(df):,}")
    print(f"  - DFT reps (clusters): {n_dft:,}")
    print(f"  - PM7 per DFT: {n_pm7_per_dft:,} (target PM7 total = {n_dft * n_pm7_per_dft:,})")
    print(f"  - Latent columns: {list(latent_cols)}")

    df_sorted = df.sort_values(smiles_col).reset_index().rename(columns={"index": "original_index"})
    coords = df_sorted[list(latent_cols)].to_numpy(dtype=float, copy=False)

    kmeans = KMeans(
        n_clusters=n_dft,
        n_init=kmeans_n_init,
        init="k-means++",
        random_state=seed,
        verbose=0,
        max_iter=kmeans_max_iter,
    )
    cluster_labels = kmeans.fit_predict(coords)
    cluster_centers = kmeans.cluster_centers_

    df_sorted["cluster_id"] = cluster_labels

    cluster_info_df = (
        pd.DataFrame(
            {
                "cluster_id": np.arange(n_dft, dtype=int),
                "centroid_latent_1": cluster_centers[:, 0],
                "centroid_latent_2": cluster_centers[:, 1],
                "centroid_latent_3": cluster_centers[:, 2],
            }
        )
        .assign(
            n_molecules_in_cluster=lambda x: pd.Series(cluster_labels).value_counts().reindex(x["cluster_id"]).fillna(0).astype(int).values
        )
        .reset_index(drop=True)
    )

    # DFT: closest point to each centroid (deterministic tie-break by smallest row index)
    dft_rows: list[int] = []
    dft_cluster_ids: list[int] = []
    dft_dist_to_centroid: list[float] = []

    for cid in tqdm(range(n_dft), desc="Selecting DFT representatives"):
        mask = cluster_labels == cid
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        sub_coords = coords[mask]
        centroid = cluster_centers[cid]
        dists = np.linalg.norm(sub_coords - centroid, axis=1)
        min_dist = float(np.min(dists))
        min_local = np.where(dists == min_dist)[0]
        chosen = int(idxs[min_local[0]])
        dft_rows.append(chosen)
        dft_cluster_ids.append(cid)
        dft_dist_to_centroid.append(min_dist)

    dft_df = df_sorted.iloc[dft_rows].copy()
    dft_df["calculation_type"] = "DFT"
    dft_df["cluster_id"] = dft_cluster_ids
    dft_df["is_cluster_center"] = True
    dft_df["distance_to_centroid"] = dft_dist_to_centroid
    dft_df["dft_representative"] = dft_rows
    dft_df["distance_to_dft"] = 0.0
    dft_df = dft_df.merge(
        cluster_info_df[["cluster_id", "centroid_latent_1", "centroid_latent_2", "centroid_latent_3"]],
        on="cluster_id",
        how="left",
    )

    # PM7: evenly distributed within each cluster via sub-kmeans
    pm7_records: list[dict] = []

    for i, cid in enumerate(tqdm(range(n_dft), desc="Selecting PM7 (sub-cluster per cluster)")):
        mask = cluster_labels == cid
        idxs = np.where(mask)[0]
        sub_coords = coords[mask]
        if len(idxs) == 0:
            continue

        n_pick = min(n_pm7_per_dft, len(idxs))
        if n_pick == 1:
            selected = [int(dft_rows[i])]
        else:
            # if a cluster is too small / degenerate, sklearn may error; fall back to just the DFT rep
            try:
                sub_kmeans = KMeans(
                    n_clusters=n_pick,
                    n_init=kmeans_n_init,
                    init="k-means++",
                    random_state=seed,
                    verbose=0,
                    max_iter=kmeans_max_iter,
                )
                sub_labels = sub_kmeans.fit_predict(sub_coords)
                sub_centers = sub_kmeans.cluster_centers_
                selected = []
                for j in range(n_pick):
                    smask = sub_labels == j
                    if not np.any(smask):
                        continue
                    sc = sub_coords[smask]
                    sidxs = idxs[smask]
                    center = sub_centers[j]
                    dists = np.linalg.norm(sc - center, axis=1)
                    min_dist = float(np.min(dists))
                    min_local = np.where(dists == min_dist)[0]
                    selected.append(int(sidxs[min_local[0]]))
            except Exception:
                selected = [int(dft_rows[i])]

        dft_coord = coords[int(dft_rows[i])]
        for row_idx in selected:
            pm7_coord = coords[int(row_idx)]
            pm7_records.append(
                {
                    "molecule_index": int(row_idx),
                    "cluster_id": int(cid),
                    "dft_representative": int(dft_rows[i]),
                    "distance_to_dft": float(np.linalg.norm(pm7_coord - dft_coord)),
                    "distance_to_centroid": float(np.linalg.norm(pm7_coord - cluster_centers[cid])),
                    "is_cluster_center": bool(int(row_idx) == int(dft_rows[i])),
                }
            )

    pm7_data = pd.DataFrame(pm7_records)
    pm7_df = df_sorted.iloc[pm7_data["molecule_index"].values].copy()
    pm7_df["calculation_type"] = "PM7"
    pm7_df["cluster_id"] = pm7_data["cluster_id"].values
    pm7_df["dft_representative"] = pm7_data["dft_representative"].values
    pm7_df["distance_to_dft"] = pm7_data["distance_to_dft"].values
    pm7_df["distance_to_centroid"] = pm7_data["distance_to_centroid"].values
    pm7_df["is_cluster_center"] = pm7_data["is_cluster_center"].values
    pm7_df = pm7_df.merge(
        cluster_info_df[["cluster_id", "centroid_latent_1", "centroid_latent_2", "centroid_latent_3"]],
        on="cluster_id",
        how="left",
    )

    dft_df = dft_df.reset_index(drop=True)
    pm7_df = pm7_df.reset_index(drop=True)

    print(f"Selected DFT: {len(dft_df):,}")
    print(f"Selected PM7: {len(pm7_df):,} (incl. {pm7_df['is_cluster_center'].sum():,} DFT molecules)")

    return dft_df, pm7_df, cluster_info_df, df_sorted


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cluster chemical space in latent coordinates and select DFT + PM7 subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input chemical space CSV (must include smiles + latent columns).",
    )
    p.add_argument("--smiles-col", default="smiles", help="Name of SMILES column.")
    p.add_argument(
        "--latent-cols",
        default="latent_1,latent_2,latent_3",
        help="Comma-separated latent coordinate columns (must be exactly 3).",
    )

    p.add_argument("--n-dft", type=int, default=256, help="Number of clusters / DFT representatives.")
    p.add_argument("--n-pm7-per-dft", type=int, default=64, help="PM7 molecules per DFT representative.")

    p.add_argument(
        "--filters",
        choices=["enhanced", "simple", "none"],
        default="enhanced",
        help="Which pre-filter to apply before clustering.",
    )
    p.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for RDKit property calculation.")

    p.add_argument("--seed", type=int, default=42, help="Random seed (KMeans + tie-breaking).")
    p.add_argument("--kmeans-n-init", type=int, default=10, help="KMeans n_init.")
    p.add_argument("--kmeans-max-iter", type=int, default=300, help="KMeans max_iter.")

    p.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write outputs (created if missing).",
    )
    p.add_argument(
        "--prefix",
        default="multifidelity",
        help="Prefix for output filenames.",
    )
    p.add_argument(
        "--no-timestamp",
        action="store_true",
        help="If set, do not append a timestamp to output filenames.",
    )
    p.add_argument(
        "--save-filtered",
        action="store_true",
        help="If set, also save the filtered chemical space CSV.",
    )
    return p


def run(args: argparse.Namespace) -> None:
    set_all_seeds(int(args.seed))

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    latent_cols = [c.strip() for c in str(args.latent_cols).split(",") if c.strip()]
    if len(latent_cols) != 3:
        raise ValueError(f"--latent-cols must contain exactly 3 columns, got {latent_cols}")

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    if args.filters == "enhanced":
        filtered_df = enhanced_molecular_screen(
            df,
            smiles_col=args.smiles_col,
            cfg=FilterConfig(),
            chunk_size=int(args.chunk_size),
        )
        filter_tag = "enhanced"
    elif args.filters == "simple":
        filtered_df = simple_molecular_screen(df, smiles_col=args.smiles_col)
        filter_tag = "simple"
        print(f"Simple filtering: {len(df):,} → {len(filtered_df):,} molecules")
    else:
        filtered_df = df
        filter_tag = "none"

    if len(filtered_df) == 0:
        raise RuntimeError("No molecules passed filters; relax filters or use --filters none.")

    stamp = "" if args.no_timestamp else datetime.now().strftime("_%Y%m%d_%H%M%S")
    base = f"{args.prefix}_{filter_tag}{stamp}"

    if args.save_filtered:
        filtered_out = os.path.join(out_dir, f"{base}_filtered_{len(filtered_df)}.csv")
        filtered_df.to_csv(filtered_out, index=False)
        print(f"Wrote: {filtered_out}")

    dft_df, pm7_df, cluster_info_df, tracked_filtered_df = kmeans_molecular_selection_corrected(
        filtered_df,
        n_dft=int(args.n_dft),
        n_pm7_per_dft=int(args.n_pm7_per_dft),
        smiles_col=str(args.smiles_col),
        latent_cols=latent_cols,
        seed=int(args.seed),
        kmeans_n_init=int(args.kmeans_n_init),
        kmeans_max_iter=int(args.kmeans_max_iter),
    )

    clusters_out = os.path.join(out_dir, f"{base}_clusters_{len(cluster_info_df)}.csv")
    dft_out = os.path.join(out_dir, f"{base}_DFT_{len(dft_df)}.csv")
    pm7_out = os.path.join(out_dir, f"{base}_PM7_{len(pm7_df)}.csv")
    tracked_out = os.path.join(out_dir, f"{base}_tracked_filtered_{len(tracked_filtered_df)}.csv")

    cluster_info_df.to_csv(clusters_out, index=False)
    dft_df.to_csv(dft_out, index=False)
    pm7_df.to_csv(pm7_out, index=False)
    tracked_filtered_df.to_csv(tracked_out, index=False)

    combined_df = pd.concat([dft_df, pm7_df[~pm7_df["is_cluster_center"]]], ignore_index=True)
    combined_out = os.path.join(out_dir, f"{base}_hierarchical_{len(combined_df)}.csv")
    combined_df.to_csv(combined_out, index=False)

    print("Wrote outputs:")
    print(f"  - {clusters_out}")
    print(f"  - {dft_out}")
    print(f"  - {pm7_out}")
    print(f"  - {combined_out}")
    print(f"  - {tracked_out}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()