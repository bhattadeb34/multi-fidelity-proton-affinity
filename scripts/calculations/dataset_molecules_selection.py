"""
dataset_molecules_selection.py
===============================
Select DFT and PM7 representative molecules from a pre-filtered ZINC library
by k-means clustering in 3D PCA latent space.

The expected input CSV contains at least:
    smiles, latent_1, latent_2, latent_3

If the CSV already contains filter columns (HBD, HBA, heavy_atoms, MW,
has_heteroatoms, has_halogens, has_charges, protonatable_sites), the
molecular filter step is skipped by default. Pass --apply-filter to re-run
filtering from an unfiltered ZINC CSV.

Outputs (written to --output-dir):
    filtered_chemical_space_<N>molecules.csv   (if filtering applied)
    DFT_<N>molecules.csv
    PM7_<N>molecules.csv
    hierarchical_<N>molecules.csv              (DFT + PM7-only combined)
    clusters_256.csv

Usage:
    python scripts/calculations/dataset_molecules_selection.py \
        --input data/screening/zinc_raw/filtered_821k.csv \
        --output-dir data/screening/kmeans_selection/ \
        --n-dft 256 --n-pm7-per-dft 64

    # Re-run filtering from raw ZINC CSV:
    python scripts/calculations/dataset_molecules_selection.py \
        --input raw_zinc.csv --apply-filter

    # Also generate latent space plot:
    python scripts/calculations/dataset_molecules_selection.py \
        --input data/screening/zinc_raw/filtered_821k.csv --plot
"""

from __future__ import annotations

import argparse
import os
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.cluster import KMeans
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Molecular filtering
# ---------------------------------------------------------------------------

def count_protonatable_sites(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    tpsa = Chem.rdMolDescriptors._CalcTPSAContribs(mol)
    sites = np.argwhere(np.array(tpsa) > 0).flatten().tolist()
    valid = 0
    for idx in sites:
        atom = mol.GetAtomWithIdx(int(idx))
        if atom.GetSymbol() in {"N", "O", "S", "P"} and atom.GetFormalCharge() == 0:
            valid += 1
    return valid


def _mol_properties(smiles: str) -> list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0, 0, 0, 0.0, False, True, True, 0]
    try:
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        heavy = mol.GetNumHeavyAtoms()
        mw = Descriptors.MolWt(mol)
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        has_het = any(s in symbols for s in ("N", "O", "S", "P"))
        has_hal = any(s in symbols for s in ("F", "Cl", "Br", "I"))
        has_chg = any(a.GetFormalCharge() != 0 for a in mol.GetAtoms())
        prot = count_protonatable_sites(smiles)
        return [hbd, hba, heavy, mw, has_het, has_hal, has_chg, prot]
    except Exception:
        return [0, 0, 0, 0.0, False, True, True, 0]


def enhanced_molecular_screen(df: pd.DataFrame, smiles_col: str = "smiles",
                               chunk_size: int = 5000) -> pd.DataFrame:
    prop_cols = ["HBD", "HBA", "heavy_atoms", "MW", "has_heteroatoms",
                 "has_halogens", "has_charges", "protonatable_sites"]
    all_props = []
    for i in tqdm(range(0, len(df), chunk_size), desc="Computing properties"):
        chunk = df[smiles_col].iloc[i:i + chunk_size].tolist()
        all_props.extend(_mol_properties(s) for s in chunk)

    props_df = pd.DataFrame(all_props, columns=prop_cols, index=df.index)
    for col in prop_cols:
        df[col] = props_df[col]

    original = len(df)
    df = df[df["has_heteroatoms"]]
    df = df[~df["has_halogens"] & ~df["has_charges"]]
    df = df[(df["heavy_atoms"] >= 5) & (df["heavy_atoms"] <= 25)
            & (df["MW"] >= 50) & (df["MW"] <= 250)]
    df = df[(df["HBD"] >= 1) & (df["HBA"] >= 1)]
    df = df[df["protonatable_sites"] >= 2]
    print(f"Filtered: {original:,} -> {len(df):,} molecules")
    return df


# ---------------------------------------------------------------------------
# K-means selection
# ---------------------------------------------------------------------------

def kmeans_molecular_selection(df: pd.DataFrame, n_dft: int = 256,
                                n_pm7_per_dft: int = 64,
                                smiles_col: str = "smiles"):
    latent_cols = ["latent_1", "latent_2", "latent_3"]
    df_sorted = df.sort_values(smiles_col).reset_index().rename(
        columns={"index": "original_index"})
    coords = df_sorted[latent_cols].values

    print(f"K-means clustering: {len(df_sorted):,} molecules -> {n_dft} clusters")
    km = KMeans(n_clusters=n_dft, n_init=10, init="k-means++",
                random_state=RANDOM_SEED, max_iter=300, verbose=0)
    labels = km.fit_predict(coords)
    centers = km.cluster_centers_
    print(f"  Converged in {km.n_iter_} iterations (inertia={km.inertia_:.2f})")

    df_sorted["cluster_id"] = labels

    cluster_info = [
        {"cluster_id": i,
         "centroid_latent_1": centers[i][0],
         "centroid_latent_2": centers[i][1],
         "centroid_latent_3": centers[i][2],
         "n_molecules_in_cluster": int(np.sum(labels == i))}
        for i in range(n_dft)
    ]
    cluster_df = pd.DataFrame(cluster_info)

    # Select DFT molecules: closest to each centroid
    dft_indices, dft_cluster_ids, dft_dists = [], [], []
    for i in tqdm(range(n_dft), desc="Selecting DFT molecules"):
        mask = labels == i
        if not np.any(mask):
            continue
        ci = np.where(mask)[0]
        d = np.linalg.norm(coords[ci] - centers[i], axis=1)
        min_d = d.min()
        tied = ci[d == min_d]
        chosen = int(tied[0])
        dft_indices.append(chosen)
        dft_cluster_ids.append(i)
        dft_dists.append(float(min_d))

    dft_df = df_sorted.iloc[dft_indices].copy()
    dft_df["calculation_type"] = "DFT"
    dft_df["cluster_id"] = dft_cluster_ids
    dft_df["is_cluster_center"] = True
    dft_df["distance_to_centroid"] = dft_dists
    dft_df["distance_to_dft"] = 0.0
    dft_df = dft_df.merge(
        cluster_df[["cluster_id", "centroid_latent_1", "centroid_latent_2",
                    "centroid_latent_3"]],
        on="cluster_id", how="left")

    # Select PM7 molecules: sub-cluster within each main cluster
    pm7_rows = []
    for i, cid in enumerate(tqdm(range(n_dft), desc="Sub-clustering for PM7")):
        mask = labels == cid
        ci = np.where(mask)[0]
        cc = coords[ci]
        if len(ci) < 2:
            continue
        n_sub = min(n_pm7_per_dft, len(ci))
        if n_sub <= 1:
            selected = [dft_indices[i]]
        else:
            sub_km = KMeans(n_clusters=n_sub, n_init=10, init="k-means++",
                            random_state=RANDOM_SEED, max_iter=300, verbose=0)
            sub_labels = sub_km.fit_predict(cc)
            sub_centers = sub_km.cluster_centers_
            selected = []
            for j in range(n_sub):
                sub_mask = sub_labels == j
                if not np.any(sub_mask):
                    continue
                sci = ci[sub_mask]
                scc = cc[sub_mask]
                d = np.linalg.norm(scc - sub_centers[j], axis=1)
                min_d = d.min()
                tied = sci[d == min_d]
                selected.append(int(tied[0]))

        dft_coord = coords[dft_indices[i]]
        for idx in selected:
            pm7_rows.append({
                "molecule_index": idx,
                "cluster_id": cid,
                "dft_representative": dft_indices[i],
                "distance_to_dft": float(np.linalg.norm(coords[idx] - dft_coord)),
                "distance_to_centroid": float(np.linalg.norm(coords[idx] - centers[cid])),
                "is_cluster_center": idx == dft_indices[i],
            })

    pm7_meta = pd.DataFrame(pm7_rows)
    pm7_df = df_sorted.iloc[pm7_meta["molecule_index"]].copy()
    pm7_df["calculation_type"] = "PM7"
    for col in ["cluster_id", "dft_representative", "distance_to_dft",
                "distance_to_centroid", "is_cluster_center"]:
        pm7_df[col] = pm7_meta[col].values
    pm7_df = pm7_df.merge(
        cluster_df[["cluster_id", "centroid_latent_1", "centroid_latent_2",
                    "centroid_latent_3"]],
        on="cluster_id", how="left")

    dft_df = dft_df.reset_index(drop=True)
    pm7_df = pm7_df.reset_index(drop=True)
    return dft_df, pm7_df, cluster_df


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------

def _distinct_colors(n: int) -> list:
    from matplotlib.colors import hsv_to_rgb
    colors = [hsv_to_rgb((i / n, 0.75 + (i % 5) * 0.05, 0.9 - (i % 7) * 0.04))
              for i in range(n)]
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(colors)
    return colors


def plot_latent_space(background_csv: str, dft_df: pd.DataFrame,
                      pm7_df: pd.DataFrame, save_path: str | None = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    bg = pd.read_csv(background_csv)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor="white")
    sns.set_style("whitegrid")

    unique_clusters = sorted(pm7_df["cluster_id"].unique())
    colors = {cid: c for cid, c in zip(unique_clusters,
                                        _distinct_colors(len(unique_clusters)))}

    pairs = [("latent_2", "latent_1"), ("latent_3", "latent_2"),
             ("latent_3", "latent_1")]
    for ax, (xc, yc) in zip(axes, pairs):
        ax.hexbin(bg[xc], bg[yc], gridsize=50, cmap="Greys", alpha=0.8, mincnt=1)
        for cid, color in colors.items():
            sub_pm7 = pm7_df[pm7_df["cluster_id"] == cid]
            sub_dft = dft_df[dft_df["cluster_id"] == cid]
            ax.scatter(sub_pm7[xc], sub_pm7[yc], s=50, color=color,
                       edgecolors="white", linewidths=0.5, alpha=0.85, zorder=4)
            ax.scatter(sub_dft[xc], sub_dft[yc], s=70, color=color,
                       edgecolors="black", linewidths=1.5, marker="s", zorder=5)
        ax.set_xlabel(xc.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(yc.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"{xc} vs {yc}", fontsize=13, fontweight="bold")
        ax.set_facecolor("#F8F8F8")

    legend = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=12, label=f"DFT (n={len(dft_df)})",
               linestyle="None"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=10, label=f"PM7 (n={len(pm7_df)})", linestyle="None"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.08),
               ncol=2, fontsize=12, frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/screening/zinc_raw/filtered_821k.csv",
                        help="Input CSV with smiles + latent_1/2/3 columns")
    parser.add_argument("--output-dir", default="data/screening/kmeans_selection/",
                        help="Directory for output CSVs")
    parser.add_argument("--n-dft", type=int, default=256,
                        help="Number of DFT cluster representatives")
    parser.add_argument("--n-pm7-per-dft", type=int, default=64,
                        help="PM7 molecules selected per DFT cluster via sub-clustering")
    parser.add_argument("--apply-filter", action="store_true",
                        help="Apply molecular filter before clustering "
                             "(use when input is unfiltered ZINC CSV)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate latent space visualization after selection")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} molecules loaded")

    for col in ("latent_1", "latent_2", "latent_3"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input CSV")

    if args.apply_filter:
        df = enhanced_molecular_screen(df)
        filtered_path = out_dir / f"filtered_chemical_space_{len(df)}molecules.csv"
        df.to_csv(filtered_path, index=False)
        print(f"Filtered CSV saved: {filtered_path}")

    dft_df, pm7_df, cluster_df = kmeans_molecular_selection(
        df, n_dft=args.n_dft, n_pm7_per_dft=args.n_pm7_per_dft)

    combined = pd.concat([dft_df, pm7_df[~pm7_df["is_cluster_center"]]],
                         ignore_index=True)

    dft_df.to_csv(out_dir / f"DFT_{len(dft_df)}molecules.csv", index=False)
    pm7_df.to_csv(out_dir / f"PM7_{len(pm7_df)}molecules.csv", index=False)
    combined.to_csv(out_dir / f"hierarchical_{len(combined)}molecules.csv", index=False)
    cluster_df.to_csv(out_dir / f"clusters_{args.n_dft}.csv", index=False)

    print(f"\nSummary")
    print(f"  Input molecules  : {len(df):,}")
    print(f"  Clusters         : {args.n_dft}")
    print(f"  DFT selected     : {len(dft_df)}")
    print(f"  PM7 selected     : {len(pm7_df)} ({len(pm7_df[~pm7_df['is_cluster_center']])} PM7-only)")
    print(f"  Outputs in       : {out_dir}")

    if args.plot:
        plot_latent_space(
            background_csv=args.input,
            dft_df=dft_df,
            pm7_df=pm7_df,
            save_path=str(out_dir / "chemical_space_cluster_analysis.png"),
        )


if __name__ == "__main__":
    main()
