"""
07_pareto_select.py
===================
Multi-objective Pareto front selection from LLM-verified candidates.

Objectives (all minimized after transformation):
  1. PA proximity  : |PA_pred - 222.5| (target center = 222.5 kcal/mol)
  2. Uncertainty   : delta_std (lower = more reliable prediction)
  3. SA score      : synthetic accessibility (lower = easier to make)
  4. Diversity     : negative min Tanimoto to already-selected molecules
                     (lower = more diverse from selected set)

Selection procedure:
  1. Filter: accepted verdict + PA in 210-235 window
  2. Compute Pareto front iteratively (Pareto + diversity)
  3. Select top 30 candidates for DFT validation

Reads from:
    data/screening/iter{N}/llm_verdicts.parquet
    data/screening/processed/zinc_fingerprints.npy
    data/screening/processed/zinc_fp_keys.npy

Writes to:
    data/screening/iter{N}/pareto_selected.csv
    data/screening/iter{N}/pareto_report.json
    screening/figures/iter{N}_pareto.pdf

Usage:
    python screening/scripts/07_pareto_select.py --iter 1
    python screening/scripts/07_pareto_select.py --iter 1 --n-select 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
DATA_DIR   = PROJECT / "data" / "screening"

PA_TARGET_LOW  = 210.0
PA_TARGET_HIGH = 235.0
PA_TARGET_MID  = (PA_TARGET_LOW + PA_TARGET_HIGH) / 2  # 222.5


# ---------------------------------------------------------------------------
# Fingerprint utilities
# ---------------------------------------------------------------------------

def smiles_to_fp(smiles: str, fp_keys: np.ndarray) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(fp_keys), dtype=np.float32)
    fp = AllChem.GetMorganFingerprint(mol, radius=2)
    counts = fp.GetNonzeroElements()
    vec = np.zeros(len(fp_keys), dtype=np.float32)
    key_to_col = {int(k): i for i, k in enumerate(fp_keys)}
    for k, v in counts.items():
        col = key_to_col.get(int(k))
        if col is not None:
            vec[col] = float(v)
    return vec


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    min_sum = np.minimum(a, b).sum()
    max_sum = np.maximum(a, b).sum()
    return float(min_sum / max_sum) if max_sum > 0 else 0.0


def max_tanimoto_to_set(fp: np.ndarray, selected_fps: list[np.ndarray]) -> float:
    if not selected_fps:
        return 0.0
    return max(tanimoto(fp, s) for s in selected_fps)


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------

def is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a is dominated by b (b is at least as good in all, better in one)."""
    return bool(np.all(b <= a) and np.any(b < a))


def pareto_front(costs: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of Pareto-optimal solutions.
    costs: (N, M) array where lower is better for all objectives.
    """
    n = costs.shape[0]
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not on_front[i]:
            continue
        for j in range(n):
            if i == j or not on_front[j]:
                continue
            if is_dominated(costs[i], costs[j]):
                on_front[i] = False
                break
    return on_front


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(iteration: int, n_select: int = 30) -> None:
    iter_dir   = DATA_DIR / f"iter{iteration}"
    verdict_path = iter_dir / "llm_verdicts.parquet"
    out_csv    = iter_dir / "pareto_selected.csv"
    out_report = iter_dir / "pareto_report.json"
    fig_dir    = SCREENING / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not verdict_path.exists():
        log.error(f"LLM verdicts not found: {verdict_path}")
        sys.exit(1)

    # ── Load and filter ──────────────────────────────────────────────────────
    df = pd.read_parquet(verdict_path)
    log.info(f"Loaded {len(df):,} molecules")

    # Filter to accepted + in PA window
    mask = (
        (df["final_verdict"] == "accept") &
        (df["pa_pred_kcalmol"] >= PA_TARGET_LOW) &
        (df["pa_pred_kcalmol"] <= PA_TARGET_HIGH)
    )
    pool = df[mask].copy().reset_index(drop=True)
    log.info(f"Pool after accept + PA filter: {len(pool):,} molecules")

    if len(pool) == 0:
        log.error("No candidates passed filters — check PA window and LLM verdicts")
        sys.exit(1)

    # Load Morgan FP keys for diversity computation
    fp_keys = np.load(DATA_DIR / "processed" / "zinc_fp_keys.npy")

    # Compute fingerprints for all pool molecules
    log.info("Computing fingerprints for diversity ...")
    pool_fps = np.array([
        smiles_to_fp(smi, fp_keys) for smi in pool["smiles"]
    ], dtype=np.float32)

    # ── Iterative Pareto + diversity selection ───────────────────────────────
    log.info(f"Selecting {n_select} candidates via Pareto + diversity ...")

    selected_indices = []
    selected_fps     = []
    remaining        = list(range(len(pool)))

    for round_idx in range(n_select):
        if not remaining:
            break

        # Build cost matrix for remaining candidates
        # Objective 1: PA proximity (|PA_pred - 222.5|)
        pa_prox = np.abs(pool.iloc[remaining]["pa_pred_kcalmol"].values - PA_TARGET_MID)

        # Objective 2: Uncertainty
        uncertainty = pool.iloc[remaining]["uncertainty"].values

        # Objective 3: SA score
        sa = pool.iloc[remaining]["sa_score"].fillna(5.0).values

        # Objective 4: Diversity (max Tanimoto to selected set — we want LOW similarity)
        if selected_fps:
            diversity_penalty = np.array([
                max_tanimoto_to_set(pool_fps[i], selected_fps)
                for i in remaining
            ])
        else:
            diversity_penalty = np.zeros(len(remaining))

        # Normalize each objective to [0, 1]
        def normalize(x):
            r = x.max() - x.min()
            return (x - x.min()) / r if r > 0 else np.zeros_like(x)

        costs = np.column_stack([
            normalize(pa_prox),
            normalize(uncertainty),
            normalize(sa),
            normalize(diversity_penalty),
        ])

        # Find Pareto front
        front_mask = pareto_front(costs)
        front_local_idx = np.where(front_mask)[0]

        # Among Pareto front, pick the one with lowest weighted sum
        # Weights: PA proximity most important, then diversity, then uncertainty, then SA
        weights = np.array([0.4, 0.2, 0.1, 0.3])
        weighted = (costs[front_local_idx] * weights).sum(axis=1)
        best_local = front_local_idx[np.argmin(weighted)]
        best_global = remaining[best_local]

        selected_indices.append(best_global)
        selected_fps.append(pool_fps[best_global])
        remaining.remove(best_global)

        if (round_idx + 1) % 5 == 0:
            log.info(f"  Selected {round_idx + 1}/{n_select} ...")

    # ── Build output dataframe ───────────────────────────────────────────────
    selected_df = pool.iloc[selected_indices].copy().reset_index(drop=True)
    selected_df["selection_rank"] = range(1, len(selected_df) + 1)

    # Add pairwise diversity stats
    selected_df["max_tanimoto_to_others"] = [
        max_tanimoto_to_set(selected_fps[i],
                           [selected_fps[j] for j in range(len(selected_fps)) if j != i])
        for i in range(len(selected_fps))
    ]

    selected_df.to_csv(out_csv, index=False)
    log.info(f"Saved {len(selected_df)} selected candidates → {out_csv}")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info(f"\n=== Pareto Selection Summary — Iteration {iteration} ===")
    log.info(f"  Pool size:          {len(pool):,}")
    log.info(f"  Selected:           {len(selected_df)}")
    log.info(f"  PA_pred range:      "
             f"{selected_df['pa_pred_kcalmol'].min():.1f} – "
             f"{selected_df['pa_pred_kcalmol'].max():.1f} kcal/mol")
    log.info(f"  PA_pred mean:       {selected_df['pa_pred_kcalmol'].mean():.1f} kcal/mol")
    log.info(f"  Uncertainty mean:   {selected_df['uncertainty'].mean():.1f} kcal/mol")
    log.info(f"  SA score mean:      {selected_df['sa_score'].mean():.2f}")
    log.info(f"  Max internal Tan.:  {selected_df['max_tanimoto_to_others'].max():.3f}")
    log.info(f"  Mean internal Tan.: {selected_df['max_tanimoto_to_others'].mean():.3f}")

    log.info(f"\n  Selected molecules:")
    for _, row in selected_df.iterrows():
        log.info(f"    [{row['selection_rank']:2d}] PA={row['pa_pred_kcalmol']:.1f} "
                 f"unc={row['uncertainty']:.1f} SA={row['sa_score']:.2f} "
                 f"MW={row['MW']:.0f}  {row['smiles']}")

    # ── Save report ──────────────────────────────────────────────────────────
    report = {
        "iteration":       iteration,
        "pool_size":       len(pool),
        "n_selected":      len(selected_df),
        "pa_target":       [PA_TARGET_LOW, PA_TARGET_HIGH],
        "pa_pred_mean":    float(selected_df["pa_pred_kcalmol"].mean()),
        "pa_pred_std":     float(selected_df["pa_pred_kcalmol"].std()),
        "uncertainty_mean":float(selected_df["uncertainty"].mean()),
        "sa_mean":         float(selected_df["sa_score"].mean()),
        "max_internal_tanimoto": float(selected_df["max_tanimoto_to_others"].max()),
        "smiles_list":     selected_df["smiles"].tolist(),
    }
    out_report.write_text(json.dumps(report, indent=2))
    log.info(f"Saved report → {out_report}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        # Panel 1: PA distribution
        ax = axes[0]
        ax.hist(pool["pa_pred_kcalmol"], bins=40, color="steelblue",
                alpha=0.6, label="Pool (accepted)")
        ax.hist(selected_df["pa_pred_kcalmol"], bins=20, color="tomato",
                alpha=0.8, label="Selected")
        ax.axvline(PA_TARGET_LOW, color="k", ls="--", lw=1)
        ax.axvline(PA_TARGET_HIGH, color="k", ls="--", lw=1)
        ax.set_xlabel("PA_pred (kcal/mol)")
        ax.set_ylabel("Count")
        ax.set_title("PA Distribution")
        ax.legend()

        # Panel 2: PA vs Uncertainty
        ax = axes[1]
        sc = ax.scatter(pool["pa_pred_kcalmol"], pool["uncertainty"],
                       c="steelblue", alpha=0.3, s=10, label="Pool")
        ax.scatter(selected_df["pa_pred_kcalmol"], selected_df["uncertainty"],
                  c="tomato", s=60, zorder=5, label="Selected")
        ax.axvline(PA_TARGET_LOW, color="k", ls="--", lw=1)
        ax.axvline(PA_TARGET_HIGH, color="k", ls="--", lw=1)
        ax.set_xlabel("PA_pred (kcal/mol)")
        ax.set_ylabel("Uncertainty (kcal/mol)")
        ax.set_title("PA vs Uncertainty")
        ax.legend()

        # Panel 3: PA vs SA score
        ax = axes[2]
        ax.scatter(pool["pa_pred_kcalmol"], pool["sa_score"],
                  c="steelblue", alpha=0.3, s=10, label="Pool")
        ax.scatter(selected_df["pa_pred_kcalmol"], selected_df["sa_score"],
                  c="tomato", s=60, zorder=5, label="Selected")
        ax.axvline(PA_TARGET_LOW, color="k", ls="--", lw=1)
        ax.axvline(PA_TARGET_HIGH, color="k", ls="--", lw=1)
        ax.set_xlabel("PA_pred (kcal/mol)")
        ax.set_ylabel("SA Score")
        ax.set_title("PA vs Synthetic Accessibility")
        ax.legend()

        plt.tight_layout()
        fig_path = fig_dir / f"iter{iteration}_pareto.pdf"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved figure → {fig_path}")

    except Exception as e:
        log.warning(f"Plotting failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pareto selection of screening candidates")
    parser.add_argument("--iter",     type=int, default=1)
    parser.add_argument("--n-select", type=int, default=30,
                        help="Number of candidates to select for DFT")
    args = parser.parse_args()
    main(iteration=args.iter, n_select=args.n_select)
