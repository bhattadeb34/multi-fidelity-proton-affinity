"""
llm_correction_benchmark.py
============================
Benchmark: Can an LLM predict PM7 correction as accurately as ExtraTrees?

Task
----
Given SMILES + PM7 PA, predict the signed correction:
  - NIST dataset:    correction = PA_exp  − PA_PM7  (experimental reference)
  - k-means dataset: correction = PA_DFT  − PA_PM7  (DFT reference)

Three LLM approaches evaluated on the same 5-fold CV splits as ExtraTrees:
  1. Zero-shot:     SMILES + PM7 PA → LLM → predicted correction
  2. Few-shot:      10 random training examples + query → LLM → prediction
  3. RAG few-shot:  5 most similar training examples (Tanimoto) + query → LLM

Comparison table (MAE kcal/mol):
  Raw PM7         |  8.21  |  17.96
  ExtraTrees      |  2.87  |   6.85   ← target to beat
  LLM zero-shot   |   ?    |    ?
  LLM few-shot    |   ?    |    ?
  LLM RAG         |   ?    |    ?

Usage
-----
  cd /noether/s0/dxb5775/proton-affinity-paper
  python "language modeling/llm_correction_benchmark.py" --dataset nist --approach all
  python "language modeling/llm_correction_benchmark.py" --dataset nist --approach rag --n-test 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

KJMOL_TO_KCAL = 1 / 4.184
API_KEYS_FILE  = SCRIPT_DIR / "api_keys.txt"
RESULTS_DIR    = SCRIPT_DIR / "benchmark_results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(dataset: str) -> pd.DataFrame:
    """Load NIST or k-means dataset with SMILES, PM7 PA, and true correction."""
    data_dir = PROJECT_DIR / "data"

    if dataset == "nist":
        # nist1155_ml.parquet is already site-level (one best site per molecule)
        # It contains: neutral_smiles, protonated_smiles, site_idx, site_name,
        #              exp_pa_kjmol, exp_pa_kcalmol, pm7_best_pa_kjmol, pm7_best_pa_kcalmol
        df = pd.read_parquet(data_dir / "targets" / "nist1155_ml.parquet")

        # PA columns
        pm7_col = "pm7_best_pa_kcalmol" if "pm7_best_pa_kcalmol" in df.columns else                   next((c for c in df.columns if "pm7" in c and "pa" in c and "kcal" in c), None)
        exp_col = "exp_pa_kcalmol" if "exp_pa_kcalmol" in df.columns else                   next((c for c in df.columns if "exp" in c and "pa" in c and "kcal" in c), None)

        log.info(f"  PM7 col: {pm7_col}, Exp col: {exp_col}")

        mol_df = df[["neutral_smiles", "protonated_smiles",
                     pm7_col, exp_col]].copy()
        mol_df = mol_df.rename(columns={
            "neutral_smiles":   "smiles",
            pm7_col:            "pm7_pa_kcalmol",
            exp_col:            "exp_pa_kcalmol",
        })
        mol_df["correction_kcalmol"] = mol_df["exp_pa_kcalmol"] - mol_df["pm7_pa_kcalmol"]
        mol_df = mol_df.dropna(subset=["pm7_pa_kcalmol", "exp_pa_kcalmol"])

        # Load ALL protonation sites from pm7_source_raw for richer prompts
        pm7_sites_path = data_dir / "pm7_source_raw" / "pm7_nist1185_pa_per_site.csv"
        if pm7_sites_path.exists():
            log.info("  Loading all protonation sites from pm7_source_raw ...")
            sites_df = pd.read_csv(pm7_sites_path)
            # Build lookup: neutral_smiles -> list of {protonated_smiles, pm7_pa}
            sites_lookup = {}
            for _, row in sites_df.iterrows():
                nsmi = row["neutral_smiles"]
                if nsmi not in sites_lookup:
                    sites_lookup[nsmi] = []
                sites_lookup[nsmi].append({
                    "protonated_smiles": row["protonated_smiles"],
                    "pm7_pa": float(row["proton_affinity_kcal_mol"]),
                })
            mol_df["all_sites"] = mol_df["smiles"].map(
                lambda s: sorted(sites_lookup.get(s, []),
                                 key=lambda x: x["pm7_pa"], reverse=True))
            n_with_sites = mol_df["all_sites"].apply(len).gt(0).sum()
            log.info(f"  All sites loaded for {n_with_sites}/{len(mol_df)} molecules")
        else:
            log.warning(f"  pm7_nist1185_pa_per_site.csv not found — using best site only")
            mol_df["all_sites"] = mol_df.apply(
                lambda r: [{"protonated_smiles": r["protonated_smiles"],
                            "pm7_pa": r["pm7_pa_kcalmol"]}], axis=1)

        log.info(f"  NIST: {len(mol_df)} molecules (one best site per molecule)")
        log.info(f"  PM7 PA: {mol_df['pm7_pa_kcalmol'].mean():.1f} ± "
                 f"{mol_df['pm7_pa_kcalmol'].std():.1f} kcal/mol")
        log.info(f"  Correction: {mol_df['correction_kcalmol'].mean():.1f} ± "
                 f"{mol_df['correction_kcalmol'].std():.1f} kcal/mol")
        log.info(f"  Raw PM7 MAE: {mol_df['correction_kcalmol'].abs().mean():.2f} kcal/mol")
        return mol_df
    else:  # kmeans — SITE LEVEL (821 sites from 251 molecules)
        df = pd.read_parquet(data_dir / "targets" / "kmeans251_ml.parquet")

        pm7_cols   = [c for c in df.columns if "pm7" in c.lower() and "pa" in c.lower()]
        dft_cols   = [c for c in df.columns if "dft" in c.lower() and "pa" in c.lower()]
        delta_cols = [c for c in df.columns if "delta" in c.lower()]

        log.info(f"PM7 cols: {pm7_cols}")
        log.info(f"DFT cols: {dft_cols}")
        log.info(f"Delta cols: {delta_cols}")

        # Per-site PM7 PA (not best-site) — site-level prediction
        pm7_col = next((c for c in pm7_cols if "best" not in c),
                       pm7_cols[0] if pm7_cols else None)
        dft_col = dft_cols[0] if dft_cols else None

        keep = ["neutral_smiles", pm7_col, dft_col]
        if "protonated_smiles" in df.columns:
            keep.insert(1, "protonated_smiles")
        mol_df = df[keep].copy()

        mol_df = mol_df.rename(columns={
            "neutral_smiles": "smiles",
            pm7_col: "pm7_pa_kjmol",
            dft_col: "dft_pa_kjmol",
        })
        mol_df["pm7_pa_kcalmol"]     = mol_df["pm7_pa_kjmol"]  * KJMOL_TO_KCAL
        mol_df["exp_pa_kcalmol"]     = mol_df["dft_pa_kjmol"]  * KJMOL_TO_KCAL
        mol_df["correction_kcalmol"] = mol_df["exp_pa_kcalmol"] - mol_df["pm7_pa_kcalmol"]
        mol_df = mol_df.dropna(subset=["pm7_pa_kcalmol","exp_pa_kcalmol"])
        log.info(f"  Site-level: {len(mol_df)} rows from "
                 f"{mol_df['smiles'].nunique()} molecules")

    mol_df = mol_df.dropna(subset=["correction_kcalmol"])
    log.info(f"Loaded {len(mol_df)} molecules")
    log.info(f"  PM7 PA:     {mol_df['pm7_pa_kcalmol'].mean():.1f} ± {mol_df['pm7_pa_kcalmol'].std():.1f} kcal/mol")
    log.info(f"  Correction: {mol_df['correction_kcalmol'].mean():.1f} ± {mol_df['correction_kcalmol'].std():.1f} kcal/mol")
    log.info(f"  Raw PM7 MAE: {mol_df['correction_kcalmol'].abs().mean():.2f} kcal/mol")

    return mol_df


# ---------------------------------------------------------------------------
# Tanimoto similarity for RAG retrieval
# ---------------------------------------------------------------------------

def compute_morgan_fps(smiles_list: list[str]) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    fps = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024),
                              dtype=np.float32)
            else:
                fp = np.zeros(1024, dtype=np.float32)
        except Exception:
            fp = np.zeros(1024, dtype=np.float32)
        fps.append(fp)
    return np.array(fps)


def tanimoto_matrix(fps_query: np.ndarray, fps_ref: np.ndarray) -> np.ndarray:
    """Compute Tanimoto similarity between each query and all ref FPs."""
    intersection = fps_query @ fps_ref.T
    sum_q = fps_query.sum(axis=1, keepdims=True)
    sum_r = fps_ref.sum(axis=1, keepdims=True)
    union = sum_q + sum_r.T - intersection
    return np.where(union > 0, intersection / union, 0.0)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_NIST = """Background — PM7 semi-empirical proton affinity:
  Proton affinity (PA) = negative enthalpy of gas-phase protonation.
  PA = H(B) + H(H+) - H(BH+), where H(H+) = 5/2 RT = 6.197 kJ/mol at 298.15 K

  PM7 calculation (MOPAC):
    PA_PM7 (kcal/mol) = HOF_neutral - HOF_protonated + 365.7
    HOF = heat of formation output from MOPAC PM7 (kcal/mol)
    365.7 kcal/mol = enthalpy of a proton (H+) at 298K in MOPAC convention
    All N, O, S, P atoms with formal charge = 0 are enumerated as protonation sites.
    Best site = site with highest PA_PM7 (most thermodynamically favorable).
    The PM7 PA reported is for the best protonation site.

Background — experimental reference:
    PA_exp = gas-phase experimental proton affinity from NIST WebBook (kcal/mol)
    N = 1155 molecules

  Correction to predict:
    correction = PA_exp - PA_PM7
    Positive: PM7 underestimates PA (molecule is more basic than PM7 predicts)
    Negative: PM7 overestimates PA (molecule is less basic than PM7 predicts)

Your task: given a molecule's neutral SMILES, all its protonated forms with their
PM7 PA values, predict the signed correction (kcal/mol) for the best protonation
site needed to match the experimental PA.

Respond with ONLY a single floating-point number (kcal/mol). No units, no explanation.
Example valid responses: 6.3   or   -2.1   or   9.45"""

SYSTEM_PROMPT_KMEANS = """Background — PM7 semi-empirical proton affinity:
  Proton affinity (PA) = negative enthalpy of gas-phase protonation.
  PA = H(B) + H(H+) - H(BH+), where H(H+) = 5/2 RT = 6.197 kJ/mol at 298.15 K

  PM7 calculation (MOPAC):
    PA_PM7 (kcal/mol) = HOF_neutral - HOF_protonated + 365.7
    HOF = heat of formation output from MOPAC PM7 (kcal/mol)
    365.7 kcal/mol = enthalpy of a proton (H+) at 298K in MOPAC convention
    All N, O, S, P atoms with formal charge = 0 are protonated and evaluated.

Background — DFT reference:
  Level of theory: B3LYP/def2-TZVP
  Software: PySCF 2.12.1 + gpu4pyscf 1.5.2 (GPU-accelerated, NVIDIA A100)
  Pipeline: SMILES -> RDKit 3D embedding (ETKDG + MMFF94) -> geometry optimization
            (geomeTRIC) -> analytical Hessian -> vibrational frequencies ->
            IGRRHO thermochemistry at 298.15 K
  PA formula: PA = H(B) + H(H+) - H(BH+)
              H = E_elec + ZPE + E_vib + E_trans + E_rot + PV  (total enthalpy)
              H(H+) = 5/2 RT = 6.197 kJ/mol (translational enthalpy, no electrons)
              ZPE scale factor: 0.9850 for B3LYP/def2-TZVP

  Correction to predict (per protonation site):
    correction = PA_DFT - PA_PM7
    Positive: PM7 underestimates DFT PA at this protonation site
    Negative: PM7 overestimates DFT PA at this protonation site

  Note: a molecule has multiple protonation sites — each site has its own PA_PM7
  and PA_DFT. The protonated SMILES explicitly shows which atom carries the proton
  ([NH+], [OH+], [N+H], [O+H] etc.). You are predicting the correction for that
  SPECIFIC protonation site, not the molecule's best site.

Your task: given the neutral SMILES, all protonated forms with their PM7 PA values,
predict the signed correction (kcal/mol) for the marked protonation site needed to
match the DFT PA at that site.

Respond with ONLY a single floating-point number (kcal/mol). No units, no explanation.
Example valid responses: 6.3   or   -2.1   or   9.45"""

def get_system_prompt(dataset: str) -> str:
    return SYSTEM_PROMPT_NIST if dataset == "nist" else SYSTEM_PROMPT_KMEANS


def make_zero_shot_prompt(smiles: str, pm7_pa: float,
                          protonated_smiles: str | None = None,
                          all_sites: list | None = None) -> str:
    if all_sites and len(all_sites) >= 1:
        # Show all protonation sites — works for both NIST and k-means
        mark = protonated_smiles  # the site we want the correction for
        sites_sorted = sorted(all_sites, key=lambda x: x["pm7_pa"], reverse=True)
        sites_block  = "\n".join(
            f"  Site {i+1}: Protonated SMILES={s['protonated_smiles']}  "
            f"PM7 PA={s['pm7_pa']:.2f} kcal/mol"
            + (" << PREDICT CORRECTION FOR THIS SITE"
               if s['protonated_smiles'] == mark else "")
            for i, s in enumerate(sites_sorted)
        )
        ref_label = "PA_DFT" if protonated_smiles and mark != sites_sorted[0].get(
            'protonated_smiles') else "PA_reference"
        return f"""Neutral SMILES: {smiles}

All protonation sites computed by PM7 (sorted by PA, highest first):
{sites_block}

For the marked site, predict: correction = {ref_label} - PA_PM7

Reply with ONLY a single number (kcal/mol)."""
    else:
        site_line = (f"\nProtonated SMILES: {protonated_smiles}"
                     if protonated_smiles else "")
        return f"""Neutral SMILES: {smiles}{site_line}
PM7 proton affinity: {pm7_pa:.2f} kcal/mol

Predict: correction = PA_reference - PA_PM7

Reply with ONLY a single number (kcal/mol)."""

def make_few_shot_prompt(smiles: str, pm7_pa: float,
                         examples: list[dict],
                         protonated_smiles: str | None = None,
                         all_sites: list | None = None) -> str:
    ex_blocks = []
    for e in examples:
        block = f"  Neutral SMILES: {e['smiles']}\n"
        # Show all protonation sites if available, else just best site
        if e.get('all_sites') and len(e['all_sites']) > 0:
            sites_sorted = sorted(e['all_sites'], key=lambda x: x['pm7_pa'], reverse=True)
            for i, s in enumerate(sites_sorted):
                marker = " << best site (correction applies here)" if i == 0 else ""
                block += (f"    Site {i+1}: Protonated={s['protonated_smiles']}  "
                          f"PM7 PA={s['pm7_pa']:.2f} kcal/mol{marker}\n")
        elif e.get('protonated_smiles'):
            block += f"  Protonated SMILES: {e['protonated_smiles']}\n"
        block += (f"  PM7 PA (best site): {e['pm7_pa']:.2f} kcal/mol\n"
                  f"  Reference PA:       {e['pm7_pa'] + e['correction']:.2f} kcal/mol\n"
                  f"  Correction:         {e['correction']:+.2f} kcal/mol\n")
        ex_blocks.append(block)
    ex_block = "\n".join(ex_blocks)

    # Query section
    if all_sites and len(all_sites) >= 1:
        sites_sorted = sorted(all_sites, key=lambda x: x["pm7_pa"], reverse=True)
        sites_str = "\n".join(
            f"    Site {i+1}: Protonated={s['protonated_smiles']}  "
            f"PM7 PA={s['pm7_pa']:.2f} kcal/mol"
            + (" << PREDICT CORRECTION FOR THIS SITE" if s['protonated_smiles'] == protonated_smiles else "")
            for i, s in enumerate(sites_sorted)
        )
        query = (f"  Neutral SMILES: {smiles}\n"
                 f"  All protonation sites:\n{sites_str}\n"
                 f"  PM7 PA (best site): {pm7_pa:.2f} kcal/mol\n"
                 f"  Reference PA: ?\n"
                 f"  Correction: ?")
    else:
        site_line = f"\n  Protonated SMILES: {protonated_smiles}" if protonated_smiles else ""
        query = (f"  Neutral SMILES: {smiles}{site_line}\n"
                 f"  PM7 PA: {pm7_pa:.2f} kcal/mol\n"
                 f"  Reference PA: ?\n"
                 f"  Correction: ?")

    return f"""Here are example molecules with their protonation sites, PM7 PA, reference PA, and correction:

{ex_block}
Now predict the correction for this molecule (reference PA is unknown):
{query}

Reply with ONLY a single number (the correction in kcal/mol)."""

def make_rag_prompt(smiles: str, pm7_pa: float,
                    neighbors: list[dict],
                    protonated_smiles: str | None = None) -> str:
    neighbor_block = "\n".join(
        "  Neutral: " + n['smiles']
        + (" | Protonated: " + n['protonated_smiles']
           if n.get('protonated_smiles') else "")
        + f"  (Tanimoto={n['tanimoto']:.2f}, "
          f"PM7={n['pm7_pa']:.1f}, "
          f"Ref PA={n['pm7_pa'] + n['correction']:.1f}, "
          f"correction={n['correction']:+.2f} kcal/mol)"
        for n in neighbors
    )
    site_line = (f"\n  Protonated SMILES: {protonated_smiles}"
                 if protonated_smiles else "")
    return f"""The following molecules are structurally most similar to the query \
(by Tanimoto similarity on Morgan fingerprints):

{neighbor_block}

Based on these chemically similar examples, predict the correction for:
  Neutral SMILES: {smiles}{site_line}
  PM7 PA: {pm7_pa:.2f} kcal/mol
  Reference PA: ?

Consider: what atom is protonated? What functional groups drive the correction?
How does this protonation site compare structurally and electronically to the examples?

Reply with ONLY a single number (the predicted correction in kcal/mol)."""


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def parse_correction(response: str | None) -> float | None:
    """Extract a single float from LLM response."""
    import re
    if response is None:
        return None
    response = response.strip()
    if not response:
        return None
    # Try direct parse first
    try:
        return float(response)
    except ValueError:
        pass
    # Look for first signed number in response
    matches = re.findall(r"[-+]?\d+\.?\d*", response)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    df: pd.DataFrame,
    llm,
    approach: str,
    dataset: str = "nist",
    n_folds: int = 5,
    n_few_shot: int = 10,
    n_rag_neighbors: int = 5,
    n_test: int | None = None,
    random_state: int = 42,
) -> dict:
    """
    Run LLM correction prediction benchmark using k-fold CV.
    
    Returns dict with per-fold MAE and all predictions.
    """
    log.info(f"Running benchmark: approach={approach}, folds={n_folds}")

    rng  = np.random.RandomState(random_state)
    kf   = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    idxs = np.arange(len(df))

    # Precompute all Morgan FPs
    log.info("  Precomputing Morgan fingerprints ...")
    all_fps = compute_morgan_fps(df["smiles"].tolist())

    fold_maes    = []
    all_preds    = []
    all_true     = []
    all_smiles   = []
    all_pm7      = []
    all_folds    = []
    api_errors   = 0

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(idxs)):
        log.info(f"  Fold {fold_idx+1}/{n_folds}: {len(train_idx)} train, {len(test_idx)} test")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)
        test_fps = all_fps[test_idx]
        train_fps = all_fps[train_idx]

        # Optionally subsample test for speed
        if n_test and len(test_df) > n_test:
            sub_idx = rng.choice(len(test_df), n_test, replace=False)
            test_df  = test_df.iloc[sub_idx].reset_index(drop=True)
            test_fps = test_fps[sub_idx]
            log.info(f"    Subsampled to {len(test_df)} test molecules")

        fold_preds = []
        fold_true  = []

        # Build all-sites lookup for k-means (neutral_smiles -> list of sites)
        all_sites_lookup = {}
        if dataset == "kmeans" and "protonated_smiles" in df.columns:
            for _, r in df.iterrows():
                smi = r["smiles"] if "smiles" in r.index else r["neutral_smiles"]
                if smi not in all_sites_lookup:
                    all_sites_lookup[smi] = []
                all_sites_lookup[smi].append({
                    "protonated_smiles": r.get("protonated_smiles", ""),
                    "pm7_pa": r["pm7_pa_kcalmol"],
                })

        for i, row in test_df.iterrows():
            smiles    = row["smiles"]
            pm7_pa    = row["pm7_pa_kcalmol"]
            true_corr = row["correction_kcalmol"]
            # Protonated SMILES — only available for k-means site-level
            prot_smiles = (row["protonated_smiles"]
                           if "protonated_smiles" in row.index else None)
            # All protonation sites for this molecule
            all_sites = all_sites_lookup.get(smiles, None)

            # Build prompt
            if approach == "zero_shot":
                prompt = make_zero_shot_prompt(
                    smiles, pm7_pa, prot_smiles, all_sites)
                messages = [{"role": "user", "content": prompt}]

            elif approach == "few_shot":
                # Sample n random training examples
                ex_idx = rng.choice(len(train_df),
                                    min(n_few_shot, len(train_df)), replace=False)
                examples = [
                    {"smiles":            train_df.iloc[j]["smiles"],
                     "protonated_smiles": (train_df.iloc[j]["protonated_smiles"]
                                          if "protonated_smiles" in train_df.columns
                                          else None),
                     "all_sites":         (train_df.iloc[j]["all_sites"]
                                          if "all_sites" in train_df.columns
                                          else None),
                     "pm7_pa":            train_df.iloc[j]["pm7_pa_kcalmol"],
                     "ref_pa":            train_df.iloc[j]["exp_pa_kcalmol"],
                     "correction":        train_df.iloc[j]["correction_kcalmol"]}
                    for j in ex_idx
                ]
                prompt = make_few_shot_prompt(smiles, pm7_pa, examples, prot_smiles)
                messages = [{"role": "user", "content": prompt}]

            elif approach == "rag":
                # Retrieve k most similar training molecules
                query_fp  = test_fps[i].reshape(1, -1)
                sims      = tanimoto_matrix(query_fp, train_fps)[0]
                top_k_idx = np.argsort(sims)[-n_rag_neighbors:][::-1]
                neighbors = [
                    {"smiles":            train_df.iloc[j]["smiles"],
                     "protonated_smiles": (train_df.iloc[j]["protonated_smiles"]
                                          if "protonated_smiles" in train_df.columns
                                          else None),
                     "all_sites":         (train_df.iloc[j]["all_sites"]
                                          if "all_sites" in train_df.columns
                                          else None),
                     "pm7_pa":            train_df.iloc[j]["pm7_pa_kcalmol"],
                     "ref_pa":            train_df.iloc[j]["exp_pa_kcalmol"],
                     "correction":        train_df.iloc[j]["correction_kcalmol"],
                     "tanimoto":          float(sims[j])}
                    for j in top_k_idx
                ]
                prompt = make_rag_prompt(smiles, pm7_pa, neighbors, prot_smiles)
                messages = [{"role": "user", "content": prompt}]

            else:
                raise ValueError(f"Unknown approach: {approach}")

            # Call LLM
            pred_corr = None
            try:
                response = llm.chat(messages, system=get_system_prompt(dataset))
                pred_corr = parse_correction(response)
                if pred_corr is None:
                    log.warning(f"    Could not parse response for "
                                f"{smiles[:30]}: {str(response)[:50]}")
                    api_errors += 1
            except Exception as e:
                log.warning(f"    API error for {smiles[:30]}: {str(e)[:80]}")
                api_errors += 1

            # Skip failed predictions from MAE — do not default to 0.0
            if pred_corr is None:
                continue

            fold_preds.append(pred_corr)
            fold_true.append(true_corr)
            all_smiles.append(smiles)
            all_pm7.append(pm7_pa)
            all_folds.append(fold_idx)

            # Checkpoint every 50 molecules
            if len(fold_preds) % 50 == 0:
                running_mae = np.mean(np.abs(
                    np.array(fold_preds) - np.array(fold_true)))
                log.info(f"    Checkpoint [{len(fold_preds)} predictions]: "
                         f"running MAE = {running_mae:.2f} kcal/mol")

            # Log progress every 10 molecules
            if (i + 1) % 10 == 0:
                running_mae = np.mean(np.abs(
                    np.array(fold_preds) - np.array(fold_true)))
                log.info(f"    [{i+1}/{len(test_df)}] running MAE = {running_mae:.2f} kcal/mol")

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        n_success = len(fold_preds)
        n_total   = len(test_df)
        fold_mae  = float(np.mean(np.abs(
            np.array(fold_preds) - np.array(fold_true)))) if fold_preds else float("nan")
        fold_maes.append(fold_mae)
        all_preds.extend(fold_preds)
        all_true.extend(fold_true)
        log.info(f"  Fold {fold_idx+1} MAE = {fold_mae:.2f} kcal/mol "
                 f"({n_success}/{n_total} successful predictions)")

    overall_mae = float(np.mean(np.abs(
        np.array(all_preds) - np.array(all_true)))) if all_preds else float("nan")
    mae_std = float(np.std([m for m in fold_maes if not np.isnan(m)]))
    n_success = len(all_preds)
    n_total_attempted = sum(len(test_df) for _, test_df in
                            list(KFold(n_splits=n_folds, shuffle=True,
                            random_state=random_state).split(df))[:n_folds])

    log.info(f"  {approach} MAE = {overall_mae:.2f} ± {mae_std:.2f} kcal/mol "
             f"on {n_success} successful predictions")
    if api_errors > 0:
        log.warning(f"  {api_errors} API errors — excluded from MAE "
                    f"(success rate: {100*n_success/(n_success+api_errors):.0f}%)")

    return {
        "approach":     approach,
        "mae":          overall_mae,
        "mae_std":      mae_std,
        "fold_maes":    fold_maes,
        "predictions":  all_preds,
        "true_values":  all_true,
        "smiles":       all_smiles,
        "pm7_pa":       all_pm7,
        "fold_indices": all_folds,
        "api_errors":   api_errors,
        "n_success":    len(all_preds),
        "success_rate": len(all_preds) / (len(all_preds) + api_errors)
                        if (len(all_preds) + api_errors) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Summary table + plot
# ---------------------------------------------------------------------------

def print_summary(dataset: str, results: dict[str, dict], df: pd.DataFrame):
    raw_pm7_mae = float(df["correction_kcalmol"].abs().mean())

    # ExtraTrees reference MAE
    et_mae = 2.87 if dataset == "nist" else 6.85

    print()
    print("=" * 65)
    print(f"  CORRECTION PREDICTION BENCHMARK — {dataset.upper()}")
    print("=" * 65)
    print(f"  {'Method':<25} {'MAE (kcal/mol)':>15}  {'vs ExtraTrees':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Raw PM7 (no correction)':<25} {raw_pm7_mae:>14.2f}  {'—':>15}")
    print(f"  {'ExtraTrees (ours)':<25} {et_mae:>14.2f}  {'baseline':>15}")
    for name, res in results.items():
        mae  = res["mae"]
        diff = mae - et_mae
        diff_str    = f"{diff:+.2f}"
        success_pct = f"{100*res.get('success_rate', 1.0):.0f}%"
        print(f"  {name:<25} {mae:>14.2f}  {diff_str:>12}  ({success_pct} parsed)")
    print("=" * 65)
    print()


def plot_parity(results: dict[str, dict], out_dir: Path, dataset: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        "font.family": "sans-serif", "axes.linewidth": 1.5,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.labelsize": 18, "figure.dpi": 300,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    n_approaches = len(results)
    fig, axes = plt.subplots(1, n_approaches, figsize=(6 * n_approaches, 5))
    if n_approaches == 1:
        axes = [axes]

    colors = {"zero_shot": "#E24B4A", "few_shot": "#F0A500", "rag": "#2166AC"}

    for ax, (name, res) in zip(axes, results.items()):
        true = np.array(res["true_values"])
        pred = np.array(res["predictions"])
        mae  = res["mae"]

        color = colors.get(name, "steelblue")
        ax.scatter(true, pred, s=10, alpha=0.5, color=color,
                   linewidths=0, rasterized=True)

        # 1:1 line
        lo = min(true.min(), pred.min()) - 2
        hi = max(true.max(), pred.max()) + 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, alpha=0.6)

        ax.set_xlabel("True correction (kcal/mol)")
        ax.set_ylabel("Predicted correction (kcal/mol)" if ax == axes[0] else "")
        ax.set_title(f"{name.replace('_',' ').title()}\nMAE = {mae:.2f} kcal/mol",
                     fontsize=14)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    fig.suptitle(f"LLM correction prediction — {dataset.upper()} dataset",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"llm_parity_{dataset}.pdf")
    fig.savefig(out_dir / f"llm_parity_{dataset}.png")
    plt.close(fig)
    log.info(f"  Saved {out_dir}/llm_parity_{dataset}.pdf/.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM correction prediction vs ExtraTrees.")
    parser.add_argument("--dataset",    default="nist",
                        choices=["nist","kmeans"])
    parser.add_argument("--approach",   default="all",
                        choices=["all","zero_shot","few_shot","rag"])
    parser.add_argument("--provider",   default="gemini",
                        choices=["gemini","openai","anthropic","claude"])
    parser.add_argument("--model",      default="gemini-2.0-flash-exp")
    parser.add_argument("--n-folds",    type=int, default=5)
    parser.add_argument("--n-few-shot", type=int, default=10,
                        help="Number of examples for few-shot prompt")
    parser.add_argument("--n-rag",      type=int, default=5,
                        help="Number of similar molecules for RAG prompt")
    parser.add_argument("--n-test",     type=int, default=None,
                        help="Subsample test set per fold (None = all). "
                             "Use 20-30 for a quick pilot run.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = deterministic (recommended for regression)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load provider
    from llm_provider import get_provider
    log.info(f"Loading {args.provider} / {args.model}")
    llm = get_provider(
        provider_name=args.provider,
        model=args.model,
        keys_file=API_KEYS_FILE,
        temperature=args.temperature,
        max_tokens=50,   # only need a single number
    )
    log.info(f"  Provider: {llm}")

    # Load data
    df = load_dataset(args.dataset)

    # Decide which approaches to run
    approaches = (["zero_shot","few_shot","rag"]
                  if args.approach == "all" else [args.approach])

    all_results = {}
    for approach in approaches:
        log.info(f"\n{'='*50}")
        log.info(f"  Approach: {approach}")
        log.info(f"{'='*50}")
        result = run_benchmark(
            df=df,
            llm=llm,
            approach=approach,
            dataset=args.dataset,
            n_folds=args.n_folds,
            n_few_shot=args.n_few_shot,
            n_rag_neighbors=args.n_rag,
            n_test=args.n_test,
            random_state=42,
        )
        all_results[approach] = result

    # Print summary
    print_summary(args.dataset, all_results, df)

    # Save results
    save = {k: {kk: vv for kk, vv in v.items()
                if kk not in ("smiles","pm7_pa")}  # keep file small
            for k, v in all_results.items()}
    # Ensure fold_indices are saved for plotting
    for k in save:
        if "fold_indices" not in save[k] and "fold_indices" in all_results[k]:
            save[k]["fold_indices"] = all_results[k]["fold_indices"]
    out_file = RESULTS_DIR / f"llm_benchmark_{args.dataset}_{args.model.replace('/','_')}.json"
    out_file.write_text(json.dumps(save, indent=2))
    log.info(f"Results saved to {out_file}")

    # Plot
    plot_parity(all_results,
                SCRIPT_DIR / "benchmark_figures",
                args.dataset)


if __name__ == "__main__":
    main()
