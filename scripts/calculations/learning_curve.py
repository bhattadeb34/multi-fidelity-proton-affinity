"""
learning_curve.py
=================
Proper nested learning curve analysis for NIST (1155 molecules) and
k-means (251 molecules / 821 sites) datasets.

For every (dataset, feature_set, seed, fraction) cell, the FULL pipeline
is re-run from scratch on the subsampled training data only:
  1. NaN imputation with training-fold column medians
  2. Four-stage feature selection (variance filter → correlation filter →
     LassoCV → 1-SE rule) on the training subset only
  3. ExtraTrees training on selected features
  4. Evaluation on a fixed held-out test set (20% of total)

This is the correct approach: at each fraction the feature selection has no
knowledge of the held-out test molecules, and small training sizes use
correspondingly noisier, smaller feature sets.

Outputs (per dataset × feature_set):
  results/learning_curve_{dataset}_{feat}/
    learning_curve_data.csv      — one row per (seed, fraction) cell
    learning_curve_detail.json   — full per-cell detail including selected
                                   feature names, importances, fold stats

Figures (in figures/model_performance/):
  learning_curve_nist_pm7.pdf
  learning_curve_nist_dft.pdf
  learning_curve_kmeans_pm7.pdf
  learning_curve_kmeans_dft.pdf
  learning_curve_combined.pdf    — 2×2 subplot: NIST and k-means side by side

Runtime estimate: 4-8 hours total
(2 datasets × 2 feature sets × 5 seeds × 11 fractions × full LassoCV)

Usage
-----
  python scripts/learning_curve.py
  python scripts/learning_curve.py --datasets nist        # NIST only
  python scripts/learning_curve.py --datasets kmeans      # k-means only
  python scripts/learning_curve.py --datasets all         # default
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR    = PROJECT_DIR / "data"
TARGET_DIR  = DATA_DIR / "targets"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR     = PROJECT_DIR / "figures"
FIG_PERF    = FIG_DIR / "model_performance"

KJMOL_TO_KCAL = 1 / 4.184

FRACTIONS = [0.08, 0.16, 0.24, 0.32, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
SEEDS     = [42, 123, 456, 789, 1024]
TEST_FRAC = 0.20

# Non-feature columns — never used as ML inputs
NON_FEATURE_COLS = {
    "record_id", "mol_id", "source", "dataset",
    "neutral_smiles", "protonated_smiles", "site_idx", "site_name",
    "mordred_geom_source",
    "exp_pa_kjmol", "exp_pa_kcalmol",
    "dft_pa_kjmol", "dft_pa_kcalmol",
    "pm7_pa_kjmol", "pm7_pa_kcalmol",
    "pm7_best_pa_kjmol", "pm7_best_pa_kcalmol",
    "delta_dft_exp", "delta_pm7_exp", "dft_correction",
    "delta_dft_pm7", "raw_pm7_error",
    "correction_kcalmol",
}

# DFT extra features — full list (NIST safe)
DFT_EXTRA_FEATURES_NIST = [
    "dft_neutral_ZPE_kjmol",
    "dft_neutral_H_total_Ha",
    "dft_neutral_n_basis",
    "dft_neutral_n_electrons",
    "dft_neutral_n_imaginary",
    "dft_neutral_freq_min_cm",
    "dft_neutral_freq_max_cm",
    "dft_neutral_n_low_freq",
    "dft_prot_ZPE_kjmol",
    "dft_prot_H_total_Ha",
    "dft_prot_n_basis",
    "dft_prot_n_electrons",
    "dft_prot_n_imaginary",
    "dft_prot_freq_min_cm",
    "dft_prot_freq_max_cm",
    "dft_prot_n_low_freq",
    "dft_delta_ZPE_kjmol",
    "dft_delta_HOMO_LUMO_gap_eV",
    "dft_delta_dipole_debye",
]

# DFT extra features — k-means safe list (absolute enthalpies excluded)
# dft_neutral_H_total_Ha and dft_prot_H_total_Ha encode PA_DFT directly
# since PA_DFT = (H_neutral - H_prot) * 2625.5 / 4.184 + const
DFT_EXTRA_FEATURES_KMEANS = [
    "dft_neutral_ZPE_kjmol",
    "dft_neutral_n_basis",
    "dft_neutral_n_electrons",
    "dft_neutral_n_imaginary",
    "dft_neutral_freq_min_cm",
    "dft_neutral_freq_max_cm",
    "dft_neutral_n_low_freq",
    "dft_prot_ZPE_kjmol",
    "dft_prot_n_basis",
    "dft_prot_n_electrons",
    "dft_prot_n_imaginary",
    "dft_prot_freq_min_cm",
    "dft_prot_freq_max_cm",
    "dft_prot_n_low_freq",
    "dft_delta_ZPE_kjmol",
    "dft_delta_HOMO_LUMO_gap_eV",
    "dft_delta_dipole_debye",
]

# Style
TICK_SIZE   = 22
LABEL_SIZE  = 26
LEGEND_SIZE = 18
TITLE_SIZE  = 22
SPINE_LW    = 1.5


# ---------------------------------------------------------------------------
# Feature selection (identical to train_models.py)
# ---------------------------------------------------------------------------

def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    lasso_cv_folds: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """
    Four-stage feature selection on training data only.
    Returns (X_selected, selected_feature_names).
    """
    names = list(feature_names)
    X = X_train.copy()

    # Impute NaN with training column medians
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # Stage 1: Variance filter
    vt = VarianceThreshold(threshold=variance_threshold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = vt.fit_transform(X)
    names = [n for n, keep in zip(names, vt.get_support()) if keep]
    if not names:
        return X, names

    # Stage 2: Correlation filter
    if len(names) > 1:
        df_tmp = pd.DataFrame(X, columns=names)
        target_corr = df_tmp.corrwith(pd.Series(y_train)).abs()
        corr_matrix = df_tmp.corr().abs()
        to_drop = set()
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            if cols[i] in to_drop:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_drop:
                    continue
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    if target_corr.get(cols[i], 0) >= target_corr.get(cols[j], 0):
                        to_drop.add(cols[j])
                    else:
                        to_drop.add(cols[i])
        keep_mask = [n not in to_drop for n in names]
        X = X[:, keep_mask]
        names = [n for n, k in zip(names, keep_mask) if k]
    if not names:
        return X, names

    # Stage 3 + 4: LassoCV with 1-SE rule
    n_cv = min(lasso_cv_folds, max(2, len(y_train) - 1))
    if n_cv < 2:
        return X, names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso_cv = LassoCV(cv=n_cv, max_iter=10000, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso_cv.fit(X_scaled, y_train)

    mean_mse  = lasso_cv.mse_path_.mean(axis=1)
    std_mse   = lasso_cv.mse_path_.std(axis=1)
    best_idx  = np.argmin(mean_mse)
    threshold = mean_mse[best_idx] + std_mse[best_idx]
    alpha_1se = lasso_cv.alphas_[mean_mse <= threshold].max()

    lasso_1se = Lasso(alpha=alpha_1se, max_iter=10000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso_1se.fit(X_scaled, y_train)

    selected_mask = lasso_1se.coef_ != 0
    if selected_mask.sum() == 0:
        lasso_best = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_best.fit(X_scaled, y_train)
        selected_mask = lasso_best.coef_ != 0
    if selected_mask.sum() == 0:
        top_k = min(10, len(names))
        top_idx = np.argsort(np.abs(lasso_cv.coef_))[-top_k:]
        selected_mask = np.zeros(len(names), dtype=bool)
        selected_mask[top_idx] = True

    X = X[:, selected_mask]
    names = [n for n, s in zip(names, selected_mask) if s]
    return X, names


def apply_selection_to_test(
    X_test: np.ndarray,
    X_train_raw: np.ndarray,
    feature_names_full: list[str],
    selected_names: list[str],
) -> np.ndarray:
    """Apply training-derived feature selection to the test set."""
    name_to_idx = {n: i for i, n in enumerate(feature_names_full)}
    sel_idx = [name_to_idx[n] for n in selected_names]
    X = X_test[:, sel_idx].copy()
    col_medians = np.nanmedian(X_train_raw[:, sel_idx], axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return X


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(model_name: str, seed: int):
    if model_name in ("ExtraTrees", "ExtraTreesRegressor"):
        return ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    elif model_name in ("RandomForest", "RandomForestRegressor"):
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    else:
        log.warning(f"  {model_name} not directly supported; using ExtraTrees")
        return ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)


def get_best_model(results_subdir: Path) -> tuple[str, float]:
    """
    Return (best_model_name, mae_pa_mean) from cv_results.json,
    excluding VotingEnsemble.
    """
    cv = json.loads((results_subdir / "cv_results.json").read_text())
    best = min(
        (m for m in cv["models"] if m != "VotingEnsemble"),
        key=lambda m: cv["models"][m].get("mae_pa_mean") or 999,
    )
    mae = cv["models"][best]["mae_pa_mean"]
    log.info(f"  Best model: {best}  (5-fold CV MAE = {mae:.3f} kcal/mol)")
    return best, mae


def load_dft_features(data_dir: Path) -> pd.DataFrame:
    """Load DFT features from dataset.json, one row per unique neutral SMILES."""
    dataset_path = data_dir / "processed" / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    rows = []
    for rec in dataset.values():
        if rec["metadata"]["source"] not in ("folder", "json"):
            continue
        neu = rec["neutral"]
        neu_smi = neu.get("smiles", "")
        if not neu_smi:
            continue
        row = {
            "neutral_smiles":           neu_smi,
            "dft_neutral_ZPE_kjmol":    neu.get("ZPE_kjmol"),
            "dft_neutral_H_total_Ha":   neu.get("H_total_Ha"),
            "dft_neutral_n_basis":      neu.get("n_basis"),
            "dft_neutral_n_electrons":  neu.get("n_electrons"),
            "dft_neutral_n_imaginary":  neu.get("n_imaginary"),
            "dft_neutral_freq_min_cm":  neu.get("freq_min_cm"),
            "dft_neutral_freq_max_cm":  neu.get("freq_max_cm"),
            "dft_neutral_n_low_freq":   neu.get("n_low_freq"),
        }
        # Use the first available site for protonated-state DFT descriptors
        for site in rec.get("all_sites", []):
            row.update({
                "protonated_smiles":           site.get("protonated_smiles", ""),
                "dft_prot_ZPE_kjmol":          site.get("ZPE_kjmol"),
                "dft_prot_H_total_Ha":         site.get("H_total_Ha"),
                "dft_prot_n_basis":            site.get("n_basis"),
                "dft_prot_n_electrons":        site.get("n_electrons"),
                "dft_prot_n_imaginary":        site.get("n_imaginary"),
                "dft_prot_freq_min_cm":        site.get("freq_min_cm"),
                "dft_prot_freq_max_cm":        site.get("freq_max_cm"),
                "dft_prot_n_low_freq":         site.get("n_low_freq"),
                "dft_delta_ZPE_kjmol":         site.get("delta_ZPE_kjmol"),
                "dft_delta_HOMO_LUMO_gap_eV":  site.get("delta_HOMO_LUMO_gap_eV"),
                "dft_delta_dipole_debye":      site.get("delta_dipole_debye"),
            })
            break
        rows.append(row)
    df = pd.DataFrame(rows).drop_duplicates(subset=["neutral_smiles"])
    log.info(f"  DFT features loaded: {len(df)} molecules")
    return df


# ---------------------------------------------------------------------------
# Core learning curve computation
# ---------------------------------------------------------------------------

def run_learning_curve(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    pa_pm7_col: str,
    pa_true_col: str,
    model_name: str,
    fractions: list[float],
    seeds: list[int],
    test_frac: float,
    unit_scale: float = KJMOL_TO_KCAL,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Nested learning curve: full pipeline re-run for every (seed, fraction) cell.

    Returns
    -------
    lc_df : pd.DataFrame
        One row per (seed, fraction) with summary statistics.
        Columns: fraction, n_train, n_train_full, n_total, seed,
                 mae_test, mae_train, n_features_selected,
                 n_features_pool (after variance+corr filter, before Lasso)

    detail : list[dict]
        One dict per (seed, fraction) with full detail:
        {
          seed, fraction, n_train, n_train_full, n_total,
          mae_test, mae_train,
          n_features_pool,       # after variance+corr filter
          n_features_selected,   # after Lasso 1-SE
          selected_feature_names: [...],
          feature_importances: {name: importance, ...},
          test_indices: [...],   # row indices used as test set for this seed
          train_indices: [...],  # row indices in full train set for this seed
          subsample_indices: [...], # which of train_indices were used at this frac
        }
    """
    avail_cols = [c for c in feature_cols if c in df.columns]
    missing = len(feature_cols) - len(avail_cols)
    if missing > 0:
        log.warning(f"  {missing} requested features not in dataset — skipped")
    log.info(f"  Feature pool: {len(avail_cols)} features")

    X_all       = df[avail_cols].values.astype(np.float64)
    y_all       = df[target_col].values.astype(np.float64)
    pa_pm7_all  = df[pa_pm7_col].values
    pa_true_all = df[pa_true_col].values
    n_total     = len(df)

    rows   = []
    detail = []
    total_cells = len(seeds) * len(fractions)
    cell = 0

    for seed in seeds:
        idx = np.arange(n_total)
        train_idx, test_idx = train_test_split(
            idx, test_size=test_frac, random_state=seed)

        X_test_raw    = X_all[test_idx].copy()
        y_test        = y_all[test_idx]
        pa_pm7_test   = pa_pm7_all[test_idx]
        pa_true_test  = pa_true_all[test_idx]

        X_train_full  = X_all[train_idx].copy()
        y_train_full  = y_all[train_idx]
        pa_pm7_train_full  = pa_pm7_all[train_idx]
        pa_true_train_full = pa_true_all[train_idx]
        n_train_full  = len(train_idx)

        for frac in fractions:
            cell += 1

            if frac >= 1.0:
                sub_idx = np.arange(n_train_full)
                n_tr = n_train_full
            else:
                n_tr    = max(20, int(frac * n_train_full))
                rng     = np.random.default_rng(seed + int(frac * 1000))
                sub_idx = rng.choice(n_train_full, size=n_tr, replace=False)

            X_tr_raw    = X_train_full[sub_idx].copy()
            y_tr        = y_train_full[sub_idx]
            pa_pm7_tr   = pa_pm7_train_full[sub_idx]
            pa_true_tr  = pa_true_train_full[sub_idx]

            log.info(f"  [{cell:3d}/{total_cells}] seed={seed}  "
                     f"frac={frac:.2f}  n_train={n_tr:4d}  — feature selection ...")

            # Full feature selection on this training subset only
            X_tr_sel, sel_names = select_features(
                X_tr_raw, y_tr, avail_cols,
                variance_threshold, correlation_threshold,
            )
            n_sel = len(sel_names)

            # Count pool size after variance+corr filter (before Lasso)
            # We re-run stages 1+2 only to get pool size
            _X = X_tr_raw.copy()
            col_med = np.nanmedian(_X, axis=0)
            nm = np.isnan(_X)
            if nm.any():
                _X[nm] = np.take(col_med, np.where(nm)[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vt = VarianceThreshold(threshold=variance_threshold)
                _X = vt.fit_transform(_X)
            n_pool = int(_X.shape[1])  # rough proxy after variance filter

            if n_sel == 0:
                log.warning(f"    No features selected — skipping")
                entry = dict(
                    seed=seed, fraction=frac, n_train=n_tr,
                    n_train_full=n_train_full, n_total=n_total,
                    mae_test=np.nan, mae_train=np.nan,
                    n_features_pool=n_pool, n_features_selected=0,
                    selected_feature_names=[],
                    feature_importances={},
                    test_indices=test_idx.tolist(),
                    train_indices=train_idx.tolist(),
                    subsample_indices=train_idx[sub_idx].tolist(),
                )
                rows.append({k: v for k, v in entry.items()
                             if k not in ("selected_feature_names",
                                          "feature_importances",
                                          "test_indices", "train_indices",
                                          "subsample_indices")})
                detail.append(entry)
                continue

            # Apply selection to test set
            X_te_sel = apply_selection_to_test(
                X_test_raw, X_tr_raw, avail_cols, sel_names)

            # Train
            model = build_model(model_name, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_sel, y_tr)
                y_pred_test  = model.predict(X_te_sel)
                y_pred_train = model.predict(X_tr_sel)

            # Evaluate on PA scale
            pa_pred_test  = pa_pm7_test + y_pred_test
            pa_pred_train = pa_pm7_tr   + y_pred_train
            mae_test  = mean_absolute_error(pa_true_test,  pa_pred_test)  * unit_scale
            mae_train = mean_absolute_error(pa_true_tr,    pa_pred_train) * unit_scale

            # Feature importances
            importances = {}
            if hasattr(model, "feature_importances_"):
                for fname, imp in zip(sel_names, model.feature_importances_):
                    importances[fname] = float(imp)

            log.info(f"    → {n_sel} features  |  "
                     f"test MAE = {mae_test:.3f}  train MAE = {mae_train:.3f} kcal/mol")

            entry = dict(
                seed=int(seed),
                fraction=float(frac),
                n_train=int(n_tr),
                n_train_full=int(n_train_full),
                n_total=int(n_total),
                mae_test=float(mae_test),
                mae_train=float(mae_train),
                n_features_pool=int(n_pool),
                n_features_selected=int(n_sel),
                selected_feature_names=sel_names,
                feature_importances=importances,
                test_indices=test_idx.tolist(),
                train_indices=train_idx.tolist(),
                subsample_indices=train_idx[sub_idx].tolist(),
            )
            rows.append({k: v for k, v in entry.items()
                         if k not in ("selected_feature_names",
                                      "feature_importances",
                                      "test_indices", "train_indices",
                                      "subsample_indices")})
            detail.append(entry)

    lc_df = pd.DataFrame(rows)
    return lc_df, detail


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _rcparams():
    plt.rcParams.update({
        "axes.linewidth":    SPINE_LW,
        "xtick.major.width": SPINE_LW,
        "ytick.major.width": SPINE_LW,
        "xtick.labelsize":   TICK_SIZE,
        "ytick.labelsize":   TICK_SIZE,
        "axes.labelsize":    LABEL_SIZE,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


def _aggregate(lc_df: pd.DataFrame) -> pd.DataFrame:
    test  = lc_df.groupby("fraction")["mae_test"].agg(
        mae_mean="mean", mae_std="std").reset_index()
    train = lc_df.groupby("fraction")["mae_train"].agg(
        train_mean="mean", train_std="std").reset_index()
    n_tr  = lc_df.groupby("fraction")["n_train"].median(
        ).round().astype(int).reset_index()
    n_sel = lc_df.groupby("fraction")["n_features_selected"].agg(
        nfeat_mean="mean", nfeat_std="std").reset_index()
    return (test.merge(train, on="fraction")
               .merge(n_tr, on="fraction")
               .merge(n_sel, on="fraction"))


def _add_secondary_xaxis(ax, summary):
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    pcts = summary["fraction"].values * 100
    ax2.set_xticks(pcts)
    ax2.set_xticklabels([str(n) for n in summary["n_train"].values],
                        rotation=45, ha="left", fontsize=TICK_SIZE - 5)
    ax2.set_xlabel("Number of training samples", fontsize=LABEL_SIZE - 6, labelpad=8)
    return ax2


def plot_single(
    lc_df: pd.DataFrame,
    title: str,
    output_stem: str,
    color: str,
    ref_mae: float | None = None,
    ref_label: str | None = None,
):
    """Single-dataset learning curve with legend outside and n_features annotation."""
    summary = _aggregate(lc_df)
    _rcparams()
    fig, ax = plt.subplots(figsize=(12, 6))
    pct = summary["fraction"] * 100

    ax.errorbar(
        pct, summary["mae_mean"], yerr=summary["mae_std"],
        fmt="o-", color=color, linewidth=2.2, markersize=8,
        capsize=5, capthick=1.8, elinewidth=1.8,
        markeredgecolor="white", markeredgewidth=0.8,
        zorder=3, label="Test MAE",
    )
    ax.errorbar(
        pct, summary["train_mean"], yerr=summary["train_std"],
        fmt="s--", color="#888888", linewidth=2.0, markersize=7,
        capsize=4, capthick=1.5, elinewidth=1.5,
        markeredgecolor="white", markeredgewidth=0.8,
        zorder=2, label="Train MAE",
    )

    if ref_mae is not None:
        ax.axhline(ref_mae, color="black", linewidth=1.5, linestyle=":",
                   zorder=1,
                   label=ref_label or f"5-fold CV MAE: {ref_mae:.2f} kcal/mol")

    # Annotate mean number of features selected at each point
    for _, row in summary.iterrows():
        ax.annotate(
            f"n={row['nfeat_mean']:.0f}",
            xy=(row["fraction"] * 100, row["mae_mean"]),
            xytext=(0, 10), textcoords="offset points",
            fontsize=9, ha="center", color=color, alpha=0.75,
        )

    _add_secondary_xaxis(ax, summary)
    ax.set_xlabel("Training set size (%)", fontsize=LABEL_SIZE)
    ax.set_ylabel("MAE (kcal/mol)", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=28, fontweight="bold")
    ax.legend(framealpha=0.9, edgecolor="lightgray",
              bbox_to_anchor=(1.01, 1), loc="upper left",
              borderaxespad=0, fontsize=LEGEND_SIZE)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linewidth=0.5, alpha=0.35, linestyle="--")
    fig.tight_layout()

    FIG_PERF.mkdir(parents=True, exist_ok=True)
    out = FIG_PERF / f"{output_stem}.pdf"
    fig.savefig(out)
    log.info(f"  Saved {out}")
    plt.close(fig)


def plot_combined(
    lc_nist_pm7: pd.DataFrame,
    lc_nist_dft: pd.DataFrame,
    lc_km_pm7:   pd.DataFrame,
    lc_km_dft:   pd.DataFrame,
    ref_nist_pm7: float,
    ref_nist_dft: float,
    ref_km_pm7:   float,
    ref_km_dft:   float,
    output_stem: str = "learning_curve_combined",
):
    """
    2×2 subplot figure:
      Top row:    NIST — PM7 only | PM7+DFT
      Bottom row: k-means — PM7 only | PM7+DFT

    Each panel shows test MAE (solid) + train MAE (dashed) with error bars,
    a reference line for the full 5-fold CV MAE, and the mean number of
    selected features annotated above each test-MAE point.
    Legend placed outside to the right.
    """
    COLOR_PM7 = "#D01C8B"
    COLOR_DFT = "#2166AC"

    panels = [
        (lc_nist_pm7, ref_nist_pm7, COLOR_PM7,
         "NIST — Molecular + PM7 features",      "ExtraTrees"),
        (lc_nist_dft, ref_nist_dft, COLOR_DFT,
         "NIST — Molecular + PM7 + DFT features",  "ExtraTrees"),
        (lc_km_pm7,   ref_km_pm7,   COLOR_PM7,
         "k-means — Molecular + PM7 features",   "ExtraTrees"),
        (lc_km_dft,   ref_km_dft,   COLOR_DFT,
         "k-means — Molecular + PM7 + DFT features","ExtraTrees"),
    ]

    _rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(22, 12))
    axes = axes.flatten()

    for ax, (lc_df, ref_mae, color, title, _model) in zip(axes, panels):
        summary = _aggregate(lc_df)
        pct = summary["fraction"] * 100

        ax.errorbar(
            pct, summary["mae_mean"], yerr=summary["mae_std"],
            fmt="o-", color=color, linewidth=2.0, markersize=7,
            capsize=4, capthick=1.5, elinewidth=1.5,
            markeredgecolor="white", markeredgewidth=0.7,
            zorder=3, label="Test MAE",
        )
        ax.errorbar(
            pct, summary["train_mean"], yerr=summary["train_std"],
            fmt="s--", color="#888888", linewidth=1.8, markersize=6,
            capsize=3, capthick=1.2, elinewidth=1.2,
            markeredgecolor="white", markeredgewidth=0.7,
            zorder=2, label="Train MAE",
        )
        ax.axhline(ref_mae, color="black", linewidth=1.4, linestyle=":",
                   zorder=1, label=f"5-fold CV: {ref_mae:.2f} kcal/mol")

        # Annotate n_features above each test MAE point
        for _, row in summary.iterrows():
            ax.annotate(
                f"n={row['nfeat_mean']:.0f}",
                xy=(row["fraction"] * 100, row["mae_mean"]),
                xytext=(0, 9), textcoords="offset points",
                fontsize=8, ha="center", color=color, alpha=0.8,
            )

        # Secondary x-axis (number of training samples)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(pct)
        ax2.set_xticklabels([str(n) for n in summary["n_train"].values],
                            rotation=45, ha="left", fontsize=TICK_SIZE - 7)
        ax2.set_xlabel("Training samples", fontsize=LABEL_SIZE - 8, labelpad=6)

        ax.set_xlabel("Training set size (%)", fontsize=LABEL_SIZE - 4)
        ax.set_ylabel("MAE (kcal/mol)", fontsize=LABEL_SIZE - 4)
        ax.set_title(title, fontsize=TITLE_SIZE - 2, pad=26, fontweight="bold")
        ax.legend(framealpha=0.9, edgecolor="lightgray",
                  bbox_to_anchor=(1.01, 1), loc="upper left",
                  borderaxespad=0, fontsize=LEGEND_SIZE - 2)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.grid(axis="y", linewidth=0.4, alpha=0.35, linestyle="--")

    fig.suptitle("Learning curves: NIST and k-means datasets",
                 fontsize=TITLE_SIZE + 2, fontweight="bold", y=1.01)
    fig.tight_layout()

    FIG_PERF.mkdir(parents=True, exist_ok=True)
    out = FIG_PERF / f"{output_stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    log.info(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_nist_pm7(data_dir: Path, target_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load NIST dataset merged with PM7 features. Returns (df, feature_cols)."""
    tgt = pd.read_parquet(target_dir / "nist1155_ml.parquet")
    if "correction_kcalmol" not in tgt.columns:
        tgt["correction_kcalmol"] = tgt["exp_pa_kcalmol"] - tgt["pm7_best_pa_kcalmol"]

    feat_all = pd.read_parquet(data_dir / "features" / "nist1185_features.parquet")
    feat_mol = feat_all.groupby("neutral_smiles").first().reset_index()

    feat_cols = [c for c in feat_mol.columns
                 if c not in NON_FEATURE_COLS
                 and not c.startswith("dft_")
                 and not c.startswith("delta_")
                 and not c.startswith("raw_")]

    df = tgt.merge(feat_mol[["neutral_smiles"] + feat_cols],
                   on="neutral_smiles", how="left", suffixes=("", "_feat"))
    log.info(f"  NIST PM7: {len(df)} molecules, {len(feat_cols)} features")
    return df, feat_cols


def build_nist_dft(df_pm7: pd.DataFrame, pm7_feat_cols: list[str],
                   data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Add DFT features to NIST dataframe."""
    dft_df = load_dft_features(data_dir)
    avail = [c for c in DFT_EXTRA_FEATURES_NIST if c in dft_df.columns]
    df = df_pm7.merge(dft_df[["neutral_smiles"] + avail],
                      on="neutral_smiles", how="left", suffixes=("", "_dft"))
    feat_cols = pm7_feat_cols + avail
    log.info(f"  NIST DFT: {len(df)} molecules, {len(feat_cols)} features")
    return df, feat_cols


def build_kmeans_pm7(data_dir: Path, target_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load k-means dataset merged with PM7 features. Returns (df, feature_cols)."""
    tgt = pd.read_parquet(target_dir / "kmeans251_ml.parquet")
    # Add kcal/mol PA columns for consistent unit handling
    tgt["pm7_pa_kcalmol"]  = tgt["pm7_pa_kjmol"]  * KJMOL_TO_KCAL
    tgt["dft_pa_kcalmol"]  = tgt["dft_pa_kjmol"]  * KJMOL_TO_KCAL
    if "correction_kcalmol" not in tgt.columns:
        tgt["correction_kcalmol"] = tgt["dft_pa_kcalmol"] - tgt["pm7_pa_kcalmol"]

    feat_all = pd.read_parquet(data_dir / "features" / "kmeans251_features.parquet")

    feat_cols = [c for c in feat_all.columns
                 if c not in NON_FEATURE_COLS
                 and not c.startswith("dft_")
                 and not c.startswith("delta_")
                 and not c.startswith("raw_")]

    # k-means features are already site-level — merge on neutral+protonated SMILES
    join_cols = ["neutral_smiles", "protonated_smiles"]
    avail_join = [c for c in join_cols if c in tgt.columns and c in feat_all.columns]
    df = tgt.merge(feat_all[avail_join + feat_cols],
                   on=avail_join, how="left", suffixes=("", "_feat"))
    log.info(f"  k-means PM7: {len(df)} sites, {len(feat_cols)} features")
    return df, feat_cols


def build_kmeans_dft(df_pm7: pd.DataFrame, pm7_feat_cols: list[str],
                     data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Add leak-free DFT features to k-means dataframe."""
    dft_df = load_dft_features(data_dir)
    avail = [c for c in DFT_EXTRA_FEATURES_KMEANS if c in dft_df.columns]
    df = df_pm7.merge(dft_df[["neutral_smiles"] + avail],
                      on="neutral_smiles", how="left", suffixes=("", "_dft"))
    feat_cols = pm7_feat_cols + avail
    log.info(f"  k-means DFT: {len(df)} sites, {len(feat_cols)} features")
    return df, feat_cols


# ---------------------------------------------------------------------------
# Per-run orchestrator
# ---------------------------------------------------------------------------

def run_one(
    label: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    pa_pm7_col: str,
    pa_true_col: str,
    unit_scale: float,
    model_name: str,
    ref_mae: float,
    color: str,
    output_stem: str,
    fractions: list[float],
    seeds: list[int],
    test_frac: float,
) -> pd.DataFrame:
    log.info("=" * 60)
    log.info(f"  {label}")
    log.info("=" * 60)

    lc_df, detail = run_learning_curve(
        df=df, feature_cols=feature_cols,
        target_col=target_col, pa_pm7_col=pa_pm7_col, pa_true_col=pa_true_col,
        model_name=model_name,
        fractions=fractions, seeds=seeds, test_frac=test_frac,
        unit_scale=unit_scale,
    )

    # Save CSV summary
    out_dir = RESULTS_DIR / f"learning_curve_{output_stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lc_df.to_csv(out_dir / "learning_curve_data.csv", index=False)

    # Save full detail JSON (feature names, importances, indices per cell)
    detail_out = out_dir / "learning_curve_detail.json"
    detail_out.write_text(json.dumps({
        "label": label,
        "model": model_name,
        "ref_mae_5fold_cv": ref_mae,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "cells": detail,
    }, indent=2))
    log.info(f"  Saved CSV  → {out_dir / 'learning_curve_data.csv'}")
    log.info(f"  Saved JSON → {detail_out}")

    # Individual plot
    plot_single(
        lc_df,
        title=label,
        output_stem=f"learning_curve_{output_stem}",
        color=color,
        ref_mae=ref_mae,
        ref_label=f"5-fold CV MAE: {ref_mae:.2f} kcal/mol ({model_name})",
    )

    return lc_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all",
                        choices=["all", "nist", "kmeans"])
    args = parser.parse_args()

    run_nist   = args.datasets in ("all", "nist")
    run_kmeans = args.datasets in ("all", "kmeans")

    # ── Placeholders so combined plot always has something ───────────────────
    lc_nist_pm7 = lc_nist_dft = lc_km_pm7 = lc_km_dft = None
    ref_nist_pm7 = ref_nist_dft = ref_km_pm7 = ref_km_dft = None

    # ── NIST ─────────────────────────────────────────────────────────────────
    if run_nist:
        log.info("Building NIST datasets ...")
        df_nist_pm7, nist_pm7_cols = build_nist_pm7(DATA_DIR, TARGET_DIR)
        df_nist_dft, nist_dft_cols = build_nist_dft(df_nist_pm7, nist_pm7_cols, DATA_DIR)

        pm7_model_nist, ref_nist_pm7 = get_best_model(RESULTS_DIR / "nist1155")
        dft_model_nist, ref_nist_dft = get_best_model(RESULTS_DIR / "nist1155_dft")

        lc_nist_pm7 = run_one(
            label        = "NIST — Molecular + PM7 features",
            df           = df_nist_pm7,
            feature_cols = nist_pm7_cols,
            target_col   = "correction_kcalmol",
            pa_pm7_col   = "pm7_best_pa_kcalmol",
            pa_true_col  = "exp_pa_kcalmol",
            unit_scale   = 1.0,      # already in kcal/mol
            model_name   = pm7_model_nist,
            ref_mae      = ref_nist_pm7,
            color        = "#D01C8B",
            output_stem  = "nist_pm7",
            fractions    = FRACTIONS,
            seeds        = SEEDS,
            test_frac    = TEST_FRAC,
        )

        lc_nist_dft = run_one(
            label        = "NIST — Molecular + PM7 + DFT features",
            df           = df_nist_dft,
            feature_cols = nist_dft_cols,
            target_col   = "correction_kcalmol",
            pa_pm7_col   = "pm7_best_pa_kcalmol",
            pa_true_col  = "exp_pa_kcalmol",
            unit_scale   = 1.0,
            model_name   = dft_model_nist,
            ref_mae      = ref_nist_dft,
            color        = "#2166AC",
            output_stem  = "nist_dft",
            fractions    = FRACTIONS,
            seeds        = SEEDS,
            test_frac    = TEST_FRAC,
        )

    # ── k-means ──────────────────────────────────────────────────────────────
    if run_kmeans:
        log.info("Building k-means datasets ...")
        df_km_pm7, km_pm7_cols = build_kmeans_pm7(DATA_DIR, TARGET_DIR)
        df_km_dft, km_dft_cols = build_kmeans_dft(df_km_pm7, km_pm7_cols, DATA_DIR)

        pm7_model_km, ref_km_pm7 = get_best_model(RESULTS_DIR / "kmeans251")
        dft_model_km, ref_km_dft = get_best_model(RESULTS_DIR / "kmeans251_dft")

        lc_km_pm7 = run_one(
            label        = "k-means — Molecular + PM7 features",
            df           = df_km_pm7,
            feature_cols = km_pm7_cols,
            target_col   = "correction_kcalmol",
            pa_pm7_col   = "pm7_pa_kcalmol",
            pa_true_col  = "dft_pa_kcalmol",
            unit_scale   = 1.0,
            model_name   = pm7_model_km,
            ref_mae      = ref_km_pm7,
            color        = "#D01C8B",
            output_stem  = "kmeans_pm7",
            fractions    = FRACTIONS,
            seeds        = SEEDS,
            test_frac    = TEST_FRAC,
        )

        lc_km_dft = run_one(
            label        = "k-means — Molecular + PM7 + DFT features",
            df           = df_km_dft,
            feature_cols = km_dft_cols,
            target_col   = "correction_kcalmol",
            pa_pm7_col   = "pm7_pa_kcalmol",
            pa_true_col  = "dft_pa_kcalmol",
            unit_scale   = 1.0,
            model_name   = dft_model_km,
            ref_mae      = ref_km_dft,
            color        = "#2166AC",
            output_stem  = "kmeans_dft",
            fractions    = FRACTIONS,
            seeds        = SEEDS,
            test_frac    = TEST_FRAC,
        )

    # ── Combined subplot (only if both datasets ran) ─────────────────────────
    if all(x is not None for x in
           [lc_nist_pm7, lc_nist_dft, lc_km_pm7, lc_km_dft]):
        log.info("Generating combined 2x2 subplot ...")
        plot_combined(
            lc_nist_pm7=lc_nist_pm7, lc_nist_dft=lc_nist_dft,
            lc_km_pm7=lc_km_pm7,     lc_km_dft=lc_km_dft,
            ref_nist_pm7=ref_nist_pm7, ref_nist_dft=ref_nist_dft,
            ref_km_pm7=ref_km_pm7,     ref_km_dft=ref_km_dft,
        )

    print(f"\n  Done. Figures in {FIG_PERF}/")
    print(f"  Results in {RESULTS_DIR}/learning_curve_*/")