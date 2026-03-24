"""
learning_curve.py
=================
Post-hoc learning curve analysis for the NIST 1155 dataset.

For each training fraction, we:
  1. Use the optimal feature set already identified from the full CV run
     (loaded from results/nist1155/feature_importance.csv — top features
     that appeared across folds, filtered by the saved cv_results.json)
  2. Train the best model (ExtraTrees by default, or whichever won)
  3. Evaluate on a fixed held-out test set (20% stratified split)
  4. Repeat with 5 different random seeds to get error bars

Fractions tested: 0.08, 0.16, 0.24, 0.32, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
(approximately 10 evenly spaced points, last point = full training set)

Two runs:
  - PM7-only features   → results/learning_curve_nist_pm7/
  - PM7+DFT features    → results/learning_curve_nist_dft/

Output files (per run):
  learning_curve_data.csv   — fraction, n_train, mae_mean, mae_std per seed
  learning_curve.pdf/.png   — publication-quality plot

Usage
-----
  python scripts/learning_curve.py
  python scripts/learning_curve.py --no-dft    # skip DFT run
  python scripts/learning_curve.py --model ExtraTrees
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR    = Path(__file__).parent
PROJECT_DIR   = SCRIPT_DIR.parent
DATA_DIR      = PROJECT_DIR / "data"
TARGET_DIR    = DATA_DIR / "targets"
RESULTS_DIR   = PROJECT_DIR / "results"
FIG_DIR       = PROJECT_DIR / "figures"
FIG_PERF    = FIG_DIR / "model_performance"

KJMOL_TO_KCAL = 1 / 4.184

# Training fractions to evaluate
FRACTIONS = [0.08, 0.16, 0.24, 0.32, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

# Random seeds for repeated subsampling (gives error bars)
SEEDS = [42, 123, 456, 789, 1024]

# Test set fraction (held out from the full dataset)
TEST_FRAC = 0.20

# Style
TICK_SIZE  = 20
LABEL_SIZE = 24
SPINE_LW   = 1.5


# ---------------------------------------------------------------------------
# Feature set recovery from saved results
# ---------------------------------------------------------------------------

def get_optimal_features_pm7(results_dir: Path) -> list[str]:
    """
    Recover the optimal PM7-only feature set from saved cv_results.
    Uses features that appeared in at least 3 out of 5 folds
    (i.e. consistently selected by the Lasso 1-SE pipeline).

    Falls back to top-N features by mean importance from feature_importance.csv
    if per-fold feature lists aren't available.
    """
    feat_imp_path = results_dir / "nist1155" / "feature_importance.csv"
    cv_path       = results_dir / "nist1155" / "cv_results.json"

    if not feat_imp_path.exists():
        raise FileNotFoundError(
            f"feature_importance.csv not found at {feat_imp_path}\n"
            f"Run train_models.py first.")

    feat_imp = pd.read_csv(feat_imp_path)

    # Get best model name from cv_results
    best_model = "ExtraTrees"
    if cv_path.exists():
        cv = json.loads(cv_path.read_text())
        best_model = min(cv["models"],
                         key=lambda m: cv["models"][m].get("mae_delta_mean") or 999)
    log.info(f"  Best model (PM7): {best_model}")

    # Features for best model, sorted by importance
    df_m = feat_imp[feat_imp["model"] == best_model].sort_values(
        "importance", ascending=False)

    if len(df_m) == 0:
        # Fallback: use any tree model available
        df_m = feat_imp.groupby("feature")["importance"].mean().reset_index()
        df_m = df_m.sort_values("importance", ascending=False)

    # Use the mean number of features selected across folds
    if cv_path.exists():
        cv = json.loads(cv_path.read_text())
        n_feat = int(round(cv["models"][best_model].get("n_features_mean", 61)))
    else:
        n_feat = min(61, len(df_m))

    features = df_m["feature"].tolist()[:n_feat]
    log.info(f"  PM7 feature set: {len(features)} features")
    return features, best_model


def get_optimal_features_dft(results_dir: Path) -> list[str]:
    """Same as above but for the PM7+DFT run (nist1155_dft)."""
    feat_imp_path = results_dir / "nist1155_dft" / "feature_importance.csv"
    cv_path       = results_dir / "nist1155_dft" / "cv_results.json"

    if not feat_imp_path.exists():
        raise FileNotFoundError(
            f"feature_importance.csv not found at {feat_imp_path}\n"
            f"Run train_models_dft.py first.")

    feat_imp  = pd.read_csv(feat_imp_path)
    best_model = "ExtraTrees"
    if cv_path.exists():
        cv = json.loads(cv_path.read_text())
        best_model = min(cv["models"],
                         key=lambda m: cv["models"][m].get("mae_delta_mean") or 999)
    log.info(f"  Best model (DFT): {best_model}")

    df_m = feat_imp[feat_imp["model"] == best_model].sort_values(
        "importance", ascending=False)

    if cv_path.exists():
        cv = json.loads(cv_path.read_text())
        n_feat = int(round(cv["models"][best_model].get("n_features_mean", 61)))
    else:
        n_feat = min(61, len(df_m))

    features = df_m["feature"].tolist()[:n_feat]
    log.info(f"  DFT feature set: {len(features)} features")
    return features, best_model


# ---------------------------------------------------------------------------
# Learning curve computation
# ---------------------------------------------------------------------------

def build_model(model_name: str, seed: int):
    """Instantiate the best model with a given seed."""
    if model_name in ("ExtraTrees", "ExtraTreesRegressor"):
        return ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    elif model_name in ("RandomForest", "RandomForestRegressor"):
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    else:
        # Default to ExtraTrees for any other model name
        log.warning(f"  Model {model_name} not directly supported for LC; using ExtraTrees")
        return ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)


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
) -> pd.DataFrame:
    """
    Run learning curve analysis.

    For each fraction × seed:
      - Hold out test_frac of full data as fixed test set
      - Sample `fraction` of remaining training data
      - Train model on optimal feature set
      - Evaluate MAE on test set (in kcal/mol)

    Returns DataFrame with columns:
      fraction, n_train, seed, mae_test
    """
    # Filter to available features
    avail_cols = [c for c in feature_cols if c in df.columns]
    missing    = len(feature_cols) - len(avail_cols)
    if missing > 0:
        log.warning(f"  {missing} feature columns not found in dataset — skipping")
    log.info(f"  Using {len(avail_cols)} features")

    X_all       = df[avail_cols].values.astype(np.float64)
    y_all       = df[target_col].values.astype(np.float64)
    pa_pm7_all  = df[pa_pm7_col].values
    pa_true_all = df[pa_true_col].values
    n_total     = len(df)

    # Impute NaN with column median
    col_medians = np.nanmedian(X_all, axis=0)
    nan_mask    = np.isnan(X_all)
    X_all[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    rows = []

    for seed in seeds:
        # Fixed train/test split for this seed
        idx = np.arange(n_total)
        train_idx, test_idx = train_test_split(
            idx, test_size=test_frac, random_state=seed)

        X_test       = X_all[test_idx]
        y_test       = y_all[test_idx]
        pa_pm7_test  = pa_pm7_all[test_idx]
        pa_true_test = pa_true_all[test_idx]

        X_train_full    = X_all[train_idx]
        y_train_full    = y_all[train_idx]
        pa_pm7_train_full  = pa_pm7_all[train_idx]
        pa_true_train_full = pa_true_all[train_idx]
        n_train_full    = len(train_idx)

        for frac in fractions:
            if frac >= 1.0:
                X_tr        = X_train_full
                y_tr        = y_train_full
                pa_pm7_tr   = pa_pm7_train_full
                pa_true_tr  = pa_true_train_full
                n_tr        = n_train_full
            else:
                n_tr    = max(10, int(frac * n_train_full))
                sub_idx = np.random.default_rng(seed + int(frac * 1000)).choice(
                    n_train_full, size=n_tr, replace=False)
                X_tr       = X_train_full[sub_idx]
                y_tr       = y_train_full[sub_idx]
                pa_pm7_tr  = pa_pm7_train_full[sub_idx]
                pa_true_tr = pa_true_train_full[sub_idx]

            model = build_model(model_name, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)
                y_pred_test  = model.predict(X_test)
                y_pred_train = model.predict(X_tr)

            # Test MAE: PA_pred = PA_PM7 + correction_ML
            pa_pred_test  = pa_pm7_test + y_pred_test
            mae_test  = mean_absolute_error(pa_true_test,  pa_pred_test)  * KJMOL_TO_KCAL

            # Train MAE
            pa_pred_train = pa_pm7_tr + y_pred_train
            mae_train = mean_absolute_error(pa_true_tr, pa_pred_train) * KJMOL_TO_KCAL

            rows.append({
                "fraction":        frac,
                "n_train":         n_tr,
                "n_train_full":    n_train_full,
                "n_total":         n_total,
                "seed":            seed,
                "mae_test":        mae_test,
                "mae_train":       mae_train,
            })
            log.info(f"    frac={frac:.2f}  n_train={n_tr:4d}  "
                     f"seed={seed}  MAE={mae_test:.3f} kcal/mol")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curve(
    lc_df: pd.DataFrame,
    title: str,
    n_total: int,
    output_stem: str,
    color: str = "#D01C8B",
    ref_mae: float | None = None,
    ref_label: str | None = None,
):
    """
    Plot learning curve: training set size vs test MAE and train MAE.
    Error bars from repeated seeds.
    """
    # Aggregate test MAE across seeds
    summary = (lc_df.groupby("fraction")["mae_test"]
               .agg(mae_mean="mean", mae_std="std")
               .reset_index())
    # Aggregate train MAE across seeds
    summary_train = (lc_df.groupby("fraction")["mae_train"]
                     .agg(train_mean="mean", train_std="std")
                     .reset_index())
    # Representative n_train per fraction
    n_trains = (lc_df.groupby("fraction")["n_train"]
                .median().round().astype(int).reset_index())
    summary = summary.merge(n_trains,       on="fraction")
    summary = summary.merge(summary_train,  on="fraction")

    plt.rcParams.update({
        "axes.linewidth":    SPINE_LW,
        "xtick.major.width": SPINE_LW,
        "ytick.major.width": SPINE_LW,
        "xtick.labelsize":   TICK_SIZE,
        "ytick.labelsize":   TICK_SIZE,
        "axes.labelsize":    LABEL_SIZE,
        "legend.fontsize":   18,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    pct = summary["fraction"] * 100

    # Test MAE
    ax.errorbar(
        pct, summary["mae_mean"],
        yerr=summary["mae_std"],
        fmt="o-", color=color,
        linewidth=2.2, markersize=8,
        capsize=5, capthick=1.8, elinewidth=1.8,
        markeredgecolor="white", markeredgewidth=0.8,
        zorder=3, label=f"Test MAE",
    )

    # Train MAE — lighter shade of same color
    train_color = "#888888"
    ax.errorbar(
        pct, summary["train_mean"],
        yerr=summary["train_std"],
        fmt="s--", color=train_color,
        linewidth=2.0, markersize=7,
        capsize=4, capthick=1.5, elinewidth=1.5,
        markeredgecolor="white", markeredgewidth=0.8,
        zorder=2, label=f"Train MAE",
    )

    # Reference line
    if ref_mae is not None:
        ax.axhline(ref_mae, color="black", linewidth=1.5, linestyle=":",
                   zorder=1, label=ref_label or f"Reference: {ref_mae:.2f} kcal/mol")

    # Annotate last test point
    last = summary.iloc[-1]
    ax.annotate(
        f"Full data\n{last['mae_mean']:.2f} kcal/mol",
        xy=(last["fraction"] * 100, last["mae_mean"]),
        xytext=(-60, 18),
        textcoords="offset points",
        fontsize=TICK_SIZE - 3,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        color=color, fontweight="bold",
    )

    # Secondary x-axis: absolute n_train
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_pcts    = summary["fraction"].values * 100
    tick_ntrains = summary["n_train"].values
    ax2.set_xticks(tick_pcts)
    ax2.set_xticklabels([str(n) for n in tick_ntrains],
                         rotation=45, ha="left", fontsize=TICK_SIZE - 4)
    ax2.set_xlabel("Number of training molecules", fontsize=LABEL_SIZE - 4,
                   labelpad=10)

    ax.set_xlabel("Training set size (%)", fontsize=LABEL_SIZE)
    ax.set_ylabel("MAE (kcal/mol)", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=28, fontweight="bold")
    ax.legend(framealpha=0.9, edgecolor="lightgray", loc="upper right")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linewidth=0.5, alpha=0.35, linestyle="--")

    fig.tight_layout()
    FIG_PERF.mkdir(parents=True, exist_ok=True)
    out = FIG_PERF / f"{output_stem}.pdf"
    fig.savefig(out)
    log.info(f"    Saved {out}")
    plt.close(fig)

def plot_comparison(lc_pm7: pd.DataFrame, lc_dft: pd.DataFrame,
                    n_total: int, output_stem: str):
    """Overlay PM7-only vs PM7+DFT learning curves (test MAE + train MAE)."""
    def aggregate(df):
        test  = df.groupby("fraction")["mae_test"].agg(
            mae_mean="mean", mae_std="std").reset_index()
        train = df.groupby("fraction")["mae_train"].agg(
            train_mean="mean", train_std="std").reset_index()
        n_tr  = df.groupby("fraction")["n_train"].median(
            ).round().astype(int).reset_index()
        return test.merge(train, on="fraction").merge(n_tr, on="fraction")

    s_pm7 = aggregate(lc_pm7)
    s_dft = aggregate(lc_dft)

    plt.rcParams.update({
        "axes.linewidth":    SPINE_LW,
        "xtick.major.width": SPINE_LW,
        "ytick.major.width": SPINE_LW,
        "xtick.labelsize":   TICK_SIZE,
        "ytick.labelsize":   TICK_SIZE,
        "axes.labelsize":    LABEL_SIZE,
        "legend.fontsize":   18,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    pct = s_pm7["fraction"] * 100

    COLOR_PM7 = "#D01C8B"
    COLOR_DFT = "#2166AC"

    # Test MAE — solid lines
    for s, color, label in [
        (s_pm7, COLOR_PM7, "PM7 features — Test"),
        (s_dft, COLOR_DFT, "PM7+DFT features — Test"),
    ]:
        ax.errorbar(pct, s["mae_mean"], yerr=s["mae_std"],
                    fmt="o-", color=color, linewidth=2.2, markersize=8,
                    capsize=5, capthick=1.8, elinewidth=1.8,
                    markeredgecolor="white", markeredgewidth=0.8,
                    zorder=3, label=label)

    # Train MAE — dashed lines
    for s, color, label in [
        (s_pm7, COLOR_PM7, "PM7 features — Train"),
        (s_dft, COLOR_DFT, "PM7+DFT features — Train"),
    ]:
        ax.errorbar(pct, s["train_mean"], yerr=s["train_std"],
                    fmt="s--", color=color, linewidth=1.8, markersize=6,
                    capsize=4, capthick=1.5, elinewidth=1.5,
                    markeredgecolor="white", markeredgewidth=0.8,
                    zorder=2, label=label, alpha=0.7)

    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_pcts    = s_pm7["fraction"].values * 100
    tick_ntrains = s_pm7["n_train"].values
    ax2.set_xticks(tick_pcts)
    ax2.set_xticklabels([str(n) for n in tick_ntrains],
                         rotation=45, ha="left", fontsize=TICK_SIZE - 4)
    ax2.set_xlabel("Number of training molecules", fontsize=LABEL_SIZE - 4,
                   labelpad=10)

    ax.set_xlabel("Training set size (%)", fontsize=LABEL_SIZE)
    ax.set_ylabel("MAE (kcal/mol)", fontsize=LABEL_SIZE)
    ax.set_title("Learning curve: PM7 vs PM7+DFT features (NIST)",
                 fontsize=TITLE_SIZE, pad=28, fontweight="bold")
    ax.legend(framealpha=0.9, edgecolor="lightgray", loc="upper right",
              fontsize=LEGEND_SIZE - 3, ncol=2)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linewidth=0.5, alpha=0.35, linestyle="--")

    fig.tight_layout()
    FIG_PERF.mkdir(parents=True, exist_ok=True)
    out = FIG_PERF / f"{output_stem}.pdf"
    fig.savefig(out)
    log.info(f"    Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    FRACTIONS  = [0.08, 0.16, 0.24, 0.32, 0.40, 0.50,
                  0.60, 0.70, 0.80, 0.90, 1.00]
    SEEDS      = [42, 123, 456, 789, 1024]
    TEST_FRAC  = 0.20

    # ── Load NIST target data ────────────────────────────────────────────────
    log.info("  Loading NIST target data ...")
    tgt = pd.read_parquet(DATA_DIR / "targets" / "nist1155_ml.parquet")
    log.info(f"    Total molecules: {len(tgt)}")

    PA_PM7_COL  = "pm7_best_pa_kcalmol"
    PA_TRUE_COL = "exp_pa_kcalmol"
    if "correction_kcalmol" not in tgt.columns:
        tgt["correction_kcalmol"] = tgt[PA_TRUE_COL] - tgt[PA_PM7_COL]
    TARGET_COL = "correction_kcalmol"

    # ── PM7-only learning curve ──────────────────────────────────────────────
    log.info("=" * 55)
    log.info("  Learning curve — PM7-only features")
    log.info("=" * 55)

    pm7_results_dir = RESULTS_DIR
    pm7_features, pm7_model = get_optimal_features_pm7(pm7_results_dir)

    # Merge PM7 features into target
    feat_all = pd.read_parquet(DATA_DIR / "features" / "nist1185_features.parquet")
    feat_mol = feat_all.groupby("neutral_smiles").first().reset_index()
    smiles_col = "neutral_smiles" if "neutral_smiles" in tgt.columns else "smiles"
    avail_pm7  = [c for c in pm7_features if c in feat_mol.columns]
    tgt_pm7    = tgt.merge(feat_mol[["neutral_smiles"] + avail_pm7],
                            left_on=smiles_col, right_on="neutral_smiles",
                            how="left", suffixes=("", "_feat"))

    lc_pm7_df = run_learning_curve(
        df           = tgt_pm7,
        feature_cols = avail_pm7,
        target_col   = TARGET_COL,
        pa_pm7_col   = PA_PM7_COL,
        pa_true_col  = PA_TRUE_COL,
        model_name   = pm7_model,
        fractions    = FRACTIONS,
        seeds        = SEEDS,
        test_frac    = TEST_FRAC,
    )

    out_pm7 = RESULTS_DIR / "learning_curve_nist_pm7" / "learning_curve_data.csv"
    out_pm7.parent.mkdir(parents=True, exist_ok=True)
    lc_pm7_df.to_csv(out_pm7, index=False)
    log.info(f"    Saved data → {out_pm7}")

    n_total     = int(lc_pm7_df["n_total"].iloc[0])
    pm7_full_mae = float(lc_pm7_df[lc_pm7_df["fraction"] >= 1.0]["mae_test"].mean())

    plot_learning_curve(
        lc_pm7_df,
        title       = "Learning curve — PM7 features (NIST)",
        n_total     = n_total,
        output_stem = "learning_curve_nist_pm7",
        color       = "#D01C8B",
    )

    # ── PM7+DFT learning curve ───────────────────────────────────────────────
    log.info("=" * 55)
    log.info("  Learning curve — PM7+DFT features")
    log.info("=" * 55)

    dft_results_dir = RESULTS_DIR / "nist1155_dft"
    dft_features, dft_model = get_optimal_features_dft(dft_results_dir)

    avail_dft = [c for c in dft_features if c in feat_mol.columns]
    tgt_dft   = tgt.merge(feat_mol[["neutral_smiles"] + avail_dft],
                           left_on=smiles_col, right_on="neutral_smiles",
                           how="left", suffixes=("", "_feat"))

    lc_dft_df = run_learning_curve(
        df           = tgt_dft,
        feature_cols = avail_dft,
        target_col   = TARGET_COL,
        pa_pm7_col   = PA_PM7_COL,
        pa_true_col  = PA_TRUE_COL,
        model_name   = dft_model,
        fractions    = FRACTIONS,
        seeds        = SEEDS,
        test_frac    = TEST_FRAC,
    )

    out_dft = RESULTS_DIR / "learning_curve_nist_dft" / "learning_curve_data.csv"
    out_dft.parent.mkdir(parents=True, exist_ok=True)
    lc_dft_df.to_csv(out_dft, index=False)
    log.info(f"    Saved data → {out_dft}")

    dft_full_mae = float(lc_dft_df[lc_dft_df["fraction"] >= 1.0]["mae_test"].mean())

    plot_learning_curve(
        lc_dft_df,
        title       = "Learning curve — PM7+DFT features (NIST)",
        n_total     = n_total,
        output_stem = "learning_curve_nist_dft",
        color       = "#2166AC",
    )

    # ── Overlay comparison ───────────────────────────────────────────────────
    log.info("  Generating overlay comparison plot ...")
    plot_comparison(lc_pm7_df, lc_dft_df, n_total,
                    output_stem="learning_curve_nist_comparison")

    print(f"\n  All learning curve figures saved to: {FIG_PERF}/")
    print(f"  Files:")
    print(f"    learning_curve_nist_pm7.*          — PM7 features only")
    print(f"    learning_curve_nist_dft.*          — PM7+DFT features")
    print(f"    learning_curve_nist_comparison.*   — overlay comparison")