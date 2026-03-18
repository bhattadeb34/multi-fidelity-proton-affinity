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

        X_train_full = X_all[train_idx]
        y_train_full = y_all[train_idx]
        n_train_full = len(train_idx)

        for frac in fractions:
            if frac >= 1.0:
                # Use all training data
                X_tr = X_train_full
                y_tr = y_train_full
                n_tr = n_train_full
            else:
                n_tr = max(10, int(frac * n_train_full))
                sub_idx = np.random.default_rng(seed + int(frac * 1000)).choice(
                    n_train_full, size=n_tr, replace=False)
                X_tr = X_train_full[sub_idx]
                y_tr = y_train_full[sub_idx]

            model = build_model(model_name, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_test)

            # PA_pred = PA_PM7 + correction_ML
            pa_pred = pa_pm7_test + y_pred
            mae = mean_absolute_error(pa_true_test, pa_pred) * KJMOL_TO_KCAL

            rows.append({
                "fraction":        frac,
                "n_train":         n_tr,
                "n_train_full":    n_train_full,
                "n_total":         n_total,
                "seed":            seed,
                "mae_test":        mae,
            })
            log.info(f"    frac={frac:.2f}  n_train={n_tr:4d}  "
                     f"seed={seed}  MAE={mae:.3f} kcal/mol")

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
    Plot learning curve: fraction of training data vs test MAE.
    Error bars from repeated seeds.
    """
    # Aggregate across seeds
    summary = (lc_df.groupby("fraction")["mae_test"]
               .agg(mae_mean="mean", mae_std="std")
               .reset_index())
    # Get representative n_train (median across seeds) for each fraction
    n_trains = (lc_df.groupby("fraction")["n_train"]
                .median().round().astype(int).reset_index())
    summary = summary.merge(n_trains, on="fraction")

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

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(
        summary["fraction"] * 100,   # convert to percentage
        summary["mae_mean"],
        yerr=summary["mae_std"],
        fmt="o-",
        color=color,
        linewidth=2.0,
        markersize=8,
        capsize=5,
        capthick=1.8,
        elinewidth=1.8,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
        label=f"Test MAE (n={n_total} total)",
    )

    # Reference line (e.g. Jin & Merz or full-data MAE)
    if ref_mae is not None:
        ax.axhline(ref_mae, color="gray", linewidth=1.5, linestyle="--",
                   zorder=1, label=ref_label or f"Reference: {ref_mae:.2f} kcal/mol")

    # Annotate last point
    last = summary.iloc[-1]
    ax.annotate(
        f"Full data\n{last['mae_mean']:.2f} kcal/mol",
        xy=(last["fraction"] * 100, last["mae_mean"]),
        xytext=(-55, 18),
        textcoords="offset points",
        fontsize=TICK_SIZE - 3,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        color="black",
    )

    # Secondary x-axis showing absolute n_train
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_fracs   = summary["fraction"].values
    tick_pcts    = tick_fracs * 100
    tick_ntrains = summary["n_train"].values
    ax2.set_xticks(tick_pcts)
    ax2.set_xticklabels([str(n) for n in tick_ntrains],
                         rotation=45, ha="left", fontsize=TICK_SIZE - 4)
    ax2.set_xlabel("Training set size (N)", fontsize=LABEL_SIZE - 4, labelpad=8)

    ax.set_xlabel("Training set size (% of total, N=1155)")
    ax.set_ylabel("Test set MAE (kcal/mol)")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linewidth=0.5, alpha=0.4, linestyle="--")
    ax.grid(axis="x", linewidth=0.3, alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="lightgray")

    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{output_stem}.{ext}"
        fig.savefig(out)
        log.info(f"  Saved {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Learning curve analysis for NIST 1155 dataset.")
    parser.add_argument("--no-dft", action="store_true",
                        help="Skip DFT-augmented learning curve")
    parser.add_argument("--model", default=None,
                        help="Override best model (e.g. ExtraTrees)")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir_pm7 = RESULTS_DIR / "learning_curve_nist_pm7"
    out_dir_dft = RESULTS_DIR / "learning_curve_nist_dft"
    out_dir_pm7.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ PM7
    log.info("=" * 55)
    log.info("  Learning curve — PM7-only features")
    log.info("=" * 55)

    log.info("Loading NIST target data ...")
    df_nist = pd.read_parquet(TARGET_DIR / "nist1155_ml.parquet")
    n_total = len(df_nist)
    log.info(f"  Total molecules: {n_total}")

    log.info("Recovering optimal PM7 feature set ...")
    pm7_features, best_model_pm7 = get_optimal_features_pm7(RESULTS_DIR)
    if args.model:
        best_model_pm7 = args.model

    log.info(f"Running PM7 learning curve ({len(FRACTIONS)} fractions × {len(SEEDS)} seeds) ...")
    lc_pm7 = run_learning_curve(
        df           = df_nist,
        feature_cols = pm7_features,
        target_col   = "delta_pm7_exp",
        pa_pm7_col   = "pm7_best_pa_kjmol",
        pa_true_col  = "exp_pa_kjmol",
        model_name   = best_model_pm7,
        fractions    = FRACTIONS,
        seeds        = SEEDS,
        test_frac    = TEST_FRAC,
    )
    lc_pm7.to_csv(out_dir_pm7 / "learning_curve_data.csv", index=False)
    log.info(f"  Saved data → {out_dir_pm7}/learning_curve_data.csv")

    # Full-data MAE from saved results (for reference line)
    mae_pm7_path = RESULTS_DIR / "nist1155" / "mae_summary.csv"
    ref_mae_pm7  = None
    if mae_pm7_path.exists():
        mae_df = pd.read_csv(mae_pm7_path)
        best_row = mae_df.sort_values("mae_delta_mean").iloc[0]
        ref_mae_pm7 = best_row["mae_delta_mean"]

    plot_learning_curve(
        lc_df       = lc_pm7,
        title       = "Learning curve — NIST 1155 (PM7 features)",
        n_total     = n_total,
        output_stem = "learning_curve_nist_pm7",
        color       = "#2166AC",
        ref_mae     = ref_mae_pm7,
        ref_label   = f"Full CV MAE: {ref_mae_pm7:.2f} kcal/mol ({best_model_pm7})" if ref_mae_pm7 else None,
    )

    # ------------------------------------------------------------------ DFT
    if args.no_dft:
        log.info("Skipping DFT learning curve (--no-dft)")
        print(f"\n  PM7 learning curve saved to: {FIG_DIR}/learning_curve_nist_pm7.*")
        return

    log.info("=" * 55)
    log.info("  Learning curve — PM7+DFT features")
    log.info("=" * 55)

    # Load DFT-augmented dataset
    try:
        from train_models_dft import load_dft_features, augment_with_dft, DFT_EXTRA_FEATURES
        dft_df   = load_dft_features()
        df_nist_dft = augment_with_dft(df_nist.copy(), dft_df,
                                        join_cols=["neutral_smiles"])
    except Exception as e:
        log.warning(f"Could not load DFT features: {e}")
        log.warning("Skipping DFT learning curve")
        print(f"\n  PM7 learning curve saved to: {FIG_DIR}/learning_curve_nist_pm7.*")
        return

    log.info("Recovering optimal PM7+DFT feature set ...")
    try:
        dft_features, best_model_dft = get_optimal_features_dft(RESULTS_DIR)
    except FileNotFoundError as e:
        log.warning(str(e))
        log.warning("Using PM7 feature set + all DFT extra features as fallback")
        dft_features   = pm7_features + DFT_EXTRA_FEATURES
        best_model_dft = best_model_pm7

    if args.model:
        best_model_dft = args.model

    out_dir_dft.mkdir(parents=True, exist_ok=True)
    log.info(f"Running DFT learning curve ({len(FRACTIONS)} fractions × {len(SEEDS)} seeds) ...")
    lc_dft = run_learning_curve(
        df           = df_nist_dft,
        feature_cols = dft_features,
        target_col   = "delta_pm7_exp",
        pa_pm7_col   = "pm7_best_pa_kjmol",
        pa_true_col  = "exp_pa_kjmol",
        model_name   = best_model_dft,
        fractions    = FRACTIONS,
        seeds        = SEEDS,
        test_frac    = TEST_FRAC,
    )
    lc_dft.to_csv(out_dir_dft / "learning_curve_data.csv", index=False)
    log.info(f"  Saved data → {out_dir_dft}/learning_curve_data.csv")

    mae_dft_path = RESULTS_DIR / "nist1155_dft" / "mae_summary.csv"
    ref_mae_dft  = None
    if mae_dft_path.exists():
        mae_df = pd.read_csv(mae_dft_path)
        best_row = mae_df.sort_values("mae_delta_mean").iloc[0]
        ref_mae_dft = best_row["mae_delta_mean"]

    plot_learning_curve(
        lc_df       = lc_dft,
        title       = "Learning curve — NIST 1155 (PM7+DFT features)",
        n_total     = n_total,
        output_stem = "learning_curve_nist_dft",
        color       = "#D01C8B",
        ref_mae     = ref_mae_dft,
        ref_label   = f"Full CV MAE: {ref_mae_dft:.2f} kcal/mol ({best_model_dft})" if ref_mae_dft else None,
    )

    # ---- Overlay plot: PM7 vs DFT on same axes ----
    log.info("Generating overlay comparison plot ...")

    summary_pm7 = (lc_pm7.groupby("fraction")["mae_test"]
                   .agg(mae_mean="mean", mae_std="std").reset_index())
    summary_dft = (lc_dft.groupby("fraction")["mae_test"]
                   .agg(mae_mean="mean", mae_std="std").reset_index())

    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300,
                          "savefig.bbox": "tight",
                          "axes.linewidth": SPINE_LW,
                          "xtick.labelsize": TICK_SIZE,
                          "ytick.labelsize": TICK_SIZE,
                          "axes.labelsize":  LABEL_SIZE,
                          "legend.fontsize": 18})

    fig, ax = plt.subplots(figsize=(9, 6))

    for summary, color, label in [
        (summary_pm7, "#2166AC", "Without DFT features"),
        (summary_dft, "#D01C8B", "With DFT features"),
    ]:
        ax.errorbar(
            summary["fraction"] * 100,
            summary["mae_mean"],
            yerr=summary["mae_std"],
            fmt="o-", color=color,
            linewidth=2.0, markersize=8,
            capsize=5, capthick=1.8, elinewidth=1.8,
            markeredgecolor="white", markeredgewidth=0.8,
            zorder=3, label=label,
        )

    ax.set_xlabel("Training set size (% of total, N=1155)")
    ax.set_ylabel("Test set MAE (kcal/mol)")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linewidth=0.5, alpha=0.4, linestyle="--")
    ax.grid(axis="x", linewidth=0.3, alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="lightgray")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"learning_curve_nist_comparison.{ext}"
        fig.savefig(out)
        log.info(f"  Saved {out}")
    plt.close(fig)

    print(f"\n  All learning curve figures saved to: {FIG_DIR}/")
    print(f"  Files:")
    print(f"    learning_curve_nist_pm7.*          — PM7 features only")
    print(f"    learning_curve_nist_dft.*          — PM7+DFT features")
    print(f"    learning_curve_nist_comparison.*   — overlay comparison")


if __name__ == "__main__":
    main()
