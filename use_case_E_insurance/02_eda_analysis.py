"""
use_case_E_insurance/02_eda_analysis.py
=========================================
Use Case E — Insurance Risk & Claims Analytics
Phase 2, Step 2: Exploratory Data Analysis (EDA)

Covers all required EDA sections from DSF504 deliverables:
  ✓ Missing values — Porto Seguro encodes -1 as missing
  ✓ Class imbalance analysis (~3.6% claim rate)
  ✓ Feature group analysis (ind / reg / car / calc)
  ✓ Correlation analysis — feature-target Pearson + Spearman
  ✓ Categorical feature distributions
  ✓ Outlier detection on continuous features
  ✓ ps_calc_* uninformativeness demonstration

ML Framework Phase: Data Gathering and Preprocessing → EDA

Run
---
    cd C:\\DSF504
    python use_case_E_insurance/02_eda_analysis.py
"""

from __future__ import annotations

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE

# ── UTF-8 encoding guard ─────────────────────────────────────────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

DATA_SUBDIR = DATA_DIR / "porto_seguro"
REPORT_DIR  = REPORTS_DIR / "use_case_E"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL  = "target"
PALETTE     = {"no_claim": "#1976D2", "claim": "#D32F2F"}
CLAIM_CLR   = PALETTE["claim"]
NO_CLM_CLR  = PALETTE["no_claim"]

sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_train() -> pd.DataFrame:
    """Load training parquet (output of Step 1). Fall back to CSV if needed."""
    parquet_path = DATA_SUBDIR / "train_raw.parquet"
    if parquet_path.exists():
        log.info(f"Loading parquet: {parquet_path}")
        return pd.read_parquet(parquet_path)
    csv_path = DATA_SUBDIR / "train.csv"
    if csv_path.exists():
        log.info(f"Loading CSV: {csv_path}")
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"No training data found in {DATA_SUBDIR}. "
        "Run 01_data_loading.py first."
    )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace -1 with NaN (Porto Seguro missing-value convention)."""
    return df.replace(-1, np.nan)


def _get_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    feat_cols = [c for c in df.columns if c not in ("id", TARGET_COL)]
    groups: dict[str, list[str]] = {
        "ind_bin":  [], "ind_cat":  [], "ind_cont": [],
        "reg":      [],
        "car_bin":  [], "car_cat":  [], "car_cont": [],
        "calc_bin": [], "calc_cont":[],
    }
    for c in feat_cols:
        if c.startswith("ps_ind"):
            if c.endswith("_bin"):   groups["ind_bin"].append(c)
            elif c.endswith("_cat"): groups["ind_cat"].append(c)
            else:                    groups["ind_cont"].append(c)
        elif c.startswith("ps_reg"):
            groups["reg"].append(c)
        elif c.startswith("ps_car"):
            if c.endswith("_bin"):   groups["car_bin"].append(c)
            elif c.endswith("_cat"): groups["car_cat"].append(c)
            else:                    groups["car_cont"].append(c)
        elif c.startswith("ps_calc"):
            if c.endswith("_bin"):   groups["calc_bin"].append(c)
            else:                    groups["calc_cont"].append(c)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# 1. Missing value analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_missing_values(df: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-column missing rate. Porto Seguro uses -1 for missing, so
    we compare raw (with -1) against clean (NaN) to tally true missing counts.

    Returns a DataFrame sorted by missing rate for downstream use in Step 3.
    """
    feat_cols = [c for c in df.columns if c not in ("id", TARGET_COL)]
    rows = []
    for col in feat_cols:
        n_miss = df_clean[col].isna().sum()
        pct    = 100 * n_miss / len(df_clean)
        dtype  = str(df[col].dtype)
        rows.append({
            "feature":    col,
            "missing_n":  int(n_miss),
            "missing_pct": round(pct, 2),
            "dtype":       dtype,
        })

    miss_df = (
        pd.DataFrame(rows)
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )
    miss_df.to_csv(REPORT_DIR / "missing_values.csv", index=False)
    log.info(
        f"Missing values — features with any missing: "
        f"{(miss_df['missing_pct'] > 0).sum()} / {len(miss_df)}"
    )

    # Bar chart of top missing features
    top = miss_df[miss_df["missing_pct"] > 0].head(30)
    if len(top) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
        colors = ["#d32f2f" if v > 20 else "#f57c00" if v > 5 else "#1976D2"
                  for v in top["missing_pct"]]
        ax.barh(top["feature"][::-1], top["missing_pct"][::-1], color=colors[::-1])
        ax.set_xlabel("% Missing  (where −1 treated as missing)")
        ax.set_title("Porto Seguro: Features with Missing Values")
        ax.axvline(x=5, color="orange", linestyle="--", alpha=0.6, label="5%")
        ax.legend()
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "missing_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved → {REPORT_DIR / 'missing_heatmap.png'}")

    return miss_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Class imbalance analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_class_imbalance(df: pd.DataFrame) -> None:
    """
    Detailed class imbalance report and visualisation.
    Motivates Gini metric and SMOTE strategy for Step 3.
    """
    counts = df[TARGET_COL].value_counts().sort_index()
    n_total = len(df)

    print("\n--- Class Imbalance Analysis ---")
    for label, count in counts.items():
        print(f"  {label} ({'Claim' if label == 1 else 'No Claim'}): "
              f"{count:>9,} ({100*count/n_total:.2f}%)")

    ratio = counts.get(0, 0) / counts.get(1, 1)
    print(f"\n  Imbalance ratio (no-claim : claim) = {ratio:.1f} : 1")
    print(
        "\n  Strategy: Normalized Gini Coefficient as primary metric;\n"
        "  SMOTE oversampling applied on training folds only (no leakage).\n"
        "  Alternative: class_weight='balanced' for tree models."
    )

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Bar
    bars = axes[0].bar(
        ["No Claim (0)", "Claim (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=[NO_CLM_CLR, CLAIM_CLR],
    )
    for bar, (_, cnt) in zip(bars, counts.items()):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2000,
            f"{100*cnt/n_total:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    axes[0].set_title("Claim Count Distribution", fontsize=12)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Pie
    axes[1].pie(
        [counts.get(0, 0), counts.get(1, 0)],
        labels=["No Claim", "Claim"],
        autopct="%1.2f%%",
        colors=[NO_CLM_CLR, CLAIM_CLR],
        explode=(0, 0.12),
        startangle=140,
        textprops={"fontsize": 10},
    )
    axes[1].set_title("Class Proportions", fontsize=12)

    # Log-scale bar
    axes[2].bar(
        ["No Claim (0)", "Claim (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=[NO_CLM_CLR, CLAIM_CLR],
        log=True,
    )
    axes[2].set_title("Claim Count (Log Scale)", fontsize=12)
    axes[2].set_ylabel("Count (log)")

    fig.suptitle(
        f"Porto Seguro: Target Class Imbalance\n"
        f"Total: {n_total:,} | Claim rate: {100*counts.get(1,0)/n_total:.3f}%",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "class_imbalance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'class_imbalance.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature-target correlation
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_target_correlation(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson and point-biserial correlation of each feature with the target.
    Porto Seguro features are all numeric, so Pearson is appropriate.
    Saves top-30 correlation chart and full CSV.
    """
    feat_cols = [c for c in df_clean.columns if c not in ("id", TARGET_COL)]
    num_cols  = df_clean[feat_cols].select_dtypes(include="number").columns.tolist()

    rows = []
    for col in num_cols:
        series = df_clean[col].dropna()
        target = df_clean.loc[series.index, TARGET_COL]
        if len(series) < 100 or series.std() < 1e-6:
            continue
        r_p, _ = stats.pearsonr(series, target)
        r_s, _ = stats.spearmanr(series, target)
        rows.append({
            "feature":         col,
            "pearson_r":       round(r_p, 4),
            "spearman_r":      round(r_s, 4),
            "abs_pearson":     abs(r_p),
        })

    corr_df = (
        pd.DataFrame(rows)
        .sort_values("abs_pearson", ascending=False)
        .reset_index(drop=True)
    )
    corr_df.to_csv(REPORT_DIR / "feature_target_correlation.csv", index=False)
    log.info(f"Top correlations:\n{corr_df.head(10).to_string(index=False)}")

    # Top-30 bar chart
    top = corr_df.head(30)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [CLAIM_CLR if v < 0 else "#1976D2" for v in top["pearson_r"]]
    ax.barh(top["feature"][::-1], top["pearson_r"][::-1], color=colors[::-1])
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with target")
    ax.set_title("Top 30 Features by |Pearson r| with 'target'")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "correlation_top30.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'correlation_top30.png'}")

    return corr_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Categorical feature distributions
# ─────────────────────────────────────────────────────────────────────────────

def analyze_categorical_features(
    df: pd.DataFrame, df_clean: pd.DataFrame, max_plots: int = 12
) -> None:
    """
    For each ps_*_cat column, plot claim rate by category level.
    Reveals which categories are high-risk, guiding target encoding in Step 3.
    """
    groups = _get_feature_groups(df)
    cat_cols = groups["ind_cat"] + groups["car_cat"]

    n = min(len(cat_cols), max_plots)
    if n == 0:
        log.info("No categorical columns found.")
        return

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols[:n]):
        ax = axes[i]
        # Compute claim rate per category level (excluding -1)
        sub = df[[col, TARGET_COL]].copy()
        sub[col] = sub[col].replace(-1, np.nan)
        grp = (
            sub.dropna(subset=[col])
            .groupby(col)[TARGET_COL]
            .agg(["mean", "count"])
            .reset_index()
        )
        grp.columns = [col, "claim_rate", "count"]
        grp = grp.sort_values("claim_rate", ascending=False)

        # Use numeric x-values to avoid matplotlib "categorical units" warning
        x_vals = grp[col].astype(int)
        ax.bar(x_vals, grp["claim_rate"], color=CLAIM_CLR, alpha=0.8,
               width=max(0.6, 0.8 / max(len(grp), 1)))
        ax.axhline(y=df[TARGET_COL].mean(), color="navy",
                   linestyle="--", linewidth=1, label="Overall mean")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Category level (int code)")
        ax.set_ylabel("Claim rate")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Claim Rate by Categorical Feature Level", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(
        REPORT_DIR / "categorical_claim_rates.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'categorical_claim_rates.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Outlier detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier detection on continuous features.
    Saves outlier_report.csv — referenced by the dashboard Data Profiling page.
    """
    groups = _get_feature_groups(df_clean)
    cont_cols = groups["ind_cont"] + groups["reg"] + groups["car_cont"]

    rows = []
    for col in cont_cols:
        series = df_clean[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((series < lo) | (series > hi)).sum())
        rows.append({
            "feature":      col,
            "q1":           round(float(q1), 4),
            "q3":           round(float(q3), 4),
            "iqr":          round(float(iqr), 4),
            "lower_fence":  round(float(lo), 4),
            "upper_fence":  round(float(hi), 4),
            "n_outliers":   n_out,
            "pct_outliers": round(100 * n_out / len(series), 2),
        })

    out_df = (
        pd.DataFrame(rows)
        .sort_values("pct_outliers", ascending=False)
        .reset_index(drop=True)
    )
    out_df.to_csv(REPORT_DIR / "outlier_report.csv", index=False)
    log.info(
        f"Outlier report: {len(out_df)} continuous features analysed — "
        f"top outlier: {out_df.iloc[0]['feature']} "
        f"({out_df.iloc[0]['pct_outliers']:.1f}%)"
        if len(out_df) > 0 else "No continuous features found."
    )
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. ps_calc_* uninformativeness demonstration
# ─────────────────────────────────────────────────────────────────────────────

def demonstrate_calc_uninformativeness(df: pd.DataFrame) -> None:
    """
    Show that ps_calc_* features have near-zero correlation with the target.
    Competition analysis confirms these are synthetic / random features.
    Removing them reduces noise and speeds up training.
    """
    groups  = _get_feature_groups(df)
    calc_cols = groups["calc_bin"] + groups["calc_cont"]
    if not calc_cols:
        log.info("No ps_calc_* features found.")
        return

    df_c = df.replace(-1, np.nan)
    corrs = []
    for col in calc_cols:
        series = df_c[col].dropna()
        if len(series) < 100:
            continue
        r, _ = stats.pearsonr(series, df_c.loc[series.index, TARGET_COL])
        corrs.append({"feature": col, "pearson_r": round(r, 4)})

    corr_df = pd.DataFrame(corrs).sort_values("pearson_r", ascending=False)
    log.info(
        f"ps_calc_* max |r|: {corr_df['pearson_r'].abs().max():.4f}  "
        f"(should be close to 0)"
    )

    if len(corr_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(corr_df) * 0.3)))
    colors = [CLAIM_CLR if v < 0 else NO_CLM_CLR for v in corr_df["pearson_r"]]
    ax.barh(corr_df["feature"][::-1], corr_df["pearson_r"][::-1], color=colors[::-1])
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with target")
    ax.set_title(
        "ps_calc_* features: correlation with target\n"
        "(all near-zero → safe to drop in Step 3)"
    )
    plt.tight_layout()
    fig.savefig(
        REPORT_DIR / "calc_features_correlation.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'calc_features_correlation.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Continuous feature distribution comparison (claim vs no-claim)
# ─────────────────────────────────────────────────────────────────────────────

def plot_continuous_distributions(
    df_clean: pd.DataFrame, max_plots: int = 12
) -> None:
    """
    KDE plots of top continuous features split by target label.
    Shows which features best discriminate claim vs non-claim policyholders.
    """
    groups    = _get_feature_groups(df_clean)
    cont_cols = groups["ind_cont"] + groups["reg"] + groups["car_cont"]

    # Pick top N by variance (proxy for signal)
    var_ranked = (
        df_clean[cont_cols].var()
        .sort_values(ascending=False)
        .head(max_plots)
        .index.tolist()
    )

    ncols = 3
    nrows = (len(var_ranked) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    claim    = df_clean[df_clean[TARGET_COL] == 1]
    no_claim = df_clean[df_clean[TARGET_COL] == 0]

    for i, col in enumerate(var_ranked):
        ax = axes[i]
        claim[col].dropna().plot.kde(ax=ax, color=CLAIM_CLR, label="Claim", linewidth=1.5)
        no_claim[col].dropna().plot.kde(ax=ax, color=NO_CLM_CLR, label="No Claim", linewidth=1.5, alpha=0.8)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(len(var_ranked), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Continuous Feature Distributions: Claim vs No-Claim", fontsize=12)
    plt.tight_layout()
    fig.savefig(
        REPORT_DIR / "continuous_distributions.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'continuous_distributions.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case E: Insurance Risk & Claims Analytics")
    print("  Step 2: Exploratory Data Analysis")
    print("=" * 65 + "\n")

    df       = _load_train()
    df_clean = _clean(df)

    print(f"Loaded: {df.shape}  |  claim rate: {df[TARGET_COL].mean():.3%}\n")

    # 1. Missing values
    print("[1] Missing value analysis…")
    analyze_missing_values(df, df_clean)

    # 2. Class imbalance
    print("\n[2] Class imbalance analysis…")
    analyze_class_imbalance(df)

    # 3. Feature-target correlation
    print("\n[3] Feature-target correlation…")
    compute_feature_target_correlation(df_clean)

    # 4. Categorical distributions
    print("\n[4] Categorical feature claim rates…")
    analyze_categorical_features(df, df_clean)

    # 5. Outlier detection
    print("\n[5] Outlier detection…")
    detect_outliers(df_clean)

    # 6. ps_calc_* uninformativeness
    print("\n[6] Demonstrating ps_calc_* uninformativeness…")
    demonstrate_calc_uninformativeness(df)

    # 7. Continuous distributions
    print("\n[7] Continuous feature distributions…")
    plot_continuous_distributions(df_clean)

    print(f"\n[✓] All EDA outputs saved to: {REPORT_DIR}")
    print("\n" + "=" * 65)
    print("  Step 2 complete. Ready for Feature Engineering (03_feature_engineering.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
