"""
use_case_G_advisory/02_eda_analysis.py
========================================
Use Case G — AmEx Credit Default Prediction
Phase 2, Step 2: EDA & Data Understanding

Analyses:
  1. Target distribution & default rate by statement count
  2. Feature group statistics (D/S/P/B/R prefixes)
  3. Time-series trends — defaulters vs non-defaulters over 13 months
  4. Missing value patterns (many AmEx features have structural missingness)
  5. Feature-target correlation (point-biserial for numeric, last statement)
  6. Categorical feature analysis (D_63, D_64)
  7. Outlier detection (delinquency counts, balance extremes)

Run
---
    cd C:\\DSF504
    python use_case_G_advisory/02_eda_analysis.py
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
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "amex_default"
REPORT_DIR  = REPORTS_DIR / "use_case_G"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
ID_COL     = "customer_ID"
CAT_COLS   = ["D_63", "D_64"]
NUM_PREFIXES = {"D": "Delinquency", "S": "Spend", "P": "Payment",
                "B": "Balance", "R": "Risk"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train splits (created in Step 1)."""
    # Try real split, fall back to synthetic
    for suffix in ("", "_synthetic"):
        tp = DATA_SUBDIR / f"train_raw{suffix}.parquet"
        lp = DATA_SUBDIR / f"train_labels_raw{suffix}.parquet"
        if not tp.exists():
            tp = DATA_SUBDIR / f"train_data{suffix}.parquet"
            lp = DATA_SUBDIR / f"train_labels{suffix}.parquet"
        if tp.exists() and lp.exists():
            return pd.read_parquet(tp), pd.read_parquet(lp)

    raise FileNotFoundError(
        "No training data found. Run 01_data_loading.py first."
    )


def _get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in (ID_COL, "S_2") + tuple(CAT_COLS)
            and pd.api.types.is_numeric_dtype(df[c])]


def _get_last_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent statement row per customer (last in time-series)."""
    if "S_2" in df.columns:
        return df.sort_values("S_2").groupby(ID_COL).last().reset_index()
    else:
        return df.groupby(ID_COL).last().reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Target distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_default_by_stmt_count(
    df_train: pd.DataFrame,
    df_labels: pd.DataFrame,
    save: bool = True,
) -> None:
    """
    Default rate by number of statements.
    Customers with fewer statements (new/young accounts) tend to have
    higher default rates — partial observation window means more uncertainty.
    """
    stmt_counts = df_train.groupby(ID_COL).size().reset_index(name="n_stmts")
    merged = stmt_counts.merge(df_labels, on=ID_COL)

    default_by_stmts = (
        merged.groupby("n_stmts")[TARGET_COL]
        .agg(["mean", "count"])
        .reset_index()
    )
    default_by_stmts.columns = ["n_stmts", "default_rate", "n_customers"]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    color_bar  = "#3949AB"
    color_line = "#D32F2F"

    bars = ax1.bar(
        default_by_stmts["n_stmts"],
        default_by_stmts["default_rate"] * 100,
        color=color_bar, alpha=0.8, label="Default Rate (%)"
    )
    ax1.set_xlabel("Number of Monthly Statements")
    ax1.set_ylabel("Default Rate (%)", color=color_bar)
    ax1.set_xticks(range(1, 14))
    ax1.axhline(y=25.9, color=color_line, linestyle="--", linewidth=1.5,
                label="Overall Default Rate (25.9%)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(default_by_stmts["n_stmts"], default_by_stmts["n_customers"],
             color=color_line, marker="o", linewidth=2, label="Customer Count")
    ax2.set_ylabel("Number of Customers", color=color_line)
    ax2.legend(loc="upper right")

    plt.title(
        "Default Rate vs Statement Count\n"
        "Fewer statements → shorter history → higher uncertainty"
    )
    plt.tight_layout()
    if save:
        p = REPORT_DIR / "default_by_stmt_count.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Time-series trends: defaulters vs non-defaulters
# ─────────────────────────────────────────────────────────────────────────────

def plot_time_series_trends(
    df_train: pd.DataFrame,
    df_labels: pd.DataFrame,
    features: list[str] | None = None,
    save: bool = True,
) -> None:
    """
    Line plots of key features over the 13-month statement window,
    split by default status. This reveals the temporal patterns that
    top competition solutions exploit.

    Key finding from competition:
    - B_1 (balance): defaulters show rising trend over time
    - P_2 (payment): defaulters show declining payments before default
    - D_39 (delinquency days): defaulters accumulate delinquency rapidly
    - R_1 (risk score): defaulters have persistently higher risk scores
    """
    num_cols = _get_numeric_cols(df_train)
    if features is None:
        # Pick one representative feature per group
        features = []
        for prefix in ["D", "P", "B", "R"]:
            cands = [c for c in num_cols if c.startswith(prefix)]
            if cands:
                features.append(cands[0])

    features = [f for f in features if f in df_train.columns][:4]
    if not features:
        log.warning("No features found for time-series plot.")
        return

    # Merge with labels
    df_merged = df_train.merge(df_labels[[ID_COL, TARGET_COL]], on=ID_COL, how="left")

    # Rank statement position (1 = oldest, N = most recent)
    df_merged["stmt_rank"] = (
        df_merged.groupby(ID_COL).cumcount() + 1
    )
    # Keep only customers with 13 statements for cleaner trend lines
    full_cust = (df_merged.groupby(ID_COL)["stmt_rank"].max() == 13)
    full_ids  = full_cust[full_cust].index
    df_plot   = df_merged[df_merged[ID_COL].isin(full_ids)]

    # Sample to avoid slow aggregation
    sample_n = min(5000, len(full_ids))
    sample_ids = pd.Series(full_ids.tolist()).sample(
        min(sample_n, len(full_ids)), random_state=RANDOM_STATE
    ).tolist()
    df_plot = df_plot[df_plot[ID_COL].isin(sample_ids)]

    trend = df_plot.groupby(["stmt_rank", TARGET_COL])[features].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, feat in enumerate(features[:4]):
        ax = axes[i]
        for label_val, color, name in [(0, "#1976D2", "Non-defaulter"),
                                        (1, "#D32F2F", "Defaulter")]:
            subset = trend[trend[TARGET_COL] == label_val]
            if len(subset) > 0:
                ax.plot(subset["stmt_rank"], subset[feat],
                        color=color, linewidth=2, marker="o", markersize=4,
                        label=name)
        ax.set_xlabel("Statement Month (1=oldest)")
        ax.set_ylabel(feat)
        ax.set_title(f"{feat} — Temporal Trend")
        ax.legend(fontsize=8)
        ax.set_xticks(range(1, 14))
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Time-Series Trends: Defaulters vs Non-Defaulters\n"
        "(Averaged over sampled customers with 13 complete statements)",
        fontsize=11,
    )
    plt.tight_layout()
    if save:
        p = REPORT_DIR / "time_series_trends.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Missing value analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_missing_by_group(df_train: pd.DataFrame, save: bool = True) -> None:
    """
    Missing value rates grouped by feature prefix.
    AmEx features have complex structural missingness:
    - Some features are only populated for certain customer segments
    - D_* delinquency features are 0 or NaN (NaN = no delinquency event)
    - Several S_* features have >90% missing for customers without those activities
    """
    num_cols = _get_numeric_cols(df_train)
    miss_pct = df_train[num_cols].isna().mean() * 100

    rows = []
    for col in num_cols:
        prefix = col.split("_")[0]
        rows.append({"feature": col, "prefix": prefix, "miss_pct": miss_pct[col]})
    miss_df = pd.DataFrame(rows)

    group_miss = (
        miss_df.groupby("prefix")["miss_pct"]
        .agg(["mean", "max", "min"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: group-level average
    colors = ["#D32F2F" if v > 20 else "#FFA726" if v > 5 else "#66BB6A"
              for v in group_miss["mean"]]
    axes[0].bar(group_miss["prefix"], group_miss["mean"], color=colors)
    axes[0].set_xlabel("Feature Group Prefix")
    axes[0].set_ylabel("Avg % Missing")
    axes[0].set_title("Average Missing % by Feature Group")
    axes[0].axhline(10, color="orange", linestyle="--", linewidth=1, label="10% threshold")
    axes[0].legend()

    # Right: distribution of individual feature missing rates
    axes[1].hist(miss_df["miss_pct"], bins=30, color="#3949AB", edgecolor="white")
    axes[1].set_xlabel("% Missing per Feature")
    axes[1].set_ylabel("Number of Features")
    axes[1].set_title("Distribution of Feature-Level Missing Rates")
    n_zero    = (miss_df["miss_pct"] == 0).sum()
    n_partial = ((miss_df["miss_pct"] > 0) & (miss_df["miss_pct"] < 50)).sum()
    n_high    = (miss_df["miss_pct"] >= 50).sum()
    axes[1].text(0.65, 0.85,
                 f"0% missing: {n_zero}\n1-50% missing: {n_partial}\n>50% missing: {n_high}",
                 transform=axes[1].transAxes, fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("AmEx: Missing Value Analysis\n"
                 "Structural missingness carries signal — missingness flags are key features",
                 fontsize=11)
    plt.tight_layout()
    if save:
        p = REPORT_DIR / "missing_by_group.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature-target correlation (last statement)
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_correlation(
    df_train: pd.DataFrame,
    df_labels: pd.DataFrame,
    top_n: int = 20,
    save: bool = True,
) -> pd.DataFrame:
    """
    Point-biserial correlation between each numeric feature (last statement)
    and the binary default target. This gives a first-pass feature importance.

    Using the last statement only — consistent with how competition winners
    found "last" features to be among the most predictive (recent behaviour
    is the strongest predictor of imminent default).
    """
    df_last  = _get_last_statement(df_train)
    num_cols = _get_numeric_cols(df_last)

    merged = df_last.merge(df_labels[[ID_COL, TARGET_COL]], on=ID_COL, how="inner")
    y = merged[TARGET_COL].values

    corr_rows = []
    for col in num_cols:
        x = merged[col].fillna(merged[col].median()).values
        if x.std() > 0:
            from scipy.stats import pointbiserialr
            try:
                r, p = pointbiserialr(y, x)
                corr_rows.append({"feature": col, "corr": r, "abs_corr": abs(r), "p_value": p})
            except Exception:
                pass

    corr_df = (
        pd.DataFrame(corr_rows)
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )
    corr_df.to_csv(REPORT_DIR / "feature_target_correlation.csv", index=False)
    log.info(f"Correlation computed for {len(corr_df)} features.")

    # Plot top N
    top = corr_df.head(top_n)
    colors = ["#D32F2F" if c > 0 else "#1976D2" for c in top["corr"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["corr"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Point-Biserial Correlation with Default Target")
    ax.set_title(f"Top {top_n} Features Correlated with Default\n(Last Statement Values)")
    ax.invert_yaxis()
    plt.tight_layout()
    if save:
        p = REPORT_DIR / "feature_target_correlation.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {p}")
    plt.close(fig)

    return corr_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Categorical feature analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_categorical_default_rates(
    df_train: pd.DataFrame,
    df_labels: pd.DataFrame,
    save: bool = True,
) -> None:
    """
    Default rate by D_63 and D_64 category.
    These categorical features encode customer account type and credit product.
    """
    df_last  = _get_last_statement(df_train)
    cat_cols = [c for c in CAT_COLS if c in df_last.columns]
    if not cat_cols:
        return

    merged = df_last.merge(df_labels[[ID_COL, TARGET_COL]], on=ID_COL, how="inner")

    fig, axes = plt.subplots(1, len(cat_cols), figsize=(6 * len(cat_cols), 5))
    if len(cat_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        dr = (
            merged.groupby(col)[TARGET_COL]
            .agg(["mean", "count"])
            .reset_index()
        )
        dr.columns = [col, "default_rate", "count"]
        dr = dr.sort_values("default_rate", ascending=False)

        bars = ax.bar(dr[col].astype(str), dr["default_rate"] * 100,
                      color="#3949AB", edgecolor="white")
        ax.axhline(25.9, color="#D32F2F", linestyle="--", linewidth=1.5,
                   label="Overall (25.9%)")
        ax.set_xlabel(f"{col} Category")
        ax.set_ylabel("Default Rate (%)")
        ax.set_title(f"Default Rate by {col}")
        ax.legend(fontsize=8)
        for bar, cnt in zip(bars, dr["count"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"n={cnt:,}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Categorical Features: Default Rate by Level", fontsize=11)
    plt.tight_layout()
    if save:
        p = REPORT_DIR / "categorical_default_rates.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Outlier analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_outlier_analysis(
    df_train: pd.DataFrame,
    df_labels: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Identify and visualise extreme values in key features.
    AmEx competition winners applied denoise: np.floor(x*100)/100 to
    remove precision noise in numeric features before aggregation.
    """
    df_last  = _get_last_statement(df_train)
    num_cols = _get_numeric_cols(df_last)

    outlier_rows = []
    for col in num_cols[:30]:  # check first 30 numeric features
        vals = df_last[col].dropna()
        if len(vals) < 10:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        n_out = ((vals < q1 - 3 * iqr) | (vals > q3 + 3 * iqr)).sum()
        outlier_rows.append({
            "feature": col,
            "n_outliers": int(n_out),
            "outlier_pct": round(100 * n_out / len(vals), 2),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "skewness": round(float(vals.skew()), 3),
        })

    outlier_df = pd.DataFrame(outlier_rows).sort_values("outlier_pct", ascending=False)
    outlier_df.to_csv(REPORT_DIR / "outlier_report.csv", index=False)

    # Plot top outlier features
    top = outlier_df.head(8)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#D32F2F" if v > 5 else "#FFA726" if v > 1 else "#66BB6A"
              for v in top["outlier_pct"]]
    ax.barh(top["feature"], top["outlier_pct"], color=colors)
    ax.set_xlabel("% Outliers (3×IQR rule)")
    ax.set_title("AmEx: Features with Highest Outlier Rates\n"
                 "(Winners applied np.floor(x×100)/100 to denoise before aggregation)")
    ax.invert_yaxis()
    plt.tight_layout()
    if save:
        p = REPORT_DIR / "outlier_report.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {p}")
    plt.close(fig)

    return outlier_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G: AmEx Credit Default Prediction")
    print("  Step 2: EDA & Data Understanding")
    print("=" * 65 + "\n")

    df_train, df_labels = _load_data()

    print(f"[1] Loaded — rows: {len(df_train):,}  |  customers: {df_labels[ID_COL].nunique():,}")

    print("\n[2] Default rate by statement count…")
    plot_default_by_stmt_count(df_train, df_labels)

    print("\n[3] Time-series trends (defaulters vs non-defaulters)…")
    plot_time_series_trends(df_train, df_labels)

    print("\n[4] Missing value analysis…")
    plot_missing_by_group(df_train)

    print("\n[5] Feature-target correlation (last statement)…")
    try:
        corr_df = compute_feature_correlation(df_train, df_labels)
        print(f"    Top 5 correlated features:\n{corr_df[['feature','corr']].head().to_string(index=False)}")
    except ImportError:
        log.warning("scipy not installed — skipping correlation. pip install scipy")

    print("\n[6] Categorical feature default rates…")
    plot_categorical_default_rates(df_train, df_labels)

    print("\n[7] Outlier analysis…")
    outlier_df = plot_outlier_analysis(df_train, df_labels)
    print(f"    Top outlier features:\n{outlier_df[['feature','outlier_pct']].head(5).to_string(index=False)}")

    print(f"\n  All EDA outputs saved → {REPORT_DIR}")
    print("\n" + "=" * 65)
    print("  Step 2 complete. Ready for Feature Engineering (03_feature_engineering.py)")
    print("=" * 65 + "\n")

    return df_train, df_labels


if __name__ == "__main__":
    main()
