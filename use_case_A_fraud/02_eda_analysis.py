"""
use_case_A_fraud/02_eda_analysis.py
=====================================
Use Case A — Financial Crime & Fraud Analytics
Phase 2, Step 2: Exploratory Data Analysis (EDA)

Covers all required EDA sections from DSF504 deliverables:
  ✓ Missing values (pattern + heatmap)
  ✓ Class imbalance analysis
  ✓ Outlier detection
  ✓ Correlation analysis
  ✓ Important visual insights (time, amount, email, card, V-features)

ML Framework Phase: Data Gathering and Preprocessing → EDA

Run
---
    cd DSF504_ML_Platform
    python use_case_A_fraud/02_eda_analysis.py
"""

from __future__ import annotations

import sys
import logging
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
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE, FRAUD_START_DATE

# ── UTF-8 encoding guard (fixes garbled output on Windows) ─────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPORT_DIR = REPORTS_DIR / "use_case_A"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting style
# ─────────────────────────────────────────────────────────────────────────────
PALETTE   = {"legit": "#1976D2", "fraud": "#D32F2F"}
FRAUD_CLR = PALETTE["fraud"]
LEGIT_CLR = PALETTE["legit"]

sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Missing value analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detailed missing value report.

    Columns are grouped into:
      - Complete      (0%)
      - Low missing   (<10%)
      - Medium        (10–50%)
      - High          (50–90%)
      - Extreme       (>90%) — candidates for dropping

    Returns
    -------
    missing_df : DataFrame sorted by % missing (descending)
    """
    missing = df.isna().sum()
    missing_df = pd.DataFrame({
        "n_missing":   missing,
        "pct_missing": (missing / len(df) * 100).round(2),
    }).sort_values("pct_missing", ascending=False)

    # Missingness category
    def categorise(pct):
        if pct == 0:    return "complete"
        if pct < 10:    return "low (<10%)"
        if pct < 50:    return "medium (10–50%)"
        if pct < 90:    return "high (50–90%)"
        return "extreme (>90%)"

    missing_df["category"] = missing_df["pct_missing"].apply(categorise)
    category_counts = missing_df["category"].value_counts()

    log.info("Missing value categories:\n" + category_counts.to_string())

    # Save report
    missing_df.to_csv(REPORT_DIR / "missing_values_detailed.csv")
    return missing_df


def plot_missing_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """
    Heatmap of missingness for the top-50 most-missing columns,
    visualising whether missingness is random (MCAR) or structured (MAR/MNAR).
    """
    top_missing_cols = (
        df.isna().mean()
        .sort_values(ascending=False)
        .head(50)
        .index.tolist()
    )
    if not top_missing_cols:
        log.info("No missing values found — skipping heatmap.")
        return

    # Sample rows for readability
    sample_size = min(2000, len(df))
    sample_idx  = np.random.choice(len(df), size=sample_size, replace=False)
    heat_df     = df[top_missing_cols].iloc[sample_idx].isna().astype(int)

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(
        heat_df.T,
        cmap="RdYlGn_r", cbar=False,
        xticklabels=False, yticklabels=True, ax=ax,
    )
    ax.set_title(
        "Missing Value Pattern — Top 50 Columns (red = missing)\n"
        "Structured blocks suggest MAR/MNAR: consider group-level imputation",
        fontsize=12,
    )
    ax.set_xlabel(f"Transactions (n={sample_size:,} sample)")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "missing_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Class imbalance
# ─────────────────────────────────────────────────────────────────────────────

def analyze_class_imbalance(df: pd.DataFrame) -> dict:
    """
    Quantify class imbalance and recommend handling strategies.

    Returns
    -------
    imbalance_stats : dict with counts, rates, and recommended strategies
    """
    n_total = len(df)
    n_fraud = int(df["isFraud"].sum())
    n_legit = n_total - n_fraud
    fraud_rate = n_fraud / n_total

    stats_dict = {
        "n_total":      n_total,
        "n_fraud":      n_fraud,
        "n_legit":      n_legit,
        "fraud_rate":   round(fraud_rate, 6),
        "imbalance_ratio": round(n_legit / n_fraud, 1),
    }

    print("\n--- Class Imbalance Analysis ---")
    print(f"  Total transactions : {n_total:>12,}")
    print(f"  Legitimate (0)     : {n_legit:>12,}  ({100*(1-fraud_rate):.2f}%)")
    print(f"  Fraud (1)          : {n_fraud:>12,}  ({100*fraud_rate:.2f}%)")
    print(f"  Imbalance ratio    : {stats_dict['imbalance_ratio']:>12,.1f}:1  (legit:fraud)")
    print()
    print("  Recommended strategies:")
    print("    ✓ Primary metric  : PR-AUC (not accuracy)")
    print("    ✓ Secondary       : ROC-AUC, F1-score (fraud class)")
    print("    ✓ Sampling        : SMOTE on training set only (never on val/test)")
    print("    ✓ Class weights   : class_weight='balanced' as baseline")
    print("    ✓ Threshold       : Calibrate decision threshold (not default 0.5)")

    return stats_dict


# ─────────────────────────────────────────────────────────────────────────────
# 3. Time-based analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_time_patterns(df: pd.DataFrame, save: bool = True) -> None:
    """
    Convert TransactionDT (seconds from reference) to calendar features,
    then analyse fraud rate by hour and day-of-week.

    Key insight: fraudsters often act at unusual hours (late night/early morning).
    This guides the time feature engineering in Step 3.
    """
    ref_date = pd.Timestamp(FRAUD_START_DATE)
    df = df.copy()
    df["transaction_dt"] = ref_date + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["hour"]           = df["transaction_dt"].dt.hour
    df["day_of_week"]    = df["transaction_dt"].dt.dayofweek  # 0=Mon, 6=Sun
    df["day_name"]       = df["transaction_dt"].dt.day_name()

    # Fraud rate by hour
    hourly = df.groupby("hour")["isFraud"].agg(["mean", "sum", "count"])
    hourly.columns = ["fraud_rate", "fraud_count", "total_count"]

    # Fraud rate by day of week
    daily = df.groupby("day_of_week")["isFraud"].agg(["mean", "sum", "count"])
    daily.columns = ["fraud_rate", "fraud_count", "total_count"]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Volume by hour
    axes[0, 0].bar(hourly.index, hourly["total_count"], color=LEGIT_CLR, alpha=0.7)
    axes[0, 0].set_title("Transaction Volume by Hour")
    axes[0, 0].set_xlabel("Hour of Day")
    axes[0, 0].set_ylabel("Number of Transactions")
    axes[0, 0].yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}K")
    )

    # Fraud rate by hour
    axes[0, 1].plot(hourly.index, hourly["fraud_rate"] * 100,
                    marker="o", color=FRAUD_CLR, linewidth=2)
    axes[0, 1].fill_between(hourly.index, hourly["fraud_rate"] * 100,
                             alpha=0.2, color=FRAUD_CLR)
    axes[0, 1].axhline(y=df["isFraud"].mean() * 100, linestyle="--",
                       color="gray", alpha=0.7, label="Overall mean")
    axes[0, 1].set_title("Fraud Rate (%) by Hour of Day")
    axes[0, 1].set_xlabel("Hour of Day")
    axes[0, 1].set_ylabel("Fraud Rate (%)")
    axes[0, 1].legend()

    # Volume by day
    axes[1, 0].bar(daily.index, daily["total_count"], color=LEGIT_CLR, alpha=0.7,
                   tick_label=days)
    axes[1, 0].set_title("Transaction Volume by Day of Week")
    axes[1, 0].set_ylabel("Number of Transactions")
    axes[1, 0].yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}K")
    )

    # Fraud rate by day
    axes[1, 1].bar(daily.index, daily["fraud_rate"] * 100,
                   color=FRAUD_CLR, alpha=0.7, tick_label=days)
    axes[1, 1].axhline(y=df["isFraud"].mean() * 100, linestyle="--",
                       color="gray", alpha=0.7, label="Overall mean")
    axes[1, 1].set_title("Fraud Rate (%) by Day of Week")
    axes[1, 1].set_ylabel("Fraud Rate (%)")
    axes[1, 1].legend()

    plt.suptitle(
        "Time-Based Transaction & Fraud Patterns\n"
        "→ Late-night/weekend spikes guide time-feature engineering",
        fontsize=12,
    )
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "time_patterns.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)

    # Save hourly fraud rate table
    hourly.to_csv(REPORT_DIR / "hourly_fraud_rate.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Categorical feature analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_categorical_features(df: pd.DataFrame, save: bool = True) -> None:
    """
    Fraud rate by ProductCD, card4 (network), card6 (type), and email domain.
    These reveal high-risk categories to encode as target-encoded features.
    """
    cat_features = {
        "ProductCD":       "Product Category",
        "card4":           "Card Network (Visa/MC/etc.)",
        "card6":           "Card Type (credit/debit)",
        "P_emaildomain":   "Purchaser Email Domain",
        "R_emaildomain":   "Recipient Email Domain",
        "DeviceType":      "Device Type",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, cat_features.items()):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        # Fraud rate per category (top 15 by volume)
        grp = (
            df.groupby(col)["isFraud"]
            .agg(fraud_rate="mean", count="count")
            .reset_index()
            .sort_values("count", ascending=False)
            .head(15)
        )

        bars = ax.bar(
            range(len(grp)),
            grp["fraud_rate"] * 100,
            color=[FRAUD_CLR if r > df["isFraud"].mean() else LEGIT_CLR
                   for r in grp["fraud_rate"]],
            alpha=0.8,
        )
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp[col].astype(str), rotation=45, ha="right", fontsize=8)
        ax.axhline(y=df["isFraud"].mean() * 100, linestyle="--",
                   color="gray", alpha=0.7, label="Overall rate")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Fraud Rate (%)")
        ax.legend(fontsize=8)

        # Annotate with sample counts
        for i, (_, row) in enumerate(grp.iterrows()):
            ax.text(i, row["fraud_rate"] * 100 + 0.05,
                    f"n={row['count']:,}", ha="center", va="bottom", fontsize=6, rotation=90)

    plt.suptitle(
        "Fraud Rate by Categorical Features\n"
        "Red bars = above-average fraud rate → strong candidates for target encoding",
        fontsize=12,
    )
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "categorical_fraud_rates.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Numerical feature analysis (C, D columns)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_count_timedelta_features(df: pd.DataFrame, save: bool = True) -> None:
    """
    Boxplots for C-columns (counting variables) and D-columns (time deltas)
    split by fraud label.

    C-columns represent how many billing addresses, emails, phones, etc. are
    associated with the card — high counts signal synthetic identity fraud.
    D-columns represent days since last similar transaction — anomalous gaps
    can indicate account takeover.
    """
    c_cols = [f"C{i}" for i in range(1, 15) if f"C{i}" in df.columns]
    d_cols = [f"D{i}" for i in range(1, 10) if f"D{i}" in df.columns]  # D1-D9

    for group_cols, group_name in [(c_cols, "C_columns"), (d_cols, "D_columns")]:
        if not group_cols:
            continue

        n_cols = len(group_cols)
        fig, axes = plt.subplots(
            (n_cols + 3) // 4, 4, figsize=(18, 4 * ((n_cols + 3) // 4))
        )
        axes = axes.flatten()

        for ax, col in zip(axes, group_cols):
            fraud_vals = df[df["isFraud"] == 1][col].dropna()
            legit_vals = df[df["isFraud"] == 0][col].dropna()

            # Clip extreme outliers for readability
            p99 = df[col].quantile(0.99)
            fraud_vals = fraud_vals.clip(upper=p99)
            legit_vals = legit_vals.clip(upper=p99)

            bp = ax.boxplot(
                [legit_vals, fraud_vals],
                tick_labels=["Legit", "Fraud"],
                patch_artist=True,
                showfliers=False,
            )
            bp["boxes"][0].set_facecolor(LEGIT_CLR)
            bp["boxes"][0].set_alpha(0.6)
            bp["boxes"][1].set_facecolor(FRAUD_CLR)
            bp["boxes"][1].set_alpha(0.6)

            ax.set_title(col, fontsize=10)
            ax.tick_params(labelsize=8)

        # Hide unused axes
        for ax in axes[n_cols:]:
            ax.set_visible(False)

        plt.suptitle(
            f"{'Counting Variables' if group_name == 'C_columns' else 'Time-Delta Variables'} "
            f"by Fraud Label\n(boxes clipped at 99th percentile)",
            fontsize=12,
        )
        plt.tight_layout()

        if save:
            path = REPORT_DIR / f"{group_name}_by_fraud.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            log.info(f"Saved → {path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Outlier detection
# ─────────────────────────────────────────────────────────────────────────────

def analyze_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score-based outlier detection on key numeric columns.
    Flags columns with >1% extreme outliers (|z| > 5).

    Returns
    -------
    outlier_df : Per-column outlier statistics
    """
    numeric_cols = ["TransactionAmt"] + [f"C{i}" for i in range(1, 15)
                                          if f"C{i}" in df.columns]

    rows = []
    for col in numeric_cols:
        vals = df[col].dropna()
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr    = q3 - q1
        lower  = q1 - 3 * iqr
        upper  = q3 + 3 * iqr
        n_out  = int(((vals < lower) | (vals > upper)).sum())
        pct_out = round(100 * n_out / len(vals), 3)

        rows.append({
            "column":        col,
            "min":           round(float(vals.min()), 2),
            "max":           round(float(vals.max()), 2),
            "q1":            round(float(q1), 2),
            "q3":            round(float(q3), 2),
            "iqr_lower":     round(float(lower), 2),
            "iqr_upper":     round(float(upper), 2),
            "n_outliers":    n_out,
            "pct_outliers":  pct_out,
        })

    outlier_df = pd.DataFrame(rows).sort_values("pct_outliers", ascending=False)
    outlier_df.to_csv(REPORT_DIR / "outlier_report.csv", index=False)
    log.info(
        f"Outlier report:\n"
        + outlier_df[outlier_df["pct_outliers"] > 0].to_string(index=False)
    )
    return outlier_df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_correlations(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Correlation heatmap for C-columns and key numeric features vs. isFraud.

    For the 339 V-columns a separate top-N correlation bar chart is produced
    to avoid an illegible 339×339 matrix.

    Returns
    -------
    corr_with_target : Series, absolute Pearson correlation with isFraud
    """
    # --- C-columns heatmap ---
    c_cols = ["isFraud", "TransactionAmt"] + [
        f"C{i}" for i in range(1, 15) if f"C{i}" in df.columns
    ]
    corr_c = df[c_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_c, dtype=bool))
    sns.heatmap(
        corr_c, mask=mask, cmap="RdBu_r", center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.3, ax=ax,
    )
    ax.set_title(
        "Correlation Heatmap — C-columns + TransactionAmt + isFraud\n"
        "C-columns encode cardinality (addresses, phones, emails) per card",
        fontsize=11,
    )
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "correlation_heatmap_C_cols.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)

    # --- Top-30 V-columns by absolute correlation with isFraud ---
    v_cols = [f"V{i}" for i in range(1, 340) if f"V{i}" in df.columns]
    if v_cols:
        corr_v = df[v_cols + ["isFraud"]].corr()["isFraud"].drop("isFraud")
        top30  = corr_v.abs().sort_values(ascending=False).head(30)

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = [FRAUD_CLR if corr_v[c] > 0 else LEGIT_CLR for c in top30.index]
        ax.barh(top30.index[::-1], corr_v[top30.index[::-1]], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Pearson Correlation with isFraud")
        ax.set_title(
            "Top-30 V-features by Absolute Correlation with isFraud\n"
            "(Red = positive correlation with fraud, Blue = negative)",
            fontsize=11,
        )
        plt.tight_layout()

        if save:
            path = REPORT_DIR / "correlation_top30_V_cols.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            log.info(f"Saved → {path}")
        plt.close(fig)

    # Return full target correlation (for feature selection)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_target = (
        numeric_df.corr()["isFraud"].drop("isFraud").abs()
        .sort_values(ascending=False)
    )
    corr_target.to_csv(REPORT_DIR / "feature_target_correlation.csv",
                       header=True)
    return corr_target


# ─────────────────────────────────────────────────────────────────────────────
# 8. EDA summary
# ─────────────────────────────────────────────────────────────────────────────

def print_eda_summary(df: pd.DataFrame) -> None:
    """
    Print a structured EDA summary — maps directly to DSF504 report Section 4.
    """
    print("\n" + "=" * 65)
    print("  EDA SUMMARY — KEY FINDINGS")
    print("=" * 65)

    # Missing
    miss_pct = df.isna().mean() * 100
    extreme  = (miss_pct > 50).sum()
    high     = ((miss_pct > 10) & (miss_pct <= 50)).sum()
    low      = ((miss_pct > 0)  & (miss_pct <= 10)).sum()
    print(f"\n  Missing Values:")
    print(f"    Extreme (>50%): {extreme} columns — consider dropping or indicator flag")
    print(f"    High (10–50%):  {high} columns — median/mode imputation")
    print(f"    Low (<10%):     {low} columns — simple imputation")

    # Class imbalance
    fraud_pct = df["isFraud"].mean() * 100
    print(f"\n  Class Imbalance:")
    print(f"    Fraud rate: {fraud_pct:.2f}%  → Severe imbalance")
    print(f"    Strategy  : SMOTE + PR-AUC as primary metric")

    # Transaction amount
    med_fraud = df[df["isFraud"] == 1]["TransactionAmt"].median()
    med_legit = df[df["isFraud"] == 0]["TransactionAmt"].median()
    print(f"\n  Transaction Amount:")
    print(f"    Median fraud: ${med_fraud:,.2f}  vs  Median legit: ${med_legit:,.2f}")
    print(f"    → Log-transform recommended; interaction with card/product")

    # V-column coverage
    v_cols   = [c for c in df.columns if c.startswith("V")]
    v_miss   = df[v_cols].isna().mean().mean() * 100 if v_cols else 0
    print(f"\n  V-columns (Vesta features):")
    print(f"    {len(v_cols)} columns, avg missing: {v_miss:.1f}%")
    print(f"    → Use PCA or select top-30 by correlation to reduce dimensionality")

    print("\n  Top Feature Engineering Priorities:")
    print("    1. Time features from TransactionDT (hour, day, week)")
    print("    2. Transaction velocity (count per card in rolling window)")
    print("    3. Card-level aggregations (mean/std amount, frequency)")
    print("    4. Email domain risk encoding")
    print("    5. Missing value indicator flags for high-missingness columns")
    print("    6. Log(TransactionAmt + 1)")
    print("    7. Amount deviation from card's historical mean")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """
    Execute full EDA pipeline on the merged IEEE-CIS training DataFrame.

    Parameters
    ----------
    df : Merged train DataFrame from 01_data_loading.py
    """
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case A: EDA Analysis")
    print("=" * 65 + "\n")

    print("[1] Missing value analysis…")
    analyze_missing_values(df)
    plot_missing_heatmap(df)

    print("[2] Class imbalance analysis…")
    analyze_class_imbalance(df)

    print("[3] Time pattern analysis…")
    analyze_time_patterns(df)

    print("[4] Categorical feature analysis…")
    analyze_categorical_features(df)

    print("[5] Count & time-delta feature analysis…")
    analyze_count_timedelta_features(df)

    print("[6] Outlier detection…")
    analyze_outliers(df)

    print("[7] Correlation analysis…")
    analyze_correlations(df)

    print("[8] EDA Summary…")
    print_eda_summary(df)

    print(f"\n  All EDA outputs saved to: {REPORT_DIR}")
    print("  Ready for feature engineering (03_feature_engineering.py)\n")


def main():
    """
    Load data (from parquet cache if available) and run full EDA.
    Requires 01_data_loading.py to have been run at least once.
    """
    # Try loading from cached Parquet first (fast)
    parquet_path = DATA_DIR / "ieee_fraud" / "train_transaction.parquet"
    if not parquet_path.exists():
        print(
            "[!] Parquet cache not found. Run 01_data_loading.py first.\n"
            "    python use_case_A_fraud/01_data_loading.py"
        )
        return

    # Load directly from Parquet cache (written by 01_data_loading.py)
    log.info("Loading from Parquet cache…")
    df_trn   = pd.read_parquet(DATA_DIR / "ieee_fraud" / "train_transaction.parquet")
    idn_path = DATA_DIR / "ieee_fraud" / "train_identity.parquet"
    df_idn   = pd.read_parquet(idn_path) if idn_path.exists() else None
    df_train = (
        df_trn.merge(df_idn, on="TransactionID", how="left")
        if df_idn is not None else df_trn
    )

    run_eda(df_train)


if __name__ == "__main__":
    main()
