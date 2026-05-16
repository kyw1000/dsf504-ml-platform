"""
use_case_B_credit/02_eda_analysis.py
=====================================
DSF504 — Use Case B: Credit Risk Modelling
Step 2: Exploratory Data Analysis

Key findings targeted
---------------------
1. Class imbalance (6.7% default rate) and SMOTE rationale
2. Outlier/error codes in delinquency columns (96, 98)
3. RevolvingUtilisation > 1 (erroneous entries)
4. Missing data patterns (MonthlyIncome MNAR, NumDependents MAR)
5. Feature-target correlations and multicollinearity
6. Credit scorecard intuition: monotonic relationships

Academic references
-------------------
- Baesens et al. (2016). Credit Risk Analytics. Wiley.
- Thomas et al. (2017). Credit Scoring and Its Applications. SIAM.
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
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR

# ── UTF-8 encoding guard (fixes garbled output on Windows) ─────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR    / "gmsc_credit"
REPORT_DIR  = REPORTS_DIR / "use_case_B"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "SeriousDlqin2yrs"
DELINQ = ["DPD_30_59", "DPD_60_89", "DPD_90plus"]
NUMERIC = [
    "RevolvingUtil", "age", "DPD_30_59", "DebtRatio",
    "MonthlyIncome", "NumOpenLoans", "DPD_90plus",
    "NumRealEstate", "DPD_60_89", "NumDependents",
]


def load() -> pd.DataFrame:
    path = DATA_SUBDIR / "cs-training.parquet"
    if not path.exists():
        path = DATA_SUBDIR / "train_raw.parquet"
    if not path.exists():
        raise FileNotFoundError("Run 01_data_loading.py first.")
    log.info(f"Loading from {path.name}…")
    return pd.read_parquet(path)


# ── 1. Class imbalance ─────────────────────────────────────────────────────────

def analyse_imbalance(df: pd.DataFrame) -> None:
    n_total   = len(df)
    n_default = df[TARGET].sum()
    rate      = n_default / n_total

    print("\n--- Class Imbalance ---")
    print(f"  Total borrowers : {n_total:,}")
    print(f"  Performing (0)  : {n_total - n_default:,}  ({1-rate:.2%})")
    print(f"  Default (1)     : {n_default:,}  ({rate:.2%})")
    print(f"  Imbalance ratio : {(n_total-n_default)/n_default:.1f}:1")
    print("  Recommended     : SMOTE on train only · primary metric = ROC-AUC")


# ── 2. Outlier / error-code analysis ──────────────────────────────────────────

def analyse_outliers(df: pd.DataFrame) -> None:
    """Flag 96/98 error codes in delinquency columns and utilisation > 1."""
    print("\n--- Outlier / Error Code Analysis ---")

    for col in DELINQ:
        if col not in df.columns:
            continue
        n96 = (df[col] == 96).sum()
        n98 = (df[col] == 98).sum()
        n_high = (df[col] > 10).sum()
        print(f"  {col}: val=96 → {n96:,}  val=98 → {n98:,}  >10 → {n_high:,}")

    if "RevolvingUtil" in df.columns:
        n_over = (df["RevolvingUtil"] > 1).sum()
        n_extreme = (df["RevolvingUtil"] > 10).sum()
        print(f"  RevolvingUtil > 1  : {n_over:,}  ({100*n_over/len(df):.2f}%)")
        print(f"  RevolvingUtil > 10 : {n_extreme:,} (likely data errors)")

    if "DebtRatio" in df.columns:
        n_extreme = (df["DebtRatio"] > 1000).sum()
        print(f"  DebtRatio > 1000   : {n_extreme:,} (likely errors or unemployed)")


# ── 3. Missing value analysis ──────────────────────────────────────────────────

def analyse_missing(df: pd.DataFrame) -> None:
    miss = df.isna().sum()
    miss = miss[miss > 0]

    print("\n--- Missing Value Analysis ---")
    for col, n in miss.items():
        print(f"  {col:35s}: {n:,} ({100*n/len(df):.1f}%)")

    # MNAR test for MonthlyIncome: do defaults have higher missingness?
    if "MonthlyIncome" in df.columns:
        miss_def  = df[df[TARGET] == 1]["MonthlyIncome"].isna().mean()
        miss_perf = df[df[TARGET] == 0]["MonthlyIncome"].isna().mean()
        print(f"\n  MonthlyIncome missingness by class:")
        print(f"    Default    : {miss_def:.2%}")
        print(f"    Performing : {miss_perf:.2%}")
        if abs(miss_def - miss_perf) > 0.02:
            print("    → MNAR suspected: missingness differs by default status")


# ── 4. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame) -> None:
    cols = [c for c in NUMERIC + [TARGET] if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix — Give Me Some Credit", fontweight="bold")
    plt.tight_layout()
    path = REPORT_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ── 5. Feature–target relationship (Information Value proxy) ──────────────────

def plot_feature_default_rates(df: pd.DataFrame) -> None:
    """Bin each numeric feature and plot default rate per bin."""
    features = [c for c in NUMERIC if c in df.columns and c != TARGET]
    n_cols = 2
    n_rows = (len(features) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()
    fig.suptitle("Default Rate by Feature Bin (Credit Scorecard View)",
                 fontsize=13, fontweight="bold")

    for ax, feat in zip(axes, features):
        clean = df[[feat, TARGET]].copy()
        # Cap extreme outliers for binning
        q01, q99 = clean[feat].quantile([0.01, 0.99])
        clean[feat] = clean[feat].clip(q01, q99)

        try:
            clean["bin"] = pd.qcut(clean[feat], q=10, duplicates="drop")
            rates = clean.groupby("bin", observed=True)[TARGET].mean() * 100
            ax.bar(range(len(rates)), rates.values, color="#1E88E5", alpha=0.8)
            ax.set_xticks(range(len(rates)))
            ax.set_xticklabels(
                [str(b.mid.round(1)) for b in rates.index],
                rotation=45, ha="right", fontsize=7
            )
        except Exception:
            ax.text(0.5, 0.5, "insufficient unique values",
                    ha="center", va="center", transform=ax.transAxes, color="gray")

        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_ylabel("Default Rate (%)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Hide unused panels
    for ax in axes[len(features):]:
        ax.set_visible(False)

    plt.tight_layout()
    path = REPORT_DIR / "feature_default_rates.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ── 6. Missing pattern heatmap ────────────────────────────────────────────────

def plot_missing_pattern(df: pd.DataFrame) -> None:
    miss = df.isna().any()
    miss_cols = miss[miss].index.tolist()
    if not miss_cols:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Missing Value Patterns", fontsize=12, fontweight="bold")

    # Bar chart of missing %
    ax = axes[0]
    miss_pct = df[miss_cols].isna().mean() * 100
    ax.barh(miss_pct.index, miss_pct.values, color="#E53935", alpha=0.8)
    ax.set_xlabel("% Missing")
    ax.set_title("Columns with Missing Values")
    for i, v in enumerate(miss_pct.values):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)

    # Missing by default class
    ax = axes[1]
    for col in miss_cols:
        rates = df.groupby(TARGET)[col].apply(lambda x: x.isna().mean() * 100)
        x_pos = [0, 1]
        ax.bar([p + miss_cols.index(col) * 0.35 for p in x_pos],
               rates.values, width=0.3,
               label=col, alpha=0.8)
    ax.set_xticks([0.18, 1.18])
    ax.set_xticklabels(["Performing", "Default"])
    ax.set_ylabel("% Missing")
    ax.set_title("Missingness by Default Status")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = REPORT_DIR / "missing_pattern.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ── 7. EDA summary ────────────────────────────────────────────────────────────

def print_eda_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  EDA SUMMARY — KEY FINDINGS")
    print("=" * 65)

    rate = df[TARGET].mean()
    print(f"  Class Imbalance : {rate:.2%} default → SMOTE + ROC-AUC metric")

    if "MonthlyIncome" in df.columns:
        pct = df["MonthlyIncome"].isna().mean()
        print(f"  MonthlyIncome   : {pct:.1%} missing (MNAR — impute with median by age)")

    if "RevolvingUtil" in df.columns:
        pct = (df["RevolvingUtil"] > 1).mean()
        print(f"  RevolvingUtil>1 : {pct:.1%} of rows → cap at 1.0 or winsorise")

    for col in DELINQ:
        if col in df.columns:
            pct = (df[col] >= 96).mean()
            if pct > 0:
                print(f"  {col:20s}: {pct:.2%} rows with error codes 96/98 → replace with NaN")

    print("\n  Feature Engineering Priorities:")
    print("    1. Cap/Winsorise RevolvingUtil, DebtRatio, DPD columns")
    print("    2. Log(1+MonthlyIncome) for skew reduction")
    print("    3. Total delinquency score = DPD_30 + DPD_60 + DPD_90")
    print("    4. Debt-to-income ratio (DebtRatio × MonthlyIncome proxy)")
    print("    5. Missing indicators for MonthlyIncome, NumDependents")
    print("    6. Age groups (non-linear relationship with default)")
    print("=" * 65 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case B: EDA Analysis")
    print("=" * 65 + "\n")

    df = load()

    print("[1] Class imbalance…")
    analyse_imbalance(df)

    print("[2] Outlier / error-code analysis…")
    analyse_outliers(df)

    print("[3] Missing value analysis…")
    analyse_missing(df)

    print("[4] Correlation heatmap…")
    plot_correlation(df)

    print("[5] Feature–default rate plots…")
    plot_feature_default_rates(df)

    print("[6] Missing pattern plot…")
    plot_missing_pattern(df)

    print_eda_summary(df)

    print(f"  All EDA outputs saved to: {REPORT_DIR}")
    print("  Ready for feature engineering (03_feature_engineering.py)")


if __name__ == "__main__":
    main()
