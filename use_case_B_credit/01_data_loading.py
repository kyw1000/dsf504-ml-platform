"""
use_case_B_credit/01_data_loading.py
=====================================
DSF504 — Use Case B: Credit Risk Modelling
Step 1: Data loading, profiling, and train/validation split

Dataset
-------
Give Me Some Credit (Kaggle / FICO, 2011)
  150,000 borrowers · 10 features · target: SeriousDlqin2yrs
  https://www.kaggle.com/c/GiveMeSomeCredit

Key challenges handled here
----------------------------
- ~19% missing MonthlyIncome, ~2.5% missing NumberOfDependents
- Delinquency columns encode errors as 96/98 (treated as outliers)
- RevolvingUtilizationOfUnsecuredLines has values > 1 (data entry errors)
- Moderate class imbalance: 6.7% default rate

Academic references
-------------------
- Siddiqi, N. (2012). Credit Risk Scorecards. Wiley.
- Lessmann et al. (2015). Benchmarking state-of-the-art classification
  algorithms for credit scoring: An update of research. EJOR.
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
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE
from utils.data_loader import KaggleLoader, DataProfiler, smart_split, reduce_mem_usage

# ── UTF-8 encoding guard (fixes garbled output on Windows) ─────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_SUBDIR  = DATA_DIR    / "gmsc_credit"
REPORT_DIR   = REPORTS_DIR / "use_case_B"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_SUBDIR.mkdir(parents=True, exist_ok=True)

TARGET       = "SeriousDlqin2yrs"
DELINQ_COLS  = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load the Give Me Some Credit training CSV (download if needed)."""
    csv_path     = DATA_SUBDIR / "cs-training.csv"
    parquet_path = DATA_SUBDIR / "cs-training.parquet"

    if parquet_path.exists():
        log.info("Loading from Parquet cache…")
        return pd.read_parquet(parquet_path)

    if not csv_path.exists():
        log.info("CSV not found — attempting Kaggle download…")
        loader = KaggleLoader("B")
        loader.download()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training CSV not found at {csv_path}.\n"
            "Download manually from https://www.kaggle.com/c/GiveMeSomeCredit/data\n"
            "and place cs-training.csv in the data/gmsc_credit/ folder."
        )

    log.info(f"Loading CSV: {csv_path.name}…")
    df = pd.read_csv(csv_path, index_col=0)   # first col is row index

    # Standardise column names (remove special chars for downstream compatibility)
    df = df.rename(columns={
        "NumberOfTime30-59DaysPastDueNotWorse": "DPD_30_59",
        "NumberOfTime60-89DaysPastDueNotWorse": "DPD_60_89",
        "NumberOfTimes90DaysLate":              "DPD_90plus",
        "NumberOfOpenCreditLinesAndLoans":      "NumOpenLoans",
        "NumberRealEstateLoansOrLines":         "NumRealEstate",
        "NumberOfDependents":                   "NumDependents",
        "RevolvingUtilizationOfUnsecuredLines": "RevolvingUtil",
        "DebtRatio":                            "DebtRatio",
        "MonthlyIncome":                        "MonthlyIncome",
        "age":                                  "age",
    })

    df = reduce_mem_usage(df, verbose=True)
    df.to_parquet(parquet_path, index=False)
    log.info(f"Cached as Parquet: {parquet_path.name}")
    return df


# ── 2. Profile ─────────────────────────────────────────────────────────────────

def profile_data(df: pd.DataFrame) -> None:
    """Print dataset profile and save column summary."""
    print("\n" + "=" * 60)
    print("  DATASET PROFILE — Give Me Some Credit")
    print("=" * 60)
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {df.shape[1]}")
    print(f"  Fraud rate   : {df[TARGET].mean():.3%}")
    print(f"  Memory       : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print()

    profiler = DataProfiler(df, target_col=TARGET)
    summary  = profiler.summary()
    profiler.print_report()

    summary.to_csv(REPORT_DIR / "train_column_summary.csv", index=False)
    log.info(f"Column summary saved → {REPORT_DIR / 'train_column_summary.csv'}")

    # Missing values report
    missing = summary[summary["n_missing"] > 0][["column", "n_missing", "pct_missing"]]
    if len(missing):
        missing.to_csv(REPORT_DIR / "train_missing_values.csv", index=False)
        log.info(f"Missing values report → {REPORT_DIR / 'train_missing_values.csv'}")


# ── 3. Visualisations ──────────────────────────────────────────────────────────

def plot_overview(df: pd.DataFrame) -> None:
    """4-panel overview: target dist, age, income, utilisation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Give Me Some Credit — Overview", fontsize=14, fontweight="bold")

    # Target
    ax = axes[0, 0]
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar(["Performing (0)", "Default (1)"],
                  counts.values, color=["#43A047", "#E53935"], alpha=0.85)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 500, f"{count:,}",
                ha="center", va="bottom", fontsize=11)
    ax.set_title("Target Distribution")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Age distribution by class
    ax = axes[0, 1]
    for label, col, clr in [(0, "Performing", "#43A047"), (1, "Default", "#E53935")]:
        subset = df[df[TARGET] == label]["age"].clip(18, 100)
        ax.hist(subset, bins=40, alpha=0.6, color=clr, label=f"{'Default' if label else 'Performing'}")
    ax.set_title("Age Distribution by Class")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.legend()

    # Monthly income (log scale)
    ax = axes[1, 0]
    income = df["MonthlyIncome"].dropna().clip(upper=30_000)
    ax.hist(income, bins=60, color="#1E88E5", alpha=0.8)
    ax.set_title("Monthly Income Distribution (capped $30K)")
    ax.set_xlabel("Monthly Income ($)")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Revolving utilisation
    ax = axes[1, 1]
    util = df["RevolvingUtil"].clip(0, 2)
    ax.hist(util, bins=50, color="#FB8C00", alpha=0.8)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Max valid (1.0)")
    ax.set_title("Revolving Utilisation (capped 2x)")
    ax.set_xlabel("Utilisation Rate")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    path = REPORT_DIR / "overview.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


def plot_delinquency(df: pd.DataFrame) -> None:
    """Delinquency columns vs default rate."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Delinquency Features vs Default Rate", fontsize=13, fontweight="bold")

    cols = [("DPD_30_59", "30–59 Days Past Due"),
            ("DPD_60_89", "60–89 Days Past Due"),
            ("DPD_90plus","90+ Days Past Due")]

    for ax, (col, title) in zip(axes, cols):
        # Clip at 10 (96/98 are error codes)
        clipped = df[col].clip(0, 10)
        rates   = df.groupby(clipped)[TARGET].mean() * 100
        ax.bar(rates.index, rates.values, color="#1E88E5", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Count of occurrences (capped 10)")
        ax.set_ylabel("Default Rate (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.tight_layout()
    path = REPORT_DIR / "delinquency_vs_default.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


def plot_debt_income(df: pd.DataFrame) -> None:
    """Debt ratio and income vs default."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Debt ratio by class (log scale)
    ax = axes[0]
    for label, clr in [(0, "#43A047"), (1, "#E53935")]:
        vals = np.log1p(df[df[TARGET] == label]["DebtRatio"].clip(0, 5000))
        ax.hist(vals, bins=50, alpha=0.6, color=clr,
                label=f"{'Default' if label else 'Performing'}")
    ax.set_title("log(1+DebtRatio) by Class")
    ax.set_xlabel("log(1 + Debt Ratio)")
    ax.legend()

    # Default rate by age bucket
    ax = axes[1]
    df2   = df.copy()
    df2["age_bucket"] = pd.cut(df2["age"].clip(18, 90),
                               bins=[18, 30, 40, 50, 60, 70, 90],
                               labels=["18–30","30–40","40–50","50–60","60–70","70+"])
    rates = df2.groupby("age_bucket", observed=True)[TARGET].mean() * 100
    ax.bar(rates.index.astype(str), rates.values, color="#7B1FA2", alpha=0.85)
    ax.set_title("Default Rate by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Default Rate (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.tight_layout()
    path = REPORT_DIR / "debt_income_analysis.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ── 4. Split ───────────────────────────────────────────────────────────────────

def split_and_save(df: pd.DataFrame) -> None:
    """Stratified 80/20 split; save parquet files."""
    df_train, df_val = smart_split(
        df, target_col=TARGET, task_type="binary_classification",
        val_size=0.20, random_state=RANDOM_STATE,
    )
    fraud_tr  = df_train[TARGET].mean()
    fraud_val = df_val[TARGET].mean()
    log.info(
        f"\nSplit → train: {len(df_train):,} ({fraud_tr:.3%} default) "
        f"| val: {len(df_val):,} ({fraud_val:.3%} default)"
    )

    df_train.to_parquet(DATA_SUBDIR / "train_raw.parquet", index=False)
    df_val.to_parquet(DATA_SUBDIR / "val_raw.parquet",   index=False)
    log.info(f"Saved raw splits to {DATA_SUBDIR}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case B: Credit Risk Data Loading")
    print("=" * 65 + "\n")

    print("[1] Loading data…")
    df = load_data()

    print("[2] Profiling dataset…")
    profile_data(df)

    print("[3] Generating overview plots…")
    plot_overview(df)
    plot_delinquency(df)
    plot_debt_income(df)

    print("[4] Splitting train / validation…")
    split_and_save(df)

    print("\n" + "=" * 65)
    print("  Step 1 complete. Ready for EDA (02_eda_analysis.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
