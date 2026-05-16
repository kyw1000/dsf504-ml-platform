"""
use_case_A_fraud/01_data_loading.py
=====================================
Use Case A — Financial Crime & Fraud Analytics
Phase 2, Step 1: Data Acquisition & Initial Inspection

Dataset : IEEE-CIS Fraud Detection (Kaggle)
          https://www.kaggle.com/c/ieee-fraud-detection/data
Files   : train_transaction.csv  (590,540 rows × 394 cols)
          train_identity.csv     (144,233 rows ×  41 cols)
          Merged on TransactionID (left join → keep all transactions)
Target  : isFraud  (binary: 1=fraud, 0=legitimate)
Imbalance: ~3.5% fraud  ← critical for model selection

ML Framework Phase: Data Gathering and Preprocessing
     └─ Gather Data / ETL
     └─ Apply Data Cleansing
     └─ Protect Data Privacy and Security
     └─ Perform Feature Extraction
     └─ Split Data Sets

Run
---
    cd DSF504_ML_Platform
    python use_case_A_fraud/01_data_loading.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import seaborn as sns

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR, REPORTS_DIR, RANDOM_STATE, DATASET_REGISTRY,
    FRAUD_START_DATE,
)
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

REPORT_DIR = REPORTS_DIR / "use_case_A"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Column groups for the IEEE-CIS dataset
# (used throughout EDA and feature engineering)
# ─────────────────────────────────────────────────────────────────────────────

# Transaction table column groups
TRANSACTION_COLS = {
    "id":          ["TransactionID"],
    "target":      ["isFraud"],
    "time":        ["TransactionDT"],
    "amount":      ["TransactionAmt"],
    "product":     ["ProductCD"],
    "card":        ["card1", "card2", "card3", "card4", "card5", "card6"],
    "address":     ["addr1", "addr2"],
    "distance":    ["dist1", "dist2"],
    "email":       ["P_emaildomain", "R_emaildomain"],
    "counting_C":  [f"C{i}" for i in range(1, 15)],   # C1–C14
    "timedelta_D": [f"D{i}" for i in range(1, 16)],   # D1–D15
    "match_M":     [f"M{i}" for i in range(1, 10)],   # M1–M9
    "vesta_V":     [f"V{i}" for i in range(1, 340)],  # V1–V339
}

# Identity table column groups
IDENTITY_COLS = {
    "id_numeric":      [f"id_{i:02d}" for i in range(1, 12)],   # id_01–id_11
    "id_categorical":  [f"id_{i:02d}" for i in range(12, 39)],  # id_12–id_38
    "device":          ["DeviceType", "DeviceInfo"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ieee_fraud(
    force_download: bool = False,
    optimize_memory: bool = True,
    sample_frac: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the IEEE-CIS Fraud Detection dataset.

    Merges transaction + identity tables on TransactionID (left join so every
    transaction is retained; identity rows without a transaction are dropped).

    Parameters
    ----------
    force_download  : Re-download from Kaggle even if local files exist.
    optimize_memory : Downcast numerics and convert low-cardinality strings
                      to category to reduce RAM.
    sample_frac     : For fast development, sample a fraction of the training
                      set (stratified by isFraud). None = full dataset.

    Returns
    -------
    df_train : Merged train DataFrame  (590,540 rows × ~433 cols)
    df_test  : Merged test DataFrame   (506,691 rows × ~432 cols, no isFraud)
    """
    loader = KaggleLoader(
        use_case_key="A",
        force_download=force_download,
        optimize_memory=optimize_memory,
    )
    df_train, df_test = loader.load()

    if df_train is None:
        _print_manual_download_instructions()
        return None, None

    # Optional stratified sample for rapid iteration
    if sample_frac is not None and 0 < sample_frac < 1.0:
        fraud   = df_train[df_train["isFraud"] == 1]
        legit   = df_train[df_train["isFraud"] == 0]
        fraud_s = fraud.sample(frac=sample_frac, random_state=RANDOM_STATE)
        legit_s = legit.sample(frac=sample_frac, random_state=RANDOM_STATE)
        df_train = pd.concat([fraud_s, legit_s]).sample(
            frac=1, random_state=RANDOM_STATE
        ).reset_index(drop=True)
        log.info(
            f"Sampled {sample_frac:.0%}: {len(df_train):,} rows "
            f"({fraud_s.shape[0]:,} fraud, {legit_s.shape[0]:,} legit)"
        )

    return df_train, df_test


def _print_manual_download_instructions() -> None:
    """Instructions printed when Kaggle API is not configured."""
    print(
        "\n" + "=" * 65 + "\n"
        "  MANUAL DOWNLOAD INSTRUCTIONS\n"
        "=" * 65 + "\n"
        "  1. Create a Kaggle account at https://kaggle.com\n"
        "  2. Accept the competition rules at:\n"
        "     https://www.kaggle.com/c/ieee-fraud-detection/data\n"
        "  3. Download these four files:\n"
        "       - train_transaction.csv\n"
        "       - train_identity.csv\n"
        "       - test_transaction.csv\n"
        "       - test_identity.csv\n"
        "  4. Place them in:\n"
        f"     {DATA_DIR / 'ieee_fraud'}\n"
        "  5. Re-run this script.\n"
        "\n"
        "  Alternatively, set up the Kaggle API:\n"
        "    pip install kaggle\n"
        "    mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/\n"
        "    chmod 600 ~/.kaggle/kaggle.json\n"
        "=" * 65 + "\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Initial data inspection report
# ─────────────────────────────────────────────────────────────────────────────

def inspect_dataset(df: pd.DataFrame, label: str = "train") -> DataProfiler:
    """
    Run the DataProfiler and save a column-level summary CSV.

    Parameters
    ----------
    df    : DataFrame to inspect
    label : Prefix for output files ("train" or "test")

    Returns
    -------
    profiler : DataProfiler instance (reusable for further analysis)
    """
    log.info(f"\n{'='*55}\nDataset inspection: {label.upper()}\n{'='*55}")

    profiler = DataProfiler(df, target_col="isFraud" if label == "train" else None)
    profiler.print_report()

    # Save column summary
    summary = profiler.summary()
    out_path = REPORT_DIR / f"{label}_column_summary.csv"
    summary.to_csv(out_path, index=False)
    log.info(f"Column summary saved → {out_path}")

    # Save missing value report
    missing = profiler.missing_heatmap_data()
    if len(missing) > 0:
        miss_path = REPORT_DIR / f"{label}_missing_values.csv"
        missing.to_csv(miss_path)
        log.info(f"Missing value report saved → {miss_path}")

    return profiler


# ─────────────────────────────────────────────────────────────────────────────
# 3. Column group analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_column_groups(df: pd.DataFrame) -> dict:
    """
    Analyse missingness and cardinality for each IEEE-CIS column group.

    The V-columns (Vesta-engineered) often have 40-80% missing; the C-columns
    (counting variables) are typically complete. This analysis guides which
    groups to impute vs. drop.

    Returns
    -------
    stats : dict mapping group_name → DataFrame with per-group summary
    """
    all_groups = {**TRANSACTION_COLS, **IDENTITY_COLS}
    stats = {}

    print("\n--- Column Group Analysis ---")
    for group, cols in all_groups.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            continue

        subset   = df[present]
        n_cols   = len(present)
        avg_miss = subset.isna().mean().mean() * 100
        dtypes   = subset.dtypes.value_counts().to_dict()

        stats[group] = {
            "n_columns":    n_cols,
            "avg_missing%": round(avg_miss, 1),
            "dtypes":       str(dtypes),
        }
        print(
            f"  {group:<18} | {n_cols:>4} cols | "
            f"avg missing: {avg_miss:5.1f}% | dtypes: {dtypes}"
        )

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. Visualise missing value patterns
# ─────────────────────────────────────────────────────────────────────────────

def plot_missing_by_group(df: pd.DataFrame, save: bool = True) -> None:
    """
    Bar chart: average % missing per column group.
    Helps decide imputation strategy before feature engineering.
    """
    all_groups = {**TRANSACTION_COLS, **IDENTITY_COLS}
    rows = []
    for group, cols in all_groups.items():
        present = [c for c in cols if c in df.columns]
        if present:
            avg_miss = df[present].isna().mean().mean() * 100
            rows.append({"group": group, "avg_missing_pct": avg_miss})

    gdf = pd.DataFrame(rows).sort_values("avg_missing_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d32f2f" if v > 50 else "#f57c00" if v > 20 else "#388e3c"
              for v in gdf["avg_missing_pct"]]
    ax.barh(gdf["group"], gdf["avg_missing_pct"], color=colors)
    ax.axvline(x=50, color="red", linestyle="--", alpha=0.6, label="50% threshold")
    ax.axvline(x=20, color="orange", linestyle="--", alpha=0.6, label="20% threshold")
    ax.set_xlabel("Average % Missing Values")
    ax.set_title("IEEE-CIS: Average Missing Values by Column Group")
    ax.legend()
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "missing_by_group.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


def plot_target_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """
    Visualise the severe class imbalance in isFraud.
    Critical: informs SMOTE strategy, threshold calibration, and metric choice.
    """
    counts = df["isFraud"].value_counts()
    labels = ["Legitimate (0)", "Fraud (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    pcts   = [100 * v / len(df) for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    colors = ["#1976D2", "#D32F2F"]
    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, pct in zip(bars, pcts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2000,
            f"{pct:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    axes[0].set_title("Transaction Class Distribution", fontsize=13)
    axes[0].set_ylabel("Count")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Pie chart (log-scale visual)
    explode = (0, 0.1)
    axes[1].pie(
        values, labels=labels, autopct="%1.2f%%",
        colors=colors, explode=explode, startangle=140,
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Class Imbalance (Fraud = 3.5%)", fontsize=13)

    fig.suptitle(
        "Class Imbalance — IEEE-CIS Fraud Detection\n"
        "Implication: Use PR-AUC, F1, not accuracy; apply SMOTE or class weights",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "target_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


def plot_transaction_amount(df: pd.DataFrame, save: bool = True) -> None:
    """
    Distribution of TransactionAmt split by isFraud.
    Fraudulent transactions often cluster at unusual amounts.
    """
    fraud = df[df["isFraud"] == 1]["TransactionAmt"].clip(upper=2000)
    legit = df[df["isFraud"] == 0]["TransactionAmt"].clip(upper=2000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Log-scale histogram overlay
    axes[0].hist(
        np.log1p(legit), bins=80, alpha=0.6, color="#1976D2", label="Legitimate"
    )
    axes[0].hist(
        np.log1p(fraud), bins=80, alpha=0.7, color="#D32F2F", label="Fraud"
    )
    axes[0].set_xlabel("log(1 + TransactionAmt)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Transaction Amount Distribution (log scale)")
    axes[0].legend()

    # Box plots
    data = [legit, fraud]
    bp = axes[1].boxplot(data, labels=["Legitimate", "Fraud"], patch_artist=True,
                         showfliers=False)
    for patch, color in zip(bp["boxes"], ["#1976D2", "#D32F2F"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel("TransactionAmt (USD, clipped at $2,000)")
    axes[1].set_title("Transaction Amount Box Plot")
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    plt.suptitle("Transaction Amount vs Fraud Label", fontsize=13)
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "transaction_amount_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train / validation split
# ─────────────────────────────────────────────────────────────────────────────

def create_train_val_split(
    df_train: pd.DataFrame,
    val_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/20 split for training and validation.

    Saves split row counts to a summary file for reproducibility tracking
    (supports the AI audit trail requirement).

    Returns
    -------
    df_tr : Training set (80%)
    df_val: Validation set (20%)
    """
    df_tr, df_val = smart_split(
        df_train,
        target_col="isFraud",
        task_type="binary_classification",
        val_size=val_size,
    )

    # Save split metadata
    meta = {
        "train_rows":    len(df_tr),
        "val_rows":      len(df_val),
        "train_fraud_n": int(df_tr["isFraud"].sum()),
        "val_fraud_n":   int(df_val["isFraud"].sum()),
        "train_fraud_%": round(df_tr["isFraud"].mean() * 100, 3),
        "val_fraud_%":   round(df_val["isFraud"].mean() * 100, 3),
        "random_state":  RANDOM_STATE,
        "val_size":      val_size,
    }
    meta_df = pd.DataFrame([meta])
    meta_df.to_csv(REPORT_DIR / "split_metadata.csv", index=False)
    log.info(
        f"\nSplit → train: {len(df_tr):,} rows ({meta['train_fraud_%']}% fraud) | "
        f"val: {len(df_val):,} rows ({meta['val_fraud_%']}% fraud)"
    )
    return df_tr, df_val


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case A: Financial Crime & Fraud Analytics")
    print("  Step 1: Data Loading & Initial Inspection")
    print("=" * 65 + "\n")

    # 1. Load data
    # Set sample_frac=0.1 for quick iteration; None for full dataset
    df_train, df_test = load_ieee_fraud(
        force_download=False,
        optimize_memory=True,
        sample_frac=None,
    )

    if df_train is None:
        print(
            "\n[!] Training data not found. Follow the download instructions above.\n"
            "    After downloading, re-run:  python use_case_A_fraud/01_data_loading.py"
        )
        return None, None, None, None

    # 2. Inspect datasets
    train_profiler = inspect_dataset(df_train, label="train")
    if df_test is not None:
        inspect_dataset(df_test, label="test")

    # 3. Column group analysis
    analyze_column_groups(df_train)

    # 4. Visualisations
    print("\n[4] Generating visualisations…")
    plot_missing_by_group(df_train)
    plot_target_distribution(df_train)
    plot_transaction_amount(df_train)
    print(f"    Figures saved to: {REPORT_DIR}")

    # 5. Train / validation split
    print("\n[5] Creating train / validation split…")
    df_tr, df_val = create_train_val_split(df_train, val_size=0.20)

    print("\n" + "=" * 65)
    print("  Step 1 complete. Ready for EDA (02_eda_analysis.py)")
    print("=" * 65 + "\n")

    return df_train, df_test, df_tr, df_val


if __name__ == "__main__":
    df_train, df_test, df_tr, df_val = main()
