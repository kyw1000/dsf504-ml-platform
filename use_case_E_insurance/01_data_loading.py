"""
use_case_E_insurance/01_data_loading.py
========================================
Use Case E — Insurance Risk & Claims Analytics
Phase 2, Step 1: Data Acquisition & Initial Inspection

Dataset : Porto Seguro Safe Driver Prediction (Kaggle)
          https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
Files   : train.csv  (595,212 rows × 59 cols  — id + 57 features + target)
          test.csv   (892,816 rows × 58 cols   — id + 57 features, no target)
Target  : target  (binary: 1 = insurance claim filed, 0 = no claim)
Imbalance: ~3.6% positive (claim rate) — requires SMOTE / class-weight strategy

Feature naming convention
  ps_ind_*   — policyholder / driver characteristics
  ps_reg_*   — registration / geographic features
  ps_car_*   — vehicle-related features
  ps_calc_*  — calculated features (known to be uninformative / random)
  *_bin      — binary indicator (0/1)
  *_cat      — categorical (integer-coded; -1 = missing)
  others     — continuous

Missing values: encoded as -1 (not NaN) in the raw CSV files.

ML Framework Phase: Data Gathering and Preprocessing
     └─ Gather Data / ETL
     └─ Apply Data Cleansing
     └─ Protect Data Privacy and Security
     └─ Perform Feature Extraction
     └─ Split Data Sets

Run
---
    cd C:\\DSF504
    python use_case_E_insurance/01_data_loading.py
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
import seaborn as sns

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE, DATASET_REGISTRY
from utils.data_loader import KaggleLoader, DataProfiler, smart_split, reduce_mem_usage

# ── UTF-8 encoding guard (fixes garbled output on Windows) ───────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_SUBDIR  = DATA_DIR / "porto_seguro"
REPORT_DIR   = REPORTS_DIR / "use_case_E"
DATA_SUBDIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "target"

# ─────────────────────────────────────────────────────────────────────────────
# Porto Seguro feature groups
# ─────────────────────────────────────────────────────────────────────────────
def _get_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Partition Porto Seguro columns into domain groups based on naming convention.
    """
    feat_cols = [c for c in df.columns if c not in ("id", TARGET_COL)]
    groups: dict[str, list[str]] = {
        "ind_bin":  [],
        "ind_cat":  [],
        "ind_cont": [],
        "reg":      [],
        "car_bin":  [],
        "car_cat":  [],
        "car_cont": [],
        "calc_bin": [],
        "calc_cont":[],
    }
    for c in feat_cols:
        if c.startswith("ps_ind"):
            if c.endswith("_bin"):
                groups["ind_bin"].append(c)
            elif c.endswith("_cat"):
                groups["ind_cat"].append(c)
            else:
                groups["ind_cont"].append(c)
        elif c.startswith("ps_reg"):
            groups["reg"].append(c)
        elif c.startswith("ps_car"):
            if c.endswith("_bin"):
                groups["car_bin"].append(c)
            elif c.endswith("_cat"):
                groups["car_cat"].append(c)
            else:
                groups["car_cont"].append(c)
        elif c.startswith("ps_calc"):
            if c.endswith("_bin"):
                groups["calc_bin"].append(c)
            else:
                groups["calc_cont"].append(c)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_porto_seguro(
    force_download: bool = False,
    optimize_memory: bool = True,
    sample_frac: float | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load the Porto Seguro Safe Driver Prediction dataset.

    Parameters
    ----------
    force_download  : Re-download from Kaggle even if local files exist.
    optimize_memory : Downcast numerics to reduce RAM footprint.
    sample_frac     : Stratified fraction for rapid development. None = full.

    Returns
    -------
    df_train : Training DataFrame  (595,212 rows × 59 cols)
    df_test  : Test DataFrame      (892,816 rows × 58 cols, no target)
    """
    # Load directly from CSV/Parquet (bypasses KaggleLoader Kaggle-API dependency)
    train_parquet = DATA_SUBDIR / "train.parquet"
    test_parquet  = DATA_SUBDIR / "test.parquet"
    train_csv     = DATA_SUBDIR / "train.csv"
    test_csv      = DATA_SUBDIR / "test.csv"

    if train_parquet.exists() and not force_download:
        log.info(f"Loading cached Parquet: {train_parquet}")
        df_train = pd.read_parquet(train_parquet)
    elif train_csv.exists():
        log.info(f"Loading CSV: {train_csv}  (this may take a moment…)")
        df_train = pd.read_csv(train_csv)
        df_train.to_parquet(train_parquet, index=False)
        log.info(f"Cached as Parquet: {train_parquet}")
    else:
        _print_manual_download_instructions()
        return None, None

    if test_parquet.exists() and not force_download:
        df_test = pd.read_parquet(test_parquet)
    elif test_csv.exists():
        df_test = pd.read_csv(test_csv)
        df_test.to_parquet(test_parquet, index=False)
    else:
        df_test = None
        log.warning("test.csv not found — continuing with train only.")

    # Optimise memory
    if optimize_memory:
        df_train = reduce_mem_usage(df_train, verbose=True)
        if df_test is not None:
            df_test = reduce_mem_usage(df_test, verbose=False)

    # Optional stratified sample for rapid iteration
    if sample_frac is not None and 0 < sample_frac < 1.0:
        pos = df_train[df_train[TARGET_COL] == 1]
        neg = df_train[df_train[TARGET_COL] == 0]
        pos_s = pos.sample(frac=sample_frac, random_state=RANDOM_STATE)
        neg_s = neg.sample(frac=sample_frac, random_state=RANDOM_STATE)
        df_train = (
            pd.concat([pos_s, neg_s])
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        log.info(
            f"Sampled {sample_frac:.0%}: {len(df_train):,} rows "
            f"({len(pos_s):,} claims, {len(neg_s):,} no-claim)"
        )

    return df_train, df_test


def _print_manual_download_instructions() -> None:
    print(
        "\n" + "=" * 65 + "\n"
        "  MANUAL DOWNLOAD INSTRUCTIONS\n"
        "=" * 65 + "\n"
        "  Dataset: Porto Seguro Safe Driver Prediction\n\n"
        "  1. Create a Kaggle account at https://kaggle.com\n"
        "  2. Accept the competition rules at:\n"
        "     https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data\n"
        "  3. Download:\n"
        "       - train.csv\n"
        "       - test.csv\n"
        f"  4. Place them in: {DATA_SUBDIR}\n"
        "  5. Re-run this script.\n\n"
        "  Alternatively, use the Kaggle API:\n"
        "    pip install kaggle\n"
        "    mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/\n"
        "    kaggle competitions download -c porto-seguro-safe-driver-prediction\n"
        "=" * 65 + "\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Initial data inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_dataset(df: pd.DataFrame, label: str = "train") -> DataProfiler:
    """
    Run DataProfiler, save column-level summary CSV.
    Missing -1 values are converted to NaN before profiling so statistics
    reflect the true missing rate.
    """
    log.info(f"\n{'='*55}\nDataset inspection: {label.upper()}\n{'='*55}")

    # Replace -1 with NaN for profiling (Porto Seguro convention)
    df_prof = df.copy()
    df_prof = df_prof.replace(-1, np.nan)

    tgt = TARGET_COL if label == "train" else None
    profiler = DataProfiler(df_prof, target_col=tgt)
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
# 3. Feature group analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_feature_groups(df: pd.DataFrame) -> dict:
    """
    Summarise missingness and cardinality for each Porto Seguro feature group.

    Porto Seguro anonymises all column names, so we rely entirely on naming
    conventions (prefix + suffix) to understand domain meaning.
    """
    groups = _get_feature_groups(df)
    stats: dict[str, dict] = {}

    print("\n--- Porto Seguro Feature Group Analysis ---")
    for group_name, cols in groups.items():
        if not cols:
            continue
        subset = df[cols].replace(-1, np.nan)
        n_cols   = len(cols)
        avg_miss = subset.isna().mean().mean() * 100
        n_unique = int(subset.nunique().mean())

        stats[group_name] = {
            "n_columns":     n_cols,
            "avg_missing%":  round(avg_miss, 1),
            "avg_n_unique":  n_unique,
        }
        print(
            f"  {group_name:<12} | {n_cols:>3} cols | "
            f"avg missing: {avg_miss:5.1f}% | avg unique: {n_unique:>4}"
        )

    # Note: ps_calc_* is known to carry no predictive signal
    print(
        "\n  [!] ps_calc_* features are known to be uninformative"
        " — will be dropped in Step 3."
    )
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """
    Visualise the class imbalance in insurance claims.
    ~3.6% of policyholders file a claim — motivates SMOTE and Gini metric.
    """
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["No Claim (0)", "Claim Filed (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    pcts   = [100 * v / len(df) for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1976D2", "#D32F2F"]

    # Bar chart
    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, pct in zip(bars, pcts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2000,
            f"{pct:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    axes[0].set_title("Claim Distribution", fontsize=13)
    axes[0].set_ylabel("Count")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Pie chart
    axes[1].pie(
        values, labels=labels, autopct="%1.2f%%",
        colors=colors, explode=(0, 0.1), startangle=140,
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Class Imbalance (Claims = 3.6%)", fontsize=13)

    fig.suptitle(
        "Target Distribution — Porto Seguro Safe Driver\n"
        "Implication: Use Gini/ROC-AUC, not accuracy; apply SMOTE or class weights",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "target_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


def plot_missing_by_group(df: pd.DataFrame, save: bool = True) -> None:
    """
    Bar chart of average % missing (where -1 = missing) per feature group.
    """
    groups = _get_feature_groups(df)
    rows = []
    for group_name, cols in groups.items():
        if not cols:
            continue
        subset = df[cols].replace(-1, np.nan)
        avg_miss = subset.isna().mean().mean() * 100
        rows.append({"group": group_name, "avg_missing_pct": avg_miss})

    gdf = pd.DataFrame(rows).sort_values("avg_missing_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [
        "#d32f2f" if v > 50 else "#f57c00" if v > 10 else "#388e3c"
        for v in gdf["avg_missing_pct"]
    ]
    ax.barh(gdf["group"], gdf["avg_missing_pct"], color=colors)
    ax.axvline(x=10, color="orange", linestyle="--", alpha=0.6, label="10% threshold")
    ax.set_xlabel("Average % Missing Values  (−1 treated as missing)")
    ax.set_title("Porto Seguro: Average Missing Values by Feature Group")
    ax.legend()
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "missing_by_group.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


def plot_feature_group_counts(df: pd.DataFrame, save: bool = True) -> None:
    """
    Stacked bar showing number of features per domain prefix × type suffix.
    Gives a structural overview of the anonymised feature space.
    """
    groups = _get_feature_groups(df)
    cats = list(groups.keys())
    cnts = [len(groups[g]) for g in cats]

    colors = [
        "#1565C0", "#1E88E5", "#42A5F5",  # ind
        "#2E7D32", "#43A047", "#66BB6A",  # reg + car
        "#E65100", "#FF7043",             # calc
    ][:len(cats)]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(cats, cnts, color=colors, edgecolor="white")
    for bar, cnt in zip(bars, cnts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(cnt), ha="center", va="bottom", fontsize=10
        )
    ax.set_xlabel("Feature Group")
    ax.set_ylabel("Number of Features")
    ax.set_title("Porto Seguro: Feature Count by Group")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "feature_group_counts.png"
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
    Stratified 80/20 split; saves Parquet + split metadata.

    Porto Seguro has a significant class imbalance (~3.6% positives).
    Stratification preserves the imbalance ratio across train and val sets.
    """
    df_tr, df_val = smart_split(
        df_train,
        target_col=TARGET_COL,
        task_type="binary_classification",
        val_size=val_size,
    )

    # Save as Parquet for fast downstream loading
    df_tr.to_parquet(DATA_SUBDIR / "train_raw.parquet",  index=False)
    df_val.to_parquet(DATA_SUBDIR / "val_raw.parquet",   index=False)
    df_train.to_parquet(DATA_SUBDIR / "train_full.parquet", index=False)
    log.info(f"Parquet splits saved to {DATA_SUBDIR}")

    # Split metadata
    meta = {
        "train_rows":     len(df_tr),
        "val_rows":       len(df_val),
        "train_claims_n": int(df_tr[TARGET_COL].sum()),
        "val_claims_n":   int(df_val[TARGET_COL].sum()),
        "train_claim_%":  round(df_tr[TARGET_COL].mean() * 100, 3),
        "val_claim_%":    round(df_val[TARGET_COL].mean() * 100, 3),
        "random_state":   RANDOM_STATE,
        "val_size":       val_size,
    }
    pd.DataFrame([meta]).to_csv(REPORT_DIR / "split_metadata.csv", index=False)
    log.info(
        f"\nSplit → train: {len(df_tr):,} rows ({meta['train_claim_%']}% claims) | "
        f"val: {len(df_val):,} rows ({meta['val_claim_%']}% claims)"
    )
    return df_tr, df_val


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case E: Insurance Risk & Claims Analytics")
    print("  Step 1: Data Loading & Initial Inspection")
    print("=" * 65 + "\n")

    # 1. Load data
    df_train, df_test = load_porto_seguro(
        force_download=False,
        optimize_memory=True,
        sample_frac=None,
    )

    if df_train is None:
        print(
            "\n[!] Training data not found. Follow the download instructions above.\n"
            "    After downloading, re-run:  python use_case_E_insurance/01_data_loading.py"
        )
        return None, None, None, None

    print(f"\n[1] Loaded train: {df_train.shape}  |  test: {df_test.shape if df_test is not None else 'N/A'}")
    print(f"    Claim rate: {df_train[TARGET_COL].mean():.3%}")

    # 2. Inspect
    print("\n[2] Profiling dataset…")
    train_profiler = inspect_dataset(df_train, label="train")
    if df_test is not None:
        inspect_dataset(df_test, label="test")

    # 3. Feature group analysis
    print("\n[3] Feature group analysis…")
    analyze_feature_groups(df_train)

    # 4. Visualisations
    print("\n[4] Generating visualisations…")
    plot_target_distribution(df_train)
    plot_missing_by_group(df_train)
    plot_feature_group_counts(df_train)
    print(f"    Figures saved to: {REPORT_DIR}")

    # 5. Split
    print("\n[5] Creating train / validation split…")
    df_tr, df_val = create_train_val_split(df_train, val_size=0.20)

    print("\n" + "=" * 65)
    print("  Step 1 complete. Ready for EDA (02_eda_analysis.py)")
    print("=" * 65 + "\n")

    return df_train, df_test, df_tr, df_val


if __name__ == "__main__":
    df_train, df_test, df_tr, df_val = main()
