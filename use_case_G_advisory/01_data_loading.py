"""
use_case_G_advisory/01_data_loading.py
========================================
Use Case G — AmEx Credit Default Prediction
Phase 2, Step 1: Data Acquisition & Initial Inspection

Dataset : American Express Default Prediction (Kaggle 2022)
          https://www.kaggle.com/competitions/amex-default-prediction
Files   : train_data.csv  (~50M rows — 458,913 customers × up to 13 monthly statements)
          train_labels.csv (458,913 rows — customer_ID, target)
          test_data.csv   (~11.4M rows — 924,621 customers, no target)
Target  : target (binary: 1 = default within 18 months, 0 = no default)
Imbalance: ~25.9% positive (default rate)

Feature groups (190 numeric + 2 categorical):
  D_*  — Delinquency variables      (e.g. D_39, D_41, D_42, …)
  S_*  — Spend variables            (e.g. S_2 = statement date, S_3, …)
  P_*  — Payment variables          (e.g. P_2 = payment amount)
  B_*  — Balance variables          (e.g. B_1, B_2, …)
  R_*  — Risk variables             (e.g. R_1, R_2, …)
  cat  — D_63, D_64 (categorical)

Time-series structure: each customer_ID appears in up to 13 monthly rows
(chronological order). Step 3 will aggregate these into one row per customer.

Competition metric: AmEx M = 0.5 × (Gini + D-rate@4%)
  Gini = Normalized Gini coefficient (= 2×AUC−1)
  D-rate@4% = Default capture rate in the top 4% of score decile

Winner insights (2022):
  1st place: LGB + GRU ensemble, denoise with np.floor(x*100)/100,
             rich time-series aggregation + diff + rank features
  2nd place: Team ensemble — LGB + CatBoost + NN (extensive infrastructure)
  3rd place: "Feature engineering is all you need" — 5,034 features from
             basic stats, diff features, last-3/6M stats, bin-unique counts

Run
---
    cd C:\\DSF504
    python use_case_G_advisory/01_data_loading.py
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

warnings.filterwarnings("ignore")

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE, DATASET_REGISTRY
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
DATA_SUBDIR = DATA_DIR / "amex_default"
REPORT_DIR  = REPORTS_DIR / "use_case_G"
DATA_SUBDIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL  = "target"
ID_COL      = "customer_ID"

# AmEx feature group prefixes
CAT_COLS    = ["D_63", "D_64"]
NUM_PREFIXES = {"D": "Delinquency", "S": "Spend", "P": "Payment",
                "B": "Balance", "R": "Risk"}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (used when real data not available)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_amex(
    n_customers: int = 5000,
    max_statements: int = 13,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic AmEx-like dataset for local development and teaching.

    Structure mirrors the real competition:
    - Each customer has 1–13 monthly statements (time-series rows)
    - 20 representative numeric features (D, S, P, B, R groups)
    - 2 categorical features: D_63 (3 levels), D_64 (2 levels)
    - Binary target with ~25.9% default rate (matches competition statistics)

    The synthetic data captures key real-world patterns:
    - Balance drift: defaulters show rising balances over time
    - Payment decline: defaulters show falling payment amounts pre-default
    - Delinquency escalation: defaulters accumulate more late payments
    - Spend volatility: higher volatility in defaulter spending patterns

    Parameters
    ----------
    n_customers : Number of synthetic customers to generate
    max_statements : Maximum monthly statements per customer (1–13)
    seed : Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # Assign targets: 25.9% default rate
    n_default = int(n_customers * 0.259)
    targets = np.array([1] * n_default + [0] * (n_customers - n_default))
    rng.shuffle(targets)

    # Statement counts (1–13 per customer, skewed toward 13)
    stmt_counts = rng.choice(
        range(1, max_statements + 1), size=n_customers,
        p=[0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.10, 0.14, 0.18, 0.25],
    )

    rows = []
    for cid, (label, n_stmts) in enumerate(zip(targets, stmt_counts)):
        customer_id = f"CUST_{cid:06d}"
        # Base financial profile
        base_balance  = rng.uniform(500, 15000)
        base_payment  = rng.uniform(50, 2000)
        base_spend    = rng.uniform(100, 3000)
        base_risk     = rng.uniform(0.1, 0.8) if label == 1 else rng.uniform(0.05, 0.4)

        for t in range(n_stmts):
            # Time-varying drift — defaulters show deterioration
            drift = (t / max_statements) * label * rng.uniform(0.3, 0.8)

            row: dict = {
                ID_COL:   customer_id,
                "S_2":    pd.Timestamp("2017-03-01") + pd.DateOffset(months=t),
                # Delinquency features
                "D_39":   max(0, rng.poisson(5 * label + t * label * 0.5)),
                "D_41":   max(0, rng.poisson(3 * label + t * label * 0.3)),
                "D_42":   max(0, rng.poisson(2 * label)),
                "D_44":   rng.uniform(0, 1 + drift),
                "D_45":   rng.integers(0, 6 + int(4 * label)),
                # Spend features
                "S_3":    max(0, base_spend * (1 + drift * 0.5) + rng.normal(0, base_spend * 0.2)),
                "S_5":    rng.uniform(0.2, 0.8),
                "S_7":    rng.uniform(0, 1),
                "S_9":    rng.uniform(0.01, 0.15),
                "S_11":   max(0, base_spend * rng.uniform(0.1, 0.5)),
                # Payment features
                "P_2":    max(0, base_payment * (1 - drift * 0.6) + rng.normal(0, base_payment * 0.15)),
                "P_3":    rng.uniform(0, 0.5),
                # Balance features
                "B_1":    max(0, base_balance * (1 + drift * 0.7) + rng.normal(0, base_balance * 0.1)),
                "B_2":    max(0, base_balance * rng.uniform(0.8, 1.2)),
                "B_3":    rng.uniform(0.1, 0.9 + 0.1 * label),
                "B_4":    rng.uniform(0, 1),
                "B_5":    rng.uniform(0, 1 + drift * 0.3),
                # Risk features
                "R_1":    min(1, base_risk + drift * 0.2 + rng.normal(0, 0.05)),
                "R_2":    rng.uniform(0, 1),
                "R_3":    min(1, base_risk * rng.uniform(0.8, 1.2)),
                # Categorical
                "D_63":   rng.choice(["CR", "CL", "XZ"], p=[0.5, 0.35, 0.15]),
                "D_64":   rng.choice(["R", "U"],          p=[0.7, 0.3]),
            }
            rows.append(row)

    df_train = pd.DataFrame(rows)

    # Labels
    customer_ids = [f"CUST_{i:06d}" for i in range(n_customers)]
    df_labels = pd.DataFrame({
        ID_COL:    customer_ids,
        TARGET_COL: targets,
    })

    log.info(
        f"Synthetic data: {len(df_train):,} rows | {n_customers:,} customers "
        f"| default rate: {df_labels[TARGET_COL].mean():.1%}"
    )
    return df_train, df_labels


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_amex_data(
    force_download: bool = False,
    use_synthetic: bool = False,
    n_synthetic: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the AmEx Default Prediction dataset.

    Priority order:
      1. Cached Parquet files (fastest)
      2. Raw CSV files (downloads from Kaggle if kaggle.json configured)
      3. Synthetic data (for local development / teaching)

    Returns
    -------
    df_train  : Time-series data (multiple rows per customer)
    df_labels : Customer-level labels (one row per customer)
    """
    train_parquet  = DATA_SUBDIR / "train_data.parquet"
    labels_parquet = DATA_SUBDIR / "train_labels.parquet"
    train_csv      = DATA_SUBDIR / "train_data.csv"
    labels_csv     = DATA_SUBDIR / "train_labels.csv"

    if use_synthetic:
        log.info("Generating synthetic AmEx dataset…")
        df_train, df_labels = generate_synthetic_amex(n_customers=n_synthetic)
        df_train.to_parquet(DATA_SUBDIR / "train_data_synthetic.parquet", index=False)
        df_labels.to_parquet(DATA_SUBDIR / "train_labels_synthetic.parquet", index=False)
        return df_train, df_labels

    # Try cached Parquet
    if train_parquet.exists() and labels_parquet.exists() and not force_download:
        log.info("Loading cached Parquet files…")
        df_train  = pd.read_parquet(train_parquet)
        df_labels = pd.read_parquet(labels_parquet)
        return df_train, df_labels

    # Try CSV
    if train_csv.exists() and labels_csv.exists():
        log.info("Loading CSVs (this may take several minutes for the full dataset)…")
        df_train  = pd.read_csv(train_csv)
        df_labels = pd.read_csv(labels_csv)
        df_train.to_parquet(train_parquet, index=False)
        df_labels.to_parquet(labels_parquet, index=False)
        log.info("Parquet cache written.")
        return df_train, df_labels

    # Fall back to synthetic
    log.warning(
        "Real data not found. Generating synthetic dataset for demonstration.\n"
        "  To use real data: download from\n"
        "  https://www.kaggle.com/competitions/amex-default-prediction/data\n"
        f"  and place train_data.csv + train_labels.csv in {DATA_SUBDIR}"
    )
    _print_download_instructions()
    df_train, df_labels = generate_synthetic_amex(n_customers=n_synthetic)
    return df_train, df_labels


def _print_download_instructions() -> None:
    print(
        "\n" + "=" * 65 + "\n"
        "  MANUAL DOWNLOAD INSTRUCTIONS\n"
        "=" * 65 + "\n"
        "  Dataset: American Express Default Prediction\n\n"
        "  1. Create a Kaggle account at https://kaggle.com\n"
        "  2. Accept the competition rules at:\n"
        "     https://www.kaggle.com/competitions/amex-default-prediction/data\n"
        "  3. Download:\n"
        "       - train_data.csv   (~5.5 GB)\n"
        "       - train_labels.csv (~10 MB)\n"
        "       - test_data.csv    (~1.4 GB)\n"
        f"  4. Place them in: {DATA_SUBDIR}\n"
        "  5. Re-run this script.\n\n"
        "  Alternatively, use the Kaggle API:\n"
        "    kaggle competitions download -c amex-default-prediction\n"
        "=" * 65 + "\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_dataset(df_train: pd.DataFrame, df_labels: pd.DataFrame) -> dict:
    """
    Profile the time-series dataset structure and basic statistics.
    Returns a summary dict for downstream use.
    """
    n_rows      = len(df_train)
    n_customers = df_train[ID_COL].nunique()
    n_features  = len([c for c in df_train.columns if c not in (ID_COL, "S_2")])
    stmts_per_cust = df_train.groupby(ID_COL).size()

    default_rate = df_labels[TARGET_COL].mean()
    n_default    = df_labels[TARGET_COL].sum()

    print(f"\n{'='*55}")
    print(f"  AmEx Dataset Overview")
    print(f"{'='*55}")
    print(f"  Total rows         : {n_rows:>12,}")
    print(f"  Unique customers   : {n_customers:>12,}")
    print(f"  Features           : {n_features:>12,}")
    print(f"  Default rate       : {default_rate:>11.1%}")
    print(f"  Defaulters         : {int(n_default):>12,}")
    print(f"  Non-defaulters     : {int(len(df_labels) - n_default):>12,}")
    print(f"  Statements/customer: {stmts_per_cust.mean():>11.1f} (avg)")
    print(f"    min={stmts_per_cust.min()}, max={stmts_per_cust.max()}, "
          f"median={stmts_per_cust.median()}")
    print(f"{'='*55}\n")

    # Feature group breakdown
    numeric_cols = [c for c in df_train.columns
                    if c not in (ID_COL, "S_2") + tuple(CAT_COLS)]
    group_counts = {}
    for col in numeric_cols:
        prefix = col.split("_")[0]
        group_counts[prefix] = group_counts.get(prefix, 0) + 1

    print("  Feature groups:")
    for prefix, label in NUM_PREFIXES.items():
        cnt = group_counts.get(prefix, 0)
        if cnt:
            print(f"    {prefix}_* ({label:<12}): {cnt:>4} features")
    print(f"    Categorical         :    2  (D_63, D_64)")

    # Missing value summary
    miss_pct = df_train[numeric_cols].isna().mean()
    miss_high = (miss_pct > 0.10).sum()
    print(f"\n  Features >10% missing: {miss_high}")

    summary = {
        "n_rows":       n_rows,
        "n_customers":  n_customers,
        "n_features":   n_features,
        "default_rate": float(default_rate),
        "stmts_mean":   float(stmts_per_cust.mean()),
        "stmts_max":    int(stmts_per_cust.max()),
        "group_counts": group_counts,
    }

    # Save column summary CSV
    col_summary = pd.DataFrame({
        "column":   [c for c in df_train.columns],
        "dtype":    [str(df_train[c].dtype) for c in df_train.columns],
        "null_pct": [round(df_train[c].isna().mean() * 100, 2) for c in df_train.columns],
        "n_unique": [df_train[c].nunique() for c in df_train.columns],
    })
    col_summary.to_csv(REPORT_DIR / "train_column_summary.csv", index=False)
    log.info(f"Column summary → {REPORT_DIR / 'train_column_summary.csv'}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(df_labels: pd.DataFrame, save: bool = True) -> None:
    """Class imbalance chart for the binary default target."""
    counts  = df_labels[TARGET_COL].value_counts().sort_index()
    labels  = ["No Default (0)", "Default (1)"]
    values  = [counts.get(0, 0), counts.get(1, 0)]
    pcts    = [100 * v / len(df_labels) for v in values]
    colors  = ["#1976D2", "#D32F2F"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, pct in zip(bars, pcts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    axes[0].set_title("Default Distribution (Customer Level)", fontsize=13)
    axes[0].set_ylabel("Number of Customers")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    axes[1].pie(values, labels=labels, autopct="%1.1f%%", colors=colors,
                explode=(0, 0.08), startangle=140, textprops={"fontsize": 11})
    axes[1].set_title("AmEx Default Rate ≈ 25.9%", fontsize=13)

    fig.suptitle(
        "Target Distribution — AmEx Default Prediction\n"
        "Less severe than fraud (~3.5%) but significant. Primary metric: AmEx M = 0.5×(Gini + D-rate@4%)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    if save:
        path = REPORT_DIR / "target_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


def plot_statements_per_customer(df_train: pd.DataFrame, save: bool = True) -> None:
    """
    Distribution of monthly statement counts per customer.
    This is critical for understanding the time-series nature of the data —
    customers with fewer statements are harder to model.
    """
    stmt_counts = df_train.groupby(ID_COL).size()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(stmt_counts, bins=range(1, 15), color="#3949AB", edgecolor="white", align="left")
    ax.set_xlabel("Number of Monthly Statements")
    ax.set_ylabel("Number of Customers")
    ax.set_title(
        "Statements per Customer — AmEx Time-Series Structure\n"
        "Most customers have 13 statements (full observation window)"
    )
    ax.set_xticks(range(1, 14))
    mean_stmts = stmt_counts.mean()
    ax.axvline(mean_stmts, color="#D32F2F", linestyle="--", label=f"Mean: {mean_stmts:.1f}")
    ax.legend()
    plt.tight_layout()
    if save:
        path = REPORT_DIR / "statements_per_customer.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


def plot_feature_group_counts(df_train: pd.DataFrame, save: bool = True) -> None:
    """Bar chart of feature counts by group prefix."""
    numeric_cols = [c for c in df_train.columns
                    if c not in (ID_COL, "S_2") + tuple(CAT_COLS)]
    groups: dict[str, int] = {}
    for col in numeric_cols:
        prefix = col.split("_")[0]
        groups[prefix] = groups.get(prefix, 0) + 1
    groups["cat"] = 2

    palette = {"D": "#EF5350", "S": "#42A5F5", "P": "#66BB6A",
               "B": "#FFA726", "R": "#AB47BC", "cat": "#78909C"}
    labels_map = {"D": "Delinquency (D_*)", "S": "Spend (S_*)",
                  "P": "Payment (P_*)",   "B": "Balance (B_*)",
                  "R": "Risk (R_*)",      "cat": "Categorical"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (prefix, cnt) in enumerate(groups.items()):
        ax.bar(labels_map.get(prefix, prefix), cnt,
               color=palette.get(prefix, "#9E9E9E"), edgecolor="white")
        ax.text(i, cnt + 0.5, str(cnt), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Feature Group")
    ax.set_ylabel("Number of Features")
    ax.set_title("AmEx: Feature Count by Group\n(190 numeric + 2 categorical = 192 total)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save:
        path = REPORT_DIR / "feature_group_counts.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Train / validation split (customer level, stratified)
# ─────────────────────────────────────────────────────────────────────────────

def create_customer_split(
    df_train: pd.DataFrame,
    df_labels: pd.DataFrame,
    val_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified customer-level 80/20 split.

    CRITICAL: The split must be done at the customer level (not row level).
    Splitting at the row level would allow statements from the same customer
    to appear in both train and validation sets, causing severe data leakage.

    The validation set retains the same default rate as the full dataset.
    """
    from sklearn.model_selection import train_test_split

    cust_labels = df_labels.set_index(ID_COL)[TARGET_COL]

    train_ids, val_ids = train_test_split(
        cust_labels.index.tolist(),
        test_size=val_size,
        stratify=cust_labels.values,
        random_state=RANDOM_STATE,
    )

    train_set = set(train_ids)
    val_set   = set(val_ids)

    df_ts_train = df_train[df_train[ID_COL].isin(train_set)].copy()
    df_ts_val   = df_train[df_train[ID_COL].isin(val_set)].copy()
    df_lb_train = df_labels[df_labels[ID_COL].isin(train_set)].copy()
    df_lb_val   = df_labels[df_labels[ID_COL].isin(val_set)].copy()

    # Save splits
    df_ts_train.to_parquet(DATA_SUBDIR / "train_raw.parquet",     index=False)
    df_ts_val.to_parquet(DATA_SUBDIR  / "val_raw.parquet",        index=False)
    df_lb_train.to_parquet(DATA_SUBDIR / "train_labels_raw.parquet", index=False)
    df_lb_val.to_parquet(DATA_SUBDIR   / "val_labels_raw.parquet",   index=False)

    # Metadata
    meta = {
        "train_customers": len(train_ids),
        "val_customers":   len(val_ids),
        "train_rows":      len(df_ts_train),
        "val_rows":        len(df_ts_val),
        "train_default_%": round(df_lb_train[TARGET_COL].mean() * 100, 3),
        "val_default_%":   round(df_lb_val[TARGET_COL].mean()   * 100, 3),
        "val_size":        val_size,
        "random_state":    RANDOM_STATE,
    }
    pd.DataFrame([meta]).to_csv(REPORT_DIR / "split_metadata.csv", index=False)

    log.info(
        f"Customer split → train: {len(train_ids):,} ({meta['train_default_%']}% default) "
        f"| val: {len(val_ids):,} ({meta['val_default_%']}% default)"
    )
    return df_ts_train, df_ts_val, df_lb_train, df_lb_val


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G: AmEx Credit Default Prediction")
    print("  Step 1: Data Loading & Initial Inspection")
    print("=" * 65 + "\n")

    # Load (uses synthetic if real data not present)
    df_train, df_labels = load_amex_data()

    if df_train is None or df_labels is None:
        print("[!] Data loading failed. Exiting.")
        return

    print(f"[1] Loaded — time-series: {df_train.shape}  |  labels: {df_labels.shape}")

    # Inspect
    print("\n[2] Dataset inspection…")
    summary = inspect_dataset(df_train, df_labels)

    # Visualisations
    print("\n[3] Generating visualisations…")
    plot_target_distribution(df_labels)
    plot_statements_per_customer(df_train)
    plot_feature_group_counts(df_train)
    print(f"    Figures saved → {REPORT_DIR}")

    # Customer-level split
    print("\n[4] Creating stratified customer-level train/val split…")
    df_ts_train, df_ts_val, df_lb_train, df_lb_val = create_customer_split(
        df_train, df_labels, val_size=0.20
    )

    print("\n" + "=" * 65)
    print("  Step 1 complete. Ready for EDA (02_eda_analysis.py)")
    print("=" * 65 + "\n")

    return df_train, df_labels, df_ts_train, df_ts_val, df_lb_train, df_lb_val


if __name__ == "__main__":
    main()
