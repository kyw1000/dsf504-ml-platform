"""
use_case_D_churn/01_data_loading.py
=====================================
DSF504 — Use Case D: Customer Churn Prediction
Step 1: Data loading, merging, profiling, and train/validation split

Dataset
-------
WSDM - KKBox's Churn Prediction Challenge — v2 refresh (Kaggle, Nov 2017)
  KKBox is Asia's leading music-streaming subscription service.
  ~971K subscribers · target: is_churn (churned within 30 days)
  https://www.kaggle.com/c/kkbox-churn-prediction-challenge

Source files (all from data/kkbox_churn/)
------------------------------------------
  data/churn_comp_refresh/train_v2.csv         — labels: msno, is_churn (~971K rows)
  data/churn_comp_refresh/transactions_v2.csv  — subscription events (~1.4M rows)
  data/churn_comp_refresh/user_logs_v2.csv     — daily listening (~18.4M rows, 1.4 GB)
  members_v3.csv                               — demographics: city, bd, gender, registered_via

Why v2 only (not v1)?
----------------------
  The root user_logs.csv is 29 GB — impractical to aggregate locally.
  The v2 files cover a consistent Nov 2017 observation window and are
  self-contained (~1.4 GB total). members_v3.csv is shared across both.

Key challenges handled here
----------------------------
- Four files joined on msno (hashed user ID)
- user_logs_v2 and transactions_v2 aggregated per user before merge
- bd (age) has erroneous values (0, negatives, >100)
- gender has many NaN values
- Moderate class imbalance: ~8.4% churn rate

Academic references
-------------------
- Verbeke et al. (2012). New insights into churn prediction in
  the telecommunication sector. EJOR.
- Hadden et al. (2007). Computer assisted customer churn management.
  Expert Systems with Applications.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE
from utils.data_loader import DataProfiler, smart_split, reduce_mem_usage

from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_SUBDIR = DATA_DIR / "kkbox_churn"
V2_DIR      = DATA_SUBDIR / "data" / "churn_comp_refresh"
REPORT_DIR  = REPORTS_DIR / "use_case_D"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_SUBDIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_churn"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Download the KKBox dataset from:\n"
            "  https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data\n"
            "and place the 'data/churn_comp_refresh/' folder and 'members_v3.csv'\n"
            f"inside:  {DATA_SUBDIR}"
        )
    return path


# ── 1. Load source files ───────────────────────────────────────────────────────

def load_train_labels() -> pd.DataFrame:
    path = _require(V2_DIR / "train_v2.csv")
    log.info("Loading train labels from %s…", path.name)
    df = pd.read_csv(path)
    log.info("  %s rows loaded", f"{len(df):,}")
    return df


def load_members() -> pd.DataFrame:
    path = _require(DATA_SUBDIR / "members_v3.csv")
    log.info("Loading members from %s…", path.name)
    df = pd.read_csv(path)
    # Clean erroneous age values (0, negatives, implausibly high)
    df["bd"] = df["bd"].where((df["bd"] >= 7) & (df["bd"] <= 100), other=np.nan)
    df["gender"] = df["gender"].fillna("unknown")
    log.info("  %s members loaded", f"{len(df):,}")
    return df


def aggregate_transactions(chunk_size: int = 200_000) -> pd.DataFrame:
    """Aggregate transactions_v2 per user."""
    parq = DATA_SUBDIR / "transactions_v2_agg.parquet"
    if parq.exists():
        log.info("Loading transactions_v2 aggregate from cache…")
        return pd.read_parquet(parq)

    path = _require(V2_DIR / "transactions_v2.csv")
    log.info("Aggregating %s (chunked)…", path.name)

    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunk["transaction_date"]       = pd.to_numeric(chunk["transaction_date"],       errors="coerce")
        chunk["membership_expire_date"] = pd.to_numeric(chunk["membership_expire_date"], errors="coerce")
        chunks.append(chunk)
    txn = pd.concat(chunks, ignore_index=True)
    log.info("  %s transaction rows loaded", f"{len(txn):,}")

    # Compute discount rate safely per group
    grp = txn.groupby("msno")
    plan_sum   = grp["plan_list_price"].sum()
    actual_sum = grp["actual_amount_paid"].sum()
    discount   = 1 - (actual_sum / plan_sum.replace(0, np.nan))

    agg = grp.agg(
        txn_count        = ("payment_method_id",    "count"),
        plan_days_mean   = ("payment_plan_days",     "mean"),
        plan_price_mean  = ("plan_list_price",       "mean"),
        actual_paid_mean = ("actual_amount_paid",    "mean"),
        auto_renew_rate  = ("is_auto_renew",         "mean"),
        cancel_rate      = ("is_cancel",             "mean"),
        last_expire_date = ("membership_expire_date","max"),
        last_txn_date    = ("transaction_date",      "max"),
    ).reset_index()
    agg = agg.merge(discount.rename("discount_rate").reset_index(), on="msno", how="left")

    agg = reduce_mem_usage(agg, verbose=False)
    agg.to_parquet(parq, index=False)
    log.info("  Aggregated %s users → %s", f"{len(agg):,}", parq.name)
    return agg


def aggregate_user_logs(chunk_size: int = 500_000) -> pd.DataFrame:
    """Aggregate user_logs_v2 per user (chunked — ~18M rows / 1.4 GB)."""
    parq = DATA_SUBDIR / "user_logs_v2_agg.parquet"
    if parq.exists():
        log.info("Loading user_logs_v2 aggregate from cache…")
        return pd.read_parquet(parq)

    path = _require(V2_DIR / "user_logs_v2.csv")
    log.info("Aggregating %s — this may take a few minutes…", path.name)

    agg_parts = []
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
        part = chunk.groupby("msno").agg(
            log_days        = ("date",      "count"),
            num_25_mean     = ("num_25",    "mean"),
            num_50_mean     = ("num_50",    "mean"),
            num_75_mean     = ("num_75",    "mean"),
            num_985_mean    = ("num_985",   "mean"),
            num_100_mean    = ("num_100",   "mean"),
            num_unq_mean    = ("num_unq",   "mean"),
            total_secs_mean = ("total_secs","mean"),
            total_secs_sum  = ("total_secs","sum"),
        ).reset_index()
        agg_parts.append(part)
        if i % 5 == 0:
            log.info("  … chunk %d processed", i + 1)

    # Re-aggregate across chunks (simple mean of means — sufficient approximation)
    combined = pd.concat(agg_parts, ignore_index=True)
    agg = combined.groupby("msno").agg(
        log_days        = ("log_days",        "sum"),
        num_25_mean     = ("num_25_mean",     "mean"),
        num_50_mean     = ("num_50_mean",     "mean"),
        num_75_mean     = ("num_75_mean",     "mean"),
        num_985_mean    = ("num_985_mean",    "mean"),
        num_100_mean    = ("num_100_mean",    "mean"),
        num_unq_mean    = ("num_unq_mean",    "mean"),
        total_secs_mean = ("total_secs_mean", "mean"),
        total_secs_sum  = ("total_secs_sum",  "sum"),
    ).reset_index()

    agg = reduce_mem_usage(agg, verbose=False)
    agg.to_parquet(parq, index=False)
    log.info("  Aggregated %s users → %s", f"{len(agg):,}", parq.name)
    return agg


# ── 2. Merge ───────────────────────────────────────────────────────────────────

def build_master(train: pd.DataFrame, members: pd.DataFrame,
                 txn_agg: pd.DataFrame, log_agg: pd.DataFrame) -> pd.DataFrame:
    log.info("Merging datasets…")
    df = train.merge(members, on="msno", how="left")
    df = df.merge(txn_agg,   on="msno", how="left")
    df = df.merge(log_agg,   on="msno", how="left")
    log.info("  Master shape: %s  |  churn rate: %.3%%", df.shape, df[TARGET].mean() * 100)
    return df


# ── 3. Profile ─────────────────────────────────────────────────────────────────

def profile_data(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  DATASET PROFILE — KKBox Churn Prediction (v2)")
    print("=" * 60)
    print(f"  Rows       : {len(df):,}")
    print(f"  Columns    : {df.shape[1]}")
    print(f"  Churn rate : {df[TARGET].mean():.3%}")
    print(f"  Memory     : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print()

    profiler = DataProfiler(df, target_col=TARGET)
    summary  = profiler.summary()
    profiler.print_report()

    summary.to_csv(REPORT_DIR / "train_column_summary.csv", index=False)
    log.info("Column summary → train_column_summary.csv")

    missing = summary[summary["n_missing"] > 0][["column", "n_missing", "pct_missing"]]
    if len(missing):
        missing.to_csv(REPORT_DIR / "train_missing_values.csv", index=False)
        log.info("Missing values → train_missing_values.csv")


# ── 4. Visualisations ──────────────────────────────────────────────────────────

def plot_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("KKBox Churn Prediction v2 — Overview", fontsize=14, fontweight="bold")

    # Target distribution
    ax = axes[0, 0]
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar(["Retained (0)", "Churned (1)"],
                  counts.values, color=["#43A047", "#E53935"], alpha=0.85)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200, f"{count:,}\n({count/len(df):.1%})",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Target Distribution")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Age by churn
    ax = axes[0, 1]
    for label, clr, lbl in [(0, "#43A047", "Retained"), (1, "#E53935", "Churned")]:
        vals = df[df[TARGET] == label]["bd"].dropna().clip(7, 80)
        ax.hist(vals, bins=40, alpha=0.6, color=clr, label=lbl)
    ax.set_title("Age Distribution by Churn")
    ax.set_xlabel("Age (years)")
    ax.legend()

    # Listening time by churn
    ax = axes[1, 0]
    for label, clr, lbl in [(0, "#43A047", "Retained"), (1, "#E53935", "Churned")]:
        vals = df[df[TARGET] == label]["total_secs_mean"].dropna().clip(upper=10_000)
        ax.hist(vals, bins=50, alpha=0.6, color=clr, label=lbl)
    ax.set_title("Mean Daily Listening (secs) by Churn")
    ax.set_xlabel("Mean total_secs per day")
    ax.legend()

    # Auto-renew rate by churn
    ax = axes[1, 1]
    for label, clr, lbl in [(0, "#43A047", "Retained"), (1, "#E53935", "Churned")]:
        vals = df[df[TARGET] == label]["auto_renew_rate"].dropna()
        ax.hist(vals, bins=20, alpha=0.6, color=clr, label=lbl)
    ax.set_title("Auto-Renew Rate by Churn")
    ax.set_xlabel("Auto-Renew Rate (0–1)")
    ax.legend()

    plt.tight_layout()
    path = REPORT_DIR / "overview.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar(["Retained (0)", "Churned (1)"],
                  counts.values, color=["#43A047", "#E53935"], alpha=0.88, width=0.5)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 100, f"{count:,}\n({count/len(df):.1%})",
                ha="center", va="bottom", fontsize=11)
    ax.set_title("Target Distribution — KKBox Churn v2", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Subscribers")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    path = REPORT_DIR / "target_distribution.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


def plot_subscription_analysis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Subscription Behaviour vs Churn (v2)", fontsize=13, fontweight="bold")

    # Plan days bucket vs churn rate
    ax = axes[0]
    bins   = [0, 7, 30, 90, 180, 365, 9999]
    labels = ["<7d", "7–30d", "30–90d", "90–180d", "180–365d", ">365d"]
    df2 = df.copy()
    df2["plan_bkt"] = pd.cut(df2["plan_days_mean"].fillna(0), bins=bins, labels=labels)
    rates = df2.groupby("plan_bkt", observed=True)[TARGET].mean() * 100
    ax.bar(rates.index.astype(str), rates.values, color="#1E88E5", alpha=0.85)
    ax.set_title("Churn Rate by Plan Duration")
    ax.set_xlabel("Mean Plan Days")
    ax.set_ylabel("Churn Rate (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.tick_params(axis="x", rotation=30)

    # Cancel rate bucket vs churn rate
    ax = axes[1]
    df2["cancel_bkt"] = pd.cut(df2["cancel_rate"].fillna(0),
                                bins=[-0.01, 0.01, 0.1, 0.3, 0.5, 1.01],
                                labels=["Never", "Rare", "Low", "Med", "High"])
    rates2 = df2.groupby("cancel_bkt", observed=True)[TARGET].mean() * 100
    ax.bar(rates2.index.astype(str), rates2.values, color="#FB8C00", alpha=0.85)
    ax.set_title("Churn Rate by Historical Cancel Rate")
    ax.set_xlabel("Cancel Rate Bucket")
    ax.set_ylabel("Churn Rate (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.tight_layout()
    path = REPORT_DIR / "subscription_analysis.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


# ── 5. Split ───────────────────────────────────────────────────────────────────

def split_and_save(df: pd.DataFrame) -> None:
    df_train, df_val = smart_split(
        df, target_col=TARGET, task_type="binary_classification",
        val_size=0.20, random_state=RANDOM_STATE,
    )
    log.info(
        "Split → train: %s (%.3f%% churn) | val: %s (%.3f%% churn)",
        f"{len(df_train):,}", df_train[TARGET].mean() * 100,
        f"{len(df_val):,}",   df_val[TARGET].mean() * 100,
    )
    df_train.to_parquet(DATA_SUBDIR / "train_raw.parquet", index=False)
    df_val.to_parquet(DATA_SUBDIR   / "val_raw.parquet",   index=False)
    log.info("Raw splits saved to %s", DATA_SUBDIR)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case D: KKBox Churn (v2) — Data Loading")
    print("=" * 65 + "\n")

    # Use cached splits if they exist
    if (DATA_SUBDIR / "train_raw.parquet").exists() and (DATA_SUBDIR / "val_raw.parquet").exists():
        log.info("Cached splits found — loading for profiling…")
        df_tr = pd.read_parquet(DATA_SUBDIR / "train_raw.parquet")
        df_va = pd.read_parquet(DATA_SUBDIR / "val_raw.parquet")
        df = pd.concat([df_tr, df_va], ignore_index=True)
    else:
        print("[1] Loading source files…")
        train   = load_train_labels()
        members = load_members()

        print("[2] Aggregating transactions_v2 (chunked)…")
        txn_agg = aggregate_transactions()

        print("[3] Aggregating user_logs_v2 (chunked — may take a few minutes)…")
        log_agg = aggregate_user_logs()

        print("[4] Merging all tables…")
        df = build_master(train, members, txn_agg, log_agg)
        df = reduce_mem_usage(df, verbose=True)

        print("[5] Splitting train / validation…")
        split_and_save(df)

    print("[6] Profiling master dataset…")
    profile_data(df)

    print("[7] Generating plots…")
    plot_overview(df)
    plot_target_distribution(df)
    plot_subscription_analysis(df)

    print("\n" + "=" * 65)
    print("  Step 1 complete. Ready for EDA (02_eda_analysis.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
