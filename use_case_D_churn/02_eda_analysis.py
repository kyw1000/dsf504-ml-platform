"""
use_case_D_churn/02_eda_analysis.py
=====================================
DSF504 — Use Case D: Customer Churn Prediction (KKBox)
Step 2: Exploratory Data Analysis

Key findings targeted
---------------------
1. Class imbalance (~8.4% churn) and SMOTE rationale
2. Listening engagement patterns: completions (num_100 / total songs)
3. Subscription plan behaviour: plan_days, auto-renew, cancellations
4. Registration channel effects (registered_via)
5. Geographic/demographic patterns (city, age, gender)
6. Feature-target correlations

Academic references
-------------------
- Verbeke et al. (2012). New insights into churn prediction
  in the telecommunication sector. EJOR.
- Hadden et al. (2007). Computer assisted customer churn
  management: State of the art and future trends.
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
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR

from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR    / "kkbox_churn"
REPORT_DIR  = REPORTS_DIR / "use_case_D"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_churn"

LISTEN_COLS = ["num_25_mean", "num_50_mean", "num_75_mean",
               "num_985_mean", "num_100_mean", "num_unq_mean", "total_secs_mean"]
PLAN_COLS   = ["plan_days_mean", "plan_price_mean", "actual_paid_mean",
               "auto_renew_rate", "cancel_rate", "txn_count"]
DEMO_COLS   = ["bd", "gender", "city", "registered_via"]


def load() -> pd.DataFrame:
    for fname in ["train_raw.parquet", "cs-training.parquet"]:
        p = DATA_SUBDIR / fname
        if p.exists():
            log.info("Loading %s…", fname)
            return pd.read_parquet(p)
    raise FileNotFoundError("Run 01_data_loading.py first.")


# ── 1. Class imbalance ─────────────────────────────────────────────────────────

def analyse_imbalance(df: pd.DataFrame) -> None:
    n       = len(df)
    n_churn = df[TARGET].sum()
    rate    = n_churn / n
    print("\n--- Class Imbalance ---")
    print(f"  Total subscribers : {n:,}")
    print(f"  Retained    (0)   : {n - n_churn:,}  ({1-rate:.2%})")
    print(f"  Churned     (1)   : {n_churn:,}  ({rate:.2%})")
    print(f"  Imbalance ratio   : {(n - n_churn)/n_churn:.1f}:1")
    print("  Recommended       : SMOTE on train fold only · primary metric = ROC-AUC")


# ── 2. Engagement analysis ─────────────────────────────────────────────────────

def analyse_engagement(df: pd.DataFrame) -> None:
    print("\n--- Engagement Patterns ---")
    for col in LISTEN_COLS:
        if col not in df.columns:
            continue
        churned   = df[df[TARGET] == 1][col].dropna().mean()
        retained  = df[df[TARGET] == 0][col].dropna().mean()
        diff_pct  = (churned - retained) / (retained + 1e-9) * 100
        print(f"  {col:20s}: churned={churned:.1f}  retained={retained:.1f}  diff={diff_pct:+.1f}%")


# ── 3. Subscription behaviour ─────────────────────────────────────────────────

def analyse_subscription(df: pd.DataFrame) -> None:
    print("\n--- Subscription Behaviour ---")
    if "auto_renew_rate" in df.columns:
        ar_churn = df[df[TARGET] == 1]["auto_renew_rate"].dropna().mean()
        ar_ret   = df[df[TARGET] == 0]["auto_renew_rate"].dropna().mean()
        print(f"  Auto-renew rate: churned={ar_churn:.3f}  retained={ar_ret:.3f}")
    if "cancel_rate" in df.columns:
        cr_churn = df[df[TARGET] == 1]["cancel_rate"].dropna().mean()
        cr_ret   = df[df[TARGET] == 0]["cancel_rate"].dropna().mean()
        print(f"  Cancel rate    : churned={cr_churn:.3f}  retained={cr_ret:.3f}")


# ── 4. Correlation analysis ────────────────────────────────────────────────────

def analyse_correlation(df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET]
    corr = df[num_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\n--- Top 15 Feature-Target Correlations ---")
    for feat, val in corr.head(15).items():
        print(f"  {feat:30s}: {val:+.4f}")
    corr_df = corr.reset_index()
    corr_df.columns = ["feature", "correlation"]
    corr_df.to_csv(REPORT_DIR / "feature_target_correlation.csv", index=False)
    log.info("Correlation CSV saved")


# ── 5. Missing values ─────────────────────────────────────────────────────────

def analyse_missing(df: pd.DataFrame) -> None:
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    print(f"\n--- Missing Values ({len(missing)} columns with missing) ---")
    for col, pct in missing.head(15).items():
        print(f"  {col:30s}: {pct:.2f}%")


# ── 6. Plots ───────────────────────────────────────────────────────────────────

def plot_engagement(df: pd.DataFrame) -> None:
    """Listening engagement by churn class."""
    cols = [c for c in LISTEN_COLS if c in df.columns]
    if not cols:
        return
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    fig.suptitle("Listening Engagement by Churn Status", fontsize=13, fontweight="bold")

    for i, col in enumerate(cols[:7]):
        ax = axes[i]
        for label, clr, lbl in [(0, "#43A047", "Retained"), (1, "#E53935", "Churned")]:
            vals = df[df[TARGET] == label][col].dropna()
            p95  = vals.quantile(0.95)
            ax.hist(vals.clip(upper=p95), bins=40, alpha=0.6, color=clr, label=lbl, density=True)
        ax.set_title(col.replace("_mean", "").replace("_", " ").title())
        ax.legend(fontsize=8)
    axes[-1].set_visible(False)

    plt.tight_layout()
    path = REPORT_DIR / "engagement_by_churn.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


def plot_subscription(df: pd.DataFrame) -> None:
    """Subscription plan behaviour vs churn."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Subscription Behaviour vs Churn", fontsize=13, fontweight="bold")

    # Plan days
    ax = axes[0]
    bins   = [0, 7, 30, 90, 180, 365, 9999]
    labels = ["<7d", "7–30d", "30–90d", "90–180d", "180–365d", ">365d"]
    df2 = df.copy()
    df2["plan_bkt"] = pd.cut(df2["plan_days_mean"].fillna(0), bins=bins, labels=labels)
    rates = df2.groupby("plan_bkt", observed=True)[TARGET].mean() * 100
    ax.bar(rates.index.astype(str), rates.values, color="#1E88E5", alpha=0.85)
    ax.set_title("Churn Rate by Plan Duration")
    ax.set_xlabel("Avg Plan Days")
    ax.set_ylabel("Churn Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(axis="x", rotation=30)

    # Auto-renew rate
    ax = axes[1]
    df2["ar_bkt"] = pd.cut(df2["auto_renew_rate"].fillna(0),
                            bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
                            labels=["Low", "Med-Low", "Med-High", "High"])
    rates2 = df2.groupby("ar_bkt", observed=True)[TARGET].mean() * 100
    ax.bar(rates2.index.astype(str), rates2.values, color="#43A047", alpha=0.85)
    ax.set_title("Churn Rate by Auto-Renew Rate")
    ax.set_xlabel("Auto-Renew Rate Bucket")
    ax.set_ylabel("Churn Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Registration channel
    ax = axes[2]
    if "registered_via" in df.columns:
        rv = df.groupby("registered_via")[TARGET].mean().sort_values() * 100
        ax.barh(rv.index.astype(str), rv.values, color="#AB47BC", alpha=0.85)
        ax.set_title("Churn Rate by Registration Channel")
        ax.set_xlabel("Churn Rate (%)")
        ax.set_ylabel("registered_via")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.tight_layout()
    path = REPORT_DIR / "subscription_eda.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET][:20]
    corr_m   = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_m, dtype=bool))
    sns.heatmap(corr_m, mask=mask, annot=False, cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.3,
                cbar_kws={"shrink": 0.75})
    ax.set_title("Feature Correlation Heatmap (top 20 numeric)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = REPORT_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


def plot_missing_heatmap(df: pd.DataFrame) -> None:
    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    if not missing_cols:
        log.info("No missing values — skipping missing heatmap")
        return
    sample = df[missing_cols].isnull().sample(min(500, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(max(8, len(missing_cols) * 0.7), 6))
    sns.heatmap(sample.T, cbar=False, yticklabels=True, ax=ax,
                cmap=["#1E88E5", "#E53935"])
    ax.set_title("Missing Value Pattern (500-row sample)", fontsize=12)
    ax.set_xlabel("Sample rows")
    plt.tight_layout()
    path = REPORT_DIR / "missing_heatmap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


def plot_outlier_report(df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET]
    rows = []
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        rows.append({"column": col, "n_outliers": n_out, "pct_outliers": n_out / len(df) * 100})
    out_df = pd.DataFrame(rows).sort_values("n_outliers", ascending=False)
    out_df.to_csv(REPORT_DIR / "outlier_report.csv", index=False)
    log.info("Outlier report saved")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case D: KKBox Churn — EDA")
    print("=" * 65 + "\n")

    df = load()

    print("[1] Imbalance…")
    analyse_imbalance(df)
    print("[2] Engagement…")
    analyse_engagement(df)
    print("[3] Subscription behaviour…")
    analyse_subscription(df)
    print("[4] Correlation…")
    analyse_correlation(df)
    print("[5] Missing values…")
    analyse_missing(df)
    print("[6] Plots…")
    plot_engagement(df)
    plot_subscription(df)
    plot_correlation_heatmap(df)
    plot_missing_heatmap(df)
    plot_outlier_report(df)

    # ── Supplemental standardised EDA plots ──────────────────────────────
    print("[7] Supplemental standardised visualizations…")
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.eda_viz import (
            plot_overview_panel, plot_target_distribution,
            plot_numeric_distributions,
        )
        plot_overview_panel(df, TARGET, REPORT_DIR, " — UC D Churn")
        plot_target_distribution(df, TARGET, REPORT_DIR, " — UC D Churn",
                                 label_map={0: "Retained", 1: "Churned"})
        plot_numeric_distributions(df, REPORT_DIR, " — UC D Churn", target_col=TARGET)
        print("    Saved: overview.png, target_distribution.png, numeric_distributions.png")
    except Exception as _e:
        print(f"    [warn] Supplemental plots skipped: {_e}")

    print("\n" + "=" * 65)
    print("  Step 2 complete. Ready for feature engineering (03_feature_engineering.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
