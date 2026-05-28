"""
use_case_G2_xai/02_eda_analysis.py
=====================================
Use Case G2 — Explainable AI for Analysts & Managers
Phase 1, Step 2: EDA & Data Understanding

Analyses
--------
  1. Financial ratio correlation with forward return (point-biserial)
  2. Sector-level outperformance rates
  3. Missing value analysis by ratio category
  4. Outlier detection (IQR method) across financial ratios
  5. Ratio decile analysis (value vs growth premium)
  6. Year-over-year target drift (temporal stability check)
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
from scipy.stats import pointbiserialr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "sec_edgar"
REPORT_DIR  = REPORTS_DIR / "use_case_G2"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RATIO_COLS = [
    "pe_ratio", "pb_ratio", "ps_ratio",
    "roe", "roa", "net_margin", "gross_margin", "ebitda_margin",
    "debt_equity", "interest_coverage", "debt_assets",
    "current_ratio", "quick_ratio",
    "asset_turnover", "revenue_growth", "eps_growth", "fcf_yield",
    "market_cap_log",
]


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_SUBDIR / "train_ratios.parquet")
    val   = pd.read_parquet(DATA_SUBDIR / "val_ratios.parquet")
    return train, val


# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_with_target(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for col in RATIO_COLS:
        if col not in df.columns:
            continue
        vals = df[[col, "outperform"]].dropna()
        r, p = pointbiserialr(vals["outperform"], vals[col])
        results.append({"feature": col, "correlation": r, "p_value": p,
                        "significant": p < 0.05})
    corr_df = pd.DataFrame(results).sort_values("correlation", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in corr_df["correlation"]]
    bars = ax.barh(corr_df["feature"], corr_df["correlation"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    # Mark significance
    for i, (_, row) in enumerate(corr_df.iterrows()):
        if row["significant"]:
            ax.text(row["correlation"] + 0.002 * np.sign(row["correlation"]),
                    i, "✓", va="center", fontsize=8, color="darkblue")
    ax.set_title("Point-Biserial Correlation: Financial Ratio vs. Outperformance",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Correlation (✓ = p < 0.05)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_target_correlation.png", dpi=120, bbox_inches="tight")
    plt.close()
    corr_df.to_csv(REPORT_DIR / "feature_target_correlation.csv", index=False)
    log.info("Saved feature_target_correlation.png/csv")
    return corr_df


def plot_sector_performance(df: pd.DataFrame) -> None:
    sec_stats = (df.groupby("sector")["outperform"]
                 .agg(["mean", "count", "std"])
                 .rename(columns={"mean": "outperform_rate", "count": "n", "std": "std"})
                 .sort_values("outperform_rate", ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Sector-Level Analysis", fontsize=13, fontweight="bold")

    bars = axes[0].barh(sec_stats.index, sec_stats["outperform_rate"],
                        color=plt.cm.RdYlGn(sec_stats["outperform_rate"].values))
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="50% base rate")
    axes[0].set_title("Outperformance Rate by Sector")
    axes[0].set_xlabel("P(Outperform)")
    axes[0].legend()
    for bar, v in zip(bars, sec_stats["outperform_rate"]):
        axes[0].text(v + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{v:.2f}", va="center", fontsize=8)

    axes[1].barh(sec_stats.index, sec_stats["n"], color="#1976D2")
    axes[1].set_title("Observations per Sector")
    axes[1].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "sector_performance.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved sector_performance.png")


def plot_ratio_deciles(df: pd.DataFrame) -> None:
    """Show outperformance rate across deciles of key financial ratios."""
    key_ratios = ["pe_ratio", "pb_ratio", "roe", "revenue_growth", "debt_equity", "net_margin"]
    present = [c for c in key_ratios if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Outperformance Rate by Financial Ratio Decile",
                 fontsize=13, fontweight="bold")

    for ax, col in zip(axes.flat, present):
        tmp = df[[col, "outperform"]].dropna().copy()
        try:
            tmp["decile"] = pd.qcut(tmp[col], q=10, labels=False, duplicates="drop") + 1
        except Exception:
            tmp["decile"] = pd.cut(tmp[col], bins=10, labels=False) + 1
        decile_rate = tmp.groupby("decile")["outperform"].mean()
        ax.bar(decile_rate.index, decile_rate.values,
               color=plt.cm.coolwarm(decile_rate.values))
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Decile (1=lowest)")
        ax.set_ylabel("P(Outperform)")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "ratio_deciles.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved ratio_deciles.png")


def plot_missing_analysis(df: pd.DataFrame) -> None:
    null_pct = df[RATIO_COLS].isna().mean().sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]

    if len(null_pct) == 0:
        log.info("No missing values in ratio columns.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(null_pct.index, null_pct.values * 100, color="#FF7043")
    ax.set_title("Missing Value Rate by Financial Ratio", fontsize=12, fontweight="bold")
    ax.set_xlabel("% Missing")
    ax.axvline(5, color="red", linestyle="--", linewidth=0.8, label="5% threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "missing_analysis.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved missing_analysis.png")


def plot_outlier_analysis(df: pd.DataFrame) -> None:
    outlier_rows = []
    for col in RATIO_COLS:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        pct = ((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).mean()
        outlier_rows.append({"feature": col, "outlier_pct": round(pct * 100, 2)})
    outlier_df = pd.DataFrame(outlier_rows).sort_values("outlier_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(outlier_df["feature"], outlier_df["outlier_pct"], color="#7B1FA2", alpha=0.8)
    ax.axvline(5, color="red", linestyle="--", label="5% threshold")
    ax.set_title("Outlier Rate by Feature (3×IQR rule)", fontsize=12, fontweight="bold")
    ax.set_xlabel("% Outliers")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "outlier_report.png", dpi=120, bbox_inches="tight")
    plt.close()
    outlier_df.to_csv(REPORT_DIR / "outlier_report.csv", index=False)
    log.info("Saved outlier_report.png/csv")


def plot_temporal_drift(df: pd.DataFrame) -> None:
    yr_stats = df.groupby("fiscal_year")["outperform"].agg(["mean", "std", "count"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yr_stats.index, yr_stats["mean"], "o-", color="#1976D2", linewidth=2)
    ax.fill_between(yr_stats.index,
                    yr_stats["mean"] - yr_stats["std"],
                    yr_stats["mean"] + yr_stats["std"],
                    alpha=0.2, color="#1976D2", label="±1 std")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, label="50% base rate")
    ax.set_title("Outperformance Rate by Year (Temporal Drift)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("P(Outperform)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "temporal_drift.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved temporal_drift.png")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G2: Explainable AI for Analysts & Managers")
    print("  Step 2: EDA & Data Understanding")
    print("=" * 65 + "\n")

    train, val = _load()
    print(f"[1] Train: {len(train):,} obs  |  Val: {len(val):,} obs")

    print("[2] Correlation analysis with outperformance target…")
    corr_df = plot_correlation_with_target(train)
    print(f"    Top-5 correlated ratios:")
    print(corr_df.head(5)[["feature", "correlation", "significant"]].to_string(index=False))

    print("[3] Sector performance analysis…")
    plot_sector_performance(train)

    print("[4] Ratio decile analysis (value vs growth patterns)…")
    plot_ratio_deciles(train)

    print("[5] Missing value analysis…")
    plot_missing_analysis(train)

    print("[6] Outlier detection…")
    plot_outlier_analysis(train)

    print("[7] Temporal drift check…")
    plot_temporal_drift(train)

    print(f"\n  All EDA outputs → {REPORT_DIR}")
    print("=" * 65)
    print("  Step 2 complete. Ready for Feature Engineering (03_)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
