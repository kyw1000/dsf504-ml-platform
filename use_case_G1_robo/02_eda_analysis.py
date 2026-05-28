"""
use_case_G1_robo/02_eda_analysis.py
=====================================
Use Case G1 — Robo-Advisory Portfolio Recommendation
Phase 1, Step 2: EDA & Data Understanding

Analyses:
  1. Customer segment profiling (risk × capacity × type)
  2. Asset popularity (long-tail distribution)
  3. Temporal activity patterns (seasonality, COVID impact)
  4. Customer–asset affinity by risk level × asset category
  5. ROI distribution analysis and risk-adjusted returns
  6. Co-purchase network (which assets are bought together)
  7. Cold-start exposure (new customers / new assets)

Run
---
    cd C:\\DSF504
    python use_case_G1_robo/02_eda_analysis.py
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
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "far_trans"
REPORT_DIR  = REPORTS_DIR / "use_case_G1"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers     = pd.read_parquet(DATA_SUBDIR / "customers.parquet")
    assets        = pd.read_parquet(DATA_SUBDIR / "assets.parquet")
    train_tx      = pd.read_parquet(DATA_SUBDIR / "train_transactions.parquet")
    profitability = pd.read_parquet(DATA_SUBDIR / "profitability.parquet")
    train_tx["transaction_date"] = pd.to_datetime(train_tx["transaction_date"])
    return customers, assets, train_tx, profitability


# ─────────────────────────────────────────────────────────────────────────────

def plot_customer_segments(customers: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Customer Segmentation Analysis", fontsize=13, fontweight="bold")

    for ax, col, title in zip(
        axes,
        ["risk_level", "customer_type", "investment_capacity"],
        ["Risk Level", "Customer Type", "Investment Capacity"],
    ):
        vc = customers[col].value_counts()
        bars = ax.barh(vc.index, vc.values, color=plt.cm.Paired.colors[:len(vc)])
        ax.set_title(title)
        ax.set_xlabel("Count")
        for bar in bars:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width()):,}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "customer_segments.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved customer_segments.png")


def plot_asset_popularity(transactions: pd.DataFrame, assets: pd.DataFrame) -> None:
    buy_tx = transactions[transactions["transaction_type"] == "Buy"]
    popularity = buy_tx.groupby("isin")["customer_id"].nunique().sort_values(ascending=False)
    merged = popularity.reset_index().merge(assets[["isin", "category"]], on="isin")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Asset Popularity — Long-Tail Distribution", fontsize=13, fontweight="bold")

    # Long-tail
    axes[0].plot(range(len(popularity)), popularity.values, color="#1565C0", linewidth=1.5)
    axes[0].fill_between(range(len(popularity)), popularity.values, alpha=0.15, color="#1565C0")
    axes[0].set_title("Rank–Popularity Curve (all assets)")
    axes[0].set_xlabel("Asset Rank (by popularity)")
    axes[0].set_ylabel("Unique Buyers")
    axes[0].set_yscale("log")
    top20_pct = popularity.iloc[:max(1, int(len(popularity)*0.2))].sum() / popularity.sum()
    axes[0].text(0.6, 0.85, f"Top 20% assets\naccounting for\n{top20_pct:.1%} of buys",
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # By category
    cat_pop = merged.groupby("category")["customer_id"].mean().sort_values()
    colors  = plt.cm.Set2.colors[:len(cat_pop)]
    axes[1].barh(cat_pop.index, cat_pop.values, color=colors)
    axes[1].set_title("Avg. Unique Buyers per Asset by Category")
    axes[1].set_xlabel("Avg. Unique Buyers")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "asset_popularity.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved asset_popularity.png")


def plot_temporal_patterns(transactions: pd.DataFrame) -> None:
    buy_tx = transactions[transactions["transaction_type"] == "Buy"].copy()
    buy_tx["month"]     = buy_tx["transaction_date"].dt.to_period("M")
    buy_tx["dayofweek"] = buy_tx["transaction_date"].dt.day_name()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Temporal Activity Patterns", fontsize=13, fontweight="bold")

    # Monthly volume
    monthly = buy_tx.groupby("month")["value"].sum() / 1e6
    monthly.index = monthly.index.to_timestamp()
    axes[0].plot(monthly.index, monthly.values, color="#1B5E20", linewidth=2)
    axes[0].fill_between(monthly.index, monthly.values, alpha=0.2, color="#1B5E20")
    axes[0].set_title("Monthly Buy Volume (€M)")
    axes[0].set_ylabel("Volume (€M)")
    # Mark COVID period
    axes[0].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"),
                    alpha=0.15, color="red", label="COVID shock (Mar–Jun 2020)")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)

    # Day of week pattern
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_counts = buy_tx["dayofweek"].value_counts().reindex(dow_order, fill_value=0)
    axes[1].bar(dow_counts.index, dow_counts.values, color="#4A148C")
    axes[1].set_title("Transaction Count by Day of Week")
    axes[1].set_ylabel("Number of Transactions")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "temporal_patterns.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved temporal_patterns.png")


def plot_risk_affinity(transactions: pd.DataFrame, customers: pd.DataFrame,
                       assets: pd.DataFrame) -> None:
    """Heatmap: risk level × asset category (buy affinity)."""
    buy_tx = transactions[transactions["transaction_type"] == "Buy"]
    merged = (buy_tx
              .merge(customers[["customer_id", "risk_level"]], on="customer_id")
              .merge(assets[["isin", "category"]], on="isin"))
    pivot  = merged.pivot_table(index="risk_level", columns="category",
                                values="value", aggfunc="count", fill_value=0)
    pivot  = pivot.div(pivot.sum(axis=1), axis=0)  # row-normalise

    risk_order = ["Conservative", "Moderate", "Balanced", "Growth", "Aggressive"]
    pivot = pivot.reindex([r for r in risk_order if r in pivot.index])

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if pivot.values[i, j] < 0.4 else "white")
    plt.colorbar(im, ax=ax, label="Share of Buys (row-normalised)")
    ax.set_title("Risk Level × Asset Category Buy Affinity", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "risk_category_affinity.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved risk_category_affinity.png")


def plot_roi_analysis(profitability: pd.DataFrame, assets: pd.DataFrame) -> None:
    merged = profitability.merge(assets[["isin", "category"]], on="isin")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Asset ROI Analysis", fontsize=13, fontweight="bold")

    # ROI distribution
    axes[0].hist(profitability["roi"], bins=50, color="#0D47A1", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Break-even")
    axes[0].axvline(profitability["roi"].median(), color="orange", linestyle="--",
                    linewidth=1.5, label=f"Median={profitability['roi'].median():.2%}")
    axes[0].set_title("Overall ROI Distribution")
    axes[0].set_xlabel("ROI")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # ROI by category — violin
    categories = merged["category"].unique()
    roi_by_cat = [merged[merged["category"] == c]["roi"].values for c in categories]
    parts = axes[1].violinplot(roi_by_cat, showmedians=True)
    for pc, color in zip(parts["bodies"], plt.cm.Set2.colors[:len(categories)]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    axes[1].set_xticks(range(1, len(categories) + 1))
    axes[1].set_xticklabels(categories, rotation=30, ha="right")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title("ROI Distribution by Asset Category")
    axes[1].set_ylabel("ROI")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roi_analysis.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved roi_analysis.png")


def compute_eda_summary(transactions: pd.DataFrame, customers: pd.DataFrame,
                        assets: pd.DataFrame, profitability: pd.DataFrame) -> None:
    buy_tx = transactions[transactions["transaction_type"] == "Buy"]
    summary = {
        "n_customers":        customers["customer_id"].nunique(),
        "n_assets":           assets["isin"].nunique(),
        "n_buy_transactions": len(buy_tx),
        "n_sell_transactions": len(transactions) - len(buy_tx),
        "sparsity":           1 - buy_tx.groupby(["customer_id", "isin"]).ngroups
                              / (customers["customer_id"].nunique() * assets["isin"].nunique()),
        "avg_assets_per_customer": buy_tx.groupby("customer_id")["isin"].nunique().mean(),
        "median_roi":          profitability["roi"].median(),
        "pct_positive_roi":    (profitability["roi"] > 0).mean(),
        "top_category":        assets["category"].value_counts().index[0],
        "dominant_channel":    transactions["channel"].value_counts().index[0],
    }
    pd.DataFrame([summary]).to_csv(REPORT_DIR / "eda_summary.csv", index=False)
    log.info("Saved eda_summary.csv")

    print("\n  EDA Summary:")
    for k, v in summary.items():
        print(f"    {k:<35}: {v}")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G1: Robo-Advisory Portfolio Recommendation")
    print("  Step 2: EDA & Data Understanding")
    print("=" * 65 + "\n")

    customers, assets, train_tx, profitability = _load()
    print(f"[1] Loaded: {len(customers):,} customers | {len(assets):,} assets | "
          f"{len(train_tx):,} train transactions")

    print("\n[2] Customer segment analysis…")
    plot_customer_segments(customers)

    print("[3] Asset popularity (long-tail)…")
    plot_asset_popularity(train_tx, assets)

    print("[4] Temporal activity patterns…")
    plot_temporal_patterns(train_tx)

    print("[5] Risk level × category affinity heatmap…")
    plot_risk_affinity(train_tx, customers, assets)

    print("[6] ROI distribution analysis…")
    plot_roi_analysis(profitability, assets)

    print("[7] EDA summary statistics…")
    compute_eda_summary(train_tx, customers, assets, profitability)

    print(f"\n  All EDA outputs → {REPORT_DIR}")
    print("=" * 65)
    print("  Step 2 complete. Ready for Feature Engineering (03_feature_engineering.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
