"""
use_case_G1_robo/01_data_loading.py
=====================================
Use Case G1 — Robo-Advisory Portfolio Recommendation
Phase 1, Step 1: Data Loading & Initial Inspection

Dataset: FAR-Trans (Financial Asset Recommendation)
Source : https://doi.org/10.5525/gla.researchdata.1658  (CC-BY 4.0)
         University of Glasgow — Jan 2018 to Nov 2022

Four tables
-----------
  customers.csv   : customer_id, customer_type, risk_level, investment_capacity
  assets.csv      : isin, asset_name, category, subcategory, market, sector, industry
  transactions.csv: customer_id, isin, transaction_type (Buy/Sell), value, units,
                    channel, market, transaction_date
  profitability.csv: isin, roi, first_date, last_date, roi_min, roi_max

ML Task
-------
  Personalized financial asset ranking:
  Given a customer's profile and history, rank candidate assets by likelihood of purchase.
  Formulated as a Learning-to-Rank (LambdaRank) problem for LightGBM.

Run
---
    cd C:\\DSF504
    python use_case_G1_robo/01_data_loading.py
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "far_trans"
REPORT_DIR  = REPORTS_DIR / "use_case_G1"
DATA_SUBDIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic FAR-Trans generator  (matches real schema)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic(
    n_customers: int = 2_000,
    n_assets:    int = 500,
    n_tx:        int = 80_000,
    seed:        int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    risk_levels  = ["Conservative", "Moderate", "Balanced", "Growth", "Aggressive"]
    cust_types   = ["Retail", "Premium", "Private"]
    categories   = ["Equity", "Bond", "Mutual Fund", "ETF", "Commodity", "Crypto"]
    subcategories = {
        "Equity":      ["Large Cap", "Mid Cap", "Small Cap", "Growth", "Value"],
        "Bond":        ["Government", "Corporate", "High Yield", "Municipal"],
        "Mutual Fund": ["Balanced", "Income", "Index", "Sector"],
        "ETF":         ["Equity ETF", "Bond ETF", "Commodity ETF", "Thematic"],
        "Commodity":   ["Precious Metal", "Energy", "Agriculture"],
        "Crypto":      ["Large Cap", "DeFi", "Layer 1"],
    }
    markets  = ["NYSE", "NASDAQ", "LSE", "EURONEXT", "XETRA", "TSX"]
    sectors  = ["Technology", "Healthcare", "Financials", "Energy", "Consumer",
                "Industrials", "Utilities", "Real Estate", "Materials", "Communication"]
    channels = ["Mobile App", "Web Platform", "Advisor", "Phone"]

    # ── Customers ────────────────────────────────────────────────────────────
    customers = pd.DataFrame({
        "customer_id":        [f"C{i:05d}" for i in range(n_customers)],
        "customer_type":      rng.choice(cust_types, n_customers, p=[0.60, 0.30, 0.10]),
        "risk_level":         rng.choice(risk_levels, n_customers, p=[0.20, 0.25, 0.25, 0.20, 0.10]),
        "investment_capacity": rng.choice(["<10K", "10K-50K", "50K-200K", ">200K"],
                                           n_customers, p=[0.30, 0.35, 0.25, 0.10]),
    })

    # ── Assets ───────────────────────────────────────────────────────────────
    asset_categories = rng.choice(categories, n_assets, p=[0.35, 0.25, 0.15, 0.12, 0.08, 0.05])
    assets = pd.DataFrame({
        "isin":        [f"ISIN{i:06d}" for i in range(n_assets)],
        "asset_name":  [f"Asset_{i:04d}" for i in range(n_assets)],
        "category":    asset_categories,
        "subcategory": [rng.choice(subcategories[c]) for c in asset_categories],
        "market":      rng.choice(markets, n_assets),
        "sector":      rng.choice(sectors, n_assets),
        "industry":    [f"Industry_{rng.integers(1, 30)}" for _ in range(n_assets)],
    })

    # ── Transactions ─────────────────────────────────────────────────────────
    # Skew: customers with higher capacity trade more; Growth/Aggressive buy Equity more
    tx_customers = rng.choice(n_customers, n_tx, replace=True)
    tx_assets    = rng.choice(n_assets,    n_tx, replace=True)
    dates = pd.date_range("2018-01-01", "2022-11-30", periods=n_tx)
    dates = dates[rng.integers(0, len(dates), n_tx)]

    transactions = pd.DataFrame({
        "customer_id":      [f"C{c:05d}" for c in tx_customers],
        "isin":             [f"ISIN{a:06d}" for a in tx_assets],
        "transaction_type": rng.choice(["Buy", "Sell"], n_tx, p=[0.65, 0.35]),
        "value":            rng.lognormal(mean=9.0, sigma=1.5, size=n_tx).round(2),
        "units":            rng.integers(1, 500, n_tx),
        "channel":          rng.choice(channels, n_tx, p=[0.40, 0.35, 0.15, 0.10]),
        "market":           rng.choice(markets, n_tx),
        "transaction_date": dates,
    })
    transactions = transactions.sort_values("transaction_date").reset_index(drop=True)

    # ── Asset Profitability ───────────────────────────────────────────────────
    # Risk-adjusted ROI: Equity/Crypto more volatile; Bond/MF more stable
    roi_mu = {"Equity": 0.08, "Bond": 0.03, "Mutual Fund": 0.05,
               "ETF": 0.07, "Commodity": 0.04, "Crypto": 0.15}
    roi_sigma = {"Equity": 0.20, "Bond": 0.05, "Mutual Fund": 0.10,
                 "ETF": 0.15, "Commodity": 0.18, "Crypto": 0.60}

    rois, roi_mins, roi_maxs = [], [], []
    for cat in assets["category"]:
        mu, sigma = roi_mu[cat], roi_sigma[cat]
        r = rng.normal(mu, sigma)
        rois.append(round(r, 4))
        roi_mins.append(round(r - abs(rng.normal(0, sigma)), 4))
        roi_maxs.append(round(r + abs(rng.normal(0, sigma)), 4))

    profitability = pd.DataFrame({
        "isin":       assets["isin"],
        "roi":        rois,
        "roi_min":    roi_mins,
        "roi_max":    roi_maxs,
        "first_date": pd.Timestamp("2018-01-01"),
        "last_date":  pd.Timestamp("2022-11-30"),
    })

    return customers, assets, transactions, profitability


# ─────────────────────────────────────────────────────────────────────────────
# Real data loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_real() -> tuple[pd.DataFrame, ...] | None:
    needed = ["customers.csv", "assets.csv", "transactions.csv", "profitability.csv"]
    if all((DATA_SUBDIR / f).exists() for f in needed):
        log.info("Loading real FAR-Trans CSV files…")
        customers     = pd.read_csv(DATA_SUBDIR / "customers.csv")
        assets        = pd.read_csv(DATA_SUBDIR / "assets.csv")
        transactions  = pd.read_csv(DATA_SUBDIR / "transactions.csv",
                                    parse_dates=["transaction_date"])
        profitability = pd.read_csv(DATA_SUBDIR / "profitability.csv")
        return customers, assets, transactions, profitability
    return None


def _print_download_instructions() -> None:
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  FAR-Trans Real Data — Download Instructions            │
  │                                                         │
  │  1. Visit: https://doi.org/10.5525/gla.researchdata.1658│
  │  2. Download the four CSV files:                        │
  │       customers.csv   assets.csv                        │
  │       transactions.csv  profitability.csv               │
  │  3. Place them in: data/far_trans/                      │
  │  4. Re-run this script.                                 │
  │                                                         │
  │  Licence: CC-BY 4.0                                     │
  └─────────────────────────────────────────────────────────┘
""")


# ─────────────────────────────────────────────────────────────────────────────
# Train / val split (customer-level)
# ─────────────────────────────────────────────────────────────────────────────

def _split_interactions(
    transactions: pd.DataFrame,
    val_months:   int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: last `val_months` of each customer's activity → val.
    This is the correct split for recommendation — prevents future leakage.
    """
    tx = transactions.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
    cutoff = tx["transaction_date"].max() - pd.DateOffset(months=val_months)
    train = tx[tx["transaction_date"] <= cutoff].reset_index(drop=True)
    val   = tx[tx["transaction_date"]  > cutoff].reset_index(drop=True)
    # Val: only customers seen in train (cold-start excluded for now)
    known = set(train["customer_id"].unique())
    val   = val[val["customer_id"].isin(known)].reset_index(drop=True)
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def _plot_overviews(
    customers:     pd.DataFrame,
    assets:        pd.DataFrame,
    transactions:  pd.DataFrame,
    profitability: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FAR-Trans Dataset — Overview", fontsize=14, fontweight="bold")

    # 1. Customer risk profile
    ax = axes[0, 0]
    order = ["Conservative", "Moderate", "Balanced", "Growth", "Aggressive"]
    counts = customers["risk_level"].value_counts().reindex(order, fill_value=0)
    bars = ax.bar(counts.index, counts.values,
                  color=["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"])
    ax.set_title("Customer Risk Profile Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Customers")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)

    # 2. Asset category distribution
    ax = axes[0, 1]
    cat_counts = assets["category"].value_counts()
    ax.pie(cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%",
           startangle=90, colors=plt.cm.Set3.colors[:len(cat_counts)])
    ax.set_title("Asset Category Distribution")

    # 3. Transaction volume over time
    ax = axes[1, 0]
    tx_monthly = (transactions.set_index("transaction_date")
                              .resample("ME")["value"].sum() / 1e6)
    ax.plot(tx_monthly.index, tx_monthly.values, color="#1565C0", linewidth=1.5)
    ax.fill_between(tx_monthly.index, tx_monthly.values, alpha=0.2, color="#1565C0")
    ax.set_title("Monthly Transaction Volume (€M)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume (€M)")
    ax.tick_params(axis="x", rotation=45)

    # 4. Asset ROI distribution by category
    ax = axes[1, 1]
    merged = profitability.merge(assets[["isin", "category"]], on="isin", how="left")
    categories_plot = merged["category"].unique()
    roi_data = [merged[merged["category"] == c]["roi"].values for c in categories_plot]
    bp = ax.boxplot(roi_data, tick_labels=categories_plot, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.Set2.colors[:len(categories_plot)]):
        patch.set_facecolor(color)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Asset ROI by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("ROI")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "dataset_overview.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved dataset_overview.png")


def _plot_interaction_sparsity(transactions: pd.DataFrame, customers: pd.DataFrame,
                                assets: pd.DataFrame) -> None:
    """Show the sparsity of the customer-asset interaction matrix."""
    n_cust  = customers["customer_id"].nunique()
    n_asset = assets["isin"].nunique()
    n_obs   = transactions[transactions["transaction_type"] == "Buy"]["customer_id"].nunique()
    buy_tx  = transactions[transactions["transaction_type"] == "Buy"]
    interactions_per_cust = buy_tx.groupby("customer_id")["isin"].nunique()
    interactions_per_asset = buy_tx.groupby("isin")["customer_id"].nunique()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Interaction Matrix Sparsity Analysis", fontsize=13, fontweight="bold")

    # Interactions per customer
    axes[0].hist(interactions_per_cust.values, bins=40, color="#1976D2", edgecolor="white")
    axes[0].set_title("Unique Assets per Customer (Buy)")
    axes[0].set_xlabel("Number of Unique Assets")
    axes[0].set_ylabel("Number of Customers")
    axes[0].axvline(interactions_per_cust.median(), color="red", linestyle="--",
                    label=f"Median={interactions_per_cust.median():.1f}")
    axes[0].legend()

    # Interactions per asset
    axes[1].hist(interactions_per_asset.values, bins=40, color="#388E3C", edgecolor="white")
    axes[1].set_title("Unique Customers per Asset (Buy)")
    axes[1].set_xlabel("Number of Unique Customers")
    axes[1].set_ylabel("Number of Assets")
    axes[1].axvline(interactions_per_asset.median(), color="red", linestyle="--",
                    label=f"Median={interactions_per_asset.median():.1f}")
    axes[1].legend()

    # Buy/Sell ratio by category
    merged = transactions.merge(assets[["isin", "category"]], on="isin", how="left")
    ratio  = (merged.groupby(["category", "transaction_type"])
              .size().unstack(fill_value=0))
    ratio["buy_ratio"] = ratio.get("Buy", 0) / (ratio.get("Buy", 0) + ratio.get("Sell", 0))
    ratio["buy_ratio"].sort_values().plot(kind="barh", ax=axes[2], color="#7B1FA2")
    axes[2].set_title("Buy/(Buy+Sell) Ratio by Category")
    axes[2].set_xlabel("Buy Ratio")
    axes[2].axvline(0.5, color="black", linestyle="--", linewidth=0.8)

    sparsity = 1 - (buy_tx.groupby(["customer_id", "isin"]).ngroups / (n_cust * n_asset))
    fig.text(0.5, -0.02,
             f"Matrix sparsity: {sparsity:.4%}  |  {n_cust:,} customers × {n_asset:,} assets",
             ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "interaction_sparsity.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved interaction_sparsity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G1: Robo-Advisory Portfolio Recommendation")
    print("  Step 1: Data Loading & Initial Inspection (FAR-Trans)")
    print("=" * 65 + "\n")

    # Load real or synthetic
    result = _load_real()
    if result is None:
        _print_download_instructions()
        log.info("Generating synthetic FAR-Trans data (2,000 customers, 500 assets)…")
        customers, assets, transactions, profitability = _generate_synthetic()
        log.info("Synthetic data generated.")
    else:
        customers, assets, transactions, profitability = result

    print(f"[1] Customers   : {len(customers):>8,}  |  cols: {list(customers.columns)}")
    print(f"    Assets      : {len(assets):>8,}  |  cols: {list(assets.columns)}")
    print(f"    Transactions: {len(transactions):>8,}  |  date range: "
          f"{transactions['transaction_date'].min().date()} → "
          f"{transactions['transaction_date'].max().date()}")
    print(f"    Profitability: {len(profitability):>7,}")

    print("\n[2] Basic statistics:")
    buy_tx = transactions[transactions["transaction_type"] == "Buy"]
    print(f"    Buy transactions  : {len(buy_tx):,} ({len(buy_tx)/len(transactions)*100:.1f}%)")
    print(f"    Unique customers  : {transactions['customer_id'].nunique():,}")
    print(f"    Unique assets     : {transactions['isin'].nunique():,}")
    print(f"    Avg tx per cust   : {len(transactions)/transactions['customer_id'].nunique():.1f}")
    print(f"    Avg tx per asset  : {len(transactions)/transactions['isin'].nunique():.1f}")
    print(f"    Total value (€M)  : {transactions['value'].sum()/1e6:.1f}")

    print("\n[3] Risk level distribution:")
    for rl, cnt in customers["risk_level"].value_counts().items():
        print(f"    {rl:<15}: {cnt:>5,}  ({cnt/len(customers)*100:.1f}%)")

    print("\n[4] Asset category breakdown:")
    for cat, cnt in assets["category"].value_counts().items():
        print(f"    {cat:<15}: {cnt:>5,}  ({cnt/len(assets)*100:.1f}%)")

    print("\n[5] Temporal train/val split (last 3 months → val)…")
    train_tx, val_tx = _split_interactions(transactions, val_months=3)
    print(f"    Train: {len(train_tx):,} transactions | "
          f"{train_tx['customer_id'].nunique():,} customers")
    print(f"    Val  : {len(val_tx):,} transactions | "
          f"{val_tx['customer_id'].nunique():,} customers")

    print("\n[6] Saving Parquet files…")
    customers.to_parquet(DATA_SUBDIR / "customers.parquet", index=False)
    assets.to_parquet(DATA_SUBDIR / "assets.parquet", index=False)
    train_tx.to_parquet(DATA_SUBDIR / "train_transactions.parquet", index=False)
    val_tx.to_parquet(DATA_SUBDIR / "val_transactions.parquet", index=False)
    profitability.to_parquet(DATA_SUBDIR / "profitability.parquet", index=False)
    log.info("Parquet files saved.")

    print("\n[7] Generating visualisations…")
    _plot_overviews(customers, assets, transactions, profitability)
    _plot_interaction_sparsity(transactions, customers, assets)

    # Column summary
    summary_rows = []
    for tbl_name, tbl in [("customers", customers), ("assets", assets),
                           ("transactions", transactions), ("profitability", profitability)]:
        for col in tbl.columns:
            summary_rows.append({
                "table": tbl_name, "column": col,
                "dtype": str(tbl[col].dtype),
                "nunique": tbl[col].nunique(),
                "null_pct": f"{tbl[col].isna().mean()*100:.2f}%",
            })
    pd.DataFrame(summary_rows).to_csv(REPORT_DIR / "data_dictionary.csv", index=False)
    log.info("Saved data_dictionary.csv")

    print(f"\n  All outputs → {DATA_SUBDIR}")
    print("=" * 65)
    print("  Step 1 complete. Ready for EDA (02_eda_analysis.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
