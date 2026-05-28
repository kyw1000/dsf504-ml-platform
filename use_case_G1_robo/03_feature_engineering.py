"""
use_case_G1_robo/03_feature_engineering.py
============================================
Use Case G1 — Robo-Advisory Portfolio Recommendation
Phase 2, Step 3: Feature Engineering

Transforms four raw FAR-Trans tables into a flat candidate-pair
(customer_id, isin) feature matrix for LightGBM LambdaRank.

Feature Groups
--------------
  USER  : customer profile features + behavioural aggregates
  ITEM  : asset metadata + profitability features
  INTER : user × item cross features (affinity signals)
  LABEL : implicit feedback — 1 if customer bought the asset in val period

Outputs
-------
  data/far_trans/train_pairs.parquet  — (user, item, features, query_id)
  data/far_trans/val_pairs.parquet
  data/far_trans/fe_stats.pkl

Run
---
    cd C:\\DSF504
    python use_case_G1_robo/03_feature_engineering.py
"""

from __future__ import annotations

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "far_trans"
REPORT_DIR  = REPORTS_DIR / "use_case_G1"
MODEL_DIR   = MODELS_DIR / "use_case_G1"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def _load():
    customers     = pd.read_parquet(DATA_SUBDIR / "customers.parquet")
    assets        = pd.read_parquet(DATA_SUBDIR / "assets.parquet")
    train_tx      = pd.read_parquet(DATA_SUBDIR / "train_transactions.parquet")
    val_tx        = pd.read_parquet(DATA_SUBDIR / "val_transactions.parquet")
    profitability = pd.read_parquet(DATA_SUBDIR / "profitability.parquet")
    train_tx["transaction_date"] = pd.to_datetime(train_tx["transaction_date"])
    val_tx["transaction_date"]   = pd.to_datetime(val_tx["transaction_date"])
    return customers, assets, train_tx, val_tx, profitability


# ─────────────────────────────────────────────────────────────────────────────
# USER features
# ─────────────────────────────────────────────────────────────────────────────

def build_user_features(customers: pd.DataFrame, train_tx: pd.DataFrame,
                        train_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    if train_stats is None:
        train_stats = {}
    is_train = "pref_cat_map" not in train_stats

    # Profile encoding
    risk_map = {"Conservative": 0, "Moderate": 1, "Balanced": 2, "Growth": 3, "Aggressive": 4}
    type_map = {"Retail": 0, "Premium": 1, "Private": 2}
    cap_map  = {"<10K": 0, "10K-50K": 1, "50K-200K": 2, ">200K": 3}

    uf = customers.copy()
    uf["user_risk_score"]   = uf["risk_level"].map(risk_map).fillna(2)
    uf["user_type_score"]   = uf["customer_type"].map(type_map).fillna(0)
    uf["user_cap_score"]    = uf["investment_capacity"].map(cap_map).fillna(1)

    # Behavioural aggregates from train transactions
    buy_tx = train_tx[train_tx["transaction_type"] == "Buy"]
    sell_tx = train_tx[train_tx["transaction_type"] == "Sell"]

    agg = buy_tx.groupby("customer_id").agg(
        user_n_buy_tx    =("value", "count"),
        user_total_buy   =("value", "sum"),
        user_avg_buy     =("value", "mean"),
        user_n_assets    =("isin", "nunique"),
        user_buy_std     =("value", "std"),
    ).fillna(0).reset_index()

    sell_agg = sell_tx.groupby("customer_id").agg(
        user_n_sell_tx=("value", "count"),
    ).fillna(0).reset_index()

    uf = uf.merge(agg,      on="customer_id", how="left")
    uf = uf.merge(sell_agg, on="customer_id", how="left")
    uf = uf.fillna(0)

    # Preferred category (mode)
    cat_merge = buy_tx.merge(
        pd.read_parquet(DATA_SUBDIR / "assets.parquet")[["isin", "category"]], on="isin", how="left"
    )
    pref_cat = cat_merge.groupby("customer_id")["category"].agg(
        lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"
    ).reset_index().rename(columns={"category": "user_pref_category"})
    uf = uf.merge(pref_cat, on="customer_id", how="left")

    if is_train:
        cats = uf["user_pref_category"].fillna("Unknown").unique().tolist()
        train_stats["pref_cat_map"] = {c: i for i, c in enumerate(cats)}
    uf["user_pref_cat_enc"] = uf["user_pref_category"].map(
        train_stats["pref_cat_map"]).fillna(-1)

    user_feat_cols = ["customer_id", "user_risk_score", "user_type_score", "user_cap_score",
                      "user_n_buy_tx", "user_total_buy", "user_avg_buy", "user_n_assets",
                      "user_buy_std", "user_n_sell_tx", "user_pref_cat_enc"]
    log.info(f"User features: {len(user_feat_cols)-1} cols for {len(uf)} customers")
    return uf[user_feat_cols], train_stats


# ─────────────────────────────────────────────────────────────────────────────
# ITEM features
# ─────────────────────────────────────────────────────────────────────────────

def build_item_features(assets: pd.DataFrame, profitability: pd.DataFrame,
                        train_tx: pd.DataFrame,
                        train_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    if train_stats is None:
        train_stats = {}
    is_train = "item_category_map" not in train_stats

    itf = assets.copy().merge(profitability, on="isin", how="left")

    # Encode categoricals
    for col in ["category", "subcategory", "market", "sector"]:
        key = f"item_{col}_map"
        if is_train:
            vals = itf[col].fillna("Unknown").unique().tolist()
            train_stats[key] = {v: i for i, v in enumerate(vals)}
        itf[f"item_{col}_enc"] = itf[col].map(train_stats[key]).fillna(-1)

    # Popularity from training interactions
    buy_tx = train_tx[train_tx["transaction_type"] == "Buy"]
    pop = buy_tx.groupby("isin").agg(
        item_n_buyers    =("customer_id", "nunique"),
        item_n_purchases =("value", "count"),
        item_total_vol   =("value", "sum"),
        item_avg_vol     =("value", "mean"),
    ).reset_index()
    itf = itf.merge(pop, on="isin", how="left").fillna(0)

    # Global popularity rank (normalised)
    if is_train:
        sorted_pop = np.sort(itf["item_n_buyers"].values)
        train_stats["item_pop_sorted"] = sorted_pop
    itf["item_pop_rank"] = itf["item_n_buyers"].apply(
        lambda x: np.searchsorted(train_stats["item_pop_sorted"], x)
        / max(len(train_stats["item_pop_sorted"]), 1)
    )

    # ROI features
    itf["item_roi"]       = itf["roi"].fillna(0)
    itf["item_roi_range"] = (itf["roi_max"] - itf["roi_min"]).fillna(0)  # volatility proxy
    itf["item_positive_roi"] = (itf["item_roi"] > 0).astype(int)

    item_feat_cols = ["isin", "item_category_enc", "item_subcategory_enc", "item_market_enc",
                      "item_sector_enc", "item_n_buyers", "item_n_purchases", "item_total_vol",
                      "item_avg_vol", "item_pop_rank", "item_roi", "item_roi_range",
                      "item_positive_roi"]
    log.info(f"Item features: {len(item_feat_cols)-1} cols for {len(itf)} assets")
    return itf[item_feat_cols], train_stats


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generation + interaction features
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_pairs(train_tx: pd.DataFrame, customers: pd.DataFrame,
                           assets: pd.DataFrame, is_train: bool = True,
                           n_neg_ratio: int = 4) -> pd.DataFrame:
    """
    Build (customer, item) pairs:
      Positive: items the customer bought in the interaction set
      Negative: random sample of unobserved items (implicit feedback assumption)
    """
    buy_tx = train_tx[train_tx["transaction_type"] == "Buy"]
    positives = (buy_tx.groupby(["customer_id", "isin"])
                 .size().reset_index(name="_count"))
    positives["label"] = 1

    rng = np.random.default_rng(RANDOM_STATE)
    neg_rows = []
    all_isins = assets["isin"].values
    bought_map = positives.groupby("customer_id")["isin"].apply(set).to_dict()

    active_customers = positives["customer_id"].unique()
    for cid in active_customers:
        bought = bought_map.get(cid, set())
        candidates = [i for i in all_isins if i not in bought]
        n_neg = min(len(candidates), len(bought) * n_neg_ratio)
        if n_neg == 0:
            continue
        sampled = rng.choice(candidates, n_neg, replace=False)
        for isin in sampled:
            neg_rows.append({"customer_id": cid, "isin": isin, "_count": 0, "label": 0})

    negatives = pd.DataFrame(neg_rows) if neg_rows else pd.DataFrame(
        columns=["customer_id", "isin", "_count", "label"])
    pairs = pd.concat([positives, negatives], ignore_index=True)

    # query_id for LambdaRank (group by customer)
    cust_idx = {c: i for i, c in enumerate(sorted(pairs["customer_id"].unique()))}
    pairs["query_id"] = pairs["customer_id"].map(cust_idx)

    log.info(f"Candidate pairs ({'train' if is_train else 'val'}): "
             f"{len(positives):,} pos + {len(negatives):,} neg = {len(pairs):,} total")
    return pairs


def build_interaction_features(pairs: pd.DataFrame, train_tx: pd.DataFrame,
                                customers: pd.DataFrame, assets: pd.DataFrame,
                                train_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Cross user×item features: has user bought this category before? etc."""
    if train_stats is None:
        train_stats = {}
    is_train = "inter_categories" not in train_stats

    buy_tx = train_tx[train_tx["transaction_type"] == "Buy"]
    buy_with_cat = buy_tx.merge(assets[["isin", "category"]], on="isin", how="left")

    # Category purchase history per user
    cat_hist = (buy_with_cat.groupby(["customer_id", "category"])
                .size().reset_index(name="inter_cat_buys"))
    cat_hist_pivot = cat_hist.pivot(index="customer_id", columns="category",
                                     values="inter_cat_buys").fillna(0)
    cat_hist_pivot.columns = [f"inter_cat_{c}" for c in cat_hist_pivot.columns]
    cat_hist_pivot = cat_hist_pivot.reset_index()

    pairs_aug = pairs.merge(cat_hist_pivot, on="customer_id", how="left").fillna(0)

    # Repeat purchase indicator: has this customer bought this exact asset before?
    past_buys = buy_tx.groupby(["customer_id", "isin"]).size().reset_index(name="inter_repeat_buys")
    pairs_aug = pairs_aug.merge(past_buys, on=["customer_id", "isin"], how="left").fillna(0)

    # Recency of last interaction with this asset
    last_buy = (buy_tx.groupby(["customer_id", "isin"])["transaction_date"]
                .max().reset_index(name="inter_last_buy_date"))
    last_buy["inter_last_buy_date"] = pd.to_datetime(last_buy["inter_last_buy_date"])
    reference_date = pd.to_datetime(train_tx["transaction_date"].max())
    last_buy["inter_days_since_last_buy"] = (
        reference_date - last_buy["inter_last_buy_date"]).dt.days
    pairs_aug = pairs_aug.merge(last_buy[["customer_id", "isin", "inter_days_since_last_buy"]],
                                 on=["customer_id", "isin"], how="left")
    pairs_aug["inter_days_since_last_buy"] = pairs_aug["inter_days_since_last_buy"].fillna(9999)

    inter_cols = (["customer_id", "isin", "label", "query_id"] +
                  [c for c in pairs_aug.columns if c.startswith("inter_")])
    log.info(f"Interaction features: {len([c for c in inter_cols if c.startswith('inter_')])} cols")
    return pairs_aug[inter_cols], train_stats


# ─────────────────────────────────────────────────────────────────────────────
# Assemble full feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def assemble_feature_matrix(pairs: pd.DataFrame, user_feats: pd.DataFrame,
                             item_feats: pd.DataFrame) -> pd.DataFrame:
    df = (pairs
          .merge(user_feats, on="customer_id", how="left")
          .merge(item_feats, on="isin",         how="left"))
    return df


def plot_feature_summary(train_df: pd.DataFrame) -> None:
    feat_cols = [c for c in train_df.columns
                 if c not in ["customer_id", "isin", "label", "query_id", "_count"]]
    null_pct = train_df[feat_cols].isna().mean().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feature Engineering Summary", fontsize=13, fontweight="bold")

    # Feature groups
    groups = {"user_": 0, "item_": 0, "inter_": 0}
    for col in feat_cols:
        for prefix in groups:
            if col.startswith(prefix):
                groups[prefix] += 1
    axes[0].bar(["User features", "Item features", "Interaction features"],
                list(groups.values()), color=["#1976D2", "#388E3C", "#F57C00"])
    axes[0].set_title("Feature Count by Group")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(groups.values()):
        axes[0].text(i, v + 0.1, str(v), ha="center", fontsize=10)

    # Null rates
    axes[1].barh(null_pct.index[:15], null_pct.values[:15], color="#7B1FA2")
    axes[1].set_title("Top-15 Features by Null Rate")
    axes[1].set_xlabel("Null Fraction")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_engineering_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved feature_engineering_summary.png")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G1: Robo-Advisory Portfolio Recommendation")
    print("  Step 3: Feature Engineering")
    print("=" * 65 + "\n")

    customers, assets, train_tx, val_tx, profitability = _load()

    print("[1] Building user features…")
    user_feats_train, train_stats = build_user_features(customers, train_tx)

    print("[2] Building item features…")
    item_feats_train, train_stats = build_item_features(
        assets, profitability, train_tx, train_stats)

    print("[3] Generating candidate pairs (train)…")
    train_pairs = build_candidate_pairs(train_tx, customers, assets, is_train=True)

    print("[4] Building interaction features (train)…")
    train_inter, train_stats = build_interaction_features(
        train_pairs, train_tx, customers, assets, train_stats)

    print("[5] Assembling train feature matrix…")
    train_df = assemble_feature_matrix(train_inter, user_feats_train, item_feats_train)

    print("[6] Building val feature matrix (using train stats — no leakage)…")
    user_feats_val, _ = build_user_features(customers, train_tx, train_stats)
    item_feats_val, _ = build_item_features(assets, profitability, train_tx, train_stats)
    val_pairs          = build_candidate_pairs(val_tx, customers, assets, is_train=False)
    val_inter, _       = build_interaction_features(val_pairs, train_tx, customers, assets, train_stats)
    val_df             = assemble_feature_matrix(val_inter, user_feats_val, item_feats_val)

    feat_cols = [c for c in train_df.columns
                 if c not in ["customer_id", "isin", "label", "query_id", "_count"]]
    print(f"\n  Train pairs: {len(train_df):,}  |  Val pairs: {len(val_df):,}")
    print(f"  Feature columns: {len(feat_cols)}")
    print(f"  Label balance (train): {train_df['label'].mean():.3f} positive")

    print("\n[7] Saving outputs…")
    train_df.to_parquet(DATA_SUBDIR / "train_pairs.parquet", index=False)
    val_df.to_parquet(DATA_SUBDIR   / "val_pairs.parquet",   index=False)
    joblib.dump(train_stats, DATA_SUBDIR / "fe_stats.pkl")
    log.info("Saved train_pairs.parquet, val_pairs.parquet, fe_stats.pkl")

    plot_feature_summary(train_df)

    print(f"\n  All outputs \u2192 {DATA_SUBDIR}")
    print("=" * 65)
    print("  Step 3 complete. Ready for Model Training (04_model_training.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
