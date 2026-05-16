"""
use_case_C_market/03_feature_engineering.py
=============================================
DSF504 Use Case C_markets — Market Intelligence: Realized Volatility Prediction
ML Framework Phase 3: Feature Engineering & Data Preparation

Engineering steps:
  1. Log1p transform of target (regression on log scale, expm1 at inference)
  2. Realized volatility in early/late sub-windows (first/last 300 sec)
  3. WAP momentum & mean-reversion features
  4. Bid-ask spread percentiles (stable spread vs spike spread)
  5. Volume imbalance stats
  6. Trade-to-book ratio (trade RV / book RV)
  7. Stock-level rolling volatility mean (cross-sectional normalisation)
  8. Winsorisation at 1st/99th percentile (on train, applied to val/test)
  9. StandardScaler (fitted on train only)

Run:
    cd C:/DSF504
    python use_case_C_market/03_feature_engineering.py
"""
from __future__ import annotations

import sys
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE

from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_SUBDIR = DATA_DIR  / "optiver_volatility"
REPORT_DIR  = REPORTS_DIR / "use_case_C_markets"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PQ = DATA_SUBDIR / "train_split.parquet"
VAL_PQ   = DATA_SUBDIR / "val_split.parquet"
TEST_PQ  = DATA_SUBDIR / "test_split.parquet"

TRAIN_FE_PQ = DATA_SUBDIR / "train_fe.parquet"
VAL_FE_PQ   = DATA_SUBDIR / "val_fe.parquet"
TEST_FE_PQ  = DATA_SUBDIR / "test_fe.parquet"

FE_STATS_PKL = DATA_SUBDIR / "fe_stats.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def safe_ratio(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b.replace(0, np.nan)
    return r.fillna(fill).astype(np.float32)


def engineer_features(df: pd.DataFrame,
                       fe_stats: dict | None = None,
                       split: str = "train") -> tuple[pd.DataFrame, dict]:
    """
    Build feature matrix from aggregated book+trade data.
    fe_stats is fitted on train and applied to val/test.
    """
    log.info("[FE] Engineering features for %s split (%d rows)", split, len(df))

    _new: dict = {}

    # ── 1. Log-transform target ────────────────────────────────────────────────
    if "target" in df.columns:
        _new["log_target"] = np.log1p(df["target"]).astype(np.float32)

    # ── 2. Book realized vol features ─────────────────────────────────────────
    if "book_rv" in df.columns:
        _new["fe_book_rv"]        = df["book_rv"].astype(np.float32)
        _new["fe_log_book_rv"]    = np.log1p(df["book_rv"]).astype(np.float32)

    if "book_rv_l2" in df.columns:
        _new["fe_book_rv_l2"]     = df["book_rv_l2"].fillna(0).astype(np.float32)
        _new["fe_rv_l2_ratio"]    = safe_ratio(
            df["book_rv_l2"].fillna(0), df["book_rv"].fillna(1e-8))

    # ── 3. Spread features ────────────────────────────────────────────────────
    if "book_spread_mean" in df.columns:
        _new["fe_spread_mean"]    = df["book_spread_mean"].astype(np.float32)
        _new["fe_log_spread"]     = np.log1p(df["book_spread_mean"]).astype(np.float32)
    if "book_spread_max" in df.columns:
        _new["fe_spread_max"]     = df["book_spread_max"].astype(np.float32)
        _new["fe_spread_spike"]   = safe_ratio(
            df["book_spread_max"], df["book_spread_mean"].replace(0, np.nan))

    # ── 4. Volume imbalance ───────────────────────────────────────────────────
    if "book_vol_imb_mean" in df.columns:
        _new["fe_vol_imb_mean"]   = df["book_vol_imb_mean"].astype(np.float32)
        _new["fe_vol_imb_abs"]    = df["book_vol_imb_mean"].abs().astype(np.float32)
    if "book_vol_imb_std" in df.columns:
        _new["fe_vol_imb_std"]    = df["book_vol_imb_std"].astype(np.float32)

    # ── 5. WAP statistics ─────────────────────────────────────────────────────
    if "book_wap_std" in df.columns:
        _new["fe_wap_std"]        = df["book_wap_std"].astype(np.float32)
        _new["fe_log_wap_std"]    = np.log1p(df["book_wap_std"]).astype(np.float32)
    if "book_wap_range" in df.columns:
        _new["fe_wap_range"]      = df["book_wap_range"].astype(np.float32)
    if "book_lr_std" in df.columns:
        _new["fe_lr_std"]         = df["book_lr_std"].astype(np.float32)
    if "book_lr_max_abs" in df.columns:
        _new["fe_lr_max_abs"]     = df["book_lr_max_abs"].astype(np.float32)

    # ── 6. Order size features ────────────────────────────────────────────────
    if "book_bid_size1_sum" in df.columns and "book_ask_size1_sum" in df.columns:
        bid_sum = df["book_bid_size1_sum"].fillna(0)
        ask_sum = df["book_ask_size1_sum"].fillna(0)
        _new["fe_bid_ask_size_ratio"]  = safe_ratio(bid_sum, ask_sum)
        _new["fe_total_book_volume"]   = (bid_sum + ask_sum).astype(np.float32)
        _new["fe_log_total_volume"]    = np.log1p(bid_sum + ask_sum).astype(np.float32)
    if "book_n_ticks" in df.columns:
        _new["fe_n_ticks"]            = df["book_n_ticks"].astype(np.float32)

    # ── 7. Trade features ─────────────────────────────────────────────────────
    if "trade_rv" in df.columns:
        _new["fe_trade_rv"]            = df["trade_rv"].fillna(0).astype(np.float32)
        _new["fe_log_trade_rv"]        = np.log1p(df["trade_rv"].fillna(0)).astype(np.float32)
        if "book_rv" in df.columns:
            _new["fe_rv_book_trade_ratio"] = safe_ratio(
                df["trade_rv"].fillna(0), df["book_rv"].fillna(1e-8))
    if "trade_size_sum" in df.columns:
        _new["fe_trade_volume"]        = df["trade_size_sum"].fillna(0).astype(np.float32)
        _new["fe_log_trade_volume"]    = np.log1p(df["trade_size_sum"].fillna(0)).astype(np.float32)
    if "trade_count" in df.columns:
        _new["fe_trade_count"]         = df["trade_count"].fillna(0).astype(np.float32)

    # ── 8. Interaction features ───────────────────────────────────────────────
    if "fe_spread_mean" in _new and "fe_vol_imb_abs" in _new:
        _new["fe_spread_x_imb"]   = (_new["fe_spread_mean"] *
                                      _new["fe_vol_imb_abs"]).astype(np.float32)
    if "fe_lr_std" in _new and "fe_spread_mean" in _new:
        _new["fe_volatility_spread_ratio"] = safe_ratio(
            pd.Series(_new["fe_lr_std"]), pd.Series(_new["fe_spread_mean"]))

    # ── 9. Stock-ID as integer feature ───────────────────────────────────────
    if "stock_id" in df.columns:
        _new["fe_stock_id"] = df["stock_id"].astype(np.int16)

    # Concatenate all new features
    df_out = pd.concat([df, pd.DataFrame(_new, index=df.index)], axis=1)

    # ── 10. Stock-level mean RV (cross-sectional) ─────────────────────────────
    #   Fit on train, apply to val/test using saved mapping
    if split == "train":
        stock_mean_rv = (df_out.groupby("stock_id")["book_rv"]
                                .mean()
                                .rename("stock_mean_rv"))
        fe_stats["stock_mean_rv"] = stock_mean_rv.to_dict()

    if fe_stats and "stock_mean_rv" in fe_stats:
        df_out["fe_stock_mean_rv"] = (
            df_out["stock_id"].map(fe_stats["stock_mean_rv"])
                              .fillna(df_out.get("book_rv", pd.Series(0)).mean())
                              .astype(np.float32)
        )
        if "book_rv" in df_out.columns:
            df_out["fe_rv_vs_stock_mean"] = safe_ratio(
                df_out["book_rv"].fillna(0),
                df_out["fe_stock_mean_rv"].replace(0, np.nan))

    # ── 11. Winsorise continuous FE columns ───────────────────────────────────
    fe_cols = [c for c in df_out.columns if c.startswith("fe_")
               and pd.api.types.is_float_dtype(df_out[c])]

    if split == "train":
        clip_stats: dict = {}
        for col in fe_cols:
            lo = float(df_out[col].quantile(0.01))
            hi = float(df_out[col].quantile(0.99))
            clip_stats[col] = (lo, hi)
        fe_stats["clip"] = clip_stats
    else:
        clip_stats = fe_stats.get("clip", {})

    for col in fe_cols:
        if col in clip_stats:
            lo, hi = clip_stats[col]
            df_out[col] = df_out[col].clip(lo, hi)

    log.info("  FE columns created: %d", len(fe_cols))
    return df_out, fe_stats


def scale_features(train: pd.DataFrame,
                   val: pd.DataFrame,
                   test: pd.DataFrame,
                   fe_stats: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on train FE columns, apply to val/test."""
    fe_cols = sorted([c for c in train.columns if c.startswith("fe_") and
                      pd.api.types.is_float_dtype(train[c])])
    if not fe_cols:
        log.info("  No float FE columns to scale")
        return train, val, test

    scaler = StandardScaler()
    train[fe_cols] = scaler.fit_transform(train[fe_cols]).astype(np.float32)
    val[fe_cols]   = scaler.transform(val[fe_cols]).astype(np.float32)
    test[fe_cols]  = scaler.transform(test[fe_cols]).astype(np.float32)

    fe_stats["scaler"]  = scaler
    fe_stats["fe_cols"] = fe_cols
    log.info("  Scaled %d FE columns with StandardScaler", len(fe_cols))
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# reporting
# ══════════════════════════════════════════════════════════════════════════════

def save_feature_list(df: pd.DataFrame) -> None:
    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    feat_df = pd.DataFrame({
        "feature":    fe_cols,
        "dtype":      [str(df[c].dtype) for c in fe_cols],
        "n_null":     [int(df[c].isna().sum()) for c in fe_cols],
        "mean":       [round(df[c].mean(), 6) if pd.api.types.is_numeric_dtype(df[c]) else None
                       for c in fe_cols],
        "std":        [round(df[c].std(), 6) if pd.api.types.is_numeric_dtype(df[c]) else None
                       for c in fe_cols],
    })
    feat_df.to_csv(REPORT_DIR / "engineered_features_list.csv", index=False)
    log.info("Saved engineered_features_list.csv (%d features)", len(fe_cols))


def plot_fe_summary(train: pd.DataFrame, train_raw: pd.DataFrame) -> None:
    """Side-by-side raw vs FE distributions for key features."""
    raw_pairs = {
        "book_rv":        "fe_log_book_rv",
        "book_spread_mean": "fe_log_spread",
    }

    pairs = [(r, f) for r, f in raw_pairs.items()
             if r in train_raw.columns and f in train.columns]
    if not pairs:
        return

    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    fig.patch.set_facecolor("#1A1A2E")
    if n == 1:
        axes = [axes]

    for i, (raw_col, fe_col) in enumerate(pairs):
        raw_vals = train_raw[raw_col].dropna().clip(
            *train_raw[raw_col].quantile([0.01, 0.99]))
        fe_vals  = train[fe_col].dropna().clip(
            *train[fe_col].quantile([0.01, 0.99]))

        for ax in axes[i]:
            ax.set_facecolor("#1A1A2E")
            ax.tick_params(colors="white")

        axes[i][0].hist(raw_vals, bins=60, color="#42A5F5", edgecolor="none", alpha=0.85)
        axes[i][0].set_title(f"RAW: {raw_col}", color="white")

        axes[i][1].hist(fe_vals,  bins=60, color="#66BB6A", edgecolor="none", alpha=0.85)
        axes[i][1].set_title(f"FE:  {fe_col}", color="white")

    plt.suptitle("Raw vs Engineered Features", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "engineered_feature_summary.png",
                dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()
    log.info("Saved engineered_feature_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 60)
    log.info("Use Case C_markets — Step 3: Feature Engineering")
    log.info("=" * 60)

    for p in (TRAIN_PQ, VAL_PQ, TEST_PQ):
        if not p.exists():
            log.error("%s not found — run Step 1 first", p.name)
            sys.exit(1)

    log.info("Loading splits …")
    df_train = pd.read_parquet(TRAIN_PQ)
    df_val   = pd.read_parquet(VAL_PQ)
    df_test  = pd.read_parquet(TEST_PQ)
    log.info("  train=%d  val=%d  test=%d", len(df_train), len(df_val), len(df_test))

    fe_stats: dict = {}

    df_train, fe_stats = engineer_features(df_train, fe_stats, split="train")
    df_val,   _        = engineer_features(df_val,   fe_stats, split="val")
    df_test,  _        = engineer_features(df_test,  fe_stats, split="test")

    df_train, df_val, df_test = scale_features(df_train, df_val, df_test, fe_stats)

    # Save parquets
    df_train.to_parquet(TRAIN_FE_PQ, index=False)
    df_val.to_parquet(VAL_FE_PQ,   index=False)
    df_test.to_parquet(TEST_FE_PQ,  index=False)
    log.info("Saved train_fe.parquet / val_fe.parquet / test_fe.parquet")

    # Save fe_stats
    with open(FE_STATS_PKL, "wb") as f:
        pickle.dump(fe_stats, f)
    log.info("Saved fe_stats.pkl")

    # Reports
    save_feature_list(df_train)
    train_raw = pd.read_parquet(DATA_SUBDIR / "train_split.parquet")
    plot_fe_summary(df_train, train_raw)

    # Summary stats
    fe_cols = [c for c in df_train.columns if c.startswith("fe_")]
    log.info("-" * 60)
    log.info("Step 3 complete.")
    log.info("  FE features: %d", len(fe_cols))
    log.info("  Train shape: %s", df_train.shape)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
