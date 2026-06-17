"""
use_case_C_market/03_feature_engineering.py
=============================================
DSF504 Use Case C_markets -- Market Intelligence: Realized Volatility Prediction
ML Framework Phase 3: Feature Engineering & Data Preparation

Performance Review Improvements (v2):
  1. HAR-RV temporal lags (lag-1, lag-5 mean, lag-22 mean) via pivot/shift
     - The HAR model (Corsi 2009) is the canonical benchmark for RV prediction
     - Lag features capture volatility persistence / clustering
  2. Cross-sectional market features per time_id
     - Market-wide RV mean/std/q75 (regime context for each stock)
     - Relative position of each stock vs the market
  3. All previously unused columns now engaged:
     - book_lr_mean, book_wap_mean/min/max, trade_order_count,
       trade_price_std, trade_price_mean, trade_size_mean
  4. Semi-variance proxy (directional volatility asymmetry)
  5. Richer interaction and ratio features
  6. Volatility acceleration (log-diff of RV vs lag)

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

DATA_SUBDIR = DATA_DIR  / "optiver_volatility"
REPORT_DIR  = REPORTS_DIR / "use_case_C_markets"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PQ    = DATA_SUBDIR / "train_split.parquet"
VAL_PQ      = DATA_SUBDIR / "val_split.parquet"
TEST_PQ     = DATA_SUBDIR / "test_split.parquet"
TRAIN_FE_PQ = DATA_SUBDIR / "train_fe.parquet"
VAL_FE_PQ   = DATA_SUBDIR / "val_fe.parquet"
TEST_FE_PQ  = DATA_SUBDIR / "test_fe.parquet"
FE_STATS_PKL = DATA_SUBDIR / "fe_stats.pkl"


def safe_ratio(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b.replace(0, np.nan)
    return r.fillna(fill).astype(np.float32)


# ===========================================================================
# HAR-RV TEMPORAL LAG FEATURES
# ===========================================================================
def compute_har_lags(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute HAR-RV lag features using the full temporal panel.

    HAR (Heterogeneous Autoregressive) Model (Corsi 2009):
        RV(t) = alpha + beta_d * RV(t-1)
                      + beta_w * mean(RV(t-1..t-5))
                      + beta_m * mean(RV(t-1..t-22))

    This is THE canonical benchmark for realized volatility forecasting.
    Adds features: fe_rv_lag1, fe_rv_lag5_mean, fe_rv_lag22_mean,
                   fe_spread_lag1, fe_rv_har_ratio, fe_rv_log_diff_l1,
                   fe_trade_rv_lag1

    Implementation: pivot to (time_id x stock_id) matrix, shift, melt back.
    Cross-split continuity: val lags use train tail; test lags use val tail.
    """
    log.info("[HAR] Computing temporal lag features ...")
    _marker = "_split_marker"

    df_all = pd.concat([
        df_train.assign(**{_marker: "train"}),
        df_val.assign(**{_marker: "val"}),
        df_test.assign(**{_marker: "test"}),
    ], ignore_index=True)

    df_all = df_all.sort_values(["stock_id", "time_id"]).reset_index(drop=True)

    def make_lags(col: str, lag_names: dict[str, int | tuple]) -> pd.DataFrame:
        """
        Build lag columns for `col` via pivot->shift->melt.

        Fix: each shifted frame is melted independently then joined via
        set_index([time_id, stock_id]) + pd.concat. The prior
        merge(on=time_id) created a Cartesian product (50x50 per time_id)
        silently dropping fe_rv_lag1 and fe_rv_lag5_mean columns.
        """
        if col not in df_all.columns:
            return pd.DataFrame(columns=["time_id", "stock_id"])
        pivot = df_all.pivot_table(
            index="time_id", columns="stock_id", values=col, aggfunc="first"
        ).sort_index()
        frames = []
        for feat_name, spec in lag_names.items():
            if isinstance(spec, int):
                shifted = pivot.shift(spec)
            else:
                lag, window = spec
                shifted = pivot.shift(lag).rolling(window, min_periods=1).mean()
            melted = (
                shifted.reset_index()
                .melt(id_vars="time_id", var_name="stock_id", value_name=feat_name)
                .set_index(["time_id", "stock_id"])
            )
            frames.append(melted)
        if not frames:
            return pd.DataFrame(columns=["time_id", "stock_id"])
        return pd.concat(frames, axis=1).reset_index()

    # book_rv lags (HAR: daily=1, weekly=mean_5, monthly=mean_22)
    rv_lags = make_lags("book_rv", {
        "fe_rv_lag1":      1,
        "fe_rv_lag5_mean": (1, 5),
        "fe_rv_lag22_mean": (1, 22),
    })

    # spread and trade_rv lags (momentum in liquidity & trade activity)
    spread_lags = make_lags("book_spread_mean", {"fe_spread_lag1": 1})
    trv_lags    = make_lags("trade_rv",         {"fe_trade_rv_lag1": 1})

    # Merge into df_all
    for lag_df in [rv_lags, spread_lags, trv_lags]:
        if "stock_id" not in lag_df.columns:
            continue
        lag_df["stock_id"] = lag_df["stock_id"].astype(df_all["stock_id"].dtype)
        df_all = df_all.merge(lag_df, on=["time_id", "stock_id"], how="left")

    # Derived: HAR ratio and log-diff
    if "fe_rv_lag1" in df_all.columns:
        df_all["fe_rv_har_ratio"] = safe_ratio(
            df_all["book_rv"].fillna(0),
            df_all["fe_rv_lag1"].fillna(df_all["book_rv"].median())
        ).astype(np.float32)
        df_all["fe_rv_log_diff_l1"] = (
            np.log1p(df_all["book_rv"].fillna(0)) -
            np.log1p(df_all["fe_rv_lag1"].fillna(0))
        ).astype(np.float32)
        # HAR component: weekly minus monthly (slope of persistence curve)
        if "fe_rv_lag5_mean" in df_all.columns and "fe_rv_lag22_mean" in df_all.columns:
            df_all["fe_rv_har_slope"] = (
                df_all["fe_rv_lag5_mean"].fillna(0) -
                df_all["fe_rv_lag22_mean"].fillna(0)
            ).astype(np.float32)

    lag_cols = [c for c in df_all.columns if c.startswith("fe_rv_lag")
                or c in ("fe_spread_lag1", "fe_trade_rv_lag1",
                         "fe_rv_har_ratio", "fe_rv_log_diff_l1", "fe_rv_har_slope")]
    log.info("[HAR] Added %d lag/HAR features", len(lag_cols))

    # Split back -- preserve original row order via merge
    def _split_back(tag: str, df_orig: pd.DataFrame) -> pd.DataFrame:
        sub = df_all[df_all[_marker] == tag].drop(columns=[_marker])
        sub = sub.reset_index(drop=True)
        return sub

    return (
        _split_back("train", df_train),
        _split_back("val",   df_val),
        _split_back("test",  df_test),
    )


# ===========================================================================
# CROSS-SECTIONAL MARKET FEATURES
# ===========================================================================
def compute_market_features(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Per-time_id cross-sectional statistics across all 50 stocks.

    Rationale: In financial markets, a stock's volatility is partially
    driven by market-wide conditions. Cross-sectional features provide
    a regime context that helps the model distinguish idiosyncratic vs
    systematic volatility spikes.

    No leakage: These are statistics across stocks AT THE SAME timestamp,
    which would be observable in production (all stocks trade simultaneously).
    """
    log.info("[Market] Computing cross-sectional time_id features ...")

    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    time_stats = (
        df_all.groupby("time_id")
        .agg(
            fe_market_rv_mean  =("book_rv", "mean"),
            fe_market_rv_std   =("book_rv", "std"),
            fe_market_rv_q75   =("book_rv", lambda x: x.quantile(0.75)),
            fe_market_spread_mean=("book_spread_mean", "mean"),
            fe_market_vol_imb  =("book_vol_imb_mean", "mean"),
        )
        .reset_index()
    )
    # cast
    for c in time_stats.columns:
        if c != "time_id" and pd.api.types.is_float_dtype(time_stats[c]):
            time_stats[c] = time_stats[c].astype(np.float32)

    def _add_market(df: pd.DataFrame) -> pd.DataFrame:
        out = df.merge(time_stats, on="time_id", how="left")
        # Stock vs market ratio (complement to existing fe_rv_vs_stock_mean)
        out["fe_rv_vs_market"] = safe_ratio(
            out["book_rv"].fillna(0),
            out["fe_market_rv_mean"].replace(0, np.nan)
        )
        # Spread relative to market spread
        out["fe_spread_vs_market"] = safe_ratio(
            out["book_spread_mean"].fillna(0),
            out["fe_market_spread_mean"].replace(0, np.nan)
        )
        # High-volatility market regime flag (soft: market_rv > 75th pct of market_rv)
        market_rv_75 = float(time_stats["fe_market_rv_mean"].quantile(0.75))
        out["fe_market_rv_regime"] = (
            out["fe_market_rv_mean"].fillna(0) > market_rv_75
        ).astype(np.float32)
        return out

    return _add_market(df_train), _add_market(df_val), _add_market(df_test)


# ===========================================================================
# MAIN FEATURE ENGINEERING
# ===========================================================================
def engineer_features(
    df: pd.DataFrame,
    fe_stats: dict | None = None,
    split: str = "train",
) -> tuple[pd.DataFrame, dict]:
    """
    Build feature matrix from aggregated book+trade data.
    fe_stats is fitted on train and applied to val/test.
    """
    log.info("[FE] Engineering features for %s split (%d rows)", split, len(df))
    if fe_stats is None:
        fe_stats = {}

    _new: dict = {}

    # 1. Log-transform target
    if "target" in df.columns:
        _new["log_target"] = np.log1p(df["target"]).astype(np.float32)

    # 2. Book realized vol features
    if "book_rv" in df.columns:
        _new["fe_book_rv"]     = df["book_rv"].astype(np.float32)
        _new["fe_log_book_rv"] = np.log1p(df["book_rv"]).astype(np.float32)

    if "book_rv_l2" in df.columns:
        _new["fe_book_rv_l2"]  = df["book_rv_l2"].fillna(0).astype(np.float32)
        _new["fe_rv_l2_ratio"] = safe_ratio(
            df["book_rv_l2"].fillna(0), df["book_rv"].fillna(1e-8))
        # Volatility acceleration: log(rv / rv_l2) - vol momentum
        _new["fe_rv_log_accel"] = (
            np.log1p(df["book_rv"].fillna(0)) -
            np.log1p(df["book_rv_l2"].fillna(0))
        ).astype(np.float32)

    # 3. Spread features
    if "book_spread_mean" in df.columns:
        _new["fe_spread_mean"] = df["book_spread_mean"].astype(np.float32)
        _new["fe_log_spread"]  = np.log1p(df["book_spread_mean"]).astype(np.float32)
    if "book_spread_max" in df.columns:
        _new["fe_spread_max"]   = df["book_spread_max"].astype(np.float32)
        _new["fe_spread_spike"] = safe_ratio(
            df["book_spread_max"], df["book_spread_mean"].replace(0, np.nan))
        # Spread range instability
        if "book_spread_mean" in df.columns:
            _new["fe_spread_range"] = (
                df["book_spread_max"] - df["book_spread_mean"]
            ).fillna(0).astype(np.float32)

    # 4. Volume imbalance
    if "book_vol_imb_mean" in df.columns:
        _new["fe_vol_imb_mean"] = df["book_vol_imb_mean"].astype(np.float32)
        _new["fe_vol_imb_abs"]  = df["book_vol_imb_mean"].abs().astype(np.float32)
    if "book_vol_imb_std" in df.columns:
        _new["fe_vol_imb_std"] = df["book_vol_imb_std"].astype(np.float32)

    # 5. WAP statistics
    if "book_wap_std" in df.columns:
        _new["fe_wap_std"]     = df["book_wap_std"].astype(np.float32)
        _new["fe_log_wap_std"] = np.log1p(df["book_wap_std"]).astype(np.float32)
    if "book_wap_range" in df.columns:
        _new["fe_wap_range"]   = df["book_wap_range"].astype(np.float32)
    if "book_lr_std" in df.columns:
        _new["fe_lr_std"]      = df["book_lr_std"].astype(np.float32)
    if "book_lr_max_abs" in df.columns:
        _new["fe_lr_max_abs"]  = df["book_lr_max_abs"].astype(np.float32)

    # 6. Log-return mean (direction bias & semi-variance proxy) -- NEW
    if "book_lr_mean" in df.columns:
        _new["fe_lr_mean"]        = df["book_lr_mean"].astype(np.float32)
        _new["fe_lr_mean_abs"]    = df["book_lr_mean"].abs().astype(np.float32)
        # Directional signed volatility contribution
        _new["fe_lr_sign"]        = np.sign(df["book_lr_mean"]).astype(np.float32)
        # Normalized return: mean / std (signal-to-noise, Sharpe-like)
        if "book_lr_std" in df.columns:
            _new["fe_lr_sharpe"]  = safe_ratio(
                df["book_lr_mean"].fillna(0), df["book_lr_std"].fillna(1e-8))
        # Semi-variance proxy: squared upside vs downside log-return contribution
        # Upside: (max(lr_mean, 0))^2; Downside: (min(lr_mean, 0))^2
        lr = df["book_lr_mean"].fillna(0)
        _new["fe_lr_up_proxy"]   = np.where(lr > 0, lr ** 2, 0).astype(np.float32)
        _new["fe_lr_down_proxy"] = np.where(lr < 0, lr ** 2, 0).astype(np.float32)
        # Asymmetry: upside - downside (positive = bullish vol skew)
        _new["fe_rv_asymmetry"]  = (_new["fe_lr_up_proxy"] -
                                     _new["fe_lr_down_proxy"]).astype(np.float32)

    # 7. WAP level features -- NEW (previously unused)
    if "book_wap_mean" in df.columns:
        _new["fe_wap_level"] = df["book_wap_mean"].astype(np.float32)
    if "book_wap_min" in df.columns and "book_wap_max" in df.columns:
        wap_range = (df["book_wap_max"] - df["book_wap_min"]).fillna(0)
        _new["fe_wap_intraday_range"] = wap_range.astype(np.float32)
        if "book_wap_mean" in df.columns:
            # Intraday range as % of WAP level (normalized volatility-in-level)
            _new["fe_wap_range_pct"] = safe_ratio(
                wap_range, df["book_wap_mean"].replace(0, np.nan))

    # 8. Order size features
    if "book_bid_size1_sum" in df.columns and "book_ask_size1_sum" in df.columns:
        bid_sum = df["book_bid_size1_sum"].fillna(0)
        ask_sum = df["book_ask_size1_sum"].fillna(0)
        _new["fe_bid_ask_size_ratio"] = safe_ratio(bid_sum, ask_sum)
        _new["fe_total_book_volume"]  = (bid_sum + ask_sum).astype(np.float32)
        _new["fe_log_total_volume"]   = np.log1p(bid_sum + ask_sum).astype(np.float32)
        # Order book depth product -- liquidity signal
        _new["fe_book_depth_product"] = np.log1p(bid_sum * ask_sum + 1).astype(np.float32)
    if "book_n_ticks" in df.columns:
        _new["fe_n_ticks"]    = df["book_n_ticks"].astype(np.float32)
        _new["fe_log_n_ticks"] = np.log1p(df["book_n_ticks"]).astype(np.float32)

    # 9. Trade features
    if "trade_rv" in df.columns:
        _new["fe_trade_rv"]          = df["trade_rv"].fillna(0).astype(np.float32)
        _new["fe_log_trade_rv"]      = np.log1p(df["trade_rv"].fillna(0)).astype(np.float32)
        if "book_rv" in df.columns:
            _new["fe_rv_book_trade_ratio"] = safe_ratio(
                df["trade_rv"].fillna(0), df["book_rv"].fillna(1e-8))
            # Trade fraction of total vol signal
            tot = df["book_rv"].fillna(0) + df["trade_rv"].fillna(0)
            _new["fe_trade_vol_frac"] = safe_ratio(
                df["trade_rv"].fillna(0), tot.replace(0, np.nan))
    if "trade_size_sum" in df.columns:
        _new["fe_trade_volume"]     = df["trade_size_sum"].fillna(0).astype(np.float32)
        _new["fe_log_trade_volume"] = np.log1p(df["trade_size_sum"].fillna(0)).astype(np.float32)
    if "trade_count" in df.columns:
        _new["fe_trade_count"]      = df["trade_count"].fillna(0).astype(np.float32)
        _new["fe_log_trade_count"]  = np.log1p(df["trade_count"].fillna(0)).astype(np.float32)

    # 10. Trade quality features -- NEW (previously unused)
    if "trade_price_std" in df.columns:
        _new["fe_trade_price_disp"] = df["trade_price_std"].fillna(0).astype(np.float32)
        _new["fe_log_trade_price_disp"] = np.log1p(df["trade_price_std"].fillna(0)).astype(np.float32)
    if "trade_size_mean" in df.columns:
        _new["fe_avg_trade_size"]   = df["trade_size_mean"].fillna(0).astype(np.float32)
        _new["fe_log_avg_trade_size"] = np.log1p(df["trade_size_mean"].fillna(0)).astype(np.float32)
    if "trade_order_count" in df.columns:
        _new["fe_order_count"]      = df["trade_order_count"].fillna(0).astype(np.float32)
        _new["fe_log_order_count"]  = np.log1p(df["trade_order_count"].fillna(0)).astype(np.float32)
        # Order-to-trade ratio: number of distinct orders per trade (fragmentation)
        if "trade_count" in df.columns:
            _new["fe_order_fragmentation"] = safe_ratio(
                df["trade_order_count"].fillna(0),
                df["trade_count"].fillna(1))

    # Book-trade price divergence -- NEW
    if "book_wap_mean" in df.columns and "trade_price_mean" in df.columns:
        wap_mid = df["book_wap_mean"].fillna(0)
        trd_mid = df["trade_price_mean"].fillna(0)
        _new["fe_book_trade_price_gap"] = (wap_mid - trd_mid).astype(np.float32)
        _new["fe_book_trade_price_gap_abs"] = (wap_mid - trd_mid).abs().astype(np.float32)

    # 11. Interaction features
    if "fe_spread_mean" in _new and "fe_vol_imb_abs" in _new:
        _new["fe_spread_x_imb"] = (
            pd.Series(_new["fe_spread_mean"]) * pd.Series(_new["fe_vol_imb_abs"])
        ).astype(np.float32)
    if "fe_lr_std" in _new and "fe_spread_mean" in _new:
        _new["fe_volatility_spread_ratio"] = safe_ratio(
            pd.Series(_new["fe_lr_std"]), pd.Series(_new["fe_spread_mean"]))
    if "fe_trade_volume" in _new and "fe_total_book_volume" in _new:
        _new["fe_trade_book_vol_ratio"] = safe_ratio(
            pd.Series(_new["fe_trade_volume"]),
            pd.Series(_new["fe_total_book_volume"]))
    # Price impact proxy: trade_rv per unit of trade volume
    if "fe_trade_rv" in _new and "fe_log_trade_volume" in _new:
        _new["fe_price_impact"] = safe_ratio(
            pd.Series(_new["fe_trade_rv"]),
            pd.Series(_new["fe_log_trade_volume"]).replace(0, np.nan))

    # 12. Stock-ID as integer feature (stock fixed-effect)
    if "stock_id" in df.columns:
        _new["fe_stock_id"] = df["stock_id"].astype(np.int16)

    df_out = pd.concat([df, pd.DataFrame(_new, index=df.index)], axis=1)

    # 13. Stock-level mean RV (cross-sectional normalisation, fit on train)
    if split == "train":
        stock_mean_rv = (df_out.groupby("stock_id")["book_rv"]
                         .mean().rename("stock_mean_rv"))
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

    # 14. Cross-sectional ranks within time_id
    if "time_id" in df_out.columns:
        for rank_col, src_col in [("fe_book_rv_rank", "book_rv"),
                                   ("fe_spread_rank", "book_spread_mean"),
                                   ("fe_trade_rv_rank", "trade_rv")]:
            if src_col in df_out.columns:
                df_out[rank_col] = (
                    df_out.groupby("time_id")[src_col]
                    .rank(pct=True).astype(np.float32)
                )

    # 15. Winsorise float FE columns (fit on train)
    fe_cols = [c for c in df_out.columns
               if c.startswith("fe_") and pd.api.types.is_float_dtype(df_out[c])]

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


def scale_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, fe_stats: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fe_cols = sorted([c for c in train.columns
                      if c.startswith("fe_") and pd.api.types.is_float_dtype(train[c])])
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


def save_feature_list(df: pd.DataFrame) -> None:
    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    feat_df = pd.DataFrame({
        "feature": fe_cols,
        "dtype":   [str(df[c].dtype) for c in fe_cols],
        "n_null":  [int(df[c].isna().sum()) for c in fe_cols],
        "mean":    [round(float(df[c].mean()), 6)
                    if pd.api.types.is_numeric_dtype(df[c]) else None for c in fe_cols],
        "std":     [round(float(df[c].std()), 6)
                    if pd.api.types.is_numeric_dtype(df[c]) else None for c in fe_cols],
    })
    feat_df.to_csv(REPORT_DIR / "engineered_features_list.csv", index=False)
    log.info("Saved engineered_features_list.csv (%d features)", len(fe_cols))


def plot_fe_summary(train: pd.DataFrame, train_raw: pd.DataFrame) -> None:
    raw_pairs = {
        "book_rv": "fe_log_book_rv",
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


def main() -> None:
    log.info("=" * 60)
    log.info("Use Case C_markets -- Step 3: Feature Engineering (v2)")
    log.info("=" * 60)

    for p in (TRAIN_PQ, VAL_PQ, TEST_PQ):
        if not p.exists():
            log.error("%s not found -- run Step 1 first", p.name)
            sys.exit(1)

    log.info("Loading splits ...")
    df_train = pd.read_parquet(TRAIN_PQ)
    df_val   = pd.read_parquet(VAL_PQ)
    df_test  = pd.read_parquet(TEST_PQ)
    log.info("  train=%d  val=%d  test=%d", len(df_train), len(df_val), len(df_test))

    # --- HAR-RV temporal lags (cross-split, uses all three splits)
    df_train, df_val, df_test = compute_har_lags(df_train, df_val, df_test)

    # --- Cross-sectional market features (per time_id)
    df_train, df_val, df_test = compute_market_features(df_train, df_val, df_test)

    # --- Per-split feature engineering (scaler/clip fitted on train only)
    fe_stats: dict = {}
    df_train, fe_stats = engineer_features(df_train, fe_stats, split="train")
    df_val,   _        = engineer_features(df_val,   fe_stats, split="val")
    df_test,  _        = engineer_features(df_test,  fe_stats, split="test")

    df_train, df_val, df_test = scale_features(df_train, df_val, df_test, fe_stats)

    # Align val/test fe_ columns to match train
    train_fe_cols = [c for c in df_train.columns if c.startswith("fe_")]
    for df_split, sname in [(df_val, "val"), (df_test, "test")]:
        for col in train_fe_cols:
            if col not in df_split.columns:
                log.warning("  [align] %s missing in %s -- filling 0", col, sname)
                df_split[col] = np.float32(0.0)
        extra = [c for c in df_split.columns
                 if c.startswith("fe_") and c not in train_fe_cols]
        if extra:
            df_split.drop(columns=extra, inplace=True)
    log.info("Column alignment done: %d fe_ columns in all splits", len(train_fe_cols))

    # Save parquets
    df_train.to_parquet(TRAIN_FE_PQ, index=False)
    df_val.to_parquet(VAL_FE_PQ,   index=False)
    df_test.to_parquet(TEST_FE_PQ,  index=False)
    log.info("Saved train_fe / val_fe / test_fe parquets")

    with open(FE_STATS_PKL, "wb") as f:
        pickle.dump(fe_stats, f)
    log.info("Saved fe_stats.pkl")

    save_feature_list(df_train)
    train_raw = pd.read_parquet(TRAIN_PQ)
    plot_fe_summary(df_train, train_raw)

    fe_cols = [c for c in df_train.columns if c.startswith("fe_")]
    log.info("-" * 60)
    log.info("Step 3 complete.")
    log.info("  FE features : %d  (was 31)", len(fe_cols))
    log.info("  New groups  : HAR lags, market xs, semi-var proxy,")
    log.info("                trade quality, WAP level, vol acceleration")
    log.info("  Train shape : %s", df_train.shape)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
