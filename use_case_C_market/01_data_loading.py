"""
use_case_C_market/01_data_loading.py
=====================================
DSF504 Use Case C_markets — Market Intelligence: Realized Volatility Prediction
ML Framework Phase 1: Data Loading & Initial Profiling

Dataset : Optiver Realized Volatility Prediction (Kaggle)
          book_train.parquet  — order-book snapshots (bid/ask price & size, L1+L2)
          trade_train.parquet — trade records (price, size, order_count)
          train.csv           — targets (stock_id, time_id, target = 10-min realized vol)
Task    : Regression (predict 10-minute realized volatility)
Metric  : RMSPE = sqrt( mean( ((y_pred - y_true) / y_true)^2 ) )

Run:
    cd C:/DSF504
    python use_case_C_market/01_data_loading.py
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
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE

# ── UTF-8 encoding guard (fixes garbled output on Windows) ────────────────────
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
DATA_SUBDIR  = DATA_DIR / "optiver_volatility"
REPORT_DIR   = REPORTS_DIR / "use_case_C_markets"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_SUBDIR.mkdir(parents=True, exist_ok=True)

BOOK_TRAIN   = DATA_SUBDIR / "book_train.parquet"
TRADE_TRAIN  = DATA_SUBDIR / "trade_train.parquet"
TRAIN_CSV    = DATA_SUBDIR / "train.csv"

# processed outputs
TRAIN_RAW_PQ = DATA_SUBDIR / "train_raw.parquet"
TRAIN_PQ     = DATA_SUBDIR / "train_split.parquet"
VAL_PQ       = DATA_SUBDIR / "val_split.parquet"
TEST_PQ      = DATA_SUBDIR / "test_split.parquet"

# ── constants ──────────────────────────────────────────────────────────────────
VAL_SIZE     = 0.10   # 10% validation
TEST_SIZE    = 0.10   # 10% test


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def realized_volatility(log_returns: np.ndarray) -> float:
    """RV = sqrt(sum of squared log-returns)."""
    return float(np.sqrt(np.sum(log_returns ** 2)))


def wap(bp1, ap1, bs1, as1):
    """Weighted Average Price from L1 order book."""
    return (bp1 * as1 + ap1 * bs1) / (bs1 + as1)


def aggregate_book(book: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw order-book ticks to one row per (stock_id, time_id).
    Returns rich features: WAP stats, bid-ask spread, volume imbalance, RV.
    """
    log.info("Aggregating book data -> %d rows", len(book))

    feats: list[dict] = []
    grouped = book.groupby(["stock_id", "time_id"], sort=False)

    for (sid, tid), grp in grouped:
        grp = grp.sort_values("seconds_in_bucket")

        # Weighted average price (L1 only)
        w = wap(grp["bid_price1"], grp["ask_price1"],
                grp["bid_size1"], grp["ask_size1"]).values

        # Log returns & realized volatility
        lr = np.log(w[1:] / w[:-1]) if len(w) > 1 else np.array([0.0])
        rv = realized_volatility(lr)

        # Bid-ask spread (L1)
        spread = ((grp["ask_price1"] - grp["bid_price1"]) /
                  ((grp["bid_price1"] + grp["ask_price1"]) / 2))

        # Volume imbalance L1
        vol_imb = ((grp["bid_size1"] - grp["ask_size1"]) /
                   (grp["bid_size1"] + grp["ask_size1"]))

        # L2 WAP if available
        has_l2 = ("bid_price2" in grp.columns and
                  grp["bid_price2"].notna().any())
        if has_l2:
            w2 = wap(grp["bid_price2"], grp["ask_price2"],
                     grp["bid_size2"], grp["ask_size2"]).values
            lr2 = np.log(w2[1:] / w2[:-1]) if len(w2) > 1 else np.array([0.0])
            rv2 = realized_volatility(lr2)
        else:
            rv2 = np.nan

        row = {
            "stock_id":         sid,
            "time_id":          tid,
            # WAP statistics
            "book_wap_mean":    float(np.mean(w)),
            "book_wap_std":     float(np.std(w)) if len(w) > 1 else 0.0,
            "book_wap_min":     float(np.min(w)),
            "book_wap_max":     float(np.max(w)),
            "book_wap_range":   float(np.max(w) - np.min(w)),
            # Realized volatility
            "book_rv":          rv,
            "book_rv_l2":       rv2,
            # Log return stats
            "book_lr_mean":     float(np.mean(lr)),
            "book_lr_std":      float(np.std(lr)) if len(lr) > 1 else 0.0,
            "book_lr_max_abs":  float(np.max(np.abs(lr))) if len(lr) > 0 else 0.0,
            # Spread & imbalance
            "book_spread_mean": float(spread.mean()),
            "book_spread_max":  float(spread.max()),
            "book_vol_imb_mean":float(vol_imb.mean()),
            "book_vol_imb_std": float(vol_imb.std()) if len(vol_imb) > 1 else 0.0,
            # Size totals
            "book_bid_size1_sum": float(grp["bid_size1"].sum()),
            "book_ask_size1_sum": float(grp["ask_size1"].sum()),
            "book_n_ticks":     len(grp),
        }
        feats.append(row)

    return pd.DataFrame(feats)


def aggregate_trade(trade: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw trade ticks to one row per (stock_id, time_id)."""
    log.info("Aggregating trade data -> %d rows", len(trade))

    feats: list[dict] = []
    grouped = trade.groupby(["stock_id", "time_id"], sort=False)

    for (sid, tid), grp in grouped:
        grp = grp.sort_values("seconds_in_bucket")

        prices = grp["price"].values
        sizes  = grp["size"].values

        # Log returns on trade prices
        lr = np.log(prices[1:] / prices[:-1]) if len(prices) > 1 else np.array([0.0])
        rv = realized_volatility(lr)

        row = {
            "stock_id":           sid,
            "time_id":            tid,
            "trade_rv":           rv,
            "trade_price_mean":   float(np.mean(prices)),
            "trade_price_std":    float(np.std(prices)) if len(prices) > 1 else 0.0,
            "trade_size_sum":     float(np.sum(sizes)),
            "trade_size_mean":    float(np.mean(sizes)),
            "trade_count":        len(grp),
            "trade_order_count":  float(grp["order_count"].sum()) if "order_count" in grp.columns else np.nan,
        }
        feats.append(row)

    return pd.DataFrame(feats)


def generate_synthetic_data(n_stocks: int = 40, n_times: int = 500) -> None:
    """
    Generate synthetic Optiver-style data when real data is absent.
    Produces book_train.parquet, trade_train.parquet, train.csv.
    """
    log.info("Real data not found — generating synthetic Optiver-style dataset")
    rng = np.random.default_rng(RANDOM_STATE)

    stock_ids = list(range(n_stocks))
    time_ids  = list(range(n_times))

    # ── book data ──────────────────────────────────────────────────────────────
    book_rows: list[dict] = []
    for sid in stock_ids:
        base_price = rng.uniform(50, 500)
        for tid in time_ids:
            for sec in range(0, 600, 10):          # 60 ticks per window
                mid  = base_price * (1 + rng.normal(0, 0.002))
                half = mid * rng.uniform(0.0001, 0.001)
                bp1  = mid - half;  ap1 = mid + half
                bp2  = bp1 - rng.uniform(0, half); ap2 = ap1 + rng.uniform(0, half)
                book_rows.append({
                    "stock_id": sid, "time_id": tid,
                    "seconds_in_bucket": sec,
                    "bid_price1": bp1, "ask_price1": ap1,
                    "bid_size1":  rng.integers(100, 1000),
                    "ask_size1":  rng.integers(100, 1000),
                    "bid_price2": bp2, "ask_price2": ap2,
                    "bid_size2":  rng.integers(50, 500),
                    "ask_size2":  rng.integers(50, 500),
                })
    book_df = pd.DataFrame(book_rows)
    book_df.to_parquet(BOOK_TRAIN, index=False)
    log.info("Saved synthetic book_train.parquet  (%d rows)", len(book_df))

    # ── trade data ─────────────────────────────────────────────────────────────
    trade_rows: list[dict] = []
    for sid in stock_ids:
        base_price = rng.uniform(50, 500)
        for tid in time_ids:
            n_trades = rng.integers(5, 30)
            secs = sorted(rng.choice(600, size=n_trades, replace=False))
            price = base_price
            for sec in secs:
                price *= (1 + rng.normal(0, 0.001))
                trade_rows.append({
                    "stock_id": sid, "time_id": tid,
                    "seconds_in_bucket": sec,
                    "price": price,
                    "size":  int(rng.integers(10, 500)),
                    "order_count": int(rng.integers(1, 10)),
                })
    trade_df = pd.DataFrame(trade_rows)
    trade_df.to_parquet(TRADE_TRAIN, index=False)
    log.info("Saved synthetic trade_train.parquet (%d rows)", len(trade_df))

    # ── target: compute realized vol from book WAP ─────────────────────────────
    target_rows: list[dict] = []
    bk = book_df.sort_values(["stock_id","time_id","seconds_in_bucket"])
    for (sid, tid), grp in bk.groupby(["stock_id","time_id"]):
        w  = wap(grp["bid_price1"], grp["ask_price1"],
                 grp["bid_size1"],  grp["ask_size1"]).values
        lr = np.log(w[1:] / w[:-1]) if len(w) > 1 else np.array([0.0])
        rv = realized_volatility(lr)
        target_rows.append({"stock_id": sid, "time_id": tid, "target": rv})
    target_df = pd.DataFrame(target_rows)
    target_df.to_csv(TRAIN_CSV, index=False)
    log.info("Saved synthetic train.csv (%d rows)", len(target_df))


# ══════════════════════════════════════════════════════════════════════════════
# profiling helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cardinality_type(n_unique: int, n_rows: int) -> str:
    """Classify a column's cardinality (matches dashboard expectations)."""
    if n_unique == 1:
        return "constant"
    if n_unique == 2:
        return "binary"
    if n_unique == n_rows:
        return "unique"
    if n_unique <= 20:
        return "low-cardinality"
    return "high-cardinality"


def profile_dataset(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a column-level summary matching the dashboard schema.

    Required columns: column, dtype, n_missing, pct_missing, n_unique,
                      cardinality_type, sample_values, mean, std, min, max.
    """
    n_rows = len(df)
    rows = []
    for col in df.columns:
        s        = df[col]
        n_miss   = int(s.isna().sum())
        n_unique = int(s.nunique())
        sample   = str(s.dropna().head(3).tolist())
        is_num   = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "column":           col,
            "dtype":            str(s.dtype),
            "n_missing":        n_miss,
            "pct_missing":      round(100 * n_miss / n_rows, 2) if n_rows else 0.0,
            "n_unique":         n_unique,
            "cardinality_type": _cardinality_type(n_unique, n_rows),
            "sample_values":    sample,
            "mean":             round(float(s.mean()), 6) if is_num else None,
            "std":              round(float(s.std()),  6) if is_num else None,
            "min":              round(float(s.min()),  6) if is_num else None,
            "max":              round(float(s.max()),  6) if is_num else None,
        })
    summary = pd.DataFrame(rows)
    out_path = REPORT_DIR / f"{label}_column_summary.csv"
    summary.to_csv(out_path, index=False)
    log.info("Saved %s -> %s", label, out_path.name)
    return summary


def plot_target_distribution(target: pd.Series) -> None:
    """Plot histogram + KDE of target (realized volatility)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1A1A2E")
    for ax in axes:
        ax.set_facecolor("#1A1A2E")

    # Raw
    axes[0].hist(target, bins=60, color="#42A5F5", edgecolor="none", alpha=0.85)
    axes[0].set_title("Target: Realized Volatility (raw)", color="white", fontsize=12)
    axes[0].set_xlabel("Realized Volatility", color="white")
    axes[0].set_ylabel("Count", color="white")
    axes[0].tick_params(colors="white")

    # Log scale
    axes[1].hist(np.log1p(target), bins=60, color="#66BB6A", edgecolor="none", alpha=0.85)
    axes[1].set_title("Target: log1p(Realized Volatility)", color="white", fontsize=12)
    axes[1].set_xlabel("log1p(Realized Volatility)", color="white")
    axes[1].set_ylabel("Count", color="white")
    axes[1].tick_params(colors="white")

    plt.tight_layout()
    out = REPORT_DIR / "target_distribution.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()
    log.info("Saved target_distribution.png")


def plot_stock_volatility_profile(df: pd.DataFrame) -> None:
    """Bar chart: mean realized volatility per stock (top 30)."""
    top = (df.groupby("stock_id")["target"]
             .mean()
             .sort_values(ascending=False)
             .head(30))

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    bars = ax.bar(top.index.astype(int), top.values, color="#AB47BC", edgecolor="none")
    ax.set_title("Mean Realized Volatility — Top 30 Stocks", color="white", fontsize=13)
    ax.set_xlabel("Stock ID", color="white")
    ax.set_ylabel("Mean Realized Volatility", color="white")
    ax.tick_params(colors="white")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = REPORT_DIR / "stock_volatility_profile.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()
    log.info("Saved stock_volatility_profile.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation of book/trade features vs target (top 20)."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if "target" not in num_cols:
        return

    corr = (df[num_cols].corr()["target"]
              .drop("target")
              .abs()
              .sort_values(ascending=False)
              .head(20))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    colors = ["#EF5350" if c > 0 else "#42A5F5"
              for c in df[num_cols].corr()["target"].drop("target")
                                    .reindex(corr.index)]
    ax.barh(corr.index[::-1], corr.values[::-1],
            color=colors[::-1], edgecolor="none")
    ax.set_title("Feature-Target Correlation |r| (Top 20)", color="white", fontsize=13)
    ax.set_xlabel("|Pearson r| vs target", color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    out = REPORT_DIR / "correlation_top20.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()
    log.info("Saved correlation_top20.png")

    # Save CSV
    corr_df = df[num_cols].corr()[["target"]].drop("target").reset_index()
    corr_df.columns = ["feature", "pearson_r"]
    corr_df["abs_r"] = corr_df["pearson_r"].abs()
    corr_df = corr_df.sort_values("abs_r", ascending=False)
    corr_df.to_csv(REPORT_DIR / "feature_target_correlation.csv", index=False)
    log.info("Saved feature_target_correlation.csv")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 60)
    log.info("Use Case C_markets — Step 1: Data Loading")
    log.info("=" * 60)

    # ── 1. Check / generate data ───────────────────────────────────────────────
    if not (BOOK_TRAIN.exists() and TRAIN_CSV.exists()):
        generate_synthetic_data(n_stocks=50, n_times=600)
    else:
        log.info("Real data found — using book_train.parquet + train.csv")

    # ── 2. Load & aggregate ────────────────────────────────────────────────────
    if TRAIN_RAW_PQ.exists():
        log.info("Loading cached train_raw.parquet")
        df = pd.read_parquet(TRAIN_RAW_PQ)
    else:
        log.info("Loading book data …")
        book  = pd.read_parquet(BOOK_TRAIN)
        log.info("  book shape: %s", book.shape)

        book_agg = aggregate_book(book)
        del book

        trade_agg = None
        if TRADE_TRAIN.exists():
            log.info("Loading trade data …")
            trade = pd.read_parquet(TRADE_TRAIN)
            log.info("  trade shape: %s", trade.shape)
            trade_agg = aggregate_trade(trade)
            del trade

        # load targets
        log.info("Loading targets …")
        targets = pd.read_csv(TRAIN_CSV)
        log.info("  targets shape: %s", targets.shape)

        # merge book + trade + targets
        df = targets.merge(book_agg, on=["stock_id", "time_id"], how="left")
        if trade_agg is not None:
            df = df.merge(trade_agg, on=["stock_id", "time_id"], how="left")

        df.to_parquet(TRAIN_RAW_PQ, index=False)
        log.info("Saved train_raw.parquet  (%d rows x %d cols)", *df.shape)

    log.info("Merged dataset: %s", df.shape)
    log.info("Target stats: mean=%.5f  std=%.5f  min=%.6f  max=%.5f",
             df["target"].mean(), df["target"].std(),
             df["target"].min(),  df["target"].max())

    # ── 3. Profile ─────────────────────────────────────────────────────────────
    profile_dataset(df, "train")

    # ── 4. Plots ───────────────────────────────────────────────────────────────
    plot_target_distribution(df["target"])
    plot_stock_volatility_profile(df)
    plot_correlation_heatmap(df)

    # ── 5. Feature-target correlation CSV (for dashboard) ─────────────────────
    # Already saved inside plot_correlation_heatmap; also save missing CSV
    missing_df = pd.DataFrame({
        "column":      df.columns.tolist(),
        "n_missing":   df.isna().sum().values,
        "pct_missing": (100 * df.isna().mean()).round(2).values,
    })
    missing_df.to_csv(REPORT_DIR / "missing_values.csv", index=False)
    log.info("Saved missing_values.csv")

    # ── 6. Train / Val / Test split ────────────────────────────────────────────
    # Split by time_id to avoid leakage (future time windows go to val/test)
    time_ids = sorted(df["time_id"].unique())
    n = len(time_ids)
    n_test  = max(1, int(n * TEST_SIZE))
    n_val   = max(1, int(n * VAL_SIZE))
    n_train = n - n_val - n_test

    train_tids = time_ids[:n_train]
    val_tids   = time_ids[n_train:n_train + n_val]
    test_tids  = time_ids[n_train + n_val:]

    df_train = df[df["time_id"].isin(train_tids)].copy()
    df_val   = df[df["time_id"].isin(val_tids)].copy()
    df_test  = df[df["time_id"].isin(test_tids)].copy()

    df_train.to_parquet(TRAIN_PQ, index=False)
    df_val.to_parquet(VAL_PQ,   index=False)
    df_test.to_parquet(TEST_PQ,  index=False)

    log.info("Split — train: %d  val: %d  test: %d rows",
             len(df_train), len(df_val), len(df_test))
    log.info("Time-based split: train time_ids 0..%d, val %d..%d, test %d..%d",
             n_train - 1, n_train, n_train + n_val - 1,
             n_train + n_val, n - 1)

    # ── 7. Summary ─────────────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("Step 1 complete. Outputs:")
    for p in [TRAIN_RAW_PQ, TRAIN_PQ, VAL_PQ, TEST_PQ]:
        log.info("  %s", p.name)
    for p in REPORT_DIR.glob("*.{csv,png}"):
        log.info("  reports/%s", p.name)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
