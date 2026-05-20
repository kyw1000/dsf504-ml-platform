"""
use_case_C_market/02_eda_analysis.py
=====================================
DSF504 Use Case C_markets — Market Intelligence: Realized Volatility Prediction
ML Framework Phase 2: Exploratory Data Analysis

Sections:
  1. Target distribution (raw + log1p)
  2. Volatility clustering by stock
  3. Feature correlation with target
  4. Order-book feature distributions
  5. Trade feature distributions
  6. Missing value analysis
  7. Outlier detection (IQR method)

Run:
    cd C:/DSF504
    python use_case_C_market/02_eda_analysis.py
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
from scipy import stats

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

TRAIN_RAW_PQ = DATA_SUBDIR / "train_raw.parquet"

# ── style constants ────────────────────────────────────────────────────────────
BG    = "#1A1A2E"
FG    = "white"
BLUE  = "#42A5F5"
GREEN = "#66BB6A"
ORG   = "#FFA726"
RED   = "#EF5350"
PURP  = "#AB47BC"
TEAL  = "#26C6DA"

PALETTE = [BLUE, GREEN, ORG, RED, PURP, TEAL,
           "#EC407A","#D4E157","#FF7043","#78909C"]


def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333366")


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Target distribution
# ══════════════════════════════════════════════════════════════════════════════

def analyze_target(df: pd.DataFrame) -> None:
    log.info("[1/7] Target distribution analysis")

    target = df["target"].dropna()
    log_target = np.log1p(target)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Target: 10-Min Realized Volatility", color=FG, fontsize=14, y=1.01)

    # Raw histogram
    axes[0, 0].hist(target, bins=80, color=BLUE, edgecolor="none", alpha=0.85)
    axes[0, 0].set_title("Raw Distribution", color=FG)
    axes[0, 0].set_xlabel("Realized Volatility", color=FG)
    style_ax(axes[0, 0])

    # Log1p histogram
    axes[0, 1].hist(log_target, bins=80, color=GREEN, edgecolor="none", alpha=0.85)
    axes[0, 1].set_title("log1p(Realized Volatility)", color=FG)
    axes[0, 1].set_xlabel("log1p(target)", color=FG)
    style_ax(axes[0, 1])

    # Empirical CDF
    sorted_vals = np.sort(target)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    axes[1, 0].plot(sorted_vals, cdf, color=ORG, linewidth=1.5)
    axes[1, 0].set_title("Empirical CDF", color=FG)
    axes[1, 0].set_xlabel("Realized Volatility", color=FG)
    axes[1, 0].set_ylabel("Cumulative Probability", color=FG)
    style_ax(axes[1, 0])

    # Q-Q plot vs log-normal
    qq_data = np.sort(log_target)
    qq_theor = stats.norm.ppf(np.linspace(0.01, 0.99, len(qq_data)),
                              loc=qq_data.mean(), scale=qq_data.std())
    axes[1, 1].scatter(qq_theor, qq_data, s=3, alpha=0.4, color=TEAL)
    axes[1, 1].plot([qq_theor.min(), qq_theor.max()],
                    [qq_theor.min(), qq_theor.max()],
                    color=RED, linewidth=1.5, linestyle="--")
    axes[1, 1].set_title("Q-Q: log1p(target) vs Normal", color=FG)
    axes[1, 1].set_xlabel("Theoretical Normal Quantiles", color=FG)
    axes[1, 1].set_ylabel("Sample Quantiles", color=FG)
    style_ax(axes[1, 1])

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_target_distribution.png",
                dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info("  skewness=%.3f  kurtosis=%.3f  mean=%.5f  p99=%.5f",
             float(stats.skew(target)), float(stats.kurtosis(target)),
             float(target.mean()), float(target.quantile(0.99)))


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Volatility clustering by stock
# ══════════════════════════════════════════════════════════════════════════════

def analyze_stock_clustering(df: pd.DataFrame) -> None:
    log.info("[2/7] Volatility clustering by stock")

    stock_stats = (df.groupby("stock_id")["target"]
                     .agg(["mean", "std", "min", "max"])
                     .sort_values("mean", ascending=False))

    top_n = min(40, len(stock_stats))
    top   = stock_stats.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor(BG)

    # Mean realized vol per stock
    axes[0].barh(range(top_n), top["mean"].values[::-1],
                 color=PURP, edgecolor="none")
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top.index.astype(int).values[::-1], fontsize=7)
    axes[0].set_title(f"Mean Realized Volatility — Top {top_n} Stocks", color=FG)
    axes[0].set_xlabel("Mean Realized Volatility", color=FG)
    style_ax(axes[0])

    # Scatter: mean vs std (vol-of-vol)
    axes[1].scatter(stock_stats["mean"], stock_stats["std"],
                    s=25, alpha=0.6, color=TEAL, edgecolors="none")
    axes[1].set_title("Volatility of Volatility per Stock\nmean(RV) vs std(RV)", color=FG)
    axes[1].set_xlabel("Mean Realized Volatility", color=FG)
    axes[1].set_ylabel("Std Realized Volatility", color=FG)
    style_ax(axes[1])

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_stock_volatility_clustering.png",
                dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()

    stock_stats.reset_index().to_csv(REPORT_DIR / "stock_volatility_stats.csv",
                                     index=False)
    log.info("  %d unique stocks analyzed", len(stock_stats))


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Feature correlation with target
# ══════════════════════════════════════════════════════════════════════════════

def analyze_correlations(df: pd.DataFrame) -> None:
    log.info("[3/7] Feature-target correlations")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if "target" not in num_cols or len(num_cols) < 3:
        log.warning("  Insufficient numeric columns — skipping correlation analysis")
        return

    feat_cols = [c for c in num_cols if c not in ("target", "stock_id", "time_id")]
    corr = df[feat_cols + ["target"]].corr()["target"].drop("target")
    corr_abs = corr.abs().sort_values(ascending=False)

    top_n = min(20, len(corr_abs))
    top   = corr_abs.head(top_n)

    colors = [RED if corr[f] < 0 else BLUE for f in top.index]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)
    ax.barh(range(top_n), top.values[::-1], color=colors[::-1], edgecolor="none")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top.index[::-1], fontsize=9)
    ax.set_title("Feature-Target |Pearson r| — Top 20", color=FG, fontsize=13)
    ax.set_xlabel("|Pearson r|", color=FG)
    style_ax(ax)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Positive corr"),
                        Patch(color=RED,  label="Negative corr")],
              facecolor=BG, labelcolor=FG, loc="lower right")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_feature_correlation.png",
                dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()

    # Save CSV
    corr_df = corr.reset_index()
    corr_df.columns = ["feature", "pearson_r"]
    corr_df["abs_r"] = corr_df["pearson_r"].abs()
    corr_df = corr_df.sort_values("abs_r", ascending=False)
    corr_df.to_csv(REPORT_DIR / "feature_target_correlation.csv", index=False)
    log.info("  Top feature: %s (|r|=%.4f)", corr_df.iloc[0]["feature"],
             corr_df.iloc[0]["abs_r"])


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Order-book feature distributions
# ══════════════════════════════════════════════════════════════════════════════

def analyze_book_features(df: pd.DataFrame) -> None:
    log.info("[4/7] Order-book feature distributions")

    book_cols = [c for c in df.columns if c.startswith("book_") and
                 pd.api.types.is_numeric_dtype(df[c])]
    if not book_cols:
        log.info("  No book_ columns found — skipping")
        return

    n_cols = min(9, len(book_cols))
    cols   = book_cols[:n_cols]
    ncols_grid = 3
    nrows_grid = (n_cols + ncols_grid - 1) // ncols_grid

    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                             figsize=(14, 4 * nrows_grid))
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten() if nrows_grid > 1 else [axes] * ncols_grid

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        vals = df[col].dropna()
        # clip extreme outliers for display
        lo, hi = vals.quantile(0.01), vals.quantile(0.99)
        vals = vals.clip(lo, hi)
        ax.hist(vals, bins=50, color=PALETTE[i % len(PALETTE)],
                edgecolor="none", alpha=0.85)
        ax.set_title(col.replace("book_", "bk_"), color=FG, fontsize=9)
        style_ax(ax)

    # hide unused subplots
    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Order-Book Feature Distributions", color=FG, fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_book_features.png",
                dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info("  Plotted %d book features", n_cols)


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Trade feature distributions
# ══════════════════════════════════════════════════════════════════════════════

def analyze_trade_features(df: pd.DataFrame) -> None:
    log.info("[5/7] Trade feature distributions")

    trade_cols = [c for c in df.columns if c.startswith("trade_") and
                  pd.api.types.is_numeric_dtype(df[c])]
    if not trade_cols:
        log.info("  No trade_ columns found — skipping")
        return

    n_cols = min(6, len(trade_cols))
    cols   = trade_cols[:n_cols]
    ncols_grid = 3
    nrows_grid = (n_cols + ncols_grid - 1) // ncols_grid

    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                             figsize=(14, 4 * nrows_grid))
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten() if nrows_grid > 1 else axes.tolist()

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        vals = df[col].dropna()
        lo, hi = vals.quantile(0.01), vals.quantile(0.99)
        vals = vals.clip(lo, hi)
        ax.hist(vals, bins=50, color=PALETTE[(i + 3) % len(PALETTE)],
                edgecolor="none", alpha=0.85)
        ax.set_title(col.replace("trade_", "tr_"), color=FG, fontsize=9)
        style_ax(ax)

    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Trade Feature Distributions", color=FG, fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_trade_features.png",
                dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info("  Plotted %d trade features", n_cols)


# ══════════════════════════════════════════════════════════════════════════════
# Section 6 — Missing value analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_missing(df: pd.DataFrame) -> None:
    log.info("[6/7] Missing value analysis")

    missing = (df.isna().sum()
                 .sort_values(ascending=False))
    missing_pct = (100 * df.isna().mean()
                          .sort_values(ascending=False))

    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) == 0:
        log.info("  No missing values found")
        # save empty CSV so dashboard doesn't error
        pd.DataFrame({"column": [], "n_missing": [], "pct_missing": []}).to_csv(
            REPORT_DIR / "missing_values.csv", index=False)
        return

    top_n = min(20, len(cols_with_missing))
    top   = cols_with_missing.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    fig.patch.set_facecolor(BG)
    ax.barh(top.index[::-1], (100 * top / len(df))[::-1],
            color=ORG, edgecolor="none")
    ax.set_title("Missing Value Rate — Top Columns", color=FG, fontsize=13)
    ax.set_xlabel("% Missing", color=FG)
    style_ax(ax)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_missing_values.png",
                dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()

    # Save CSV
    mv_df = pd.DataFrame({
        "column":   missing.index,
        "n_missing":   missing.values,
        "pct_missing": missing_pct.values,
    })
    mv_df.to_csv(REPORT_DIR / "missing_values.csv", index=False)
    log.info("  %d columns have missing values (max pct=%.1f%%)",
             len(cols_with_missing), missing_pct.max())


# ══════════════════════════════════════════════════════════════════════════════
# Section 7 — Outlier detection
# ══════════════════════════════════════════════════════════════════════════════

def analyze_outliers(df: pd.DataFrame) -> None:
    log.info("[7/7] Outlier detection (IQR method)")

    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in ("stock_id", "time_id")]

    records: list[dict] = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        records.append({
            "column":     col,
            "q1":         round(q1, 6),
            "q3":         round(q3, 6),
            "iqr":        round(iqr, 6),
            "n_outliers": n_out,
            "pct_outliers": round(100 * n_out / len(s), 2),
        })

    out_df = pd.DataFrame(records).sort_values("pct_outliers", ascending=False)
    out_df.to_csv(REPORT_DIR / "outlier_report.csv", index=False)

    top = out_df.head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.barh(top["column"].values[::-1], top["pct_outliers"].values[::-1],
            color=RED, edgecolor="none")
    ax.set_title("Outlier Rate by Feature (IQR, Top 15)", color=FG, fontsize=13)
    ax.set_xlabel("% Outliers", color=FG)
    style_ax(ax)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_outliers.png",
                dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info("  Max outlier rate: %s (%.1f%%)",
             out_df.iloc[0]["column"], out_df.iloc[0]["pct_outliers"])


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 60)
    log.info("Use Case C_markets — Step 2: EDA Analysis")
    log.info("=" * 60)

    if not TRAIN_RAW_PQ.exists():
        log.error("train_raw.parquet not found — run Step 1 first")
        sys.exit(1)

    log.info("Loading train_raw.parquet …")
    df = pd.read_parquet(TRAIN_RAW_PQ)
    log.info("  shape: %s", df.shape)

    analyze_target(df)
    analyze_stock_clustering(df)
    analyze_correlations(df)
    analyze_book_features(df)
    analyze_trade_features(df)
    analyze_missing(df)
    analyze_outliers(df)

    # ── Save column summary (schema must match dashboard expectations) ─────────
    def _card_type(n_unique, n_rows):
        if n_unique == 1:         return "constant"
        if n_unique == 2:         return "binary"
        if n_unique == n_rows:    return "unique"
        if n_unique <= 20:        return "low-cardinality"
        return "high-cardinality"

    n_rows = len(df)
    rows = []
    for col in df.columns:
        s        = df[col]
        n_miss   = int(s.isna().sum())
        n_unique = int(s.nunique())
        is_num   = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "column":           col,
            "dtype":            str(s.dtype),
            "n_missing":        n_miss,
            "pct_missing":      round(100 * n_miss / n_rows, 2) if n_rows else 0.0,
            "n_unique":         n_unique,
            "cardinality_type": _card_type(n_unique, n_rows),
            "sample_values":    str(s.dropna().head(3).tolist()),
            "mean":             round(float(s.mean()), 6) if is_num else None,
            "std":              round(float(s.std()),  6) if is_num else None,
            "min":              round(float(s.min()),  6) if is_num else None,
            "max":              round(float(s.max()),  6) if is_num else None,
        })
    pd.DataFrame(rows).to_csv(REPORT_DIR / "train_column_summary.csv", index=False)
    log.info("Saved train_column_summary.csv")

    # ── Supplemental standardised EDA plots ──────────────────────────────
    log.info("Generating supplemental standardised EDA visualizations …")
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.eda_viz import (
            plot_overview_panel, plot_target_distribution,
            plot_missing_heatmap, plot_correlation_heatmap,
            plot_numeric_distributions, plot_engineered_feature_summary,
        )
        plot_overview_panel(df, "target", REPORT_DIR, " — UC C_market Volatility")
        plot_target_distribution(df, "target", REPORT_DIR, " — UC C_market Volatility")
        plot_missing_heatmap(df, REPORT_DIR, " — UC C_market Volatility")
        plot_correlation_heatmap(df, REPORT_DIR, " — UC C_market Volatility",
                                 top_n=25, target_col="target")
        plot_numeric_distributions(df, REPORT_DIR, " — UC C_market Volatility",
                                   target_col=None)
        log.info("  Saved: overview, target_distribution, missing_heatmap, correlation_heatmap, numeric_distributions")
    except Exception as _e:
        log.warning("Supplemental plots skipped: %s", _e)

    log.info("-" * 60)
    log.info("Step 2 complete. Reports saved to %s", REPORT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
