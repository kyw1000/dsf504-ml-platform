"""
use_case_G2_xai/01_data_loading.py
=====================================
Use Case G2 — Explainable AI for Analysts & Managers
Phase 1, Step 1: Data Loading & Initial Inspection

Dataset sources
---------------
  PRIMARY  : SEC EDGAR Company Facts (Kaggle: chad116/sec-company-facts-all-10q-10k-financial-data)
             Structured financial statement data from 10-K/10-Q filings
  SECONDARY: Yahoo Finance price data (yfinance library or Kaggle S&P 500 dataset)
  SYNTHETIC: Full synthetic generator if neither source is available

Target variable
---------------
  Binary: does the stock outperform the S&P 500 index in the next 12 months?
  (Top 40% of forward_return → 1, Bottom 40% → 0, middle 20% dropped for clarity)

Financial Ratio Features (from 10-K/10-Q)
------------------------------------------
  Valuation : P/E, P/B, EV/EBITDA, Price/Sales
  Profitability: ROE, ROA, Net Margin, Gross Margin, EBITDA Margin
  Leverage  : Debt/Equity, Interest Coverage, Debt/Assets
  Liquidity : Current Ratio, Quick Ratio, Cash Ratio
  Efficiency: Asset Turnover, Inventory Turnover, Receivables Turnover
  Growth    : Revenue YoY, EPS YoY, Asset YoY, FCF YoY

Run
---
    cd C:\\DSF504
    python use_case_G2_xai/01_data_loading.py
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

DATA_SUBDIR = DATA_DIR / "sec_edgar"
REPORT_DIR  = REPORTS_DIR / "use_case_G2"
DATA_SUBDIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["Technology", "Healthcare", "Financials", "Energy", "Consumer Staples",
           "Consumer Discretionary", "Industrials", "Utilities", "Real Estate",
           "Materials", "Communication Services"]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic generator  (matches SEC EDGAR + Yahoo Finance schema)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic(
    n_companies: int = 800,
    n_years:     int = 5,
    seed:        int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate synthetic panel data: one row per (ticker, fiscal_year).
    Financial ratios are drawn from realistic distributions calibrated
    to S&P 500 historical ranges (2018–2022).
    """
    rng = np.random.default_rng(seed)
    tickers = [f"TICK{i:04d}" for i in range(n_companies)]
    years   = list(range(2018, 2018 + n_years))

    rows = []
    # Company-level fixed effects (stable characteristics)
    sector_assignments = rng.choice(SECTORS, n_companies)
    size_factor = rng.lognormal(0, 1, n_companies)   # market cap proxy
    quality_factor = rng.normal(0, 1, n_companies)   # latent quality

    for yi, year in enumerate(years):
        for ci, ticker in enumerate(tickers):
            sector = sector_assignments[ci]
            sf     = size_factor[ci]
            qf     = quality_factor[ci]

            # Ratios with sector-appropriate calibration
            pe_base = {"Technology": 28, "Healthcare": 22, "Financials": 12,
                       "Energy": 14, "Consumer Staples": 20}.get(sector, 18)
            pe = max(0, rng.normal(pe_base + qf * 3, 8))
            pb = max(0.1, rng.normal(2.5 + qf * 0.5, 1.5))
            ps = max(0.1, rng.normal(2.0, 1.0))

            roe = rng.normal(0.12 + qf * 0.05, 0.08)
            roa = rng.normal(0.06 + qf * 0.03, 0.04)
            net_margin    = rng.normal(0.08 + qf * 0.04, 0.06)
            gross_margin  = max(0, rng.normal(0.35, 0.15))
            ebitda_margin = max(0, rng.normal(0.18, 0.10))

            de_ratio = max(0, rng.lognormal(-0.1, 0.8))
            interest_cov = max(0, rng.lognormal(2.0, 1.0))
            debt_assets = np.clip(rng.normal(0.35, 0.15), 0, 0.9)

            current_ratio = max(0.5, rng.normal(1.8, 0.6))
            quick_ratio   = max(0.2, current_ratio - rng.uniform(0.2, 0.8))

            asset_turnover = max(0.1, rng.normal(0.8, 0.4))
            rev_growth     = rng.normal(0.07 + qf * 0.03, 0.15)
            eps_growth     = rng.normal(0.08 + qf * 0.04, 0.20)
            fcf_yield      = rng.normal(0.04, 0.03)

            # Forward 12-month return (driven by quality + value + noise)
            forward_return = (
                qf * 0.08                      # quality premium
                - (pe / pe_base - 1) * 0.05   # valuation penalty
                + rev_growth * 0.15            # growth premium
                + rng.normal(0, 0.20)          # market noise
            )
            # Macro year effect
            year_effect = {2018: -0.05, 2019: 0.25, 2020: -0.10,
                           2021: 0.25, 2022: -0.18}.get(year, 0)
            forward_return += year_effect + rng.normal(0, 0.05)

            rows.append({
                "ticker":           ticker,
                "fiscal_year":      year,
                "sector":           sector,
                "market_cap_log":   np.log(sf * 1e9),
                # Valuation
                "pe_ratio":         round(pe, 2),
                "pb_ratio":         round(pb, 2),
                "ps_ratio":         round(ps, 2),
                # Profitability
                "roe":              round(roe, 4),
                "roa":              round(roa, 4),
                "net_margin":       round(net_margin, 4),
                "gross_margin":     round(gross_margin, 4),
                "ebitda_margin":    round(ebitda_margin, 4),
                # Leverage
                "debt_equity":      round(de_ratio, 3),
                "interest_coverage":round(interest_cov, 2),
                "debt_assets":      round(debt_assets, 3),
                # Liquidity
                "current_ratio":    round(current_ratio, 2),
                "quick_ratio":      round(quick_ratio, 2),
                # Efficiency
                "asset_turnover":   round(asset_turnover, 3),
                # Growth
                "revenue_growth":   round(rev_growth, 4),
                "eps_growth":       round(eps_growth, 4),
                "fcf_yield":        round(fcf_yield, 4),
                # Target
                "forward_return_12m": round(forward_return, 4),
            })

    df = pd.DataFrame(rows)

    # Binary target: outperform S&P500 proxy (top 40% = 1, bottom 40% = 0)
    thresholds = df.groupby("fiscal_year")["forward_return_12m"].quantile([0.40, 0.60])
    labels = []
    for _, row in df.iterrows():
        lo = thresholds.loc[row["fiscal_year"], 0.40]
        hi = thresholds.loc[row["fiscal_year"], 0.60]
        if row["forward_return_12m"] >= hi:
            labels.append(1)
        elif row["forward_return_12m"] <= lo:
            labels.append(0)
        else:
            labels.append(-1)  # middle band → dropped
    df["outperform"] = labels
    df = df[df["outperform"] != -1].reset_index(drop=True)

    log.info(f"Synthetic data: {len(df):,} company-year observations, "
             f"{n_companies} tickers, {n_years} years")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Real data loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_real() -> pd.DataFrame | None:
    for fname in ["company_facts.parquet", "company_facts.csv", "sec_edgar_ratios.csv"]:
        p = DATA_SUBDIR / fname
        if p.exists():
            log.info(f"Loading real data from {fname}…")
            df = pd.read_parquet(p) if fname.endswith(".parquet") else pd.read_csv(p)
            return df
    return None


def _print_download_instructions() -> None:
    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │  SEC EDGAR Real Data — Download Instructions                 │
  │                                                              │
  │  Option 1 (Kaggle SEC Company Facts):                        │
  │    kaggle datasets download chad116/sec-company-facts-all-   │
  │    10q-10k-financial-data                                    │
  │    Place in: data/sec_edgar/                                 │
  │                                                              │
  │  Option 2 (SEC EDGAR API):                                   │
  │    https://data.sec.gov/api/xbrl/companyfacts/               │
  │    Use the secedgar or edgar Python packages                 │
  │                                                              │
  │  Option 3 (Yahoo Finance):                                   │
  │    pip install yfinance                                      │
  │    Fetch financials for S&P 500 tickers                      │
  └──────────────────────────────────────────────────────────────┘
""")


# ─────────────────────────────────────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────────────────────────────────────

def _temporal_split(df: pd.DataFrame, val_year: int = 2022) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train: years < val_year | Val: val_year. Prevents look-ahead bias."""
    train = df[df["fiscal_year"] < val_year].reset_index(drop=True)
    val   = df[df["fiscal_year"] == val_year].reset_index(drop=True)
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def _plot_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SEC EDGAR + Yahoo Finance Dataset — Overview", fontsize=14, fontweight="bold")

    # Target distribution
    ax = axes[0, 0]
    vc = df["outperform"].value_counts()
    ax.bar(["Underperform (0)", "Outperform (1)"], [vc.get(0, 0), vc.get(1, 0)],
           color=["#F44336", "#4CAF50"])
    ax.set_title("Target Distribution (Outperform S&P 500)")
    ax.set_ylabel("Count")
    for i, v in enumerate([vc.get(0, 0), vc.get(1, 0)]):
        ax.text(i, v + 5, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center")

    # Observations per year
    ax = axes[0, 1]
    yr_counts = df.groupby("fiscal_year").size()
    ax.bar(yr_counts.index.astype(str), yr_counts.values, color="#1976D2")
    ax.set_title("Observations per Fiscal Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")

    # Sector distribution
    ax = axes[1, 0]
    sec_counts = df["sector"].value_counts()
    ax.barh(sec_counts.index, sec_counts.values, color=plt.cm.Set3.colors[:len(sec_counts)])
    ax.set_title("Companies by Sector")
    ax.set_xlabel("Observations")

    # Forward return distribution
    ax = axes[1, 1]
    ax.hist(df["forward_return_12m"], bins=50, color="#7B1FA2", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="0% return")
    ax.axvline(df["forward_return_12m"].median(), color="orange", linestyle="--",
               label=f"Median={df['forward_return_12m'].median():.2%}")
    ax.set_title("Forward 12M Return Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "target_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved target_distribution.png")


def _plot_ratio_overview(df: pd.DataFrame) -> None:
    ratio_cols = ["pe_ratio", "pb_ratio", "roe", "net_margin",
                  "debt_equity", "current_ratio", "revenue_growth", "eps_growth"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Financial Ratio Distributions by Target Class", fontsize=13, fontweight="bold")

    for ax, col in zip(axes.flat, ratio_cols):
        for label, color in [(0, "#F44336"), (1, "#4CAF50")]:
            vals = df[df["outperform"] == label][col].dropna()
            # Clip extreme values for display
            p1, p99 = vals.quantile(0.01), vals.quantile(0.99)
            vals = vals.clip(p1, p99)
            ax.hist(vals, bins=30, alpha=0.5, color=color,
                    label="Under" if label == 0 else "Over", density=True)
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("Density")

    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "ratio_distributions.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved ratio_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G2: Explainable AI for Analysts & Managers")
    print("  Step 1: Data Loading & Initial Inspection")
    print("=" * 65 + "\n")

    df = _load_real()
    if df is None:
        _print_download_instructions()
        log.info("Generating synthetic SEC EDGAR / Yahoo Finance data…")
        df = _generate_synthetic(n_companies=800, n_years=5)

    print(f"[1] Dataset: {len(df):,} company-year observations")
    print(f"    Tickers  : {df['ticker'].nunique():,}")
    print(f"    Years    : {df['fiscal_year'].min()} – {df['fiscal_year'].max()}")
    print(f"    Features : {df.shape[1] - 3} (excl. ticker/year/target)")
    print(f"    Positive rate: {df['outperform'].mean():.3f}")

    print("\n[2] Class balance by year:")
    yr_balance = df.groupby("fiscal_year")["outperform"].agg(["mean", "count"])
    yr_balance.columns = ["positive_rate", "n"]
    print(yr_balance.to_string())

    print("\n[3] Sector breakdown:")
    sec_stats = df.groupby("sector")["outperform"].agg(["mean", "count"]).sort_values("count", ascending=False)
    print(sec_stats.to_string())

    print("\n[4] Financial ratio summary (medians):")
    ratio_cols = ["pe_ratio", "pb_ratio", "roe", "net_margin", "debt_equity",
                  "current_ratio", "revenue_growth", "eps_growth"]
    print(df[ratio_cols].median().round(4).to_string())

    print("\n[5] Temporal train/val split (train: <2022, val: 2022)…")
    train, val = _temporal_split(df, val_year=df["fiscal_year"].max())
    print(f"    Train: {len(train):,} obs  ({train['fiscal_year'].min()}–{train['fiscal_year'].max()})")
    print(f"    Val  : {len(val):,} obs   (year {val['fiscal_year'].max()})")

    print("\n[6] Saving parquet files…")
    df.to_parquet(DATA_SUBDIR / "company_ratios.parquet", index=False)
    train.to_parquet(DATA_SUBDIR / "train_ratios.parquet", index=False)
    val.to_parquet(DATA_SUBDIR / "val_ratios.parquet",   index=False)
    log.info("Saved company_ratios.parquet, train_ratios.parquet, val_ratios.parquet")

    # Column summary
    summary_rows = []
    for col in df.columns:
        summary_rows.append({
            "column": col, "dtype": str(df[col].dtype),
            "nunique": df[col].nunique(),
            "null_pct": f"{df[col].isna().mean()*100:.2f}%",
            "median": df[col].median() if df[col].dtype != object else "",
        })
    pd.DataFrame(summary_rows).to_csv(REPORT_DIR / "train_column_summary.csv", index=False)

    print("\n[7] Generating visualisations…")
    _plot_overview(df)
    _plot_ratio_overview(df)

    print(f"\n  All outputs → {DATA_SUBDIR}")
    print("=" * 65)
    print("  Step 1 complete. Ready for EDA (02_eda_analysis.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
