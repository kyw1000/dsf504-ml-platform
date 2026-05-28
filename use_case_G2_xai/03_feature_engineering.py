"""
use_case_G2_xai/03_feature_engineering.py
==========================================
Use Case G2 — Explainable AI for Analysts & Managers
Phase 2, Step 3: Feature Engineering

Feature Groups
--------------
  RAW     : 17 financial ratios from 10-K/10-Q (as-is after clipping)
  SECTOR  : sector one-hot encoding (11 GICS sectors)
  DERIVED : ratio combinations (e.g. PEG = PE/earnings_growth, net_debt/EBITDA)
  RANK    : cross-sectional percentile rank of each ratio within the fiscal year
  COMPOSITE: value score, quality score, growth score (like factor investing)
  REGIME  : fiscal year macro indicator (crisis vs bull vs bear)

All train statistics are stored in fe_stats.pkl and applied to val.
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

DATA_SUBDIR = DATA_DIR / "sec_edgar"
REPORT_DIR  = REPORTS_DIR / "use_case_G2"
MODEL_DIR   = MODELS_DIR / "use_case_G2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RATIO_COLS = [
    "pe_ratio", "pb_ratio", "ps_ratio",
    "roe", "roa", "net_margin", "gross_margin", "ebitda_margin",
    "debt_equity", "interest_coverage", "debt_assets",
    "current_ratio", "quick_ratio",
    "asset_turnover", "revenue_growth", "eps_growth", "fcf_yield",
    "market_cap_log",
]

SECTORS = ["Technology", "Healthcare", "Financials", "Energy", "Consumer Staples",
           "Consumer Discretionary", "Industrials", "Utilities", "Real Estate",
           "Materials", "Communication Services"]

MACRO_REGIMES = {2018: 0, 2019: 1, 2020: -1, 2021: 1, 2022: -1}  # -1=crisis, 0=neutral, 1=bull


def _load():
    train = pd.read_parquet(DATA_SUBDIR / "train_ratios.parquet")
    val   = pd.read_parquet(DATA_SUBDIR / "val_ratios.parquet")
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1: Clip + impute raw ratios
# ─────────────────────────────────────────────────────────────────────────────

def clip_and_impute(df: pd.DataFrame, train_stats: dict | None = None,
                    clip_pct: float = 0.01) -> tuple[pd.DataFrame, dict]:
    is_train = train_stats is None
    if is_train:
        train_stats = {}

    df = df.copy()
    for col in RATIO_COLS:
        if col not in df.columns:
            continue
        if is_train:
            lo = df[col].quantile(clip_pct)
            hi = df[col].quantile(1 - clip_pct)
            med = df[col].median()
            train_stats[f"clip_{col}"] = (lo, hi, med)
        lo, hi, med = train_stats.get(f"clip_{col}", (-np.inf, np.inf, 0))
        df[col] = df[col].clip(lo, hi).fillna(med)

    log.info(f"Clipped and imputed {len(RATIO_COLS)} ratio features.")
    return df, train_stats


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2: Sector encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_sector(df: pd.DataFrame, train_stats: dict) -> pd.DataFrame:
    is_train = "sector_map" not in train_stats
    if is_train:
        sectors = df["sector"].fillna("Unknown").unique().tolist()
        train_stats["sector_map"] = {s: i for i, s in enumerate(sectors)}

    df = df.copy()
    df["sector_enc"] = df["sector"].map(train_stats["sector_map"]).fillna(-1)
    log.info("Sector encoded.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 3: Derived ratio features
# ─────────────────────────────────────────────────────────────────────────────

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # PEG ratio (P/E ÷ EPS growth) — classic value/growth bridge
    df["peg_ratio"] = df["pe_ratio"] / (df["eps_growth"].replace(0, np.nan).abs() * 100 + 1)
    df["peg_ratio"] = df["peg_ratio"].clip(-50, 50).fillna(0)

    # Net debt coverage proxy
    df["interest_burden"] = 1 / (df["interest_coverage"].replace(0, np.nan) + 1)
    df["interest_burden"] = df["interest_burden"].fillna(1)

    # Quality spread: high ROE + low leverage = quality
    df["quality_spread"] = df["roe"] - df["debt_equity"] * 0.1

    # Value composite (lower PE + PB = more value)
    df["value_composite"] = -(df["pe_ratio"] / df["pe_ratio"].max().clip(1)
                               + df["pb_ratio"] / df["pb_ratio"].max().clip(1)) / 2

    # Growth composite
    df["growth_composite"] = (df["revenue_growth"] + df["eps_growth"]) / 2

    # Profitability composite
    df["profitability_composite"] = (df["roe"] + df["net_margin"] + df["ebitda_margin"]) / 3

    # Leverage risk score
    df["leverage_risk"] = df["debt_equity"] + df["debt_assets"] - df["interest_coverage"] * 0.01

    log.info("Derived 7 composite features.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 4: Cross-sectional percentile ranks (within fiscal year)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rank_features(df: pd.DataFrame, train_stats: dict) -> pd.DataFrame:
    """
    Rank each company's ratio within its fiscal year cohort.
    This makes features scale-invariant across years and removes
    macro-level trends (e.g., all P/Es rise in a bull market).
    Ranks computed on TRAIN years only; val companies ranked relative
    to train-year medians to prevent leakage.
    """
    is_train = "rank_medians" not in train_stats
    df = df.copy()
    rank_cols = [c for c in RATIO_COLS if c in df.columns] + [
        "peg_ratio", "quality_spread", "growth_composite", "profitability_composite"
    ]
    rank_cols = [c for c in rank_cols if c in df.columns]

    if is_train:
        # Store year-level medians/stds for val normalisation
        yr_stats = df.groupby("fiscal_year")[rank_cols].agg(["median", "std"])
        train_stats["rank_medians"] = yr_stats

    # Compute within-year rank (for train, genuine; for val, approximate)
    for col in rank_cols:
        df[f"{col}__rank"] = (df.groupby("fiscal_year")[col]
                               .rank(pct=True, na_option="keep")
                               .fillna(0.5))

    log.info(f"Rank features: {len(rank_cols)} ratios ranked within fiscal year.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 5: Macro regime
# ─────────────────────────────────────────────────────────────────────────────

def add_macro_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["macro_regime"] = df["fiscal_year"].map(MACRO_REGIMES).fillna(0)
    df["is_crisis_year"] = (df["macro_regime"] == -1).astype(int)
    df["is_bull_year"]   = (df["macro_regime"] == 1).astype(int)
    log.info("Macro regime features added.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Assemble + save
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame,
                             train_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    is_train = train_stats is None
    if is_train:
        train_stats = {}

    df, train_stats = clip_and_impute(df, train_stats)
    df = encode_sector(df, train_stats)
    df = derive_features(df)
    df = compute_rank_features(df, train_stats)
    df = add_macro_regime(df)
    return df, train_stats


def plot_feature_summary(train_fe: pd.DataFrame) -> None:
    feat_cols = [c for c in train_fe.columns
                 if c not in ["ticker", "fiscal_year", "sector", "forward_return_12m",
                               "outperform"]]
    groups = {"Raw ratios": sum(1 for c in feat_cols if not c.endswith("__rank")
                                and c not in ["sector_enc", "macro_regime",
                                               "is_crisis_year", "is_bull_year"]),
              "Rank features": sum(1 for c in feat_cols if c.endswith("__rank")),
              "Derived/Composite": sum(1 for c in feat_cols if c in [
                  "peg_ratio", "interest_burden", "quality_spread", "value_composite",
                  "growth_composite", "profitability_composite", "leverage_risk"]),
              "Macro/Sector": sum(1 for c in feat_cols if c in [
                  "sector_enc", "macro_regime", "is_crisis_year", "is_bull_year"])}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Feature Engineering Summary — G2 XAI", fontsize=12, fontweight="bold")

    axes[0].bar(list(groups.keys()), list(groups.values()),
                color=["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"])
    axes[0].set_title(f"Features by Group (Total: {sum(groups.values())})")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(groups.values()):
        axes[0].text(i, v + 0.2, str(v), ha="center")

    # Correlation heatmap of top composite features vs target
    composite_cols = ["value_composite", "growth_composite", "profitability_composite",
                      "leverage_risk", "quality_spread", "peg_ratio"]
    composite_cols = [c for c in composite_cols if c in train_fe.columns]
    if composite_cols:
        corr = train_fe[composite_cols + ["outperform"]].corr()["outperform"][:-1]
        corr.plot(kind="barh", ax=axes[1], color=["#4CAF50" if v > 0 else "#F44336"
                                                    for v in corr.values])
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_title("Composite Features vs Target Correlation")
        axes[1].set_xlabel("Correlation")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "engineered_feature_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved engineered_feature_summary.png")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G2: Explainable AI for Analysts & Managers")
    print("  Step 3: Feature Engineering")
    print("=" * 65 + "\n")

    train, val = _load()
    print(f"[1] Loaded: {len(train):,} train | {len(val):,} val observations")

    print("[2] Running feature engineering on train set…")
    train_fe, train_stats = run_feature_engineering(train)

    print("[3] Applying to val (using train stats — no leakage)…")
    val_fe, _ = run_feature_engineering(val, train_stats)

    feat_cols = [c for c in train_fe.columns
                 if c not in ["ticker", "fiscal_year", "sector", "forward_return_12m", "outperform"]]
    print(f"\n  Train FE shape: {train_fe.shape}  |  Val FE shape: {val_fe.shape}")
    print(f"  Engineered features: {len(feat_cols)}")
    print(f"  Label balance (train): {train_fe['outperform'].mean():.3f}")

    print("\n[4] Saving outputs…")
    train_fe.to_parquet(DATA_SUBDIR / "train_fe.parquet", index=False)
    val_fe.to_parquet(DATA_SUBDIR / "val_fe.parquet",   index=False)
    joblib.dump(train_stats, DATA_SUBDIR / "fe_stats.pkl")

    feat_summary = pd.DataFrame({"feature": feat_cols,
                                  "null_pct": [train_fe[c].isna().mean() for c in feat_cols]})
    feat_summary.to_csv(REPORT_DIR / "engineered_features_list.csv", index=False)
    log.info("Saved train_fe.parquet, val_fe.parquet, fe_stats.pkl")

    print("[5] Plotting feature summary…")
    plot_feature_summary(train_fe)

    print(f"\n  All outputs → {DATA_SUBDIR}")
    print("=" * 65)
    print("  Step 3 complete. Ready for Model Training (04_)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
