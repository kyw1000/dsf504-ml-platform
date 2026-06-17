"""
use_case_F_esg/02_eda_analysis.py
====================================
DSF504 — Use Case F: ESG & Greenwashing Risk Scoring
Step 2: Exploratory Data Analysis

Outputs (saved to reports/use_case_F/)
----------------------------------------
  overview.png                  — dataset summary stats
  target_distribution.png       — greenwashing risk class balance
  missing_heatmap.png           — missing value profile
  numeric_distributions.png     — ESG score distributions by risk tier
  esg_gap_analysis.png          — score inflation gap deep-dive
  sector_analysis.png           — sector breakdown of risk tiers
  text_length_distribution.png  — disclosure text length analysis
  correlation_heatmap.png       — structured feature correlations
  engineered_feature_summary.png — key signal summary
  raw_vs_processed_distributions.png — before/after view
  eda_summary.csv               — numeric statistics table
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR
from utils.encoding_guard import ensure_utf8

ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "sec_esg"
REPORT_DIR  = REPORTS_DIR / "use_case_F"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET  = "greenwashing_risk"
ORDER   = ["Low", "Medium", "High"]
PALETTE = {"Low": "#66BB6A", "Medium": "#FFA726", "High": "#EF5350"}
BG      = "#1A1A2E"


def _style(fig: plt.Figure) -> None:
    fig.patch.set_facecolor("white")


def main() -> None:
    log.info("Step 2 — ESG & Greenwashing: EDA")

    train = pd.read_parquet(DATA_SUBDIR / "train.parquet")
    full  = pd.read_parquet(DATA_SUBDIR / "full.parquet")
    log.info("Loaded full dataset: %d rows, %d cols", *full.shape)

    num_cols = ["e_score","s_score","g_score","composite_esg",
                "reported_e","reported_s","reported_g",
                "e_gap","s_gap","g_gap","avg_gap",
                "market_cap_bn","revenue_bn","emissions_intensity"]
    text_col = "disclosure_text" if "disclosure_text" in full.columns else "text"

    # ── 1. Overview ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    _style(fig)
    fig.suptitle("UC-F — ESG & Greenwashing Dataset Overview", fontsize=14, fontweight="bold")

    # class balance
    counts = full[TARGET].value_counts().reindex(ORDER, fill_value=0)
    axes[0,0].bar(ORDER, counts.values, color=[PALETTE[r] for r in ORDER], edgecolor="white")
    axes[0,0].set_title("Greenwashing Risk Distribution")
    axes[0,0].set_ylabel("Sentences")
    for j, v in enumerate(counts.values):
        axes[0,0].text(j, v + 4, f"{v} ({v/len(full)*100:.1f}%)", ha="center", fontsize=9)

    # env_claim breakdown
    ct = pd.crosstab(full[TARGET], full["env_claim_label"]).reindex(ORDER, fill_value=0)
    ct.columns = ["No Claim", "Env Claim"]
    ct.plot(kind="bar", ax=axes[0,1], color=["#42A5F5","#AB47BC"], edgecolor="white")
    axes[0,1].set_title("Risk Tier × Environmental Claim Label")
    axes[0,1].set_xlabel(""); axes[0,1].tick_params(axis="x", rotation=0)
    axes[0,1].legend(title="Claim Label")

    # ESG gap by risk tier boxplot
    gap_data = [full.loc[full[TARGET]==r, "avg_gap"].values for r in ORDER]
    bp = axes[1,0].boxplot(gap_data, patch_artist=True, tick_labels=ORDER, widths=0.5)
    for patch, r in zip(bp["boxes"], ORDER):
        patch.set_facecolor(PALETTE[r])
        patch.set_alpha(0.8)
    axes[1,0].set_title("ESG Score Gap (Reported − Assessed) by Risk Tier")
    axes[1,0].set_ylabel("Average Gap (pts)")
    axes[1,0].axhline(0, color="grey", linestyle="--", linewidth=0.8)

    # Sector distribution
    top_sectors = full["sector"].value_counts().head(8)
    axes[1,1].barh(top_sectors.index[::-1], top_sectors.values[::-1], color="#42A5F5")
    axes[1,1].set_title("Top Sectors by Record Count")
    axes[1,1].set_xlabel("Sentences")

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "overview.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved overview.png")

    # ── 2. Numeric distributions by risk tier ────────────────────────────────
    score_cols = ["e_score","s_score","g_score","composite_esg",
                  "reported_e","reported_s","reported_g"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    _style(fig)
    fig.suptitle("ESG Score Distributions by Greenwashing Risk Tier", fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()
    for i, col in enumerate(score_cols):
        ax = axes_flat[i]
        for tier in ORDER:
            vals = full.loc[full[TARGET]==tier, col].dropna()
            ax.hist(vals, bins=20, alpha=0.5, color=PALETTE[tier], label=tier, density=True)
        ax.set_title(col.replace("_"," ").title(), fontsize=10)
        ax.set_xlabel("Score (0–100)")
        if i == 0:
            ax.legend(title="Risk Tier", fontsize=8)
    axes_flat[-1].set_visible(False)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "numeric_distributions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved numeric_distributions.png")

    # ── 3. ESG gap deep-dive ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _style(fig)
    fig.suptitle("ESG Score Inflation Gap (Reported − Independently Assessed)", fontsize=13, fontweight="bold")
    for i, (col, label) in enumerate([("e_gap","Environmental Gap"),
                                       ("s_gap","Social Gap"),
                                       ("g_gap","Governance Gap")]):
        ax = axes[i]
        for tier in ORDER:
            vals = full.loc[full[TARGET]==tier, col].dropna()
            ax.hist(vals, bins=25, alpha=0.55, color=PALETTE[tier], label=tier, density=True)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero gap")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Gap (pts)"); ax.set_ylabel("Density")
        if i == 0:
            ax.legend(title="Risk Tier", fontsize=8)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "esg_gap_analysis.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved esg_gap_analysis.png")

    # ── 4. Sector analysis ────────────────────────────────────────────────────
    sector_risk = (
        full.groupby("sector")[TARGET]
        .value_counts(normalize=True)
        .mul(100).rename("pct")
        .reset_index()
    )
    sectors = full["sector"].unique()
    high_pct = (
        sector_risk[sector_risk[TARGET]=="High"]
        .set_index("sector")["pct"]
        .reindex(sectors, fill_value=0)
        .sort_values(ascending=True)
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _style(fig)
    axes[0].barh(high_pct.index, high_pct.values, color="#EF5350", edgecolor="white")
    axes[0].set_title("% High Greenwashing Risk by Sector", fontweight="bold")
    axes[0].set_xlabel("% of Sentences Classified High Risk")
    axes[0].axvline(high_pct.mean(), color="grey", linestyle="--",
                    label=f"Mean: {high_pct.mean():.1f}%")
    axes[0].legend()

    pivot = pd.crosstab(full["sector"], full[TARGET]).reindex(columns=ORDER, fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100)
    pivot_pct.plot(kind="barh", stacked=True, ax=axes[1],
                   color=[PALETTE[r] for r in ORDER], edgecolor="white")
    axes[1].set_title("Risk Tier Composition by Sector (%)", fontweight="bold")
    axes[1].set_xlabel("% of Sentences"); axes[1].set_ylabel("")
    axes[1].legend(title="Risk Tier", loc="lower right")

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "sector_analysis.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved sector_analysis.png")

    # ── 5. Text length distribution ───────────────────────────────────────────
    full["text_len"] = full[text_col].str.len()
    full["word_count"] = full[text_col].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _style(fig)
    for tier in ORDER:
        vals = full.loc[full[TARGET]==tier, "text_len"]
        axes[0].hist(vals, bins=30, alpha=0.55, color=PALETTE[tier], label=tier, density=True)
    axes[0].set_title("Disclosure Text Length (chars) by Risk Tier", fontweight="bold")
    axes[0].set_xlabel("Character count"); axes[0].set_ylabel("Density")
    axes[0].legend(title="Risk Tier")

    for tier in ORDER:
        vals = full.loc[full[TARGET]==tier, "word_count"]
        axes[1].hist(vals, bins=25, alpha=0.55, color=PALETTE[tier], label=tier, density=True)
    axes[1].set_title("Word Count by Risk Tier", fontweight="bold")
    axes[1].set_xlabel("Word count"); axes[1].set_ylabel("Density")
    axes[1].legend(title="Risk Tier")

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "text_length_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved text_length_distribution.png")

    # ── 6. Correlation heatmap ────────────────────────────────────────────────
    corr_cols = ["e_score","s_score","g_score","composite_esg",
                 "e_gap","s_gap","g_gap","avg_gap",
                 "market_cap_bn","revenue_bn","emissions_intensity","env_claim_label"]
    corr = full[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    _style(fig)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols, fontsize=8)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(corr.iloc[i,j]) < 0.6 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pearson Correlation — ESG Structured Features", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved correlation_heatmap.png")

    # ── 7. Missing heatmap ────────────────────────────────────────────────────
    missing_pct = full[num_cols].isna().mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    _style(fig)
    colors_m = ["#EF5350" if v > 5 else "#FFA726" if v > 0 else "#66BB6A"
                for v in missing_pct.values]
    ax.bar(missing_pct.index, missing_pct.values, color=colors_m, edgecolor="white")
    ax.set_title("Missing Value Rate per Feature (%)", fontweight="bold")
    ax.set_ylabel("% Missing"); ax.tick_params(axis="x", rotation=45)
    ax.axhline(5, color="red", linestyle="--", linewidth=0.8, label="5% threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "missing_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved missing_heatmap.png")

    # ── 8. Engineered feature summary ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _style(fig)
    # avg_gap vs composite_esg scatter by risk tier
    for tier in ORDER:
        sub = full[full[TARGET]==tier]
        axes[0].scatter(sub["composite_esg"], sub["avg_gap"],
                        alpha=0.4, s=15, color=PALETTE[tier], label=tier)
    axes[0].set_title("ESG Score vs Score Inflation Gap", fontweight="bold")
    axes[0].set_xlabel("Composite ESG Score (assessed)")
    axes[0].set_ylabel("Avg Gap (reported − assessed)")
    axes[0].legend(title="Risk Tier", markerscale=2)

    # emissions intensity by sector (top 5)
    top5 = full["sector"].value_counts().head(5).index
    ei_by_sector = full[full["sector"].isin(top5)].groupby("sector")["emissions_intensity"].median().sort_values()
    axes[1].barh(ei_by_sector.index, ei_by_sector.values, color="#42A5F5")
    axes[1].set_title("Median Emissions Intensity by Sector", fontweight="bold")
    axes[1].set_xlabel("tCO₂e per $M revenue")

    # env_claim_label proportion by sector (top 5)
    claim_by_sector = (
        full[full["sector"].isin(top5)]
        .groupby("sector")["env_claim_label"].mean().mul(100)
        .sort_values()
    )
    axes[2].barh(claim_by_sector.index, claim_by_sector.values, color="#AB47BC")
    axes[2].set_title("% Sentences with Environmental Claim", fontweight="bold")
    axes[2].set_xlabel("% of Sentences (label=1)")

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "engineered_feature_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved engineered_feature_summary.png")

    # ── 9. Raw vs processed ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _style(fig)
    axes[0].hist(full["avg_gap"], bins=40, color="#42A5F5", edgecolor="white", alpha=0.8)
    axes[0].set_title("Raw: avg_gap (score inflation)", fontweight="bold")
    axes[0].set_xlabel("Average Gap (pts)"); axes[0].set_ylabel("Count")
    axes[0].axvline(full["avg_gap"].mean(), color="red", linestyle="--",
                    label=f"Mean: {full['avg_gap'].mean():.1f}")
    axes[0].legend()

    full["avg_gap_clipped"] = full["avg_gap"].clip(lower=0)
    axes[1].hist(full["avg_gap_clipped"], bins=40, color="#66BB6A", edgecolor="white", alpha=0.8)
    axes[1].set_title("Processed: avg_gap clipped to [0, ∞]", fontweight="bold")
    axes[1].set_xlabel("Average Gap (pts, clipped)"); axes[1].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "raw_vs_processed_distributions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved raw_vs_processed_distributions.png")

    # ── 10. Summary CSV ───────────────────────────────────────────────────────
    summary = full[num_cols].describe().T
    summary.to_csv(REPORT_DIR / "eda_summary.csv")
    log.info("Saved eda_summary.csv")

    log.info("\n=== EDA Summary ===")
    log.info("Total sentences : %d", len(full))
    log.info("Companies       : %d", full["company_id"].nunique())
    log.info("Sectors         : %d", full["sector"].nunique())
    log.info("Risk Low / Med / High : %d / %d / %d",
             (full[TARGET]=="Low").sum(),
             (full[TARGET]=="Medium").sum(),
             (full[TARGET]=="High").sum())
    log.info("Env claim rate  : %.1f%%", full["env_claim_label"].mean()*100)
    log.info("Avg ESG gap     : %.1f pts", full["avg_gap"].mean())
    log.info("Step 2 complete ✓")


if __name__ == "__main__":
    main()
