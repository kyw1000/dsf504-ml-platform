"""
use_case_F_esg/01_data_loading.py
====================================
DSF504 — Use Case F: ESG & Greenwashing Risk Scoring
Step 1: Data loading, augmentation, profiling, and train/val/test split

Dataset — Hybrid approach
--------------------------
PRIMARY (requires internet): ClimateBERT environmental_claims
  Real sentences from corporate annual reports, sustainability reports,
  and earnings call transcripts, expert-annotated for environmental claims.
  Source: Stammbach et al. (2022) — University of Zurich
  HuggingFace: climatebert/environmental_claims  (CC-BY-NC-SA 4.0)
  Size: ~2,647 rows | Format: parquet | No auth required

FALLBACK (offline / sandbox): Synthetic ESG Disclosure Dataset
  Template-generated sentences following ClimateBERT vocabulary patterns,
  with the same schema. Activates automatically when HuggingFace is
  unreachable (e.g. sandboxed classroom environments).

Hybrid Augmentation
-------------------
Each sentence (real or synthetic) is assigned to a fictional listed company
and augmented with synthetic structured ESG features:
  - sector, market_cap_bn, revenue_bn, emissions_intensity
  - e_score / s_score / g_score  (independently assessed, 0-100)
  - reported_e / reported_s / reported_g  (company self-reported, may be inflated)
  - e_gap / s_gap / g_gap / avg_gap  (reported minus assessed)

Target: greenwashing_risk  (Low / Medium / High)
  Derived from the intersection of sentence-level environmental claim label
  and company-level ESG gap (inflation between reported and assessed scores).

Greenwashing Risk Logic
------------------------
  HIGH  : sentence makes environmental claim (label=1)
          AND company avg_gap > 20 pts (significant score inflation)
  MEDIUM: sentence makes environmental claim (label=1) AND gap 8–20 pts
          OR sentence makes no claim (label=0) AND gap > 25 pts (concealment)
  LOW   : aligned companies making substantiated claims, or neutral sentences

EDGAR Extension Exercise
-------------------------
Students can pull real 10-K Item 1A (Risk Factors) text for the same
companies using the SEC EDGAR Full-Text Search API:

  https://efts.sec.gov/LATEST/search-index?q=%22climate%22
    &dateRange=custom&startdt=2020-01-01&enddt=2024-12-31&forms=10-K

Replace the `disclosure_text` column with real 10-K excerpts to see how
model performance changes on authentic (noisier, longer) disclosure language.

Academic references
-------------------
- Stammbach et al. (2022). A Dataset for Detecting Real-World Environmental
  Claims. arXiv:2209.00507.
- Lyon & Maxwell (2011). Greenwash: Corporate environmental disclosure under
  threat of audit. Journal of Economics & Management Strategy, 20(1).
- Escrig-Olmedo et al. (2019). Rating the raters: Evaluating how ESG rating
  agencies integrate sustainability principles. Sustainability, 11(3).
- Loughran & McDonald (2011). When is a liability not a liability?
  Journal of Finance, 66(1).
"""

from __future__ import annotations

import sys
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_SUBDIR = DATA_DIR / "sec_esg"
REPORT_DIR  = REPORTS_DIR / "use_case_F"
DATA_SUBDIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "greenwashing_risk"
N_COMPANIES  = 400   # fictional companies sentences are assigned to
SEED         = RANDOM_STATE
HF_REPO      = "climatebert/environmental_claims"

# ── Sector definitions ───────────────────────────────────────────────────────
SECTORS = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials", "Information Technology",
    "Communication Services", "Utilities", "Real Estate",
]

SECTOR_BASE_ESG = {
    "Energy":                 (35, 50, 55),
    "Materials":              (42, 48, 58),
    "Industrials":            (52, 55, 62),
    "Consumer Discretionary": (55, 60, 65),
    "Consumer Staples":       (58, 62, 68),
    "Health Care":            (60, 65, 70),
    "Financials":             (50, 58, 72),
    "Information Technology": (62, 65, 68),
    "Communication Services": (55, 60, 65),
    "Utilities":              (45, 55, 60),
    "Real Estate":            (50, 52, 65),
}

# ── Synthetic text vocabulary (used in fallback mode) ────────────────────────
_ENV_CLAIM_SENTENCES = [
    "We are committed to achieving net-zero emissions across our operations by 2040.",
    "Our renewable energy procurement reached 100% across all owned facilities this year.",
    "The company reduced Scope 1 and 2 greenhouse gas emissions by 34% versus our 2019 baseline.",
    "We have set science-based targets aligned with a 1.5°C pathway under the Paris Agreement.",
    "Our circular economy initiatives diverted over 85% of operational waste from landfill.",
    "We invested $420 million in clean energy infrastructure and energy efficiency this fiscal year.",
    "Our water recycling program reduced freshwater withdrawal intensity by 28% year-over-year.",
    "We advocate for national climate policies that advance the Paris Agreement objectives.",
    "Emissions from our supply chain (Scope 3) are now tracked and disclosed annually.",
    "We partnered with WWF to restore 50,000 hectares of degraded forest in Southeast Asia.",
    "Our facilities achieved ISO 14001 environmental management certification across all regions.",
    "We reduced average carbon intensity per unit of production by 19% this reporting period.",
    "The Board's Sustainability Committee oversees climate risk integration into strategy.",
    "We are accelerating electrification of our vehicle fleet, targeting 100% EV by 2030.",
    "Our product portfolio is now 60% low-carbon or climate-positive solutions.",
    "We aligned capital allocation decisions with TCFD recommendations this fiscal year.",
    "Hydro has started working on initiatives to reduce direct CO2 emissions in primary production.",
    "We advocate for and supported the Government's ban on new petrol and diesel vehicles.",
    "Greater use of renewables has the added benefit of reducing our carbon footprint.",
    "We set the goal of reducing total GHG emissions by 43% by 2021 compared to 2015.",
    "The fund was 62% less carbon-intensive than its MSCI benchmark.",
    "We reduce water consumption and identify opportunities to install water-efficient practices.",
    "The partnership is an example of how we work with our supply chain to reduce emissions.",
    "Building a smarter energy grid that better serves customers is at the heart of our initiative.",
    "Our ambition is to provide more environmentally sustainable products to our customers.",
]

_NON_CLAIM_SENTENCES = [
    "Net revenue for the fiscal year was $4.2 billion, up 7% from the prior year.",
    "The Board approved a $500 million share repurchase program effective immediately.",
    "Operating margins improved to 18.4% driven by cost efficiencies and pricing actions.",
    "We completed the acquisition of Nexus Technologies for a total consideration of $1.1 billion.",
    "Capital expenditure guidance for the year is set at $320–350 million.",
    "The company reported diluted earnings per share of $3.87 for the fiscal year.",
    "We continue to see strong demand in our North American and Asia-Pacific segments.",
    "Our debt-to-EBITDA ratio improved to 2.1x following the refinancing in Q3.",
    "Headcount increased by 1,200 employees, primarily in R&D and customer success functions.",
    "The Audit Committee reviewed and approved the internal controls framework for the year.",
    "Currency headwinds reduced reported revenue growth by approximately 3 percentage points.",
    "We expect first-quarter revenue to be in the range of $980 million to $1.02 billion.",
    "Our credit facility was extended to 2028 with improved covenant terms.",
    "The pension deficit narrowed to $140 million, reflecting actuarial gain assumptions.",
    "Free cash flow conversion remained strong at 92% of adjusted net income.",
    "The company operates in 42 countries with approximately 38,000 employees worldwide.",
    "We completed the divestiture of our non-core consumer division for $680 million.",
    "Inventory levels were reduced by $180 million through working capital initiatives.",
    "Risk factors include foreign exchange volatility, interest rate movements, and cyber threats.",
    "The regulatory environment in our key markets remains complex and subject to change.",
    "We assess interest rate risk through sensitivity analysis on our floating-rate instruments.",
    "Our principal markets are the United States, Germany, Japan, and the United Kingdom.",
    "We rely on third-party contract manufacturers for approximately 60% of our production.",
    "The company maintains a diversified customer base with no single customer exceeding 8% of revenue.",
    "Backlog at year-end stood at $2.3 billion, up 12% from the prior year.",
]

_BOILERPLATE = [
    "Our sustainability strategy is aligned with the UN Sustainable Development Goals.",
    "We publish an annual sustainability report consistent with GRI Standards.",
    "Climate risk is integrated into our enterprise risk management framework.",
    "We engage regularly with investors, employees, and communities on ESG matters.",
    "The company is subject to evolving environmental regulations in its operating jurisdictions.",
    "We continue to refine our ESG data collection and reporting processes.",
]


def _load_climatebert() -> pd.DataFrame | None:
    """Attempt to load ClimateBERT environmental_claims from HuggingFace."""
    # Try datasets library first
    try:
        from datasets import load_dataset  # type: ignore
        log.info("Downloading ClimateBERT environmental_claims via datasets library...")
        ds = load_dataset(HF_REPO)
        frames = []
        for split_name, split in ds.items():
            df = split.to_pandas()
            df["hf_split"] = split_name
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.rename(columns={"label": "env_claim_label"})
        log.info("Downloaded %d rows from HuggingFace (%s)", len(combined), HF_REPO)
        return combined
    except Exception as e:
        log.warning("datasets library failed: %s", e)

    # Try direct parquet download
    try:
        import urllib.request
        splits_urls = {
            "train": (
                "https://huggingface.co/datasets/climatebert/environmental_claims"
                "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
            ),
            "validation": (
                "https://huggingface.co/datasets/climatebert/environmental_claims"
                "/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
            ),
            "test": (
                "https://huggingface.co/datasets/climatebert/environmental_claims"
                "/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
            ),
        }
        frames = []
        for split_name, url in splits_urls.items():
            cache = DATA_SUBDIR / f"climatebert_{split_name}.parquet"
            if not cache.exists():
                log.info("Downloading %s split from HuggingFace CDN...", split_name)
                req = urllib.request.Request(url, headers={"User-Agent": "DSF504/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    cache.write_bytes(resp.read())
            df = pd.read_parquet(cache)
            df["hf_split"] = split_name
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.rename(columns={"label": "env_claim_label"})
        log.info("Downloaded %d rows via direct parquet CDN", len(combined))
        return combined
    except Exception as e:
        log.warning("Direct parquet download failed: %s", e)

    return None


def _generate_synthetic_sentences(n_claim: int, n_nonclaim: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic sentences following ClimateBERT vocabulary patterns."""
    rows = []
    for _ in range(n_claim):
        base = rng.choice(_ENV_CLAIM_SENTENCES)
        # occasionally prepend boilerplate
        if rng.random() < 0.3:
            bp = rng.choice(_BOILERPLATE)
            text = f"{bp} {base}"
        else:
            text = base
        rows.append({"text": text, "env_claim_label": 1, "hf_split": "synthetic"})
    for _ in range(n_nonclaim):
        text = rng.choice(_NON_CLAIM_SENTENCES)
        rows.append({"text": text, "env_claim_label": 0, "hf_split": "synthetic"})
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=int(rng.integers(0, 99999))).reset_index(drop=True)


def _build_company_table(n: int, rng: np.random.Generator, py_rng: random.Random) -> pd.DataFrame:
    """Generate N synthetic companies with ESG scores and gap features."""
    rows = []
    for i in range(n):
        sector = py_rng.choice(SECTORS)
        base_e, base_s, base_g = SECTOR_BASE_ESG[sector]

        e_score = float(np.clip(rng.normal(base_e, 12), 5, 95))
        s_score = float(np.clip(rng.normal(base_s, 10), 5, 95))
        g_score = float(np.clip(rng.normal(base_g, 8),  5, 95))

        # Greenwash tendency — how much the company inflates its self-reported scores
        # Explicit 3-tier greenwash generation ensures realistic class balance
        # 60% Low-gap companies, 25% Medium-gap, 15% High-gap greenwashers
        gw_tier = rng.choice([0, 1, 2], p=[0.60, 0.25, 0.15])
        if gw_tier == 2:        # High greenwasher: avg_gap in [22, 40]
            target_avg_gap = float(rng.uniform(22, 40))
        elif gw_tier == 1:      # Medium greenwasher: avg_gap in [9, 21]
            target_avg_gap = float(rng.uniform(9, 21))
        else:                   # Low greenwasher: avg_gap in [0, 8]
            target_avg_gap = float(rng.uniform(0, 8))

        # Decompose avg_gap into correlated individual gaps with mild variance
        noise_e = float(rng.normal(0, 2.0))
        noise_s = float(rng.normal(0, 1.5))
        noise_g = float(rng.normal(0, 1.0))
        e_gap_raw = float(np.clip(target_avg_gap + noise_e, 0, 50))
        s_gap_raw = float(np.clip(target_avg_gap + noise_s, 0, 40))
        g_gap_raw = float(np.clip(target_avg_gap + noise_g, 0, 30))

        reported_e = float(np.clip(e_score + e_gap_raw, 0, 100))
        reported_s = float(np.clip(s_score + s_gap_raw, 0, 100))
        reported_g = float(np.clip(g_score + g_gap_raw, 0, 100))
        # Recompute gw_tendency as a [0,1] normalised proxy for backward compat
        gw_tendency = float(np.clip(target_avg_gap / 40.0, 0, 1))

        rows.append({
            "company_id":          f"CO{i:04d}",
            "sector":              sector,
            "market_cap_bn":       round(float(np.clip(rng.lognormal(2.5, 1.5), 0.1, 2000)), 2),
            "revenue_bn":          round(float(np.clip(rng.lognormal(1.8, 1.4), 0.05, 800)), 2),
            "emissions_intensity": round(float(np.clip(rng.lognormal(3.0, 1.2), 1, 5000)), 1),
            "e_score":             round(e_score, 1),
            "s_score":             round(s_score, 1),
            "g_score":             round(g_score, 1),
            "composite_esg":       round((e_score + s_score + g_score) / 3, 1),
            "reported_e":          round(reported_e, 1),
            "reported_s":          round(reported_s, 1),
            "reported_g":          round(reported_g, 1),
            "e_gap":               round(reported_e - e_score, 1),
            "s_gap":               round(reported_s - s_score, 1),
            "g_gap":               round(reported_g - g_score, 1),
            "avg_gap":             round(((reported_e - e_score) + (reported_s - s_score) + (reported_g - g_score)) / 3, 1),
            "gw_tendency":         round(gw_tendency, 4),
        })
    return pd.DataFrame(rows)


def _assign_risk_label(env_claim: int, avg_gap: float, rng: np.random.Generator) -> str:
    """
    Map (sentence claim label, company ESG gap) → greenwashing risk tier.

    HIGH  : sentence makes an environmental claim AND company has inflated scores (gap > 20)
    MEDIUM: moderate mismatch — claim with modest gap, OR no claim but large concealment gap
    LOW   : aligned company OR neutral sentence from a low-gap company
    """
    noise = float(rng.normal(0, 2.0))
    effective_gap = avg_gap + noise

    if env_claim == 1:
        if effective_gap > 20:
            return "High"
        elif effective_gap > 8:
            return "Medium"
        else:
            return "Low"
    else:  # no environmental claim
        if effective_gap > 25:
            return "Medium"   # company hides bad practices without claiming good ones
        else:
            return "Low"


def main() -> None:
    log.info("=" * 60)
    log.info("Step 1 — ESG & Greenwashing: Data loading & augmentation")
    log.info("=" * 60)

    rng    = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)

    # ── 1. Load or generate text data ────────────────────────────────────────
    text_df = _load_climatebert()
    if text_df is not None:
        data_source = "ClimateBERT (real)"
        log.info("Using REAL ClimateBERT sentences (%d rows)", len(text_df))
    else:
        log.warning(
            "HuggingFace unreachable — falling back to synthetic sentences.\n"
            "  Students: run this script with internet access to get real data.\n"
            "  Extension: use EDGAR EFTS endpoint for real 10-K Item 1A text."
        )
        n_total   = 2_647
        n_claim   = int(n_total * 0.37)   # match ClimateBERT's ~37% positive rate
        n_nonclaim = n_total - n_claim
        text_df   = _generate_synthetic_sentences(n_claim, n_nonclaim, rng)
        data_source = "Synthetic (fallback)"
        log.info("Generated %d synthetic sentences", len(text_df))

    log.info(
        "Sentence label distribution:\n  label=1 (env claim): %d | label=0 (no claim): %d",
        (text_df["env_claim_label"] == 1).sum(),
        (text_df["env_claim_label"] == 0).sum(),
    )

    # ── 2. Build company table and assign sentences ───────────────────────────
    log.info("Building %d synthetic companies...", N_COMPANIES)
    companies = _build_company_table(N_COMPANIES, rng, py_rng)

    # Assign each sentence to a company (roughly equal distribution)
    n_rows = len(text_df)
    company_ids = np.tile(companies["company_id"].values, int(np.ceil(n_rows / N_COMPANIES)))[:n_rows]
    rng.shuffle(company_ids)
    text_df = text_df.copy()
    text_df["company_id"] = company_ids

    # ── 3. Merge company features onto sentences ──────────────────────────────
    df = text_df.merge(companies, on="company_id", how="left")

    # ── 4. Assign greenwashing risk labels ────────────────────────────────────
    df[TARGET] = [
        _assign_risk_label(row["env_claim_label"], row["avg_gap"], rng)
        for _, row in df.iterrows()
    ]
    df["data_source"] = data_source

    log.info(
        "Greenwashing risk distribution:\n%s",
        df[TARGET].value_counts().to_string()
    )
    log.info(
        "Risk × env_claim cross-tab:\n%s",
        pd.crosstab(df[TARGET], df["env_claim_label"]).to_string()
    )

    # ── 5. Company-stratified train / val / test split ────────────────────────
    # Split by company to prevent sentences from the same company leaking
    # across train/val/test (data leakage prevention)
    from sklearn.model_selection import train_test_split

    # Majority risk label per company for stratification
    company_risk = (
        df.groupby("company_id")[TARGET]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={TARGET: "majority_risk"})
    )

    co_train, co_temp = train_test_split(
        company_risk, test_size=0.30,
        stratify=company_risk["majority_risk"], random_state=SEED
    )
    co_val, co_test = train_test_split(
        co_temp, test_size=0.50,
        stratify=co_temp["majority_risk"], random_state=SEED
    )

    df_train = df[df["company_id"].isin(co_train["company_id"])].reset_index(drop=True)
    df_val   = df[df["company_id"].isin(co_val["company_id"])].reset_index(drop=True)
    df_test  = df[df["company_id"].isin(co_test["company_id"])].reset_index(drop=True)

    log.info(
        "Company-stratified split → train: %d rows (%d cos) | val: %d rows (%d cos) | test: %d rows (%d cos)",
        len(df_train), len(co_train),
        len(df_val),   len(co_val),
        len(df_test),  len(co_test),
    )

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    df_train.to_parquet(DATA_SUBDIR / "train.parquet", index=False)
    df_val.to_parquet(DATA_SUBDIR   / "val.parquet",   index=False)
    df_test.to_parquet(DATA_SUBDIR  / "test.parquet",  index=False)
    df.to_parquet(DATA_SUBDIR       / "full.parquet",  index=False)
    companies.to_parquet(DATA_SUBDIR / "companies.parquet", index=False)
    log.info("Saved parquet artefacts to %s", DATA_SUBDIR)

    # ── 7. Summary plots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    order  = ["Low", "Medium", "High"]
    colors = ["#66BB6A", "#FFA726", "#EF5350"]

    # (a) Risk distribution
    counts = df[TARGET].value_counts().reindex(order, fill_value=0)
    axes[0].bar(order, counts.values, color=colors, edgecolor="white")
    axes[0].set_title("Greenwashing Risk Distribution", fontweight="bold")
    axes[0].set_xlabel("Risk Tier"); axes[0].set_ylabel("Sentences")
    for j, v in enumerate(counts.values):
        axes[0].text(j, v + 5, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

    # (b) Risk by env_claim label
    ct = pd.crosstab(df[TARGET], df["env_claim_label"]).reindex(order, fill_value=0)
    ct.columns = ["No Claim (0)", "Env Claim (1)"]
    ct.plot(kind="bar", ax=axes[1], color=["#42A5F5", "#AB47BC"], edgecolor="white")
    axes[1].set_title("Risk Tier by Environmental Claim Label", fontweight="bold")
    axes[1].set_xlabel("Risk Tier"); axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Sentence Label")

    # (c) ESG gap distribution by risk tier
    for tier, col in zip(order, colors):
        vals = df.loc[df[TARGET] == tier, "avg_gap"]
        axes[2].hist(vals, bins=20, alpha=0.6, color=col, label=tier, density=True)
    axes[2].set_title("ESG Score Gap (Reported − Assessed) by Risk", fontweight="bold")
    axes[2].set_xlabel("Average Gap (pts)"); axes[2].set_ylabel("Density")
    axes[2].legend(title="Risk Tier")

    plt.tight_layout()
    out = REPORT_DIR / "target_distribution.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved target_distribution.png")

    log.info("Step 1 complete ✓  [data source: %s]", data_source)
    log.info(
        "\n  EDGAR extension exercise:\n"
        "  https://efts.sec.gov/LATEST/search-index"
        "?q=%%22climate%%22&forms=10-K&dateRange=custom"
        "&startdt=2020-01-01&enddt=2024-12-31\n"
        "  Replace disclosure_text column with real Item 1A excerpts."
    )


if __name__ == "__main__":
    main()
