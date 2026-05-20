"""
use_case_F_esg/03_feature_engineering.py
==========================================
DSF504 — Use Case F: ESG & Greenwashing Risk Scoring
Step 3: Feature Engineering

Features produced
------------------
TEXT FEATURES (from disclosure_text / text column)
  tfidf_*         : TF-IDF unigram/bigram matrix (top 200 features by chi-squared)
  fe_text_len     : character count of disclosure text
  fe_word_count   : word count
  fe_avg_word_len : average word length (proxy for financial jargon density)
  fe_claim_density: proportion of climate-positive keywords in text

ESG GAP FEATURES (structured — claim inflation signals)
  fe_avg_gap_clipped  : avg_gap clipped to [0, ∞] (negative gaps → not greenwashing)
  fe_e_gap_clipped    : E pillar gap clipped
  fe_s_gap_clipped    : S pillar gap clipped
  fe_g_gap_clipped    : G pillar gap clipped
  fe_gap_cv           : coefficient of variation of E/S/G gaps (consistency of inflation)
  fe_max_gap          : max single-pillar gap (worst-case inflation)
  fe_composite_delta  : reported_composite − assessed_composite

ABSOLUTE ESG SCORE FEATURES
  fe_composite_esg    : assessed composite ESG score (0–100)
  fe_esg_low          : binary — composite_esg < 40 (low underlying ESG quality)
  fe_esg_high         : binary — composite_esg ≥ 70 (high underlying ESG quality)
  fe_e_score_norm     : e_score / 100 (normalised)
  fe_s_score_norm     : s_score / 100
  fe_g_score_norm     : g_score / 100

COMPANY FINANCIAL FEATURES
  fe_log_market_cap   : log1p(market_cap_bn)
  fe_log_revenue      : log1p(revenue_bn)
  fe_log_emissions    : log1p(emissions_intensity)
  fe_emissions_high   : binary — emissions_intensity > 500 tCO2e/$M (high-carbon)

SECTOR ENCODING
  sector_*            : target-encoded sector (mean greenwashing risk score per sector)
  fe_sector_<name>    : one-hot for top-8 sectors (others → "Other")

INTERACTION FEATURES
  fe_claim_x_gap      : env_claim_label × fe_avg_gap_clipped (claim-weighted gap)
  fe_gap_x_emissions  : fe_avg_gap_clipped × fe_emissions_high (high-carbon inflators)

TARGET ENCODING
  fe_sector_risk_te   : mean numeric risk score per sector (train only, no leakage)
"""

from __future__ import annotations

import sys
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder

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

DATA_SUBDIR = DATA_DIR / "sec_esg"
REPORT_DIR  = REPORTS_DIR / "use_case_F"

TARGET     = "greenwashing_risk"
RISK_MAP   = {"Low": 0, "Medium": 1, "High": 2}
N_TFIDF    = 200   # top TF-IDF features by chi-squared

CLIMATE_KEYWORDS = [
    "emission", "carbon", "climate", "renewable", "sustainable", "green",
    "net.zero", "scope", "esg", "environmental", "biodiversity", "recycl",
    "circulat", "clean.energy", "decarboni", "offset", "ghg", "footprint",
]
_KW_PATTERN = re.compile("|".join(CLIMATE_KEYWORDS), re.IGNORECASE)


def _text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Derive hand-crafted text features."""
    texts = df[text_col].fillna("").astype(str)
    out = pd.DataFrame(index=df.index)
    out["fe_text_len"]      = texts.str.len()
    out["fe_word_count"]    = texts.str.split().str.len()
    out["fe_avg_word_len"]  = texts.apply(
        lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0.0
    )
    out["fe_claim_density"] = texts.apply(
        lambda t: len(_KW_PATTERN.findall(t)) / max(len(t.split()), 1)
    )
    return out


def _gap_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in ["e_gap", "s_gap", "g_gap", "avg_gap"]:
        out[f"fe_{col}_clipped"] = df[col].clip(lower=0)

    gaps = df[["e_gap","s_gap","g_gap"]].clip(lower=0)
    gap_std  = gaps.std(axis=1).fillna(0)
    gap_mean = gaps.mean(axis=1).replace(0, np.nan)
    out["fe_gap_cv"]         = (gap_std / gap_mean).fillna(0)
    out["fe_max_gap"]        = gaps.max(axis=1)
    rep_comp = (df["reported_e"] + df["reported_s"] + df["reported_g"]) / 3
    asd_comp = df["composite_esg"]
    out["fe_composite_delta"] = rep_comp - asd_comp
    return out


def _esg_score_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["fe_composite_esg"] = df["composite_esg"]
    out["fe_esg_low"]       = (df["composite_esg"] < 40).astype(int)
    out["fe_esg_high"]      = (df["composite_esg"] >= 70).astype(int)
    out["fe_e_score_norm"]  = df["e_score"] / 100
    out["fe_s_score_norm"]  = df["s_score"] / 100
    out["fe_g_score_norm"]  = df["g_score"] / 100
    return out


def _financial_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["fe_log_market_cap"]  = np.log1p(df["market_cap_bn"].clip(lower=0))
    out["fe_log_revenue"]     = np.log1p(df["revenue_bn"].clip(lower=0))
    out["fe_log_emissions"]   = np.log1p(df["emissions_intensity"].clip(lower=0))
    out["fe_emissions_high"]  = (df["emissions_intensity"] > 500).astype(int)
    return out


def _sector_features(df: pd.DataFrame, train_risk_map: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """One-hot encode top sectors + target encode sector using train risk map."""
    top8 = df["sector"].value_counts().head(8).index.tolist()
    sector_safe = df["sector"].where(df["sector"].isin(top8), other="Other")

    ohe = pd.get_dummies(sector_safe, prefix="fe_sector", dtype=int)

    # Target encoding (train-only map to prevent leakage)
    if train_risk_map is None:
        risk_numeric = df[TARGET].map(RISK_MAP)
        train_risk_map = df.groupby("sector").apply(
            lambda g: risk_numeric.loc[g.index].mean()
        ).to_dict()

    out = ohe.copy()
    out["fe_sector_risk_te"] = df["sector"].map(train_risk_map).fillna(
        np.mean(list(train_risk_map.values()))
    )
    return out, train_risk_map


def _interaction_features(df_struct: pd.DataFrame, env_claim: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=df_struct.index)
    out["fe_claim_x_gap"]      = env_claim.values * df_struct["fe_avg_gap_clipped"].values
    out["fe_gap_x_emissions"]  = df_struct["fe_avg_gap_clipped"].values * df_struct["fe_emissions_high"].values
    return out


def engineer(df: pd.DataFrame, tfidf: TfidfVectorizer | None = None,
             selector: SelectKBest | None = None,
             sector_risk_map: dict | None = None,
             fit: bool = True) -> tuple[pd.DataFrame, TfidfVectorizer, SelectKBest, dict]:
    """Full feature engineering pipeline. fit=True fits transformers on df."""
    text_col = "disclosure_text" if "disclosure_text" in df.columns else "text"
    texts = df[text_col].fillna("").astype(str)

    # TF-IDF
    if fit:
        tfidf = TfidfVectorizer(
            max_features=2000, ngram_range=(1, 2),
            min_df=2, sublinear_tf=True, strip_accents="unicode",
        )
        X_tfidf_raw = tfidf.fit_transform(texts)
        y_num = df[TARGET].map(RISK_MAP).values
        selector = SelectKBest(chi2, k=N_TFIDF)
        selector.fit(X_tfidf_raw, y_num)
    else:
        X_tfidf_raw = tfidf.transform(texts)

    X_tfidf = selector.transform(X_tfidf_raw).toarray()
    selected_names = np.array(tfidf.get_feature_names_out())[selector.get_support()]
    tfidf_df = pd.DataFrame(X_tfidf, columns=[f"tfidf_{n.replace(' ','_')}" for n in selected_names],
                             index=df.index)

    # Structured features
    text_feats    = _text_features(df, text_col)
    gap_feats     = _gap_features(df)
    esg_feats     = _esg_score_features(df)
    fin_feats     = _financial_features(df)
    sec_feats, sector_risk_map = _sector_features(df, train_risk_map=sector_risk_map)
    interaction   = _interaction_features(
        pd.concat([gap_feats, esg_feats, fin_feats], axis=1),
        df["env_claim_label"].astype(int)
    )

    fe = pd.concat([tfidf_df, text_feats, gap_feats, esg_feats,
                    fin_feats, sec_feats, interaction], axis=1)

    # Append target + identifiers
    fe[TARGET]           = df[TARGET].values
    fe["company_id"]     = df["company_id"].values
    fe["env_claim_label"] = df["env_claim_label"].values

    log.info("Feature matrix: %d rows × %d features (+target)", *fe.shape)
    return fe, tfidf, selector, sector_risk_map


def main() -> None:
    log.info("Step 3 — ESG & Greenwashing: Feature Engineering")

    train_raw = pd.read_parquet(DATA_SUBDIR / "train.parquet")
    val_raw   = pd.read_parquet(DATA_SUBDIR / "val.parquet")
    test_raw  = pd.read_parquet(DATA_SUBDIR / "test.parquet")

    log.info("Train: %d | Val: %d | Test: %d", len(train_raw), len(val_raw), len(test_raw))

    # Fit on train, transform all splits
    train_fe, tfidf, selector, sec_map = engineer(train_raw, fit=True)
    val_fe,   *_  = engineer(val_raw,   tfidf=tfidf, selector=selector,
                             sector_risk_map=sec_map, fit=False)
    test_fe,  *_  = engineer(test_raw,  tfidf=tfidf, selector=selector,
                             sector_risk_map=sec_map, fit=False)

    # Save
    train_fe.to_parquet(DATA_SUBDIR / "train_fe.parquet", index=False)
    val_fe.to_parquet(DATA_SUBDIR   / "val_fe.parquet",   index=False)
    test_fe.to_parquet(DATA_SUBDIR  / "test_fe.parquet",  index=False)

    # Save feature list
    feature_cols = [c for c in train_fe.columns if c not in [TARGET, "company_id", "env_claim_label"]]
    pd.DataFrame({"feature": feature_cols}).to_csv(DATA_SUBDIR / "feature_list.csv", index=False)
    log.info("Saved %d features to feature_list.csv", len(feature_cols))

    # Quick stats
    log.info("\nTop 10 TF-IDF features by chi-squared score:")
    tfidf_cols = [c for c in train_fe.columns if c.startswith("tfidf_")]
    y_num = train_fe[TARGET].map(RISK_MAP)
    scores = {}
    for col in tfidf_cols[:50]:  # sample for speed
        from scipy.stats import f_oneway
        groups = [train_fe.loc[train_fe[TARGET]==r, col].values for r in ["Low","Medium","High"]]
        try:
            f, p = f_oneway(*groups)
            scores[col] = f
        except Exception:
            scores[col] = 0.0
    top10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, score in top10:
        log.info("  %-40s F=%.2f", name, score)

    log.info("Step 3 complete ✓")


if __name__ == "__main__":
    main()
