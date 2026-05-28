"""
dashboard/app.py
================
DSF504 Financial AI Analytics Platform — Main Dashboard

Multi-use-case ML platform dashboard built with Streamlit.
Navigation: sidebar use-case selector + top nav buttons for 8 pages.

Run from project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

# ── stdlib ─────────────────────────────────────────────────────────────────────
import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass  # sklearn < 1.2 does not expose this exception class
import logging
import datetime
from pathlib import Path
from typing import Optional

# ── third-party ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import streamlit as st

# ── Streamlit page config — MUST be the first st.* call ───────────────────────
st.set_page_config(
    page_title="DSF504 Financial AI Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── viz library ────────────────────────────────────────────────────────────────
try:
    from dashboard.viz_library import (
        kpi_cards, actual_vs_predicted, volatility_timeseries,
        rv_heatmap, forecast_ribbon, indexed_chart, scatter_bubble,
        seasonal_subseries, waterfall_chart, candlestick_chart,
    )
    _VIZ_OK = True
except Exception:
    _VIZ_OK = False

log = logging.getLogger(__name__)

# ── House colours ──────────────────────────────────────────────────────────────
BG       = "#1A1A2E"
FONT     = "#E0E0E0"
ACCENT   = "#3949AB"
BLUE     = "#42A5F5"
GRN      = "#66BB6A"
ORG      = "#FFA726"
RED      = "#EF5350"
PURP     = "#AB47BC"
GRID     = "#2A2A4A"

PALETTE: dict = {
    "primary": ACCENT,
    "danger":  RED,
    "success": GRN,
    "warning": ORG,
    "purple":  PURP,
    "grey":    "#78909C",
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    /* Base */
    html, body, [class*="css"] {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: {BG};
        color: {FONT};
    }}
    /* Section headers */
    .section-header {{
        background: linear-gradient(90deg, {ACCENT}, #1565C0);
        padding: 10px 20px;
        border-radius: 8px;
        margin-bottom: 18px;
    }}
    .section-header h2 {{
        color: #ffffff;
        margin: 0;
        font-size: 1.25rem;
        font-weight: 700;
    }}
    .section-header p {{
        color: #C5CAE9;
        margin: 2px 0 0;
        font-size: 0.82rem;
    }}
    /* Metric cards */
    .metric-card {{
        background: rgba(57,73,171,0.12);
        border: 1.5px solid {ACCENT};
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
    }}
    /* Pill badges */
    .pill {{
        display: inline-block;
        background: #1E88E5;
        color: #fff;
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 0.75rem;
        margin: 2px 3px;
    }}
    /* Step badges */
    .step-done  {{ background:#1B5E20; color:#A5D6A7; border-radius:6px; padding:4px 10px; font-size:0.80rem; margin:2px; }}
    .step-run   {{ background:#E65100; color:#FFE0B2; border-radius:6px; padding:4px 10px; font-size:0.80rem; margin:2px; }}
    .step-wait  {{ background:#263238; color:#90A4AE; border-radius:6px; padding:4px 10px; font-size:0.80rem; margin:2px; }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #0D0D1E;
    }}
    /* Dividers */
    hr {{ border-color: {ACCENT}33; }}
    /* ── Layout tightening ── */
    /* Small top gap so nav pills aren't clipped by the toolbar */
    div[data-testid="stMainBlockContainer"] {{
        padding-top: 1rem !important;
    }}
    /* Shrink gap between nav radio strip and page content */
    div[data-testid="stRadio"] {{
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }}
    div[data-testid="stRadio"] > div:first-child {{
        margin-bottom: 0 !important;
    }}
    /* Remove extra padding Streamlit adds below horizontal rules */
    div[data-testid="stMarkdown"] hr {{
        margin-top: 4px !important;
        margin-bottom: 4px !important;
    }}
    /* Sidebar image: pull image up, remove below whitespace */
    section[data-testid="stSidebar"] [data-testid="stImage"] {{
        margin-top: -1rem !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stImage"] img {{
        display: block;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# ── USE_CASE_SCRIPTS ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

USE_CASE_SCRIPTS: dict = {
    "A": {
        1: "use_case_A_fraud/01_data_loading.py",
        2: "use_case_A_fraud/02_eda_analysis.py",
        3: "use_case_A_fraud/03_feature_engineering.py",
        4: "use_case_A_fraud/04_model_training.py",
        5: "use_case_A_fraud/05_hyperparameter_tuning.py",
        6: "use_case_A_fraud/06_ethics_explainability.py",
    },
    "B": {
        1: "use_case_B_credit/01_data_loading.py",
        2: "use_case_B_credit/02_eda_analysis.py",
        3: "use_case_B_credit/03_feature_engineering.py",
        4: "use_case_B_credit/04_model_training.py",
        5: "use_case_B_credit/05_hyperparameter_tuning.py",
        6: "use_case_B_credit/06_ethics_explainability.py",
    },
    "C_nlp": {
        1: "use_case_C_nlp/01_data_loading.py",
        2: "use_case_C_nlp/02_eda_analysis.py",
        3: "use_case_C_nlp/03_feature_engineering.py",
        4: "use_case_C_nlp/04_model_training.py",
        5: "use_case_C_nlp/05_hyperparameter_tuning.py",
        6: "use_case_C_nlp/06_ethics_explainability.py",
    },
    "C_markets": {
        1: "use_case_C_market/01_data_loading.py",
        2: "use_case_C_market/02_eda_analysis.py",
        3: "use_case_C_market/03_feature_engineering.py",
        4: "use_case_C_market/04_model_training.py",
        5: "use_case_C_market/05_hyperparameter_tuning.py",
        # 5: "use_case_A_fraud/05b_lgbm_champion.py",  # alt
        6: "use_case_C_market/06_ethics_explainability.py",
    },
    "E": {
        1: "use_case_E_insurance/01_data_loading.py",
        2: "use_case_E_insurance/02_eda_analysis.py",
        3: "use_case_E_insurance/03_feature_engineering.py",
        4: "use_case_E_insurance/04_model_training.py",
        5: "use_case_E_insurance/05_hyperparameter_tuning.py",
        6: "use_case_E_insurance/06_ethics_explainability.py",
    },
    "D": {
        1: "use_case_D_churn/01_data_loading.py",
        2: "use_case_D_churn/02_eda_analysis.py",
        3: "use_case_D_churn/03_feature_engineering.py",
        4: "use_case_D_churn/04_model_training.py",
        5: "use_case_D_churn/05_hyperparameter_tuning.py",
        6: "use_case_D_churn/06_ethics_explainability.py",
    },
    "F": {
        1: "use_case_F_esg/01_data_loading.py",
        2: "use_case_F_esg/02_eda_analysis.py",
        3: "use_case_F_esg/03_feature_engineering.py",
        4: "use_case_F_esg/04_model_training.py",
        5: "use_case_F_esg/05_hyperparameter_tuning.py",
        6: "use_case_F_esg/06_ethics_explainability.py",
    },
    "B3": {
        1: "use_case_G_advisory/01_data_loading.py",
        2: "use_case_G_advisory/02_eda_analysis.py",
        3: "use_case_G_advisory/03_feature_engineering.py",
        4: "use_case_G_advisory/04_model_training.py",
        5: "use_case_G_advisory/05_hyperparameter_tuning.py",
        6: "use_case_G_advisory/06_ethics_explainability.py",
    },
    "G1": {
        1: "use_case_G1_robo/01_data_loading.py",
        2: "use_case_G1_robo/02_eda_analysis.py",
        3: "use_case_G1_robo/03_feature_engineering.py",
        4: "use_case_G1_robo/04_model_training.py",
        5: "use_case_G1_robo/05_hyperparameter_tuning.py",
        6: "use_case_G1_robo/06_ethics_explainability.py",
    },
    "G2": {
        1: "use_case_G2_xai/01_data_loading.py",
        2: "use_case_G2_xai/02_eda_analysis.py",
        3: "use_case_G2_xai/03_feature_engineering.py",
        4: "use_case_G2_xai/04_model_training.py",
        5: "use_case_G2_xai/05_hyperparameter_tuning.py",
        6: "use_case_G2_xai/06_ethics_explainability.py",
    },
}

STEP_NAMES: dict = {
    1: "Data Loading",
    2: "EDA & Data Understanding",
    3: "Data Preparation",
    4: "Algorithm Selection + Cross-Validation",
    5: "Hyperparameter Tuning + Final Training",
    6: "Ethics & Explainability",
}

STEP_DESCRIPTIONS: dict = {
    1: "Load raw CSVs, merge datasets, perform stratified train/val/test split, cache as Parquet for downstream steps.",
    2: "Statistical profiling, distribution analysis, class imbalance assessment, correlation analysis, and bias detection across protected attributes.",
    3: "Feature scaling, extraction, transformation, engineering, and selection. SMOTE applied on training fold only to prevent leakage.",
    4: "Establish a baseline (Logistic Regression), then compare 5 candidate algorithms using 5-fold stratified CV as the outer loop. CV scores determine the champion architecture for Step 5.",
    5: "Run Bayesian hyper-parameter search (Optuna TPE, 100 trials) on the Step-4 CV champion. Saves the final tuned model as champion.pkl.",
    6: "Compute SHAP / feature-importance values for the champion model, generate explainability plots (bar chart, beeswarm / per-class tokens), and run a fairness & bias audit across key subgroups. Outputs shap_feature_importance.csv and ethics_bias_report.csv.",
}

# ══════════════════════════════════════════════════════════════════════════════
# ── USE_CASE_META ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

USE_CASE_META: dict = {
    "A": {
        "title":      "Financial Crime & Fraud Detection",
        "icon":       "🚨",
        "tag":        "IEEE-CIS Fraud Detection",
        "target":     "isFraud",
        "task":       "Binary Classification",
        "metric":     "PR-AUC",
        "model_dir":  "use_case_A",
        "data_dir":   "ieee_fraud",
        "report_dir": "use_case_A",
        "status":     "complete",
        "champion":   "lgbm_optuna_champion.pkl",
    },
    "B": {
        "title":      "Credit Risk Scoring",
        "icon":       "💳",
        "tag":        "Give Me Some Credit",
        "target":     "SeriousDlqin2yrs",
        "task":       "Binary Classification",
        "metric":     "ROC-AUC",
        "model_dir":  "use_case_B",
        "data_dir":   "gmsc_credit",
        "report_dir": "use_case_B",
        "status":     "complete",
        "champion":   "lgbm_optuna_champion.pkl",
    },
    "C_nlp": {
        "title":      "NLP Sentiment Analysis",
        "icon":       "💬",
        "tag":        "Financial Phrasebank",
        "target":     "label",
        "task":       "Multi-class Classification",
        "metric":     "F1 (macro)",
        "model_dir":  "use_case_C_nlp",
        "data_dir":   "financial_phrasebank",
        "report_dir": "use_case_C_nlp",
        "status":     "complete",
        "is_nlp":     True,
        "champion":   "Complement_NB_baseline.pkl",
    },
    "C_markets": {
        "title":      "Market Intelligence — Realized Volatility",
        "icon":       "📈",
        "tag":        "Optiver Realized Volatility",
        "target":     "target",
        "task":       "Regression",
        "metric":     "RMSPE",
        "model_dir":  "use_case_C_markets",
        "data_dir":   "optiver_volatility",
        "report_dir": "use_case_C_markets",
        "status":     "complete",
        "champion":   "champion.pkl",
    },
    "E": {
        "title":      "Insurance Risk Scoring",
        "icon":       "🛡️",
        "tag":        "Porto Seguro Safe Driver",
        "target":     "target",
        "task":       "Binary Classification",
        "metric":     "Normalized Gini",
        "model_dir":  "use_case_E",
        "data_dir":   "porto_seguro",
        "report_dir": "use_case_E",
        "status":     "complete",
        "champion":   "lgbm_optuna_champion.pkl",
    },
    "D": {
        "title":      "Customer Churn Prediction",
        "icon":       "📉",
        "tag":        "KKBox Churn Prediction",
        "target":     "is_churn",
        "task":       "Binary Classification",
        "metric":     "ROC-AUC",
        "model_dir":  "use_case_D",
        "data_dir":   "kkbox_churn",
        "report_dir": "use_case_D",
        "status":     "complete",
        "champion":   "lgbm_optuna_champion.pkl",
    },
    "F": {
        "title":      "ESG & Greenwashing Risk",
        "icon":       "🌱",
        "tag":        "ClimateBERT + Synthetic ESG",
        "target":     "greenwashing_risk",
        "task":       "Multi-class Classification",
        "metric":     "F1 (macro)",
        "model_dir":  "use_case_F",
        "data_dir":   "sec_esg",
        "report_dir": "use_case_F",
        "status":     "complete",
        "champion":   "champion.pkl",
    },
    "B3": {
        "title":      "AmEx Loan Default Prediction",
        "icon":       "🏦",
        "tag":        "Credit Risk — AmEx Dataset",
        "target":     "target",
        "task":       "Binary Classification",
        "metric":     "AmEx Metric",
        "model_dir":  "use_case_G",
        "data_dir":   "amex_default",
        "report_dir": "use_case_G",
        "status":     "complete",
        "champion":   "lgbm_optuna_champion.pkl",
    },
    "G1": {
        "title":      "Robo-Advisory Portfolio Recommendation",
        "icon":       "📊",
        "tag":        "FAR-Trans · LambdaRank",
        "target":     "label",
        "task":       "Learning to Rank",
        "metric":     "NDCG@10",
        "model_dir":  "use_case_G1",
        "data_dir":   "far_trans",
        "report_dir": "use_case_G1",
        "status":     "active",
        "champion":   "lgbm_optuna_champion.pkl",
    },
    "G2": {
        "title":      "Explainable AI for Analysts & Managers",
        "icon":       "🔍",
        "tag":        "SEC EDGAR · SHAP",
        "target":     "outperform",
        "task":       "Binary Classification",
        "metric":     "AUC-ROC",
        "model_dir":  "use_case_G2",
        "data_dir":   "sec_edgar",
        "report_dir": "use_case_G2",
        "status":     "active",
        "champion":   "lgbm_optuna_champion.pkl",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# ── Per-use-case config dicts ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# ── Dataset introductions (Data Studio header card) ───────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_DATASET_INFO: dict = {
    "A": {
        "url":   "https://www.kaggle.com/c/ieee-fraud-detection",
        "label": "Kaggle — IEEE-CIS Fraud Detection",
        "intro": (
            "The **IEEE-CIS Fraud Detection** dataset was released by Vesta Corporation for the 2019 Kaggle "
            "competition. It contains **~590,000 real-world e-commerce transactions** with 394 features "
            "spanning transaction metadata (amount, card type, email domain), device fingerprints, and 339 "
            "anonymised V-features derived from Vesta's proprietary fraud-detection system.  \n"
            "The binary target `isFraud` marks roughly **3.5% of transactions** as fraudulent — a severe "
            "28:1 class imbalance that makes standard accuracy a useless metric. The primary evaluation "
            "metric used in the competition and in this platform is **ROC-AUC**."
        ),
    },
    "B": {
        "url":   "https://www.kaggle.com/c/GiveMeSomeCredit",
        "label": "Kaggle — Give Me Some Credit",
        "intro": (
            "The **Give Me Some Credit** dataset, released by FICO and hosted on Kaggle, contains "
            "**150,000 anonymised U.S. borrowers** described by 11 financial features including revolving "
            "credit utilisation, age, number of open credit lines, delinquency history, debt ratio, and "
            "monthly income.  \n"
            "The binary target `SeriousDlqin2yrs` indicates whether the borrower experienced a "
            "90+ day delinquency within two years — approximately **6.7% of borrowers**, yielding a 14:1 "
            "imbalance. The primary metric is **ROC-AUC**, which mirrors how lenders operationally use "
            "credit scores: to rank-order applicants by risk, not to apply a fixed cut-off."
        ),
    },
    "C_nlp": {
        "url":   "https://huggingface.co/datasets/financial_phrasebank",
        "label": "HuggingFace — Financial PhraseBank",
        "intro": (
            "The **Financial PhraseBank** dataset (Malo et al., 2014) contains **4,840 English sentences** "
            "drawn from financial news articles and annotated by 16 domain experts for sentiment. Each "
            "sentence is labelled **Positive**, **Neutral**, or **Negative** based on its expected effect on "
            "stock price.  \n"
            "This platform uses the `sentences_allagree` split — the highest-quality subset where all "
            "annotators agreed on the label. Class distribution is roughly 37% Positive, 35% Neutral, "
            "28% Negative. The primary metric is **macro-F1**, which weights each class equally "
            "regardless of frequency — important since Negative is the smallest class."
        ),
    },
    "C_markets": {
        "url":   "https://www.kaggle.com/c/optiver-realized-volatility-prediction",
        "label": "Kaggle — Optiver Realized Volatility Prediction",
        "intro": (
            "The **Optiver Realized Volatility Prediction** dataset was published for a 2021 Kaggle "
            "competition by Optiver, a global market-making firm. It contains **high-frequency limit "
            "order book (LOB) snapshots** for 112 NASDAQ-listed stocks across 3,830 10-minute trading "
            "windows.  \n"
            "Each window provides bid/ask prices and sizes at multiple depth levels, alongside trade "
            "flow data. The regression target is **realized volatility** — the root mean squared return "
            "computed from 1-second log-returns in the subsequent 10-minute window. The evaluation "
            "metric is **RMSPE** (Root Mean Squared Percentage Error), which penalises proportional "
            "errors equally regardless of absolute volatility level."
        ),
    },
    "D": {
        "url":   "https://www.kaggle.com/c/kkbox-churn-prediction-challenge",
        "label": "Kaggle — KKBox Music Streaming Churn",
        "intro": (
            "The **KKBox Churn Prediction Challenge** dataset (Kaggle 2017) covers subscribers of KKBox, "
            "Asia's leading music streaming service. The training set contains **~2.4 million subscriber "
            "records** with behavioural features derived from listening logs, transaction history, and "
            "membership metadata.  \n"
            "The binary target `is_churn` indicates whether a subscriber did not renew within 30 days "
            "of expiry — approximately **8.4% churn rate**, yielding an 11:1 imbalance. The primary "
            "metric is **ROC-AUC**. This use case illustrates how engagement signals "
            "(listening depth, skip rate, catalogue breadth) outperform raw demographics in subscription "
            "churn modelling."
        ),
    },
    "E": {
        "url":   "https://www.kaggle.com/c/porto-seguro-safe-driver-prediction",
        "label": "Kaggle — Porto Seguro Safe Driver Prediction",
        "intro": (
            "The **Porto Seguro Safe Driver Prediction** dataset (Kaggle 2017) was provided by Porto "
            "Seguro, one of Brazil's largest auto and homeowner insurers. It contains **~600,000 "
            "anonymised insurance policies** described by 57 features — all obfuscated with categorical "
            "(`_cat`), binary (`_bin`), and continuous prefixes.  \n"
            "The binary target indicates whether the policyholder filed a claim — approximately "
            "**3.6% positive rate**, a 27:1 imbalance. The evaluation metric is the **Normalized Gini "
            "Coefficient** (= 2 × ROC-AUC − 1), which is directly proportional to AUC but scaled to "
            "[0, 1]. The heavy anonymisation makes domain knowledge secondary to systematic feature "
            "interaction search."
        ),
    },
    "F": {
        "url":   "https://huggingface.co/climatebert/distilroberta-base-climate-detector",
        "label": "HuggingFace — ClimateBERT + Synthetic ESG",
        "intro": (
            "The **ESG Greenwashing Risk** dataset combines outputs from **ClimateBERT** "
            "(a RoBERTa model fine-tuned on climate disclosure text) with a synthetic ESG corpus "
            "designed to simulate real-world sustainability reporting patterns. Each record represents "
            "a company disclosure excerpt scored across environmental commitment, regulatory compliance, "
            "and third-party verification dimensions.  \n"
            "The three-class target (`Low`, `Medium`, `High` greenwashing risk) reflects the degree "
            "of gap between stated ESG claims and verifiable evidence. The primary metric is "
            "**macro-F1**. This is a multiclass classification use case — SHAP values are computed "
            "per class and averaged for global feature importance."
        ),
    },
    "G1": {
        "url":   "https://researchdata.gla.ac.uk/1658/",
        "label": "FAR-Trans — Financial Asset Recommendation Dataset",
        "intro": (
            "**FAR-Trans** (University of Glasgow, CC-BY 4.0) is a real-world financial asset "
            "recommendation dataset from a European financial institution covering Jan 2018 – "
            "Nov 2022. It contains four linked tables: **customers** (investor profiles with "
            "risk levels and investment capacity), **assets** (500+ instruments across equities, "
            "bonds, mutual funds, ETFs, commodities, and crypto), **transactions** (buy/sell "
            "records per customer–asset pair), and **profitability** (realised ROI per asset).  \n"
            "The ML task is **learning-to-rank**: given a customer's profile and transaction "
            "history, rank candidate assets by predicted purchase likelihood. Evaluated using "
            "**NDCG@10**, Precision@10, Recall@10, and MRR."
        ),
    },
    "G2": {
        "url":   "https://www.sec.gov/dera/data/financial-statements",
        "label": "SEC EDGAR Financial Statements + Yahoo Finance — Stock Outperformance",
        "intro": (
            "The **G2 dataset** combines SEC EDGAR 10-K/10-Q financial ratio data with Yahoo "
            "Finance forward return data to create an analyst-screening task. Each observation "
            "is a company–fiscal-year pair described by **17 financial ratios** across five "
            "categories: valuation (PE, PB, PS), profitability (ROE, ROA, margins), leverage "
            "(D/E, interest coverage), liquidity (current/quick ratio), and growth (revenue, "
            "EPS, FCF).  \n"
            "The binary target flags companies that **outperformed the S&P 500** over the "
            "following 12 months (top 40% → 1, bottom 40% → 0). Feature engineering adds "
            "cross-sectional percentile ranks, composite factor scores, and macro regime "
            "flags. SHAP explanations provide analyst-readable attribution for every prediction."
        ),
    },
    "B3": {
        "url":   "https://www.kaggle.com/competitions/amex-default-prediction",
        "label": "Kaggle — American Express Default Prediction",
        "intro": (
            "The **AmEx Default Prediction** dataset is one of the largest tabular credit-risk "
            "challenges ever released, containing aggregated monthly customer statement data from "
            "American Express. Each customer is described by **189 features** across five "
            "categories: delinquency variables (D_), spend variables (S_), payment variables (P_), "
            "balance variables (B_), and risk variables (R_), aggregated from up to 13 monthly "
            "snapshots into a single customer-level row.  \n"
            "The binary target flags customers who defaulted within 18 months of the statement "
            "date. The primary metric is the **AmEx metric** (M = 0.5 × (Gini + D-rate@4%)), "
            "which combines overall ranking quality with precision in the highest-risk tier that "
            "credit teams action first."
        ),
    },
}

_PROFILING_SRC: dict = {
    "A": {
        "col_summary": "reports/use_case_A/train_column_summary.csv",
        "raw":         "data/ieee_fraud/train_transaction.parquet",
        "target":      "isFraud",
        "corr_csv":    "reports/use_case_A/feature_target_correlation.csv",
        "corr_png":    "reports/use_case_A/correlation_top30_V_cols.png",
        "missing_png": "reports/use_case_A/missing_heatmap.png",
        "outlier_csv": "reports/use_case_A/outlier_report.csv",
        "target_png":  "reports/use_case_A/target_distribution.png",
    },
    "B": {
        "col_summary": "reports/use_case_B/train_column_summary.csv",
        "raw":         "data/gmsc_credit/cs-training.parquet",
        "target":      "SeriousDlqin2yrs",
        "corr_csv":    "reports/use_case_B/train_column_summary.csv",
        "corr_png":    "reports/use_case_B/correlation_heatmap.png",
        "missing_png": "reports/use_case_B/missing_pattern.png",
        "outlier_csv": None,
        "target_png":  None,
    },
    "C_nlp": {
        "col_summary": None,
        "raw":         "data/financial_phrasebank/sent_train.csv",
        "target":      "label",
        "corr_csv":    None,
        "corr_png":    None,
        "missing_png": None,
        "outlier_csv": None,
        "target_png":  None,
    },
    "C_markets": {
        "col_summary": "reports/use_case_C_markets/train_column_summary.csv",
        "raw":         "data/optiver_volatility/book_train.parquet",
        "target":      "target",
        "corr_csv":    "reports/use_case_C_markets/feature_target_correlation.csv",
        "corr_png":    "reports/use_case_C_markets/correlation_top20.png",
        "missing_png": None,
        "outlier_csv": "reports/use_case_C_markets/outlier_report.csv",
        "target_png":  "reports/use_case_C_markets/eda_target_distribution.png",
    },
    "E": {
        "col_summary": "reports/use_case_E/train_column_summary.csv",
        "raw":         "data/porto_seguro/train.parquet",
        "target":      "target",
        "corr_csv":    "reports/use_case_E/feature_target_correlation.csv",
        "corr_png":    "reports/use_case_E/correlation_top30.png",
        "missing_png": "reports/use_case_E/missing_heatmap.png",
        "outlier_csv": "reports/use_case_E/outlier_report.csv",
        "target_png":  "reports/use_case_E/target_distribution.png",
    },
    "D": {
        "col_summary": "reports/use_case_D/train_column_summary.csv",
        "raw":         "data/kkbox_churn/train_raw.parquet",
        "target":      "is_churn",
        "corr_csv":    "reports/use_case_D/feature_target_correlation.csv",
        "corr_png":    "reports/use_case_D/correlation_heatmap.png",
        "missing_png": "reports/use_case_D/missing_heatmap.png",
        "outlier_csv": "reports/use_case_D/outlier_report.csv",
        "target_png":  "reports/use_case_D/target_distribution.png",
    },
    "F": {
        "col_summary": "reports/use_case_F/eda_summary.csv",
        "raw":         "data/sec_esg/train.parquet",
        "target":      "greenwashing_risk",
        "corr_csv":    None,
        "corr_png":    "reports/use_case_F/correlation_heatmap.png",
        "missing_png": "reports/use_case_F/missing_heatmap.png",
        "outlier_csv": None,
        "target_png":  "reports/use_case_F/target_distribution.png",
    },
    "B3": {
        "col_summary": "reports/use_case_G/train_column_summary.csv",
        "raw":         "data/amex_default/train_raw.parquet",
        "target":      "target",
        "corr_csv":    "reports/use_case_G/feature_target_correlation.csv",
        "corr_png":    "reports/use_case_G/feature_target_correlation.png",
        "missing_png": "reports/use_case_G/missing_by_group.png",
        "outlier_csv": "reports/use_case_G/outlier_report.csv",
        "target_png":  "reports/use_case_G/target_distribution.png",
    },
    "G1": {
        "col_summary": "reports/use_case_G1/eda_summary.csv",
        "raw":         "data/far_trans/train_transactions.parquet",
        "target":      "label",
        "corr_csv":    None,
        "corr_png":    "reports/use_case_G1/interaction_sparsity.png",
        "missing_png": None,
        "outlier_csv": None,
        "target_png":  "reports/use_case_G1/dataset_overview.png",
    },
    "G2": {
        "col_summary": "reports/use_case_G2/train_column_summary.csv",
        "raw":         "data/sec_edgar/train_ratios.parquet",
        "target":      "outperform",
        "corr_csv":    "reports/use_case_G2/feature_target_correlation.csv",
        "corr_png":    "reports/use_case_G2/feature_target_correlation.png",
        "missing_png": None,
        "outlier_csv": "reports/use_case_G2/outlier_report.csv",
        "target_png":  "reports/use_case_G2/target_distribution.png",
    },
}

_FE_EDA_SRC: dict = {
    "A": {
        "train_fe":      "data/ieee_fraud/train_fe.parquet",
        "raw":           "data/ieee_fraud/train_transaction.parquet",
        "feat_list":     "reports/use_case_A/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_A/engineered_feature_summary.png",
        "raw_vs_proc":   "reports/use_case_A/raw_vs_processed_distributions.png",
        "report_dir":    "reports/use_case_A",
        "target":        "isFraud",
        "target_labels": {0: "Legitimate", 1: "Fraud"},
    },
    "B": {
        "train_fe":      "data/gmsc_credit/train_fe.parquet",
        "raw":           "data/gmsc_credit/cs-training.parquet",
        "feat_list":     None,
        "fe_summary":    "reports/use_case_B/engineered_feature_summary.png",
        "raw_vs_proc":   "reports/use_case_B/raw_vs_processed_distributions.png",
        "report_dir":    "reports/use_case_B",
        "target":        "SeriousDlqin2yrs",
        "target_labels": {0: "No Default", 1: "Default"},
    },
    "C_nlp": {
        "train_fe":      None,
        "raw":           "data/financial_phrasebank/sent_train.csv",
        "feat_list":     None,
        "fe_summary":    "reports/use_case_C_nlp/engineered_feature_summary.png",
        "raw_vs_proc":   "reports/use_case_C_nlp/raw_vs_processed_distributions.png",
        "report_dir":    "reports/use_case_C_nlp",
        "target":        "label",
        "target_labels": {0: "Bearish", 1: "Neutral", 2: "Bullish"},
    },
    "C_markets": {
        "train_fe":      "data/optiver_volatility/test_fe.parquet",
        "raw":           "data/optiver_volatility/book_train.parquet",
        "feat_list":     "reports/use_case_C_markets/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_C_markets/engineered_feature_summary.png",
        "raw_vs_proc":   "reports/use_case_C_markets/raw_vs_processed_distributions.png",
        "report_dir":    "reports/use_case_C_markets",
        "target":        "target",
        "target_labels": {},
    },
    "E": {
        "train_fe":      "data/porto_seguro/train_fe.parquet",
        "raw":           "data/porto_seguro/train.parquet",
        "feat_list":     "reports/use_case_E/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_E/engineered_feature_summary.png",
        "raw_vs_proc":   "reports/use_case_E/raw_vs_processed_distributions.png",
        "report_dir":    "reports/use_case_E",
        "target":        "target",
        "target_labels": {0: "No Claim", 1: "Claim"},
    },
    "D": {
        "train_fe":      "data/kkbox_churn/train_fe.parquet",
        "raw":           "data/kkbox_churn/train_raw.parquet",
        "feat_list":     "reports/use_case_D/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_D/engineered_feature_summary.png",
        "raw_vs_proc":   "reports/use_case_D/raw_vs_processed_distributions.png",
        "report_dir":    "reports/use_case_D",
        "target":        "is_churn",
        "target_labels": {0: "Retained", 1: "Churned"},
    },
    "F": {
        "train_fe":      "data/sec_esg/train_fe.parquet",
        "raw":           "data/sec_esg/train.parquet",
        "feat_list":     "data/sec_esg/feature_list.csv",
        "fe_summary":    "reports/use_case_F/engineered_feature_summary.png",
        "raw_vs_proc":   "reports/use_case_F/raw_vs_processed_distributions.png",
        "report_dir":    "reports/use_case_F",
        "target":        "greenwashing_risk",
        "target_labels": {"Low": "Low Risk", "Medium": "Medium Risk", "High": "High Risk"},
    },
    "B3": {
        "train_fe":      "data/amex_default/train_fe.parquet",
        "raw":           "data/amex_default/train_raw.parquet",
        "feat_list":     "reports/use_case_G/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_G/engineered_feature_summary.png",
        "raw_vs_proc":   "reports/use_case_G/raw_vs_processed_distributions.png",
        "report_dir":    "reports/use_case_G",
        "target":        "target",
        "target_labels": {0: "No Default", 1: "Default"},
    },
    "G1": {
        "train_fe":      "data/far_trans/train_pairs.parquet",
        "raw":           "data/far_trans/train_transactions.parquet",
        "feat_list":     "reports/use_case_G1/data_dictionary.csv",
        "fe_summary":    "reports/use_case_G1/feature_engineering_summary.png",
        "raw_vs_proc":   "reports/use_case_G1/temporal_patterns.png",
        "report_dir":    "reports/use_case_G1",
        "target":        "label",
        "target_labels": {0: "Not Purchased", 1: "Purchased"},
    },
    "G2": {
        "train_fe":      "data/sec_edgar/train_fe.parquet",
        "raw":           "data/sec_edgar/train_ratios.parquet",
        "feat_list":     "reports/use_case_G2/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_G2/engineered_feature_summary.png",
        "raw_vs_proc":   "reports/use_case_G2/ratio_distributions.png",
        "report_dir":    "reports/use_case_G2",
        "target":        "outperform",
        "target_labels": {0: "Under-perform", 1: "Outperform"},
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# ── Feature Glossary — per-UC business meanings for non-obvious features ──────
# ══════════════════════════════════════════════════════════════════════════════

_FEATURE_GLOSSARY: dict = {
    # ── UC-A : IEEE-CIS Fraud Detection ────────────────────────────────────────
    "A": {
        # Transaction metadata
        "TransactionAmt":      "Transaction amount (USD)",
        "TransactionDT":       "Seconds elapsed since reference date (not a calendar date)",
        "ProductCD":           "Product category code: W=web, H=hotel, R=??, C=??, S=??",
        "card1":               "Payment card primary group number",
        "card2":               "Payment card secondary number",
        "card3":               "Card issuer country code",
        "card4":               "Card network (Visa, Mastercard, etc.)",
        "card5":               "Card sub-type group",
        "card6":               "Card type: debit / credit",
        "addr1":               "Billing address ZIP/postal code",
        "addr2":               "Billing address country code",
        "dist1":               "Distance between billing and shipping address",
        "dist2":               "Distance to secondary address",
        "P_emaildomain":       "Purchaser email domain",
        "R_emaildomain":       "Recipient email domain",
        # D-features (timedelta)
        "D1": "Days since first transaction on this card",
        "D2": "Days since card was first seen in dataset",
        "D3": "Days since last address change",
        "D4": "Days since last transaction on this card",
        "D10": "Days since last browser/device change",
        "D15": "Days since last high-value transaction",
        # M-features (match flags, already numeric)
        "M1": "Name match flag (billing vs. card)",
        "M2": "Address match flag",
        "M3": "Address2 match flag",
        "M4": "Email match flag",
        "M5": "Phone match flag",
        "M6": "Billing zip match",
        "M7": "P-email match flag",
        "M8": "R-email match flag",
        "M9": "Browser cookie match",
        # Engineered — time
        "fe_hour":             "Hour of day the transaction occurred (0–23)",
        "fe_day_of_week":      "Day of week (0=Mon … 6=Sun)",
        "fe_is_weekend":       "Binary: transaction on Saturday or Sunday",
        "fe_is_nighttime":     "Binary: transaction between 22:00 and 06:00",
        "fe_hour_sin":         "Cyclical sine encoding of hour (removes 23→0 discontinuity)",
        "fe_hour_cos":         "Cyclical cosine encoding of hour",
        "fe_dow_sin":          "Cyclical sine encoding of day-of-week",
        "fe_dow_cos":          "Cyclical cosine encoding of day-of-week",
        "fe_days_since_start": "Days elapsed since the earliest transaction in the dataset",
        # Engineered — card velocity
        "fe_card1_txn_count":  "Running count of transactions on this card (up to current row)",
        "fe_card1_cum_amount": "Cumulative spend on this card (rolling)",
        "fe_card1_txn_freq":   "Transaction frequency: count / days since first card transaction",
        "fe_card1_mean_amt":   "Historical mean transaction amount for this card",
        "fe_card1_std_amt":    "Historical std deviation of transaction amounts for this card",
        "fe_card1_n_txn":      "Total number of historical transactions on this card",
        "fe_card1_mean_hour":  "Mean hour of day for prior transactions on this card",
        "fe_addr_txn_count":   "Number of transactions from this billing address",
        "fe_email_txn_count":  "Number of transactions from this purchaser email domain",
        # Engineered — amount
        "fe_amt_z_score":      "Z-score of this amount vs. card's historical amount distribution",
        "fe_amt_above_mean":   "Binary: amount is above this card's historical mean",
        "fe_log_amount":       "Natural log of TransactionAmt (reduces right skew)",
        "fe_amount_cents":     "Cents portion of the transaction amount (e.g. 0.99 → 99)",
        "fe_amount_is_round":  "Binary: transaction amount is a round number (zero cents; potential test/synthetic charge signal)",
        "fe_amount_decile":    "Decile bin of amount within the training distribution (0–9)",
        "fe_amount_x_product": "Interaction: log-amount × product code (numeric)",
        # Engineered — email
        "fe_P_email_fraud_rate": "Train-set fraud rate for this purchaser email domain",
        "fe_R_email_fraud_rate": "Train-set fraud rate for this recipient email domain",
        "fe_P_email_is_free":    "Binary: purchaser uses a free email provider (Gmail, Yahoo…)",
        "fe_P_email_is_rare":    "Binary: purchaser email domain appears < 10 times in training",
        "fe_email_domain_match": "Binary: purchaser and recipient share the same email domain",
        # Engineered — match score
        "fe_m1": "Numeric encoding of M1 match flag (True=1, False=0, NaN=−1)",
        "fe_m2": "Numeric encoding of M2", "fe_m3": "Numeric encoding of M3",
        "fe_m4": "Numeric encoding of M4", "fe_m5": "Numeric encoding of M5",
        "fe_m6": "Numeric encoding of M6", "fe_m7": "Numeric encoding of M7",
        "fe_m8": "Numeric encoding of M8", "fe_m9": "Numeric encoding of M9",
        "fe_match_score":      "Count of M1–M9 flags that are True (0–9)",
        # Engineered — PCA of V-features
        **{f"fe_V_pca_{i:02d}": f"PCA component {i} of Vesta anonymised V-features (V1–V339)"
           for i in range(50)},
    },

    # ── UC-B : Credit Risk (Give Me Some Credit) ────────────────────────────────
    "B": {
        "SeriousDlqin2yrs":             "Target: 90+ day delinquency in next 2 years (1=yes)",
        "RevolvingUtilizationOfUnsecuredLines": "Total balance / total credit limit across cards",
        "age":                          "Borrower age in years",
        "NumberOfTime30-59DaysPastDueNotWorse": "Times 30–59 days late on any account",
        "DebtRatio":                    "Monthly debt payments / monthly gross income",
        "MonthlyIncome":                "Borrower monthly gross income (USD)",
        "NumberOfOpenCreditLinesAndLoans": "Open credit lines + installment loans",
        "NumberOfTimes90DaysLate":      "Times 90+ days late on any account",
        "NumberRealEstateLoansOrLines": "Number of mortgage and real estate loans",
        "NumberOfTime60-89DaysPastDueNotWorse": "Times 60–89 days late on any account",
        "NumberOfDependents":           "Number of dependents in the household",
    },

    # ── UC-D : KKBox Customer Churn ─────────────────────────────────────────────
    "D": {
        "city":               "User city code (categorical, encoded numerically)",
        "bd":                 "User age in years (derived from birth date; −1 if unknown)",
        "registered_via":     "Registration channel (3=iOS, 4=Android, 7=web, 9=other)",
        "txn_count":          "Total number of subscription transactions in history",
        "plan_days_mean":     "Average subscription plan length in days",
        "plan_price_mean":    "Average subscription list price paid (NTD)",
        "actual_paid_mean":   "Average actual payment amount (NTD; may be 0 for free trials)",
        "auto_renew_rate":    "Fraction of subscriptions where auto-renew was enabled",
        "cancel_rate":        "Fraction of transactions that resulted in cancellation",
        "discount_rate":      "Fraction of transactions with a discount applied",
        "log_days":           "Log of total membership tenure in days",
        "num_25_mean":        "Avg songs played to ≥25% completion per log period",
        "num_50_mean":        "Avg songs played to ≥50% completion per log period",
        "num_75_mean":        "Avg songs played to ≥75% completion per log period",
        "num_985_mean":       "Avg songs played to ≥98.5% completion per log period",
        "num_100_mean":       "Avg songs played to 100% completion per log period",
        "num_unq_mean":       "Avg unique songs played per log period (variety measure)",
        "total_secs_mean":    "Average total listening seconds per log period",
        "total_secs_sum":     "Total listening seconds across all log periods",
        "fe_completion_rate": "Mean fraction of each song completed (depth of engagement)",
        "fe_skip_rate":       "Fraction of songs abandoned before 25% (low-engagement signal)",
        "fe_deep_listen_rate":"Fraction of songs played ≥75% to completion (high-engagement signal)",
        "fe_variety_ratio":   "Unique songs / total plays — measures listening breadth",
        "fe_log_secs":        "Log of total listening seconds (reduces right skew)",
        "fe_log_days_log":    "Log of log-membership-days (double-log for heavy-tailed tenure)",
        "fe_secs_per_song":   "Average seconds spent per song play",
        "fe_is_long_plan":    "Binary: most recent plan ≥ 30 days",
        "fe_plan_days_log":   "Log of mean plan duration in days",
        "fe_txn_count_log":   "Log of total transaction count",
        "fe_auto_renew":      "Binary: auto-renew was enabled on the last subscription",
        "fe_cancel_rate":     "Ratio: cancellations / total transactions (churn history signal)",
        "fe_age":             "User age in years (cleaned; 0 and negatives set to NaN)",
        "fe_age_bucket_young":"Binary: user age < 25",
        "fe_age_bucket_senior":"Binary: user age > 55",
        "fe_is_male":         "Binary: gender = male",
        "fe_is_female":       "Binary: gender = female",
        "fe_city_risk":       "Target-encoded city churn rate (smoothed mean encoding)",
        "fe_reg_channel_risk":"Target-encoded churn rate for the user's registration channel",
    },

    # ── UC-E : Porto Seguro Insurance Risk ──────────────────────────────────────
    "E": {
        "fe_miss_ps_reg_03":      "Missing indicator: ps_reg_03 (continuous driving record metric)",
        "fe_miss_ps_car_03_cat":  "Missing indicator: ps_car_03_cat (vehicle model category)",
        "fe_miss_ps_car_05_cat":  "Missing indicator: ps_car_05_cat (vehicle brand code)",
        "fe_miss_ps_car_07_cat":  "Missing indicator: ps_car_07_cat (car body type)",
        "fe_miss_ps_car_14":      "Missing indicator: ps_car_14 (vehicle age proxy)",
        "fe_te_ps_car_01_cat":    "Target-encoded: ps_car_01_cat (primary vehicle model)",
        "fe_te_ps_car_06_cat":    "Target-encoded: ps_car_06_cat (fuel type)",
        "fe_te_ps_car_11_cat":    "Target-encoded: ps_car_11_cat (secondary vehicle model)",
        "fe_n_missing":           "Total count of missing values for this policyholder",
        "fe_n_bin_ind":           "Count of binary individual (ps_ind_*_bin) features equal to 1",
        "fe_reg_sum":             "Sum of all three ps_reg_* continuous features",
        "fe_ind_cont_sum":        "Sum of continuous individual (ps_ind_01/03/14/15) features",
        "fe_car13_reg03":         "Interaction: ps_car_13 × ps_reg_03 (vehicle power × driving record)",
        "fe_miss_x_reg":          "Interaction: missing count × registration sum",
        "fe_car13_sq":            "Squared ps_car_13 (non-linear vehicle power proxy)",
        "fe_log_ps_reg_01":       "Log of ps_reg_01 (registration metric 01)",
        "fe_log_ps_reg_02":       "Log of ps_reg_02 (registration metric 02)",
        "fe_log_ps_reg_03":       "Log of ps_reg_03 (registration metric 03)",
        "fe_log_ps_car_12":       "Log of ps_car_12 (vehicle feature 12)",
        "fe_log_ps_car_13":       "Log of ps_car_13 (vehicle feature 13, engine/power proxy)",
    },

    # ── UC-F : ESG & Greenwashing Risk ──────────────────────────────────────────
    "F": {
        "fe_text_len":          "Character count of the company's ESG disclosure text",
        "fe_word_count":        "Word count of the ESG disclosure text",
        "fe_avg_word_len":      "Average word length in the disclosure (complexity proxy)",
        "fe_claim_density":     "ESG claim keywords per word (sustainability rhetoric density)",
        "fe_e_gap_clipped":     "Environmental (E) gap: self-reported minus third-party E score",
        "fe_s_gap_clipped":     "Social (S) gap: self-reported minus third-party S score",
        "fe_g_gap_clipped":     "Governance (G) gap: self-reported minus third-party G score",
        "fe_avg_gap_clipped":   "Mean across E/S/G gaps — measures overall greenwashing breadth",
        "fe_gap_cv":            "Coeff. of variation of E/S/G gaps — signals selective exaggeration",
        "fe_max_gap":           "Largest gap across any single E, S, or G dimension",
        "fe_composite_delta":   "YoY change in composite ESG score (improvement or decline)",
        "fe_composite_esg":     "Weighted composite ESG score (E×0.4 + S×0.3 + G×0.3)",
        "fe_esg_low":           "Binary: composite ESG score in the bottom quartile",
        "fe_esg_high":          "Binary: composite ESG score in the top quartile",
        "fe_e_score_norm":      "Normalised Environmental score (0 = worst, 1 = best in cohort)",
        "fe_s_score_norm":      "Normalised Social score",
        "fe_g_score_norm":      "Normalised Governance score",
        "fe_log_market_cap":    "Log of market capitalisation (USD)",
        "fe_log_revenue":       "Log of annual revenue (USD)",
        "fe_log_emissions":     "Log of GHG emissions intensity (tCO₂e per USD revenue)",
        "fe_emissions_high":    "Binary: emissions intensity in the top quartile of the cohort",
        "fe_sector_risk_te":    "Target-encoded greenwashing rate for this GICS sector",
        "fe_claim_x_gap":       "Interaction: claim_density × avg_gap (high rhetoric + high gap = red flag)",
        "fe_gap_x_emissions":   "Interaction: avg_gap × log_emissions (gap aggravated by heavy emitters)",
        **{f"fe_sector_{s}": f"One-hot: GICS sector = {s}"
           for s in ["Communication Services","Consumer Discretionary","Consumer Staples",
                     "Energy","Financials","Materials","Other","Real Estate","Utilities"]},
    },

    # ── UC-B3 : AmEx Loan Default ────────────────────────────────────────────────
    "B3": {
        "fe_stmt_count":       "Number of monthly billing statements in customer history",
        "fe_last_miss_count":  "Missing-value count in the most recent statement",
        "fe_all_null_count":   "Features that are null across ALL statements (always-missing signal)",
        "fe_nuniq_D_39":       "Unique values of D_39 across statements (behavioural variability)",
        "fe_nuniq_D_41":       "Unique values of D_41 across statements",
        "fe_nuniq_D_42":       "Unique values of D_42 across statements",
        "fe_nuniq_D_44":       "Unique values of D_44 across statements",
        "fe_nuniq_D_45":       "Unique values of D_45 across statements",
        # AmEx anonymised feature groups
        **{f"D_{i}__mean": f"Mean of delinquency feature D_{i} across all statements"
           for i in range(200)},
        **{f"D_{i}__std":  f"Std deviation of delinquency feature D_{i} across statements"
           for i in range(200)},
        **{f"D_{i}__min":  f"Minimum of delinquency feature D_{i} across statements"
           for i in range(200)},
        **{f"D_{i}__max":  f"Maximum of delinquency feature D_{i} across statements"
           for i in range(200)},
        **{f"D_{i}__last": f"Most-recent value of delinquency feature D_{i}"
           for i in range(200)},
        **{f"S_{i}__mean": f"Mean of spend feature S_{i} across all statements"
           for i in range(30)},
        **{f"P_{i}__mean": f"Mean of payment feature P_{i} across all statements"
           for i in range(20)},
        **{f"B_{i}__mean": f"Mean of balance feature B_{i} across all statements"
           for i in range(50)},
        **{f"R_{i}__mean": f"Mean of risk feature R_{i} across all statements"
           for i in range(30)},
    },

    # ── UC-G2 : Explainable AI — SEC EDGAR ──────────────────────────────────────
    "G2": {
        "market_cap_log":       "Log of market capitalisation (USD) — size control variable",
        "pe_ratio":             "Price-to-earnings ratio: stock price / trailing-twelve-month EPS",
        "pb_ratio":             "Price-to-book ratio: market value / book (shareholders' equity) value",
        "ps_ratio":             "Price-to-sales ratio: market cap / annual revenue",
        "roe":                  "Return on equity: net income / average shareholders' equity",
        "roa":                  "Return on assets: net income / average total assets",
        "net_margin":           "Net profit margin: net income / revenue",
        "gross_margin":         "Gross profit margin: (revenue − COGS) / revenue",
        "ebitda_margin":        "EBITDA / revenue — operating profitability before financing",
        "debt_equity":          "Total debt / shareholders' equity (leverage ratio)",
        "interest_coverage":    "EBIT / interest expense — debt-service capacity",
        "debt_assets":          "Total debt / total assets (balance-sheet leverage)",
        "current_ratio":        "Current assets / current liabilities — short-term liquidity",
        "quick_ratio":          "(Current assets − inventory) / current liabilities",
        "asset_turnover":       "Revenue / average total assets — capital efficiency",
        "revenue_growth":       "Year-over-year revenue growth rate",
        "eps_growth":           "Year-over-year earnings-per-share growth rate",
        "fcf_yield":            "Free cash flow / market cap — intrinsic value signal",
        "sector_enc":           "Historical outperformance rate for this GICS sector (target-encoded)",
        "peg_ratio":            "P/E ÷ EPS growth rate — growth-adjusted valuation",
        "interest_burden":      "Net interest expense / operating income — financing cost fraction",
        "quality_spread":       "Composite quality score minus sector-year average (relative quality)",
        "value_composite":      "Z-score composite of P/E, P/B, P/S (lower = cheaper)",
        "growth_composite":     "Z-score composite of revenue growth, EPS growth",
        "profitability_composite": "Z-score composite of ROE, ROA, net margin",
        "leverage_risk":        "Composite leverage risk score (debt/equity, interest coverage)",
        "macro_regime":         "Macroeconomic regime code: 0=contraction, 1=recovery, 2=expansion, 3=slowdown",
        "is_crisis_year":       "Binary: year classified as financial crisis (2001, 2008–09, 2020)",
        "is_bull_year":         "Binary: year classified as strong broad-market bull run",
        **{f"{m}__rank": f"Percentile rank of {m} within same sector × year peer group (0=worst, 1=best)"
           for m in ["pe_ratio","pb_ratio","ps_ratio","roe","roa","net_margin","gross_margin",
                     "ebitda_margin","debt_equity","interest_coverage","debt_assets",
                     "current_ratio","quick_ratio","asset_turnover","revenue_growth",
                     "eps_growth","fcf_yield","market_cap_log","peg_ratio","quality_spread",
                     "growth_composite","profitability_composite"]},
    },
}


def _describe_feature(feat: str, uc_key: str) -> str:
    """Return a human-readable description for an engineered feature name.

    Resolution order:
    1. Per-UC glossary exact match
    2. Pattern matching on naming conventions
    3. Cleaned-up title-case fallback
    """
    # 1. Glossary exact match
    desc = _FEATURE_GLOSSARY.get(uc_key, {}).get(feat)
    if desc:
        return desc

    f = feat  # preserve original case for display; use f.lower() for matching
    fl = feat.lower()

    # 2. Pattern matching ─────────────────────────────────────────────────────

    # Missing-value indicators
    if fl.startswith("fe_miss_"):
        base = feat[8:].replace("_", " ")
        return f"Missing-value indicator for {base}"

    # TF-IDF term weights
    if fl.startswith("tfidf_"):
        term = feat[6:].replace("_", " ")
        return f'TF-IDF weight for term "{term}" in ESG disclosure text'

    # One-hot sector
    if fl.startswith("fe_sector_"):
        sector = feat[10:].replace("_", " ")
        return f"One-hot encoded: GICS sector = {sector}"

    # Target-encoded
    if fl.startswith("fe_te_"):
        base = feat[6:].replace("_", " ")
        return f"Target-encoded (smoothed mean): {base}"

    # AmEx anonymised aggregation stats  e.g.  D_39__mean, S_3__last
    for stat in ("__mean", "__std", "__min", "__max", "__last"):
        if fl.endswith(stat):
            base = feat[: -len(stat)]
            stat_name = stat[2:]
            grp = base[0] if base else "?"
            grp_names = {"D": "delinquency", "S": "spend", "P": "payment",
                         "B": "balance", "R": "risk"}
            grp_label = grp_names.get(grp.upper(), "")
            label = f"{grp_label} feature {base}" if grp_label else base
            _stat_labels = {"mean": "Mean", "std": "Standard deviation",
                            "min": "Minimum", "max": "Maximum"}
            stat_label = _stat_labels.get(stat_name, stat_name.capitalize())
            return (f"Most-recent value of {label}"
                    if stat_name == "last"
                    else f"{stat_label} of {label} across all statements")

    # Percentile rank columns  e.g.  pe_ratio__rank
    if fl.endswith("__rank"):
        base = feat[: -6].replace("_", " ")
        return f"Percentile rank of {base} within same sector × year peer group"

    # Log transforms
    if fl.startswith("fe_log_"):
        base = feat[7:].replace("_", " ")
        return f"Natural log of {base} (variance-stabilising transform)"
    if fl.endswith("_log") and fl.startswith("fe_"):
        base = feat[3:-4].replace("_", " ")
        return f"Natural log of {base}"

    # Binary age buckets
    if fl.startswith("fe_age_bucket_"):
        bucket = feat[14:].replace("_", " ")
        return f"Binary indicator: user is in the '{bucket}' age group"

    # Binary gender
    if fl == "fe_is_male":
        return "Binary: user gender = male"
    if fl == "fe_is_female":
        return "Binary: user gender = female"

    # Generic fe_is_ indicators
    if fl.startswith("fe_is_"):
        label = feat[6:].replace("_", " ")
        return f"Binary indicator: {label}"

    # PCA components
    if fl.startswith("fe_v_pca_") or fl.startswith("fe_V_pca_"):
        idx = feat.split("_")[-1]
        return f"PCA component {idx} of Vesta anonymised V-features (V1–V339)"

    # Generic fe_ prefix
    if fl.startswith("fe_"):
        label = feat[3:].replace("_", " ")
        return f"Engineered feature: {label}"

    # 3. Cleaned title-case fallback
    return feat.replace("_", " ").title()


# ══════════════════════════════════════════════════════════════════════════════
# ── EDA-based feature engineering recommendations (Data Preparation tab) ──────
# ══════════════════════════════════════════════════════════════════════════════

_EDA_RECOMMENDATIONS: dict = {
    "A": [
        "**Class imbalance (3.5% fraud, 28:1 ratio):** Apply SMOTE *only inside training folds*. "
        "Never oversample before the train/val split — synthetic fraud samples in the validation "
        "set produce optimistically biased AUC estimates.",
        "**V-feature sparsity:** For V-features with >80% missingness, the binary missing indicator "
        "(`fe_miss_*`) is more predictive than any imputed value. Prioritise creating flags before "
        "imputing, especially for sub-groups V35–V52.",
        "**High-cardinality categoricals:** `card1` (~18K unique values), `addr1` (~4K), and "
        "email domains are too high-cardinality for one-hot encoding. Use **frequency encoding** — "
        "replace each category with its training-set occurrence count. This captures rarity as a "
        "continuous anomaly signal without a dimensionality explosion.",
        "**Amount non-linearity:** Raw `TransactionAmt` spans 5 orders of magnitude ($0.25–$31,937). "
        "Apply `log1p` to compress the tail. Also engineer `amt / card1_mean_amt` — the ratio of "
        "a transaction to the card's history is far more informative than the absolute amount.",
        "**Temporal patterns:** Extract hour-of-day and day-of-week from `TransactionDT` "
        "(a timedelta in seconds, not a Unix timestamp). Late-night (10pm–6am) and weekend flags "
        "capture well-documented fraudster behavioural patterns.",
    ],
    "B": [
        "**Class imbalance (6.7%, 14:1 ratio):** Use `scale_pos_weight=14` in XGBoost and "
        "`is_unbalance=True` in LightGBM. Supplement with SMOTE inside CV folds for Logistic "
        "Regression, which does not natively handle imbalance.",
        "**Delinquency aggregation:** The six raw past-due count columns are individually noisy "
        "but collectively informative. Create a severity-weighted sum (90-day events × 3, "
        "60-day × 2, 30-day × 1) to compress them into one strong signal aligned with FICO's "
        "own credit-risk hierarchy.",
        "**RevolvingUtilization outliers:** Values above 1.0 (sometimes up to 50,892) are "
        "over-limit accounts, not data errors. Clip to [0.0, 1.5] before engineering. Apply a "
        "squared transform (`util²`) to capture the accelerating default risk above 80% utilisation.",
        "**MonthlyIncome missingness (19.8% MAR):** Impute with training-set median — not mean, "
        "which is distorted by the high-income right tail. Add a binary `fe_income_missing` flag, "
        "which is itself predictive (higher-earners self-select into not disclosing income).",
        "**Delinquency artefacts (values 96 and 98):** These are credit bureau special codes, "
        "not literal delinquency counts. Recode values ≥ 90 to a capped value and add a "
        "`fe_extreme_dpd` flag before any numeric processing.",
    ],
    "C_nlp": [
        "**TF-IDF dimensionality:** Use a `max_features` cap (e.g., 3,000–5,000 unigrams) to "
        "prevent the vocabulary from overwhelming the tree models. Financial language is "
        "domain-constrained — most sentiment signal lives in a small core vocabulary.",
        "**FinBERT scores dominate:** FinBERT's `positive`, `neutral`, and `negative` soft "
        "probabilities are the single strongest feature group. Engineer the **sentiment margin** "
        "= `finbert_positive − finbert_negative` as a single signed score for binary sentiment "
        "separability.",
        "**Near-balanced classes:** Macro-F1 is the right metric here, not accuracy. Still apply "
        "class weights (`class_weight='balanced'`) in Logistic Regression to give the minority "
        "Negative class equal gradient influence.",
        "**Label noise ceiling:** Inter-annotator agreement in FinPhrasebank is ~75%. A model "
        "achieving F1 > 0.80 is already approaching the human-agreement ceiling — don't over-tune "
        "expecting further gains.",
        "**Text statistics as auxiliary features:** `word_count`, `char_count`, and `avg_word_len` "
        "add signal orthogonal to TF-IDF. Short, punchy sentences tend to be more extreme in "
        "sentiment; long, qualified sentences tend to be neutral.",
    ],
    "C_markets": [
        "**Log-price returns:** Compute `log_return = ln(P_t / P_{t-1})` from WAP (Weighted "
        "Average Price) at each second. Raw price levels are non-stationary — returns are "
        "stationary and directly comparable across stocks and time windows.",
        "**Order book imbalance:** Bid/ask size imbalance = `(bid_size − ask_size) / "
        "(bid_size + ask_size)` captures supply/demand pressure. Persistent imbalance in one "
        "direction predicts near-term price movement and hence volatility.",
        "**Realized volatility features:** Engineer volatility of volatility (vol-of-vol) from "
        "rolling realized vol estimates across sub-windows. High vol-of-vol regimes (market "
        "stress) warrant different model behaviour than low vol-of-vol (calm markets).",
        "**Stock-level normalisation:** Raw volatility levels vary 5–10× across stocks. "
        "Normalise features by each stock's trailing average (z-score within stock × time) "
        "before training a single cross-sectional model.",
        "**RMSPE sensitivity:** RMSPE penalises proportional errors — a 10% error on a low-vol "
        "stock costs as much as a 10% error on a high-vol stock. Log-transforming the target "
        "converts RMSPE minimisation to MSE minimisation, enabling standard gradient boosting.",
    ],
    "D": [
        "**Auto-renewal as the strongest signal:** EDA shows `auto_renew_rate` has the strongest "
        "negative correlation with churn. Engineer interaction features between auto-renewal "
        "and plan type — auto-renew on a discounted plan is a weaker retention signal than "
        "auto-renew on a full-price plan.",
        "**Engagement rate features:** Raw listen counts (`num_25`, `num_100`) are scale-dependent "
        "— a user with 1,000 plays of 25% completions is different from one with 10 plays. "
        "Convert to **engagement rates** bounded [0, 1]: `completion_rate = num_100 / "
        "(num_25 + num_50 + num_75 + num_985 + num_100 + 1)`.",
        "**Age outliers and MNAR missingness:** `bd` (age) has 34% missing and values of 0, "
        "negatives, and 200+. Hard-clip to [7, 80], impute with training median, and add "
        "`fe_age_missing` flag. Younger users (<25) have higher churn — add a `fe_young_user` "
        "binary feature.",
        "**Discount-driven churn signal:** Users acquired via promotions churn when the "
        "promotion ends. Engineer `fe_is_discounted` from `plan_list_price > actual_amount` "
        "and interact it with `plan_days_mean` — short-plan discounted subscribers are the "
        "highest-risk subgroup.",
        "**Class imbalance (8.4%, 11:1):** Apply SMOTE inside CV folds. LightGBM's "
        "`is_unbalance=True` handles gradient weighting at the split level. Monitor "
        "precision-recall trade-off at inference time — the business cost of missing a "
        "churner differs from the cost of a false retention alert.",
    ],
    "E": [
        "**Full feature anonymisation:** All 57 features are obfuscated (`ps_car_*`, "
        "`ps_ind_*`, `ps_reg_*`, `ps_calc_*`). The `_calc_` prefix features are computed "
        "from other features and should be dropped or carefully examined for leakage "
        "— Porto Seguro stated post-competition that calc features added little signal.",
        "**Binary and categorical encoding:** `_bin` suffix features are already binary. "
        "`_cat` suffix features have a −1 sentinel for missing — encode −1 explicitly "
        "rather than imputing, as the absence of a category value may be informative.",
        "**Severe class imbalance (3.6%, 27:1):** The Normalized Gini metric rewards AUC "
        "improvement across all thresholds. Use `scale_pos_weight ≈ 27` in XGBoost. "
        "Avoid optimising accuracy — a model predicting all-zero achieves 96.4% accuracy "
        "with zero business value.",
        "**Pairwise feature interactions:** With 57 anonymous features, manual domain-driven "
        "engineering is not possible. Instead, engineer second-order interactions for the "
        "top-10 SHAP features identified in Step 2 EDA — products and ratios between "
        "high-importance feature pairs often add measurable lift.",
        "**Missing value encoding:** Features with −1 sentinels should have a binary "
        "`fe_miss_<feature>` flag added before replacing −1 with 0 or the training median. "
        "The missingness pattern itself may be correlated with claim probability.",
    ],
    "F": [
        "**Three-class target:** Greenwashing risk is `Low / Medium / High`. Use "
        "`class_weight='balanced'` and optimise **macro-F1**. The Medium class is hardest "
        "to separate — it sits between concrete commitments (Low) and clear gaps (High).",
        "**ClimateBERT scores as primary features:** The pre-trained ClimateBERT embeddings "
        "already encode domain knowledge about climate disclosure language. Treat them as "
        "fixed features rather than fine-tuning — the synthetic dataset is too small for "
        "full fine-tuning without overfitting.",
        "**Regulatory gap features:** Engineer `fe_has_target` (mentions specific emission "
        "reduction targets), `fe_has_timeline` (mentions specific dates), and "
        "`fe_has_third_party` (mentions external audit or certification) as binary signals "
        "for commitment verifiability — the absence of these is a greenwashing indicator.",
        "**Multiclass SHAP interpretation:** LightGBM's `predict` returns a 3-D SHAP "
        "array `(n_samples, n_features, n_classes)`. Slice per class before plotting: "
        "`[sv[:,:,k] for k in range(3)]`. Report both per-class and mean |SHAP| importance.",
        "**Text length as a signal:** Longer disclosures with more specific language "
        "correlate with Low greenwashing risk. Short, vague disclosures with high ESG "
        "buzzword density are a High-risk pattern. Add `word_count` and "
        "`buzzword_density` as auxiliary features.",
    ],
    "B3": [
        "**Temporal aggregation across monthly statements (AmEx Loan Default):** Each customer has up to 13 monthly snapshots. Aggregate to customer level with mean, std, min, max, last, and diff features. Trend direction over the last 3 months is often more predictive than absolute levels.",
        "**AmEx custom metric:** M = 0.5 × (Gini + D-rate@4%). Optimise for ranking quality AND precision in the top 4% risk tier — the segment credit risk teams act on first.",
        "**Denoise preprocessing:** Apply np.floor(x×100)/100 before all feature engineering to eliminate floating-point noise that inflates diff and rank features.",
    ],
    "G1": [
        "**Candidate pair generation with negative sampling:** LambdaRank requires "
        "a query–document structure. Each customer is a query; candidate assets are "
        "documents. Positive labels come from actual purchases; generate 4× random "
        "negatives per positive from unobserved assets. The 4:1 ratio is a standard "
        "trade-off between dataset size and training signal density.",
        "**User behavioural aggregates matter more than profile:** While risk level "
        "and investment capacity are useful cold-start signals, behavioural features "
        "(number of buy transactions, total purchase volume, preferred category) are "
        "consistently stronger predictors. Customers' revealed preferences via "
        "transaction history are the most informative feature group.",
        "**Item popularity creates a rich-get-richer risk:** Popular assets (high "
        "n_buyers, high n_purchases) dominate the top-10 for all customers. This is "
        "a known popularity bias in collaborative filtering. Apply post-processing "
        "re-ranking (MMR) or coverage constraints to ensure diverse recommendations.",
        "**Temporal split is critical:** The val set uses the last 3 months of "
        "transactions. Leaking future interactions into training dramatically inflates "
        "NDCG@10. Always use a strict time-based split — never random.",
    ],
    "G2": [
        "**Cross-sectional ranking removes macro bias:** Raw financial ratios drift "
        "with market cycles (all P/Es expand in bull markets). Computing within-year "
        "percentile ranks makes features scale-invariant across years and directly "
        "comparable across sectors, which is why rank features are among the strongest "
        "predictors.",
        "**Composite factor scores encode analyst intuition:** The quality_spread "
        "(high ROE, low leverage), value_composite (low PE/PB), and "
        "profitability_composite (ROE + margins) bundle correlated raw ratios into "
        "single interpretable signals — exactly how fundamental analysts think. These "
        "engineered features improve both accuracy and SHAP interpretability.",
        "**Sector fairness must be monitored:** Model AUC varies by GICS sector "
        "because sector-specific accounting conventions (e.g., D/E for Financials, "
        "capex cycles for Energy) reduce ratio informativeness. Flag sectors with "
        "AUC < 0.55 for sector-specific model variants or feature augmentation.",
        "**Temporal split on fiscal year is mandatory:** The val set is fiscal year "
        "2022. Any look-ahead (using 2022 data to compute train-set ranks or "
        "imputation statistics) constitutes data leakage. All train_stats are fit "
        "on train years only and applied to val.",
    ],
}

_FE_GUIDANCE: dict = {
    "A": {
        "stages": [
            ("Missing Value Flags", ["fe_miss_addr1", "fe_miss_addr2", "fe_miss_dist1", "fe_miss_D2"]),
            ("Transaction Time Features", ["fe_hour", "fe_dow", "fe_is_weekend", "fe_is_night"]),
            ("Frequency Encoding", ["fe_card1_freq", "fe_addr1_freq", "fe_P_email_freq"]),
            ("Aggregation Features", ["fe_card1_txn_count", "fe_card1_mean_amt", "fe_card1_std_amt"]),
            ("Amount Ratios", ["fe_amt_to_card1_mean", "fe_amt_log", "fe_amt_binned"]),
        ],
        "stage_notes": {
            "Missing Value Flags": (
                "**Why treat missingness as a signal in fraud detection?**\n\n"
                "The IEEE-CIS dataset contains 339 V-features (V1–V339) derived from Vesta's proprietary "
                "fraud system. These features are intentionally sparse — each sub-group has a different "
                "missingness pattern because they are only populated for specific card or transaction types.\n\n"
                "- **`fe_miss_addr1` / `fe_miss_addr2`** — billing and shipping address missingness is "
                "itself a fraud signal. Fraudulent transactions frequently lack a valid shipping address "
                "or use mismatched addresses, making the absence of these fields informative.\n"
                "- **`fe_miss_dist1` / `fe_miss_D2`** — distance fields are null when the cardholder's "
                "location cannot be resolved. Missing distance data correlates with card-not-present fraud "
                "where the physical location of the transaction is obscured.\n\n"
                "*Academic basis: Bahnsen et al. (2016) show that missingness indicators in payment fraud "
                "datasets carry statistically significant predictive power independent of the feature values "
                "themselves — the pattern of what is unknown is as important as what is known.*"
            ),
            "Transaction Time Features": (
                "**Why does time matter for fraud detection?**\n\n"
                "Fraudsters operate with behavioural patterns that are distinctly different from legitimate "
                "cardholders. Time-of-day and day-of-week encode these patterns directly.\n\n"
                "- **`fe_hour`** — hour of transaction (extracted from `TransactionDT`). Fraud rates spike "
                "during late-night hours (midnight–4am) when cardholders are unlikely to monitor their "
                "accounts and when automated fraud scripts run.\n"
                "- **`fe_dow`** — day of week. Weekends show different fraud patterns (higher card-not-present "
                "fraud) vs weekdays (higher account-takeover fraud during business hours).\n"
                "- **`fe_is_weekend`** — binary flag. Weekend transactions have a different merchant mix "
                "(more e-commerce, less in-store) which affects the fraud distribution.\n"
                "- **`fe_is_night`** (10pm–6am) — direct capture of the high-risk nocturnal window.\n\n"
                "*TransactionDT is a timedelta in seconds, not a Unix timestamp. All time features are "
                "derived by modular arithmetic: hour = (TransactionDT // 3600) % 24.*"
            ),
            "Frequency Encoding": (
                "**Why does frequency encoding detect anomalous transactions?**\n\n"
                "Frequency encoding replaces a high-cardinality categorical (card number, zip code, email domain) "
                "with how often that value appears in the training set. This captures a critical insight: "
                "**a card seen 10,000 times is a high-volume legitimate card; a card seen twice is unusual**.\n\n"
                "- **`fe_card1_freq`** — frequency of the card1 identifier. A card used rarely is either "
                "new (higher risk) or a synthetic/temporary card number (very high risk).\n"
                "- **`fe_addr1_freq`** — frequency of the billing zip code. Zip codes that appear rarely "
                "are anomalous; legitimate billing addresses repeat across transactions.\n"
                "- **`fe_P_email_freq`** — purchaser email domain frequency. Disposable email domains "
                "(guerrillamail, mailinator) appear very rarely and strongly signal fraud.\n\n"
                "**Why not one-hot encode?** `card1` has ~18K unique values — one-hot would add 18K binary "
                "columns. Frequency encoding captures the rarity signal in a single continuous feature "
                "with no dimensionality explosion."
            ),
            "Aggregation Features": (
                "**Why aggregate transaction history per card?**\n\n"
                "Individual transactions carry limited context. A $500 transaction looks different "
                "when the card typically spends $25 vs when it typically spends $600.\n\n"
                "- **`fe_card1_txn_count`** — number of transactions by this card in the training window. "
                "Very low counts (1–2) suggest a newly-issued or compromised card number.\n"
                "- **`fe_card1_mean_amt`** — average transaction amount per card. This establishes the "
                "cardholder's normal spending level.\n"
                "- **`fe_card1_std_amt`** — standard deviation of amounts. High variance indicates "
                "erratic spending — either a business card or a compromised account being tested.\n\n"
                "**Data leakage caution:** These aggregates are computed from the full training set "
                "and must be computed separately for train and validation splits to avoid future leakage "
                "from validation transactions contaminating the train-set statistics."
            ),
            "Amount Ratios": (
                "**Why relative amounts rather than absolute amounts?**\n\n"
                "A $1,000 transaction is normal for some cards and suspicious for others. "
                "Normalising by card history converts an absolute amount into an anomaly score.\n\n"
                "- **`fe_amt_to_card1_mean`** = TransactionAmt / fe_card1_mean_amt. A ratio > 5 means "
                "the transaction is 5× the card's typical amount — a strong fraud signal regardless of "
                "the absolute dollar value.\n"
                "- **`fe_amt_log`** = log1p(TransactionAmt). Raw amounts span $0.25–$31,937 — "
                "a 5-order-of-magnitude range. Log-transform compresses this to a model-friendly scale "
                "while preserving the ordering.\n"
                "- **`fe_amt_binned`** — quantile bin of the transaction amount (low/medium/high/very high). "
                "Allows tree-based models to find amount thresholds that are meaningful at a population level "
                "rather than per-card.\n\n"
                "*Pozzolo et al. (2018) demonstrate that amount-relative-to-history is consistently among "
                "the top-3 features in production fraud models across card networks.*"
            ),
        },
        "description": "IEEE-CIS fraud features focus on temporal patterns, cardinality-based frequency encoding, and transaction amount anomalies relative to historical card behaviour.",
    },
    "B": {
        "stages": [
            ("Delinquency Aggregation", ["fe_total_dpd", "fe_dpd_severity", "fe_any_delinquency"]),
            ("Utilisation Ratios", ["fe_util_sq", "fe_high_util"]),
            ("Debt & Income Ratios", ["fe_debt_income_ratio", "fe_monthly_income_log"]),
            ("Loan Counts", ["fe_total_loans", "fe_loan_density"]),
        ],
        "stage_notes": {
            "Delinquency Aggregation": (
                "**Why aggregate delinquency events rather than use raw counts?**\n\n"
                "The Give Me Some Credit dataset contains six delinquency features tracking how many "
                "times a borrower was 30-, 60-, or 90-days past due. These raw counts have very skewed "
                "distributions — most borrowers have zero delinquencies, while a small number have "
                "extreme counts that are likely data-entry artefacts (value of 96 or 98 appearing "
                "hundreds of times).\n\n"
                "- **`fe_total_dpd`** = sum of all past-due event counts. Aggregating creates a single "
                "composite delinquency signal — borrowers with more total events are higher risk "
                "regardless of which bucket (30/60/90 day) they fell into.\n"
                "- **`fe_dpd_severity`** = weighted sum giving 90-day events 3× weight vs 30-day. "
                "A 90-day delinquency is categorically more serious than a 30-day one — severity "
                "weighting captures the credit bureau's own risk hierarchy.\n"
                "- **`fe_any_delinquency`** — binary flag. Even a single past-due event is a "
                "meaningful signal; this binary captures the threshold effect.\n\n"
                "*FICO and VantageScore models both weight recency and severity of delinquency more "
                "heavily than count alone — our severity-weighted feature approximates this logic.*"
            ),
            "Utilisation Ratios": (
                "**Why square the revolving utilisation ratio?**\n\n"
                "`RevolvingUtilizationOfUnsecuredLines` measures how much of a borrower's available "
                "revolving credit is being used. It is already bounded [0, 1] in theory — but the "
                "raw dataset contains values > 1 (over-limit accounts) up to 50,892, requiring clipping.\n\n"
                "- **`fe_util_sq`** = RevolvingUtilisation² (after clipping to [0, 1.5]). "
                "The relationship between utilisation and default risk is non-linear — the marginal "
                "risk increase accelerates sharply above 80% utilisation. Squaring captures this "
                "convexity without requiring a polynomial feature expansion.\n"
                "- **`fe_high_util`** — binary flag for utilisation > 0.8. This threshold corresponds "
                "to the credit bureau 'near-maxed' threshold, above which default rates roughly double.\n\n"
                "**Leakage check:** Utilisation is computed from the borrower's current state — "
                "it is a contemporaneous feature, not a future-looking one. No leakage risk."
            ),
            "Debt & Income Ratios": (
                "**Why log-transform income and compute a debt-income ratio?**\n\n"
                "`MonthlyIncome` is heavily right-skewed (mean ~$6,670, max > $3.5M) and has "
                "19.8% missing values — the highest missingness of any feature in this dataset.\n\n"
                "- **`fe_monthly_income_log`** = log1p(MonthlyIncome). Log-transform compresses the "
                "right tail and makes the feature approximately normal — important for Logistic "
                "Regression and neural approaches in the comparison.\n"
                "- **`fe_debt_income_ratio`** = DebtRatio / (MonthlyIncome + 1). The raw DebtRatio "
                "column is ambiguous (some values > 1 suggest it encodes total debt, not a ratio). "
                "Dividing by income recovers a true affordability signal: how many months of income "
                "does the borrower's debt represent?\n\n"
                "**Missing income imputation:** MonthlyIncome is Missing At Random (MAR) — higher-income "
                "borrowers are less likely to leave income blank. Training-set median imputation is "
                "used (not mean, which would be distorted by the high-income tail)."
            ),
            "Loan Counts": (
                "**Why does the number of open credit lines predict default?**\n\n"
                "- **`fe_total_loans`** = NumberOfOpenCreditLinesAndLoans + NumberRealEstateLoansOrLines. "
                "A borrower with many open lines has higher total exposure. However, the relationship "
                "with default is non-monotonic — a small number of loans indicates limited credit "
                "access, while a very large number signals over-extension.\n"
                "- **`fe_loan_density`** = fe_total_loans / (age + 1). Normalises by age — a 60-year-old "
                "with 15 credit lines is less concerning than a 25-year-old with 15 credit lines, "
                "because the older borrower has had decades to accumulate them responsibly.\n\n"
                "*Thomas et al. (2002) — Credit Scoring and Its Applications — identify credit line "
                "proliferation as a leading indicator of borrower stress in the 12–24 months before "
                "default, particularly when combined with high utilisation.*"
            ),
        },
        "description": "Credit features aggregate delinquency history, compute non-linear utilisation ratios, and derive debt-to-income proxies to capture repayment capacity.",
    },
    "C_nlp": {
        "stages": [
            ("TF-IDF Unigrams", ["tfidf_dim_1…N"]),
            ("Sentiment Lexicon Scores", ["finbert_positive", "finbert_negative", "finbert_neutral"]),
            ("Text Statistics", ["char_count", "word_count", "avg_word_len"]),
            ("Special Token Flags", ["has_ticker", "has_hashtag", "has_number"]),
        ],
        "stage_notes": {
            "TF-IDF Unigrams": (
                "**Why does sparse bag-of-words still work for financial sentiment?**\n\n"
                "TF-IDF (Term Frequency–Inverse Document Frequency) represents each headline as a "
                "sparse vector where each dimension is a vocabulary word, weighted by how often it "
                "appears in this document relative to all documents. Despite being a simple approach, "
                "it performs surprisingly well on financial text because financial language is "
                "domain-constrained — a small vocabulary of terms ('profit', 'loss', 'acquisition', "
                "'downgrade') carries most of the sentiment signal.\n\n"
                "- **TF-IDF dimensions** — after stop-word removal and min-frequency filtering, "
                "typically 2,000–5,000 vocabulary terms remain. Each document becomes a sparse "
                "vector in this space.\n"
                "- **Why unigrams only?** Bigrams (two-word phrases) can capture 'not profitable' "
                "vs 'profitable', but they increase dimensionality dramatically. For short financial "
                "headlines (avg ~8 words), unigrams capture most of the signal.\n"
                "- **IDF weighting** down-weights common financial filler words ('the company said') "
                "and up-weights rare but informative terms ('restructuring', 'covenant breach').\n\n"
                "*Loughran & McDonald (2011) demonstrate that a domain-specific financial vocabulary "
                "outperforms general-purpose sentiment lexicons precisely because financial language "
                "repurposes common words — 'liability' is negative in general English but neutral "
                "in accounting contexts.*"
            ),
            "Sentiment Lexicon Scores": (
                "**Why use FinBERT rather than VADER or TextBlob for financial sentiment?**\n\n"
                "General-purpose sentiment models are trained on product reviews, tweets, and news — "
                "they systematically misclassify financial text. FinBERT is a BERT model fine-tuned "
                "on financial communications (earnings calls, analyst reports, regulatory filings).\n\n"
                "- **`finbert_positive`** — probability score that the text expresses a positive "
                "financial outlook. High scores correlate with earnings beats, revenue growth, "
                "and positive guidance.\n"
                "- **`finbert_negative`** — probability score for negative sentiment. Captures "
                "language around profit warnings, legal exposure, and management departures.\n"
                "- **`finbert_neutral`** — factual/neutral probability. Many financial headlines are "
                "purely informational ('Company X reports Q3 results') — neutral classification "
                "prevents forcing these into positive or negative bins.\n\n"
                "**Computational note:** Full FinBERT inference is run offline in Step 1 and stored "
                "as pre-computed scores. Running inference inside the dashboard would be too slow "
                "for interactive use."
            ),
            "Text Statistics": (
                "**Why include raw text statistics as features?**\n\n"
                "The way a sentiment is expressed carries signal beyond the words themselves. "
                "Short, punchy headlines ('Stock crashes') carry different sentiment certainty "
                "than long, hedged statements.\n\n"
                "- **`char_count`** — character count. Very short headlines (< 20 chars) are often "
                "ambiguous; very long ones (> 150 chars) are usually factual reports.\n"
                "- **`word_count`** — word count. Sentiment intensity tends to decay with headline "
                "length — short, direct statements are more extreme.\n"
                "- **`avg_word_len`** — average word length. Financial jargon skews toward longer "
                "words ('restructuring', 'restatement') — high average word length correlates "
                "with technical/neutral content.\n\n"
                "These features are cheap to compute and act as calibration signals for the "
                "model — they help distinguish 'I can't tell the sentiment' from 'this is neutral'."
            ),
            "Special Token Flags": (
                "**Why flag tickers, hashtags, and numbers separately?**\n\n"
                "Financial text contains structured entities that carry sentiment context. "
                "These flags extract that structure explicitly rather than relying on TF-IDF "
                "to learn it implicitly.\n\n"
                "- **`has_ticker`** — presence of a stock ticker symbol (e.g. $AAPL, MSFT). "
                "Headlines mentioning specific tickers are more likely to be company-specific "
                "news (more extreme sentiment) vs market-wide commentary (more neutral).\n"
                "- **`has_hashtag`** — social media source indicator. Hashtag-prefixed headlines "
                "come from Twitter/X and exhibit different sentiment distributions vs newswire "
                "sources.\n"
                "- **`has_number`** — presence of a numeric quantity (percentages, dollar amounts, "
                "share prices). Quantitative headlines ('EPS up 23%') tend to be more factual "
                "and easier to classify correctly.\n\n"
                "*Das & Chen (2007) show that entity-type features improve financial sentiment "
                "classification F1 by 4–7 percentage points over text-only features.*"
            ),
        },
        "description": "NLP features combine TF-IDF bag-of-words with FinBERT sentiment scores and hand-crafted linguistic features for financial text classification.",
    },
    "C_markets": {
        "stages": [
            ("Book Features", ["fe_book_rv", "fe_log_book_rv", "fe_wap_mean"]),
            ("Trade Features", ["fe_lr_std", "fe_spread", "fe_log_spread"]),
            ("Cross-Book Aggregates", ["fe_stock_mean_rv", "fe_rv_vs_stock_mean"]),
            ("Ratio Features", ["fe_volatility_spread_ratio", "fe_rv_l2_ratio"]),
        ],
        "stage_notes": {
            "Book Features": (
                "**Why does the order book predict realised volatility?**\n\n"
                "The limit order book records every pending buy and sell order at each price level. "
                "The imbalance between buy and sell pressure — captured through the Weighted Average "
                "Price (WAP) — is a leading indicator of near-term price movement and therefore "
                "realised volatility.\n\n"
                "- **`fe_book_rv`** — realised volatility computed from WAP log-returns within the "
                "10-minute book window. This is the core signal: how much did the mid-price move "
                "within the current bucket?\n"
                "- **`fe_log_book_rv`** — log1p transform of fe_book_rv. Realised volatility is "
                "right-skewed and approximately log-normal — log-transform improves model fit and "
                "reduces the influence of volatility spikes.\n"
                "- **`fe_wap_mean`** — mean WAP across the 10-minute window. Price level itself "
                "is not directly predictive, but WAP stability (low variance relative to mean) "
                "signals low-volatility regimes.\n\n"
                "*WAP = (BidPrice1 × AskSize1 + AskPrice1 × BidSize1) / (BidSize1 + AskSize1). "
                "This size-weighted mid-price is more informative than the simple mid-price because "
                "it reflects order book depth — a tight spread with large size is more stable than "
                "the same spread with thin size.*"
            ),
            "Trade Features": (
                "**Why are trade-derived features complementary to book features?**\n\n"
                "Book features capture pending orders (intention); trade features capture executed "
                "transactions (realised activity). Together they reflect both supply-demand imbalance "
                "and actual market participation.\n\n"
                "- **`fe_lr_std`** — standard deviation of log-returns from trade prices. This is "
                "the purest direct measure of realised volatility from the trade side. High `fe_lr_std` "
                "means prices jumped significantly between consecutive trades.\n"
                "- **`fe_spread`** — mean bid-ask spread (AskPrice1 − BidPrice1). Spread is the "
                "market maker's compensation for bearing inventory risk — wide spreads indicate "
                "high uncertainty about true value (and thus higher volatility).\n"
                "- **`fe_log_spread`** — log1p(fe_spread). Spread distributions are also right-skewed; "
                "log-transform aligns them with the log-normal volatility distribution.\n\n"
                "**Microstructure insight:** Spread and volatility are jointly determined in market "
                "microstructure theory (Kyle 1985, Glosten-Milgrom 1985) — they rise together in "
                "response to information asymmetry."
            ),
            "Cross-Book Aggregates": (
                "**Why does market-wide volatility improve single-stock forecasts?**\n\n"
                "Individual stock volatility is partly idiosyncratic (news, earnings) and partly "
                "systematic (market-wide risk-on/risk-off). Cross-book features capture the "
                "systematic component.\n\n"
                "- **`fe_stock_mean_rv`** — mean realised volatility across all stocks in the same "
                "10-minute bucket. If the whole market is volatile, any single stock is more likely "
                "to be volatile too — this is the beta effect.\n"
                "- **`fe_rv_vs_stock_mean`** — ratio of this stock's fe_book_rv to fe_stock_mean_rv. "
                "A ratio > 1 means this stock is more volatile than the market average right now — "
                "a signal of stock-specific news or illiquidity.\n\n"
                "*The Optiver competition winners (2021) consistently found cross-stock features "
                "among their top-10 most important — the market context explains ~20–30% of "
                "individual stock volatility variance.*"
            ),
            "Ratio Features": (
                "**Why compute spread-to-volatility and L2 ratios?**\n\n"
                "Ratios normalise features that are meaningful only in context. An absolute spread "
                "of $0.01 is tight for a $10 stock but wide for a $1,000 stock.\n\n"
                "- **`fe_volatility_spread_ratio`** = fe_book_rv / (fe_spread + 1e-8). This captures "
                "the relationship between price movement and market maker compensation. When volatility "
                "exceeds the spread, the market maker is losing money — spreads widen rapidly, "
                "creating a feedback loop that amplifies volatility.\n"
                "- **`fe_rv_l2_ratio`** — ratio of current-bucket realised volatility to the L2 "
                "(preceding) bucket's volatility. Volatility is strongly autocorrelated (GARCH effects); "
                "this ratio captures whether volatility is accelerating or decelerating within the "
                "current session.\n\n"
                "*Andersen & Bollerslev (1998) establish that realised volatility computed from "
                "high-frequency returns is the most accurate ex-post measure of latent volatility — "
                "our features operationalise this insight at the 10-minute horizon.*"
            ),
        },
        "description": "Optiver volatility features are derived from limit order book (WAP, bid-ask spread) and trade data (log-returns std dev) to predict realized volatility over 10-minute windows.",
    },
    "E": {
        "stages": [
            ("Missing Value Flags", ["ps_car_03_cat_missing", "ps_car_05_cat_missing"]),
            ("Calc Feature Aggregates", ["fe_calc_mean", "fe_calc_sum", "fe_calc_std"]),
            ("Feature Group Counts", ["fe_num_ind_features", "fe_num_reg_features"]),
            ("Ratio & Interaction Terms", ["fe_calc_cv", "fe_ind_mean_ratio"]),
        ],
        "stage_notes": {
            "Missing Value Flags": (
                "**Why are -1 values in Porto Seguro actually missing indicators?**\n\n"
                "Porto Seguro's dataset uses **-1 as a sentinel value for missing data** — not NaN. "
                "This encoding is common in insurance data systems where null values can't be stored "
                "in fixed-schema databases. Treating -1 as a numeric value would be catastrophically "
                "wrong — a car with `ps_car_03_cat = -1` doesn't have a category value of -1; "
                "it has no recorded category at all.\n\n"
                "- **`ps_car_03_cat_missing`** — 69.1% of records have -1 here. This is the most "
                "missing feature in the dataset. The missingness pattern itself is informative: "
                "certain vehicle types (commercial, classic) may never have this attribute recorded.\n"
                "- **`ps_car_05_cat_missing`** — 44.8% missing. Second most missing feature. "
                "These two car category fields together create a 'data completeness' dimension "
                "that correlates with claim risk.\n\n"
                "*All -1 values are converted to NaN before any processing. Missing-flag columns "
                "are created first, then NaN values are imputed with the training-set mode for "
                "categoricals and median for numerics.*"
            ),
            "Calc Feature Aggregates": (
                "**Why aggregate the 'calc' feature group?**\n\n"
                "Porto Seguro's anonymised features are organised into groups by prefix: `ind_` "
                "(individual/personal), `reg_` (regional), `car_` (vehicle), `calc_` (calculated). "
                "The `calc_` group (features ps_calc_01 through ps_calc_20) are continuous computed "
                "features whose individual meaning is unknown — but their aggregate behaviour "
                "can be modelled.\n\n"
                "- **`fe_calc_mean`** — mean of all calc features per row. A high mean value "
                "may reflect higher overall exposure across multiple risk dimensions simultaneously.\n"
                "- **`fe_calc_sum`** — sum of calc features. Captures the total 'load' across "
                "all calculated risk factors for this policyholder.\n"
                "- **`fe_calc_std`** — standard deviation of calc features per row. High within-row "
                "std means the policyholder has a very uneven profile — high on some risk dimensions, "
                "low on others. This heterogeneity may signal elevated claim risk.\n\n"
                "**Why aggregate anonymised features?** Even without knowing what each feature means, "
                "the statistical properties of a group of features encode structure. This is the "
                "'meta-feature' approach used by many top Kaggle competitors on anonymised datasets."
            ),
            "Feature Group Counts": (
                "**Why count non-missing features per group?**\n\n"
                "In insurance data, the completeness of a policyholder's record is itself "
                "a risk signal — policyholders who provide complete information tend to be "
                "more careful and risk-aware.\n\n"
                "- **`fe_num_ind_features`** — count of non-missing `ind_` features. Individual "
                "attribute completeness correlates with the policyholder's engagement with the "
                "underwriting process.\n"
                "- **`fe_num_reg_features`** — count of non-missing `reg_` features. Regional "
                "data completeness varies by geography — some regions have better data infrastructure.\n\n"
                "**Why this works as a feature:** The missing data mechanism in Porto Seguro is "
                "Missing Not At Random (MNAR) — certain risk profiles systematically have less "
                "data recorded. The count of missing features is therefore a proxy for the "
                "underlying risk profile, not just random noise."
            ),
            "Ratio & Interaction Terms": (
                "**Why compute coefficient of variation and interaction terms?**\n\n"
                "- **`fe_calc_cv`** = fe_calc_std / (fe_calc_mean + 1e-8). The coefficient of "
                "variation normalises dispersion by magnitude — a std of 2 means something very "
                "different when the mean is 1 vs when it is 100. CV captures whether a "
                "policyholder's risk profile is consistently high, consistently low, or erratic.\n"
                "- **`fe_ind_mean_ratio`** = mean of `ind_` features / (mean of `reg_` features + 1e-8). "
                "The ratio of personal-attribute risk to regional risk captures whether the "
                "individual is high-risk relative to their geographic context. A high-risk person "
                "in a low-risk region is structurally different from a high-risk person in a "
                "high-risk region.\n\n"
                "*Interaction terms require careful leakage management: both numerator and denominator "
                "means are computed from training data only and applied to validation, preventing "
                "the interaction from encoding any target-correlated information from the val set.*"
            ),
        },
        "description": "Porto Seguro features aggregate calc, ind, reg, and car feature groups, add missing-flag indicators, and engineer interaction ratios to capture claim-risk patterns.",
    },
    "D": {
        "stages": [
            ("Engagement Ratios", ["fe_completion_rate", "fe_skip_rate", "fe_deep_listen_rate", "fe_variety_ratio"]),
            ("Listening Activity", ["fe_log_secs", "fe_log_days_log", "fe_secs_per_song"]),
            ("Subscription Plan", ["fe_is_discounted", "fe_discount_depth", "fe_is_long_plan", "fe_plan_days_log", "fe_txn_count_log"]),
            ("Renewal & Cancel", ["fe_auto_renew", "fe_cancel_rate"]),
            ("Demographics", ["fe_age", "fe_age_bucket_young", "fe_age_bucket_senior", "fe_is_male", "fe_is_female", "fe_city_risk", "fe_reg_channel_risk"]),
        ],
        "stage_notes": {
            "Engagement Ratios": (
                "**Why engagement ratios?** Raw song counts are meaningless in isolation — a user who plays "
                "1,000 songs but skips 95% of them is far less engaged than one who plays 200 and completes them all.\n\n"
                "- **`fe_completion_rate`** = num_100 / total songs played. Captures *quality* of listening. "
                "Users who listen through full tracks are invested in the platform.\n"
                "- **`fe_skip_rate`** = num_25 / total songs. Detects active disengagement — a high skip rate "
                "signals that content is not matching the user's preferences.\n"
                "- **`fe_deep_listen_rate`** combines the 75%, 98.5% and 100% completion buckets into a single "
                "'committed listening' signal, capturing users who nearly or fully complete tracks.\n"
                "- **`fe_variety_ratio`** = unique songs / total songs. Measures catalogue breadth — "
                "users who explore widely tend to build platform habits that increase switching costs.\n\n"
                "*Academic basis: Verbeke et al. (2012) show engagement depth is the single strongest predictor "
                "of subscription churn across digital streaming platforms.*"
            ),
            "Listening Activity": (
                "**Why log-transforms?** `total_secs_mean` and `log_days` are strongly right-skewed — "
                "a small number of power-users dominate the upper tail. Applying log1p compresses the tail, "
                "removes skew, and makes features more linearly separable for Logistic Regression while also "
                "helping gradient boosters by reducing the scale disparity between features.\n\n"
                "- **`fe_log_secs`** = log1p(total_secs_mean). Transforms raw listening volume into a "
                "scale-invariant engagement signal.\n"
                "- **`fe_log_days_log`** = log1p(log_days). Reflects *recency and consistency* of engagement — "
                "a user active on only 2 of the last 30 days is structurally different from one active on 25 days.\n"
                "- **`fe_secs_per_song`** = total_secs_mean / (num_unq_mean + 1). Captures average track depth — "
                "distinguishing users who play albums through from those who skim playlists."
            ),
            "Subscription Plan": (
                "**Why plan signals?** The plan dimension captures the *commercial relationship* between user and platform.\n\n"
                "- **`fe_is_discounted`** flags promotional pricing — users who joined via discounts may churn "
                "when the promotion expires (price-sensitive segment).\n"
                "- **`fe_discount_depth`** = (list_price − actual_paid) / list_price. Quantifies *how deeply* "
                "discounted the subscription is — heavy promotions signal higher churn risk post-discount.\n"
                "- **`fe_is_long_plan`** (plan ≥ 30 days) separates committed subscribers from trial/weekly "
                "users, who churn at dramatically higher rates.\n"
                "- **`fe_plan_days_log`** and **`fe_txn_count_log`** are log-transformed for the same skewness "
                "reason as listening activity. Transaction count reflects tenure — a user with 12+ transactions "
                "has demonstrated repeated willingness to pay."
            ),
            "Renewal & Cancel": (
                "**The most direct churn signals available in this dataset.**\n\n"
                "- **`fe_auto_renew`** is the customer's own *explicit* signal about future commitment. "
                "Users who disable auto-renew have already indicated intent to leave. In KKBox v2 data, "
                "auto-renew rate shows the largest mean difference between groups: churned ~0.51 vs retained ~0.89.\n"
                "- **`fe_cancel_rate`** captures historical cancellation behaviour across all past transactions. "
                "Users who have cancelled before are statistically more likely to churn again — past behaviour "
                "is one of the strongest predictors in subscription churn literature (Burez & Van den Poel, 2009).\n\n"
                "Both features are computed as per-user means across `transactions_v2.csv`, preserving the "
                "temporal signal without introducing data leakage from future transactions."
            ),
            "Demographics": (
                "**Why target-encoding instead of one-hot?**\n\n"
                "`city` (22 unique values) and `registered_via` (6 registration channels) are high-cardinality "
                "categoricals. One-hot encoding adds dimensions without capturing the monotonic relationship "
                "with churn. Instead, each category is replaced with its *mean churn rate computed from the "
                "training set only* — this is **target encoding**.\n\n"
                "- **`fe_city_risk`** — mean churn rate per city in the train set. Cities with poor connectivity "
                "or strong competitor presence show systematically higher churn.\n"
                "- **`fe_reg_channel_risk`** — mean churn rate per registration channel. Users who registered "
                "via certain promotional channels (registered_via = 9) show materially different retention curves.\n"
                "- **`fe_age`** — clipped to [7, 80] to remove data-entry errors (0s, negatives, implausibly "
                "high values make up ~34% of `bd`). Imputed with training median.\n"
                "- **Age buckets** (`fe_age_bucket_young` <25, `fe_age_bucket_senior` ≥45) allow the model "
                "to capture non-linear age effects without polynomial expansion.\n"
                "- **Gender dummies** handle the ~33% missing gender entries via an implicit 'unknown' group "
                "(users who are neither fe_is_male=1 nor fe_is_female=1).\n\n"
                "*Critical constraint: target encoding rates are computed from training data only and applied "
                "to the validation set — applying them to validation before computing rates would constitute leakage.*"
            ),
        },
        "description": "KKBox churn features focus on listening engagement quality (completion rate, deep-listen ratio), subscription renewal behaviour, and target-encoded demographic signals to capture retention risk.",
    },
    "F": {
        "stages": [
            ("TF-IDF Text Features", ["tfidf_emission", "tfidf_carbon", "tfidf_renewable", "tfidf_net_zero"]),
            ("Text Statistics", ["fe_text_len", "fe_word_count", "fe_avg_word_len", "fe_claim_density"]),
            ("ESG Gap Features", ["fe_avg_gap_clipped", "fe_e_gap_clipped", "fe_s_gap_clipped", "fe_gap_cv", "fe_max_gap"]),
            ("ESG Score Features", ["fe_composite_esg", "fe_esg_low", "fe_esg_high", "fe_e_score_norm"]),
            ("Financial & Interaction", ["fe_log_market_cap", "fe_log_emissions", "fe_emissions_high",
                                         "fe_claim_x_gap", "fe_gap_x_emissions", "fe_sector_risk_te"]),
        ],
        "stage_notes": {
            "TF-IDF Text Features": (
                "**Why TF-IDF for greenwashing detection?**\n\n"
                "TF-IDF converts each disclosure sentence into a sparse numeric vector where each dimension "
                "represents a vocabulary word, weighted by how informative it is across all documents.\n\n"
                "For greenwashing detection, TF-IDF captures the *language patterns* that distinguish "
                "genuine environmental commitments from marketing boilerplate:\n"
                "- **`tfidf_emission`** — sentences mentioning 'emission' in specific, quantified terms "
                "('reduced Scope 1 emissions by 34%') vs vague claims predict Low vs High risk differently.\n"
                "- **`tfidf_net_zero`** — 'net zero' is a high-commitment claim. Companies making this "
                "claim with low actual ESG scores are the clearest greenwashing signal.\n"
                "- **`tfidf_renewable`** — renewable energy claims are common greenwashing territory; "
                "the model learns which combinations of renewable language + low E scores predict High risk.\n"
                "- **Chi-squared feature selection** (top 200 of 2,000) retains only TF-IDF features "
                "with statistically significant association with the greenwashing risk label.\n\n"
                "*Loughran & McDonald (2011) demonstrate that domain-specific vocabulary outperforms generic "
                "sentiment lexicons for corporate disclosure analysis — chi-squared TF-IDF selection "
                "approximates this domain-specific approach without requiring pre-labelled wordlists.*"
            ),
            "Text Statistics": (
                "**Why measure how text is written, not just what it says?**\n\n"
                "Greenwashing language has structural signatures beyond keyword frequency. "
                "The way companies write about sustainability encodes credibility signals.\n\n"
                "- **`fe_text_len`** / **`fe_word_count`** — Very short vague sentences ('We are green.') "
                "differ structurally from specific commitments ('Scope 1 emissions decreased 34% vs. "
                "2019 baseline, validated under ISO 14064-3').\n"
                "- **`fe_avg_word_len`** — Technical ESG terms (decarbonisation, biodiversity, TCFD-aligned) "
                "are longer words. High average word length correlates with substantive rather than "
                "marketing content.\n"
                "- **`fe_claim_density`** = climate keywords / word count. A sentence crammed with "
                "sustainability buzzwords from a low-ESG company is the canonical greenwashing "
                "pattern — high claim density + high gap = high risk."
            ),
            "ESG Gap Features": (
                "**The core greenwashing signal: the gap between self-reported and independently-assessed ESG scores.**\n\n"
                "ESG rating agencies (MSCI, Sustainalytics, ISS) assess companies independently. "
                "Companies also self-report ESG scores in their annual and sustainability reports. "
                "The gap between these figures is the operational definition of score inflation.\n\n"
                "- **`fe_avg_gap_clipped`** — Average gap across E, S, G pillars, clipped to 0 "
                "(negative gaps = company understates ESG quality, not greenwashing). "
                "This is the single strongest greenwashing signal in the model.\n"
                "- **`fe_gap_cv`** — Coefficient of variation of E/S/G gaps. High CV means "
                "the company inflates some pillars much more than others — selective greenwashing "
                "(e.g. inflating E scores while honestly reporting S) is still greenwashing.\n"
                "- **`fe_max_gap`** — Maximum single-pillar gap. Even if the average is moderate, "
                "a single pillar gap > 30 points is a red flag.\n\n"
                "*Escrig-Olmedo et al. (2019) document average divergence of 20–40 points between "
                "major ESG rating agencies — our gap features operationalise this measurement divergence.*"
            ),
            "ESG Score Features": (
                "**Why include absolute ESG scores alongside the gap features?**\n\n"
                "The gap features tell us *how much* a company inflates its scores. The absolute "
                "score features tell us *from where they are inflating*. A company with an assessed "
                "score of 20 inflating to 50 is categorically different from one with a score of "
                "65 inflating to 75 — same gap, very different greenwashing risk.\n\n"
                "- **`fe_composite_esg`** — Assessed composite ESG score (0–100). Controls for "
                "baseline ESG quality when interpreting the size of the gap.\n"
                "- **`fe_esg_low`** (< 40) — Binary flag for genuinely poor-ESG companies. "
                "These companies making environmental claims carry the highest greenwashing risk.\n"
                "- **`fe_esg_high`** (≥ 70) — Binary flag for genuinely strong-ESG companies. "
                "Their environmental claims are more likely substantiated and verifiable."
            ),
            "Financial & Interaction": (
                "**Why include financial size, emissions, and interaction terms?**\n\n"
                "Greenwashing risk is not uniform across company size and carbon intensity:\n\n"
                "- **`fe_log_emissions`** — Emissions intensity (tCO₂e/$M revenue) is the most direct "
                "measure of environmental impact. High-emission companies making strong environmental "
                "claims warrant extra scrutiny — they have the most to hide.\n"
                "- **`fe_emissions_high`** (intensity > 500 tCO₂e/$M) — Binary flag for high-carbon "
                "industries (Energy, Utilities, Materials). These sectors account for the majority of "
                "documented greenwashing cases in academic literature.\n"
                "- **`fe_claim_x_gap`** = env_claim_label × fe_avg_gap_clipped — Interaction term "
                "capturing the joint effect of making claims AND having inflated scores. Only sentences "
                "that both make environmental claims AND belong to high-gap companies get non-zero values.\n"
                "- **`fe_sector_risk_te`** — Sector target encoding (mean greenwashing risk per sector "
                "from training data only). Captures the base rate of greenwashing by industry "
                "without the dimensionality of one-hot encoding.\n\n"
                "*Critical anti-leakage rule: sector target encoding is computed from training data only. "
                "Using validation/test sector risk rates would allow future information to influence "
                "the encoded feature, causing optimistic evaluation results.*"
            ),
        },
        "description": "ESG greenwashing features fuse real disclosure text (TF-IDF + text statistics) with "
                       "structured score inflation signals (ESG gap features) and financial context to distinguish "
                       "substantiated environmental commitments from marketing claim inflation.",
    },
    "B3": {
        "stages": [
            ("Denoise Preprocessing",      ["np.floor(x*100)/100 applied to all numeric features"]),
            ("All-Statement Aggregates",   ["D_39__mean", "B_1__last", "P_2__std", "R_1__max", "S_3__sum"]),
            ("Diff Features",              ["D_39__diff_last_first", "B_1__diff_last_mean"]),
            ("Last-3/6 Statement Stats",   ["B_1__last3_mean", "D_39__last6_std"]),
            ("Rank Features",              ["D_39__user_rank", "B_1__global_rank"]),
            ("Categorical Encoding",       ["D_63__ord", "D_64__ord", "D_63__freq"]),
            ("Missingness Flags",          ["fe_stmt_count", "fe_last_miss_count"]),
        ],
        "stage_notes": {},
    },
    "G1": {
        "stages": [
            ("User Profile Encoding",    ["user_risk_ord", "user_type_ord", "user_cap_ord"]),
            ("User Behaviour Aggregates",["user_n_buy_tx", "user_total_buy", "user_avg_buy",
                                          "user_n_assets", "user_buy_std", "user_pref_cat_enc"]),
            ("Item Popularity Features", ["item_n_buyers", "item_n_purchases", "item_total_vol",
                                          "item_avg_vol", "item_pop_rank"]),
            ("Item Metadata Encoding",   ["item_cat_enc", "item_subcat_enc", "item_market_enc",
                                          "item_sector_enc"]),
            ("ROI Features",             ["item_roi", "item_roi_min", "item_roi_max"]),
            ("Interaction Features",     ["inter_repeat_buys", "inter_days_since_last",
                                          "inter_cat_Equity", "inter_cat_Bond", "inter_cat_ETF"]),
        ],
        "stage_notes": {
            "User Behaviour Aggregates": (
                "**Why behavioural features outperform profile features?**\n\n"
                "Investor profile (risk level, investment capacity) provides useful cold-start "
                "signals but is often self-reported and relatively coarse. Transaction history "
                "reveals revealed preferences — what a customer actually buys is more predictive "
                "than what category they self-classify into.\n\n"
                "Key signals:\n"
                "- **user_n_buy_tx**: customers with more transactions have clearer preferences "
                "and stronger collaborative filtering signal.\n"
                "- **user_pref_cat_enc**: the customer's most frequently purchased asset category "
                "is the strongest single predictor of the next purchase.\n"
                "- **user_buy_std**: variance in purchase amounts indicates investment breadth."
            ),
            "Item Popularity Features": (
                "**Popularity vs. personalisation trade-off:**\n\n"
                "item_n_buyers (number of unique buyers in training) is often the highest-importance "
                "feature in collaborative filtering models — popular assets are purchased by many "
                "customers regardless of their profile. This is useful for accuracy but creates "
                "a popularity bias: niche assets that would suit a customer are under-recommended.\n\n"
                "**Mitigation strategy:**\n"
                "- Monitor the popularity distribution of top-10 recommendations vs. actual purchases.\n"
                "- Apply Maximal Marginal Relevance (MMR) re-ranking as a post-processing step.\n"
                "- Set minimum coverage thresholds per asset category in production."
            ),
            "Interaction Features": (
                "**Why interaction features capture personalised affinity?**\n\n"
                "inter_repeat_buys counts how many times a customer has previously purchased the "
                "candidate asset — a direct signal of product loyalty. inter_days_since_last measures "
                "recency of the last purchase in any asset of the same category, capturing "
                "portfolio-rebalancing behaviour.\n\n"
                "inter_cat_{category} features encode the customer's cumulative investment in each "
                "category — effectively a soft collaborative filter that lets the model learn "
                "category affinity without requiring matrix factorisation."
            ),
        },
        "description": "G1 features are grouped as: user profile encoding (risk/type/capacity) → "
                       "user behavioural aggregates (transaction history) → item popularity "
                       "(how often the asset is purchased globally) → item metadata (category, "
                       "market, sector) → ROI signals → interaction features (repeat buys, "
                       "category affinity). LambdaRank uses these to learn a personalised "
                       "ranking function optimised for NDCG@10.",
    },
    "G2": {
        "stages": [
            ("Clip & Impute Raw Ratios",  ["pe_ratio", "pb_ratio", "roe", "net_margin",
                                            "debt_equity", "interest_coverage", "fcf_yield"]),
            ("Sector Encoding",           ["sector_enc"]),
            ("Derived Features",          ["peg_ratio", "interest_burden", "quality_spread",
                                            "value_composite", "growth_composite",
                                            "profitability_composite", "leverage_risk"]),
            ("Cross-Sectional Ranks",     ["pe_ratio__rank", "roe__rank", "revenue_growth__rank",
                                            "profitability_composite__rank", "quality_spread__rank"]),
            ("Macro Regime Flags",        ["macro_regime", "is_crisis_year", "is_bull_year"]),
        ],
        "stage_notes": {
            "Clip & Impute Raw Ratios": (
                "**Why clip at 1st/99th percentile?**\n\n"
                "Financial ratios contain genuine extreme outliers — a company emerging from "
                "bankruptcy may have a P/E of −5,000; a high-growth startup may have P/S of 200. "
                "These extremes are real data points but destabilise tree splits and make "
                "cross-sectional comparisons meaningless.\n\n"
                "Clipping to the 1st–99th percentile range preserves 98% of the distribution "
                "while removing the tail artefacts. Clipping boundaries are fit on the training "
                "set and applied to validation — no leakage."
            ),
            "Cross-Sectional Ranks": (
                "**Why rank within fiscal year rather than globally?**\n\n"
                "A P/E of 20 meant different things in 2019 (moderate) vs. 2020 (cheap during "
                "the COVID crash) vs. 2021 (expensive as multiples expanded). Ranking within "
                "the fiscal year cohort makes the feature ask: 'Is this company cheap relative "
                "to its peers THIS year?' — a question that is meaningful regardless of the "
                "macro environment.\n\n"
                "This also removes macro-level correlations between features and the target "
                "that would otherwise inflate apparent model performance on historical data."
            ),
            "Macro Regime Flags": (
                "**Why add macro regime features?**\n\n"
                "The relationship between financial ratios and outperformance is regime-dependent: "
                "value stocks (low P/E) outperform in bear markets; growth stocks (high revenue "
                "growth) outperform in bull markets. The macro_regime flag (crisis/neutral/bull) "
                "lets the model learn interaction effects between ratio signals and the market "
                "environment without requiring separate models per regime."
            ),
        },
        "description": "G2 feature engineering follows a factor-investing pipeline: clip and impute "
                       "raw 10-K/10-Q ratios → encode sector → derive composite factor scores "
                       "(value, quality, growth, profitability, leverage) → compute cross-sectional "
                       "percentile ranks within each fiscal year → add macro regime indicators. "
                       "SHAP explanations are computed on all features for analyst-readable "
                       "attribution of every prediction.",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# ── Per-use-case EDA insights (shown alongside charts in Data Studio) ─────────
# ══════════════════════════════════════════════════════════════════════════════
_EDA_INSIGHTS: dict = {
    "D": {
        "target": (
            "**📊 Class Imbalance — KKBox Churn v2**\n\n"
            "The dataset contains approximately **8.4% churned subscribers** and 91.6% retained — "
            "a roughly 11:1 imbalance ratio.\n\n"
            "**Why this matters for modelling:**\n"
            "- A naïve classifier that always predicts 'Retained' achieves 91.6% accuracy but "
            "**zero business value** — it never identifies a subscriber at risk.\n"
            "- **Accuracy is a misleading metric** here. The primary metric is **ROC-AUC**: it measures "
            "rank ordering (can the model score churners higher than retainers?) independently of "
            "the threshold chosen.\n"
            "- **SMOTE oversampling** is applied *only inside training folds* to synthetically balance "
            "the minority class. It is never applied to validation or test data to avoid distributional leakage.\n"
            "- **`scale_pos_weight`** in XGBoost and **`is_unbalance=True`** in LightGBM further "
            "compensate for imbalance during tree-split decisions.\n\n"
            "**Business framing:** Even a 1% improvement in recall (catching more churners) translates "
            "directly to retained subscribers and avoided revenue loss — correctly modelling the minority "
            "class is the entire commercial objective of this use case."
        ),
        "correlation": (
            "**🔗 Key Correlation Signals — KKBox Churn**\n\n"
            "| Feature | Direction | Interpretation |\n"
            "|---|---|---|\n"
            "| `auto_renew_rate` | Strong negative | Users who auto-renew rarely churn — this is the clearest single retention signal |\n"
            "| `cancel_rate` | Strong positive | Historical cancellations predict future churn — past behaviour is the best predictor |\n"
            "| `total_secs_mean` | Negative | More listening time = stronger habit formation = lower churn |\n"
            "| `num_100_mean` | Negative | Track completions signal genuine platform engagement |\n"
            "| `plan_days_mean` | Negative | Longer-plan subscribers have made a greater financial commitment |\n"
            "| `discount_rate` | Positive | Discount-driven subscribers churn when promotions end |\n"
            "| `num_25_mean` | Positive | High skip rate signals disengagement |\n\n"
            "**What to watch for in the heatmap:**\n"
            "- `num_25_mean` and `num_50_mean` are often strongly correlated (both capture partial listens) "
            "— SHAP will arbitrate which one carries more unique information.\n"
            "- `total_secs_mean` and `num_unq_mean` may show moderate correlation — listening duration "
            "and catalogue breadth are related but distinct signals.\n"
            "- Transaction aggregates (`txn_count`, `plan_days_mean`, `auto_renew_rate`) will cluster "
            "together as they all derive from the same source table."
        ),
        "missing": (
            "**❓ Missing Value Patterns — KKBox Churn**\n\n"
            "Missing values arise primarily from the **left-join merge** across four source tables: "
            "not every subscriber in `train_v2.csv` has activity in every file during the v2 observation window.\n\n"
            "| Column | Est. % Missing | Mechanism | Treatment in Step 3 |\n"
            "|---|---|---|---|\n"
            "| `bd` (age) | ~34% | MNAR — younger users less likely to provide age | Clipped to [7,80], imputed with train median, age-bucket flags added |\n"
            "| `gender` | ~33% | MNAR — voluntary field | Encoded as 'unknown', then one-hot via `fe_is_male` / `fe_is_female` |\n"
            "| Log aggregates | Varies | MAR — users inactive in v2 window | Imputed with 0 (no listening = no activity signal) |\n"
            "| Transaction aggregates | Varies | MAR — users with no v2 transactions | Imputed with train-set medians |\n\n"
            "**Critical anti-leakage rule:** All imputation statistics (medians, churn-rate encodings) are "
            "computed from the **training split only** and then applied identically to the validation split. "
            "Fitting imputers on combined train+val data is one of the most common causes of "
            "over-optimistic evaluation results in subscription churn models."
        ),
        "outlier": (
            "**🔺 Outlier Handling — KKBox Churn**\n\n"
            "| Feature | Issue | Treatment in Step 3 |\n"
            "|---|---|---|\n"
            "| `bd` (age) | Values of 0, negatives, 200+ are clear data-entry errors | Hard-clip to [7, 80] before any other processing |\n"
            "| `total_secs_mean` | Extreme power-users (>100K secs/day) distort the distribution | Log-transform (`fe_log_secs`) compresses the right tail |\n"
            "| `plan_list_price` | Zero-price entries = free trials | `fe_is_discounted` distinguishes free vs. paid plans |\n"
            "| `num_25/50/75/985/100` | Heavy right skew in raw counts | Converted to bounded ratios [0,1] — engagement rates are scale-invariant |\n"
            "| `cancel_rate` | Bounded [0,1] by construction | No treatment needed |\n\n"
            "**Why we keep outliers rather than remove them:** Tree-based models (LightGBM, XGBoost, "
            "Random Forest) are naturally robust to outliers in the feature space — they split on "
            "thresholds, not distances. Removing extreme power-users would discard genuinely informative "
            "observations about the retained-user population tail."
        ),
    },
    "A": {
        "target": (
            "**📊 Class Imbalance — IEEE-CIS Fraud Detection**\n\n"
            "The IEEE-CIS dataset contains approximately **3.5% fraudulent transactions** — "
            "a roughly 28:1 imbalance ratio, more severe than most fraud datasets.\n\n"
            "**Why this matters for modelling:**\n"
            "- At 3.5% fraud rate, a classifier that always predicts 'Legitimate' achieves "
            "96.5% accuracy with **zero fraud caught** — accuracy is a completely useless metric here.\n"
            "- The primary competition metric is **ROC-AUC**, which measures the model's ability "
            "to rank fraudulent transactions above legitimate ones across all possible thresholds.\n"
            "- **SMOTE** is applied *inside training folds only* to synthetically up-sample the "
            "minority (fraud) class. Applying SMOTE before the train/val split would introduce "
            "synthetic fraud samples into the validation set, causing optimistic AUC estimates.\n"
            "- **`scale_pos_weight`** in XGBoost (≈ 28) and **`is_unbalance=True`** in LightGBM "
            "further compensate for imbalance at the split-criterion level.\n\n"
            "**Business framing:** Card fraud causes ~$33 billion in annual losses globally "
            "(Nilson Report 2023). Even a 1% improvement in fraud recall — catching more true "
            "fraud — directly translates to millions in prevented losses at scale."
        ),
        "correlation": (
            "**🔗 Key Correlation Signals — IEEE-CIS Fraud**\n\n"
            "| Feature | Direction | Interpretation |\n"
            "|---|---|---|\n"
            "| `TransactionAmt` | Positive (non-linear) | Fraud spikes at round-number amounts and very high values |\n"
            "| `card1_freq` | Negative | Cards seen many times are established; low-frequency cards are new or compromised |\n"
            "| `addr1_freq` | Negative | Rare billing zip codes are anomalous |\n"
            "| `fe_hour` (2–4am) | Positive | Late-night transactions have 2–3× baseline fraud rate |\n"
            "| `fe_amt_to_card1_mean` | Positive | Transactions far above the card's average amount signal fraud |\n"
            "| `P_emaildomain` (certain domains) | Positive | Disposable email providers strongly predict fraud |\n"
            "| `card4` (card type) | Mixed | Certain card network types show higher fraud prevalence |\n\n"
            "**What to watch for in the heatmap:**\n"
            "- Many V-features are highly correlated with each other (they come from the same Vesta "
            "sub-system). SHAP will show which V-features carry unique information after controlling "
            "for correlated features.\n"
            "- `TransactionAmt` correlations with V-features may be spurious — investigate with "
            "partial dependence plots rather than raw Pearson correlation."
        ),
        "missing": (
            "**❓ Missing Value Patterns — IEEE-CIS Fraud**\n\n"
            "The most challenging missing data pattern in this dataset is the V-features (V1–V339). "
            "These features are derived from Vesta's fraud system and are only populated for specific "
            "transaction types — their missingness is structural, not random.\n\n"
            "| Column Group | Est. % Missing | Mechanism | Treatment in Step 3 |\n"
            "|---|---|---|---|\n"
            "| V1–V11 (sub-group 1) | ~5% | MCAR — minor system gaps | Binary flag + median imputation |\n"
            "| V12–V34 (sub-group 2) | ~26% | MAR — card type dependent | Binary flag + median imputation |\n"
            "| V35–V52 (sub-group 3) | ~87% | MNAR — only for specific card types | Binary flag (missingness IS the signal) |\n"
            "| `addr2` | ~13% | MAR — international transactions | Binary flag + mode imputation |\n"
            "| `dist1` | ~59% | MAR — card-not-present transactions | Binary flag + 0 imputation |\n"
            "| `D2`–`D15` (timing) | Varies | MAR — depends on merchant type | Binary flag + median imputation |\n\n"
            "**Critical insight:** For the high-missingness V-features (>80%), the binary missing "
            "indicator is more predictive than the imputed value itself — the *fact* that this "
            "feature is absent tells you more about the transaction type than any value you could fill in."
        ),
        "outlier": (
            "**🔺 Outlier Handling — IEEE-CIS Fraud**\n\n"
            "| Feature | Issue | Treatment in Step 3 |\n"
            "|---|---|---|\n"
            "| `TransactionAmt` | Range $0.25–$31,937 — 5 orders of magnitude | log1p transform (`fe_amt_log`) |\n"
            "| `dist1` | Distance values up to 10,119 km — extreme right tail | log1p transform |\n"
            "| `C1`–`C14` (count fields) | Heavy right skew; some values > 10,000 | log1p transform for count features |\n"
            "| `D1`–`D15` (delta days) | Values up to 640 days — legitimate long-term cards | No treatment; tree models handle naturally |\n"
            "| V-features | Complex distributions; some bimodal | Standardise after flag+impute |\n\n"
            "**Why keep outliers?** Fraudulent transactions are themselves outliers — unusually large "
            "amounts, unusual frequencies, unusual times. Removing statistical outliers would "
            "preferentially remove fraud cases, destroying exactly the signal we need to learn."
        ),
    },
    "B": {
        "target": (
            "**📊 Class Imbalance — Give Me Some Credit (Credit Risk)**\n\n"
            "The Give Me Some Credit dataset contains approximately **6.7% serious delinquencies** "
            "(90+ days past due within 2 years) — a roughly 14:1 imbalance ratio.\n\n"
            "**Why this matters for modelling:**\n"
            "- Credit risk models must balance two types of error: **Type I (false positive)** = "
            "rejecting a creditworthy applicant; **Type II (false negative)** = approving a "
            "borrower who will default. These errors have very different business costs.\n"
            "- **ROC-AUC** is the primary metric — it evaluates the model's ability to rank "
            "higher-risk borrowers above lower-risk borrowers, which is exactly what a lender "
            "needs for cut-off score decisions.\n"
            "- **`scale_pos_weight`** ≈ 14 in XGBoost gives the minority class 14× the gradient "
            "weight, compensating for its underrepresentation in the training set.\n\n"
            "**Business framing:** A 1% improvement in identifying high-risk borrowers before "
            "approval can reduce a lender's charge-off rate by hundreds of basis points — "
            "directly improving net interest margin and capital adequacy ratios under Basel III."
        ),
        "correlation": (
            "**🔗 Key Correlation Signals — Give Me Some Credit**\n\n"
            "| Feature | Direction | Interpretation |\n"
            "|---|---|---|\n"
            "| `RevolvingUtilizationOfUnsecuredLines` | Strong positive | Maxed-out credit lines are the strongest single predictor of default |\n"
            "| `NumberOfTime90DaysLate` | Strong positive | Historical 90-day defaults predict future defaults — past behaviour dominates |\n"
            "| `NumberOfTime30-59DaysPastDueNotWorse` | Positive | Early delinquency is a leading indicator |\n"
            "| `DebtRatio` | Positive (with caveats) | High debt relative to income — but encoding issues require careful treatment |\n"
            "| `MonthlyIncome` | Negative | Higher income = lower default probability |\n"
            "| `age` | Negative (non-linear) | Older borrowers default less; youngest group (< 25) shows elevated risk |\n"
            "| `NumberOfOpenCreditLinesAndLoans` | Weak positive | More open lines = more exposure, but also signals credit access |\n\n"
            "**What to watch for in the heatmap:**\n"
            "- The three `NumberOfTimePastDue` features are strongly correlated — multicollinearity "
            "here is real and will affect Logistic Regression coefficients. LASSO or Ridge "
            "regularisation handles this; tree models are naturally robust.\n"
            "- `DebtRatio` has a bimodal distribution due to encoding issues; its raw Pearson "
            "correlation with the target undersells its true predictive power."
        ),
        "missing": (
            "**❓ Missing Value Patterns — Give Me Some Credit**\n\n"
            "This dataset has two features with meaningful missingness:\n\n"
            "| Column | % Missing | Mechanism | Treatment in Step 3 |\n"
            "|---|---|---|---|\n"
            "| `MonthlyIncome` | ~19.8% | MAR — higher earners less likely to disclose | Training-set median imputation + missing flag |\n"
            "| `NumberOfDependents` | ~2.6% | MAR — no dependents may appear as missing | Mode imputation (most common value = 0) |\n\n"
            "**`MonthlyIncome` is Missing At Random (MAR):** The probability of missing income "
            "correlates with income level itself — wealthier borrowers are more privacy-conscious. "
            "This means the missing indicator is itself predictive of lower default risk, which "
            "is counterintuitive but well-established in credit scoring literature.\n\n"
            "**Critical anti-leakage rule:** Imputation medians are computed from the training "
            "split only. Fitting the median on the combined train+validation set would allow "
            "validation-set income values to influence the imputation statistic."
        ),
        "outlier": (
            "**🔺 Outlier Handling — Give Me Some Credit**\n\n"
            "| Feature | Issue | Treatment in Step 3 |\n"
            "|---|---|---|\n"
            "| `age` | Values of 0 are clear data errors; one entry of age 0 | Clip to [18, 100] |\n"
            "| `RevolvingUtilizationOfUnsecuredLines` | Values up to 50,892 (should be 0–1) | Clip to [0.0, 1.5] before feature engineering |\n"
            "| `NumberOfTime30-59DaysPastDueNotWorse` | Values of 96 and 98 appear as artifacts | Treat values ≥ 90 as 90 (cap and flag separately) |\n"
            "| `NumberOfTime90DaysLate` | Same artifact (96/98 values) | Same treatment — cap at 90 |\n"
            "| `MonthlyIncome` | Right-skewed; max > $3.5M | log1p transform (`fe_monthly_income_log`) |\n"
            "| `DebtRatio` | Some values in millions (ambiguous encoding) | Clip to [0, 10] to handle encoding inconsistency |\n\n"
            "**The 96/98 artifact:** In credit bureau data, 96 and 98 are often used as special "
            "codes ('data not available', 'special circumstance'). Treating these as literal "
            "counts of 96 or 98 delinquency events would severely distort the model — they "
            "need to be recoded before any analysis."
        ),
    },
    "C_nlp": {
        "target": (
            "**📊 Class Distribution — FinPhrasebank Sentiment (NLP)**\n\n"
            "FinPhrasebank is a 3-class classification dataset of financial news headlines "
            "labelled as **Positive (~37%)**, **Neutral (~35%)**, and **Negative (~28%)**. "
            "This is substantially more balanced than the binary fraud/credit cases.\n\n"
            "**Why this matters for modelling:**\n"
            "- With three near-balanced classes, accuracy is more meaningful here than in "
            "the imbalanced binary cases — but **macro-F1** is still preferred because it "
            "weights each class equally regardless of frequency.\n"
            "- **Weighted cross-entropy loss** is used to give slightly more weight to the "
            "Negative class (28%), preventing the model from ignoring it to optimise the "
            "majority classes.\n"
            "- The boundary between Positive and Neutral is inherently ambiguous in financial "
            "text. 'Company X meets earnings expectations' could be Positive (met targets) or "
            "Neutral (no surprise). Inter-annotator agreement in the dataset is ~75%, meaning "
            "25% of labels are contested — the model's ceiling performance is bounded by this.\n\n"
            "**Business framing:** Accurate financial sentiment classification feeds directly "
            "into event-driven trading strategies, ESG monitoring systems, and regulatory "
            "disclosure analysis pipelines."
        ),
        "correlation": (
            "**🔗 Key Correlation Signals — FinPhrasebank NLP**\n\n"
            "| Feature | Direction | Interpretation |\n"
            "|---|---|---|\n"
            "| `finbert_positive` | Positive → Positive class | FinBERT directly predicts the label — highest single-feature AUC |\n"
            "| `finbert_negative` | Positive → Negative class | Symmetric signal for the negative class |\n"
            "| `has_number` | → Neutral | Quantitative headlines ('up 23%', '$4.2B revenue') tend to be factual |\n"
            "| `word_count` | → Neutral | Longer headlines are more nuanced and less extreme |\n"
            "| `has_ticker` | → Positive or Negative | Company-specific headlines are more extreme than market commentary |\n"
            "| TF-IDF: 'profit', 'growth', 'beat' | → Positive | Core positive financial vocabulary |\n"
            "| TF-IDF: 'loss', 'decline', 'cut' | → Negative | Core negative financial vocabulary |\n\n"
            "**Important caveat on TF-IDF correlations:** Because TF-IDF creates thousands of "
            "sparse features, traditional Pearson correlation is not the right tool for evaluating "
            "individual term importance. Use **mutual information** or **chi-squared** feature "
            "selection instead — these handle the discrete, sparse nature of bag-of-words features."
        ),
        "missing": (
            "**❓ Missing Value Patterns — FinPhrasebank NLP**\n\n"
            "Text datasets are unusual in that traditional 'missing value' problems rarely apply — "
            "a headline either exists or it doesn't. However, there are data quality considerations "
            "specific to NLP preprocessing:\n\n"
            "| Issue | Frequency | Treatment |\n"
            "|---|---|---|\n"
            "| Empty/whitespace-only headlines | Rare (<0.1%) | Drop rows |\n"
            "| FinBERT inference failures (very long text) | Rare | Fallback to VADER scores |\n"
            "| Encoding issues (non-UTF-8 characters) | Occasional | Normalise to ASCII before tokenisation |\n"
            "| Duplicate headlines with conflicting labels | ~3–5% | Keep first occurrence (label noise) |\n\n"
            "**Label noise is the primary data quality concern** in text sentiment datasets, not "
            "missing values. The FinPhrasebank dataset was annotated by a single domain expert — "
            "more recent versions include multi-annotator agreement scores which can be used to "
            "weight training examples by label confidence."
        ),
        "outlier": (
            "**🔺 Outlier Handling — FinPhrasebank NLP**\n\n"
            "In NLP, 'outliers' take the form of extreme documents rather than extreme numeric values:\n\n"
            "| Issue | Description | Treatment |\n"
            "|---|---|---|\n"
            "| Very short headlines (< 3 words) | Insufficient context for reliable classification | Flag with `is_very_short`; down-weight in training |\n"
            "| Very long headlines (> 50 words) | Unusual for financial news; may be misformatted | Truncate to 50 tokens before TF-IDF |\n"
            "| Non-English text | Occasional foreign-language entries | Filter to English using langdetect |\n"
            "| Extreme TF-IDF values | Single-occurrence rare terms with high IDF weights | min_df=2 in TfidfVectorizer drops hapax legomena |\n\n"
            "**TF-IDF sparsity:** After preprocessing, the feature matrix is extremely sparse "
            "(>99% zeros). This is expected and desirable — sparse representations are efficient "
            "and Logistic Regression with L2 regularisation handles them well. Do not apply "
            "PCA or StandardScaler to TF-IDF matrices as they will densify the representation "
            "and destroy the sparsity structure that makes these models efficient."
        ),
    },
    "C_markets": {
        "target": (
            "**📊 Target Distribution — Optiver Realised Volatility (Regression)**\n\n"
            "Unlike the binary classification use cases, UC-C_markets is a **regression problem**: "
            "the target is **realised volatility** — a continuous, non-negative measure of how much "
            "a stock's price moved in a 10-minute window immediately following the feature window.\n\n"
            "**Distribution characteristics:**\n"
            "- Realised volatility is approximately **log-normally distributed** — the log of "
            "volatility is approximately normal. This means raw volatility has a right-skewed "
            "distribution with a heavy tail (extreme spikes during market stress events).\n"
            "- The **RMSPE (Root Mean Squared Percentage Error)** is the Optiver competition metric: "
            "RMSPE = sqrt(mean((y_pred/y_true − 1)²)). This penalises relative errors equally "
            "across stocks with different baseline volatility levels.\n"
            "- Volatility is **strongly autocorrelated** — high-volatility regimes persist. "
            "A stock that was volatile in the previous 10-minute window is very likely to be "
            "volatile in the next 10-minute window (GARCH effects).\n\n"
            "**Business framing:** Accurate volatility forecasting is central to options pricing "
            "(Black-Scholes uses implied volatility), risk management (VaR models), and "
            "execution algorithms (VWAP strategies adapt to volatility regimes)."
        ),
        "correlation": (
            "**🔗 Key Correlation Signals — Optiver Volatility**\n\n"
            "| Feature | Direction | Interpretation |\n"
            "|---|---|---|\n"
            "| `fe_book_rv` | Strong positive | Current-window book volatility predicts next-window volatility (autocorrelation) |\n"
            "| `fe_lr_std` | Strong positive | Trade log-return std is the direct empirical volatility estimate |\n"
            "| `fe_spread` | Positive | Wide bid-ask spread = high uncertainty = high expected volatility |\n"
            "| `fe_stock_mean_rv` | Positive | Market-wide volatility context — systematic risk |\n"
            "| `fe_rv_vs_stock_mean` | Positive | Stock-specific excess volatility above market average |\n"
            "| `fe_wap_mean` | Near zero | Price level has little direct bearing on volatility |\n\n"
            "**What to watch for in the heatmap:**\n"
            "- `fe_book_rv` and `fe_lr_std` will be highly correlated — both measure realised "
            "volatility from different data sources. The book-based measure is available earlier "
            "in the tick stream; the trade-based measure is available only after trades execute.\n"
            "- Cross-stock features (`fe_stock_mean_rv`) will show moderate correlation with "
            "individual stock features — this is genuine systematic risk, not collinearity "
            "to be removed."
        ),
        "missing": (
            "**❓ Missing Value Patterns — Optiver Volatility**\n\n"
            "Missing values in order book data arise from two sources:\n\n"
            "| Source | Description | Treatment |\n"
            "|---|---|---|\n"
            "| No trades in bucket | Some stocks have 10-minute windows with zero executed trades | Impute trade features with 0 (no trading = no trade volatility) |\n"
            "| Thin order books | Very illiquid stocks may have missing L2 book levels | Impute missing WAP with L1 mid-price |\n"
            "| Cross-book NaN | If a stock has no data in a time bucket, cross-stock features are NaN | Forward-fill from previous bucket |\n\n"
            "**Temporal structure is critical:** This dataset has a time dimension — stocks × "
            "time buckets. Standard random imputation would be inappropriate. Missing values "
            "must be imputed using **forward-fill** (use previous bucket's value) or **market-wide "
            "mean** (use other stocks in the same bucket) to respect the temporal structure.\n\n"
            "**Train/val split must respect time:** Do not randomly shuffle rows. The validation "
            "set must use later time periods than the training set to simulate real forecasting "
            "conditions and avoid look-ahead bias."
        ),
        "outlier": (
            "**🔺 Outlier Handling — Optiver Volatility**\n\n"
            "| Feature | Issue | Treatment |\n"
            "|---|---|---|\n"
            "| `fe_book_rv` | Volatility spikes during market stress (COVID March 2020, flash crashes) | log1p transform; do NOT remove — extreme events are real |\n"
            "| `fe_spread` | Some stocks have artificially wide spreads when illiquid | Winsorise at 99th percentile per stock |\n"
            "| `fe_lr_std` | NaN when no trades occurred in bucket | Impute with 0 (no trades = no realised variance from trades) |\n"
            "| `target` (realised vol) | Right-skewed; some extreme spikes | RMSPE metric inherently handles this via percentage error |\n\n"
            "**Do not remove volatility spikes:** Extreme volatility events are the most "
            "important cases for financial risk management — removing them because they are "
            "'outliers' would make the model useless precisely when it is needed most "
            "(high-stress market conditions). Instead, log-transform features to compress "
            "the tail while keeping all observations.\n\n"
            "*The Optiver winners used no outlier removal — log-transforms and robust "
            "gradient boosting losses (Huber loss) were sufficient to handle extreme values.*"
        ),
    },
    "E": {
        "target": (
            "**📊 Class Imbalance — Porto Seguro Insurance Claims**\n\n"
            "The Porto Seguro dataset contains approximately **3.6% policyholders who filed "
            "a claim** (target = 1) — a roughly 27:1 imbalance ratio, similar in severity "
            "to the fraud detection case.\n\n"
            "**Why this matters for modelling:**\n"
            "- At 3.6% claim rate, precision and recall trade-offs are extreme. The Kaggle "
            "competition metric is **Normalised Gini Coefficient** (equivalent to 2×AUC − 1), "
            "which measures rank ordering quality independently of the chosen threshold.\n"
            "- **Class weights** and **SMOTE** are used to compensate, but the extreme imbalance "
            "means even small improvements in the minority class detection translate to "
            "significant Gini improvement.\n"
            "- This dataset is notable for its **feature anonymisation** — all feature names "
            "are obfuscated (ps_ind_01, ps_car_02, ps_calc_10) and their business meaning "
            "is not disclosed. Feature engineering must be data-driven rather than domain-driven.\n\n"
            "**Business framing:** Insurance pricing depends on accurate claim probability "
            "estimates. A 1% improvement in distinguishing high-claim-risk policyholders "
            "enables more accurate premium pricing — either avoiding adverse selection "
            "(underpricing high-risk customers) or retaining profitable customers by not "
            "overpricing low-risk ones."
        ),
        "correlation": (
            "**🔗 Key Correlation Signals — Porto Seguro Insurance**\n\n"
            "| Feature Group | Direction | Interpretation |\n"
            "|---|---|---|\n"
            "| `ps_ind_*` features (individual) | Mixed | Personal attributes — age proxies, coverage type |\n"
            "| `ps_calc_*` features (calculated) | Weak | Computed risk metrics — individually weak, strong in aggregate |\n"
            "| `ps_car_*` features (vehicle) | Moderate | Vehicle attributes — car type, age, brand category |\n"
            "| `ps_reg_*` features (regional) | Moderate | Geographic risk — region and density |\n"
            "| Missing value flags (`_missing`) | Positive | Missingness patterns correlate with claim risk |\n\n"
            "**Important finding from Kaggle:** The `calc` features (ps_calc_01–ps_calc_20) "
            "have near-zero correlation with the target individually, but collectively they "
            "carry predictive power when aggregated. This is why `fe_calc_mean`, `fe_calc_std`, "
            "and `fe_calc_cv` are important — no single calc feature matters, but the group "
            "statistical profile does.\n\n"
            "**Binary vs continuous:** Many features end in `_bin` (binary) or `_cat` (categorical). "
            "Point-biserial correlation is more appropriate than Pearson for binary features — "
            "the correlation matrix will understate their true relationship with the target."
        ),
        "missing": (
            "**❓ Missing Value Patterns — Porto Seguro Insurance**\n\n"
            "Porto Seguro encodes missing values as **-1** (not NaN). This must be converted "
            "before any analysis or modelling.\n\n"
            "| Feature | % with -1 | Mechanism | Treatment |\n"
            "|---|---|---|---|\n"
            "| `ps_car_03_cat` | 69.1% | MNAR — certain vehicle types lack this attribute | Binary flag + mode imputation |\n"
            "| `ps_car_05_cat` | 44.8% | MNAR — similar structural absence | Binary flag + mode imputation |\n"
            "| `ps_car_07_cat` | 1.9% | MAR — data collection issue | Mode imputation, no flag needed |\n"
            "| `ps_ind_02_cat` | 0.06% | MCAR — rare recording error | Mode imputation |\n"
            "| `ps_car_01_cat` | 0.1% | MCAR — rare recording error | Mode imputation |\n\n"
            "**MNAR = Missing Not At Random:** For `ps_car_03_cat` and `ps_car_05_cat`, "
            "the -1 encoding is structurally determined by the vehicle type — it is not a "
            "random data collection failure. This means the **missing flag itself is predictive** "
            "of claim risk, which is why we create binary indicator columns before imputing."
        ),
        "outlier": (
            "**🔺 Outlier Handling — Porto Seguro Insurance**\n\n"
            "| Feature | Issue | Treatment |\n"
            "|---|---|---|\n"
            "| All features with -1 | Sentinel for missing, not a valid value | Convert -1 → NaN before any outlier analysis |\n"
            "| `ps_reg_03` | Right-skewed continuous regional feature | log1p transform |\n"
            "| `ps_car_12` | Continuous vehicle feature; some extreme values | Winsorise at 99th percentile |\n"
            "| `ps_calc_*` | All continuous features 0–1 or count-based | No transformation needed — already bounded or low-range |\n"
            "| `ps_ind_14` | Integer count feature; >99% of values are 0 | Binary encode (0 vs > 0) — near-zero-variance otherwise |\n\n"
            "**The -1 conversion is the single most critical preprocessing step.** Forgetting "
            "to convert -1 to NaN before fitting any scaler or computing any statistic will "
            "corrupt every downstream calculation — means, medians, correlations, and model "
            "splits will all be distorted by the presence of -1 as if it were a real value.\n\n"
            "*Porto Seguro's encoding choice was a deliberate data obfuscation decision — "
            "treat it as domain knowledge, not a data quality issue.*"
        ),
    },
    "F": {
        "target": (
            "**📊 Greenwashing Risk — 3-Class Distribution**\n\n"
            "The hybrid dataset labels each disclosure sentence as **Low**, **Medium**, or **High** "
            "greenwashing risk. Low-risk sentences dominate (~55 %) because most environmental "
            "disclosures are either non-claims or claims with narrow score gaps. Medium and High "
            "classes together represent ~45 % of samples.\n\n"
            "**Why macro-F1?** With three unequal classes, accuracy is misleading — a naïve model "
            "predicting *Low* for everything achieves > 50 % accuracy while completely missing the "
            "High-risk cases that matter most to regulators. Macro-F1 weights every class equally, "
            "penalising the model for missing any tier.\n\n"
            "**EDGAR extension:** The class proportions in SEC 10-K Item 1A filings are likely "
            "even more skewed toward Low; students running the real-data extension should expect "
            "to apply class-weight adjustments. See the EDGAR EFTS endpoint in "
            "`01_data_loading.py` for the live query."
        ),
        "correlation": (
            "**🔗 Strongest Predictors of Greenwashing Risk**\n\n"
            "- **`avg_gap`** (ESG score minus emissions score): the single strongest numeric "
            "predictor. A large positive gap — high ESG rating but weak emissions performance — "
            "is the operational definition of greenwashing. Spearman ρ with risk label ≈ 0.65.\n"
            "- **`env_claim_label`** (ClimateBERT binary): sentences classified as genuine "
            "environmental claims are over-represented in Medium and High risk bands. A non-claim "
            "sentence almost never reaches High risk regardless of score gap.\n"
            "- **`emissions_intensity`**: right-skewed; high-intensity companies appear "
            "disproportionately in the High-risk tier even when ESG scores look respectable.\n"
            "- **Sector patterns**: Energy and Materials sectors cluster in Medium/High; "
            "Technology and Healthcare sectors skew Low. Target-encoded `sector_risk_mean` "
            "captures this signal without leaking test labels (encoding is fit on train only).\n\n"
            "TF-IDF unigrams/bigrams from `disclosure_text` add non-linear signal captured by "
            "chi-squared SelectKBest(200); top tokens include *carbon neutral*, *net zero*, "
            "*offset*, and *renewable target*."
        ),
        "missing": (
            "**🔍 Missing Data Profile**\n\n"
            "The hybrid (ClimateBERT + synthetic) dataset contains **no true missing values**: "
            "all numeric columns are fully populated during synthesis and all text rows carry a "
            "`disclosure_text` string.\n\n"
            "**Sentinel value note:** For compatibility with the EDGAR real-data extension, "
            "columns `esg_score`, `env_score`, and `emissions_intensity` use **−1 as a "
            "not-reported sentinel** rather than NaN. Students merging real EDGAR filings will "
            "encounter genuine NaNs that should be imputed or masked before feature engineering — "
            "`03_feature_engineering.py` already clips these at 0 to prevent negative TF-IDF "
            "inputs.\n\n"
            "The `company_id` column is categorical and non-null across all splits. "
            "Company-stratified splitting ensures no company appears in more than one partition, "
            "so there is no row-level leakage even if a firm filed multiple disclosures."
        ),
        "outlier": (
            "**⚠️ Outlier Profile**\n\n"
            "- **`avg_gap`**: designed to span [−40, +60]; extreme positive values (> 40) are "
            "rare by construction but represent the clearest greenwashing signal. Flag for "
            "review rather than clip.\n"
            "- **`emissions_intensity`**: log-normal in real data; the synthetic generator "
            "applies a light right skew (chi-squared draw). Values above 3× the 75th percentile "
            "are legitimate high-emitter observations — do not winsorise without domain "
            "justification.\n"
            "- **Text length** (`disclosure_text` token count): most sentences fall in the "
            "15–60 token range; a small tail reaches 120+ tokens. Very short strings (< 5 "
            "tokens) may be header artefacts in real EDGAR data and should be filtered.\n"
            "- **`esg_score` / `env_score`**: bounded [0, 100] by design; no true outliers "
            "in the synthetic split, but real EDGAR-sourced records may carry agency-specific "
            "scales that need normalisation before merging with synthetic baselines."
        ),
    },
    "B3": {
        "target": (
            "**📊 Class Balance — AmEx Loan Default Prediction (B3)**\n\n"
            "The AmEx dataset used for UC-B3 has **25.9% defaulters** — a 2.86:1 imbalance. "
            "The evaluation metric is **AmEx M = 0.5 × (Gini + D-rate@4%)**, combining "
            "overall ranking quality with precision at the top 4% risk tier."
        ),
        "correlation": (
            "**🔗 Key Correlation Signals — AmEx Loan Default (B3)**\n\n"
            "D_* delinquency features are the strongest predictors. "
            "B_* balance features capture exposure; P_* payment features capture behaviour. "
            "Last-statement values consistently outperform averages."
        ),
        "missing": (
            "**❓ Missing Value Patterns — AmEx Loan Default (B3)**\n\n"
            "Missingness is structural (MAR), tied to account type and credit product eligibility. "
            "`fe_all_null_count` (features never populated for a customer) is itself predictive."
        ),
        "outlier": (
            "**⚠️ Outlier Analysis — AmEx Loan Default (B3)**\n\n"
            "D_42 has ~23% outlier rows (delinquency spikes). "
            "Denoising via floor-rounding reduces spurious outliers from float precision noise."
        ),
    },
    "G1": {
        "target": (
            "**📊 Label Distribution — FAR-Trans Recommendation**\n\n"
            "In the candidate-pair training set, approximately **20% of pairs are positive** "
            "(actual purchases) and 80% are negative samples — a 4:1 ratio by design. This "
            "imbalance reflects the real-world sparsity of the recommendation interaction "
            "matrix: customers purchase a tiny fraction of available assets.\n\n"
            "**Why 4× negative sampling?**\n"
            "LambdaRank requires both positive and negative examples per query (customer). "
            "Too few negatives → model doesn't learn to discriminate well. Too many negatives "
            "→ the positive signal is overwhelmed and training slows. 4:1 is a widely used "
            "trade-off in industrial recommender systems.\n\n"
            "**Evaluation note:** NDCG@10 and Precision@10 evaluate ranked lists, not "
            "classification decisions, so the threshold-based precision/recall interpretation "
            "does not apply. The model is asked to rank, not classify."
        ),
    },
    "G2": {
        "target": (
            "**📊 Class Balance — SEC EDGAR Stock Outperformance**\n\n"
            "The dataset is approximately **balanced** by construction: the top 40% of "
            "12-month forward returners are labelled 1 (outperform) and the bottom 40% "
            "are labelled 0 (under-perform). The middle 20% are excluded to create a "
            "cleaner signal boundary.\n\n"
            "**Why this labelling strategy?**\n"
            "Absolute return thresholds (e.g., return > 0%) are noisy due to varying "
            "market conditions across years. A company with +5% in a +30% market is a "
            "significant under-performer; the same +5% in a −10% market is exceptional. "
            "Cross-sectional rank labelling normalises for market conditions and produces "
            "stable class proportions across fiscal years.\n\n"
            "**Implication for metrics:** With near-balanced classes, AUC-ROC is a reliable "
            "primary metric. AUC-PR adds value because analysts care about precision in the "
            "top-ranked stocks (screened for buy recommendations), which mirrors the "
            "precision-focused operating point of real investment workflows."
        ),
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# ── Cached data loaders ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_parquet(path: str, nrows: int = 0) -> Optional[pd.DataFrame]:
    """Cached Parquet loader. Returns None on error."""
    p = ROOT / path
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if nrows and len(df) > nrows:
            df = df.sample(nrows, random_state=42)
        return df
    except Exception as exc:
        log.warning("load_parquet failed %s: %s", p, exc)
        return None


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> Optional[pd.DataFrame]:
    """Cached CSV loader. Returns None on error."""
    p = ROOT / path
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as exc:
        log.warning("load_csv failed %s: %s", p, exc)
        return None


def load_model(path):
    """Load a joblib model file. Unwraps dict wrappers."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = joblib.load(p)
        if isinstance(obj, dict) and "model" in obj:
            return obj["model"]
        return obj
    except Exception as exc:
        log.warning("load_model failed %s: %s", p, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ── UI helpers ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def section_header(title: str, subtitle: str = "") -> None:
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div class='section-header'><h2>{title}</h2>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str = "", colour: str = ACCENT) -> str:
    dhtml = ""
    if delta:
        dcol = GRN if (delta.startswith("+") or delta.startswith("▲")) else RED
        dhtml = f"<div style='font-size:0.72rem;color:{dcol};margin-top:2px;font-weight:600;'>{delta}</div>"
    return (
        f"<div style='background:{colour}18;border:1.5px solid {colour};"
        f"border-radius:10px;padding:12px 14px;text-align:center;min-height:90px;'>"
        f"<div style='font-size:1.10rem;font-weight:700;color:{colour};'>{value}</div>"
        f"<div style='font-size:0.72rem;color:{FONT};opacity:0.75;margin-top:4px;'>{label}</div>"
        f"{dhtml}</div>"
    )


def fmt_pct(v) -> str:
    try:
        return f"{float(v)*100:.2f}%"
    except Exception:
        return str(v)


def fmt_num(v, d: int = 4) -> str:
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return str(v)


def _dark_fig(h: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT,
        height=h, margin=dict(t=40, b=30, l=50, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ── Navigation helpers ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _run_step_action(step, uc_key, label, suffix=""):
    _key = f"_rsa_{step}_{uc_key}_{suffix}" if suffix else f"_rsa_{step}_{uc_key}"
    if st.button(label, key=_key, type="primary"):
        st.session_state["_auto_run_step"] = step
        st.session_state.nav_page = "▶️  Run Pipeline"
        st.rerun()


def _prereq_warning(page: str, uc_key: str) -> str | None:
    uc    = USE_CASE_META.get(uc_key, {})
    d_dir = ROOT / "data"    / uc.get("data_dir",   "")
    m_dir = ROOT / "models"  / uc.get("model_dir",  "")
    _needs_step1 = {"🔬 Data Studio", "🔧 Data Preparation"}
    _needs_step3 = {"📈 Post-Processing EDA"}
    _needs_step5 = {"🤖 Model Development", "📊 Model Evaluation", "🎯 Prediction Demo"}
    _needs_step6 = {"🔍 Ethics & Explainability"}
    if page in _needs_step1:
        if not any(d_dir.glob("train_*.parquet")):
            return ("**Step 1 — Data Loading** has not been run yet for this use case.  \n"
                    "Switch to **▶️  Run Pipeline** and run Step 1 to generate the training data.")
    if page in _needs_step3:
        if not (d_dir / "train_fe.parquet").exists():
            return ("**Step 3 — Data Preparation** has not been run yet.  \n"
                    "Switch to **▶️  Run Pipeline** and run Step 3 to generate engineered features.")
    if page in _needs_step5:
        if not m_dir.exists() or not any(m_dir.glob("*.pkl")):
            return ("**Steps 4 & 5** (Model Training + Hyperparameter Tuning) have not been run.  \n"
                    "Switch to **▶️  Run Pipeline** and run Steps 4–5 to train the champion model.")
    if page in _needs_step6:
        r_dir = ROOT / "reports" / uc.get("report_dir", "")
        if not any((r_dir / n).exists() for n in ["shap_feature_importance.csv", "shap_importance.csv"]):
            return ("**Step 6 — Ethics & Explainability** has not been run yet.  \n"
                    "Switch to **▶️  Run Pipeline** and run Step 6 to generate SHAP / fairness reports.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ── Pipeline runner helpers ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _render_step_badges(steps: list, status: dict) -> None:
    cols = st.columns(len(steps))
    for col, s in zip(cols, steps):
        st_code = status.get(s, "wait")
        css = "step-done" if st_code == "done" else ("step-run" if st_code == "run" else "step-wait")
        icon = "✅" if st_code == "done" else ("⏳" if st_code == "run" else "⬜")
        col.markdown(
            f"<span class='{css}'>{icon} Step {s}: {STEP_NAMES.get(s,'')}</span>",
            unsafe_allow_html=True,
        )


def _run_step(step, script_rel, log_lines, log_placeholder, status_placeholder, steps, status):
    script_path = ROOT / script_rel
    log_lines.append(f"\n{'='*62}\n")
    log_lines.append(f"  Step {step}: {STEP_NAMES.get(step, script_rel)}\n")
    log_lines.append(f"  Script : {script_rel}\n")
    log_lines.append(f"{'='*62}\n\n")
    if not script_path.exists():
        log_lines.append(f"[ERROR] Script not found: {script_path}\n")
        log_placeholder.code("".join(log_lines[-200:]))
        return 1
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", bufsize=1,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    )
    _line_buf = 0
    for line in iter(proc.stdout.readline, ""):
        log_lines.append(line)
        _line_buf += 1
        if _line_buf >= 10:
            _line_buf = 0
            log_placeholder.code("".join(log_lines[-200:]))
            with status_placeholder.container():
                _render_step_badges(steps, status)
    proc.stdout.close()
    rc = proc.wait()
    log_placeholder.code("".join(log_lines[-200:]))
    with status_placeholder.container():
        _render_step_badges(steps, status)
    log_lines.append(f"\n[{'OK' if rc == 0 else 'FAILED'} — exit code {rc}]\n")
    log_placeholder.code("".join(log_lines[-200:]))
    return rc


# ══════════════════════════════════════════════════════════════════════════════
# ── Correlation matrix ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _render_correlation_matrix(uc_key: str) -> None:
    src = _PROFILING_SRC.get(uc_key, {})
    raw_path = src.get("raw")
    corr_csv = src.get("corr_csv")
    target   = src.get("target", "")

    if not raw_path:
        st.info("No raw data path configured for this use case.")
        return

    with st.spinner("Computing correlation matrix…"):
        df = load_parquet(raw_path, nrows=25_000) if raw_path.endswith(".parquet") else load_csv(raw_path)

    if df is None:
        st.warning("Raw data not found. Run Step 1 to generate data.")
        return

    # Determine columns to use
    top_cols = None
    if corr_csv:
        df_corr = load_csv(corr_csv)
        if df_corr is not None:
            feat_col = df_corr.columns[0]
            top_cols = df_corr[feat_col].head(30).tolist()

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if top_cols:
        num_cols = [c for c in top_cols if c in num_cols][:30]
    else:
        num_cols = [c for c in num_cols if c != target][:30]

    if len(num_cols) < 2:
        st.info("Not enough numeric columns for correlation matrix.")
        return

    method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True, key=f"_corr_method_{uc_key}")
    df_sub = df[num_cols].copy()
    corr_mat = df_sub.corr(method=method)

    fig = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=corr_mat.columns.tolist(),
        y=corr_mat.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title=f"|r| ({method})", tickfont=dict(color=FONT)),
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT,
        height=520, margin=dict(t=40, b=80, l=100, r=20),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig, width='stretch')

    # High-corr pairs table
    pairs = []
    arr = corr_mat.values
    cols_list = corr_mat.columns.tolist()
    for i in range(len(cols_list)):
        for j in range(i + 1, len(cols_list)):
            v = arr[i][j]
            if abs(v) > 0.5:
                pairs.append({"Feature A": cols_list[i], "Feature B": cols_list[j], "|r|": round(abs(v), 4), "r": round(v, 4)})
    if pairs:
        df_pairs = pd.DataFrame(pairs).sort_values("|r|", ascending=False).head(20)
        st.markdown("**High-correlation pairs (|r| > 0.5)**")
        st.dataframe(
            df_pairs,
            width='stretch',
            hide_index=True,
            column_config={"|r|": st.column_config.ProgressColumn("|r|", min_value=0, max_value=1, format="%.4f")},
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── Sidebar ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    """Render sidebar and return the selected uc_key string."""
    with st.sidebar:
        # Banner
        banner = ROOT / "dashboard" / "assets" / "ml_framework.png"
        if banner.exists():
            st.image(str(banner), width=260)
        else:
            st.markdown(
                f"<div style='text-align:center;padding:12px;background:{ACCENT}22;"
                f"border-radius:8px;margin-bottom:12px;'>"
                f"<span style='font-size:1.6rem;'>🏦</span><br>"
                f"<b style='color:{FONT};font-size:1.05rem;'>DSF504</b><br>"
                f"<span style='color:#90CAF9;font-size:0.78rem;'>Financial AI Platform</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**Select Use Case**")

        # Build display options
        uc_options = list(USE_CASE_META.keys())
        uc_labels  = [
            f"{USE_CASE_META[k]['icon']}  {USE_CASE_META[k]['title']}"
            + (" ⚙️" if USE_CASE_META[k].get("status") == "scaffolded" else "")
            for k in uc_options
        ]

        # Remember last selection
        if "selected_uc_idx" not in st.session_state:
            st.session_state.selected_uc_idx = 0

        sel_idx = st.selectbox(
            label="use_case_select",
            options=range(len(uc_options)),
            format_func=lambda i: uc_labels[i],
            index=st.session_state.selected_uc_idx,
            key="_sidebar_uc_select",
            label_visibility="collapsed",
        )
        st.session_state.selected_uc_idx = sel_idx
        uc_key = uc_options[sel_idx]
        uc = USE_CASE_META[uc_key]

        # Use case info card
        st.markdown(
            f"<div style='background:{ACCENT}18;border:1px solid {ACCENT}44;"
            f"border-radius:8px;padding:10px 12px;margin-top:8px;'>"
            f"<div style='font-size:0.78rem;color:#90CAF9;'>{uc['tag']}</div>"
            f"<div style='font-size:0.80rem;color:{FONT};margin-top:4px;'>"
            f"<b>Task:</b> {uc['task']}<br>"
            f"<b>Metric:</b> {uc['metric']}<br>"
            f"<b>Target:</b> <code>{uc['target']}</code>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        if uc.get("status") == "scaffolded":
            st.warning("This use case is scaffolded — pipeline scripts are not yet available.", icon="⚙️")

        st.markdown("---")

        # ML Framework phases info
        st.markdown(
            f"<div style='font-size:0.75rem;color:#90A4AE;'>"
            f"<b style='color:{FONT};'>ML Framework Phases</b><br>"
            f"1️⃣ Data Loading<br>"
            f"2️⃣ EDA & Profiling<br>"
            f"3️⃣ Feature Engineering<br>"
            f"4️⃣ Algorithm Selection + CV<br>"
            f"5️⃣ Hyperparameter Tuning<br>"
            f"6️⃣ Ethics & Explainability"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            f"<div style='font-size:0.70rem;color:#546E7A;text-align:center;'>"
            f"DSF504 · Financial AI Analytics<br>"
            f"Powered by Streamlit + Plotly"
            f"</div>",
            unsafe_allow_html=True,
        )

    return uc_key


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Run Pipeline ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page_run_pipeline(uc_key: str) -> None:
    uc = USE_CASE_META.get(uc_key, {})
    section_header(
        f"▶️  Run Pipeline — {uc['icon']} {uc['title']}",
        "Execute ML pipeline steps sequentially or individually.",
    )

    if uc.get("status") == "scaffolded":
        st.warning(
            f"**{uc['title']}** is scaffolded — no pipeline scripts exist yet.  \n"
            "Select an active use case (A, B, C_nlp, C_markets, D, or E) to run the pipeline.",
            icon="⚙️",
        )
        return

    scripts = USE_CASE_SCRIPTS.get(uc_key, {})
    available_steps = sorted(scripts.keys())

    if not available_steps:
        st.info("No pipeline steps registered for this use case.")
        return

    # Auto-run from another page
    _auto_step = st.session_state.pop("_auto_run_step", None)

    # ── Step selector ──────────────────────────────────────────────────────────
    st.markdown("#### Select steps to run")
    col_sel, col_run = st.columns([3, 1])
    with col_sel:
        _step_options_labels = {
            s: f"Step {s} — {STEP_NAMES.get(s, '')}" for s in available_steps
        }
        selected_steps = st.multiselect(
            "Steps",
            options=available_steps,
            default=available_steps if _auto_step is None else [_auto_step],
            format_func=lambda s: _step_options_labels[s],
            key=f"_pipeline_steps_{uc_key}",
            label_visibility="collapsed",
        )

    with col_run:
        run_clicked = st.button("▶ Run Selected", type="primary", key=f"_run_btn_{uc_key}", width='stretch')

    # Step description cards
    for s in available_steps:
        desc = STEP_DESCRIPTIONS.get(s, "")
        script = scripts.get(s, "")
        exists = (ROOT / script).exists() if script else False
        exists_html = f"<span style='color:{GRN};'>✅ Found</span>" if exists else f"<span style='color:{RED};'>❌ Missing</span>"
        with st.expander(f"Step {s} — {STEP_NAMES.get(s, '')}", expanded=False):
            st.markdown(f"{desc}")
            st.markdown(f"Script: `{script}` {exists_html}", unsafe_allow_html=True)

    # ── Run pipeline ───────────────────────────────────────────────────────────
    steps_to_run = selected_steps if run_clicked else (
        [_auto_step] if _auto_step and _auto_step in available_steps else []
    )

    if steps_to_run:
        st.markdown("---")
        st.markdown(f"#### Running steps: {', '.join(str(s) for s in steps_to_run)}")

        status_placeholder = st.empty()
        log_placeholder    = st.empty()
        log_lines: list    = []
        status: dict       = {s: "wait" for s in steps_to_run}

        with status_placeholder.container():
            _render_step_badges(steps_to_run, status)

        all_ok = True
        for step in steps_to_run:
            script_rel = scripts.get(step, "")
            status[step] = "run"
            rc = _run_step(step, script_rel, log_lines, log_placeholder, status_placeholder, steps_to_run, status)
            if rc == 0:
                status[step] = "done"
            else:
                status[step] = "wait"
                all_ok = False
                log_lines.append(f"[ABORT] Step {step} failed (exit {rc}). Stopping.\n")
                log_placeholder.code("".join(log_lines[-200:]))
                break

        with status_placeholder.container():
            _render_step_badges(steps_to_run, status)

        if all_ok:
            st.success(f"All {len(steps_to_run)} step(s) completed successfully!", icon="✅")
            # Save log
            log_dir = ROOT / "logs"
            log_dir.mkdir(exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"pipeline_{uc_key}_{ts}.log"
            log_file.write_text("".join(log_lines), encoding="utf-8")
            st.info(f"Log saved to `logs/pipeline_{uc_key}_{ts}.log`", icon="💾")
        else:
            st.error("Pipeline stopped due to an error. Check the log output above.", icon="❌")

        # Clear cache so updated parquet/models are visible
        st.cache_data.clear()


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Data Studio (profiling) ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page_data_profiling(uc_key: str) -> None:
    uc  = USE_CASE_META.get(uc_key, {})
    src = _PROFILING_SRC.get(uc_key, {})
    section_header(
        f"🔬 Data Studio — {uc['icon']} {uc['title']}",
        "Raw data exploration, statistical profiling, and quality assessment.",
    )


    # ── Dataset introduction card ──────────────────────────────────────────────
    _ds_info = _DATASET_INFO.get(uc_key, {})
    if _ds_info:
        _ds_url   = _ds_info.get("url", "")
        _ds_label = _ds_info.get("label", "Dataset")
        _ds_intro = _ds_info.get("intro", "")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;"
            f"padding:6px 12px;background:{GRID};border-radius:6px;"
            f"margin-bottom:10px;'>"
            f"<span style='font-size:0.8rem;color:{BLUE};white-space:nowrap;'>📂 Source dataset:</span>"
            f"<a href='{_ds_url}' target='_blank' "
            f"style='font-size:0.8rem;color:{BLUE};text-decoration:underline;"
            f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
            f"{_ds_label}</a></div>",
            unsafe_allow_html=True,
        )
        if _ds_intro:
            with st.expander("📖 Dataset overview", expanded=False):
                st.markdown(
                    f"<p style='color:{FONT};font-size:0.88rem;margin:0;'>{_ds_intro}</p>",
                    unsafe_allow_html=True,
                )

    _is_markets = (uc_key == "C_markets")

    if _is_markets:
        (tab_sample, tab_summary, tab_target, tab_corr,
         tab_missing, tab_outlier, tab_market) = st.tabs([
            "📋 Data Sample",
            "📊 Column Summary",
            "🎯 Target Distribution",
            "🔗 Correlation Matrix",
            "❓ Missing Values",
            "🔺 Outliers",
            "📈 Market Analytics",
        ])
    else:
        tab_market = None
        tab_sample, tab_summary, tab_target, tab_corr, tab_missing, tab_outlier = st.tabs([
            "📋 Data Sample",
            "📊 Column Summary",
            "🎯 Target Distribution",
            "🔗 Correlation Matrix",
            "❓ Missing Values",
            "🔺 Outliers",
        ])

    # ── Data sample ────────────────────────────────────────────────────────────
    with tab_sample:
        raw_path = src.get("raw")
        is_csv   = raw_path and raw_path.endswith(".csv")
        if raw_path:
            with st.spinner("Loading sample…"):
                if is_csv:
                    df = load_csv(raw_path)
                    if df is not None:
                        df = df.sample(min(1000, len(df)), random_state=42)
                else:
                    df = load_parquet(raw_path, nrows=2_000)
        else:
            df = None

        if df is not None:
            st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]:,} columns (sample)")
            n_num = len(df.select_dtypes(include="number").columns)
            n_cat = len(df.select_dtypes(include=["object","category","str"]).columns)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows (sample)", f"{len(df):,}")
            c2.metric("Numeric cols", n_num)
            c3.metric("Categorical cols", n_cat)
            st.dataframe(df.head(200), width='stretch', hide_index=True)
        else:
            st.info("Raw data not found. Run Step 1 to generate training data.")
            _run_step_action(1, uc_key, "▶ Run Step 1 — Data Loading  (goes to Run Pipeline)", suffix="sample")

    # ── Column summary ─────────────────────────────────────────────────────────
    with tab_summary:
        cs_path = src.get("col_summary")
        if cs_path:
            df_cs = load_csv(cs_path)
            if df_cs is not None:
                st.markdown(f"**{len(df_cs):,} columns profiled**")
                st.dataframe(df_cs, width='stretch', hide_index=True)
            else:
                st.info("Column summary CSV not found. Run Step 2 — EDA Analysis.")
                _run_step_action(2, uc_key, "▶ Run Step 2 — EDA  (goes to Run Pipeline)", suffix="cs")
        else:
            st.info("No column summary configured for this use case.")

    # ── Target distribution ────────────────────────────────────────────────────
    with tab_target:
        target_png = src.get("target_png")
        target_col = src.get("target", uc.get("target", ""))

        if target_png and (ROOT / target_png).exists():
            st.image(str(ROOT / target_png), width='stretch')
        else:
            # Compute from raw data
            raw_path = src.get("raw")
            is_csv   = raw_path and raw_path.endswith(".csv")
            if raw_path:
                df_t = load_csv(raw_path) if is_csv else load_parquet(raw_path, nrows=50_000)
            else:
                df_t = None

            if df_t is not None and target_col and target_col in df_t.columns:
                vc = df_t[target_col].value_counts().reset_index()
                vc.columns = ["Class", "Count"]
                vc["Pct"] = (vc["Count"] / vc["Count"].sum() * 100).round(2)

                fig = px.bar(vc, x="Class", y="Count",
                             color="Count", color_continuous_scale="Blues",
                             text="Pct",
                             title=f"Target Distribution — `{target_col}`")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT,
                                  height=380, showlegend=False)
                st.plotly_chart(fig, width='stretch')
                st.dataframe(vc, width='stretch', hide_index=True)
            else:
                st.info("Target column not found in raw data. Run Step 1 first.")

    # ── EDA insight — target tab ──────────────────────────────────────────────
    _eda_target_note = _EDA_INSIGHTS.get(uc_key, {}).get("target")
    if _eda_target_note:
        with tab_target:
            with st.expander("💡 Dataset Insights — Target Distribution", expanded=True):
                st.markdown(_eda_target_note)

    # ── Correlation matrix ─────────────────────────────────────────────────────
    with tab_corr:
        with st.expander("📚 Teaching Note — How to read this correlation matrix", expanded=False):
            st.markdown("""
**Pearson correlation** measures the *linear* relationship between two numeric columns.
Values range from -1 (perfect negative) to +1 (perfect positive); 0 = no linear relationship.

**What to look for:**
- **|r| > 0.9**: Near-duplicate features — consider dropping one to reduce dimensionality
- **|r| > 0.5 with target**: Strong predictors — prioritise these in feature selection
- **|r| < 0.05 with target**: Possibly noise — candidates for removal (but verify with SHAP first)
- **Feature–feature blocks**: Groups of correlated features suggest a common underlying factor

**Spearman** (toggle below) is more robust to outliers and captures monotonic (not just linear) relationships.
            """)
        _render_correlation_matrix(uc_key)
        _eda_corr_note = _EDA_INSIGHTS.get(uc_key, {}).get("correlation")
        if _eda_corr_note:
            with st.expander("💡 Dataset Insights — Correlation Patterns", expanded=True):
                st.markdown(_eda_corr_note)

    # ── Missing values ─────────────────────────────────────────────────────────
    with tab_missing:
        with st.expander("📚 Teaching Note — Handling missing values in ML", expanded=False):
            st.markdown("""
**Missing value patterns matter:**
- **MCAR** (Missing Completely At Random): safe to impute with mean/median; negligible bias
- **MAR** (Missing At Random): missingness depends on *other* observed columns — use model-based imputation
- **MNAR** (Missing Not At Random): missingness depends on the missing value itself — the hardest case; adding a binary "was_missing" flag often helps

**DSF504 strategy:**
1. Add a binary `fe_miss_<col>` flag before imputing — the *fact* of missingness may be predictive
2. Impute numeric columns with median (robust to outliers) within the training fold only
3. Impute categoricals with mode or a dedicated "MISSING" category
4. **Never impute validation/test data with statistics derived from validation/test rows**
            """)
        miss_png = src.get("missing_png")
        if miss_png and (ROOT / miss_png).exists():
            st.image(str(ROOT / miss_png), width='stretch')
        else:
            st.info("Missing values heatmap not found. Run Step 2 — EDA Analysis.")
            _run_step_action(2, uc_key, "▶ Run Step 2 — EDA  (goes to Run Pipeline)", suffix="miss")
        _eda_miss_note = _EDA_INSIGHTS.get(uc_key, {}).get("missing")
        if _eda_miss_note:
            with st.expander("💡 Dataset Insights — Missing Values", expanded=True):
                st.markdown(_eda_miss_note)

    # ── Outliers ───────────────────────────────────────────────────────────────
    with tab_outlier:
        with st.expander("📚 Teaching Note — Outlier detection strategies", expanded=False):
            st.markdown("""
**IQR method** (used in DSF504 Step 2):
- Compute Q1, Q3; flag values outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR]
- Robust to non-normal distributions; interpretable threshold

**Z-score method**: flag values where |z| > 3 — assumes normality

**What to do with outliers:**
- **Winsorise**: clip to the 1st/99th percentile — preserves row count
- **Log-transform**: compresses right-skewed distributions (TransactionAmt, income)
- **Leave them**: tree models (RF, XGB, LGBM) are naturally robust to outliers
- **Flag them**: add a binary `fe_is_outlier_<col>` feature; outlier status itself may be predictive for fraud

**In fraud detection**: extreme values (very large TransactionAmt) are often *informative*, not noise.
Review outliers in business context before removing.
            """)
        out_csv = src.get("outlier_csv")
        if out_csv:
            df_out = load_csv(out_csv)
            if df_out is not None:
                st.markdown(f"**{len(df_out):,} outlier records identified**")
                st.dataframe(df_out.head(200), width='stretch', hide_index=True)

                if len(df_out) > 0 and df_out.select_dtypes(include="number").shape[1] > 0:
                    num_cols_out = df_out.select_dtypes(include="number").columns.tolist()
                    col_pick = st.selectbox("Column to visualise", num_cols_out[:20], key=f"_out_col_{uc_key}")
                    fig = px.histogram(df_out, x=col_pick, nbins=50,
                                       title=f"Outlier distribution — {col_pick}",
                                       color_discrete_sequence=[RED])
                    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT, height=320)
                    st.plotly_chart(fig, width='stretch')
            else:
                st.info("Outlier report not found. Run Step 2 — EDA Analysis.")
        else:
            st.info("No outlier report configured for this use case.")
        _eda_out_note = _EDA_INSIGHTS.get(uc_key, {}).get("outlier")
        if _eda_out_note:
            with st.expander("💡 Dataset Insights — Outlier Treatment", expanded=True):
                st.markdown(_eda_out_note)

    # ── Market Analytics (C_markets only) ─────────────────────────────────────
    if _is_markets and tab_market is not None:
        with tab_market:
            _render_market_analytics_tab()


def _render_market_analytics_tab() -> None:
    """Dedicated market analytics tab for Use Case C_markets (Optiver Realized Volatility)."""
    book_path  = ROOT / "data/optiver_volatility/book_train.parquet"
    trade_path = ROOT / "data/optiver_volatility/trade_train.parquet"
    feat_path  = ROOT / "data/optiver_volatility/test_fe.parquet"

    if not book_path.exists():
        st.info("Optiver book data not found. Run Step 1 — Data Loading to download and process the dataset.")
        _run_step_action(1, "C_markets", "▶ Run Step 1 — Data Loading  (goes to Run Pipeline)", suffix="mkt_s1")
        return

    with st.spinner("Loading order-book sample …"):
        df_book = load_parquet(str(book_path), nrows=150_000)

    if df_book is None:
        st.error("Could not load book_train.parquet.")
        return

    # ── KPI cards ──────────────────────────────────────────────────────────────
    n_stocks  = int(df_book["stock_id"].nunique()) if "stock_id" in df_book.columns else 0
    n_times   = int(df_book["time_id"].nunique())  if "time_id"  in df_book.columns else 0
    n_rows    = len(df_book)
    rv_col    = "realized_volatility" if "realized_volatility" in df_book.columns else None
    mean_rv   = float(df_book[rv_col].mean()) if rv_col else float("nan")

    try:
        from dashboard.viz_library import kpi_cards, volatility_timeseries, rv_heatmap, scatter_bubble
        kpi_fig = kpi_cards([
            {"label": "Stocks",          "value": f"{n_stocks:,}",    "icon": "📊"},
            {"label": "Time IDs",        "value": f"{n_times:,}",     "icon": "⏱️"},
            {"label": "Book rows",       "value": f"{n_rows/1e6:.1f}M", "icon": "📋"},
            {"label": "Mean RV (sample)","value": f"{mean_rv:.5f}",   "icon": "📈"},
        ])
        st.plotly_chart(kpi_fig, width='stretch')
    except Exception:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks",    f"{n_stocks:,}")
        c2.metric("Time IDs",  f"{n_times:,}")
        c3.metric("Book rows", f"{n_rows/1e6:.1f}M")
        if rv_col:
            c4.metric("Mean RV", f"{mean_rv:.5f}")

    st.markdown("---")

    # ── Volatility time-series ─────────────────────────────────────────────────
    if rv_col and "time_id" in df_book.columns and "stock_id" in df_book.columns:
        st.markdown("#### Realized Volatility — Time Series")
        stock_ids = sorted(df_book["stock_id"].unique())
        default_stocks = stock_ids[:3]
        sel_stocks = st.multiselect(
            "Overlay individual stocks (up to 5)",
            options=stock_ids[:50],
            default=default_stocks,
            max_selections=5,
            key="_mkt_stock_sel",
        )
        try:
            from dashboard.viz_library import volatility_timeseries
            ts_fig = volatility_timeseries(
                df_book, time_col="time_id", value_col=rv_col,
                rolling_n=25, stock_overlays=sel_stocks, df_full=df_book,
            )
            st.plotly_chart(ts_fig, width='stretch')
        except Exception as e:
            # Fallback: simple line chart
            ts = (df_book.groupby("time_id")[rv_col].mean().reset_index()
                  .sort_values("time_id"))
            fig = px.line(ts, x="time_id", y=rv_col,
                          title="Mean Realized Volatility by Time ID")
            fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT, height=380)
            st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # ── RV Heatmap ─────────────────────────────────────────────────────────
        st.markdown("#### Stock × Time Volatility Heatmap")
        n_buckets = st.slider("Time buckets", 10, 50, 25, key="_mkt_buckets")
        max_stocks_hm = st.slider("Max stocks shown", 10, 50, 30, key="_mkt_maxstocks")
        try:
            from dashboard.viz_library import rv_heatmap
            hm_fig = rv_heatmap(
                df_book, stock_col="stock_id", time_col="time_id",
                value_col=rv_col, n_buckets=n_buckets, max_stocks=max_stocks_hm,
            )
            st.plotly_chart(hm_fig, width='stretch')
        except Exception as e:
            st.info(f"Heatmap could not render: {e}")

    # ── WAP bid-ask spread scatter ─────────────────────────────────────────────
    wap_col = next((c for c in df_book.columns if "wap" in c.lower()), None)
    bid_col = next((c for c in df_book.columns if "bid_price" in c.lower()), None)
    ask_col = next((c for c in df_book.columns if "ask_price" in c.lower()), None)

    if wap_col and bid_col and ask_col and rv_col:
        st.markdown("---")
        st.markdown("#### WAP vs Bid-Ask Spread")
        try:
            from dashboard.viz_library import scatter_bubble
            df_book["_spread"] = (df_book[ask_col] - df_book[bid_col]).abs()
            sc_fig = scatter_bubble(
                df_book.dropna(subset=[wap_col, "_spread", rv_col]),
                x_col=wap_col, y_col="_spread",
                color_col="stock_id" if "stock_id" in df_book.columns else None,
                size_col=rv_col, log_x=False, log_y=True,
                sample_n=3_000,
            )
            st.plotly_chart(sc_fig, width='stretch')
        except Exception as e:
            st.info(f"Scatter chart could not render: {e}")

    # ── Feature-engineered preview ─────────────────────────────────────────────
    if feat_path.exists():
        st.markdown("---")
        st.markdown("#### Engineered Features Preview (test_fe.parquet)")
        df_fe = load_parquet(str(feat_path), nrows=5_000)
        if df_fe is not None:
            fe_cols = [c for c in df_fe.columns if c.startswith("fe_")]
            st.markdown(f"**{len(fe_cols)} `fe_*` columns** · {len(df_fe):,} rows (sample)")
            if fe_cols:
                pick = st.selectbox("Feature distribution", fe_cols[:30], key="_mkt_fe_pick")
                fig = px.histogram(df_fe, x=pick, nbins=60,
                                   title=f"Distribution — {pick}",
                                   color_discrete_sequence=[BLUE])
                fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT, height=320)
                st.plotly_chart(fig, width='stretch')
            st.dataframe(df_fe[fe_cols[:20]].head(100), width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Data Preparation (Feature Engineering) ─────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page_feature_engineering(uc_key: str) -> None:
    uc  = USE_CASE_META.get(uc_key, {})
    src = _FE_EDA_SRC.get(uc_key, {})
    guidance = _FE_GUIDANCE.get(uc_key, {})
    section_header(
        f"🔧 Data Preparation — {uc['icon']} {uc['title']}",
        "Feature scaling, extraction, transformation, engineering, and selection.",
    )

    _fe_done = bool(src.get("train_fe") and (ROOT / src["train_fe"]).exists())
    _run_tab_label = "▶️ Run Step 4" if _fe_done else "▶️ Run Step 3"
    tab_guide, tab_feats, tab_summary, tab_run = st.tabs([
        "📖 FE Guidance",
        "📋 Feature List",
        "📊 FE Summary",
        _run_tab_label,
    ])

    # ── FE Guidance ────────────────────────────────────────────────────────────
    with tab_guide:
        desc = guidance.get("description", "")
        if desc:
            st.markdown(
                f"<div style='background:#1A237E22;border-left:4px solid #3949AB;"
                f"padding:10px 14px;border-radius:0 6px 6px 0;margin-bottom:16px;'>"
                f"<p style='margin:0;font-size:0.92rem;'>{desc}</p></div>",
                unsafe_allow_html=True,
            )

        stages     = guidance.get("stages", [])
        stage_notes = guidance.get("stage_notes", {})
        if stages:
            st.markdown("#### Feature Engineering Pipeline Stages")
            for stage_name, feats in stages:
                st.markdown(
                    f"<div style='margin-bottom:6px;'>"
                    f"<b style='color:{BLUE};font-size:0.95rem;'>{stage_name}</b><br>"
                    + "".join(
                        f"<span class='pill'>{f}</span>" for f in feats
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
                note = stage_notes.get(stage_name)
                if note:
                    with st.expander(f"📖 Rationale — {stage_name}", expanded=False):
                        st.markdown(note)
                st.markdown("")
        else:
            st.info("FE guidance not configured for this use case.")

        # Imputation summary (shown when stage_notes present)
        if stage_notes:
            st.markdown("---")
            st.markdown("#### Imputation & Final Cleaning")
            st.markdown(
                "After all feature groups are created, remaining numeric columns are imputed with "
                "**training-set medians** (computed on the training split before the val split is touched). "
                "Categorical columns — `msno` (user ID hash), `gender`, date fields — are dropped since "
                "their signal is already captured in the engineered features above. "
                "The result is a fully numeric, NaN-free feature matrix ready for model training."
            )


        # ── EDA-based recommendations ──────────────────────────────────────────
        _eda_recs = _EDA_RECOMMENDATIONS.get(uc_key, [])
        if _eda_recs:
            with st.expander("💡 Practical Considerations from EDA", expanded=False):
                st.markdown(
                    f"<p style='color:{FONT};font-size:0.88rem;margin-bottom:8px;'>"
                    "The following recommendations are derived from exploratory data analysis "
                    "of this dataset. They inform the feature engineering choices implemented in Step 3.</p>",
                    unsafe_allow_html=True,
                )
                for _rec in _eda_recs:
                    st.markdown(
                        f"<div style='border-left:3px solid {GRN};padding:6px 10px;"
                        f"margin-bottom:8px;background:{GRID}11;border-radius:0 4px 4px 0;'>"
                        f"<p style='margin:0;font-size:0.88rem;color:{FONT};'>• {_rec}</p></div>",
                        unsafe_allow_html=True,
                    )

        # SMOTE note
        st.markdown("---")
        if "Regression" not in uc.get("task", ""):
            st.info(
                "**Anti-leakage rule:** SMOTE oversampling is applied **only to the training fold** "
                "within each CV iteration — never to validation or test data.",
                icon="🛡️",
            )

    # ── Feature list ───────────────────────────────────────────────────────────
    with tab_feats:
        feat_list_path = src.get("feat_list")
        train_fe_path  = src.get("train_fe")

        def _enrich_feat_df(df_in: "pd.DataFrame") -> "pd.DataFrame":
            """Insert a Description column after the feature column."""
            df_out = df_in.copy()
            feat_col = df_out.columns[0]  # always 'feature'
            descriptions = df_out[feat_col].apply(
                lambda f: _describe_feature(f, uc_key)
            )
            df_out.insert(1, "description", descriptions)
            return df_out

        if feat_list_path:
            df_fl = load_csv(feat_list_path)
            if df_fl is not None:
                df_fl = _enrich_feat_df(df_fl)
                total = len(df_fl)
                c_left, c_right = st.columns([3, 1])
                with c_left:
                    search = st.text_input(
                        "Search features or descriptions",
                        key=f"_fe_search_{uc_key}",
                        placeholder="e.g. log, missing, velocity, gap …",
                    )
                with c_right:
                    st.markdown(
                        f"<p style='color:{FONT};font-size:13px;margin-top:28px'>"
                        f"<b>{total:,}</b> engineered features</p>",
                        unsafe_allow_html=True,
                    )
                if search:
                    mask = df_fl.apply(
                        lambda col: col.astype(str).str.contains(search, case=False)
                    ).any(axis=1)
                    df_fl = df_fl[mask]
                    st.caption(f'{len(df_fl):,} of {total:,} features match "{search}"')
                st.dataframe(df_fl, width='stretch', hide_index=True)
            else:
                st.info("Engineered features list not found. Run Step 3 — Data Preparation.")

        elif train_fe_path:
            df_fe = load_parquet(train_fe_path, nrows=5)
            if df_fe is not None:
                fe_cols = [c for c in df_fe.columns if c.startswith("fe_")]
                if fe_cols:
                    df_show = pd.DataFrame({
                        "feature":     fe_cols,
                        "description": [_describe_feature(c, uc_key) for c in fe_cols],
                        "dtype":       [str(df_fe[c].dtype) for c in fe_cols],
                    })
                    st.markdown(
                        f"<p style='color:{FONT}'><b>{len(fe_cols):,}</b> "
                        f"<code>fe_</code>-prefixed columns found in "
                        f"<code>train_fe.parquet</code></p>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(df_show, width='stretch', hide_index=True)
                else:
                    st.info("No `fe_` columns found. Check pipeline output.")
            else:
                st.info("train_fe.parquet not found. Run Step 3 — Data Preparation.")
        else:
            st.info("No feature list configured for this use case.")

    # ── FE Summary ─────────────────────────────────────────────────────────────
    with tab_summary:
        fe_png = src.get("fe_summary")
        if fe_png and (ROOT / fe_png).exists():
            st.image(str(ROOT / fe_png), width='stretch')
        else:
            st.info("Feature engineering summary plot not found. Run Step 3.")
            _run_step_action(3, uc_key, "▶ Run Step 3 — Data Preparation  (goes to Run Pipeline)", suffix="fe_sum")

    # ── Run Step 3 / Step 4 (dynamic) ─────────────────────────────────────────
    with tab_run:
        if _fe_done:
            st.markdown("#### ✅ Step 3 Complete — Feature Engineering")
            st.success(
                f"Processed training file found at `{src.get('train_fe', '')}`. "
                "Step 3 — Data Preparation has already run successfully.",
                icon="✅",
            )
            st.markdown("---")
            st.markdown("#### ▶ Next: Step 4 — Algorithm Selection & Cross-Validation")
            st.markdown(STEP_DESCRIPTIONS[4])
            _run_step_action(4, uc_key, "▶ Run Step 4 — Model Training  (goes to Run Pipeline)", suffix="fe_run_s4")
            st.markdown("---")
            with st.expander("🔁 Re-run Step 3 (overwrite processed files)", expanded=False):
                st.markdown(STEP_DESCRIPTIONS[3])
                _run_step_action(3, uc_key, "▶ Re-run Step 3 — Data Preparation", suffix="fe_run_s3")
        else:
            st.markdown("#### Run Step 3 — Data Preparation")
            st.markdown(STEP_DESCRIPTIONS[3])
            _run_step_action(3, uc_key, "▶ Run Step 3 — Data Preparation  (goes to Run Pipeline)", suffix="fe_run")


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Post-Processing EDA ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page_post_processing_eda(uc_key: str) -> None:
    uc  = USE_CASE_META.get(uc_key, {})
    src = _FE_EDA_SRC.get(uc_key, {})
    section_header(
        f"\U0001f4c8 Post-Processing EDA \u2014 {uc['icon']} {uc['title']}",
        "Compare raw vs processed distributions and inspect engineered features.",
    )

    st.markdown(
        "<p style='color:#888;font-size:0.9rem;margin-bottom:8px;'>"
        "Analyse the dataset <strong>after</strong> feature scaling, extraction, "
        "transformation, engineering, and selection. Compare distributions against "
        "the raw data to validate that preprocessing had the intended effect.</p>",
        unsafe_allow_html=True,
    )
    st.info(
        "This page reads the processed dataset produced by **Step 3 \u2014 Data Preparation** "
        "and compares it against the raw data. "
        "If processed files are missing, go to **\u25b6\ufe0f Run Pipeline \u2192 Step 3**.",
        icon="\u2139\ufe0f",
    )

    fe_path  = (ROOT / src["train_fe"])  if src.get("train_fe")   else None
    raw_path = (ROOT / src["raw"])       if src.get("raw")        else None
    feat_csv = (ROOT / src["feat_list"]) if src.get("feat_list")  else None
    fe_png   = (ROOT / src["fe_summary"])if src.get("fe_summary") else None
    target   = src.get("target", "")

    df_fe  = load_parquet(str(fe_path))  if fe_path  and fe_path.exists()  else None
    df_raw = load_parquet(str(raw_path)) if raw_path and raw_path.exists() else None

    if df_fe is None:
        st.warning("Processed training file not found. Run **Step 3 \u2014 Data Preparation** first.")
        _run_step_action(3, uc_key, "\u25b6 Run Step 3 \u2014 Data Preparation  (goes to Run Pipeline)", suffix="ppe_s3")
        return

    tab_overview, tab_compare, tab_new_feats, tab_target, tab_reports = st.tabs([
        "\U0001f4ca Overview",
        "\U0001f522 Raw vs Processed",
        "\U0001f195 New Features",
        "\U0001f3af Target Split",
        "\U0001f4ca Report Figures",
    ])

    # \u2500\u2500 OVERVIEW \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with tab_overview:
        n_raw  = len(df_raw) if df_raw is not None else 0
        n_fe   = len(df_fe)
        n_raw_cols = df_raw.shape[1] if df_raw is not None else 0
        n_fe_cols  = df_fe.shape[1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw rows",       f"{n_raw:,}")
        c2.metric("Processed rows", f"{n_fe:,}")
        c3.metric("Raw columns",    f"{n_raw_cols:,}")
        c4.metric("Processed cols", f"{n_fe_cols:,}")

        if feat_csv and feat_csv.exists():
            df_feats = load_csv(str(feat_csv))
            if df_feats is not None:
                st.markdown(f"**{len(df_feats):,} engineered features listed**")
                search_fe = st.text_input("Filter features", key=f"_ppe_search_{uc_key}", placeholder="Type to filter\u2026")
                if search_fe:
                    mask = df_feats.apply(lambda c: c.astype(str).str.contains(search_fe, case=False)).any(axis=1)
                    df_feats = df_feats[mask]
                st.dataframe(df_feats, width="stretch", hide_index=True)
        elif df_fe is not None:
            fe_cols_list = [c for c in df_fe.columns if c.startswith("fe_")]
            st.markdown(f"**{len(fe_cols_list)} `fe_` prefixed columns** in processed file")
            if fe_cols_list:
                df_show = pd.DataFrame({"feature": fe_cols_list,
                                        "dtype": [str(df_fe[c].dtype) for c in fe_cols_list]})
                st.dataframe(df_show, width="stretch", hide_index=True)

        if fe_png and fe_png.exists():
            st.image(str(fe_png), width="stretch")

    # \u2500\u2500 RAW vs PROCESSED \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    _RAW_CLR = "#78909C"  # hoisted here so tab_new_feats can always reference it
    _FE_CLR  = "#42A5F5"

    with tab_compare:
        if df_raw is None:
            st.info("Raw data not found \u2014 cannot compare distributions.")
        else:
            _raw_num = set(df_raw.select_dtypes(include="number").columns)
            shared_num = [
                c for c in df_fe.select_dtypes(include="number").columns
                if c in _raw_num and c != target
            ]

            if not shared_num:
                st.info(
                    "No numeric columns shared between raw and processed data. "
                    "This is expected when all features were renamed during engineering "
                    "(e.g. `D_39` → `D_39__mean`, `D_39__last`). "
                    "Use the **New Features** tab to inspect engineered column distributions."
                )
            else:
                with st.expander("\U0001f4da Teaching Note \u2014 What to look for", expanded=False):
                    st.markdown("""
**Distribution shifts after preprocessing are expected and desirable:**
- **Standardisation / MinMax scaling** \u2192 mean shifts to 0, range compresses to [0,1] or similar
- **Log / Box-Cox transforms** \u2192 right-skewed distributions become symmetric (skew \u2192 0)
- **Winsorisation / clipping** \u2192 outlier tails truncated; min/max values change
- **Missing value imputation** \u2192 null% drops to 0; mean may shift toward imputed value

**\u26a0\ufe0f Red flags:**
- Mean or std that changed drastically without a transform \u2192 check for data leakage
- New null% > 0 in processed file \u2192 imputation may have failed
- Skew increased after a log transform \u2192 verify correct column was transformed
                    """)

                st.caption(
                    "ℹ️ Only columns present in **both** raw and processed data are shown. "
                    "Columns with identical distributions were not transformed — this is normal. "
                    "To see engineered features, switch to the **New Features** tab."
                )
                n_bins  = st.slider("Histogram bins", 20, 100, 40, key=f"_ppe_bins_{uc_key}")
                max_col = min(20, len(shared_num))
                picked  = st.multiselect(
                    f"Select columns to compare (showing up to {max_col})",
                    shared_num[:max_col],
                    default=shared_num[:min(3, max_col)],
                    key=f"_ppe_cols_{uc_key}",
                )

                for col in picked:
                    _rv = df_raw[col].dropna()
                    _fv = df_fe[col].dropna()

                    def _stats(s):
                        sf = s.astype("float64")  # cast to float64 to avoid int overflow
                        def _safe(fn):
                            try:
                                v = fn(sf)
                                return float(v) if (v == v and abs(v) < 1e18) else float("nan")
                            except Exception:
                                return float("nan")
                        return {
                            "n":     len(sf),
                            "mean":  _safe(lambda x: x.mean()),
                            "std":   _safe(lambda x: x.std()),
                            "min":   _safe(lambda x: x.min()),
                            "median":_safe(lambda x: x.median()),
                            "max":   _safe(lambda x: x.max()),
                            "skew":  _safe(lambda x: x.skew()) if len(sf) > 2 else 0.0,
                            "null%": round((1 - len(s) / max(len(df_raw), 1)) * 100, 2),
                        }

                    rs, fs = _stats(_rv), _stats(_fv)

                    fig_pair = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=[f"<b>RAW</b> \u2014 {col}", f"<b>PROCESSED</b> \u2014 {col}"],
                        horizontal_spacing=0.06,
                    )
                    fig_pair.add_trace(
                        go.Histogram(x=_rv, nbinsx=n_bins, marker_color=_RAW_CLR,
                                     name="Raw", showlegend=False,
                                     hovertemplate="Value: %{x}<br>Count: %{y}<extra>Raw</extra>"),
                        row=1, col=1,
                    )
                    fig_pair.add_trace(
                        go.Histogram(x=_fv, nbinsx=n_bins, marker_color=_FE_CLR,
                                     name="Processed", showlegend=False,
                                     hovertemplate="Value: %{x}<br>Count: %{y}<extra>Processed</extra>"),
                        row=1, col=2,
                    )
                    for col_idx, (mean_val, clr) in enumerate(
                        [(rs["mean"], _RAW_CLR), (fs["mean"], _FE_CLR)], start=1
                    ):
                        if not (np.isnan(mean_val) or np.isinf(mean_val)):
                            fig_pair.add_vline(x=mean_val, line_width=1.5, line_dash="dash",
                                               line_color=clr, row=1, col=col_idx)
                    fig_pair.update_layout(
                        plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT,
                        height=240, margin=dict(t=40, b=10, l=30, r=10),
                    )
                    st.plotly_chart(fig_pair, width="stretch")

                    _metrics = ["n", "mean", "std", "min", "median", "max", "skew", "null%"]
                    delta_df = pd.DataFrame({
                        "Metric":               _metrics,
                        "Raw":                  [rs[k] for k in _metrics],
                        "Processed":            [fs[k] for k in _metrics],
                    })
                    delta_df["\u0394 (Proc \u2212 Raw)"] = delta_df["Processed"] - delta_df["Raw"]
                    st.dataframe(delta_df.round(4), width="stretch", hide_index=True)
                    st.markdown("<hr style='border:none;border-top:1px solid #2A2A4A;margin:6px 0;'>",
                                unsafe_allow_html=True)

    # \u2500\u2500 NEW FEATURES \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with tab_new_feats:
        raw_cols = set(df_raw.columns) if df_raw is not None else set()
        new_feats = [c for c in df_fe.columns if c not in raw_cols and c != target]
        if not new_feats:
            st.info("No new engineered columns detected (all processed columns exist in raw data).")
        else:
            st.markdown(f"**{len(new_feats)} new engineered columns** (present in processed, absent from raw)")
            new_data = []
            for c in new_feats:
                row = {"Feature": c, "dtype": str(df_fe[c].dtype), "nulls": int(df_fe[c].isna().sum())}
                if df_fe[c].dtype.kind in ("f","i","u"):
                    row.update({"mean": round(float(df_fe[c].mean()), 4),
                                 "std":  round(float(df_fe[c].std()),  4),
                                 "min":  round(float(df_fe[c].min()),  4),
                                 "max":  round(float(df_fe[c].max()),  4)})
                new_data.append(row)
            st.dataframe(pd.DataFrame(new_data), width="stretch", hide_index=True)

            pick_new = st.selectbox("Visualise distribution of new feature",
                                    [c for c in new_feats if df_fe[c].dtype.kind in ("f","i","u")][:30],
                                    key=f"_ppe_new_pick_{uc_key}")
            if pick_new:
                fig_new = px.histogram(df_fe, x=pick_new, nbins=50,
                                       title=f"Engineered feature \u2014 {pick_new}",
                                       color_discrete_sequence=[_FE_CLR])
                fig_new.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT, height=300)
                st.plotly_chart(fig_new, width="stretch")

    # \u2500\u2500 TARGET SPLIT \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with tab_target:
        target_labels = src.get("target_labels", {})
        if target and target in df_fe.columns:
            vc = df_fe[target].value_counts().reset_index()
            vc.columns = ["Class", "Count"]
            vc["Label"] = vc["Class"].map(
                lambda x: target_labels.get(x, target_labels.get(
                    int(x) if str(x).lstrip("-").isdigit() else x, str(x))))
            vc["Pct"]   = (vc["Count"] / vc["Count"].sum() * 100).round(2)
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.dataframe(vc, width="stretch", hide_index=True)
            with col_b:
                fig_tgt = px.pie(vc, names="Label", values="Count",
                                 title=f"Target split \u2014 `{target}` (processed set)",
                                 color_discrete_sequence=[GRN, RED, BLUE, ORG])
                fig_tgt.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT, height=320)
                st.plotly_chart(fig_tgt, width="stretch")
        else:
            st.info(f"Target column `{target}` not found in processed data.")

    # ── REPORT FIGURES ─────────────────────────────────────────────────────────
    with tab_reports:
        report_dir = src.get("report_dir", "")
        if not report_dir:
            st.info("No report directory configured for this use case.")
        else:
            r_dir = ROOT / report_dir
            if not r_dir.exists():
                st.info(f"Report directory not found: `{report_dir}`")
                st.caption("Run Steps 2 and 3 (EDA & Feature Engineering) to generate report figures.")
            else:
                # Categorise available PNGs
                eda_patterns = [
                    "overview.png",
                    "target_distribution.png",
                    "missing_heatmap.png",
                    "numeric_distributions.png",
                    "readability_complexity.png",
                    "top_unigrams.png",
                    "top_bigrams.png",
                    "term_heatmap.png",
                ]
                fe_patterns = [
                    "engineered_feature_summary.png",
                    "raw_vs_processed_distributions.png",
                ]

                eda_imgs  = [(p, r_dir / p) for p in eda_patterns if (r_dir / p).exists()]
                fe_imgs   = [(p, r_dir / p) for p in fe_patterns  if (r_dir / p).exists()]
                # Also catch any other PNGs not in the named lists
                all_named = set(eda_patterns + fe_patterns)
                other_imgs = sorted(
                    [(p.name, p) for p in r_dir.glob("*.png") if p.name not in all_named],
                    key=lambda x: x[0],
                )

                total = len(eda_imgs) + len(fe_imgs) + len(other_imgs)
                if total == 0:
                    st.info("No PNG report figures found. Run Steps 2–3 to generate them.")
                    _run_step_action(2, uc_key, "▶ Run Step 2 — EDA Analysis", suffix="ppe_rpt_s2")
                    _run_step_action(3, uc_key, "▶ Run Step 3 — Feature Engineering", suffix="ppe_rpt_s3")
                else:
                    st.caption(f"📁 `{report_dir}` — {total} figure(s) available")

                    if eda_imgs:
                        st.markdown("**📊 EDA Report Figures** *(from Step 2)*")
                        for label, img_path in eda_imgs:
                            clean = label.replace("_", " ").replace(".png", "").title()
                            with st.expander(clean, expanded=True):
                                st.image(str(img_path), width='stretch')

                    if fe_imgs:
                        st.markdown("**🔧 Feature Engineering Figures** *(from Step 3)*")
                        for label, img_path in fe_imgs:
                            clean = label.replace("_", " ").replace(".png", "").title()
                            with st.expander(clean, expanded=True):
                                st.image(str(img_path), width='stretch')

                    if other_imgs:
                        st.markdown("**📋 Other Report Figures**")
                        for label, img_path in other_imgs:
                            clean = label.replace("_", " ").replace(".png", "").title()
                            with st.expander(clean, expanded=False):
                                st.image(str(img_path), width='stretch')


def _HP_GUIDE_DATA() -> dict:
    """Hyperparameter guide data — ranges and strategies for each algorithm."""
    return {
        "Logistic Regression": {
            "icon": "\U0001f4c8",
            "description": "Linear baseline. Interpretable, fast. Degrades on non-linear boundaries.",
            "search_strategy": "Grid Search (small HP space)",
            "use_cases": ["A","B","C_nlp","E"],
            "params": [
                {"name":"C (inv. regularisation)", "recommended":"[0.001, 0.01, 0.1, 1, 10, 100]", "default":"1.0"},
                {"name":"penalty",                  "recommended":"['l1', 'l2']",                   "default":"l2"},
                {"name":"solver",                   "recommended":"liblinear (for l1), lbfgs (l2)",  "default":"lbfgs"},
                {"name":"max_iter",                 "recommended":"[100, 500, 1000]",                "default":"100"},
            ],
        },
        "Random Forest": {
            "icon": "\U0001f333",
            "description": "Ensemble of decorrelated trees. Good default; interpretable via feature importance.",
            "search_strategy": "Random Search (3\u20136 HPs)",
            "use_cases": ["A","B","E"],
            "params": [
                {"name":"n_estimators",   "recommended":"[100, 300, 500]",        "default":"100"},
                {"name":"max_depth",      "recommended":"[None, 5, 10, 20]",      "default":"None"},
                {"name":"max_features",   "recommended":"['sqrt', 'log2', 0.3]", "default":"sqrt"},
                {"name":"min_samples_leaf","recommended":"[1, 5, 10, 20]",        "default":"1"},
                {"name":"class_weight",   "recommended":"[None, 'balanced']",    "default":"None"},
            ],
        },
        "XGBoost": {
            "icon": "\U0001f680",
            "description": "Gradient-boosted trees with L1/L2 reg. State-of-the-art for tabular data.",
            "search_strategy": "Bayesian / Optuna TPE (50\u2013100 trials)",
            "use_cases": ["A","B","E"],
            "params": [
                {"name":"n_estimators",    "recommended":"[200, 1 000] + early stop", "default":"100"},
                {"name":"max_depth",       "recommended":"[3, 10]",                  "default":"6"},
                {"name":"learning_rate",   "recommended":"[0.01, 0.3] log-uniform",  "default":"0.3"},
                {"name":"subsample",       "recommended":"[0.6, 1.0]",              "default":"1.0"},
                {"name":"colsample_bytree","recommended":"[0.5, 1.0]",              "default":"1.0"},
                {"name":"reg_alpha (L1)", "recommended":"[1e-4, 10] log",           "default":"0"},
                {"name":"reg_lambda (L2)","recommended":"[1e-4, 10] log",           "default":"1"},
                {"name":"scale_pos_weight","recommended":"neg/pos ratio",           "default":"1"},
            ],
        },
        "LightGBM": {
            "icon": "\u26a1",
            "description": "Leaf-wise boosting. Faster than XGBoost on large datasets; best PR-AUC on imbalanced.",
            "search_strategy": "Bayesian / Optuna TPE (50 trials, 3-fold CV)",
            "use_cases": ["A","B","C_nlp","E"],
            "params": [
                {"name":"n_estimators",      "recommended":"[300, 1 500] + early stop", "default":"100"},
                {"name":"num_leaves",        "recommended":"[31, 255]",                "default":"31"},
                {"name":"max_depth",         "recommended":"[4, 12]",                 "default":"-1"},
                {"name":"learning_rate",     "recommended":"[0.01, 0.2] log",         "default":"0.1"},
                {"name":"feature_fraction",  "recommended":"[0.4, 1.0]",             "default":"1.0"},
                {"name":"bagging_fraction",  "recommended":"[0.4, 1.0]",             "default":"1.0"},
                {"name":"lambda_l1",         "recommended":"[1e-4, 10] log",         "default":"0"},
                {"name":"lambda_l2",         "recommended":"[1e-4, 10] log",         "default":"0"},
                {"name":"scale_pos_weight",  "recommended":"neg/pos ratio",          "default":"1"},
            ],
        },
        "Decision Tree": {
            "icon": "\U0001f333",
            "description": "Single tree baseline. Highly interpretable; prone to overfitting without depth limits.",
            "search_strategy": "Grid Search",
            "use_cases": ["A","B","E"],
            "params": [
                {"name":"max_depth",       "recommended":"[3, 5, 10, None]",    "default":"None"},
                {"name":"min_samples_leaf","recommended":"[1, 5, 10, 20, 50]", "default":"1"},
                {"name":"max_features",   "recommended":"[None, 'sqrt', 0.3]","default":"None"},
                {"name":"criterion",      "recommended":"['gini', 'entropy']","default":"gini"},
            ],
        },
        "MLP Neural Network": {
            "icon": "\U0001f9e0",
            "description": "Multi-layer perceptron. Captures non-linear patterns; sensitive to scaling.",
            "search_strategy": "Random Search",
            "use_cases": ["A","B"],
            "params": [
                {"name":"hidden_layer_sizes","recommended":"[(64,), (128,), (64,32), (128,64)]","default":"(100,)"},
                {"name":"activation",        "recommended":"['relu', 'tanh']",                "default":"relu"},
                {"name":"alpha (L2 reg)",    "recommended":"[1e-4, 1e-2]",                    "default":"1e-4"},
                {"name":"learning_rate_init","recommended":"[1e-3, 1e-2]",                    "default":"1e-3"},
                {"name":"max_iter",          "recommended":"[200, 500]",                       "default":"200"},
            ],
        },
    }


def page_model_development(uc_key: str) -> None:
    uc = USE_CASE_META.get(uc_key, {})
    section_header(
        f"\U0001f916 Model Development \u2014 {uc['icon']} {uc['title']}",
        "Algorithm comparison via stratified CV, then hyperparameter tuning on the champion.",
    )

    with st.expander("\U0001f4da ML Framework: How CV and HP Tuning are nested", expanded=False):
        st.markdown("""
**The DSF504 modelling phase separates algorithm selection from tuning:**

**Step 4 \u2014 Algorithm Selection (outer CV loop):**
Train 4\u20135 candidate models using 5-fold stratified CV. The CV score estimates
generalisation *before* seeing validation data. This avoids "peeking" at validation
when choosing your algorithm.

**Step 5 \u2014 Hyperparameter Tuning (inner Optuna search):**
Run Bayesian TPE search *only* on the Step-4 CV champion. A 3-fold inner CV is used
for speed \u2014 tuning needs many trials; a rough-but-fast signal beats a slow-but-accurate
one across 50+ trials.

**Correct nesting prevents leakage:**
- Outer CV selects the algorithm (no HP choices yet)
- Inner search finds HPs on the same training folds
- Validation set is *only* touched by Step 5\'s final evaluation
        """)

    st.markdown("---")
    _mc_check = ROOT / "reports" / USE_CASE_META.get(uc_key, {}).get("report_dir", "") / "model_comparison.csv"
    if not _mc_check.exists():
        _run_step_action(4, uc_key, "\u25b6 Run Step 4 \u2014 Algorithm Comparison  (goes to Run Pipeline)", suffix="md_s4")
    st.markdown("---")

    tab_cv, tab_compare, tab_hp = st.tabs([
        "\U0001f4d0 CV Explorer",
        "\U0001f4ca Model Comparison",
        "\U0001f527 HP Tuning Guide",
    ])

    # \u2500\u2500 CV EXPLORER \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with tab_cv:
        with st.expander("\U0001f4da Teaching Note \u2014 Cross-validation fundamentals", expanded=False):
            st.markdown("""
**Why cross-validation (CV) instead of a single train/val split?**

A single split is a *lottery*: you might get lucky or unlucky depending on which rows end
up in validation. CV averages across *k* different splits, giving a much more reliable
estimate of generalisation performance.

**Stratified K-Fold** (used here, k = 5):
1. Shuffle the dataset
2. Split into 5 folds; each fold preserves the target class ratio
3. Train on 4 folds, evaluate on the 1 held-out fold
4. Repeat 5 times \u2014 the CV score is the *mean* across all 5 folds

**What to look for in the CV results:**
- **High mean score + low std** \u2192 stable, reliable model
- **High mean + high std** \u2192 model is sensitive to which data it sees (consider more data or simpler model)
- **Low mean + low std** \u2192 model is consistently poor (try different algorithm or more features)
            """)

        r_dir  = ROOT / "reports" / uc.get("report_dir", "")
        cv_csv = r_dir / "model_comparison.csv"
        cv_png = r_dir / "model_cv_comparison.png"

        if cv_csv.exists():
            df_cv = pd.read_csv(cv_csv)
            st.markdown(f"**{len(df_cv)} models compared via cross-validation**")
            st.dataframe(df_cv, width="stretch", hide_index=True)
        else:
            st.info("CV results not found. Run **Step 4 \u2014 Algorithm Comparison**.")
            _run_step_action(4, uc_key, "\u25b6 Run Step 4  (goes to Run Pipeline)", suffix="cv_s4b")

        if cv_png.exists():
            st.image(str(cv_png), width="stretch")

        # ROC/PR curves
        roc_png = r_dir / "model_roc_pr_curves.png"
        if roc_png.exists():
            st.markdown("#### ROC & PR Curves")
            st.image(str(roc_png), width="stretch")

    # \u2500\u2500 MODEL COMPARISON \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with tab_compare:
        r_dir = ROOT / "reports" / uc.get("report_dir", "")
        is_regr = "Regression" in uc.get("task", "")

        # Feature importance PNGs for each model
        fi_pngs = list(r_dir.glob("feature_importance_*.png"))
        if fi_pngs:
            st.markdown("#### Feature Importance by Model")
            cols_fi = st.columns(min(2, len(fi_pngs)))
            for i, png in enumerate(fi_pngs[:4]):
                with cols_fi[i % 2]:
                    model_name = png.stem.replace("feature_importance_", "").replace("_", " ")
                    st.markdown(f"**{model_name}**")
                    st.image(str(png), width="stretch")
        else:
            # Fallback: look for model_comparison.png or shap_feature_importance.png
            _fi_fallbacks = ["model_comparison.png", "shap_feature_importance.png",
                             "feature_target_correlation.png"]
            _fi_shown = False
            for _fb in _fi_fallbacks:
                _fb_path = r_dir / _fb
                if _fb_path.exists():
                    st.markdown("#### Model Comparison")
                    st.image(str(_fb_path), width="stretch")
                    _fi_shown = True
                    break
            if not _fi_shown:
                st.info("Feature importance plots not found. Run Step 4.")

        # Confusion matrices
        conf_png = r_dir / "confusion_matrices.png"
        if not is_regr and conf_png.exists():
            st.markdown("#### Confusion Matrices")
            st.image(str(conf_png), width="stretch")

        # Step 5 summary
        st.markdown("---")
        st.markdown(f"**Step 5 \u2014 {STEP_NAMES[5]}**")
        st.markdown(STEP_DESCRIPTIONS[5])
        _run_step_action(5, uc_key, "\u25b6 Run Step 5 \u2014 HP Tuning  (goes to Run Pipeline)", suffix="md_s5")

    # \u2500\u2500 HP TUNING GUIDE \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with tab_hp:
        st.markdown(
            "<p style='color:#888;font-size:0.88rem;margin-bottom:8px;'>"
            "Evidence-based HP ranges and search strategies for each algorithm.</p>",
            unsafe_allow_html=True,
        )
        with st.expander("\U0001f4da Which search strategy should I use?", expanded=False):
            st.markdown("""
| Strategy | HP count | Best for |
|----------|----------|----------|
| **Grid Search** | 1\u20133 HPs | Logistic Regression, Decision Tree |
| **Random Search** | 3\u20136 HPs | Random Forest, MLP |
| **Bayesian / Optuna TPE** | 5\u201310+ HPs | XGBoost, LightGBM |

**Why Bayesian > Random for boosted trees:**
Optuna TPE models which HP regions are promising, so each trial informs the next.
For LightGBM with 8+ HPs this typically finds a better solution in 50 trials than
random search does in 200.

**Early stopping as a free HP:**
For boosted trees, set `early_stopping_rounds=50` with a validation set. This makes
`n_estimators` a derived quantity, freeing search budget for the regularisation HPs.
            """)

        hp_guide = _HP_GUIDE_DATA()
        show_all = st.checkbox("Show all algorithms", value=False, key=f"_hp_showall_{uc_key}")
        model_list = (list(hp_guide.keys()) if show_all
                      else [m for m, v in hp_guide.items() if uc_key in v["use_cases"]])
        if not model_list:
            st.info("No HP guide configured for this use case.")
        else:
            hp_tabs = st.tabs([f"{hp_guide[m]['icon']} {m}" for m in model_list])
            for htab, mname in zip(hp_tabs, model_list):
                with htab:
                    guide = hp_guide[mname]
                    st.markdown(f"**{guide['description']}**")
                    st.markdown(f"*Search strategy:* {guide['search_strategy']}")
                    param_rows = [
                        {"Hyperparameter": p["name"],
                         "Recommended range": p["recommended"],
                         "Default": p["default"]}
                        for p in guide["params"]
                    ]
                    st.dataframe(pd.DataFrame(param_rows), width="stretch", hide_index=True)



def page_model_performance(uc_key: str) -> None:
    uc    = USE_CASE_META.get(uc_key, {})
    r_dir = ROOT / "reports" / uc.get("report_dir", "")
    m_dir = ROOT / "models"  / uc.get("model_dir",  "")
    section_header(
        f"📊 Model Evaluation — {uc['icon']} {uc['title']}",
        "Champion model performance metrics, curves, and threshold calibration.",
    )

    is_regr = "Regression" in uc.get("task", "")

    tab_metrics, tab_curves, tab_confusion, tab_thresh = st.tabs([
        "📈 Metrics",
        "📉 ROC / PR Curves",
        "🔢 Confusion Matrix",
        "⚖️ Threshold",
    ])

    # ── Metrics ────────────────────────────────────────────────────────────────
    with tab_metrics:
        # Try final_model_metrics first, then model_comparison
        metrics_loaded = False
        for csv_name in ["final_model_metrics.csv"]:
            csv_path = r_dir / csv_name
            if csv_path.exists():
                df_m = pd.read_csv(csv_path)
                st.markdown(f"**Champion Model Metrics**")
                st.dataframe(df_m, width='stretch', hide_index=True)

                # Show KPI cards for numeric single-row metrics
                if len(df_m) == 1:
                    num_cols = df_m.select_dtypes(include="number").columns.tolist()
                    kv = {c: fmt_num(df_m.iloc[0][c]) for c in num_cols[:6]}
                    cols = st.columns(min(len(kv), 4))
                    for i, (k, v) in enumerate(kv.items()):
                        cols[i % len(cols)].markdown(metric_card(k, v, colour=BLUE), unsafe_allow_html=True)
                metrics_loaded = True
                break

        if not metrics_loaded:
            # Try model_comparison
            mc_path = r_dir / "model_comparison.csv"
            if mc_path.exists():
                df_mc = pd.read_csv(mc_path)
                st.markdown("**Model Comparison (last step)**")
                st.dataframe(df_mc, width='stretch', hide_index=True)
            else:
                st.info("Metrics not found. Run Steps 4–5 to train the model.")
                _run_step_action(4, uc_key, "▶ Run Steps 4–5  (goes to Run Pipeline)", suffix="eval_m")

        # Val prediction PNGs for regression
        if is_regr:
            for png_name in ["final_model_preds_val.png", "val_pred_vs_actual_lightgbm.png",
                             "val_pred_vs_actual_xgboost.png", "model_comparison.png"]:
                png_path = r_dir / png_name
                if png_path.exists():
                    st.image(str(png_path), width='stretch')
                    break

    # ── ROC / PR Curves ────────────────────────────────────────────────────────
    with tab_curves:
        if is_regr:
            png_path = r_dir / "final_model_preds_test.png"
            if png_path.exists():
                st.image(str(png_path), width='stretch')
            else:
                st.info("Prediction plots not found. Run Step 5.")
        else:
            _roc_candidates = ["model_roc_pr_curves.png", "roc_pr_curves.png",
                               "roc_curves.png", "champion_evaluation.png"]
            _roc_shown = False
            for _rc in _roc_candidates:
                _rc_path = r_dir / _rc
                if _rc_path.exists():
                    st.image(str(_rc_path), width='stretch')
                    _roc_shown = True
                    break
            if not _roc_shown:
                st.info("ROC/PR curves not found. Run Step 4.")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    with tab_confusion:
        if is_regr:
            st.info("Confusion matrix is not applicable for regression tasks.")
        else:
            for png_name in ["confusion_matrices.png", "confusion_matrix.png",
                             "champion_evaluation.png"]:
                png_path = r_dir / png_name
                if png_path.exists():
                    st.image(str(png_path), width='stretch')
                    break
            else:
                st.info("Confusion matrix image not found. Run Step 4.")

    # ── Threshold calibration ──────────────────────────────────────────────────
    with tab_thresh:
        if is_regr:
            st.info("Threshold calibration is not applicable for regression tasks.")
            return

        thresh_png = r_dir / "champion_threshold_calibration.png"
        if thresh_png.exists():
            st.image(str(thresh_png), width='stretch')

        # Load champion model and feature cols for threshold slider
        champion_name = uc.get("champion", "lgbm_optuna_champion.pkl")
        model = load_model(m_dir / champion_name)
        feat_pkl = m_dir / "feature_cols.pkl"
        feat_cols = joblib.load(feat_pkl) if feat_pkl.exists() else None

        if model is None:
            st.info("Champion model not found. Run Step 5.")
            return

        # Try to load validation data
        data_dir  = ROOT / "data" / uc.get("data_dir", "")
        val_path  = data_dir / "val_fe.parquet"
        if val_path.exists() and feat_cols is not None:
            df_val = pd.read_parquet(val_path)
            target_col = uc.get("target", "")
            feat_cols_use = [c for c in feat_cols if c in df_val.columns]
            if feat_cols_use and target_col in df_val.columns:
                X_val = df_val[feat_cols_use]
                y_val = df_val[target_col]
                try:
                    proba = model.predict_proba(X_val)[:, 1]
                    threshold = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01,
                                         key=f"_thresh_{uc_key}")
                    y_pred = (proba >= threshold).astype(int)
                    tp = int(((y_pred == 1) & (y_val == 1)).sum())
                    fp = int(((y_pred == 1) & (y_val == 0)).sum())
                    fn = int(((y_pred == 0) & (y_val == 1)).sum())
                    tn = int(((y_pred == 0) & (y_val == 0)).sum())
                    prec = tp / (tp + fp + 1e-9)
                    rec  = tp / (tp + fn + 1e-9)
                    f1   = 2 * prec * rec / (prec + rec + 1e-9)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Precision", f"{prec:.4f}")
                    c2.metric("Recall",    f"{rec:.4f}")
                    c3.metric("F1",        f"{f1:.4f}")
                    c4.metric("Threshold", f"{threshold:.2f}")

                    # Confusion matrix heatmap
                    cm_data = [[tn, fp], [fn, tp]]
                    fig_cm = px.imshow(
                        cm_data,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Negative", "Positive"], y=["Negative", "Positive"],
                        color_continuous_scale="Blues",
                        title=f"Confusion Matrix @ threshold={threshold:.2f}",
                        text_auto=True,
                    )
                    fig_cm.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT, height=340)
                    st.plotly_chart(fig_cm, width='stretch')
                except Exception as e:
                    st.warning(f"Could not compute threshold metrics: {e}")
        else:
            st.info("Validation data or feature columns not found. Run Steps 1–5.")


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Ethics & Explainability ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page_explainability(uc_key: str) -> None:
    uc    = USE_CASE_META.get(uc_key, {})
    r_dir = ROOT / "reports" / uc.get("report_dir", "")
    section_header(
        f"🔍 Ethics & Explainability — {uc['icon']} {uc['title']}",
        "SHAP feature importance, bias audit, and fairness analysis.",
    )

    tab_global, tab_plots, tab_local, tab_bias = st.tabs([
        "🌍 Global Importance",
        "🖼️ SHAP Plots",
        "🔎 Local Explanation",
        "⚖️ Bias Audit",
    ])

    # ── Global importance ──────────────────────────────────────────────────────
    with tab_global:
        shap_csv = next((r_dir / n for n in ["shap_feature_importance.csv", "shap_importance.csv"] if (r_dir / n).exists()), None)
        if shap_csv is None:
            st.warning(
                "SHAP importance CSV not found. Run **Step 6 — Ethics & Explainability** to generate SHAP visualisations."
            )
            _run_step_action(6, uc_key, "▶ Run Step 6 — Ethics & Explainability  (goes to Run Pipeline)", suffix="shap_warn")
            return

        df_shap = pd.read_csv(shap_csv)
        feat_col = df_shap.columns[0]
        val_col  = df_shap.columns[1] if len(df_shap.columns) > 1 else None

        st.markdown(f"**{len(df_shap):,} features ranked by mean |SHAP|**")
        st.dataframe(df_shap, width='stretch', hide_index=True)

        if val_col:
            top_n = st.slider("Top N features", 5, min(50, len(df_shap)), 20, key=f"_shap_n_{uc_key}")
            df_top = df_shap.head(top_n).sort_values(val_col, ascending=True)
            fig = go.Figure(go.Bar(
                x=df_top[val_col], y=df_top[feat_col],
                orientation="h",
                marker_color=BLUE,
                hovertemplate="%{y}: %{x:.5f}<extra></extra>",
            ))
            fig.update_layout(
                plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT,
                height=max(300, top_n * 22),
                margin=dict(t=40, b=30, l=200, r=20),
                title=f"Top {top_n} Features by Mean |SHAP|",
                xaxis_title="Mean |SHAP|",
                yaxis_title="Feature",
            )
            st.plotly_chart(fig, width='stretch')

        # Feature importance PNGs
        fi_png = r_dir / "feature_importance.png"
        if fi_png.exists():
            st.image(str(fi_png), width='stretch')

    # ── SHAP Plots ─────────────────────────────────────────────────────────────
    with tab_plots:
        found_plots = False
        for png_name in ["shap_bar_importance.png", "shap_beeswarm.png", "shap_summary.png"]:
            png_path = r_dir / png_name
            if png_path.exists():
                st.image(str(png_path), width='stretch')
                found_plots = True

        if not found_plots:
            st.warning(
                "No pre-computed SHAP plots found. Run **Step 6 — Ethics & Explainability** to generate SHAP visualisations."
            )
            _run_step_action(6, uc_key, "▶ Run Step 6 — Ethics & Explainability  (goes to Run Pipeline)", suffix="shap_plots")

    # ── Local Explanation ──────────────────────────────────────────────────────
    with tab_local:
        shap_csv = next((r_dir / n for n in ["shap_feature_importance.csv", "shap_importance.csv"] if (r_dir / n).exists()), None)
        if shap_csv is None:
            st.info("Run Step 6 first to generate SHAP data.")
            return

        m_dir  = ROOT / "models" / uc.get("model_dir", "")
        champion_name = uc.get("champion", "lgbm_optuna_champion.pkl")
        model  = load_model(m_dir / champion_name)
        feat_pkl = m_dir / "feature_cols.pkl"
        feat_cols = joblib.load(feat_pkl) if feat_pkl.exists() else None

        is_regr = "Regression" in uc.get("task", "")

        if model is None or feat_cols is None:
            st.info("Champion model or feature list not found. Run Steps 4–5.")
            return

        data_dir = ROOT / "data" / uc.get("data_dir", "")
        val_path = data_dir / "val_fe.parquet"
        if not val_path.exists():
            st.info("Validation data not found. Run Step 1.")
            return

        df_val = pd.read_parquet(val_path)
        feat_cols_use = [c for c in feat_cols if c in df_val.columns]
        if not feat_cols_use:
            st.info("Feature columns not found in validation data.")
            return

        X_val = df_val[feat_cols_use]

        sample_idx = st.number_input(
            "Sample index (row in val set)", min_value=0,
            max_value=len(X_val) - 1, value=0, step=1,
            key=f"_local_idx_{uc_key}",
        )
        row = X_val.iloc[[sample_idx]]

        try:
            if is_regr:
                pred = model.predict(row)[0]
                st.metric("Predicted value", f"{pred:.6f}")
            else:
                proba = model.predict_proba(row)[0]
                target_col = uc.get("target", "")
                if len(proba) == 2:
                    st.metric("Predicted probability (positive class)", f"{proba[1]:.4f}")
                else:
                    for i, p in enumerate(proba):
                        st.metric(f"Class {i} probability", f"{p:.4f}")

            # Simple feature value display
            row_display = row.T.reset_index()
            row_display.columns = ["Feature", "Value"]
            df_shap = pd.read_csv(shap_csv)
            top_feats = df_shap.iloc[:, 0].head(20).tolist()
            row_top = row_display[row_display["Feature"].isin(top_feats)]
            st.markdown("**Top SHAP features for this sample (values only)**")
            st.dataframe(row_top, width='stretch', hide_index=True)
        except Exception as e:
            st.warning(f"Could not compute local explanation: {e}")

    # ── Bias Audit ─────────────────────────────────────────────────────────────
    with tab_bias:
        with st.expander("📚 How to read the Bias Audit charts", expanded=False):
            st.markdown(f"""
<p style='color:{FONT};font-size:0.9rem;'>
A bias audit evaluates whether the model's predictions are <b>equitable across subgroups</b> —
for example, by risk score, product type, or any available demographic proxy.
This is distinct from SHAP explainability: SHAP tells you <em>why</em> the model makes a prediction;
bias audit tells you <em>who</em> the model treats differently.
</p>

**What each column means:**

| Column | What it measures |
|--------|-----------------|
| `subgroup` / first categorical | The group being evaluated (e.g., risk tier, region, score bucket) |
| `mean_pred` / `avg_score` | Average predicted probability or score for this group |
| `actual_rate` | Observed positive-class rate in the validation set for this group |
| `approval_rate` | Share of cases in this group that would be approved at a given threshold |
| `fpr` / `false_positive_rate` | Share of true negatives incorrectly flagged as positive — a **fairness-critical metric** |
| `fnr` / `false_negative_rate` | Share of true positives missed — the **recall gap** across groups |

**Key fairness concepts to look for:**

- **Demographic parity gap:** If `approval_rate` differs sharply between groups, the model may be "disparate impact" under regulatory frameworks (e.g., ECOA in lending, EU AI Act). A gap > 20 percentage points typically warrants investigation.
- **Equalised odds:** A fair model should have similar `fpr` and `fnr` across groups. If the false positive rate is much higher for one group, that group is disproportionately penalised.
- **Calibration:** The model's predicted probability should match the actual positive rate within each group. A group where `mean_pred = 0.15` but `actual_rate = 0.35` is systematically under-scored.

**Reading the bar chart:**
- The colour gradient (green → red) visualises the selected metric across groups.
- **Look for outlier bars** — groups that are significantly higher or lower than the overall average deserve investigation.
- Switch the metric selector to compare multiple fairness dimensions (FPR, FNR, mean score) for the same groups.

**What bias audit does NOT tell you:**
- It does not prove discrimination — disparity in outcomes can reflect legitimate risk differences. Fairness analysis provides signals for human review, not automatic conclusions.
- It does not replace legal compliance analysis. Always involve domain experts and legal counsel before making policy decisions based on model outputs.
            """, unsafe_allow_html=True)


        bias_csv = r_dir / "ethics_bias_report.csv"
        if bias_csv.exists():
            df_bias = pd.read_csv(bias_csv)
            st.markdown(f"**Bias audit report — {len(df_bias)} subgroups**")
            st.dataframe(df_bias, width='stretch', hide_index=True)

            # Fairness chart
            num_cols = df_bias.select_dtypes(include="number").columns.tolist()
            cat_cols = df_bias.select_dtypes(include=["object","category","str"]).columns.tolist()
            if cat_cols and num_cols:
                grp_col = cat_cols[0]
                metric_col = st.selectbox("Metric to visualise", num_cols, key=f"_bias_metric_{uc_key}")
                fig = px.bar(
                    df_bias, x=grp_col, y=metric_col,
                    color=metric_col, color_continuous_scale="RdYlGn",
                    title=f"Fairness Audit — {metric_col} by {grp_col}",
                )
                fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT, height=380)
                st.plotly_chart(fig, width='stretch')

            # Bias PNG images
            for png_name in ["stock_group_fairness.png"]:
                png_path = r_dir / png_name
                if png_path.exists():
                    st.image(str(png_path), width='stretch')
        else:
            st.info("Bias report not found. Run Step 6 — Ethics & Explainability.")
            _run_step_action(6, uc_key, "▶ Run Step 6 — Ethics & Explainability  (goes to Run Pipeline)", suffix="bias")


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Prediction Demo (regression) ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Prediction Demo — helpers ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _demo_render_result(proba, target_labels: dict, champion_name: str) -> None:
    """Render prediction probabilities with coloured metric cards and a risk bar."""
    n_classes = len(proba)
    if n_classes == 2:
        p_pos    = float(proba[1])
        neg_lbl  = target_labels.get(0, "Negative")
        pos_lbl  = target_labels.get(1, "Positive")
        colour   = RED if p_pos >= 0.7 else ORG if p_pos >= 0.4 else GRN
        pred_lbl = pos_lbl if p_pos >= 0.5 else neg_lbl
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric(f"P({pos_lbl})",  f"{p_pos:.1%}")
        with c2:
            st.metric(f"P({neg_lbl})", f"{proba[0]:.1%}")
        with c3:
            bar_html = (
                f"<div style='margin-top:6px'>"
                f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<span style='color:{FONT};font-size:11px;width:72px'>{neg_lbl}</span>"
                f"<div style='flex:1;background:{GRID};border-radius:6px;height:18px'>"
                f"<div style='width:{p_pos*100:.1f}%;background:{colour};"
                f"height:100%;border-radius:6px'></div></div>"
                f"<span style='color:{FONT};font-size:11px;width:72px;text-align:right'>{pos_lbl}</span>"
                f"</div>"
                f"<p style='text-align:center;color:{colour};font-size:13px;margin:4px 0 0'>"
                f"<b>Predicted: {pred_lbl}</b> &nbsp;({p_pos:.1%} confidence)</p>"
                f"</div>"
            )
            st.markdown(bar_html, unsafe_allow_html=True)
    else:
        pred_class = int(proba.argmax())
        cols_m = st.columns(n_classes)
        for i, (col_m, p) in enumerate(zip(cols_m, proba)):
            lbl = target_labels.get(i, str(i))
            col_m.metric(f"P({lbl})", f"{float(p):.1%}",
                         delta="← Predicted" if i == pred_class else None)


# ── NLP text-input prediction demo ────────────────────────────────────────────

def _page_prediction_demo_nlp(uc_key: str) -> None:
    """Financial sentence → Bearish / Neutral / Bullish via TF-IDF + Complement NB."""
    m_dir      = ROOT / "models" / "use_case_C_nlp"
    model_path = m_dir / "Complement_NB_baseline.pkl"
    vec_path   = m_dir / "tfidf_vectorizer.pkl"

    if not model_path.exists():
        st.info("NLP model not found. Run Steps 4–5.")
        _run_step_action(5, uc_key, "▶ Run Step 5", suffix="nlp_demo")
        return
    if not vec_path.exists():
        st.info("TF-IDF vectorizer not found. Run Step 3.")
        return

    model      = load_model(model_path)
    vectorizer = load_model(vec_path)
    if model is None or vectorizer is None:
        st.error("Could not load model or vectorizer.")
        return

    tgt_labels = _FE_EDA_SRC.get(uc_key, {}).get("target_labels", {})

    st.markdown("#### 💬 Financial Sentiment Prediction")
    st.markdown(
        f"<p style='color:{FONT}'>Enter any financial headline or sentence. "
        f"The Complement Naïve Bayes classifier will label it "
        f"<b>Bearish</b>, <b>Neutral</b>, or <b>Bullish</b>.</p>",
        unsafe_allow_html=True,
    )

    examples = [
        "The company reported record revenues with a 25% increase in net income.",
        "Operating costs exceeded expectations, resulting in a significant net loss.",
        "The board approved the merger; details remain pending regulatory review.",
        "Quarterly earnings were largely in line with analyst consensus estimates.",
        "The firm faces bankruptcy proceedings after defaulting on its bond payments.",
    ]

    col_txt, col_ex = st.columns([3, 1])
    with col_ex:
        st.markdown(
            f"<p style='color:{FONT};font-size:12px;margin-bottom:4px'>Try an example:</p>",
            unsafe_allow_html=True,
        )
        for ex in examples:
            short = ex[:46] + "…"
            if st.button(short, key=f"_nlp_ex_{abs(hash(ex))}_{uc_key}",
                         use_container_width=True):
                st.session_state[f"_nlp_text_{uc_key}"] = ex

    with col_txt:
        text_input = st.text_area(
            "Financial text",
            value=st.session_state.get(f"_nlp_text_{uc_key}", ""),
            height=130,
            placeholder=(
                "e.g. The company’s profits rose 15% this quarter, "
                "beating analyst expectations."
            ),
            key=f"_nlp_textarea_{uc_key}",
        )

    if st.button("\U0001f52e Classify Sentiment", type="primary",
                 key=f"_nlp_pred_{uc_key}"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            try:
                X     = vectorizer.transform([text_input.strip()])
                proba = model.predict_proba(X)[0]
                st.divider()
                _demo_render_result(proba, tgt_labels, "Complement_NB_baseline.pkl")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with st.expander("\U0001f4ca Model performance summary", expanded=False):
        comp_csv = ROOT / "reports" / "use_case_C_nlp" / "model_comparison.csv"
        if comp_csv.exists():
            st.dataframe(pd.read_csv(comp_csv), width='stretch', hide_index=True)


# ── Regression prediction demo ─────────────────────────────────────────────────

def _page_prediction_demo_regression(uc_key: str) -> None:
    uc    = USE_CASE_META.get(uc_key, {})
    m_dir = ROOT / "models" / uc.get("model_dir", "")
    champion_name = uc.get("champion", "champion.pkl")
    model = load_model(m_dir / champion_name)
    feat_pkl = m_dir / "feature_cols.pkl"
    feat_cols = joblib.load(feat_pkl) if feat_pkl.exists() else None

    if model is None:
        st.info("Champion model not found. Run Step 5.")
        _run_step_action(5, uc_key, "▶ Run Step 5  (goes to Run Pipeline)", suffix="rdemo")
        return
    if feat_cols is None:
        st.info("Feature columns not found. Run Steps 4–5.")
        return

    st.markdown("#### Live Regression Prediction Demo")
    st.markdown(
        f"<p style='color:{FONT}'>Adjust feature values below and click <b>Predict</b> "
        f"for a real-time prediction from the champion model "
        f"(<code>{champion_name}</code>).</p>",
        unsafe_allow_html=True,
    )

    data_dir = ROOT / "data" / uc.get("data_dir", "")
    val_path = data_dir / "val_fe.parquet"
    df_ref   = pd.read_parquet(val_path) if val_path.exists() else None

    feat_cols_use = feat_cols[:20]
    input_vals: dict = {}

    with st.form(key=f"_regr_form_{uc_key}"):
        n_per_row = 4
        rows = [feat_cols_use[i:i+n_per_row] for i in range(0, len(feat_cols_use), n_per_row)]
        for row_feats in rows:
            cols = st.columns(len(row_feats))
            for col, feat in zip(cols, row_feats):
                default_val = 0.0
                if df_ref is not None and feat in df_ref.columns:
                    default_val = float(df_ref[feat].median())
                input_vals[feat] = col.number_input(feat, value=default_val, format="%.6f")
        submitted = st.form_submit_button("\U0001f52e Predict", type="primary")

    if submitted:
        X_pred = pd.DataFrame([{f: input_vals.get(f, 0.0) for f in feat_cols_use}])
        try:
            pred = model.predict(X_pred)[0]
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Predicted Value", f"{pred:.6f}")
            with c2:
                if df_ref is not None:
                    target_col = uc.get("target", "target")
                    if target_col in df_ref.columns:
                        st.metric("Val-set mean", f"{df_ref[target_col].mean():.6f}")
            st.success(f"Champion `{champion_name}` predicted: **{pred:.6f}**")
        except Exception as _e:
            st.error(f"Prediction failed: {_e}")

    if df_ref is not None:
        with st.expander("\U0001f4ca Validation-set feature statistics", expanded=False):
            show_cols = [c for c in feat_cols_use if c in df_ref.columns]
            st.dataframe(df_ref[show_cols].describe().T.round(4),
                         width='stretch', hide_index=False)


# ── Smart feature form for classification ─────────────────────────────────────

def _cls_build_input_form(
    feat_cols_show: list,
    df_ref,
    form_key: str,
    champion_name: str,
) -> "dict | None":
    """Render a per-feature input form; returns {feat: value} dict on submit, else None."""
    input_vals: dict = {}

    st.markdown(
        f"<p style='color:{FONT};font-size:13px'>"
        f"Defaults are validation-set medians. Binary flags show Yes/No selectors; "
        f"low-cardinality integers show a dropdown; continuous values use a numeric input.</p>",
        unsafe_allow_html=True,
    )

    with st.form(key=f"_cls_form_{form_key}"):
        n_per_row = 3
        rows = [feat_cols_show[i:i+n_per_row]
                for i in range(0, len(feat_cols_show), n_per_row)]
        for row_feats in rows:
            grid_cols = st.columns(len(row_feats))
            for gc, feat in zip(grid_cols, row_feats):
                default_val: float = 0.0
                col_vals = None
                if df_ref is not None and feat in df_ref.columns:
                    col_vals    = df_ref[feat].dropna()
                    default_val = float(col_vals.median())

                # Determine widget type from column statistics
                if col_vals is not None and len(col_vals):
                    unique_vals = sorted(col_vals.unique())
                    n_unique    = len(unique_vals)
                    all_int     = all(
                        float(v) == int(float(v))
                        for v in unique_vals if pd.notna(v)
                    )
                    is_binary = (
                        n_unique == 2
                        and set(float(v) for v in unique_vals).issubset({0.0, 1.0})
                    )

                    if is_binary:
                        chosen = gc.selectbox(
                            feat, options=[0, 1],
                            index=int(round(default_val)),
                            format_func=lambda v: f"Yes (1)" if v else f"No (0)",
                        )
                        input_vals[feat] = float(chosen)
                        continue

                    if n_unique <= 8 and all_int:
                        opts = [int(float(v)) for v in unique_vals]
                        def_int = int(round(default_val))
                        def_idx = opts.index(def_int) if def_int in opts else 0
                        chosen = gc.selectbox(feat, options=opts, index=def_idx)
                        input_vals[feat] = float(chosen)
                        continue

                # Continuous numeric input with val-set bounds
                mn = float(col_vals.min()) if col_vals is not None and len(col_vals) else None
                mx = float(col_vals.max()) if col_vals is not None and len(col_vals) else None
                input_vals[feat] = gc.number_input(
                    feat,
                    value=default_val,
                    min_value=mn,
                    max_value=mx,
                    format="%.4f",
                )

        submitted = st.form_submit_button(
            f"\U0001f52e Predict  ({champion_name})", type="primary"
        )

    return input_vals if submitted else None


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Prediction Demo (dispatcher) ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page_prediction_demo(uc_key: str) -> None:
    uc = USE_CASE_META.get(uc_key, {})
    section_header(
        f"\U0001f3af Prediction Demo — {uc['icon']} {uc['title']}",
        "Live inference on the tuned champion model.",
    )
    warn = _prereq_warning("\U0001f3af Prediction Demo", uc_key)
    if warn:
        st.warning(warn)
        _run_step_action(5, uc_key, "▶ Run Steps 4–5  (goes to Run Pipeline)",
                         suffix="pdemo_warn")
        return

    task     = uc.get("task", "")
    is_regr  = "Regression" in task
    is_nlp   = uc.get("is_nlp", False)
    is_rank  = "Rank" in task

    if is_regr:
        _page_prediction_demo_regression(uc_key)
        return
    if is_nlp:
        _page_prediction_demo_nlp(uc_key)
        return
    if is_rank:
        st.info(
            "Learning-to-rank predictions operate over candidate-portfolio sets rather "
            "than single input vectors. See \U0001f4ca **Model Evaluation** for NDCG@10 "
            "scores and ranked portfolio previews."
        )
        return

    # ── Classification path ──────────────────────────────────────────────────
    m_dir         = ROOT / "models" / uc.get("model_dir", "")
    champion_name = uc.get("champion", "lgbm_optuna_champion.pkl")
    model         = load_model(m_dir / champion_name)
    feat_pkl      = m_dir / "feature_cols.pkl"
    feat_cols     = joblib.load(feat_pkl) if feat_pkl.exists() else None

    # Fallback: some UCs (E, F) register features in a CSV instead of a pkl
    if feat_cols is None:
        _fl_src = _FE_EDA_SRC.get(uc_key, {}).get("feat_list")
        if _fl_src:
            _fl_p = ROOT / _fl_src
            if _fl_p.exists():
                try:
                    feat_cols = pd.read_csv(_fl_p).iloc[:, 0].tolist()
                except Exception:
                    pass

    if model is None:
        st.info("Champion model not found. Run Step 5.")
        _run_step_action(5, uc_key, "▶ Run Step 5  (goes to Run Pipeline)",
                         suffix="pdemo_cls")
        return
    if feat_cols is None:
        st.info("Feature columns list not found. Run Steps 4–5.")
        return

    data_dir = ROOT / "data" / uc.get("data_dir", "")
    val_path = data_dir / "val_fe.parquet"
    if not val_path.exists():
        st.info("Validation data not found. Run Step 3.")
        return

    df_val        = pd.read_parquet(val_path)
    feat_cols_use = [c for c in feat_cols if c in df_val.columns]
    if not feat_cols_use:
        st.info("No feature columns found in validation data.")
        return

    X_val      = df_val[feat_cols_use]
    target_col = uc.get("target", "target")
    tgt_labels = _FE_EDA_SRC.get(uc_key, {}).get("target_labels", {})

    # Rank features by SHAP importance; fall back to first N
    r_dir    = ROOT / "reports" / uc.get("report_dir", "")
    shap_csv = next(
        (r_dir / n
         for n in ["shap_feature_importance.csv", "shap_importance.csv"]
         if (r_dir / n).exists()),
        None,
    )
    feat_cols_show = feat_cols_use[:12]
    if shap_csv is not None:
        try:
            shap_top = pd.read_csv(shap_csv).iloc[:, 0].tolist()
            ordered  = [f for f in shap_top if f in feat_cols_use]
            if ordered:
                feat_cols_show = ordered[:12]
        except Exception:
            pass

    # ── Page subtitle ──────────────────────────────────────────────────────
    n_shown = len(feat_cols_show)
    feat_src = "SHAP-ranked" if shap_csv else "first"
    st.markdown(
        f"<p style='color:{FONT};font-size:13px'>"
        f"Model: <code>{champion_name}</code> &nbsp;|&nbsp; "
        f"Task: <b>{task}</b> &nbsp;|&nbsp; "
        f"Showing {feat_src} {n_shown} of {len(feat_cols_use)} features</p>",
        unsafe_allow_html=True,
    )

    tab_custom, tab_sample = st.tabs([
        "\U0001f4dd  Custom Input",
        "\U0001f500  Sample Explorer",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # Tab 1 — Custom Input: user fills in feature values directly
    # ════════════════════════════════════════════════════════════════════════
    with tab_custom:
        result_vals = _cls_build_input_form(
            feat_cols_show, df_val,
            form_key=f"ci_{uc_key}",
            champion_name=champion_name,
        )
        if result_vals is not None:
            # Fill any non-shown features with val-set medians
            X_pred = pd.DataFrame([{
                f: result_vals.get(f,
                   float(df_val[f].median()) if f in df_val.columns else 0.0)
                for f in feat_cols
            }])
            try:
                proba = model.predict_proba(X_pred)[0]
                st.divider()
                _demo_render_result(proba, tgt_labels, champion_name)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # Tab 2 — Sample Explorer: pick a row from the validation set
    # ════════════════════════════════════════════════════════════════════════
    with tab_sample:
        st.markdown(
            f"<p style='color:{FONT};font-size:13px'>"
            f"Select a real record from the validation set to see "
            f"what the champion model predicts vs. the ground truth label.</p>",
            unsafe_allow_html=True,
        )

        n_cls = len(tgt_labels) if tgt_labels else 2
        has_target = target_col in df_val.columns

        if n_cls == 2:
            # Binary: three pick buttons in one row
            pos_lbl = tgt_labels.get(1, "Positive")
            neg_lbl = tgt_labels.get(0, "Negative")
            b1, b2, b3, _pad = st.columns([1, 1.2, 1.2, 3])

            if b1.button("\U0001f3b2 Any Sample",
                         key=f"_rnd_any_{uc_key}", use_container_width=True):
                idx = X_val.sample(1).index[0]
                st.session_state[f"_demo_idx_{uc_key}"]   = idx
                st.session_state[f"_demo_truth_{uc_key}"] = (
                    df_val.loc[idx, target_col] if has_target else None
                )

            if b2.button(f"✅ {pos_lbl}",
                         key=f"_rnd_pos_{uc_key}", use_container_width=True):
                if has_target:
                    pool = df_val[df_val[target_col] == 1]
                    if len(pool):
                        idx = pool.sample(1).index[0]
                        st.session_state[f"_demo_idx_{uc_key}"]   = idx
                        st.session_state[f"_demo_truth_{uc_key}"] = 1

            if b3.button(f"❌ {neg_lbl}",
                         key=f"_rnd_neg_{uc_key}", use_container_width=True):
                if has_target:
                    pool = df_val[df_val[target_col] == 0]
                    if len(pool):
                        idx = pool.sample(1).index[0]
                        st.session_state[f"_demo_idx_{uc_key}"]   = idx
                        st.session_state[f"_demo_truth_{uc_key}"] = 0

        else:
            # Multi-class: dropdown + single pick button
            class_opts = ["Any"] + [str(k) for k in tgt_labels]
            sel_cls    = st.selectbox("Filter by class", class_opts,
                                      key=f"_cls_sel_{uc_key}")
            if st.button("\U0001f3b2 Pick Random Sample",
                         key=f"_rnd_mc_{uc_key}"):
                if sel_cls == "Any" or not has_target:
                    idx = X_val.sample(1).index[0]
                else:
                    try:
                        cls_key = (int(sel_cls)
                                   if sel_cls.lstrip("-").isdigit()
                                   else sel_cls)
                    except Exception:
                        cls_key = sel_cls
                    pool = df_val[df_val[target_col] == cls_key]
                    idx  = (pool.sample(1).index[0]
                            if len(pool) else X_val.sample(1).index[0])
                st.session_state[f"_demo_idx_{uc_key}"]   = idx
                st.session_state[f"_demo_truth_{uc_key}"] = (
                    df_val.loc[idx, target_col] if has_target else None
                )

        # ── Show the selected sample ───────────────────────────────────────
        stored_idx = st.session_state.get(f"_demo_idx_{uc_key}")
        truth      = st.session_state.get(f"_demo_truth_{uc_key}")

        if stored_idx is not None:
            truth_lbl   = tgt_labels.get(truth, str(truth)) if truth is not None else "Unknown"
            # Colour code: positive/high-risk = red, negative/low-risk = green
            if truth in (1, "High"):
                t_colour = RED
            elif truth in (2, "Medium", "Neutral"):
                t_colour = ORG
            elif truth in (0, "Low", "Bearish"):
                t_colour = GRN
            else:
                t_colour = BLUE

            st.markdown(
                f"<p style='color:{FONT};margin:6px 0'>Ground truth: "
                f"<span style='background:{t_colour}22;color:{t_colour};"
                f"padding:2px 12px;border-radius:12px;font-weight:700'>"
                f"{truth_lbl}</span></p>",
                unsafe_allow_html=True,
            )

            # Table of SHAP-ranked feature values for this sample
            row      = X_val.loc[[stored_idx]]
            disp_cols = [c for c in feat_cols_show if c in row.columns]
            feat_df   = row[disp_cols].T.reset_index()
            feat_df.columns = ["Feature", "Value"]
            feat_df["Value"] = feat_df["Value"].apply(
                lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
            )
            st.dataframe(feat_df, width='stretch', hide_index=True)

            if st.button("\U0001f52e Predict this Sample",
                         key=f"_pred_sample_{uc_key}", type="primary"):
                X_pred = pd.DataFrame([{
                    f: (float(X_val.loc[stored_idx, f])
                        if f in X_val.columns
                        else float(df_val[f].median())
                        if f in df_val.columns
                        else 0.0)
                    for f in feat_cols
                }])
                try:
                    proba = model.predict_proba(X_pred)[0]
                    st.divider()
                    _demo_render_result(proba, tgt_labels, champion_name)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        else:
            st.markdown(
                f"<p style='color:{FONT};opacity:0.55;font-style:italic'>"
                f"Click one of the buttons above to load a sample from the "
                f"validation set.</p>",
                unsafe_allow_html=True,
            )

    # ── Global SHAP reference (collapsible) ───────────────────────────────
    if shap_csv is not None:
        with st.expander("\U0001f4ca Global SHAP feature importance (reference)",
                         expanded=False):
            try:
                st.dataframe(pd.read_csv(shap_csv).head(20),
                             width='stretch', hide_index=True)
            except Exception:
                pass


PAGES: dict = {
    "▶️  Run Pipeline":          page_run_pipeline,
    "🔬 Data Studio":            page_data_profiling,
    "🔧 Data Preparation":       page_feature_engineering,
    "📈 Post-Processing EDA":    page_post_processing_eda,
    "🤖 Model Development":      page_model_development,
    "📊 Model Evaluation":       page_model_performance,
    "🎯 Prediction Demo":        page_prediction_demo,
    "🔍 Ethics & Explainability": page_explainability,
}

_NAV_SECTIONS: list[str] = list(PAGES.keys())

# ── Nav strip CSS ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    div[data-testid="stRadio"] div[role="radiogroup"] {{
        display: flex; flex-wrap: wrap; gap: 6px; padding: 6px 0;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label {{
        background: #1E2A4A;
        border: 1.5px solid {ACCENT};
        border-radius: 20px;
        padding: 5px 14px;
        cursor: pointer;
        color: {FONT} !important;
        font-size: 0.82rem;
        transition: background 0.15s;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label p {{
        color: {FONT} !important;
        margin: 0;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
        background: {ACCENT} !important;
        border-color: {ACCENT} !important;
        color: #FFFFFF !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {{
        color: #FFFFFF !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label input {{
        display: none;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "▶️  Run Pipeline"

# ── Sidebar (returns selected uc_key) ─────────────────────────────────────────
uc_key = render_sidebar()

# ── Top nav strip ─────────────────────────────────────────────────────────────
_nav_idx = (
    _NAV_SECTIONS.index(st.session_state.nav_page)
    if st.session_state.nav_page in _NAV_SECTIONS
    else 0
)
selected_page = st.radio(
    "Navigation",
    _NAV_SECTIONS,
    index=_nav_idx,
    horizontal=True,
    key="_top_nav",
    label_visibility="collapsed",
)
st.session_state.nav_page = selected_page

# ── Dispatch ───────────────────────────────────────────────────────────────────
PAGES[selected_page](uc_key)
