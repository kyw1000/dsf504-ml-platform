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
        "champion":   "lgbm_optuna_champion.pkl",
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
        "status":     "scaffolded",
        "champion":   "champion.pkl",
    },
    "F": {
        "title":      "ESG & Greenwashing Risk",
        "icon":       "🌱",
        "tag":        "SEC EDGAR ESG",
        "target":     "esg_label",
        "task":       "Multi-class Classification",
        "metric":     "F1 (macro)",
        "model_dir":  "use_case_F",
        "data_dir":   "sec_esg",
        "report_dir": "use_case_F",
        "status":     "scaffolded",
        "champion":   "champion.pkl",
    },
    "G": {
        "title":      "Robo-Advisory & Portfolio AI",
        "icon":       "🤖",
        "tag":        "AmEx Default Prediction",
        "target":     "target",
        "task":       "Binary Classification",
        "metric":     "AmEx Metric",
        "model_dir":  "use_case_G",
        "data_dir":   "amex_default",
        "report_dir": "use_case_G",
        "status":     "scaffolded",
        "champion":   "champion.pkl",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# ── Per-use-case config dicts ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

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
}

_FE_EDA_SRC: dict = {
    "A": {
        "train_fe":      "data/ieee_fraud/train_fe.parquet",
        "raw":           "data/ieee_fraud/train_transaction.parquet",
        "feat_list":     "reports/use_case_A/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_A/engineered_feature_summary.png",
        "target":        "isFraud",
        "target_labels": {0: "Legitimate", 1: "Fraud"},
    },
    "B": {
        "train_fe":      "data/gmsc_credit/train_fe.parquet",
        "raw":           "data/gmsc_credit/cs-training.parquet",
        "feat_list":     None,
        "fe_summary":    "reports/use_case_B/engineered_features.png",
        "target":        "SeriousDlqin2yrs",
        "target_labels": {0: "No Default", 1: "Default"},
    },
    "C_nlp": {
        "train_fe":      None,
        "raw":           "data/financial_phrasebank/sent_train.csv",
        "feat_list":     None,
        "fe_summary":    None,
        "target":        "label",
        "target_labels": {0: "Bearish", 1: "Neutral", 2: "Bullish"},
    },
    "C_markets": {
        "train_fe":      "data/optiver_volatility/test_fe.parquet",
        "raw":           "data/optiver_volatility/book_train.parquet",
        "feat_list":     "reports/use_case_C_markets/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_C_markets/engineered_feature_summary.png",
        "target":        "target",
        "target_labels": {},
    },
    "E": {
        "train_fe":      "data/porto_seguro/train_fe.parquet",
        "raw":           "data/porto_seguro/train.parquet",
        "feat_list":     "reports/use_case_E/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_E/engineered_feature_summary.png",
        "target":        "target",
        "target_labels": {0: "No Claim", 1: "Claim"},
    },
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
        "description": "IEEE-CIS fraud features focus on temporal patterns, cardinality-based frequency encoding, and transaction amount anomalies relative to historical card behaviour.",
    },
    "B": {
        "stages": [
            ("Delinquency Aggregation", ["fe_total_dpd", "fe_dpd_severity", "fe_any_delinquency"]),
            ("Utilisation Ratios", ["fe_util_sq", "fe_high_util"]),
            ("Debt & Income Ratios", ["fe_debt_income_ratio", "fe_monthly_income_log"]),
            ("Loan Counts", ["fe_total_loans", "fe_loan_density"]),
        ],
        "description": "Credit features aggregate delinquency history, compute non-linear utilisation ratios, and derive debt-to-income proxies to capture repayment capacity.",
    },
    "C_nlp": {
        "stages": [
            ("TF-IDF Unigrams", ["tfidf_dim_1…N"]),
            ("Sentiment Lexicon Scores", ["finbert_positive", "finbert_negative", "finbert_neutral"]),
            ("Text Statistics", ["char_count", "word_count", "avg_word_len"]),
            ("Special Token Flags", ["has_ticker", "has_hashtag", "has_number"]),
        ],
        "description": "NLP features combine TF-IDF bag-of-words with FinBERT sentiment scores and hand-crafted linguistic features for financial text classification.",
    },
    "C_markets": {
        "stages": [
            ("Book Features", ["fe_book_rv", "fe_log_book_rv", "fe_wap_mean"]),
            ("Trade Features", ["fe_lr_std", "fe_spread", "fe_log_spread"]),
            ("Cross-Book Aggregates", ["fe_stock_mean_rv", "fe_rv_vs_stock_mean"]),
            ("Ratio Features", ["fe_volatility_spread_ratio", "fe_rv_l2_ratio"]),
        ],
        "description": "Optiver volatility features are derived from limit order book (WAP, bid-ask spread) and trade data (log-returns std dev) to predict realized volatility over 10-minute windows.",
    },
    "E": {
        "stages": [
            ("Missing Value Flags", ["ps_car_03_cat_missing", "ps_car_05_cat_missing"]),
            ("Calc Feature Aggregates", ["fe_calc_mean", "fe_calc_sum", "fe_calc_std"]),
            ("Feature Group Counts", ["fe_num_ind_features", "fe_num_reg_features"]),
            ("Ratio & Interaction Terms", ["fe_calc_cv", "fe_ind_mean_ratio"]),
        ],
        "description": "Porto Seguro features aggregate calc, ind, reg, and car feature groups, add missing-flag indicators, and engineer interaction ratios to capture claim-risk patterns.",
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
        if not (r_dir / "shap_feature_importance.csv").exists():
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
            "Select an active use case (A, B, C_nlp, C_markets, or E) to run the pipeline.",
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
        run_clicked = st.button("▶ Run Selected", type="primary", key=f"_run_btn_{uc_key}", use_container_width=True)

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

    tab_guide, tab_feats, tab_summary, tab_run = st.tabs([
        "📖 FE Guidance",
        "📋 Feature List",
        "📊 FE Summary",
        "▶️ Run Step 3",
    ])

    # ── FE Guidance ────────────────────────────────────────────────────────────
    with tab_guide:
        desc = guidance.get("description", "")
        if desc:
            st.markdown(f"> {desc}")

        stages = guidance.get("stages", [])
        if stages:
            st.markdown("#### Feature Engineering Pipeline Stages")
            for stage_name, feats in stages:
                st.markdown(
                    f"<div style='margin-bottom:10px;'>"
                    f"<b style='color:{BLUE};'>{stage_name}</b><br>"
                    + "".join(
                        f"<span class='pill'>{f}</span>" for f in feats
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("FE guidance not configured for this use case.")

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

        if feat_list_path:
            df_fl = load_csv(feat_list_path)
            if df_fl is not None:
                st.markdown(f"**{len(df_fl):,} engineered features**")
                search = st.text_input("Search features", key=f"_fe_search_{uc_key}", placeholder="Type to filter…")
                if search:
                    mask = df_fl.apply(lambda col: col.astype(str).str.contains(search, case=False)).any(axis=1)
                    df_fl = df_fl[mask]
                st.dataframe(df_fl, width='stretch', hide_index=True)
            else:
                st.info("Engineered features list not found. Run Step 3 — Data Preparation.")

        elif train_fe_path:
            df_fe = load_parquet(train_fe_path, nrows=5)
            if df_fe is not None:
                fe_cols = [c for c in df_fe.columns if c.startswith("fe_")]
                st.markdown(f"**{len(fe_cols):,} `fe_` prefixed columns found in train_fe.parquet**")
                if fe_cols:
                    df_show = pd.DataFrame({"feature": fe_cols, "dtype": [str(df_fe[c].dtype) for c in fe_cols]})
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

    # ── Run Step 3 ─────────────────────────────────────────────────────────────
    with tab_run:
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

    tab_overview, tab_compare, tab_new_feats, tab_target = st.tabs([
        "\U0001f4ca Overview",
        "\U0001f522 Raw vs Processed",
        "\U0001f195 New Features",
        "\U0001f3af Target Split",
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
                st.info("No numeric columns shared between raw and processed data.")
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

                n_bins  = st.slider("Histogram bins", 20, 100, 40, key=f"_ppe_bins_{uc_key}")
                max_col = min(20, len(shared_num))
                picked  = st.multiselect(
                    f"Select columns to compare (showing up to {max_col})",
                    shared_num[:max_col],
                    default=shared_num[:min(3, max_col)],
                    key=f"_ppe_cols_{uc_key}",
                )

                _RAW_CLR = "#78909C"
                _FE_CLR  = "#42A5F5"

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
            vc["Label"] = vc["Class"].map(lambda x: target_labels.get(int(x), str(x)))
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
            png_path = r_dir / "model_roc_pr_curves.png"
            if png_path.exists():
                st.image(str(png_path), width='stretch')
            else:
                st.info("ROC/PR curves not found. Run Step 4.")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    with tab_confusion:
        if is_regr:
            st.info("Confusion matrix is not applicable for regression tasks.")
        else:
            for png_name in ["confusion_matrices.png"]:
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
        shap_csv = r_dir / "shap_feature_importance.csv"
        if not shap_csv.exists():
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
        for png_name in ["shap_bar_importance.png", "shap_beeswarm.png"]:
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
        shap_csv = r_dir / "shap_feature_importance.csv"
        if not shap_csv.exists():
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
        f"Adjust feature values below and click **Predict** to get a real-time "
        f"prediction from the champion model (`{champion_name}`)."
    )

    # Load val data for default values and ranges
    data_dir = ROOT / "data" / uc.get("data_dir", "")
    val_path = data_dir / "val_fe.parquet"
    df_ref   = None
    if val_path.exists():
        df_ref = pd.read_parquet(val_path)

    feat_cols_use = feat_cols[:20]  # Show top 20 for UI simplicity
    input_vals = {}

    with st.form(key=f"_regr_form_{uc_key}"):
        n_per_row = 4
        rows = [feat_cols_use[i:i+n_per_row] for i in range(0, len(feat_cols_use), n_per_row)]
        for row_feats in rows:
            cols = st.columns(len(row_feats))
            for col, feat in zip(cols, row_feats):
                default_val = 0.0
                if df_ref is not None and feat in df_ref.columns:
                    default_val = float(df_ref[feat].median())
                input_vals[feat] = col.number_input(
                    feat, value=default_val,
                    format="%.6f",
                    key=f"_ri_{uc_key}_{feat}",
                )
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        # Fill remaining features with median
        all_input = {}
        if df_ref is not None:
            for feat in feat_cols:
                if feat in df_ref.columns:
                    all_input[feat] = float(df_ref[feat].median())
        for feat in feat_cols_use:
            all_input[feat] = input_vals.get(feat, 0.0)

        X_input = pd.DataFrame([{f: all_input.get(f, 0.0) for f in feat_cols}])
        try:
            prediction = model.predict(X_input)[0]
            st.success(f"**Predicted {uc.get('target', 'value')}: {prediction:.6f}**")

            # Gauge-style display
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(prediction),
                title={"text": f"Predicted {uc.get('metric', 'Value')}"},
                gauge={
                    "axis": {"range": [0, max(0.05, float(prediction) * 3)]},
                    "bar":  {"color": BLUE},
                    "bgcolor": BG,
                    "steps": [
                        {"range": [0, float(prediction) * 0.5], "color": "#1B5E20"},
                        {"range": [float(prediction) * 0.5, float(prediction) * 1.5], "color": "#E65100"},
                    ],
                },
            ))
            fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT, height=320)
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Prediction Demo ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page_prediction_demo(uc_key: str) -> None:
    uc = USE_CASE_META.get(uc_key, {})
    section_header(
        f"🎯 Prediction Demo — {uc['icon']} {uc['title']}",
        "Live inference using the tuned champion model.",
    )

    # Route regression to its own helper
    if "Regression" in uc.get("task", ""):
        _page_prediction_demo_regression(uc_key)
        return

    # ── Classification demo ────────────────────────────────────────────────────
    m_dir = ROOT / "models" / uc.get("model_dir", "")
    champion_name = uc.get("champion", "lgbm_optuna_champion.pkl")
    model = load_model(m_dir / champion_name)
    feat_pkl = m_dir / "feature_cols.pkl"
    feat_cols = joblib.load(feat_pkl) if feat_pkl.exists() else None

    if model is None:
        st.info("Champion model not found. Run Step 5.")
        _run_step_action(5, uc_key, "▶ Run Step 5  (goes to Run Pipeline)", suffix="cdemo")
        return

    if feat_cols is None:
        st.info("Feature columns not found. Run Steps 4–5.")
        return

    is_nlp = uc.get("is_nlp", False)
    if is_nlp:
        _page_prediction_demo_nlp(uc_key, model, feat_cols)
        return

    st.markdown("#### Live Classification Prediction Demo")
    st.markdown(
        f"Adjust feature values below and click **Predict** to see the champion model's "
        f"probability output. Model: `{champion_name}`"
    )

    # Load val data for defaults and ranges
    data_dir = ROOT / "data" / uc.get("data_dir", "")
    val_path = data_dir / "val_fe.parquet"
    df_ref   = None
    if val_path.exists():
        df_ref = pd.read_parquet(val_path)

    # Load optimal threshold
    thresh = 0.5
    thresh_file = m_dir / "lgbm_optimal_threshold.txt"
    if thresh_file.exists():
        try:
            thresh = float(thresh_file.read_text().strip())
        except Exception:
            pass

    # Feature importance for ordering sliders
    shap_csv = ROOT / "reports" / uc.get("report_dir", "") / "shap_feature_importance.csv"
    if shap_csv.exists():
        df_shap = pd.read_csv(shap_csv)
        ordered_feats = df_shap.iloc[:, 0].tolist()
        feat_cols_ui = [f for f in ordered_feats if f in feat_cols][:15]
    else:
        feat_cols_ui = feat_cols[:15]

    input_vals = {}
    with st.form(key=f"_clf_form_{uc_key}"):
        n_per_row = 3
        feat_rows = [feat_cols_ui[i:i+n_per_row] for i in range(0, len(feat_cols_ui), n_per_row)]
        for feat_row in feat_rows:
            cols = st.columns(len(feat_row))
            for col, feat in zip(cols, feat_row):
                default_val = 0.0
                min_val, max_val = 0.0, 1.0
                if df_ref is not None and feat in df_ref.columns:
                    default_val = float(df_ref[feat].median())
                    min_val     = float(df_ref[feat].quantile(0.01))
                    max_val     = float(df_ref[feat].quantile(0.99))
                    if min_val == max_val:
                        min_val = default_val - 1.0
                        max_val = default_val + 1.0
                input_vals[feat] = col.slider(
                    feat,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    key=f"_ci_{uc_key}_{feat}",
                )
        threshold_override = st.slider(
            "Decision threshold", 0.01, 0.99, float(thresh), 0.01,
            key=f"_thresh_demo_{uc_key}",
        )
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        all_input = {}
        if df_ref is not None:
            for feat in feat_cols:
                if feat in df_ref.columns:
                    all_input[feat] = float(df_ref[feat].median())
        for feat in feat_cols_ui:
            all_input[feat] = input_vals.get(feat, 0.0)

        X_input = pd.DataFrame([{f: all_input.get(f, 0.0) for f in feat_cols}])
        try:
            proba = model.predict_proba(X_input)[0]
            target_names = uc.get("target_labels", {})

            if len(proba) == 2:
                pos_prob = proba[1]
                decision = "POSITIVE" if pos_prob >= threshold_override else "NEGATIVE"
                colour   = RED if decision == "POSITIVE" else GRN

                col1, col2 = st.columns(2)
                col1.markdown(
                    metric_card("Positive Class Probability", f"{pos_prob:.4f}", colour=colour),
                    unsafe_allow_html=True,
                )
                col2.markdown(
                    metric_card("Decision", decision, colour=colour),
                    unsafe_allow_html=True,
                )

                fig = go.Figure(go.Bar(
                    x=[target_names.get(0, "Negative"), target_names.get(1, "Positive")],
                    y=[proba[0], proba[1]],
                    marker_color=[GRN, RED],
                    text=[f"{proba[0]:.4f}", f"{proba[1]:.4f}"],
                    textposition="outside",
                ))
                fig.update_layout(
                    plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT,
                    height=300, yaxis_range=[0, 1.1],
                    title="Predicted Class Probabilities",
                    yaxis_title="Probability",
                )
                st.plotly_chart(fig, width='stretch')

            else:
                # Multi-class
                class_names = [target_names.get(i, f"Class {i}") for i in range(len(proba))]
                fig = go.Figure(go.Bar(
                    x=class_names, y=list(proba),
                    marker_color=[BLUE, GRN, ORG, RED, PURP][:len(proba)],
                    text=[f"{p:.4f}" for p in proba],
                    textposition="outside",
                ))
                fig.update_layout(
                    plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT,
                    height=320, yaxis_range=[0, 1.1],
                    title="Predicted Class Probabilities",
                    yaxis_title="Probability",
                )
                st.plotly_chart(fig, width='stretch')

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ======================================================================
# PAGE: Prediction Demo (NLP)
# ======================================================================

def _page_prediction_demo_nlp(uc_key: str, model, feat_cols: list) -> None:
    uc = USE_CASE_META.get(uc_key, {})
    target_labels = uc.get("target_labels", {})

    st.markdown("#### Live NLP Sentiment Demo")
    st.markdown(
        "Enter a financial news headline or excerpt below. "
        "The champion model will classify it as **Bearish / Neutral / Bullish**."
    )

    m_dir    = ROOT / "models" / uc.get("model_dir", "")
    vec_path = m_dir / "tfidf_vectorizer.pkl"
    vectorizer = None
    if vec_path.exists():
        try:
            vectorizer = joblib.load(vec_path)
        except Exception:
            pass

    text_input = st.text_area(
        "Financial text",
        value="The company reported record profits, beating analyst expectations.",
        height=100,
        key=f"_nlp_text_{uc_key}",
    )

    if st.button("Classify", type="primary", key=f"_nlp_btn_{uc_key}"):
        try:
            if vectorizer is not None:
                X_vec = vectorizer.transform([text_input])
                if hasattr(X_vec, "toarray"):
                    X_input = pd.DataFrame(
                        X_vec.toarray(),
                        columns=vectorizer.get_feature_names_out(),
                    )
                    for c in feat_cols:
                        if c not in X_input.columns:
                            X_input[c] = 0.0
                    X_input = X_input[feat_cols]
                else:
                    X_input = X_vec
            else:
                X_input = pd.DataFrame([[0.0] * len(feat_cols)], columns=feat_cols)

            proba       = model.predict_proba(X_input)[0]
            class_names = [target_labels.get(i, f"Class {i}") for i in range(len(proba))]
            pred_idx    = int(np.argmax(proba))
            pred_label  = class_names[pred_idx]
            colour_map  = {0: RED, 1: PALETTE["grey"], 2: GRN}
            colour      = colour_map.get(pred_idx, BLUE)

            st.markdown(
                metric_card("Prediction", pred_label, colour=colour),
                unsafe_allow_html=True,
            )

            fig = go.Figure(go.Bar(
                x=class_names, y=list(proba),
                marker_color=[RED, PALETTE["grey"], GRN][:len(proba)],
                text=[f"{p:.4f}" for p in proba],
                textposition="outside",
            ))
            fig.update_layout(
                plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT,
                height=300, yaxis_range=[0, 1.1],
                title="Sentiment Class Probabilities",
                yaxis_title="Probability",
            )
            st.plotly_chart(fig, width='stretch')

        except Exception as e:
            st.error(f"Classification failed: {e}")
            st.info("Ensure Steps 4-5 have been run successfully so the model and vectorizer are saved.")


# ======================================================================
# PAGES registry
# ======================================================================

PAGES: dict = {
    "\u25b6\ufe0f  Run Pipeline":           page_run_pipeline,
    "\U0001f52c Data Studio":             page_data_profiling,
    "\U0001f527 Data Preparation":        page_feature_engineering,
    "\U0001f4c8 Post-Processing EDA":     page_post_processing_eda,
    "\U0001f916 Model Development":       page_model_development,
    "\U0001f4ca Model Evaluation":        page_model_performance,
    "\U0001f50d Ethics & Explainability": page_explainability,
    "\U0001f3af Prediction Demo":         page_prediction_demo,
}

# ── Main execution ─────────────────────────────────────────────────────────────
uc_key = render_sidebar()

# ── Top navigation bar (radio-based tab strip) ────────────────────────────────
st.markdown(
    """
    <style>
    div[data-testid="stRadio"] > label { display:none; }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display:flex; flex-wrap:wrap; gap:2px;
        border-bottom:2px solid #3949AB;
        margin-bottom:18px; padding-bottom:0;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background:#FFFFFF;
        border:1px solid #C5CAE9;
        border-bottom:2px solid transparent;
        border-radius:6px 6px 0 0;
        padding:7px 14px;
        margin-bottom:-2px;
        font-size:0.80rem;
        white-space:nowrap;
        color:#1A237E !important;
        cursor:pointer;
        font-weight:500;
        transition:background 0.12s, color 0.12s;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label p {
        color:#1A237E !important;
        font-size:0.80rem;
        margin:0;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background:#E8EAF6;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover p {
        color:#1A237E !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background:#3949AB !important;
        border-color:#3949AB !important;
        border-bottom-color:#3949AB !important;
        color:#FFFFFF !important;
        font-weight:700;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
        color:#FFFFFF !important;
        font-weight:700;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display:none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_NAV_LABELS = list(PAGES.keys())

if "nav_page" not in st.session_state or st.session_state.nav_page not in _NAV_LABELS:
    st.session_state.nav_page = _NAV_LABELS[0]
_selected_nav = st.radio(
    "Navigation",
    _NAV_LABELS,
    index=_NAV_LABELS.index(st.session_state.nav_page),
    key="_top_nav_radio",
    horizontal=True,
    label_visibility="collapsed",
)
if _selected_nav != st.session_state.nav_page:
    st.session_state.nav_page = _selected_nav
    st.rerun()

_active_page = st.session_state.nav_page
_warning_msg = _prereq_warning(_active_page, uc_key)
if _warning_msg:
    st.warning(_warning_msg, icon="\u26a0\ufe0f")
    st.info(
        "Once the required step is complete, come back to this page \u2014 it will load automatically.",
        icon="\U0001f4a1",
    )
    _run_step_hint = {
        "\U0001f52c Data Studio":             1,
        "\U0001f527 Data Preparation":        1,
        "\U0001f4c8 Post-Processing EDA":     3,
        "\U0001f916 Model Development":       4,
        "\U0001f4ca Model Evaluation":        4,
        "\U0001f3af Prediction Demo":         5,
        "\U0001f50d Ethics & Explainability": 6,
    }
    _hint_step = _run_step_hint.get(_active_page)
    if _hint_step is not None:
        if st.button(
            f"\u25b6\ufe0f Go to Run Pipeline \u2192 Step {_hint_step}",
            key="_prereq_goto_run",
            type="primary",
        ):
            st.session_state["_auto_run_step"] = _hint_step
            st.session_state.nav_page = "\u25b6\ufe0f  Run Pipeline"
            st.rerun()
else:
    PAGES[_active_page](uc_key)
