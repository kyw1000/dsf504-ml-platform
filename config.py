"""
config.py
=========
DSF504 Big Data/AI-Powered Financial Analytics Platform
Platform-wide configuration: paths, dataset registry, and shared constants.

All use-case modules import from here so paths and settings are changed in
one place only.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (this file lives at the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
REPORTS_DIR  = PROJECT_ROOT / "reports"
MODELS_DIR   = PROJECT_ROOT / "models"

# Create directories if they don't exist
for _d in [DATA_DIR, REPORTS_DIR, MODELS_DIR]:
    _d.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Random seed — used everywhere for reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Dataset registry
# Each entry maps a use-case key to its Kaggle competition/dataset slug
# and the local subdirectory name under DATA_DIR.
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "A": {
        "name":        "Financial Crime & Fraud Analytics",
        "use_cases":   ["Credit Card Fraud Detection", "Fraud/Anomaly Detection", "AML Detection"],
        "kaggle_slug": "ieee-fraud-detection",          # competition slug
        "data_subdir": "ieee_fraud",
        "target_col":  "isFraud",
        "task_type":   "binary_classification",
        "files": {
            "train_transaction": "train_transaction.csv",
            "train_identity":    "train_identity.csv",
            "test_transaction":  "test_transaction.csv",
            "test_identity":     "test_identity.csv",
        },
        "merge_key":   "TransactionID",
        "positive_rate": 0.035,                        # ~3.5% fraud rate
        "eval_metrics": ["roc_auc", "pr_auc", "f1", "precision", "recall"],
        "primary_metric": "roc_auc",
    },
    "B": {
        "name":        "Credit Risk & Default Intelligence",
        "use_cases":   ["Loan Default Prediction", "Early Warning Signals"],
        "kaggle_slug": "GiveMeSomeCredit",
        "data_subdir": "gmsc_credit",
        "target_col":  "SeriousDlqin2yrs",
        "task_type":   "binary_classification",
        "files": {
            "train": "cs-training.csv",
            "test":  "cs-test.csv",
        },
        "merge_key":   None,
        "positive_rate": 0.067,
        "eval_metrics": ["roc_auc", "pr_auc", "f1"],
        "primary_metric": "roc_auc",
    },
    "C_markets": {
        "name":        "Investment & Market Intelligence — Volatility",
        "use_cases":   ["Stock Return/Volatility Prediction"],
        "kaggle_slug": "optiver-realized-volatility-prediction",
        "data_subdir": "optiver_volatility",
        "target_col":  "target",
        "task_type":   "regression",
        "eval_metrics": ["rmse", "mae", "r2"],
        "primary_metric": "rmse",
    },
    "C_nlp": {
        "name":        "Investment & Market Intelligence — Sentiment",
        "use_cases":   ["Financial News Sentiment Analysis"],
        "kaggle_slug": "zeroshot/twitter-financial-news-sentiment",
        "data_subdir": "twitter_fin_sentiment",
        "target_col":  "label",
        "task_type":   "multiclass_classification",
        "eval_metrics": ["accuracy", "f1_macro", "roc_auc_ovr"],
        "primary_metric": "f1_macro",
    },
    "D": {
        "name":        "Customer Intelligence & Retention",
        "use_cases":   ["Customer Churn Prediction"],
        "kaggle_slug": "kkbox-churn-prediction-challenge",
        "data_subdir": "kkbox_churn",
        "target_col":  "is_churn",
        "task_type":   "binary_classification",
        "positive_rate": 0.07,
        "eval_metrics": ["roc_auc", "pr_auc", "f1"],
        "primary_metric": "roc_auc",
    },
    "E": {
        "name":        "Insurance Risk & Claims Analytics",
        "use_cases":   ["Insurance Claim Risk Prediction"],
        "kaggle_slug": "porto-seguro-safe-driver-prediction",
        "data_subdir": "porto_seguro",
        "target_col":  "target",
        "task_type":   "binary_classification",
        "positive_rate": 0.036,
        "eval_metrics": ["normalized_gini", "roc_auc", "pr_auc"],
        "primary_metric": "roc_auc",
        "files": {
            "train": "train.csv",
            "test":  "test.csv",
        },
        "merge_key": None,
    },
    "F": {
        "name":        "ESG, Compliance & Reputation Risk",
        "use_cases":   ["ESG / Greenwashing Risk Scoring"],
        "kaggle_slug": "sec-edgar-10k",                # SEC EDGAR / custom
        "data_subdir": "sec_esg",
        "target_col":  "esg_label",
        "task_type":   "multiclass_classification",
        "eval_metrics": ["accuracy", "f1_macro"],
        "primary_metric": "f1_macro",
    },
    "B3": {
        "name":          "Credit Risk & Default Intelligence",
        "use_cases":     ["AmEx Loan Default Prediction"],
        "kaggle_slug":   "amex-default-prediction",
        "data_subdir":   "amex_default",
        "target_col":    "target",
        "task_type":     "binary_classification",
        "positive_rate": 0.259,
        "eval_metrics":  ["amex_metric", "roc_auc", "f1"],
        "primary_metric": "amex_metric",
    },
    "G1": {
        "name":          "Robo-Advisory Portfolio Recommendation",
        "use_cases":     ["Personalised Asset Recommendation", "Investor–Asset Ranking"],
        "kaggle_slug":   "far-trans-financial-asset-recommendation",
        "data_subdir":   "far_trans",
        "target_col":    "label",
        "task_type":     "ranking",
        "eval_metrics":  ["ndcg_at_10", "precision_at_10", "recall_at_10", "mrr"],
        "primary_metric": "ndcg_at_10",
    },
    "G2": {
        "name":          "Explainable AI for Analysts & Managers",
        "use_cases":     ["Stock Outperformance Prediction", "Analyst Screening"],
        "kaggle_slug":   "chad116/sec-company-facts-all-10q-10k-financial-data",
        "data_subdir":   "sec_edgar",
        "target_col":    "outperform",
        "task_type":     "binary_classification",
        "positive_rate": 0.40,
        "eval_metrics":  ["auc_roc", "auc_pr", "f1", "precision", "recall"],
        "primary_metric": "auc_roc",
    },
}

# ---------------------------------------------------------------------------
# Imbalance handling thresholds
# ---------------------------------------------------------------------------
IMBALANCE_THRESHOLD = 0.10   # If positive rate < 10%, apply imbalance strategy
SMOTE_RATIO         = 0.10   # Target minority ratio after SMOTE

# ---------------------------------------------------------------------------
# Cross-validation defaults
# ---------------------------------------------------------------------------
CV_FOLDS            = 5
STRATIFIED_CV       = True   # Use StratifiedKFold for classification tasks

# ---------------------------------------------------------------------------
# Hyperparameter tuning defaults
# ---------------------------------------------------------------------------
TUNING_TRIALS       = 50     # Optuna trials
TUNING_TIMEOUT      = 600    # seconds
TUNING_CV_FOLDS     = 3      # Folds used during tuning (faster than full CV)

# ---------------------------------------------------------------------------
# Feature engineering constants
# ---------------------------------------------------------------------------
# Fraud: time reference (TransactionDT is seconds from this date)
FRAUD_START_DATE    = "2017-12-01"

# High-cardinality threshold: encode if unique values < threshold, else target-encode
CARDINALITY_THRESHOLD = 50

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
DASHBOARD_HOST      = "localhost"
DASHBOARD_PORT      = 8501     # Streamlit default
DASHBOARD_TITLE     = "DSF504 Financial AI Analytics Platform"
