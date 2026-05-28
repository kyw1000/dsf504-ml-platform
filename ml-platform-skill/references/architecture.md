# DSF504 Architecture Reference

## File Tree

```
C:\DSF504\
├── config.py                    <- DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE, CV_FOLDS
├── run_platform.py              <- CLI runner (alternative to dashboard)
├── generate_report.py           <- Self-contained HTML report (UC-A PNGs embedded as base64)
├── requirements.txt
├── claude.md                    <- 12-rule working rules for Claude sessions
├── dashboard/
│   ├── app.py                   <- ALL dashboard code (~4,241 lines)
│   ├── viz_library.py           <- Plotly chart helpers for C_markets market analytics tab
│   └── assets/
│       └── ml_framework.png     <- sidebar banner (place here to activate)
├── utils/
│   ├── data_loader.py           <- KaggleLoader, DataProfiler, smart_split, reduce_mem_usage
│   ├── eda_viz.py               <- Shared matplotlib EDA/FE visualization helpers (PNG + insight)
│   ├── encoding_guard.py        <- ensure_utf8() -- call before logging.basicConfig()
│   ├── ethics_viz.py            <- SHAP/fairness plot helpers (used by UC-E Step 6)
│   ├── file_guard.py            <- Check file integrity + strip null bytes
│   └── __init__.py
├── use_case_A_fraud/            <- Steps 01-06 + 05b_lgbm_champion.py (status: complete)
├── use_case_B_credit/           <- Steps 01-06 (status: complete)
├── use_case_C_nlp/              <- Steps 01-06 (status: complete)
├── use_case_C_market/           <- Steps 01-06 + _run_step4_fast.py + _run_step5_fast.py
│                                   NOTE: folder has no trailing 's'
│                                   model_dir/report_dir = use_case_C_markets (WITH 's')
├── use_case_D_churn/            <- Steps 01-06 (status: complete)
├── use_case_E_insurance/        <- Steps 01-06 (status: complete)
├── use_case_F_esg/              <- Steps 01-06 (status: complete)
├── use_case_G_advisory/         <- Steps 01-06 (status: complete)
├── use_case_G1_robo/            <- Steps 01-06 (status: active -- Robo-Advisory / LambdaRank)
├── use_case_G2_xai/             <- Steps 01-06 (status: active -- XAI for Analysts)
├── data/
│   ├── ieee_fraud/              <- UC-A  (train_transaction.parquet)
│   ├── gmsc_credit/             <- UC-B  (cs-training.parquet)
│   ├── financial_phrasebank/    <- UC-C_nlp (sent_train.csv)
│   ├── optiver_volatility/      <- UC-C_markets (book_train.parquet)
│   ├── kkbox_churn/             <- UC-D  (train_raw.parquet, train_fe.parquet)
│   ├── porto_seguro/            <- UC-E  (train.parquet)
│   ├── sec_esg/ + greenwashing/ <- UC-F
│   ├── amex_default/            <- UC-G  (train_raw.parquet -- NOT train_data_synthetic)
│   ├── amex-advisory/           <- UC-G advisory variant
│   ├── far-trans/               <- UC-G1 (company_ratios.parquet, train_fe.parquet)
│   └── sec_edgar/               <- UC-G2
├── models/
│   ├── use_case_A/              <- lgbm_optuna_champion.pkl, final_model.pkl
│   ├── use_case_B/
│   ├── use_case_C_nlp/
│   ├── use_case_C_markets/      <- WITH trailing 's' (unlike script folder)
│   ├── use_case_D/
│   ├── use_case_E/
│   ├── use_case_F/
│   ├── use_case_G/
│   ├── use_case_G1/
│   └── use_case_G2/
└── reports/
    ├── use_case_A/ ... use_case_G/
    ├── use_case_C_markets/      <- WITH trailing 's'
    ├── use_case_G1/
    └── use_case_G2/
```

## USE_CASE_META Schema

```python
USE_CASE_META = {
    "A": {
        "title":      str,   # human-readable name
        "icon":       str,   # emoji
        "tag":        str,   # dataset name / Kaggle tag
        "target":     str,   # target column name
        "task":       str,   # "Binary Classification" | "Regression" | "Multi-class Classification" | "Learning to Rank"
        "metric":     str,   # primary metric, e.g. "PR-AUC", "NDCG@10"
        "model_dir":  str,   # relative path under models/
        "data_dir":   str,   # relative path under data/
        "report_dir": str,   # relative path under reports/
        "status":     str,   # "complete" | "active" | "scaffolded"
        "champion":   str,   # model filename e.g. "lgbm_optuna_champion.pkl" or "champion.pkl"
        # optional:
        "is_nlp":     bool,  # True for C_nlp -- enables text-feature tab
    },
}
```

Status meanings:
- "complete" -- all 6 steps verified end-to-end: A, B, C_nlp, C_markets, D, E, F, G
- "active"   -- scripts written, artefacts not fully validated: G1, G2
- "scaffolded" -- placeholder only (none currently in codebase)

## Pipeline Script Boilerplate

Every `01-06_*.py` must start with this pattern (order matters):

```python
"""use_case_X_name/0N_script_name.py"""
from __future__ import annotations
import sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (DATA_DIR, REPORTS_DIR, RANDOM_STATE, ...)
from utils.encoding_guard import ensure_utf8
ensure_utf8()   # MUST be before logging.basicConfig()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
```

`ensure_utf8()` before `logging.basicConfig()` is mandatory -- the StreamHandler
captures stderr at that point; cp1252 on Windows garbles all emoji/unicode.

## Subprocess Calls in app.py

Both `subprocess.Popen()` calls in `_run_step()` must have:

```python
proc = subprocess.Popen(
    [sys.executable, str(script_path)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    cwd=str(ROOT),
    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
)
```

## PAGES Dict + _NAV_SECTIONS

`_NAV_SECTIONS` is derived directly from PAGES: `_NAV_SECTIONS = list(PAGES.keys())`.
Adding a page only requires updating PAGES -- no separate list to maintain.
Labels must be byte-for-byte identical; verify with `.encode('utf-8').hex()`.

## _PROFILING_SRC Schema (Data Studio)

```python
_PROFILING_SRC = {
    "A": {
        "col_summary":  "reports/use_case_A/train_column_summary.csv",
        "raw":          "data/ieee_fraud/train_transaction.parquet",
        "target":       "isFraud",
        "corr_csv":     "reports/use_case_A/feature_target_correlation.csv",
        "corr_png":     "reports/use_case_A/correlation_top30_V_cols.png",
        "missing_png":  "reports/use_case_A/missing_heatmap.png",
        "outlier_csv":  "reports/use_case_A/outlier_report.csv",
        "target_png":   "reports/use_case_A/target_distribution.png",
    }, ...
}
```

## _FE_EDA_SRC Schema (Post-Processing EDA)

```python
_FE_EDA_SRC = {
    "A": {
        "train_fe":      "data/ieee_fraud/train_fe.parquet",
        "raw":           "data/ieee_fraud/train_transaction.parquet",
        "feat_list":     "reports/use_case_A/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_A/engineered_feature_summary.png",
        "raw_vs_proc":   "reports/use_case_A/raw_vs_processed_distributions.png",
        "target":        "isFraud",
        "target_labels": {0: "Legitimate", 1: "Fraud"},
    }, ...
}
```

## Correlation Matrix Data Flow

`_render_correlation_matrix(uc_key)`:
1. Reads `_PROFILING_SRC[uc_key]["corr_csv"]` -> gets top-N feature names
2. Samples up to 25,000 rows from `_PROFILING_SRC[uc_key]["raw"]`
3. Filters to numeric columns present in both raw and FE datasets
4. Computes `df.corr(method=pearson|spearman)`
5. Renders `go.Heatmap` with diverging RdBu_r colour scale
6. Shows "high correlation pairs" table with `ProgressColumn` for |r|
