# Adding Use Case D — Customer Churn Prediction (Telco, ROC-AUC)

## Files to Create or Edit

### Files to Edit

| File | What to Change |
|------|---------------|
| `dashboard/app.py` | (1) Add `"D"` block to `USE_CASE_SCRIPTS`. (2) Update `USE_CASE_META["D"]["metric"]` from `"F1"` to `"ROC-AUC"`. (3) Add `"D"` entries to `_PROFILING_SRC`, `_FE_EDA_SRC`, `_FE_GUIDANCE`, and `_DATASET_ANALYSIS_CONFIG`. |
| `config.py` | Update/confirm `DATASET_REGISTRY["D"]` with `target_col: "Churn"`, `kaggle_slug`, `primary_metric: "roc_auc"`, and `data_subdir: "telco_churn"`. |
| `run_platform.py` | Add `"D"` block to `USE_CASE_SCRIPTS`. |

### Files to Create

| File | Purpose |
|------|---------|
| `use_case_D_churn/__init__.py` | Makes the folder a Python package. |
| `use_case_D_churn/01_data_loading.py` | Load Telco CSV, split into train/val/test, cache as Parquet under `data/telco_churn/`. |
| `use_case_D_churn/02_eda_analysis.py` | Statistical profiling, class-balance check (~26.5% churn), correlation heatmap, missing-value report saved to `reports/use_case_D/`. |
| `use_case_D_churn/03_feature_engineering.py` | Encode categoricals (tenure groups, contract type, payment method), SMOTE on train fold only, save `train_fe.parquet` and `val_fe.parquet`. |
| `use_case_D_churn/04_model_training.py` | Baseline (Logistic Regression) + 5-fold stratified CV over candidate algorithms, save all candidate `.pkl` files to `models/use_case_D/`. |
| `use_case_D_churn/05_hyperparameter_tuning.py` | Optuna Bayesian search on CV champion, save `lgbm_optuna_champion.pkl` and `lgbm_optimal_threshold.txt` to `models/use_case_D/`. |
| `data/telco_churn/` | Data directory (place `WA_Fn-UseC_-Telco-Customer-Churn.csv` here; Kaggle slug: `blastchar/telco-customer-churn`). |
| `models/use_case_D/` | Directory for saved model artefacts. |
| `reports/use_case_D/` | Directory for PNG/CSV outputs of Steps 2–5. |

---

## Note on the Existing Scaffolded Entry

`USE_CASE_META["D"]` already exists in `dashboard/app.py` with `"status": "scaffolded"` and `"metric": "F1"`. The two corrections needed before wiring up the pipeline are:

1. Change `"metric": "F1"` → `"metric": "ROC-AUC"`.
2. Change `"status": "scaffolded"` → `"status": "complete"` once all five pipeline scripts run cleanly.

`USE_CASE_SCRIPTS` in both `app.py` and `run_platform.py` does **not** yet have a `"D"` key — that must be added.

---

## Scaffolded `USE_CASE_META` Entry

Add this block to the `USE_CASE_META` dict in `dashboard/app.py` (replacing the existing scaffolded `"D"` entry around line 215):

```python
    "D": {
        "title":      "Customer Churn Analytics",
        "icon":       "👥",
        "tag":        "Telco Customer Churn",
        "target":     "Churn",
        "task":       "Binary Classification",
        "metric":     "ROC-AUC",
        "model_dir":  "use_case_D",
        "data_dir":   "telco_churn",
        "report_dir": "use_case_D",
        "status":     "complete",           # set to "scaffolded" until scripts run cleanly
        "champion":   "lgbm_optuna_champion.pkl",
        "threshold_file": "lgbm_optimal_threshold.txt",
        "description": (
            "Identifies customers likely to churn from a telecom subscription "
            "using the IBM Telco dataset (7 043 customers, 20 features). "
            "Handles class imbalance (~26.5% churn rate) with SMOTE and "
            "Bayesian-tuned LightGBM. Optimises ROC-AUC; SHAP provides "
            "per-customer explainability for proactive retention targeting."
        ),
    },
```

---

## Scaffolded `USE_CASE_SCRIPTS` Entry

Add this block to `USE_CASE_SCRIPTS` in **both** `dashboard/app.py` (after the `"C_nlp"` block, around line 71) and `run_platform.py` (same location):

```python
    "D": {
        1: "use_case_D_churn/01_data_loading.py",
        2: "use_case_D_churn/02_eda_analysis.py",
        3: "use_case_D_churn/03_feature_engineering.py",
        4: "use_case_D_churn/04_model_training.py",
        5: "use_case_D_churn/05_hyperparameter_tuning.py",
    },
```

---

## Supporting Dict Entries

### `config.py` — `DATASET_REGISTRY["D"]`

Confirm or add the following entry in `config.py`:

```python
    "D": {
        "name":          "Customer Intelligence & Retention",
        "use_cases":     ["Customer Churn Prediction"],
        "kaggle_slug":   "blastchar/telco-customer-churn",
        "data_subdir":   "telco_churn",
        "target_col":    "Churn",
        "task_type":     "binary_classification",
        "positive_rate": 0.265,
        "eval_metrics":  ["roc_auc", "pr_auc", "f1"],
        "primary_metric": "roc_auc",
    },
```

### `_PROFILING_SRC["D"]` entry (add in `app.py`)

```python
    "D": {
        "col_summary": "reports/use_case_D/train_column_summary.csv",
        "raw":         "data/telco_churn/train_raw.parquet",
        "target":      "Churn",
        "corr_csv":    "reports/use_case_D/feature_target_correlation.csv",
        "corr_png":    "reports/use_case_D/correlation_top30.png",
        "missing_png": "reports/use_case_D/missing_heatmap.png",
        "outlier_csv": "reports/use_case_D/outlier_report.csv",
    },
```

### `_FE_EDA_SRC["D"]` entry (add in `app.py`)

```python
    "D": {
        "train_fe":      "data/telco_churn/train_fe.parquet",
        "raw":           "data/telco_churn/train_raw.parquet",
        "feat_list":     "reports/use_case_D/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_D/engineered_feature_summary.png",
        "target":        "Churn",
        "target_labels": {0: "Retained", 1: "Churned"},
    },
```

---

## Pipeline Script Boilerplate (for each `01–05_*.py`)

Every script in `use_case_D_churn/` must follow this header pattern (from `architecture.md`):

```python
"""
use_case_D_churn/0N_script_name.py
====================================
DSF504 — Use Case D: Customer Churn Prediction
Step N: <description>

Dataset
-------
IBM Telco Customer Churn (Kaggle: blastchar/telco-customer-churn)
  7 043 customers · 20 features · target: Churn (Yes/No → 1/0)
  ~26.5% churn rate
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE

# ── UTF-8 encoding guard (MUST come before logging.basicConfig) ──────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR    / "telco_churn"
REPORT_DIR  = REPORTS_DIR / "use_case_D"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_SUBDIR.mkdir(parents=True, exist_ok=True)

TARGET = "Churn"
```

Use Use Case B (`use_case_B_credit/`) as the closest copy-template since it is also tabular binary classification optimised for ROC-AUC with SMOTE and LightGBM. Replace all `gmsc_credit` / `use_case_B` / `SeriousDlqin2yrs` references with their Use Case D equivalents.

---

## Verification Steps

After creating all files and making edits:

```powershell
cd C:\DSF504

# 1. Confirm app.py is still valid Python
python -c "import ast; ast.parse(open('dashboard/app.py').read()); print('Syntax OK')"

# 2. Confirm config.py is still valid Python
python -c "import ast; ast.parse(open('config.py').read()); print('Syntax OK')"

# 3. Run Step 1 to verify data loading works
python run_platform.py --use-case D --steps data

# 4. Launch dashboard and confirm Use Case D appears in sidebar with ROC-AUC metric
python run_platform.py --dashboard
```

---

## Key Checklist

| Item | Detail |
|------|--------|
| `USE_CASE_SCRIPTS` key | `"D"` (not `"D_churn"`) — must match the `USE_CASE_META` key exactly |
| Metric correction | Existing entry uses `"F1"`; task requires `"ROC-AUC"` |
| Status field | Leave as `"scaffolded"` until all five pipeline scripts run end-to-end |
| `champion` field | Set to `None` while scaffolded; update to `"lgbm_optuna_champion.pkl"` after Step 5 |
| `ensure_utf8()` position | Must appear **before** `logging.basicConfig()` in every pipeline script |
| `USE_CASE_SCRIPTS` locations | Appears in **both** `dashboard/app.py` and `run_platform.py` — update both |
| Telco target column | `"Churn"` (capital C, string `"Yes"`/`"No"` in raw CSV — must be binarised to 1/0 in Step 1) |
| Data directory | `data/telco_churn/` — does not exist yet; must be created |
| Model/report directories | `models/use_case_D/` and `reports/use_case_D/` — must be created |
