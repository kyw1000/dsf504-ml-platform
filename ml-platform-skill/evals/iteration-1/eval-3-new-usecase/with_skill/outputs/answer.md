# Adding Use Case D — Customer Churn Prediction (Telco, ROC-AUC)

## Current State

Use Case D is already partially wired into `dashboard/app.py` as a **scaffolded** entry with `metric: "F1"`. The task requires:

1. Correcting the metric from `"F1"` to `"ROC-AUC"` (as specified).
2. Adding `"D"` to `USE_CASE_SCRIPTS` (currently absent — only A, B, C_nlp are present).
3. Creating the five pipeline scripts in `use_case_D_churn/`.
4. Creating the supporting data and output directories.
5. Adding `"D"` entries to the per-page config dicts (`_PROFILING_SRC`, `_FE_EDA_SRC`, `_FE_GUIDANCE`, `_DATASET_ANALYSIS_CONFIG`).

---

## Step-by-Step Plan

### Step 1 — Create directory scaffold

```
mkdir C:\DSF504\data\telco_churn\
mkdir C:\DSF504\models\use_case_D\
mkdir C:\DSF504\reports\use_case_D\
```

`use_case_D_churn\` already exists (scaffolded folder is present).

### Step 2 — Download the Telco dataset

Place the Kaggle Telco Customer Churn CSV at:

```
C:\DSF504\data\telco_churn\WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Kaggle slug: `blastchar/telco-customer-churn`

### Step 3 — Edit `dashboard/app.py`

Make **two** edits (use the Python-read-replace-ast.parse-write method — never `sed -i` on a file with emoji):

**Edit A — `USE_CASE_SCRIPTS`**: Add `"D"` key after the `"C_nlp"` block (around line 71).

**Edit B — `USE_CASE_META["D"]`**: Change `"metric": "F1"` to `"metric": "ROC-AUC"` and set `"status": "complete"` once scripts exist (leave as `"scaffolded"` until Step 4 pipeline scripts are done).

**Edit C — `_PROFILING_SRC`** (around line 1561): Add `"D"` entry.

**Edit D — `_FE_EDA_SRC`** (around line 1856): Add `"D"` entry.

**Edit E — `_FE_GUIDANCE`** and **`_DATASET_ANALYSIS_CONFIG`**: Add `"D"` entries (follow the pattern of `"A"` and `"B"`).

### Step 4 — Create five pipeline scripts in `use_case_D_churn\`

Files to create:
- `use_case_D_churn\01_data_loading.py`
- `use_case_D_churn\02_eda_analysis.py`
- `use_case_D_churn\03_feature_engineering.py`
- `use_case_D_churn\04_model_training.py`
- `use_case_D_churn\05_hyperparameter_tuning.py`

Copy Use Case B (credit) as the closest structural template (also binary classification, tabular, ROC-AUC). Patch every file: replace dataset-specific identifiers, ensure `ensure_utf8()` is called before `logging.basicConfig()`.

### Step 5 — Verify

```
cd C:\DSF504
python -c "import ast; ast.parse(open('dashboard/app.py').read()); print('OK')"
streamlit run dashboard/app.py
```

---

## Ready-to-Paste Dict Entries

### `USE_CASE_SCRIPTS` entry (add after the `"C_nlp"` block, around line 71)

```python
    "D": {
        1: "use_case_D_churn/01_data_loading.py",
        2: "use_case_D_churn/02_eda_analysis.py",
        3: "use_case_D_churn/03_feature_engineering.py",
        4: "use_case_D_churn/04_model_training.py",
        5: "use_case_D_churn/05_hyperparameter_tuning.py",
    },
```

### `USE_CASE_META["D"]` entry (corrected metric; replaces the existing scaffolded entry around line 215)

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
        "status":     "complete",          # change to "scaffolded" until scripts run cleanly
        "champion":   "lgbm_optuna_champion.pkl",
        "threshold_file": "lgbm_optimal_threshold.txt",
        "description": (
            "Identifies customers likely to churn from a telecom subscription "
            "using the IBM Telco dataset (7 043 customers, 20 features). "
            "Handles class imbalance (~26.5% churn rate) with SMOTE and "
            "Bayesian-tuned LightGBM. Optimises ROC-AUC; SHAP provides "
            "per-customer explainability for retention targeting."
        ),
    },
```

### `_PROFILING_SRC["D"]` entry

```python
    "D": {
        "col_summary": "reports/use_case_D/train_column_summary.csv",
        "raw":         "data/telco_churn/train_transaction.parquet",
        "target":      "Churn",
        "corr_csv":    "reports/use_case_D/feature_target_correlation.csv",
        "corr_png":    "reports/use_case_D/correlation_top30.png",
        "missing_png": "reports/use_case_D/missing_heatmap.png",
        "outlier_csv": "reports/use_case_D/outlier_report.csv",
    },
```

### `_FE_EDA_SRC["D"]` entry

```python
    "D": {
        "train_fe":      "data/telco_churn/train_fe.parquet",
        "raw":           "data/telco_churn/train_transaction.parquet",
        "feat_list":     "reports/use_case_D/engineered_features_list.csv",
        "fe_summary":    "reports/use_case_D/engineered_feature_summary.png",
        "target":        "Churn",
        "target_labels": {0: "Retained", 1: "Churned"},
    },
```

---

## Key Notes

| Item | Detail |
|------|--------|
| Existing metric in app.py | `"F1"` — must be changed to `"ROC-AUC"` |
| Existing status | `"scaffolded"` — no scripts exist yet |
| `USE_CASE_SCRIPTS` key | `"D"` (not `"D_churn"`) — matches `USE_CASE_META` key |
| Data directory | `data/telco_churn/` — must be created; no data present yet |
| Model/report dirs | `models/use_case_D/` and `reports/use_case_D/` — must be created |
| Script boilerplate rule | `ensure_utf8()` MUST appear before `logging.basicConfig()` in every script |
| Dashboard will show warning | Until `"status": "complete"` and scripts are present, Run Pipeline shows a scaffold warning — this is intentional |
| `champion` field | Set to `None` while scaffolded; change to `"lgbm_optuna_champion.pkl"` after Step 5 completes |
