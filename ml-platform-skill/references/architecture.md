# DSF504 Architecture Reference

## File Tree

```
C:\DSF504\
├── config.py                    ← DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE, CV_FOLDS
├── run_platform.py              ← CLI runner (alternative to dashboard)
├── requirements.txt
├── dashboard/
│   ├── app.py                   ← ALL dashboard code (~5 300 lines)
│   └── assets/
│       └── ml_framework.png     ← sidebar banner (place here to activate)
├── utils/
│   ├── data_loader.py           ← KaggleLoader, DataProfiler, smart_split, reduce_mem_usage
│   ├── encoding_guard.py        ← ensure_utf8() — call before logging.basicConfig()
│   └── __init__.py
├── use_case_A_fraud/
│   ├── 01_data_loading.py
│   ├── 02_eda_analysis.py
│   ├── 03_feature_engineering.py
│   ├── 04_model_training.py
│   └── 05_hyperparameter_tuning.py
├── use_case_B_credit/           ← same 01–05 structure
├── use_case_C_nlp/              ← same 01–05 structure
├── use_case_C_market/           ← scaffolded (no scripts)
├── use_case_D_churn/            ← scaffolded
├── use_case_E_insurance/        ← scaffolded
├── use_case_F_esg/              ← scaffolded
├── use_case_G_advisory/         ← scaffolded
├── data/
│   ├── ieee_fraud/              ← Use Case A raw + parquet splits
│   ├── gmsc_credit/             ← Use Case B
│   └── nlp_sentiment/           ← Use Case C_nlp
├── models/
│   ├── use_case_A/              ← champion.pkl, final_model.pkl, threshold.txt
│   ├── use_case_B/
│   └── use_case_C_nlp/
└── reports/
    ├── use_case_A/              ← CSV + PNG outputs of Steps 2–5
    ├── use_case_B/
    └── use_case_C_nlp/
```

## Pipeline Script Boilerplate

Every `01–05_*.py` must start with this pattern (order matters):

```python
"""
use_case_X_name/0N_script_name.py
<docstring with dataset, ML framework phase, run instructions>
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

# ... other stdlib / third-party imports ...

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (DATA_DIR, REPORTS_DIR, RANDOM_STATE, ...)
from utils.data_loader import ...

# ── UTF-8 encoding guard (fixes garbled output on Windows) ───────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
```

`ensure_utf8()` must appear **before** `logging.basicConfig()` — the
StreamHandler captures stderr at that point; if stderr is still cp1252,
all emoji/unicode in log messages will be garbled on Windows.

## Subprocess Calls in app.py

Both `subprocess.Popen()` calls in `_run_step()` must have:

```python
proc = subprocess.Popen(
    [sys.executable, str(script_path)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",          # ← decode stdout as UTF-8
    cwd=str(ROOT),
    env={**os.environ, "PYTHONIOENCODING": "utf-8"},  # ← subprocess inherits UTF-8
)
```

## USE_CASE_META Schema

```python
USE_CASE_META = {
    "A": {
        "title":      str,   # human-readable name
        "icon":       str,   # emoji
        "tag":        str,   # dataset name / Kaggle tag
        "target":     str,   # target column name
        "task":       str,   # "Binary Classification" | "Regression" | ...
        "metric":     str,   # primary metric, e.g. "PR-AUC"
        "model_dir":  str,   # relative path under models/
        "data_dir":   str,   # relative path under data/
        "report_dir": str,   # relative path under reports/
        "status":     str,   # "complete" | "scaffolded"
    },
    ...
}
```

## _PROFILING_SRC Schema (Data Profiling Studio)

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
    },
    ...
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
        "target":        "isFraud",
        "target_labels": {0: "Legitimate", 1: "Fraud"},
    },
    ...
}
```

## app.py Top-Level Constants

```python
ROOT = Path(__file__).resolve().parents[1]      # C:\DSF504

USE_CASE_SCRIPTS: dict  # {uc_key: {step_int: "relative/path.py"}}
STEP_NAMES: dict        # {1..5: str}
STEP_DESCRIPTIONS: dict # {1..5: str}
PALETTE: dict           # color tokens (primary, danger, success, warning, purple, grey)
USE_CASE_META: dict     # per use-case metadata
```

## PAGES Dict + _NAV_SECTIONS Alignment

The sidebar button label must be **byte-for-byte identical** to the PAGES key.
Both live in `render_sidebar()` and the global `PAGES` dict at the bottom
of app.py. To add a page:

1. Write `def page_foo(uc_key: str) -> None: ...`
2. In `_NAV_SECTIONS`: `("MY SECTION", ["🆕 My Page"])`
3. In `PAGES`: `"🆕 My Page": page_foo,`

Any whitespace difference (e.g. trailing space, different emoji variant)
breaks the active-page highlighting — verify with `.encode('utf-8').hex()`.

## Correlation Matrix Data Flow

`_render_correlation_matrix(uc_key)`:
1. Reads `_PROFILING_SRC[uc_key]["corr_csv"]` → gets top-N feature names
2. Reads a sample (configurable, default 25 000 rows) from `_PROFILING_SRC[uc_key]["raw"]`
3. Filters to columns that are numeric in the raw parquet
4. Computes `df.corr(method=pearson|spearman)` on those columns
5. Renders a `go.Heatmap` with diverging RdBu_r colour scale
6. Shows a "high correlation pairs" table with `ProgressColumn` for |r|
