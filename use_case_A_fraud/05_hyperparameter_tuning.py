"""
use_case_A_fraud/05_hyperparameter_tuning.py
=============================================
DSF504 ML Platform — Use Case A: Financial Crime & Fraud Detection
Phase 5: Hyperparameter Tuning

Objectives (DSF504 Section 5)
------------------------------
1. GridSearchCV  — exhaustive search on Random Forest (interpretable baseline)
2. RandomizedSearchCV — efficient sampling on XGBoost (tree ensemble)
3. Optuna Bayesian optimisation — surrogate-model-guided search on LightGBM
   (expected champion model from Step 4)
4. Tuned vs untuned comparison table
5. Optimal probability threshold calibration (PR-curve, Youden's J)
6. SHAP global feature importance for the champion model
7. Persist best tuned model to models/use_case_A/

Why Bayesian optimisation matters for fraud detection
------------------------------------------------------
* Fraud datasets are large (590 K rows × 400+ features after engineering).
* Grid search is computationally intractable at this scale.
* Bayesian optimisation with a Tree-structured Parzen Estimator (TPE) converges
  on good hyperparameters in O(50) trials vs O(1000+) for grid search.
* PR-AUC is the objective because the positive class is only ~3.5%.
* All search loops apply SMOTE inside CV folds via ImbPipeline — no leakage.

Usage
-----
    python use_case_A_fraud/05_hyperparameter_tuning.py
"""

from __future__ import annotations

import sys
import time
import logging
import warnings
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── project root on sys.path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    MODELS_DIR, REPORTS_DIR, DATA_DIR,
    RANDOM_STATE, CV_FOLDS, TUNING_TRIALS,
)

# ── UTF-8 encoding guard (fixes garbled output on Windows) ─────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
MODEL_DIR   = MODELS_DIR / "use_case_A"
REPORT_DIR  = REPORTS_DIR / "use_case_A"
DATA_UC_A   = DATA_DIR / "ieee_fraud"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── columns to exclude before training ───────────────────────────────────────
EXCLUDE_COLS = {
    "TransactionID", "TransactionDT", "isFraud",
    "P_emaildomain", "R_emaildomain",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "ProductCD", "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28",
    "id_29", "id_31", "id_33", "id_34", "id_35", "id_36",
    "id_37", "id_38",
}
TARGET_COL  = "isFraud"


# ============================================================================
# Helpers
# ============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns to use as model features."""
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and df[c].dtype in [np.float64, np.float32, np.float16,
                             np.int64, np.int32, np.int16, np.int8]
    ]


def prepare_Xy(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract X, y; replace inf/NaN with column median."""
    X = df[feature_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in X.columns:
        if X[col].isna().any():
            X[col].fillna(X[col].median(), inplace=True)
    y = df[TARGET_COL].values.astype(int)
    return X.values, y


def pr_auc_scorer(estimator, X, y):
    """Custom scorer: average precision (PR-AUC) for imblearn pipelines."""
    y_prob = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y, y_prob)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Return the probability threshold that maximises F1-score.
    Searches [0.05, 0.95] in steps of 0.01.
    """
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


def evaluate_model(
    name: str,
    pipeline,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Full validation-set evaluation at a given threshold (default: optimal)."""
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    if threshold is None:
        threshold = find_optimal_threshold(y_val, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "Model":     name,
        "ROC-AUC":   round(roc_auc_score(y_val, y_prob), 4),
        "PR-AUC":    round(average_precision_score(y_val, y_prob), 4),
        "F1":        round(f1_score(y_val, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_val, y_pred, zero_division=0), 4),
        "Threshold": round(threshold, 3),
        "_y_prob":   y_prob,
        "_y_pred":   y_pred,
    }


def smote_pipeline(estimator) -> ImbPipeline:
    """Wrap estimator with SMOTE (sampling_strategy=0.10) in ImbPipeline."""
    return ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.10, random_state=RANDOM_STATE)),
        ("clf",   estimator),
    ])


def load_trained_model(name_fragment: str):
    """Load the first .pkl file whose name contains `name_fragment`."""
    candidates = list(MODEL_DIR.glob(f"*{name_fragment}*.pkl"))
    if not candidates:
        return None
    path = sorted(candidates)[-1]          # most recent
    log.info(f"Loading pre-trained model: {path.name}")
    return joblib.load(path)


# ============================================================================
# 1. GridSearchCV — Random Forest
# ============================================================================

def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    n_jobs:  int = -1,
) -> Tuple[ImbPipeline, Dict]:
    """
    GridSearchCV over a moderate RF grid.

    Rationale
    ---------
    Random Forest is the most interpretable ensemble in this suite.
    GridSearchCV is feasible here because the RF grid is deliberately small:
    n_estimators × max_depth × min_samples_leaf = 3 × 3 × 2 = 18 combos
    × 5 folds = 90 fits — manageable in ~10 min on a modern laptop.

    Search dimensions
    -----------------
    n_estimators    : tree count; more trees → lower variance, diminishing returns
    max_depth       : controls bias-variance; None = fully grown (overfit risk)
    min_samples_leaf: minimum leaf size; regularises splits on noisy fraud signals
    """
    log.info("=" * 60)
    log.info("GridSearchCV — Random Forest")
    log.info("=" * 60)

    param_grid = {
        "clf__n_estimators":    [200, 400, 600],
        "clf__max_depth":       [8, 12, None],
        "clf__min_samples_leaf":[1, 5],
        "clf__class_weight":    ["balanced_subsample"],
    }

    base_pipeline = smote_pipeline(
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        )
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    log.info(f"Grid size: {np.prod([len(v) for v in param_grid.values()])} combos × {CV_FOLDS} folds")
    t0 = time.time()

    gs = GridSearchCV(
        estimator  = base_pipeline,
        param_grid = param_grid,
        scoring    = pr_auc_scorer,
        cv         = cv,
        n_jobs     = 1,          # SMOTE + RF already parallel internally
        verbose    = 1,
        refit      = True,
    )
    gs.fit(X_train, y_train)

    elapsed = time.time() - t0
    log.info(f"GridSearchCV complete in {elapsed/60:.1f} min")
    log.info(f"Best params : {gs.best_params_}")
    log.info(f"Best CV PR-AUC : {gs.best_score_:.4f}")

    best_pipeline = gs.best_estimator_
    results = evaluate_model("RF (tuned)", best_pipeline, X_val, y_val)
    log.info(f"Val PR-AUC={results['PR-AUC']}  ROC-AUC={results['ROC-AUC']}  "
             f"F1={results['F1']}  Threshold={results['Threshold']}")

    # Save CV results to CSV
    cv_df = pd.DataFrame(gs.cv_results_)
    cv_df.to_csv(REPORT_DIR / "rf_grid_search_results.csv", index=False)

    return best_pipeline, results


# ============================================================================
# 2. RandomizedSearchCV — XGBoost
# ============================================================================

def tune_xgboost(
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    X_val:        np.ndarray,
    y_val:        np.ndarray,
    n_iter:       int = 40,
    n_jobs:       int = -1,
    scale_pos_weight: float = 28.0,
) -> Tuple[ImbPipeline, Dict]:
    """
    RandomizedSearchCV over a wide XGBoost distribution.

    Rationale
    ---------
    XGBoost has ~15 tunable hyperparameters; grid search is intractable.
    RandomizedSearch samples uniformly from distributions, achieving good
    coverage in n_iter=40 trials (each evaluated 5-fold = 200 fits).

    Key hyperparameters
    -------------------
    n_estimators     : iteration count; early stopping not available in pipeline
    max_depth        : tree depth; deeper → more complex interactions, overfit risk
    learning_rate    : shrinkage; lower = slower but better generalisation
    subsample        : row fraction per tree; prevents memorisation
    colsample_bytree : feature fraction per tree; injects diversity
    reg_alpha/lambda : L1/L2 regularisation; critical for high-dimensional data
    min_child_weight : minimum sum of hessians in a child; counteracts SMOTE noise
    gamma            : minimum loss reduction to split; global pruning
    """
    from scipy.stats import uniform, randint, loguniform

    log.info("=" * 60)
    log.info("RandomizedSearchCV — XGBoost")
    log.info("=" * 60)

    param_dist = {
        "clf__n_estimators":    randint(300, 800),
        "clf__max_depth":       randint(4, 10),
        "clf__learning_rate":   loguniform(0.01, 0.3),
        "clf__subsample":       uniform(0.6, 0.4),        # [0.6, 1.0]
        "clf__colsample_bytree":uniform(0.5, 0.5),        # [0.5, 1.0]
        "clf__reg_alpha":       loguniform(1e-4, 10),
        "clf__reg_lambda":      loguniform(1e-4, 10),
        "clf__min_child_weight":randint(1, 20),
        "clf__gamma":           uniform(0, 5),
    }

    base_pipeline = smote_pipeline(
        xgb.XGBClassifier(
            scale_pos_weight = scale_pos_weight,
            tree_method      = "hist",
            eval_metric      = "aucpr",
            random_state     = RANDOM_STATE,
            n_jobs           = n_jobs,
            verbosity        = 0,
        )
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    log.info(f"RandomizedSearch: {n_iter} iterations × {CV_FOLDS} folds = {n_iter * CV_FOLDS} fits")
    t0 = time.time()

    rs = RandomizedSearchCV(
        estimator          = base_pipeline,
        param_distributions = param_dist,
        n_iter             = n_iter,
        scoring            = pr_auc_scorer,
        cv                 = cv,
        n_jobs             = 1,
        verbose            = 1,
        random_state       = RANDOM_STATE,
        refit              = True,
    )
    rs.fit(X_train, y_train)

    elapsed = time.time() - t0
    log.info(f"RandomizedSearchCV complete in {elapsed/60:.1f} min")
    log.info(f"Best params : {rs.best_params_}")
    log.info(f"Best CV PR-AUC : {rs.best_score_:.4f}")

    best_pipeline = rs.best_estimator_
    results = evaluate_model("XGB (tuned)", best_pipeline, X_val, y_val)
    log.info(f"Val PR-AUC={results['PR-AUC']}  ROC-AUC={results['ROC-AUC']}  "
             f"F1={results['F1']}  Threshold={results['Threshold']}")

    cv_df = pd.DataFrame(rs.cv_results_)
    cv_df.to_csv(REPORT_DIR / "xgb_random_search_results.csv", index=False)

    return best_pipeline, results


# ============================================================================
# 3. Optuna Bayesian Optimisation — LightGBM
# ============================================================================

def tune_lightgbm_optuna(
    X_train:          np.ndarray,
    y_train:          np.ndarray,
    X_val:            np.ndarray,
    y_val:            np.ndarray,
    n_trials:         int = TUNING_TRIALS,
    scale_pos_weight: float = 28.0,
) -> Tuple[lgb.LGBMClassifier, Dict]:
    """
    Bayesian hyperparameter optimisation of LightGBM via Optuna TPE sampler.

    Why Optuna + LightGBM?
    ----------------------
    LightGBM is typically the fastest and most accurate model on tabular fraud
    data.  Its large hyperparameter space (num_leaves, min_child_samples,
    feature_fraction, bagging_fraction, lambda_l1/l2, max_bin, min_split_gain)
    benefits most from adaptive search.

    Optuna's TPE sampler builds a probabilistic model of good vs bad regions,
    focusing subsequent trials where improvement is expected — typically
    converging in 30–50 trials vs hundreds for random search.

    SMOTE is applied inside each validation fold using a manual CV loop
    (not sklearn's CV wrapper) to give Optuna direct access to per-trial
    PR-AUC without the overhead of ImbPipeline serialisation.

    Trial pruning
    -------------
    MedianPruner halts unpromising trials after their first k folds evaluate
    below the running median — saving ~30% compute.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("Optuna not installed. Run: pip install optuna --break-system-packages")
        log.warning("Falling back to RandomizedSearch for LightGBM.")
        return _tune_lgbm_fallback(X_train, y_train, X_val, y_val, scale_pos_weight)

    log.info("=" * 60)
    log.info("Optuna Bayesian Optimisation — LightGBM")
    log.info(f"Trials: {n_trials}  |  CV folds: {CV_FOLDS}")
    log.info("=" * 60)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 300, 1000, step=50),
            "num_leaves":         trial.suggest_int("num_leaves", 31, 255),
            "max_depth":          trial.suggest_int("max_depth", 4, 12),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction":   trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction":   trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":       trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1":          trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2":          trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "min_split_gain":     trial.suggest_float("min_split_gain", 0.0, 1.0),
            "max_bin":            trial.suggest_int("max_bin", 63, 511, step=32),
            "scale_pos_weight":   scale_pos_weight,
            "random_state":       RANDOM_STATE,
            "verbosity":          -1,
            "n_jobs":             -1,
        }

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_tr, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_tr, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Apply SMOTE only to fold training data
            smote = SMOTE(sampling_strategy=0.10, random_state=RANDOM_STATE)
            X_fold_tr_res, y_fold_tr_res = smote.fit_resample(X_fold_tr, y_fold_tr)

            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                X_fold_tr_res, y_fold_tr_res,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
            )

            y_prob = clf.predict_proba(X_fold_val)[:, 1]
            pr_auc = average_precision_score(y_fold_val, y_prob)
            fold_scores.append(pr_auc)

            # Report for pruning
            trial.report(np.mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    sampler = TPESampler(seed=RANDOM_STATE)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study   = optuna.create_study(
        direction = "maximize",
        sampler   = sampler,
        pruner    = pruner,
        study_name= "lgbm_fraud_pr_auc",
    )

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    log.info(f"Optuna complete in {elapsed/60:.1f} min")
    log.info(f"Best trial  : #{study.best_trial.number}")
    log.info(f"Best CV PR-AUC : {study.best_value:.4f}")
    log.info(f"Best params : {study.best_params}")

    # Save optimisation history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(REPORT_DIR / "lgbm_optuna_trials.csv", index=False)

    # Retrain champion on full training set with best params + SMOTE
    best_params = study.best_params.copy()
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"]     = RANDOM_STATE
    best_params["verbosity"]        = -1
    best_params["n_jobs"]           = -1

    smote      = SMOTE(sampling_strategy=0.10, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    champion = lgb.LGBMClassifier(**best_params)
    champion.fit(
        X_res, y_res,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    results = evaluate_model("LightGBM (Optuna)", champion, X_val, y_val)
    log.info(f"Val PR-AUC={results['PR-AUC']}  ROC-AUC={results['ROC-AUC']}  "
             f"F1={results['F1']}  Threshold={results['Threshold']}")

    # Wrap champion in a minimal pipeline stub so interface is consistent
    # (champion is a raw LGBMClassifier — SMOTE was applied outside pipeline here)
    return champion, results, study


def _tune_lgbm_fallback(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    scale_pos_weight: float,
) -> Tuple:
    """RandomizedSearch fallback when Optuna is not installed."""
    from scipy.stats import uniform, randint, loguniform

    log.info("Fallback: RandomizedSearchCV for LightGBM")
    param_dist = {
        "clf__n_estimators":    randint(300, 800),
        "clf__num_leaves":      randint(31, 200),
        "clf__learning_rate":   loguniform(0.01, 0.2),
        "clf__feature_fraction":uniform(0.5, 0.5),
        "clf__bagging_fraction":uniform(0.5, 0.5),
        "clf__lambda_l1":       loguniform(1e-4, 10),
        "clf__lambda_l2":       loguniform(1e-4, 10),
        "clf__min_child_samples":randint(10, 100),
    }
    base_pipeline = smote_pipeline(
        lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        )
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(
        base_pipeline, param_dist, n_iter=TUNING_TRIALS,
        scoring=pr_auc_scorer, cv=cv, n_jobs=1,
        verbose=1, random_state=RANDOM_STATE, refit=True,
    )
    rs.fit(X_train, y_train)
    best = rs.best_estimator_
    results = evaluate_model("LightGBM (tuned)", best, X_val, y_val)
    return best, results, None


# ============================================================================
# 4. Load untuned baselines from Step 4 for comparison
# ============================================================================

def load_step4_results() -> pd.DataFrame:
    """
    Load model_comparison.csv saved by Step 4.
    Returns empty DataFrame if file not found.
    """
    csv_path = REPORT_DIR / "model_comparison.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        log.info(f"Loaded Step 4 results: {len(df)} models from {csv_path.name}")
        return df
    log.warning("model_comparison.csv not found — Step 4 results unavailable.")
    return pd.DataFrame()


# ============================================================================
# 5. Threshold calibration plots
# ============================================================================

def plot_threshold_calibration(
    y_val:    np.ndarray,
    y_prob:   np.ndarray,
    model_name: str,
    save_path: Path,
) -> float:
    """
    Plot Precision / Recall / F1 vs threshold curve.
    Marks the F1-optimal and Youden's J optimal thresholds.
    Returns F1-optimal threshold.
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    precisions, recalls, f1s = [], [], []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        precisions.append(precision_score(y_val, y_pred, zero_division=0))
        recalls.append(recall_score(y_val, y_pred, zero_division=0))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))

    # Youden's J = sensitivity + specificity − 1 = TPR − FPR
    fpr_arr, tpr_arr, roc_thr = roc_curve(y_val, y_prob)
    j_scores = tpr_arr - fpr_arr
    youden_thr = roc_thr[np.argmax(j_scores)]

    optimal_f1_thr = thresholds[np.argmax(f1s)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Threshold Calibration — {model_name}", fontsize=13, fontweight="bold")

    # Left: P / R / F1 vs threshold
    ax = axes[0]
    ax.plot(thresholds, precisions, label="Precision", color="#2196F3")
    ax.plot(thresholds, recalls,    label="Recall",    color="#FF9800")
    ax.plot(thresholds, f1s,        label="F1-Score",  color="#4CAF50", linewidth=2)
    ax.axvline(optimal_f1_thr, linestyle="--", color="#4CAF50", alpha=0.7,
               label=f"F1-opt = {optimal_f1_thr:.2f}")
    ax.axvline(youden_thr, linestyle=":", color="#9C27B0", alpha=0.7,
               label=f"Youden = {youden_thr:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Precision-Recall curve with operating point
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_prob)
    pr_auc_val = average_precision_score(y_val, y_prob)

    ax2 = axes[1]
    ax2.plot(recall_curve, precision_curve,
             label=f"PR curve (AUC={pr_auc_val:.3f})", color="#2196F3", linewidth=2)
    # Operating point at optimal threshold
    op_idx   = np.argmax(f1s)
    ax2.scatter([recalls[op_idx]], [precisions[op_idx]],
                marker="*", s=200, color="#4CAF50", zorder=5,
                label=f"F1-opt operating point")
    baseline = y_val.mean()
    ax2.axhline(baseline, linestyle="--", color="grey", alpha=0.5,
                label=f"No-skill baseline ({baseline:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Threshold calibration plot saved → {save_path.name}")

    return float(optimal_f1_thr)


# ============================================================================
# 6. SHAP global feature importance
# ============================================================================

def compute_shap_importance(
    champion,
    X_val:        np.ndarray,
    feature_cols: List[str],
    max_display:  int = 30,
    sample_n:     int = 2000,
) -> pd.DataFrame:
    """
    Compute SHAP values for the champion model on a validation sample.

    For tree models (LGBM, XGB, RF): uses TreeExplainer (fast, exact).
    For linear/other models: uses LinearExplainer or KernelExplainer.

    Returns a DataFrame of mean |SHAP| per feature, sorted descending.
    """
    log.info("Computing SHAP values…")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_val), min(sample_n, len(X_val)), replace=False)
    X_sample = X_val[idx]

    model_type = type(champion).__name__

    try:
        if model_type in ("LGBMClassifier",):
            explainer = shap.TreeExplainer(champion)
            shap_vals = explainer.shap_values(X_sample)
            # Binary classification: shap_values returns list [neg, pos]
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

        elif model_type in ("XGBClassifier",):
            # Extract from ImbPipeline if needed
            clf = champion["clf"] if hasattr(champion, "__getitem__") else champion
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_sample)

        elif model_type in ("RandomForestClassifier",):
            clf = champion["clf"] if hasattr(champion, "__getitem__") else champion
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

        else:
            # Fallback: LinearExplainer (LogReg) or KernelExplainer (others)
            log.info(f"Using KernelExplainer for {model_type} (slow, sampling 200 rows)")
            background = shap.sample(X_sample, 100)
            predict_fn = champion.predict_proba
            explainer  = shap.KernelExplainer(
                lambda x: predict_fn(x)[:, 1], background
            )
            shap_vals  = explainer.shap_values(X_sample[:200])

    except Exception as exc:
        log.warning(f"SHAP computation failed: {exc}")
        return pd.DataFrame()

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":        feature_cols[:len(mean_abs_shap)],
        "mean_abs_shap":  mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    return importance_df


def plot_shap_summary(
    champion,
    X_val:        np.ndarray,
    feature_cols: List[str],
    importance_df: pd.DataFrame,
    save_dir:     Path,
    max_display:  int = 30,
) -> None:
    """
    1. Bar chart: top-N features by mean |SHAP|
    2. SHAP dot plot (beeswarm) showing directionality
    """
    if importance_df.empty:
        log.warning("SHAP importance DataFrame is empty — skipping plots.")
        return

    # --- Bar chart ---
    top_n  = importance_df.head(max_display)
    fig, ax = plt.subplots(figsize=(10, 8))
    colours = ["#E53935" if i < 5 else "#1E88E5" if i < 15 else "#7CB342"
               for i in range(len(top_n))]
    ax.barh(top_n["feature"][::-1], top_n["mean_abs_shap"][::-1], color=colours[::-1])
    ax.set_xlabel("Mean |SHAP value|  (average impact on fraud probability)")
    ax.set_title(f"Top {max_display} Features — Global SHAP Importance\n"
                 "Champion Model: LightGBM (Optuna-tuned)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = save_dir / "shap_bar_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"SHAP bar chart → {path.name}")

    # --- Beeswarm / dot plot ---
    sample_n = min(1000, len(X_val))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_val), sample_n, replace=False)
    X_sample = X_val[idx]

    model_type = type(champion).__name__
    try:
        if model_type == "LGBMClassifier":
            explainer = shap.TreeExplainer(champion)
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        elif model_type in ("XGBClassifier", "RandomForestClassifier"):
            clf = champion["clf"] if hasattr(champion, "__getitem__") else champion
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            log.info("Skipping SHAP beeswarm for non-tree model.")
            return

        # Limit to top-N features
        top_feats = importance_df["feature"].head(max_display).tolist()
        feat_idx  = [feature_cols.index(f) for f in top_feats if f in feature_cols]
        shap_top  = shap_vals[:, feat_idx]
        X_top     = X_sample[:, feat_idx]

        plt.figure(figsize=(10, 9))
        shap.summary_plot(
            shap_top, X_top,
            feature_names=top_feats,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Beeswarm — Feature Impact on Fraud Probability",
                  fontsize=12, fontweight="bold", pad=14)
        plt.tight_layout()
        path2 = save_dir / "shap_beeswarm.png"
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"SHAP beeswarm → {path2.name}")

    except Exception as exc:
        log.warning(f"SHAP beeswarm failed: {exc}")


# ============================================================================
# 7. Comparison plots — tuned vs untuned
# ============================================================================

def plot_tuned_vs_untuned(
    comparison_df: pd.DataFrame,
    save_path:     Path,
) -> None:
    """
    Grouped bar chart comparing PR-AUC, ROC-AUC, and F1 across all model
    variants (untuned from Step 4 + tuned from Step 5).
    """
    if comparison_df.empty:
        log.warning("comparison_df is empty — skipping tuned vs untuned plot.")
        return

    metrics = ["PR-AUC", "ROC-AUC", "F1"]
    models  = comparison_df["Model"].tolist()
    x       = np.arange(len(models))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(models) * 2), 6))
    colours = ["#1E88E5", "#E53935", "#43A047"]

    for i, (metric, colour) in enumerate(zip(metrics, colours)):
        values = comparison_df[metric].tolist()
        bars   = ax.bar(x + i * width, values, width, label=metric, color=colour, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=45,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Untuned vs Tuned Model Comparison\n"
                 "(PR-AUC is primary metric for imbalanced fraud detection)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Shade tuned models lightly
    for i, mdl in enumerate(models):
        if "(tuned)" in mdl or "(Optuna)" in mdl:
            ax.axvspan(i - 0.5 + 0.5 * width, i + 0.5 + 1.5 * width,
                       alpha=0.06, color="#4CAF50")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Tuned vs untuned comparison → {save_path.name}")


def plot_optuna_history(study, save_path: Path) -> None:
    """Plot Optuna optimisation history (trial PR-AUC over iterations)."""
    try:
        import optuna
    except ImportError:
        return
    if study is None:
        return

    df = study.trials_dataframe()
    df_complete = df[df["state"] == "COMPLETE"].copy()

    if df_complete.empty:
        return

    df_complete["best_so_far"] = df_complete["value"].cummax()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df_complete["number"], df_complete["value"],
               alpha=0.4, s=20, color="#1E88E5", label="Trial PR-AUC")
    ax.plot(df_complete["number"], df_complete["best_so_far"],
            color="#E53935", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("CV PR-AUC")
    ax.set_title("Optuna Optimisation History — LightGBM",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Optuna history plot → {save_path.name}")


# ============================================================================
# 8. Business impact of threshold calibration
# ============================================================================

def print_threshold_business_impact(
    y_val:          np.ndarray,
    y_prob:         np.ndarray,
    default_thr:    float = 0.50,
    optimal_thr:    float = None,
    avg_txn_value:  float = 150.0,
    fraud_loss_rate: float = 0.85,
    alert_cost:     float = 12.0,
) -> None:
    """
    Translate threshold calibration into dollar terms.

    Parameters
    ----------
    avg_txn_value   : Average fraudulent transaction value ($)
    fraud_loss_rate : Fraction of fraud loss prevented when fraud is caught
    alert_cost      : Analyst cost to review one false positive alert ($)
    """
    if optimal_thr is None:
        optimal_thr = find_optimal_threshold(y_val, y_prob)

    def business_metrics(thr: float, label: str) -> None:
        y_pred  = (y_prob >= thr).astype(int)
        cm      = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_fraud     = tp + fn
        fraud_caught    = tp
        false_alerts    = fp
        losses_saved    = fraud_caught * avg_txn_value * fraud_loss_rate
        false_alert_cost= false_alerts * alert_cost
        net_benefit     = losses_saved - false_alert_cost
        precision       = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall          = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"\n  ── {label} (threshold={thr:.2f}) ──")
        print(f"  Fraud caught (TP)   : {fraud_caught:,} / {total_fraud:,} "
              f"({100*recall:.1f}% recall)")
        print(f"  False alerts (FP)   : {false_alerts:,} "
              f"(precision={100*precision:.1f}%)")
        print(f"  Est. losses saved   : ${losses_saved:,.0f}")
        print(f"  False-alert cost    : ${false_alert_cost:,.0f}")
        print(f"  Net benefit         : ${net_benefit:,.0f}")

    print("\n" + "=" * 60)
    print("THRESHOLD CALIBRATION — BUSINESS IMPACT")
    print("=" * 60)
    print(f"  Assumptions:")
    print(f"    Avg fraud transaction : ${avg_txn_value}")
    print(f"    Loss prevention rate  : {100*fraud_loss_rate:.0f}%")
    print(f"    False alert review    : ${alert_cost}/alert")

    business_metrics(default_thr, "Default threshold (0.50)")
    business_metrics(optimal_thr, "Optimal threshold (F1-maximising)")

    improvement = (
        (y_prob >= optimal_thr).astype(int).sum() -
        (y_prob >= default_thr).astype(int).sum()
    )
    print(f"\n  → Optimal threshold detects {improvement:+,} more alerts.")
    print("=" * 60)


# ============================================================================
# 9. Main orchestration
# ============================================================================

def main() -> None:
    """
    Full hyperparameter tuning pipeline.

    Steps
    -----
    1. Load feature-engineered parquet from Step 3
    2. Load Step 4 untuned model results for comparison baseline
    3. GridSearchCV → Random Forest
    4. RandomizedSearchCV → XGBoost
    5. Optuna Bayesian → LightGBM (champion)
    6. Threshold calibration for champion
    7. SHAP global feature importance for champion
    8. Comparison table + plots
    9. Save champion model
    """
    print("\n" + "=" * 60)
    print("DSF504 ML Platform — Use Case A: Fraud Detection")
    print("Step 5: Hyperparameter Tuning")
    print("=" * 60 + "\n")

    # ── Load data ────────────────────────────────────────────────────────────
    train_path = DATA_UC_A / "train_fe.parquet"
    val_path   = DATA_UC_A / "val_fe.parquet"

    if not train_path.exists() or not val_path.exists():
        log.error(
            "Feature-engineered parquet files not found.\n"
            f"  Expected: {train_path}\n"
            f"            {val_path}\n"
            "  → Run 03_feature_engineering.py first."
        )
        return

    log.info(f"Loading train parquet from {train_path.name}…")
    df_train = pd.read_parquet(train_path)
    log.info(f"Loading val parquet from {val_path.name}…")
    df_val   = pd.read_parquet(val_path)

    log.info(f"Train: {df_train.shape}  |  Val: {df_val.shape}")
    log.info(f"Fraud rate — train: {df_train[TARGET_COL].mean():.4f}  "
             f"val: {df_val[TARGET_COL].mean():.4f}")

    # ── Prepare arrays ───────────────────────────────────────────────────────
    feature_cols = get_feature_columns(df_train)
    log.info(f"Feature columns: {len(feature_cols)}")

    X_train, y_train = prepare_Xy(df_train, feature_cols)
    X_val,   y_val   = prepare_Xy(df_val,   feature_cols)

    # Estimate class imbalance ratio for scale_pos_weight
    n_neg         = int((y_train == 0).sum())
    n_pos         = int((y_train == 1).sum())
    scale_pos_wt  = round(n_neg / n_pos, 1)
    log.info(f"Class ratio (neg/pos): {scale_pos_wt}")

    # ── Step 4 baseline results ──────────────────────────────────────────────
    step4_df = load_step4_results()
    all_results: List[Dict] = []
    if not step4_df.empty:
        for _, row in step4_df.iterrows():
            all_results.append({
                "Model":     row.get("Model", ""),
                "ROC-AUC":   float(row.get("ROC-AUC", 0)),
                "PR-AUC":    float(row.get("PR-AUC", 0)),
                "F1":        float(row.get("F1", 0)),
                "Precision": float(row.get("Precision", 0)),
                "Recall":    float(row.get("Recall", 0)),
                "Threshold": float(row.get("Threshold", 0.5)),
            })

    # ── Tune models ──────────────────────────────────────────────────────────
    champion      = None
    champion_prob = None
    optuna_study  = None

    # 1. Random Forest — GridSearchCV
    try:
        rf_tuned, rf_results = tune_random_forest(X_train, y_train, X_val, y_val)
        all_results.append({k: v for k, v in rf_results.items() if not k.startswith("_")})
        joblib.dump(rf_tuned, MODEL_DIR / "rf_tuned.pkl")
        log.info("Saved: rf_tuned.pkl")
    except Exception as exc:
        log.error(f"RF tuning failed: {exc}")
        rf_results = {}

    # 2. XGBoost — RandomizedSearchCV
    try:
        xgb_tuned, xgb_results = tune_xgboost(
            X_train, y_train, X_val, y_val,
            n_iter=TUNING_TRIALS, scale_pos_weight=scale_pos_wt,
        )
        all_results.append({k: v for k, v in xgb_results.items() if not k.startswith("_")})
        joblib.dump(xgb_tuned, MODEL_DIR / "xgb_tuned.pkl")
        log.info("Saved: xgb_tuned.pkl")
    except Exception as exc:
        log.error(f"XGB tuning failed: {exc}")
        xgb_results = {}

    # 3. LightGBM — Optuna
    try:
        lgb_result = tune_lightgbm_optuna(
            X_train, y_train, X_val, y_val,
            n_trials=TUNING_TRIALS, scale_pos_weight=scale_pos_wt,
        )
        if len(lgb_result) == 3:
            lgb_tuned, lgb_results, optuna_study = lgb_result
        else:
            lgb_tuned, lgb_results = lgb_result[:2]
            optuna_study = None

        all_results.append({k: v for k, v in lgb_results.items() if not k.startswith("_")})
        joblib.dump(lgb_tuned, MODEL_DIR / "lgbm_optuna_champion.pkl")
        log.info("Saved: lgbm_optuna_champion.pkl")

        champion      = lgb_tuned
        champion_prob = lgb_results.get("_y_prob")
        if champion_prob is None:
            champion_prob = champion.predict_proba(X_val)[:, 1]

    except Exception as exc:
        log.error(f"LightGBM Optuna tuning failed: {exc}")
        lgb_results = {}

    # ── Comparison table ─────────────────────────────────────────────────────
    comparison_cols = ["Model", "ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Threshold"]
    comparison_df   = pd.DataFrame(all_results)[comparison_cols] if all_results else pd.DataFrame()

    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("PR-AUC", ascending=False).reset_index(drop=True)
        print("\n" + "=" * 60)
        print("MODEL COMPARISON — UNTUNED vs TUNED (Val set)")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(REPORT_DIR / "tuning_comparison.csv", index=False)
        log.info("Saved: tuning_comparison.csv")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_tuned_vs_untuned(comparison_df, REPORT_DIR / "tuned_vs_untuned_comparison.png")

    if optuna_study is not None:
        plot_optuna_history(optuna_study, REPORT_DIR / "optuna_history.png")

    # ── Threshold calibration for champion ───────────────────────────────────
    if champion is not None and champion_prob is not None:
        optimal_thr = plot_threshold_calibration(
            y_val, champion_prob,
            model_name="LightGBM (Optuna-tuned)",
            save_path=REPORT_DIR / "champion_threshold_calibration.png",
        )
        print_threshold_business_impact(
            y_val, champion_prob,
            default_thr=0.50,
            optimal_thr=optimal_thr,
        )

        # Save threshold alongside model
        threshold_path = MODEL_DIR / "lgbm_optimal_threshold.txt"
        threshold_path.write_text(str(optimal_thr))
        log.info(f"Optimal threshold ({optimal_thr:.3f}) saved → {threshold_path.name}")

    # ── SHAP global importance ────────────────────────────────────────────────
    if champion is not None:
        importance_df = compute_shap_importance(
            champion, X_val, feature_cols,
            max_display=30, sample_n=2000,
        )
        if not importance_df.empty:
            importance_df.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
            log.info("Saved: shap_feature_importance.csv")

            plot_shap_summary(
                champion, X_val, feature_cols, importance_df,
                save_dir=REPORT_DIR, max_display=30,
            )

            print("\n" + "=" * 60)
            print("TOP 20 FEATURES BY SHAP IMPORTANCE (Champion: LightGBM Optuna)")
            print("=" * 60)
            print(importance_df.head(20).to_string(index=False))

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 COMPLETE")
    print("=" * 60)
    print(f"  Output directory : {REPORT_DIR}")
    print(f"  Models saved     : {MODEL_DIR}")
    print("  Key outputs:")
    for fname in [
        "tuning_comparison.csv",
        "tuned_vs_untuned_comparison.png",
        "optuna_history.png",
        "champion_threshold_calibration.png",
        "shap_bar_importance.png",
        "shap_beeswarm.png",
        "shap_feature_importance.csv",
        "lgbm_optimal_threshold.txt",
        "lgbm_optuna_champion.pkl",
        "rf_tuned.pkl",
        "xgb_tuned.pkl",
        "rf_grid_search_results.csv",
        "xgb_random_search_results.csv",
        "lgbm_optuna_trials.csv",
    ]:
        status = "✓" if (REPORT_DIR / fname).exists() or (MODEL_DIR / fname).exists() else "·"
        print(f"    {status} {fname}")
    print("\n  → Ready for dashboard (dashboard/app.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
