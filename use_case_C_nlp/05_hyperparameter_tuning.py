"""
use_case_C_nlp/05_hyperparameter_tuning.py
============================================
DSF504 — Use Case C (NLP): Market Intelligence — Financial Sentiment
Step 5: Hyperparameter Tuning

Objectives
----------
1. GridSearchCV      — Logistic Regression (C, penalty, solver)
2. RandomizedSearchCV — XGBoost (wide distribution sampling)
3. Optuna Bayesian   — LightGBM on FinBERT embeddings (expected champion)
4. Tuned vs untuned comparison table
5. SHAP feature importance for TF-IDF champion (interpretable terms)
6. Confusion matrix analysis and per-class error breakdown
7. Persist champion model to models/use_case_C_nlp/

Why Optuna + LightGBM on FinBERT?
----------------------------------
FinBERT embeddings are dense 768-dim representations containing rich
domain semantics. LightGBM navigates this high-dimensional space more
efficiently than decision trees alone, and Optuna's TPE sampler finds
good hyperparameter regions in ~50 trials versus exhaustive search
requiring thousands of evaluations.

Primary metric: Macro-F1 (balances performance across all 3 sentiment classes)
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
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
)
from sklearn.preprocessing import label_binarize

from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

MODEL_DIR  = MODELS_DIR  / "use_case_C_nlp"
REPORT_DIR = REPORTS_DIR / "use_case_C_nlp"
DATA_DIR_C = DATA_DIR    / "financial_phrasebank"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["negative", "neutral", "positive"]
N_CLASSES   = 3


# ============================================================================
# Helpers
# ============================================================================

def load_features() -> dict:
    """Load all feature arrays saved by Step 3."""
    required = ["X_tfidf_train.npy", "X_tfidf_val.npy",
                "X_combined_train.npy", "X_combined_val.npy",
                "y_train.npy", "y_val.npy"]
    for f in required:
        if not (DATA_DIR_C / f).exists():
            raise FileNotFoundError(f"Missing {f}. Run 03_feature_engineering.py first.")

    data = {
        "X_tfidf_train":    np.load(DATA_DIR_C / "X_tfidf_train.npy"),
        "X_tfidf_val":      np.load(DATA_DIR_C / "X_tfidf_val.npy"),
        "X_combined_train": np.load(DATA_DIR_C / "X_combined_train.npy"),
        "X_combined_val":   np.load(DATA_DIR_C / "X_combined_val.npy"),
        "y_train":          np.load(DATA_DIR_C / "y_train.npy").astype(int),
        "y_val":            np.load(DATA_DIR_C / "y_val.npy").astype(int),
    }
    for key in ["X_finbert_train.npy", "X_finbert_val.npy"]:
        path = DATA_DIR_C / key
        data[key.replace(".npy", "")] = np.load(path) if path.exists() else None

    return data


def macro_f1_scorer(estimator, X, y):
    """Custom scorer returning macro-F1 for CV search."""
    y_pred = estimator.predict(X)
    return f1_score(y, y_pred, average="macro", zero_division=0)


def evaluate_model(
    name: str,
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Full validation-set evaluation."""
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    y_bin  = label_binarize(y_val, classes=[0, 1, 2])
    try:
        roc_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")

    per_class = f1_score(y_val, y_pred, average=None, zero_division=0)

    return {
        "Model":      name,
        "Macro-F1":   round(f1_score(y_val, y_pred, average="macro", zero_division=0), 4),
        "Accuracy":   round(accuracy_score(y_val, y_pred), 4),
        "ROC-AUC":    round(roc_auc, 4),
        "F1-neg":     round(per_class[0], 4),
        "F1-neu":     round(per_class[1], 4),
        "F1-pos":     round(per_class[2], 4),
        "_y_pred":    y_pred,
        "_y_prob":    y_prob,
    }


def load_step4_results() -> pd.DataFrame:
    """Load model_comparison.csv from Step 4."""
    csv_path = REPORT_DIR / "model_comparison.csv"
    if not csv_path.exists():
        log.warning("model_comparison.csv not found — Step 4 results unavailable.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    log.info(f"Loaded Step 4 results: {len(df)} models")

    # Normalise to Step 5 column names
    rename = {
        "model":        "Model",
        "macro_f1":     "Macro-F1",
        "accuracy":     "Accuracy",
        "roc_auc_ovr":  "ROC-AUC",
        "f1_negative":  "F1-neg",
        "f1_neutral":   "F1-neu",
        "f1_positive":  "F1-pos",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ["Model", "Macro-F1", "Accuracy", "ROC-AUC",
                        "F1-neg", "F1-neu", "F1-pos"] if c in df.columns]
    return df[keep]


# ============================================================================
# 1. GridSearchCV — Logistic Regression
# ============================================================================

def tune_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> Tuple:
    """
    GridSearchCV over regularisation strength and penalty type.

    Rationale
    ---------
    Logistic Regression is the standard text classification baseline.
    The C parameter controls L1/L2 regularisation strength — critical
    for high-dimensional TF-IDF features (5000+ dimensions).
    L1 (Lasso) drives sparse coefficients, aiding interpretation;
    L2 (Ridge) retains all features but shrinks their weights.
    """
    log.info("=" * 60)
    log.info("GridSearchCV — Logistic Regression")
    log.info("=" * 60)

    param_grid = {
        "C":            [0.01, 0.1, 1.0, 10.0],
        "penalty":      ["l2"],
        "solver":       ["saga"],
        "max_iter":     [2000],
        "class_weight": ["balanced"],
        "multi_class":  ["multinomial"],
    }
    base = LogisticRegression(random_state=RANDOM_STATE)
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    n_combos = len(param_grid["C"]) * len(param_grid["penalty"])
    log.info(f"Grid: {n_combos} combos × {CV_FOLDS} folds = {n_combos * CV_FOLDS} fits")
    t0 = time.time()

    gs = GridSearchCV(
        estimator  = base,
        param_grid = param_grid,
        scoring    = macro_f1_scorer,
        cv         = cv,
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,
    )
    gs.fit(X_train, y_train)

    elapsed = time.time() - t0
    log.info(f"GridSearch complete in {elapsed/60:.1f} min")
    log.info(f"Best params     : {gs.best_params_}")
    log.info(f"Best CV Macro-F1: {gs.best_score_:.4f}")

    best = gs.best_estimator_
    results = evaluate_model("LR (tuned)", best, X_val, y_val)
    log.info(f"Val Macro-F1={results['Macro-F1']}  Acc={results['Accuracy']}")

    pd.DataFrame(gs.cv_results_).to_csv(REPORT_DIR / "lr_grid_search_results.csv", index=False)
    return best, results


# ============================================================================
# 2. RandomizedSearchCV — XGBoost
# ============================================================================

def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    n_iter:  int = 30,
) -> Tuple:
    """
    RandomizedSearchCV over XGBoost hyperparameter distributions.

    Key parameters for NLP/text features
    -------------------------------------
    subsample / colsample_bytree : Important for sparse TF-IDF —
      column subsampling injects diversity across high-dim feature space.
    max_depth 4–8: Financial sentences are short; deep trees overfit.
    learning_rate: Lower rate with more estimators → better generalisation.
    """
    from scipy.stats import uniform, randint, loguniform

    log.info("=" * 60)
    log.info("RandomizedSearchCV — XGBoost")
    log.info("=" * 60)

    param_dist = {
        "n_estimators":    randint(200, 600),
        "max_depth":       randint(3, 8),
        "learning_rate":   loguniform(0.01, 0.3),
        "subsample":       uniform(0.6, 0.4),
        "colsample_bytree":uniform(0.5, 0.5),
        "reg_alpha":       loguniform(1e-4, 10),
        "reg_lambda":      loguniform(1e-4, 10),
        "min_child_weight":randint(1, 10),
    }
    base = xgb.XGBClassifier(
        objective  = "multi:softprob",
        num_class  = N_CLASSES,
        eval_metric= "mlogloss",
        tree_method= "hist",
        random_state=RANDOM_STATE,
        n_jobs     = -1,
        verbosity  = 0,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    log.info(f"RandomizedSearch: {n_iter} iterations × {CV_FOLDS} folds")
    t0 = time.time()

    rs = RandomizedSearchCV(
        estimator           = base,
        param_distributions = param_dist,
        n_iter              = n_iter,
        scoring             = macro_f1_scorer,
        cv                  = cv,
        n_jobs              = 1,
        verbose             = 1,
        random_state        = RANDOM_STATE,
        refit               = True,
    )
    rs.fit(X_train, y_train)

    elapsed = time.time() - t0
    log.info(f"RandomizedSearch complete in {elapsed/60:.1f} min")
    log.info(f"Best params     : {rs.best_params_}")
    log.info(f"Best CV Macro-F1: {rs.best_score_:.4f}")

    best = rs.best_estimator_
    results = evaluate_model("XGB (tuned)", best, X_val, y_val)
    log.info(f"Val Macro-F1={results['Macro-F1']}  Acc={results['Accuracy']}")

    pd.DataFrame(rs.cv_results_).to_csv(REPORT_DIR / "xgb_random_search_results.csv", index=False)
    return best, results


# ============================================================================
# 3. Optuna Bayesian Optimisation — LightGBM on FinBERT
# ============================================================================

def tune_lightgbm_optuna(
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    X_val:    np.ndarray,
    y_val:    np.ndarray,
    n_trials: int = TUNING_TRIALS,
    feat_label: str = "FinBERT",
) -> Tuple:
    """
    Bayesian HPO of LightGBM via Optuna TPE sampler.

    Objective metric: macro-F1 (averaged across 3 sentiment classes).
    MedianPruner terminates unpromising trials after initial folds.

    FinBERT variant: best expected champion — semantic embeddings + boosting.
    TF-IDF fallback: used when FinBERT embeddings not available.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("Optuna not installed. Run: pip install optuna --break-system-packages")
        return _lgbm_fallback(X_train, y_train, X_val, y_val, feat_label)

    log.info("=" * 60)
    log.info(f"Optuna Bayesian Optimisation — LightGBM ({feat_label})")
    log.info(f"Trials: {n_trials}  |  CV folds: {CV_FOLDS}")
    log.info("=" * 60)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800, step=50),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "class_weight":      "balanced",
            "random_state":      RANDOM_STATE,
            "verbosity":         -1,
            "n_jobs":            -1,
        }

        fold_scores = []
        for fold_idx, (tr_idx, vl_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
            y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]

            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            y_pred = clf.predict(X_vl)
            score  = f1_score(y_vl, y_pred, average="macro", zero_division=0)
            fold_scores.append(score)

            trial.report(np.mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    sampler = TPESampler(seed=RANDOM_STATE)
    pruner  = MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    study   = optuna.create_study(
        direction  = "maximize",
        sampler    = sampler,
        pruner     = pruner,
        study_name = f"lgbm_sentiment_{feat_label.lower()}",
    )

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    log.info(f"Optuna complete in {elapsed/60:.1f} min")
    log.info(f"Best trial: #{study.best_trial.number}  macro-F1={study.best_value:.4f}")
    log.info(f"Best params: {study.best_params}")

    study.trials_dataframe().to_csv(REPORT_DIR / "lgbm_optuna_trials.csv", index=False)

    # Retrain champion on full training set
    best_params = study.best_params.copy()
    best_params.update({"class_weight": "balanced", "random_state": RANDOM_STATE,
                         "verbosity": -1, "n_jobs": -1})

    champion = lgb.LGBMClassifier(**best_params)
    champion.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    results = evaluate_model(f"LightGBM-{feat_label} (Optuna)", champion, X_val, y_val)
    log.info(f"Val Macro-F1={results['Macro-F1']}  Acc={results['Accuracy']}")

    return champion, results, study


def _lgbm_fallback(X_train, y_train, X_val, y_val, feat_label: str) -> Tuple:
    """RandomizedSearch fallback when Optuna is not installed."""
    from scipy.stats import uniform, randint, loguniform
    param_dist = {
        "clf__n_estimators":     randint(200, 600),
        "clf__num_leaves":       randint(15, 127),
        "clf__learning_rate":    loguniform(0.01, 0.3),
        "clf__feature_fraction": uniform(0.5, 0.5),
        "clf__bagging_fraction": uniform(0.5, 0.5),
        "clf__lambda_l1":        loguniform(1e-4, 10),
        "clf__lambda_l2":        loguniform(1e-4, 10),
    }
    base = lgb.LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                               verbosity=-1, n_jobs=-1)
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rs   = RandomizedSearchCV(base, param_dist, n_iter=TUNING_TRIALS,
                               scoring=macro_f1_scorer, cv=cv, n_jobs=1,
                               verbose=1, random_state=RANDOM_STATE, refit=True)
    rs.fit(X_train, y_train)
    best    = rs.best_estimator_
    results = evaluate_model(f"LightGBM-{feat_label} (tuned)", best, X_val, y_val)
    return best, results, None


# ============================================================================
# 4. SHAP feature importance — TF-IDF champion
# ============================================================================

def compute_shap_tfidf(
    model,
    X_val:          np.ndarray,
    vectorizer,
    max_display:    int = 25,
    sample_n:       int = 300,
) -> None:
    """
    SHAP bar chart + beeswarm for TF-IDF LightGBM.

    Uses TreeExplainer (exact, fast for LGBM/XGB/RF).
    Shows which TF-IDF terms most drive each sentiment class.
    """
    log.info("Computing SHAP values for TF-IDF champion…")
    rng     = np.random.RandomState(RANDOM_STATE)
    idx     = rng.choice(len(X_val), min(sample_n, len(X_val)), replace=False)
    X_sample= X_val[idx]

    feature_names = list(vectorizer.get_feature_names_out())

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        # shap_vals is list of (N, F) arrays for each class
        if isinstance(shap_vals, list):
            # Per-class bar chart
            fig, axes = plt.subplots(1, N_CLASSES, figsize=(18, 7))
            for cls_idx, (ax, label) in enumerate(zip(axes, LABEL_NAMES)):
                sv   = np.abs(shap_vals[cls_idx]).mean(axis=0)
                top  = np.argsort(sv)[-max_display:]
                terms = [feature_names[i] for i in top]
                vals  = sv[top]
                colours = ["#E53935" if label == "negative"
                           else "#FB8C00" if label == "neutral"
                           else "#43A047"] * len(terms)
                ax.barh(terms, vals, color=colours, alpha=0.85)
                ax.set_title(f"SHAP — {label.capitalize()}", fontsize=11)
                ax.set_xlabel("Mean |SHAP|")
                ax.tick_params(axis="y", labelsize=7)

            fig.suptitle(
                f"Top {max_display} TF-IDF Terms by SHAP — LightGBM Sentiment Classifier",
                fontsize=12, fontweight="bold",
            )
            plt.tight_layout()
            path = REPORT_DIR / "shap_tfidf_per_class.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"Saved → {path.name}")

            # Global mean |SHAP| across all classes
            global_shap = np.stack([np.abs(sv).mean(axis=0) for sv in shap_vals]).mean(axis=0)
            top_global  = np.argsort(global_shap)[-max_display:]
            importance_df = pd.DataFrame({
                "feature":       [feature_names[i] for i in top_global[::-1]],
                "mean_abs_shap": global_shap[top_global[::-1]],
            })
            importance_df.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
            log.info("Saved: shap_feature_importance.csv")

    except Exception as exc:
        log.warning(f"SHAP computation failed: {exc}")


# ============================================================================
# 5. Comparison plots
# ============================================================================

def plot_confusion_matrix_champion(
    y_val: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    cm  = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=ax, cbar=False,
    )
    ax.set_title(f"Champion Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    # Add classification report as text
    report = classification_report(y_val, y_pred, target_names=LABEL_NAMES)
    fig.text(0.5, -0.02, report, ha="center", fontsize=8, family="monospace")

    plt.tight_layout()
    path = REPORT_DIR / "champion_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def plot_tuned_vs_untuned(comparison_df: pd.DataFrame) -> None:
    if comparison_df.empty or "Macro-F1" not in comparison_df.columns:
        return

    metrics = ["Macro-F1", "Accuracy", "ROC-AUC"]
    avail   = [m for m in metrics if m in comparison_df.columns]
    models  = comparison_df["Model"].tolist()
    x       = np.arange(len(models))
    width   = 0.25
    colours = ["#1E88E5", "#43A047", "#E53935"]

    fig, ax = plt.subplots(figsize=(max(12, len(models) * 2), 6))
    for i, (metric, colour) in enumerate(zip(avail, colours)):
        vals = comparison_df[metric].tolist()
        bars = ax.bar(x + i * width, vals, width, label=metric,
                      color=colour, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Untuned vs Tuned — Financial Sentiment Classifier\n"
        "(Primary metric: Macro-F1)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    for i, mdl in enumerate(models):
        if "(tuned)" in mdl or "(Optuna)" in mdl:
            ax.axvspan(i - 0.5 + 0.5 * width, i + 0.5 + 1.5 * width,
                       alpha=0.06, color="#4CAF50")

    plt.tight_layout()
    path = REPORT_DIR / "tuned_vs_untuned_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def plot_optuna_history(study, save_path: Path) -> None:
    if study is None:
        return
    try:
        df = study.trials_dataframe()
        df = df[df["state"] == "COMPLETE"].copy()
        if df.empty:
            return
        df["best"] = df["value"].cummax()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(df["number"], df["value"], alpha=0.4, s=20,
                   color="#1E88E5", label="Trial Macro-F1")
        ax.plot(df["number"], df["best"], color="#E53935",
                linewidth=2, label="Best so far")
        ax.set_xlabel("Trial number")
        ax.set_ylabel("CV Macro-F1")
        ax.set_title("Optuna History — LightGBM Sentiment Classifier",
                     fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Optuna history → {save_path.name}")
    except Exception as exc:
        log.warning(f"Optuna history plot failed: {exc}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("\n" + "=" * 60)
    print("DSF504 — Use Case C (NLP): Financial Sentiment")
    print("Step 5: Hyperparameter Tuning")
    print("=" * 60 + "\n")

    data = load_features()
    X_combined_train = data["X_combined_train"]
    X_combined_val   = data["X_combined_val"]
    y_train          = data["y_train"]
    y_val            = data["y_val"]

    # Determine FinBERT availability
    finbert_available = data["X_finbert_train"] is not None
    if finbert_available:
        # FinBERT + hand-crafted (combine for champion)
        hc_train = np.load(DATA_DIR_C / "X_hc_train.npy")
        hc_val   = np.load(DATA_DIR_C / "X_hc_val.npy")
        X_fb_train = np.concatenate([data["X_finbert_train"], hc_train], axis=1)
        X_fb_val   = np.concatenate([data["X_finbert_val"],   hc_val],   axis=1)
        log.info(f"FinBERT+HC train shape: {X_fb_train.shape}")
    else:
        log.warning("FinBERT embeddings not found — using TF-IDF for LightGBM champion.")
        X_fb_train, X_fb_val = X_combined_train, X_combined_val

    # Load Step 4 baselines
    step4_df   = load_step4_results()
    all_results: List[Dict] = []
    if not step4_df.empty:
        for _, row in step4_df.iterrows():
            all_results.append({k: row.get(k, float("nan")) for k in
                                 ["Model", "Macro-F1", "Accuracy", "ROC-AUC",
                                  "F1-neg", "F1-neu", "F1-pos"]})

    champion      = None
    champion_pred = None
    champion_name = None
    optuna_study  = None
    vectorizer    = None

    # ── 1. Logistic Regression — GridSearchCV ───────────────────────────────
    try:
        lr_tuned, lr_results = tune_logistic_regression(
            X_combined_train, y_train, X_combined_val, y_val
        )
        all_results.append({k: v for k, v in lr_results.items() if not k.startswith("_")})
        joblib.dump(lr_tuned, MODEL_DIR / "lr_tuned.pkl")
        log.info("Saved: lr_tuned.pkl")
    except Exception as exc:
        log.error(f"LR tuning failed: {exc}")

    # ── 2. XGBoost — RandomizedSearchCV ─────────────────────────────────────
    try:
        xgb_tuned, xgb_results = tune_xgboost(
            X_combined_train, y_train, X_combined_val, y_val,
            n_iter=TUNING_TRIALS,
        )
        all_results.append({k: v for k, v in xgb_results.items() if not k.startswith("_")})
        joblib.dump(xgb_tuned, MODEL_DIR / "xgb_tuned.pkl")
        log.info("Saved: xgb_tuned.pkl")
    except Exception as exc:
        log.error(f"XGB tuning failed: {exc}")

    # ── 3. LightGBM — Optuna ─────────────────────────────────────────────────
    try:
        feat_label = "FinBERT" if finbert_available else "TF-IDF"
        lgb_result = tune_lightgbm_optuna(
            X_fb_train, y_train, X_fb_val, y_val,
            n_trials=TUNING_TRIALS, feat_label=feat_label,
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
        champion_pred = lgb_results.get("_y_pred")
        champion_name = lgb_results["Model"]
        if champion_pred is None:
            champion_pred = champion.predict(X_fb_val)

    except Exception as exc:
        log.error(f"LightGBM Optuna tuning failed: {exc}")

    # ── Comparison table ──────────────────────────────────────────────────────
    comparison_cols = ["Model", "Macro-F1", "Accuracy", "ROC-AUC",
                       "F1-neg", "F1-neu", "F1-pos"]
    comparison_df = pd.DataFrame([{k: r.get(k, float("nan")) for k in comparison_cols}
                                   for r in all_results if "Model" in r])

    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("Macro-F1", ascending=False).reset_index(drop=True)
        print("\n" + "=" * 60)
        print("MODEL COMPARISON — UNTUNED vs TUNED (Val set)")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(REPORT_DIR / "tuning_comparison.csv", index=False)
        log.info("Saved: tuning_comparison.csv")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_tuned_vs_untuned(comparison_df)

    if optuna_study is not None:
        plot_optuna_history(optuna_study, REPORT_DIR / "optuna_history.png")

    # ── Champion confusion matrix ──────────────────────────────────────────────
    if champion is not None and champion_pred is not None:
        plot_confusion_matrix_champion(y_val, champion_pred, champion_name)

    # ── SHAP on TF-IDF LightGBM (interpretable terms) ─────────────────────────
    tfidf_lgbm_path = MODEL_DIR / "LightGBM_TF-IDF_.pkl"
    if not tfidf_lgbm_path.exists():
        # Try alternative name format
        candidates = list(MODEL_DIR.glob("LightGBM*TF*.pkl"))
        if candidates:
            tfidf_lgbm_path = candidates[0]

    vectorizer_path = DATA_DIR_C / "tfidf_vectorizer.pkl"
    if tfidf_lgbm_path.exists() and vectorizer_path.exists():
        try:
            tfidf_model = joblib.load(tfidf_lgbm_path)
            vectorizer  = joblib.load(vectorizer_path)
            compute_shap_tfidf(
                tfidf_model, X_combined_val, vectorizer, max_display=20, sample_n=300
            )
        except Exception as exc:
            log.warning(f"SHAP failed: {exc}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 COMPLETE — Use Case C NLP: Sentiment Classification")
    print("=" * 60)
    print(f"  Output directory : {REPORT_DIR}")
    print(f"  Models saved     : {MODEL_DIR}")
    print("  Key outputs:")
    for fname in [
        "tuning_comparison.csv",
        "tuned_vs_untuned_comparison.png",
        "optuna_history.png",
        "champion_confusion_matrix.png",
        "shap_tfidf_per_class.png",
        "shap_feature_importance.csv",
        "lgbm_optuna_champion.pkl",
        "lr_tuned.pkl",
        "xgb_tuned.pkl",
        "lr_grid_search_results.csv",
        "xgb_random_search_results.csv",
        "lgbm_optuna_trials.csv",
    ]:
        status = "✓" if (REPORT_DIR / fname).exists() or (MODEL_DIR / fname).exists() else "·"
        print(f"    {status} {fname}")
    print("\n  → Ready for dashboard (python run_platform.py --dashboard)")
    print("=" * 60)


if __name__ == "__main__":
    main()
