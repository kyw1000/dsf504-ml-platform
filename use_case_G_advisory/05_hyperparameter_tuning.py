"""
use_case_G_advisory/05_hyperparameter_tuning.py
================================================
Use Case G — AmEx Credit Default Prediction
Phase 3, Step 5: Hyperparameter Tuning + Final Training

Bayesian optimisation (Optuna TPE sampler) on the Step-4 champion (LightGBM).
Objective: maximise AmEx metric = 0.5 × (Gini + D-rate@4%)
Search: 3-fold inner CV × Optuna TPE (50 trials default, configurable)
Output: final_model.pkl (tuned), tuning_log.csv, optuna_history.png

Key hyperparameters tuned for LightGBM:
  - num_leaves        (tree complexity)
  - learning_rate     (shrinkage)
  - n_estimators      (ensemble size, with early stopping)
  - subsample         (row sampling per tree)
  - colsample_bytree  (feature sampling per tree)
  - min_child_samples (leaf regularisation)
  - reg_alpha / reg_lambda (L1/L2 regularisation)

Competition context:
  1st place used 5-fold CV as the outer loop with seed 42.
  3rd place used LGB with LGB(TOP 6 features by CV) → Private LB 0.8087.
  The winning hyperparameter space is consistent with standard LGB tuning.

Run
---
    cd C:\\DSF504
    python use_case_G_advisory/05_hyperparameter_tuning.py
"""

from __future__ import annotations

import sys
import time
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE
from config import TUNING_TRIALS, TUNING_TIMEOUT, TUNING_CV_FOLDS
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "amex_default"
REPORT_DIR  = REPORTS_DIR / "use_case_G"
MODEL_DIR   = MODELS_DIR / "use_case_G"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
ID_COL     = "customer_ID"


# ─────────────────────────────────────────────────────────────────────────────
# AmEx metric (replicated from step 4)
# ─────────────────────────────────────────────────────────────────────────────

def amex_metric(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AmEx M = 0.5 × (Gini + D-rate@4%)."""
    from sklearn.metrics import roc_auc_score
    labels_df = pd.DataFrame({"target": y_true, "score": y_score})
    labels_df = labels_df.sort_values("score", ascending=False).reset_index(drop=True)
    n     = len(labels_df)
    n_pos = int(labels_df["target"].sum())
    if n_pos == 0:
        return 0.0
    auc    = roc_auc_score(y_true, y_score)
    gini   = 2 * auc - 1
    top4   = max(1, int(np.ceil(0.04 * n)))
    d_rate = float(labels_df.head(top4)["target"].sum()) / n_pos
    return 0.5 * (gini + d_rate)


def lgb_amex_eval(y_pred: np.ndarray, dataset) -> tuple[str, float, bool]:
    """LightGBM custom eval metric callback."""
    y_true = dataset.get_label()
    score  = amex_metric(y_true, y_pred)
    return "amex_metric", score, True


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_data():
    train_path = DATA_SUBDIR / "train_fe.parquet"
    val_path   = DATA_SUBDIR / "val_fe.parquet"
    if not train_path.exists():
        raise FileNotFoundError("Run 03_feature_engineering.py first.")
    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)
    feat_cols = [c for c in df_train.columns
                 if c not in (ID_COL, TARGET_COL)
                 and df_train[c].dtype != object]
    X_train = df_train[feat_cols].fillna(0)
    y_train = df_train[TARGET_COL]
    X_val   = df_val[feat_cols].fillna(0)
    y_val   = df_val[TARGET_COL]
    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

def _make_objective(X: pd.DataFrame, y: pd.Series, cv_folds: int, random_state: int):
    """
    Returns an Optuna objective function for LightGBM tuning.

    The search space is informed by the top-3 AmEx competition solutions:
    - num_leaves 64–512: competition winners used 64–256 (balanced depth vs. overfitting)
    - learning_rate 0.01–0.1: small enough for 300–1000 trees
    - subsample / colsample_bytree 0.5–1.0: standard stochastic gradient boosting range
    - min_child_samples 20–200: critical regulariser for financial tabular data
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "num_leaves":       trial.suggest_int("num_leaves", 64, 512),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 20, 200),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "is_unbalance":     True,
            "random_state":     random_state,
            "n_jobs":           -1,
            "verbose":          -1,
        }

        fold_scores = []
        for _, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_vl, y_vl)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
            y_pred = model.predict_proba(X_vl)[:, 1]
            fold_scores.append(amex_metric(y_vl.values, y_pred))

        return float(np.mean(fold_scores))

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Grid fallback (no Optuna)
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Manual grid search when Optuna is not installed."""
    log.info("Running manual grid search (Optuna not available)…")

    param_grid = [
        {"num_leaves": 64,  "learning_rate": 0.05, "n_estimators": 300},
        {"num_leaves": 128, "learning_rate": 0.05, "n_estimators": 500},
        {"num_leaves": 256, "learning_rate": 0.03, "n_estimators": 700},
        {"num_leaves": 128, "learning_rate": 0.03, "n_estimators": 700},
    ]

    best_score = -1
    best_params = {}
    rows = []
    for params in param_grid:
        full_params = {
            **params,
            "is_unbalance": True,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**full_params)
        model.fit(X_train, y_train)
        score = amex_metric(y_val.values, model.predict_proba(X_val)[:, 1])
        rows.append({**params, "amex_metric": round(score, 5)})
        log.info(f"  {params} → AmEx={score:.5f}")
        if score > best_score:
            best_score = score
            best_params = full_params

    pd.DataFrame(rows).to_csv(REPORT_DIR / "tuning_log.csv", index=False)
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = TUNING_TRIALS,
    timeout: int  = TUNING_TIMEOUT,
) -> tuple[dict, object]:
    """Run Optuna Bayesian search, return best params and study."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    objective = _make_objective(
        X_train, y_train,
        cv_folds=TUNING_CV_FOLDS,
        random_state=RANDOM_STATE,
    )

    log.info(f"Starting Optuna: {n_trials} trials, timeout={timeout}s")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    elapsed = time.time() - t0

    best = study.best_params
    log.info(
        f"Best AmEx (CV): {study.best_value:.5f}  "
        f"({len(study.trials)} trials, {elapsed:.0f}s)"
    )
    log.info(f"Best params: {best}")

    # Save trial log
    trials_df = study.trials_dataframe()
    trials_df.to_csv(REPORT_DIR / "tuning_log.csv", index=False)

    return best, study


def plot_optuna_history(study, save: bool = True) -> None:
    """Plot optimisation history and parameter importances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # History
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    best_so_far = pd.Series(values).cummax().values
    axes[0].plot(values, alpha=0.5, color="#3949AB", label="Trial AmEx score")
    axes[0].plot(best_so_far, color="#D32F2F", linewidth=2, label="Best so far")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("AmEx Metric (CV)")
    axes[0].set_title("Optuna Optimisation History")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Parameter importance (top params)
    try:
        importances = optuna.importance.get_param_importances(study)
        params_names = list(importances.keys())[:6]
        imp_vals     = [importances[p] for p in params_names]
        axes[1].barh(params_names, imp_vals, color="#66BB6A")
        axes[1].set_xlabel("Relative Importance")
        axes[1].set_title("Hyperparameter Importance\n(Optuna FAnova)")
        axes[1].invert_yaxis()
    except Exception:
        axes[1].text(0.3, 0.5, "Importance\nnot available", transform=axes[1].transAxes)

    plt.tight_layout()
    if save:
        p = REPORT_DIR / "optuna_history.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Final model training
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(
    best_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> object:
    """
    Train the final LightGBM model with best hyperparameters on the full
    training set (train + val) for deployment, and evaluate on val for reporting.

    Important: the val set should NOT be used for final tuning — only for
    reporting final performance. In production, you would train on all
    available labelled data.
    """
    final_params = {
        **best_params,
        "is_unbalance": True,
        "random_state":  RANDOM_STATE,
        "n_jobs":       -1,
        "verbose":      -1,
    }
    # Remove Optuna-specific keys if present
    for k in list(final_params.keys()):
        if k not in lgb.LGBMClassifier().get_params():
            del final_params[k]

    model = lgb.LGBMClassifier(**final_params)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)[:, 1]
    final_score = amex_metric(y_val.values, y_proba)
    from sklearn.metrics import roc_auc_score
    final_auc = roc_auc_score(y_val, y_proba)

    log.info(f"Final model → AmEx: {final_score:.5f}  |  AUC: {final_auc:.5f}")

    # Save
    joblib.dump(model, MODEL_DIR / "final_model.pkl")
    joblib.dump(model, MODEL_DIR / "lgbm_optuna_champion.pkl")
    pd.DataFrame([{
        "model": "LightGBM (tuned)",
        "amex_metric": round(final_score, 5),
        "roc_auc": round(final_auc, 5),
        **{k: v for k, v in best_params.items()},
    }]).to_csv(REPORT_DIR / "final_model_metrics.csv", index=False)

    log.info(f"Final model saved → {MODEL_DIR / 'lgbm_optuna_champion.pkl'}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G: AmEx Credit Default Prediction")
    print("  Step 5: Hyperparameter Tuning + Final Training")
    print("=" * 65 + "\n")

    if not LGB_AVAILABLE:
        print("[!] LightGBM not installed. Run: pip install lightgbm")
        return

    X_train, y_train, X_val, y_val = _load_data()
    print(f"[1] Data loaded — X_train: {X_train.shape}  |  default rate: {y_train.mean():.1%}")

    if OPTUNA_AVAILABLE:
        print(f"\n[2] Optuna Bayesian search ({TUNING_TRIALS} trials)…")
        best_params, study = run_optuna_tuning(X_train, y_train)
        print(f"\n[3] Plotting optimisation history…")
        plot_optuna_history(study)
    else:
        print("\n[2] Optuna not installed — running grid search…")
        print("    Install with: pip install optuna")
        best_params = run_grid_search(X_train, y_train, X_val, y_val)

    print(f"\n[4] Training final model with best hyperparameters…")
    final_model = train_final_model(best_params, X_train, y_train, X_val, y_val)

    print(f"\n[✓] Champion model saved → {MODEL_DIR / 'lgbm_optuna_champion.pkl'}")
    print("\n" + "=" * 65)
    print("  Step 5 complete. Ready for Ethics & Explainability (06_ethics_explainability.py)")
    print("=" * 65 + "\n")

    return final_model


if __name__ == "__main__":
    main()
