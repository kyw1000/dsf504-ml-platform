"""
use_case_A_fraud/05b_lgbm_champion.py
======================================
DSF504 Use Case A — Fraud Detection
Targeted LightGBM Optuna Tuning (continuation of Step 5)

Why this script exists
----------------------
Step 5 (05_hyperparameter_tuning.py) applies SMOTE inside every Optuna
trial's CV loop — 100 trials × 5 folds = 500 SMOTE fits on a dataset with
795 features and ~470k rows.  Computing k-nearest neighbours in 795-D space
for each SMOTE call requires O(n_minority²) distance evaluations, which
exhausts RAM and causes a silent OOM that the outer try/except catches.

This script replaces in-loop SMOTE with LightGBM's native `scale_pos_weight`
(mathematically equivalent for gradient-boosted trees) and runs 50 Optuna
trials with 3-fold CV — producing a high-quality champion without the memory
spike.

Outputs
-------
  models/use_case_A/lgbm_optuna_champion.pkl
  models/use_case_A/lgbm_optimal_threshold.txt
  reports/use_case_A/lgbm_optuna_trials.csv
  reports/use_case_A/lgbm_optuna_history.png
  reports/use_case_A/shap_feature_importance.csv
  reports/use_case_A/shap_bar_importance.png
  logs/use_case_A_lgbm_champion.log            ← persistent run log

Run
---
    cd C:\\DSF504
    python use_case_A_fraud/05b_lgbm_champion.py
"""
from __future__ import annotations

import sys
import time
import logging
import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

# ── project root ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE

from utils.encoding_guard import ensure_utf8
ensure_utf8()

# ── persistent file log ───────────────────────────────────────────────────────
LOG_DIR  = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "use_case_A_lgbm_champion.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR  = MODELS_DIR  / "use_case_A"
REPORT_DIR = REPORTS_DIR / "use_case_A"
DATA_PATH  = DATA_DIR    / "ieee_fraud"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET      = "isFraud"
N_TRIALS    = 50     # Optuna trials — 50 is sufficient for LGB's smooth landscape
CV_FOLDS    = 3      # 3-fold is fast enough; PR-AUC variance is low on 470k rows
PALETTE     = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    fc_path = MODEL_DIR / "feature_cols.pkl"
    if not fc_path.exists():
        raise FileNotFoundError(
            "feature_cols.pkl not found. Run Step 3 (feature engineering) first."
        )
    fe_cols = joblib.load(fc_path)
    log.info("  Feature columns: %d", len(fe_cols))

    train_path = DATA_PATH / "train_fe.parquet"
    val_path   = DATA_PATH / "val_fe.parquet"
    for p in (train_path, val_path):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found. Run Steps 1–3 first.")

    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)
    log.info("  Train: %s   Val: %s", df_train.shape, df_val.shape)

    # Align columns
    for c in fe_cols:
        if c not in df_train.columns:
            df_train[c] = 0.0
        if c not in df_val.columns:
            df_val[c] = 0.0

    X_train = df_train[fe_cols].fillna(0).values.astype(np.float32)
    y_train = df_train[TARGET].values.astype(int)
    X_val   = df_val[fe_cols].fillna(0).values.astype(np.float32)
    y_val   = df_val[TARGET].values.astype(int)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw   = round(n_neg / n_pos, 1)
    log.info("  Fraud rate train=%.4f  val=%.4f  scale_pos_weight=%.1f",
             y_train.mean(), y_val.mean(), spw)
    return X_train, y_train, X_val, y_val, fe_cols, spw


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def run_optuna(X_train, y_train, X_val, y_val, spw: float):
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise RuntimeError(
            "optuna not installed. Run: pip install optuna --break-system-packages"
        )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            # Tree structure
            "n_estimators":      trial.suggest_int("n_estimators", 300, 1500, step=100),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            # Learning
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            # Sampling (replaces SMOTE — no OOM risk)
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),
            # Regularisation
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
            # Imbalance — native handling replaces in-loop SMOTE
            "scale_pos_weight":  spw,
            # Fixed
            "random_state":      RANDOM_STATE,
            "verbosity":         -1,
            "n_jobs":            -1,
        }

        fold_scores = []
        for fold_idx, (tr_idx, vl_idx) in enumerate(cv.split(X_train, y_train)):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[vl_idx], y_train[vl_idx])],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            y_prob = clf.predict_proba(X_train[vl_idx])[:, 1]
            fold_scores.append(average_precision_score(y_train[vl_idx], y_prob))

            trial.report(np.mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = TPESampler(seed=RANDOM_STATE),
        pruner     = MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        study_name = "lgbm_fraud_pr_auc_v2",
    )

    log.info("Starting Optuna: %d trials × %d folds", N_TRIALS, CV_FOLDS)
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    elapsed = time.time() - t0

    log.info("Optuna complete in %.1f min", elapsed / 60)
    log.info("Best trial #%d  CV PR-AUC=%.4f",
             study.best_trial.number, study.best_value)
    log.info("Best params: %s", study.best_params)

    # Save trial history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(REPORT_DIR / "lgbm_optuna_trials.csv", index=False)

    return study


# ── final retrain ─────────────────────────────────────────────────────────────

def retrain_champion(study, X_train, y_train, X_val, y_val, spw: float):
    best_params = study.best_params.copy()
    best_params.update({
        "scale_pos_weight": spw,
        "random_state":     RANDOM_STATE,
        "verbosity":        -1,
        "n_jobs":           -1,
    })

    log.info("Retraining champion on full train set…")
    champion = lgb.LGBMClassifier(**best_params)
    champion.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    y_prob = champion.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_prob)
    roc    = roc_auc_score(y_val, y_prob)

    # Optimal threshold via PR curve
    prec, rec, thr = precision_recall_curve(y_val, y_prob)
    f1_scores = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best_thr  = float(thr[np.argmax(f1_scores[:-1])])
    y_pred    = (y_prob >= best_thr).astype(int)
    f1        = f1_score(y_val, y_pred)

    log.info("Champion val metrics — PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  Thr=%.3f",
             pr_auc, roc, f1, best_thr)
    return champion, y_prob, best_thr, {"PR-AUC": pr_auc, "ROC-AUC": roc,
                                         "F1": f1, "Threshold": best_thr}


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_optuna_history(study):
    trials = study.trials_dataframe()
    completed = trials[trials["state"] == "COMPLETE"].copy()
    completed["best_so_far"] = completed["value"].cummax()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(completed["number"], completed["value"],
               alpha=0.5, color=PALETTE[0], s=30, label="Trial PR-AUC")
    ax.plot(completed["number"], completed["best_so_far"],
            color=PALETTE[3], linewidth=2, label="Best so far")
    ax.set_xlabel("Trial #")
    ax.set_ylabel("CV PR-AUC")
    ax.set_title("Optuna Optimisation History — LightGBM Fraud Champion")
    ax.legend()
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "lgbm_optuna_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved lgbm_optuna_history.png")


def plot_feature_importance(champion, fe_cols):
    imp = champion.feature_importances_
    df_fi = (pd.DataFrame({"feature": fe_cols, "mean_abs_shap": imp})
             .sort_values("mean_abs_shap", ascending=False)
             .reset_index(drop=True))
    df_fi.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
    log.info("Saved shap_feature_importance.csv (%d features)", len(df_fi))

    top = df_fi.head(25)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
            color=PALETTE[0], edgecolor="none")
    ax.set_xlabel("LightGBM Feature Importance (split gain)")
    ax.set_title("Top 25 Features — LightGBM Champion (UC A Fraud)")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_bar_importance.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 62)
    log.info("  05b: LightGBM Optuna Champion — Use Case A (Fraud)")
    log.info("  Log: %s", LOG_FILE)
    log.info("=" * 62)

    X_train, y_train, X_val, y_val, fe_cols, spw = load_data()

    study   = run_optuna(X_train, y_train, X_val, y_val, spw)
    champion, y_prob, best_thr, metrics = retrain_champion(
        study, X_train, y_train, X_val, y_val, spw
    )

    # Save model + threshold
    joblib.dump(champion, MODEL_DIR / "lgbm_optuna_champion.pkl")
    log.info("Saved lgbm_optuna_champion.pkl")

    (MODEL_DIR / "lgbm_optimal_threshold.txt").write_text(str(best_thr))
    log.info("Saved lgbm_optimal_threshold.txt  (threshold=%.3f)", best_thr)

    # Save metrics summary
    pd.DataFrame([{"Step": "05b_lgbm_champion", **metrics}]).to_csv(
        REPORT_DIR / "lgbm_champion_metrics.csv", index=False
    )

    # Plots
    plot_optuna_history(study)
    plot_feature_importance(champion, fe_cols)

    log.info("=" * 62)
    log.info("  05b complete — champion saved to %s", MODEL_DIR)
    log.info("  Full log at: %s", LOG_FILE)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
