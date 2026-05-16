"""
use_case_E_insurance/05_hyperparameter_tuning.py
==================================================
Use Case E — Insurance Risk & Claims Analytics
Phase 4: Hyperparameter Tuning + Final Model Training

Runs Bayesian hyperparameter optimisation (Optuna TPE) on the champion
model selected in Step 4. Optimises for Normalized Gini Coefficient
(the official Porto Seguro competition metric = 2 × ROC-AUC − 1).

Default champion: LightGBM (best on Porto Seguro in practice)
  - 50 Optuna trials (configurable via TUNING_TRIALS in config.py)
  - 3-fold inner CV per trial (fast; outer fold already done in Step 4)
  - Early stopping on each tree to prevent overfitting

Outputs (saved to models/use_case_E/):
  lgbm_optuna_champion.pkl    — final tuned model
  lgbm_optimal_threshold.txt  — optimal decision threshold (Youden J)
  optuna_tuning_log.csv       — per-trial results

ML Framework Phase: Hyperparameter Optimisation → Final Training → Evaluation

Run
---
    cd C:\\DSF504
    python use_case_E_insurance/05_hyperparameter_tuning.py
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("[!] lightgbm not installed — run: pip install lightgbm")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[!] optuna not installed — run: pip install optuna")

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR, REPORTS_DIR, MODELS_DIR,
    RANDOM_STATE, TUNING_TRIALS, TUNING_TIMEOUT, TUNING_CV_FOLDS,
)

# ── UTF-8 encoding guard ─────────────────────────────────────────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "porto_seguro"
REPORT_DIR  = REPORTS_DIR / "use_case_E"
MODEL_DIR   = MODELS_DIR / "use_case_E"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalized_gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Normalized Gini = 2 × ROC-AUC − 1  (Porto Seguro official metric)."""
    return 2 * roc_auc_score(y_true, y_score) - 1


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Youden's J statistic: threshold that maximises TPR − FPR.
    Better than 0.5 for imbalanced datasets where the decision boundary
    needs to reflect the asymmetric cost of missed claims vs. false alarms.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_stats  = tpr - fpr
    best_idx = np.argmax(j_stats)
    return float(thresholds[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective for LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def lgbm_objective(
    trial: "optuna.trial.Trial",
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = TUNING_CV_FOLDS,
) -> float:
    """
    Optuna objective: maximise mean Normalized Gini across n_folds inner CV.

    Search space covers:
      - n_estimators     : number of boosting rounds
      - num_leaves       : model complexity (main LightGBM parameter)
      - max_depth        : tree depth cap (prevents overfitting)
      - learning_rate    : shrinkage
      - subsample        : row sampling per tree
      - colsample_bytree : feature sampling per tree
      - min_child_samples: minimum data in leaves (regularisation)
      - reg_alpha        : L1 regularisation
      - reg_lambda       : L2 regularisation
    """
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, step=100),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
        "max_depth":         trial.suggest_int("max_depth", 4, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        # Fixed params
        "class_weight":      "balanced",
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        y_proba = model.predict_proba(X_vl)[:, 1]
        gini    = normalized_gini(y_vl, y_proba)
        scores.append(gini)

    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Grid-search fallback (if Optuna not installed)
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_fallback(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Simple manual grid search over key LightGBM parameters.
    Used when Optuna is not installed.
    """
    log.info("Running grid search fallback (Optuna not available)…")
    from sklearn.model_selection import StratifiedKFold

    best_params: dict = {}
    best_gini   = -1.0

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    candidate_grids = [
        {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 600, "num_leaves": 50, "learning_rate": 0.03,
         "subsample": 0.7, "colsample_bytree": 0.7},
        {"n_estimators": 300, "num_leaves": 20, "learning_rate": 0.1,
         "subsample": 0.9, "colsample_bytree": 0.9},
    ]

    for params in candidate_grids:
        scores = []
        for tr_idx, vl_idx in skf.split(X_train, y_train):
            X_tr, X_vl = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
            y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[vl_idx]
            m = lgb.LGBMClassifier(
                **params, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            )
            m.fit(X_tr, y_tr)
            y_proba = m.predict_proba(X_vl)[:, 1]
            scores.append(normalized_gini(y_vl, y_proba))
        mean_gini = float(np.mean(scores))
        log.info(f"  Grid params {params} → Gini={mean_gini:.4f}")
        if mean_gini > best_gini:
            best_gini   = mean_gini
            best_params = params

    log.info(f"Best grid params: {best_params}  (Gini={best_gini:.4f})")
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_tuning_history(trials_df: pd.DataFrame) -> None:
    """Line chart of best Gini over Optuna trials."""
    if trials_df.empty:
        return

    trials_df = trials_df.sort_values("number")
    running_best = trials_df["value"].cummax()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(trials_df["number"], trials_df["value"],
                 ".", color="#1976D2", alpha=0.5, markersize=4, label="Trial Gini")
    axes[0].plot(trials_df["number"], running_best,
                 color="#D32F2F", linewidth=2, label="Best Gini")
    axes[0].set_xlabel("Trial number")
    axes[0].set_ylabel("Normalized Gini (CV mean)")
    axes[0].set_title("Optuna Tuning History")
    axes[0].legend()

    axes[1].hist(trials_df["value"], bins=30, color="#42A5F5", edgecolor="white")
    axes[1].axvline(x=trials_df["value"].max(), color="#D32F2F",
                    linestyle="--", linewidth=1.5, label=f"Best={trials_df['value'].max():.4f}")
    axes[1].set_xlabel("Normalized Gini")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Gini Score Distribution Across Trials")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "optuna_tuning_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'optuna_tuning_history.png'}")


def plot_final_model_evaluation(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float,
) -> None:
    """4-panel evaluation for the final tuned LightGBM model."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    auc   = roc_auc_score(y_val, y_proba)
    gini  = 2 * auc - 1
    prauc = average_precision_score(y_val, y_proba)
    f1    = f1_score(y_val, y_pred, zero_division=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
                xticklabels=["No Claim", "Claim"],
                yticklabels=["No Claim", "Claim"])
    axes[0, 0].set_title(f"Confusion Matrix (thresh={threshold:.3f})\nF1={f1:.4f}")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")

    # Score distribution
    axes[0, 1].hist(y_proba[y_val == 0], bins=60, alpha=0.6, color="#1976D2",
                    label="No Claim", density=True)
    axes[0, 1].hist(y_proba[y_val == 1], bins=60, alpha=0.7, color="#D32F2F",
                    label="Claim", density=True)
    axes[0, 1].axvline(x=threshold, color="black", linestyle="--",
                       label=f"threshold={threshold:.3f}")
    axes[0, 1].set_xlabel("Predicted probability")
    axes[0, 1].set_title("Score Distributions")
    axes[0, 1].legend(fontsize=8)

    # ROC
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    axes[1, 0].plot(fpr, tpr, color="#388E3C", linewidth=2,
                    label=f"AUC={auc:.4f}  Gini={gini:.4f}")
    axes[1, 0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[1, 0].set_xlabel("FPR")
    axes[1, 0].set_ylabel("TPR")
    axes[1, 0].set_title("ROC Curve — Tuned LightGBM")
    axes[1, 0].legend()

    # Feature importance (top 20)
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(
            model.feature_importances_,
            index=X_val.columns if hasattr(X_val, "columns") else range(len(model.feature_importances_)),
        ).sort_values(ascending=False).head(20)
        imp[::-1].plot.barh(ax=axes[1, 1], color="#7B1FA2")
        axes[1, 1].set_xlabel("Feature importance (gain)")
        axes[1, 1].set_title("Top 20 Feature Importances")
    else:
        axes[1, 1].text(0.5, 0.5, "Feature importance\nnot available",
                        ha="center", va="center", transform=axes[1, 1].transAxes)

    fig.suptitle(
        f"Final Tuned LightGBM — Porto Seguro Insurance Risk Scoring\n"
        f"Gini={gini:.4f}  ROC-AUC={auc:.4f}  PR-AUC={prauc:.4f}  F1={f1:.4f}",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "final_model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'final_model_evaluation.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case E: Insurance Risk & Claims Analytics")
    print("  Step 5: Hyperparameter Tuning + Final Model Training")
    print("=" * 65 + "\n")

    if not LGB_AVAILABLE:
        print("[!] LightGBM not available. Install it and re-run.")
        return None

    # Load feature-engineered data
    train_path = DATA_SUBDIR / "train_fe.parquet"
    val_path   = DATA_SUBDIR / "val_fe.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            "train_fe.parquet not found. Run 03_feature_engineering.py first."
        )

    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)

    feat_cols = [c for c in df_train.columns
                 if c not in ("id", TARGET_COL)
                 and df_train[c].dtype != object]

    X_train = df_train[feat_cols].fillna(0)
    y_train = df_train[TARGET_COL]
    X_val   = df_val[feat_cols].fillna(0)
    y_val   = df_val[TARGET_COL]

    print(f"[1] X_train: {X_train.shape}  |  claim rate: {y_train.mean():.3%}")
    print(f"    X_val  : {X_val.shape}    |  claim rate: {y_val.mean():.3%}")

    # ── Hyperparameter search ─────────────────────────────────────────────────
    best_params: dict = {}

    if OPTUNA_AVAILABLE:
        print(f"\n[2] Running Optuna TPE — {TUNING_TRIALS} trials…")
        t0 = time.time()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )
        study.optimize(
            lambda trial: lgbm_objective(trial, X_train, y_train),
            n_trials=TUNING_TRIALS,
            timeout=TUNING_TIMEOUT,
            show_progress_bar=True,
        )

        elapsed = time.time() - t0
        best_trial  = study.best_trial
        best_params = best_trial.params
        best_gini   = best_trial.value

        log.info(
            f"\nOptuna complete: {len(study.trials)} trials in {elapsed:.0f}s\n"
            f"  Best Gini (CV) = {best_gini:.4f}\n"
            f"  Best params    = {best_params}"
        )

        # Save Optuna log
        trials_df = study.trials_dataframe()[["number", "value", "datetime_complete"]]
        trials_df.to_csv(REPORT_DIR / "optuna_tuning_log.csv", index=False)

        # Plot tuning history
        plot_tuning_history(trials_df)

    else:
        print("\n[2] Optuna not available — using grid search fallback…")
        best_params = grid_search_fallback(X_train, y_train)

    # ── Retrain on full training set with best params ─────────────────────────
    print("\n[3] Retraining final model on full training set…")
    final_params = {
        **best_params,
        "class_weight": "balanced",
        "random_state":  RANDOM_STATE,
        "n_jobs":        -1,
        "verbose":       -1,
    }

    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[4] Evaluating final model on validation set…")
    y_proba = final_model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    gini    = normalized_gini(y_val, y_proba)
    pr_auc  = average_precision_score(y_val, y_proba)

    threshold = find_optimal_threshold(y_val.values, y_proba)
    y_pred    = (y_proba >= threshold).astype(int)
    f1        = f1_score(y_val, y_pred, zero_division=0)

    print(
        f"\n  Final Model Results:\n"
        f"    ROC-AUC          : {roc_auc:.4f}\n"
        f"    Normalized Gini  : {gini:.4f}\n"
        f"    PR-AUC           : {pr_auc:.4f}\n"
        f"    F1  (opt thresh) : {f1:.4f}\n"
        f"    Optimal threshold: {threshold:.4f}"
    )

    # ── Save artefacts ────────────────────────────────────────────────────────
    print("\n[5] Saving final model artefacts…")

    # Champion model (matches app.py USE_CASE_META["E"]["champion"])
    champion_path = MODEL_DIR / "lgbm_optuna_champion.pkl"
    joblib.dump(final_model, champion_path)
    log.info(f"Final model saved → {champion_path}")

    # Threshold file
    threshold_path = MODEL_DIR / "lgbm_optimal_threshold.txt"
    threshold_path.write_text(f"{threshold:.6f}", encoding="utf-8")
    log.info(f"Threshold saved → {threshold_path}")

    # Summary metrics CSV
    metrics_df = pd.DataFrame([{
        "roc_auc":   round(roc_auc, 4),
        "gini":      round(gini, 4),
        "pr_auc":    round(pr_auc, 4),
        "f1":        round(f1, 4),
        "threshold": round(threshold, 4),
    }])
    metrics_df.to_csv(REPORT_DIR / "final_model_metrics.csv", index=False)

    # Feature importance
    if hasattr(final_model, "feature_importances_"):
        imp_df = pd.DataFrame({
            "feature":    X_train.columns,
            "importance": final_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
        log.info("Feature importance saved.")

    # Evaluation plots
    plot_final_model_evaluation(final_model, X_val, y_val, threshold)

    print("\n" + "=" * 65)
    print(f"  Step 5 complete.")
    print(f"  Final Normalized Gini : {gini:.4f}")
    print(f"  Final ROC-AUC         : {roc_auc:.4f}")
    print(f"  Champion saved to     : {champion_path}")
    print("=" * 65 + "\n")

    return final_model


if __name__ == "__main__":
    final_model = main()
