"""
use_case_C_market/05_hyperparameter_tuning.py
===============================================
DSF504 Use Case C_markets — Market Intelligence: Realized Volatility Prediction
ML Framework Phase 5: Hyperparameter Tuning & Final Model

Strategy : Optuna TPE, 50 trials on LightGBM
Objective : minimise 3-fold CV RMSPE on training set
Final eval: validation + test sets (original scale RMSPE)

Run:
    cd C:/DSF504
    python use_case_C_market/05_hyperparameter_tuning.py
"""
from __future__ import annotations

import sys
import logging
import pickle
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE

from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_SUBDIR = DATA_DIR  / "optiver_volatility"
MODEL_DIR   = MODELS_DIR / "use_case_C_markets"
REPORT_DIR  = REPORTS_DIR / "use_case_C_markets"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FE_PQ = DATA_SUBDIR / "train_fe.parquet"
VAL_FE_PQ   = DATA_SUBDIR / "val_fe.parquet"
TEST_FE_PQ  = DATA_SUBDIR / "test_fe.parquet"

TARGET_COL   = "target"
LOG_TARGET   = "log_target"
N_TRIALS     = 50
TUNING_FOLDS = 3
MAX_ROWS     = 60_000   # cap for tuning speed


# ══════════════════════════════════════════════════════════════════════════════
# metric
# ══════════════════════════════════════════════════════════════════════════════

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean(((y_pred[mask] - y_true[mask]) / y_true[mask]) ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════

def build_objective(X: np.ndarray, y_log: np.ndarray, y_orig: np.ndarray):
    from sklearn.model_selection import KFold
    import lightgbm as lgb

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 16, 127),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "objective":         "regression",
            "metric":            "rmse",
            "random_state":      RANDOM_STATE,
            "verbose":           -1,
            "n_jobs":            -1,
        }
        kf = KFold(n_splits=TUNING_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_rmspe = []

        for tr_idx, vl_idx in kf.split(X):
            model = lgb.LGBMRegressor(**params)
            model.fit(X[tr_idx], y_log[tr_idx])
            preds = np.expm1(np.clip(model.predict(X[vl_idx]), -10, 10))
            fold_rmspe.append(rmspe(y_orig[vl_idx], preds))

        return float(np.mean(fold_rmspe))

    return objective


# ══════════════════════════════════════════════════════════════════════════════
# reporting
# ══════════════════════════════════════════════════════════════════════════════

def plot_tuning_history(study) -> None:
    trials_df = study.trials_dataframe()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1A1A2E")

    for ax in axes:
        ax.set_facecolor("#1A1A2E")
        ax.tick_params(colors="white")

    # Trial values
    values = [t.value for t in study.trials if t.value is not None]
    axes[0].plot(values, color="#42A5F5", linewidth=1.5)
    best_cummin = np.minimum.accumulate(values)
    axes[0].plot(best_cummin, color="#EF5350", linewidth=2, linestyle="--",
                 label="Best so far")
    axes[0].set_title("Optuna: RMSPE per Trial", color="white", fontsize=12)
    axes[0].set_xlabel("Trial", color="white")
    axes[0].set_ylabel("CV RMSPE", color="white")
    axes[0].legend(facecolor="#1A1A2E", labelcolor="white")

    # Param importance (top 8)
    try:
        import optuna
        imp = optuna.importance.get_param_importances(study)
        top = dict(list(imp.items())[:8])
        keys = list(top.keys())[::-1]
        vals = [top[k] for k in keys]
        axes[1].barh(keys, vals, color="#66BB6A", edgecolor="none")
        axes[1].set_title("Hyperparameter Importance", color="white", fontsize=12)
        axes[1].set_xlabel("Importance", color="white")
    except Exception:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "tuning_history.png",
                dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()
    log.info("Saved tuning_history.png")


def plot_feature_importance(model, fe_cols: list[str]) -> None:
    imp = pd.Series(model.feature_importances_, index=fe_cols)
    top = imp.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")
    ax.barh(top.index[::-1], top.values[::-1], color="#AB47BC", edgecolor="none")
    ax.set_title("LightGBM Feature Importance — Top 20", color="white", fontsize=13)
    ax.set_xlabel("Importance (gain)", color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png",
                dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()

    imp_df = imp.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    log.info("Saved feature_importance.csv + feature_importance.png")


def plot_final_preds(y_true_orig: np.ndarray, y_pred_orig: np.ndarray,
                     split: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1A1A2E")
    for ax in axes:
        ax.set_facecolor("#1A1A2E"); ax.tick_params(colors="white")

    axes[0].scatter(y_true_orig, y_pred_orig, s=5, alpha=0.25, color="#26C6DA")
    lo = min(y_true_orig.min(), y_pred_orig.min())
    hi = max(y_true_orig.max(), y_pred_orig.max())
    axes[0].plot([lo, hi], [lo, hi], color="#EF5350", linewidth=1.5)
    axes[0].set_title(f"Final Model — Pred vs Actual ({split})", color="white")
    axes[0].set_xlabel("Actual RV", color="white")
    axes[0].set_ylabel("Predicted RV", color="white")

    resid = y_pred_orig - y_true_orig
    axes[1].hist(resid, bins=60, color="#FFA726", edgecolor="none", alpha=0.85)
    axes[1].axvline(0, color="#EF5350", linewidth=1.5, linestyle="--")
    axes[1].set_title("Residuals (original scale)", color="white")
    axes[1].set_xlabel("Pred - Actual", color="white")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"final_model_preds_{split}.png",
                dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 60)
    log.info("Use Case C_markets — Step 5: Hyperparameter Tuning")
    log.info("=" * 60)

    for p in (TRAIN_FE_PQ, VAL_FE_PQ, TEST_FE_PQ):
        if not p.exists():
            log.error("%s not found — run Steps 1-3 first", p.name)
            sys.exit(1)

    # ── check LightGBM ─────────────────────────────────────────────────────────
    try:
        import lightgbm as lgb
        log.info("LightGBM version: %s", lgb.__version__)
    except ImportError:
        log.error("LightGBM not installed. Run: pip install lightgbm")
        sys.exit(1)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.error("Optuna not installed. Run: pip install optuna")
        sys.exit(1)

    log.info("Loading feature-engineered splits …")
    df_train = pd.read_parquet(TRAIN_FE_PQ)
    df_val   = pd.read_parquet(VAL_FE_PQ)
    df_test  = pd.read_parquet(TEST_FE_PQ)

    fe_cols = sorted([c for c in df_train.columns
                      if c.startswith("fe_") and
                      pd.api.types.is_numeric_dtype(df_train[c])])
    log.info("  Features: %d  train=%d  val=%d  test=%d",
             len(fe_cols), len(df_train), len(df_val), len(df_test))

    X_train     = df_train[fe_cols].fillna(0).values.astype(np.float32)
    y_train_log = df_train[LOG_TARGET].fillna(0).values
    y_train_orig= df_train[TARGET_COL].fillna(1e-8).values

    X_val       = df_val[fe_cols].fillna(0).values.astype(np.float32)
    y_val_log   = df_val[LOG_TARGET].fillna(0).values
    y_val_orig  = df_val[TARGET_COL].fillna(1e-8).values

    X_test      = df_test[fe_cols].fillna(0).values.astype(np.float32)
    y_test_log  = df_test[LOG_TARGET].fillna(0).values
    y_test_orig = df_test[TARGET_COL].fillna(1e-8).values

    # Sample for tuning speed
    if len(X_train) > MAX_ROWS:
        log.info("  Sampling %d rows for tuning (full: %d)", MAX_ROWS, len(X_train))
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train), MAX_ROWS, replace=False)
        X_cv = X_train[idx]; y_cv_log = y_train_log[idx]; y_cv_orig = y_train_orig[idx]
    else:
        X_cv = X_train; y_cv_log = y_train_log; y_cv_orig = y_train_orig

    # ── Optuna search ──────────────────────────────────────────────────────────
    log.info("Starting Optuna TPE search (%d trials, %d-fold CV) …", N_TRIALS, TUNING_FOLDS)
    t0 = time.time()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        build_objective(X_cv, y_cv_log, y_cv_orig),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    log.info("Tuning done in %.1fs | Best RMSPE=%.5f", time.time() - t0,
             study.best_value)
    log.info("Best params: %s", study.best_params)

    plot_tuning_history(study)

    # Save best params
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df.to_csv(REPORT_DIR / "best_hyperparams.csv", index=False)

    # ── Retrain final model on full training set ───────────────────────────────
    log.info("Retraining on full training set with best params …")
    best_params = {**study.best_params,
                   "objective": "regression", "metric": "rmse",
                   "random_state": RANDOM_STATE, "verbose": -1, "n_jobs": -1}

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train_log)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    def full_eval(X, y_log, y_orig, split_name):
        preds_log  = final_model.predict(X)
        preds_orig = np.expm1(np.clip(preds_log, -10, 10))
        rp  = rmspe(y_orig, preds_orig)
        rmse_log = float(np.sqrt(np.mean((preds_log - y_log) ** 2)))
        r2  = float(1 - np.sum((y_log - preds_log) ** 2) /
                        np.sum((y_log - y_log.mean()) ** 2))
        log.info("  %s — RMSPE=%.5f  RMSE(log)=%.5f  R²=%.4f",
                 split_name, rp, rmse_log, r2)
        plot_final_preds(y_orig, preds_orig, split_name)
        return rp, rmse_log, r2

    val_rmspe,  val_rmse,  val_r2  = full_eval(X_val,  y_val_log,  y_val_orig,  "val")
    test_rmspe, test_rmse, test_r2 = full_eval(X_test, y_test_log, y_test_orig, "test")

    # Save feature importance
    plot_feature_importance(final_model, fe_cols)

    # ── Save final model ───────────────────────────────────────────────────────
    payload = {
        "model":          final_model,
        "fe_cols":        fe_cols,
        "best_params":    study.best_params,
        "val_rmspe":      val_rmspe,
        "test_rmspe":     test_rmspe,
    }
    with open(MODEL_DIR / "lgbm_optuna_champion.pkl", "wb") as f:
        pickle.dump(payload, f)
    log.info("Saved lgbm_optuna_champion.pkl")

    # Final metrics CSV
    metrics_df = pd.DataFrame([{
        "model":           "lgbm_optuna",
        "val_rmspe":       round(val_rmspe, 6),
        "val_rmse_log":    round(val_rmse, 6),
        "val_r2":          round(val_r2, 4),
        "test_rmspe":      round(test_rmspe, 6),
        "test_rmse_log":   round(test_rmse, 6),
        "test_r2":         round(test_r2, 4),
        "best_trial_rmspe":round(study.best_value, 6),
        "n_trials":        N_TRIALS,
    }])
    metrics_df.to_csv(REPORT_DIR / "final_model_metrics.csv", index=False)
    log.info("Saved final_model_metrics.csv")

    log.info("-" * 60)
    log.info("Step 5 complete.")
    log.info("  Final model : LightGBM (Optuna-tuned)")
    log.info("  Val  RMSPE  : %.5f", val_rmspe)
    log.info("  Test RMSPE  : %.5f", test_rmspe)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
