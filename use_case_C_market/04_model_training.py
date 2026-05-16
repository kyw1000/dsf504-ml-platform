"""
use_case_C_market/04_model_training.py
========================================
DSF504 Use Case C_markets — Market Intelligence: Realized Volatility Prediction
ML Framework Phase 4: Algorithm Selection & Cross-Validation

Models evaluated (regression):
  1. Ridge Regression (baseline — linear)
  2. Random Forest Regressor
  3. XGBoost Regressor
  4. LightGBM Regressor (expected champion)

Primary metric : RMSPE = sqrt( mean( ((y_pred - y_true) / y_true)^2 ) )
Secondary       : RMSE, MAE, R²

Note: we model log1p(target) and expm1 the predictions to compute RMSPE
on the original scale.

Run:
    cd C:/DSF504
    python use_case_C_market/04_model_training.py
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE, CV_FOLDS

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

TARGET_COL   = "target"       # original scale
LOG_TARGET   = "log_target"   # log1p scale (what we train on)


# ══════════════════════════════════════════════════════════════════════════════
# metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Percentage Error (original scale)."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    pct_err = (y_pred[mask] - y_true[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(pct_err ** 2)))


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray,
             y_true_orig: np.ndarray) -> dict:
    """Compute all metrics. Predictions are on log scale; convert for RMSPE."""
    y_pred_orig = np.expm1(np.clip(y_pred_log, -10, 10))

    rmse = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae  = float(mean_absolute_error(y_true_log, y_pred_log))
    r2   = float(r2_score(y_true_log, y_pred_log))
    rp   = rmspe(y_true_orig, y_pred_orig)

    return {"rmspe": round(rp, 6), "rmse": round(rmse, 6),
            "mae": round(mae, 6), "r2": round(r2, 6)}


# ══════════════════════════════════════════════════════════════════════════════
# model registry
# ══════════════════════════════════════════════════════════════════════════════

def get_models(n_features: int) -> dict:
    models = {
        "ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=8,
            min_samples_leaf=10, n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    except ImportError:
        log.warning("XGBoost not installed — skipping")

    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05,
            num_leaves=63, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20,
            objective="regression",
            metric="rmse",
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    except ImportError:
        log.warning("LightGBM not installed — skipping")

    return models


# ══════════════════════════════════════════════════════════════════════════════
# cross-validation
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate(model, X: np.ndarray, y_log: np.ndarray,
                   y_orig: np.ndarray, name: str) -> dict:
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_rmspe, fold_rmse, fold_mae, fold_r2 = [], [], [], []

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X), 1):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = y_log[tr_idx], y_log[vl_idx]
        y_orig_vl  = y_orig[vl_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_vl)
        m = evaluate(y_vl, preds, y_orig_vl)

        fold_rmspe.append(m["rmspe"])
        fold_rmse.append(m["rmse"])
        fold_mae.append(m["mae"])
        fold_r2.append(m["r2"])
        log.info("  %s fold %d — RMSPE=%.5f  RMSE=%.5f  R²=%.4f",
                 name, fold, m["rmspe"], m["rmse"], m["r2"])

    return {
        "rmspe_mean": round(np.mean(fold_rmspe), 5),
        "rmspe_std":  round(np.std(fold_rmspe), 5),
        "rmse_mean":  round(np.mean(fold_rmse), 5),
        "mae_mean":   round(np.mean(fold_mae), 5),
        "r2_mean":    round(np.mean(fold_r2), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# reporting
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(results: pd.DataFrame) -> None:
    metrics = ["rmspe_mean", "rmse_mean", "r2_mean"]
    titles  = ["RMSPE (lower = better)", "RMSE on log scale (lower = better)",
               "R² on log scale (higher = better)"]
    colors  = ["#EF5350", "#FFA726", "#66BB6A"]
    ascending = [True, True, False]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#1A1A2E")

    for ax, metric, title, col, asc in zip(axes, metrics, titles, colors, ascending):
        ax.set_facecolor("#1A1A2E")
        df_sorted = results.sort_values(metric, ascending=asc)
        bars = ax.barh(df_sorted["model"], df_sorted[metric],
                       color=col, edgecolor="none")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")

        # annotate bars
        for bar, val in zip(bars, df_sorted[metric]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.5f}", va="center", color="white", fontsize=8)

    plt.suptitle("Model Comparison — 5-Fold CV", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png",
                dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()
    log.info("Saved model_comparison.png")


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str) -> None:
    """Scatter plot of predictions vs actuals on validation set."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1A1A2E")

    for ax in axes:
        ax.set_facecolor("#1A1A2E")
        ax.tick_params(colors="white")

    # Log scale
    axes[0].scatter(y_true, y_pred, s=5, alpha=0.3, color="#42A5F5")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], color="#EF5350", linewidth=1.5)
    axes[0].set_xlabel("Actual log1p(target)", color="white")
    axes[0].set_ylabel("Predicted log1p(target)", color="white")
    axes[0].set_title(f"{model_name} — Pred vs Actual (log scale)", color="white")

    # Residuals
    resid = y_pred - y_true
    axes[1].hist(resid, bins=60, color="#AB47BC", edgecolor="none", alpha=0.85)
    axes[1].axvline(0, color="#EF5350", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Residual (pred - actual)", color="white")
    axes[1].set_ylabel("Count", color="white")
    axes[1].set_title("Residual Distribution", color="white")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"val_pred_vs_actual_{model_name}.png",
                dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 60)
    log.info("Use Case C_markets — Step 4: Model Training")
    log.info("=" * 60)

    for p in (TRAIN_FE_PQ, VAL_FE_PQ):
        if not p.exists():
            log.error("%s not found — run Step 3 first", p.name)
            sys.exit(1)

    log.info("Loading feature-engineered splits …")
    df_train = pd.read_parquet(TRAIN_FE_PQ)
    df_val   = pd.read_parquet(VAL_FE_PQ)

    # Feature columns
    fe_cols = sorted([c for c in df_train.columns
                      if c.startswith("fe_") and
                      pd.api.types.is_numeric_dtype(df_train[c])])
    log.info("  Feature cols: %d", len(fe_cols))

    if not fe_cols:
        log.error("No fe_ columns found — check Step 3 output")
        sys.exit(1)

    # ── prepare arrays ─────────────────────────────────────────────────────────
    X_train     = df_train[fe_cols].fillna(0).values.astype(np.float32)
    y_train_log = df_train[LOG_TARGET].fillna(0).values.astype(np.float64)
    y_train_orig= df_train[TARGET_COL].fillna(1e-6).values.astype(np.float64)

    X_val       = df_val[fe_cols].fillna(0).values.astype(np.float32)
    y_val_log   = df_val[LOG_TARGET].fillna(0).values.astype(np.float64)
    y_val_orig  = df_val[TARGET_COL].fillna(1e-6).values.astype(np.float64)

    # ── sample for speed if very large ────────────────────────────────────────
    MAX_ROWS = 80_000
    if len(X_train) > MAX_ROWS:
        log.info("  Sampling %d rows for CV (full dataset: %d)", MAX_ROWS, len(X_train))
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train), MAX_ROWS, replace=False)
        X_cv = X_train[idx]; y_cv_log = y_train_log[idx]; y_cv_orig = y_train_orig[idx]
    else:
        X_cv = X_train; y_cv_log = y_train_log; y_cv_orig = y_train_orig

    models = get_models(X_train.shape[1])

    results: list[dict] = []
    trained_models: dict = {}

    for name, model in models.items():
        log.info("--- %s ---", name.upper())
        t0 = time.time()

        # Cross-validation
        cv_metrics = cross_validate(model, X_cv, y_cv_log, y_cv_orig, name)
        log.info("  CV  RMSPE=%.5f±%.5f  R²=%.4f",
                 cv_metrics["rmspe_mean"], cv_metrics["rmspe_std"],
                 cv_metrics["r2_mean"])

        # Refit on full train
        model.fit(X_train, y_train_log)

        # Val metrics
        val_preds = model.predict(X_val)
        val_m = evaluate(y_val_log, val_preds, y_val_orig)
        log.info("  Val RMSPE=%.5f  RMSE=%.5f  R²=%.4f  (%.1fs)",
                 val_m["rmspe"], val_m["rmse"], val_m["r2"], time.time() - t0)

        plot_pred_vs_actual(y_val_log, val_preds, name)

        results.append({
            "model":         name,
            "cv_rmspe_mean": cv_metrics["rmspe_mean"],
            "cv_rmspe_std":  cv_metrics["rmspe_std"],
            "cv_r2_mean":    cv_metrics["r2_mean"],
            **{f"rmspe_mean": cv_metrics["rmspe_mean"],
               f"rmse_mean":  cv_metrics["rmse_mean"],
               f"mae_mean":   cv_metrics["mae_mean"],
               f"r2_mean":    cv_metrics["r2_mean"]},
            "val_rmspe":     val_m["rmspe"],
            "val_rmse":      val_m["rmse"],
            "val_r2":        val_m["r2"],
        })
        trained_models[name] = model

        pkl_path = MODEL_DIR / f"{name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"model": model, "fe_cols": fe_cols}, f)
        log.info("  Saved %s.pkl", name)

    # ── select champion ────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    log.info("Saved model_comparison.csv")
    plot_model_comparison(results_df)

    champion_row = results_df.sort_values("cv_rmspe_mean").iloc[0]
    champion_name = champion_row["model"]
    log.info("Champion (lowest CV RMSPE): %s  (RMSPE=%.5f)",
             champion_name, champion_row["cv_rmspe_mean"])

    # Save champion
    champion_model = trained_models[champion_name]
    with open(MODEL_DIR / "champion.pkl", "wb") as f:
        pickle.dump({"model": champion_model, "fe_cols": fe_cols,
                     "champion_name": champion_name}, f)
    log.info("Saved champion.pkl -> %s", champion_name)

    log.info("=" * 60)
    log.info("Step 4 complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
