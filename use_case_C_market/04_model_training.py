"""
use_case_C_market/04_model_training.py
========================================
DSF504 Use Case C_markets -- Market Intelligence: Realized Volatility Prediction
ML Framework Phase 4: Algorithm Selection & Cross-Validation

Performance Review Improvements (v2):
  1. Time-series-aware CV: BlockTimeSeriesSplit splits by unique time_id order
     (prevents data leakage from random K-fold on financial panel data)
  2. HAR-RV OLS baseline: canonical benchmark from volatility literature
     (Corsi 2009: RV ~ lag1 + lag5_mean + lag22_mean)
  3. ElasticNet added alongside Ridge (explores L1+L2 blend)
  4. HistGradientBoostingRegressor added (faster than RandomForest, native NaN)
  5. RMSPE sample weights applied consistently across all tree-based models
  6. Expanded XGBoost config: more trees, tree_method=hist for speed

Primary metric : RMSPE = sqrt( mean( ((y_pred - y_true) / y_true)^2 ) )
Secondary       : RMSE on log scale, R^2

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
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

DATA_SUBDIR = DATA_DIR  / "optiver_volatility"
MODEL_DIR   = MODELS_DIR / "use_case_C_markets"
REPORT_DIR  = REPORTS_DIR / "use_case_C_markets"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FE_PQ = DATA_SUBDIR / "train_fe.parquet"
VAL_FE_PQ   = DATA_SUBDIR / "val_fe.parquet"
TARGET_COL  = "target"
LOG_TARGET  = "log_target"


# ===========================================================================
# METRIC HELPERS
# ===========================================================================
def rmspe_weights(y_orig: np.ndarray) -> np.ndarray:
    """
    Sample weights = 1/y^2, equivalent to RMSPE loss weighting.
    Minimising weighted MSE(w_i=1/y_i^2) == minimising RMSPE.
    Normalised so mean(w) = 1 to avoid interaction with learning rate.
    """
    eps = np.percentile(y_orig[y_orig > 0], 1) if (y_orig > 0).any() else 1e-6
    safe = np.where(y_orig > 0, y_orig, eps)
    w = 1.0 / (safe ** 2)
    return (w / w.mean()).astype(np.float32)


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    pct = (y_pred[mask] - y_true[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(pct ** 2)))


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray,
             y_true_orig: np.ndarray) -> dict:
    y_pred_orig = np.expm1(np.clip(y_pred_log, -10, 10))
    return {
        "rmspe": round(rmspe(y_true_orig, y_pred_orig), 6),
        "rmse":  round(float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))), 6),
        "mae":   round(float(mean_absolute_error(y_true_log, y_pred_log)), 6),
        "r2":    round(float(r2_score(y_true_log, y_pred_log)), 6),
    }


# ===========================================================================
# BLOCK TIME-SERIES CV
# ===========================================================================
class BlockTimeSeriesSplit:
    """
    Time-series cross-validation that respects temporal ordering.

    Splits on unique time_id values, not on rows. Each fold uses
    all data up to a cutoff time_id for training and the next block
    for validation. This prevents leakage from random shuffling.

    Why this matters for financial data:
    - Log-returns are autocorrelated at short horizons (realized vol clustering)
    - Random K-fold uses future data in training, inflating CV scores
    - Block CV gives realistic forward-looking estimates of generalisation
    """
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X: np.ndarray, time_ids: np.ndarray):
        unique_times = np.sort(np.unique(time_ids))
        n_times = len(unique_times)
        # Each test block = n_times // (n_splits + 1) time_ids
        block = max(1, n_times // (self.n_splits + 1))

        for i in range(self.n_splits):
            train_end = (i + 1) * block
            val_end   = (i + 2) * block
            train_times = set(unique_times[:train_end])
            val_times   = set(unique_times[train_end:val_end])
            tr_idx = np.where(np.isin(time_ids, list(train_times)))[0]
            vl_idx = np.where(np.isin(time_ids, list(val_times)))[0]
            if len(tr_idx) == 0 or len(vl_idx) == 0:
                continue
            yield tr_idx, vl_idx


# ===========================================================================
# HAR-RV LINEAR BASELINE
# ===========================================================================
def fit_har_rv(X_train: np.ndarray, y_train_log: np.ndarray,
               y_train_orig: np.ndarray, fe_cols: list[str],
               X_val: np.ndarray, y_val_log: np.ndarray,
               y_val_orig: np.ndarray) -> dict:
    """
    HAR-RV (Corsi 2009) linear regression on lag features.

    RV(t) ~ const + b_d*RV(lag1) + b_w*RV(lag5_mean) + b_m*RV(lag22_mean)

    This is the academic benchmark for all realized volatility models.
    A good ML model should beat HAR-RV; if it does not, feature engineering
    or model selection needs revision.
    """
    har_features = ["fe_rv_lag1", "fe_rv_lag5_mean", "fe_rv_lag22_mean"]
    avail = [c for c in har_features if c in fe_cols]
    if not avail:
        log.warning("[HAR-RV] Lag features not found -- run Step 3 (v2) first")
        return {}

    har_idx = [fe_cols.index(c) for c in avail]
    X_har_train = X_train[:, har_idx]
    X_har_val   = X_val[:,   har_idx]

    # Weighted Ridge (RMSPE weights) as the HAR estimator
    model = Ridge(alpha=1.0)
    sw = rmspe_weights(y_train_orig)
    model.fit(X_har_train, y_train_log, sample_weight=sw)

    val_preds = model.predict(X_har_val)
    val_m = evaluate(y_val_log, val_preds, y_val_orig)
    log.info("[HAR-RV] Val RMSPE=%.5f  R^2=%.4f  (features: %s)",
             val_m["rmspe"], val_m["r2"], avail)
    return {"model": model, "val_metrics": val_m, "har_features": avail}


# ===========================================================================
# MODEL REGISTRY
# ===========================================================================
def get_models() -> dict:
    models = {
        "ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "elasticnet": ElasticNet(
            alpha=0.01, l1_ratio=0.5,
            max_iter=2000, random_state=RANDOM_STATE,
        ),
        "hist_gbm": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05,
            max_leaf_nodes=63, max_depth=None,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=8,
            min_samples_leaf=10, n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=1,
        )
    except ImportError:
        log.warning("XGBoost not installed -- skipping")

    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05,
            num_leaves=127, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.01, reg_lambda=0.1,
            objective="regression",
            metric="rmse",
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        )
    except ImportError:
        log.warning("LightGBM not installed -- skipping")

    return models


# ===========================================================================
# CROSS-VALIDATION
# ===========================================================================
def cross_validate(
    model, X: np.ndarray, y_log: np.ndarray, y_orig: np.ndarray,
    time_ids: np.ndarray, name: str,
    use_rmspe_weights: bool = True,
) -> dict:
    """Time-series block CV — no random shuffling."""
    splitter = BlockTimeSeriesSplit(n_splits=CV_FOLDS)
    fold_rmspe, fold_rmse, fold_mae, fold_r2 = [], [], [], []

    for fold, (tr_idx, vl_idx) in enumerate(splitter.split(X, time_ids), 1):
        X_tr, X_vl     = X[tr_idx],      X[vl_idx]
        y_tr, y_vl     = y_log[tr_idx],  y_log[vl_idx]
        y_orig_vl      = y_orig[vl_idx]

        fit_kwargs: dict = {}
        if use_rmspe_weights:
            sw = rmspe_weights(y_orig[tr_idx])
            # Only LightGBM and XGBoost support sample_weight directly;
            # sklearn models accept it via fit(sample_weight=...)
            try:
                model.fit(X_tr, y_tr, sample_weight=sw)
            except TypeError:
                model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, y_tr)

        preds = model.predict(X_vl)
        m = evaluate(y_vl, preds, y_orig_vl)

        fold_rmspe.append(m["rmspe"])
        fold_rmse.append(m["rmse"])
        fold_mae.append(m["mae"])
        fold_r2.append(m["r2"])
        log.info("  %s fold %d — RMSPE=%.5f  R^2=%.4f",
                 name, fold, m["rmspe"], m["r2"])

    return {
        "rmspe_mean": round(float(np.mean(fold_rmspe)), 5),
        "rmspe_std":  round(float(np.std(fold_rmspe)), 5),
        "rmse_mean":  round(float(np.mean(fold_rmse)), 5),
        "mae_mean":   round(float(np.mean(fold_mae)), 5),
        "r2_mean":    round(float(np.mean(fold_r2)), 4),
    }


# ===========================================================================
# REPORTING
# ===========================================================================
def plot_model_comparison(results: pd.DataFrame) -> None:
    metrics = ["rmspe_mean", "rmse_mean", "r2_mean"]
    titles  = ["RMSPE -- lower is better", "RMSE (log) -- lower is better",
               "R^2 -- higher is better"]
    colors  = ["#EF5350", "#FFA726", "#66BB6A"]
    ascending = [True, True, False]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#1A1A2E")

    for ax, metric, title, col, asc in zip(axes, metrics, titles, colors, ascending):
        ax.set_facecolor("#1A1A2E")
        df_sorted = results.sort_values(metric, ascending=asc)
        bars = ax.barh(df_sorted["model"], df_sorted[metric],
                       color=col, edgecolor="none")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white")
        for bar, val in zip(bars, df_sorted[metric]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.5f}", va="center", color="white", fontsize=8)

    plt.suptitle("Model Comparison -- Block Time-Series CV", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png",
                dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()
    log.info("Saved model_comparison.png")


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1A1A2E")
    for ax in axes:
        ax.set_facecolor("#1A1A2E")
        ax.tick_params(colors="white")

    axes[0].scatter(y_true, y_pred, s=5, alpha=0.3, color="#42A5F5")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], color="#EF5350", linewidth=1.5)
    axes[0].set_xlabel("Actual log1p(target)", color="white")
    axes[0].set_ylabel("Predicted log1p(target)", color="white")
    axes[0].set_title(f"{model_name} -- Pred vs Actual (log)", color="white")

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


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    log.info("=" * 60)
    log.info("Use Case C_markets -- Step 4: Model Training (v2)")
    log.info("=" * 60)

    for p in (TRAIN_FE_PQ, VAL_FE_PQ):
        if not p.exists():
            log.error("%s not found -- run Step 3 first", p.name)
            sys.exit(1)

    log.info("Loading feature-engineered splits ...")
    df_train = pd.read_parquet(TRAIN_FE_PQ)
    df_val   = pd.read_parquet(VAL_FE_PQ)

    fe_cols = sorted([
        c for c in df_train.columns
        if c.startswith("fe_")
        and pd.api.types.is_numeric_dtype(df_train[c])
        and c in df_val.columns
    ])
    log.info("  Feature cols: %d", len(fe_cols))

    if not fe_cols:
        log.error("No fe_ columns -- check Step 3 output")
        sys.exit(1)

    X_train      = df_train[fe_cols].fillna(0).values.astype(np.float32)
    y_train_log  = df_train[LOG_TARGET].fillna(0).values.astype(np.float64)
    y_train_orig = df_train[TARGET_COL].fillna(1e-6).values.astype(np.float64)
    time_ids_tr  = df_train["time_id"].values if "time_id" in df_train.columns                    else np.arange(len(df_train))

    X_val        = df_val[fe_cols].fillna(0).values.astype(np.float32)
    y_val_log    = df_val[LOG_TARGET].fillna(0).values.astype(np.float64)
    y_val_orig   = df_val[TARGET_COL].fillna(1e-6).values.astype(np.float64)

    # Sample for CV speed (preserves time ordering)
    MAX_ROWS = 100_000
    if len(X_train) > MAX_ROWS:
        unique_t = np.sort(np.unique(time_ids_tr))
        cut = int(len(unique_t) * (MAX_ROWS / len(X_train)))
        keep_times = set(unique_t[:cut])
        mask = np.isin(time_ids_tr, list(keep_times))
        log.info("  Sampling %d rows for CV (full: %d)", mask.sum(), len(X_train))
        X_cv        = X_train[mask]
        y_cv_log    = y_train_log[mask]
        y_cv_orig   = y_train_orig[mask]
        time_ids_cv = time_ids_tr[mask]
    else:
        X_cv, y_cv_log, y_cv_orig, time_ids_cv = (
            X_train, y_train_log, y_train_orig, time_ids_tr
        )

    # --- HAR-RV baseline (academic benchmark)
    log.info("--- HAR-RV BASELINE ---")
    har_result = fit_har_rv(
        X_train, y_train_log, y_train_orig, fe_cols,
        X_val,   y_val_log,   y_val_orig,
    )

    models = get_models()
    results: list[dict] = []
    trained_models: dict = {}

    for name, model in models.items():
        log.info("--- %s ---", name.upper())
        t0 = time.time()

        cv_m = cross_validate(
            model, X_cv, y_cv_log, y_cv_orig, time_ids_cv, name,
            use_rmspe_weights=True,
        )
        log.info("  CV  RMSPE=%.5f+-%.5f  R^2=%.4f",
                 cv_m["rmspe_mean"], cv_m["rmspe_std"], cv_m["r2_mean"])

        # Refit on full train with RMSPE weights
        sw_full = rmspe_weights(y_train_orig)
        try:
            model.fit(X_train, y_train_log, sample_weight=sw_full)
        except TypeError:
            model.fit(X_train, y_train_log)

        val_preds = np.clip(model.predict(X_val), 0, None)
        val_m = evaluate(y_val_log, val_preds, y_val_orig)
        log.info("  Val RMSPE=%.5f  R^2=%.4f  (%.1fs)",
                 val_m["rmspe"], val_m["r2"], time.time() - t0)

        plot_pred_vs_actual(y_val_log, val_preds, name)

        results.append({
            "model":         name,
            "cv_rmspe_mean": cv_m["rmspe_mean"],
            "cv_rmspe_std":  cv_m["rmspe_std"],
            "cv_r2_mean":    cv_m["r2_mean"],
            "rmspe_mean":    cv_m["rmspe_mean"],
            "rmse_mean":     cv_m["rmse_mean"],
            "mae_mean":      cv_m["mae_mean"],
            "r2_mean":       cv_m["r2_mean"],
            "val_rmspe":     val_m["rmspe"],
            "val_rmse":      val_m["rmse"],
            "val_r2":        val_m["r2"],
        })
        trained_models[name] = model

        with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump({"model": model, "fe_cols": fe_cols}, f)
        log.info("  Saved %s.pkl", name)

    # HAR-RV as comparison row
    if har_result and "val_metrics" in har_result:
        hm = har_result["val_metrics"]
        results.append({
            "model": "har_rv_baseline",
            "cv_rmspe_mean": hm["rmspe"],
            "cv_rmspe_std":  0.0,
            "cv_r2_mean":    hm["r2"],
            "rmspe_mean":    hm["rmspe"],
            "rmse_mean":     hm["rmse"],
            "mae_mean":      hm["mae"],
            "r2_mean":       hm["r2"],
            "val_rmspe":     hm["rmspe"],
            "val_rmse":      hm["rmse"],
            "val_r2":        hm["r2"],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    log.info("Saved model_comparison.csv")
    plot_model_comparison(results_df)

    champion_row  = results_df[results_df["model"] != "har_rv_baseline"]                     .sort_values("cv_rmspe_mean").iloc[0]
    champion_name = champion_row["model"]
    log.info("Champion: %s  (CV RMSPE=%.5f)", champion_name, champion_row["cv_rmspe_mean"])

    with open(MODEL_DIR / "champion.pkl", "wb") as f:
        pickle.dump({
            "model":         trained_models[champion_name],
            "fe_cols":       fe_cols,
            "champion_name": champion_name,
        }, f)
    log.info("Saved champion.pkl -> %s", champion_name)

    log.info("=" * 60)
    log.info("Step 4 complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
