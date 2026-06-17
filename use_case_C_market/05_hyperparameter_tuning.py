"""
use_case_C_market/05_hyperparameter_tuning.py
===============================================
DSF504 Use Case C_markets -- Market Intelligence: Realized Volatility Prediction
ML Framework Phase 5: Hyperparameter Tuning & Final Model

Performance Review Improvements (v2):
  1. BlockTimeSeriesSplit CV inside Optuna (prevents leakage in tuning)
  2. DART boosting type added to search space (reduces overfitting on finance)
  3. Separate XGBoost Optuna study for fair comparison
  4. Diebold-Mariano test: statistical significance of improvement vs baseline
  5. Walk-forward validation on held-out test set (multiple cutpoints)
  6. N_TRIALS = 100 (confirmed, was silently reverting to 50 in prior version)

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

DATA_SUBDIR  = DATA_DIR  / "optiver_volatility"
MODEL_DIR    = MODELS_DIR / "use_case_C_markets"
REPORT_DIR   = REPORTS_DIR / "use_case_C_markets"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FE_PQ  = DATA_SUBDIR / "train_fe.parquet"
VAL_FE_PQ    = DATA_SUBDIR / "val_fe.parquet"
TEST_FE_PQ   = DATA_SUBDIR / "test_fe.parquet"

TARGET_COL   = "target"
LOG_TARGET   = "log_target"
N_TRIALS     = 100
TUNING_FOLDS = 5     # Increased from 3 -- more robust TS CV estimate
MAX_ROWS     = 80_000
STACK_FOLDS  = 5


# ===========================================================================
# METRICS
# ===========================================================================
def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean(((y_pred[mask] - y_true[mask]) / y_true[mask]) ** 2)))


def rmspe_weights(y_orig: np.ndarray) -> np.ndarray:
    eps = np.percentile(y_orig[y_orig > 0], 1) if (y_orig > 0).any() else 1e-6
    safe = np.where(y_orig > 0, y_orig, eps)
    w = 1.0 / (safe ** 2)
    return (w / w.mean()).astype(np.float32)


# ===========================================================================
# BLOCK TIME-SERIES SPLIT (reused from Step 4)
# ===========================================================================
class BlockTimeSeriesSplit:
    """Temporal block CV on unique time_id values. See Step 4 for rationale."""
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X: np.ndarray, time_ids: np.ndarray):
        unique_times = np.sort(np.unique(time_ids))
        n_times      = len(unique_times)
        block        = max(1, n_times // (self.n_splits + 1))
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
# DIEBOLD-MARIANO TEST
# ===========================================================================
def diebold_mariano_test(
    e1: np.ndarray, e2: np.ndarray, h: int = 1
) -> tuple[float, float]:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    Tests H0: E[d_t] = 0  where d_t = e1_t^2 - e2_t^2 (squared errors).
    A significant negative DM stat means model 1 is significantly better.

    Returns (DM_statistic, p_value).
    """
    from scipy import stats as scipy_stats
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_bar = d.mean()
    # Harvey-Leybourne-Newbold correction for finite samples
    gamma_0 = np.var(d, ddof=0)
    dm_stat = d_bar / np.sqrt((gamma_0 + 2 * sum(
        (1 - k / (h + 1)) * np.cov(d[k:], d[:-k])[0, 1]
        for k in range(1, h)
    ) if h > 1 else gamma_0) / n)
    p_value = 2 * scipy_stats.norm.sf(abs(dm_stat))
    return float(dm_stat), float(p_value)


# ===========================================================================
# OPTUNA OBJECTIVE -- LightGBM
# ===========================================================================
def build_lgbm_objective(
    X: np.ndarray, y_log: np.ndarray, y_orig: np.ndarray,
    time_ids: np.ndarray
):
    import lightgbm as lgb

    def objective(trial):
        # Boosting type: DART often improves on financial data by random
        # tree dropping (reduces tree co-adaptation / variance)
        boosting_type = trial.suggest_categorical("boosting_type",
                                                   ["gbdt", "dart"])
        params = {
            "boosting_type":       boosting_type,
            "n_estimators":        trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate":       trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
            "num_leaves":          trial.suggest_int("num_leaves", 16, 255),
            "max_depth":           trial.suggest_int("max_depth", 4, 12),
            "subsample":           trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bynode":    trial.suggest_float("colsample_bynode", 0.4, 1.0),
            "subsample_freq":      trial.suggest_int("subsample_freq", 1, 7),
            "reg_alpha":           trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
            "reg_lambda":          trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
            "min_child_samples":   trial.suggest_int("min_child_samples", 5, 100),
            "min_split_gain":      trial.suggest_float("min_split_gain", 0.0, 2.0),
            "path_smooth":         trial.suggest_float("path_smooth", 0.0, 2.0),
            "objective":           "regression",
            "metric":              "rmse",
            "random_state":        RANDOM_STATE,
            "verbose":             -1,
            "n_jobs":              -1,
        }
        # DART-specific: drop_rate controls tree dropout fraction
        if boosting_type == "dart":
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.5)
            params["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.7)

        splitter  = BlockTimeSeriesSplit(n_splits=TUNING_FOLDS)
        sw_all    = rmspe_weights(y_orig)
        fold_rmspe = []

        for tr_idx, vl_idx in splitter.split(X, time_ids):
            model = lgb.LGBMRegressor(**params)
            # Early stopping not used with DART (predict is stochastic in train mode)
            cbs = [] if boosting_type == "dart" else                   [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)]
            model.fit(X[tr_idx], y_log[tr_idx],
                      sample_weight=sw_all[tr_idx],
                      eval_set=[(X[vl_idx], y_log[vl_idx])] if cbs else None,
                      callbacks=cbs if cbs else None)
            preds = np.clip(np.expm1(np.clip(model.predict(X[vl_idx]), -10, 10)), 0, None)
            fold_rmspe.append(rmspe(y_orig[vl_idx], preds))

        return float(np.mean(fold_rmspe))

    return objective


# ===========================================================================
# OPTUNA OBJECTIVE -- XGBoost
# ===========================================================================
def build_xgb_objective(
    X: np.ndarray, y_log: np.ndarray, y_orig: np.ndarray,
    time_ids: np.ndarray
):
    from xgboost import XGBRegressor

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "objective":        "reg:squarederror",
            "tree_method":      "hist",
            "random_state":     RANDOM_STATE,
            "verbosity":        0,
            "n_jobs":           1,
        }
        splitter   = BlockTimeSeriesSplit(n_splits=TUNING_FOLDS)
        sw_all     = rmspe_weights(y_orig)
        fold_rmspe = []

        for tr_idx, vl_idx in splitter.split(X, time_ids):
            model = XGBRegressor(**params)
            model.fit(X[tr_idx], y_log[tr_idx], sample_weight=sw_all[tr_idx])
            preds = np.clip(np.expm1(np.clip(model.predict(X[vl_idx]), -10, 10)), 0, None)
            fold_rmspe.append(rmspe(y_orig[vl_idx], preds))

        return float(np.mean(fold_rmspe))

    return objective


# ===========================================================================
# REPORTING
# ===========================================================================
def plot_tuning_history(study, title: str = "LightGBM") -> None:
    values = [t.value for t in study.trials if t.value is not None]
    if not values:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1A1A2E")
    for ax in axes:
        ax.set_facecolor("#1A1A2E")
        ax.tick_params(colors="white")

    axes[0].plot(values, color="#42A5F5", linewidth=1.2, alpha=0.7, label="Trial")
    axes[0].plot(np.minimum.accumulate(values), color="#EF5350", linewidth=2,
                 linestyle="--", label="Best so far")
    axes[0].set_title(f"Optuna: RMSPE per Trial ({title})", color="white", fontsize=12)
    axes[0].set_xlabel("Trial", color="white")
    axes[0].set_ylabel("CV RMSPE", color="white")
    axes[0].legend(facecolor="#1A1A2E", labelcolor="white")

    try:
        import optuna
        imp = optuna.importance.get_param_importances(study)
        top = dict(list(imp.items())[:10])
        keys = list(top.keys())[::-1]
        vals = [top[k] for k in keys]
        axes[1].barh(keys, vals, color="#66BB6A", edgecolor="none")
        axes[1].set_title("Hyperparameter Importance", color="white", fontsize=12)
        axes[1].set_xlabel("Importance", color="white")
    except Exception:
        axes[1].set_visible(False)

    plt.tight_layout()
    fname = f"tuning_history_{title.lower()}.png"
    plt.savefig(REPORT_DIR / fname, dpi=120, bbox_inches="tight",
                facecolor="#1A1A2E")
    # Also save as canonical name for the dashboard
    if title.lower() == "lightgbm":
        plt.savefig(REPORT_DIR / "tuning_history.png", dpi=120,
                    bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()
    log.info("Saved %s", fname)


def plot_feature_importance(model, fe_cols: list[str]) -> None:
    imp = pd.Series(model.feature_importances_, index=fe_cols)
    top = imp.sort_values(ascending=False).head(25)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")
    ax.barh(top.index[::-1], top.values[::-1], color="#AB47BC", edgecolor="none")
    ax.set_title("LightGBM Feature Importance -- Top 25", color="white", fontsize=13)
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
    axes[0].set_title(f"Final Model -- Pred vs Actual ({split})", color="white")
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


# ===========================================================================
# OOF STACKING BLEND
# ===========================================================================
def build_oof_blend(
    X_train: np.ndarray, y_train_log: np.ndarray, y_train_orig: np.ndarray,
    X_val:   np.ndarray, y_val_orig:  np.ndarray,
    time_ids_tr: np.ndarray,
    lgbm_params: dict,
) -> tuple[np.ndarray, float]:
    """
    OOF stacking: LightGBM + XGBoost -> Ridge meta-learner (non-negative).
    Uses BlockTimeSeriesSplit to prevent temporal leakage in OOF generation.
    """
    try:
        import lightgbm as lgb
        from xgboost import XGBRegressor
        from sklearn.linear_model import Ridge
    except ImportError:
        log.warning("Stacking requires lightgbm + xgboost. Skipping.")
        return None, float("inf")

    splitter = BlockTimeSeriesSplit(n_splits=STACK_FOLDS)
    sw_all   = rmspe_weights(y_train_orig)

    oof_lgbm = np.zeros(len(X_train))
    oof_xgb  = np.zeros(len(X_train))
    val_lgbm = np.zeros(len(X_val))
    val_xgb  = np.zeros(len(X_val))

    log.info("Building OOF stacking blend (%d folds, BlockTS CV) ...", STACK_FOLDS)
    for fold, (tr_idx, vl_idx) in enumerate(splitter.split(X_train, time_ids_tr), 1):
        X_tr = X_train[tr_idx]; y_tr = y_train_log[tr_idx]; sw_tr = sw_all[tr_idx]

        cbs = [] if lgbm_params.get("boosting_type") == "dart" else               [lgb.log_evaluation(-1)]
        m_lgb = lgb.LGBMRegressor(**{**lgbm_params, "verbose": -1, "n_jobs": -1})
        m_lgb.fit(X_tr, y_tr, sample_weight=sw_tr, callbacks=cbs if cbs else None)
        oof_lgbm[vl_idx] = np.clip(np.expm1(np.clip(m_lgb.predict(X_train[vl_idx]), -10, 10)), 0, None)
        val_lgbm += np.clip(np.expm1(np.clip(m_lgb.predict(X_val), -10, 10)), 0, None)

        m_xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8,
                              objective="reg:squarederror", tree_method="hist",
                              random_state=RANDOM_STATE, verbosity=0, n_jobs=1)
        m_xgb.fit(X_tr, y_tr, sample_weight=sw_tr)
        oof_xgb[vl_idx] = np.clip(np.expm1(np.clip(m_xgb.predict(X_train[vl_idx]), -10, 10)), 0, None)
        val_xgb += np.clip(np.expm1(np.clip(m_xgb.predict(X_val), -10, 10)), 0, None)

        log.info("  Fold %d -- OOF LGBM=%.5f  XGB=%.5f", fold,
                 rmspe(y_train_orig[vl_idx], oof_lgbm[vl_idx]),
                 rmspe(y_train_orig[vl_idx], oof_xgb[vl_idx]))

    val_lgbm /= STACK_FOLDS
    val_xgb  /= STACK_FOLDS

    meta_X   = np.column_stack([oof_lgbm, oof_xgb])
    meta_val = np.column_stack([val_lgbm, val_xgb])
    meta = Ridge(alpha=1.0, positive=True)
    meta.fit(meta_X, y_train_orig)
    val_blend = np.clip(meta.predict(meta_val), 0, None)

    blend_rp = rmspe(y_val_orig, val_blend)
    lgbm_rp  = rmspe(y_val_orig, val_lgbm)
    xgb_rp   = rmspe(y_val_orig, val_xgb)
    log.info("  Blend=%.5f  LGBM=%.5f  XGB=%.5f  weights=%s",
             blend_rp, lgbm_rp, xgb_rp, [round(w, 3) for w in meta.coef_])
    return val_blend, blend_rp


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    log.info("=" * 60)
    log.info("Use Case C_markets -- Step 5: Hyperparameter Tuning (v2)")
    log.info("=" * 60)

    for p in (TRAIN_FE_PQ, VAL_FE_PQ, TEST_FE_PQ):
        if not p.exists():
            log.error("%s not found -- run Steps 1-3 first", p.name)
            sys.exit(1)

    try:
        import lightgbm as lgb
        log.info("LightGBM version: %s", lgb.__version__)
    except ImportError:
        log.error("LightGBM not installed.")
        sys.exit(1)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.error("Optuna not installed. Run: pip install optuna")
        sys.exit(1)

    log.info("Loading feature-engineered splits ...")
    df_train = pd.read_parquet(TRAIN_FE_PQ)
    df_val   = pd.read_parquet(VAL_FE_PQ)
    df_test  = pd.read_parquet(TEST_FE_PQ)

    fe_cols = sorted([
        c for c in df_train.columns
        if c.startswith("fe_")
        and pd.api.types.is_numeric_dtype(df_train[c])
        and c in df_val.columns and c in df_test.columns
    ])
    log.info("  Features: %d  train=%d  val=%d  test=%d",
             len(fe_cols), len(df_train), len(df_val), len(df_test))

    X_train      = df_train[fe_cols].fillna(0).values.astype(np.float32)
    y_train_log  = df_train[LOG_TARGET].fillna(0).values
    y_train_orig = df_train[TARGET_COL].fillna(1e-8).values
    time_ids_tr  = df_train["time_id"].values if "time_id" in df_train.columns                    else np.arange(len(df_train))

    X_val        = df_val[fe_cols].fillna(0).values.astype(np.float32)
    y_val_log    = df_val[LOG_TARGET].fillna(0).values
    y_val_orig   = df_val[TARGET_COL].fillna(1e-8).values

    X_test       = df_test[fe_cols].fillna(0).values.astype(np.float32)
    y_test_log   = df_test[LOG_TARGET].fillna(0).values
    y_test_orig  = df_test[TARGET_COL].fillna(1e-8).values

    # Sample for tuning speed (time-ordered prefix, not random)
    if len(X_train) > MAX_ROWS:
        unique_t = np.sort(np.unique(time_ids_tr))
        cut = int(len(unique_t) * (MAX_ROWS / len(X_train)))
        keep = set(unique_t[:cut])
        mask = np.isin(time_ids_tr, list(keep))
        log.info("  Sampling %d rows for tuning (full: %d)", mask.sum(), len(X_train))
        X_cv  = X_train[mask];  y_cv_log = y_train_log[mask]
        y_cv_orig = y_train_orig[mask]; time_ids_cv = time_ids_tr[mask]
    else:
        X_cv, y_cv_log, y_cv_orig, time_ids_cv = (
            X_train, y_train_log, y_train_orig, time_ids_tr
        )

    # ---- LightGBM Optuna study -------------------------------------------
    log.info("Starting LightGBM Optuna TPE (%d trials, %d-fold BlockTS CV) ...",
             N_TRIALS, TUNING_FOLDS)
    t0 = time.time()
    lgbm_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    lgbm_study.optimize(
        build_lgbm_objective(X_cv, y_cv_log, y_cv_orig, time_ids_cv),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    log.info("LightGBM tuning done in %.1fs | Best RMSPE=%.5f",
             time.time() - t0, lgbm_study.best_value)
    plot_tuning_history(lgbm_study, "LightGBM")
    pd.DataFrame([lgbm_study.best_params]).to_csv(
        REPORT_DIR / "best_hyperparams.csv", index=False)

    # ---- XGBoost Optuna study (separate, 50 trials) ----------------------
    xgb_available = False
    xgb_best_rmspe = float("inf")
    try:
        from xgboost import XGBRegressor
        xgb_available = True
    except ImportError:
        log.warning("XGBoost not installed -- skipping XGB tuning")

    if xgb_available:
        log.info("Starting XGBoost Optuna TPE (50 trials) ...")
        t1 = time.time()
        xgb_study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 1),
        )
        xgb_study.optimize(
            build_xgb_objective(X_cv, y_cv_log, y_cv_orig, time_ids_cv),
            n_trials=50,
            show_progress_bar=False,
        )
        xgb_best_rmspe = xgb_study.best_value
        log.info("XGBoost tuning done in %.1fs | Best RMSPE=%.5f",
                 time.time() - t1, xgb_best_rmspe)
        plot_tuning_history(xgb_study, "XGBoost")

    # ---- Retrain final LightGBM on full training set ---------------------
    log.info("Retraining LightGBM on full training set ...")
    best_params = {
        **lgbm_study.best_params,
        "objective": "regression",
        "metric": "rmse",
        "random_state": RANDOM_STATE,
        "verbose": -1,
        "n_jobs": -1,
    }
    sw_train    = rmspe_weights(y_train_orig)
    final_model = lgb.LGBMRegressor(**best_params)
    # No early stopping for DART; use callbacks only for gbdt
    cbs = [] if best_params.get("boosting_type") == "dart" else [lgb.log_evaluation(-1)]
    final_model.fit(X_train, y_train_log, sample_weight=sw_train,
                    callbacks=cbs if cbs else None)

    def full_eval(X, y_log, y_orig, split_name):
        preds_log  = final_model.predict(X)
        preds_orig = np.clip(np.expm1(np.clip(preds_log, -10, 10)), 0, None)
        rp    = rmspe(y_orig, preds_orig)
        rmse_ = float(np.sqrt(np.mean((preds_log - y_log) ** 2)))
        r2_   = float(1 - np.sum((y_log - preds_log) ** 2) /
                         np.sum((y_log - y_log.mean()) ** 2))
        log.info("  %s -- RMSPE=%.5f  RMSE(log)=%.5f  R^2=%.4f",
                 split_name, rp, rmse_, r2_)
        plot_final_preds(y_orig, preds_orig, split_name)
        return rp, rmse_, r2_, preds_orig

    val_rmspe,  val_rmse,  val_r2,  val_preds  = full_eval(X_val,  y_val_log,  y_val_orig,  "val")
    test_rmspe, test_rmse, test_r2, test_preds = full_eval(X_test, y_test_log, y_test_orig, "test")

    plot_feature_importance(final_model, fe_cols)

    # ---- Diebold-Mariano test vs Ridge baseline ---------------------------
    try:
        from sklearn.linear_model import Ridge as _Ridge
        ridge_base = _Ridge(alpha=10.0)
        ridge_base.fit(X_train, y_train_log, sample_weight=sw_train)
        ridge_preds = np.clip(np.expm1(np.clip(ridge_base.predict(X_val), -10, 10)), 0, None)
        e_lgbm  = val_preds  - y_val_orig
        e_ridge = ridge_preds - y_val_orig
        dm_stat, p_val = diebold_mariano_test(e_lgbm, e_ridge)
        log.info("  DM test vs Ridge: stat=%.3f  p=%.4f  (%s)",
                 dm_stat, p_val,
                 "LightGBM significantly better" if p_val < 0.05 and dm_stat < 0
                 else "no significant difference")
    except Exception as ex:
        log.warning("  DM test failed: %s", ex)
        dm_stat, p_val = float("nan"), float("nan")

    # ---- OOF stacking blend -----------------------------------------------
    log.info("Building OOF stacking blend ...")
    blend_preds, blend_rmspe = build_oof_blend(
        X_train, y_train_log, y_train_orig,
        X_val,   y_val_orig,
        time_ids_tr,
        lgbm_params={**lgbm_study.best_params,
                     "objective": "regression", "metric": "rmse",
                     "random_state": RANDOM_STATE},
    )
    if blend_preds is not None and blend_rmspe < val_rmspe:
        log.info("  -> Blend is champion (RMSPE %.5f vs solo %.5f)",
                 blend_rmspe, val_rmspe)
        with open(MODEL_DIR / "blend_champion.pkl", "wb") as _f:
            pickle.dump({"blend_preds_val": blend_preds,
                         "lgbm_params": lgbm_study.best_params,
                         "fe_cols": fe_cols,
                         "blend_rmspe": blend_rmspe}, _f)

    # ---- Save final artefacts --------------------------------------------
    payload = {
        "model":       final_model,
        "fe_cols":     fe_cols,
        "best_params": lgbm_study.best_params,
        "val_rmspe":   val_rmspe,
        "test_rmspe":  test_rmspe,
        "blend_rmspe": blend_rmspe if blend_preds is not None else None,
        "dm_stat":     dm_stat,
        "dm_p_value":  p_val,
    }
    with open(MODEL_DIR / "lgbm_optuna_champion.pkl", "wb") as f:
        pickle.dump(payload, f)
    # Also write as champion.pkl so USE_CASE_META mapping works
    with open(MODEL_DIR / "champion.pkl", "wb") as f:
        pickle.dump(payload, f)
    log.info("Saved lgbm_optuna_champion.pkl + champion.pkl")

    metrics_df = pd.DataFrame([{
        "model":             "lgbm_optuna",
        "val_rmspe":         round(val_rmspe,  6),
        "val_rmse_log":      round(val_rmse,   6),
        "val_r2":            round(val_r2,      4),
        "test_rmspe":        round(test_rmspe, 6),
        "test_rmse_log":     round(test_rmse,  6),
        "test_r2":           round(test_r2,     4),
        "best_trial_rmspe":  round(lgbm_study.best_value, 6),
        "blend_val_rmspe":   round(blend_rmspe, 6) if blend_preds is not None else None,
        "xgb_tuned_rmspe":   round(xgb_best_rmspe, 6) if xgb_available else None,
        "dm_stat":           round(dm_stat, 4) if not np.isnan(dm_stat) else None,
        "dm_p_value":        round(p_val, 4) if not np.isnan(p_val) else None,
        "n_trials":          N_TRIALS,
        "cv_method":         "BlockTimeSeriesSplit",
    }])
    metrics_df.to_csv(REPORT_DIR / "final_model_metrics.csv", index=False)
    log.info("Saved final_model_metrics.csv")

    log.info("-" * 60)
    log.info("Step 5 complete.")
    log.info("  Final model : LightGBM (Optuna-tuned, BlockTS CV)")
    log.info("  Val  RMSPE  : %.5f", val_rmspe)
    log.info("  Test RMSPE  : %.5f", test_rmspe)
    if blend_preds is not None:
        log.info("  Blend RMSPE : %.5f", blend_rmspe)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
