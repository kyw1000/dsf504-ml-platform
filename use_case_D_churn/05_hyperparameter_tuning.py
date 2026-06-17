"""
use_case_D_churn/05_hyperparameter_tuning.py
==============================================
DSF504 — Use Case D: Customer Churn Prediction (KKBox)
Step 5: Hyperparameter Tuning

Objectives
----------
1. GridSearchCV        — Random Forest (fast exhaustive on small grid)
2. RandomizedSearchCV  — XGBoost (efficient random sampling)
3. Optuna Bayesian     — LightGBM (surrogate-model TPE, expected champion)
4. Tuned vs untuned comparison
5. Optimal threshold calibration (F1-maximising & Youden's J)
6. SHAP global feature importance for final champion
7. Persist best tuned model → models/use_case_D/

Why Bayesian optimisation for churn
-------------------------------------
- KKBox train set is ~720K rows; gradient boosting space is high-dimensional.
- Optuna TPE sampler converges in ~50 trials vs. thousands for grid search.
- ROC-AUC is the primary objective (industry standard for churn scorecards).
- SMOTE applied inside CV folds to prevent leakage.

Academic references
-------------------
- Verbeke et al. (2012): ensemble methods outperform LR for churn prediction
- Burez & Van den Poel (2009): SMOTE + gradient boosting for imbalanced churn
"""

from __future__ import annotations

import sys
import time
import logging
import warnings
import joblib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
    roc_curve, confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import lightgbm as lgb

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    MODELS_DIR, REPORTS_DIR, DATA_DIR,
    RANDOM_STATE, CV_FOLDS, TUNING_TRIALS,
)

from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR  = MODELS_DIR  / "use_case_D"
REPORT_DIR = REPORTS_DIR / "use_case_D"
DATA_DIR_D = DATA_DIR    / "kkbox_churn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "is_churn"
CHURN_RATE   = 0.084
EXCLUDE_COLS = {TARGET_COL}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data():
    df_train = pd.read_parquet(DATA_DIR_D / "train_fe.parquet")
    df_val   = pd.read_parquet(DATA_DIR_D / "val_fe.parquet")
    feat_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns
                 if c not in EXCLUDE_COLS]
    X_train = df_train[feat_cols].fillna(0).values.astype(np.float32)
    y_train = df_train[TARGET_COL].values
    X_val   = df_val[feat_cols].fillna(0).values.astype(np.float32)
    y_val   = df_val[TARGET_COL].values
    return X_train, y_train, X_val, y_val, feat_cols


def calibrate_threshold(model, X_val, y_val) -> float:
    proba = model.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_f1_thr = float(thr[np.argmax(f1s[:-1])])
    fpr, tpr, thresholds = roc_curve(y_val, proba)
    youden_thr = float(thresholds[np.argmax(tpr - fpr)])
    log.info("  Threshold — F1-max: %.3f  Youden's J: %.3f", best_f1_thr, youden_thr)
    return youden_thr


def score_model(model, X_val, y_val, thr: float) -> dict:
    proba = model.predict_proba(X_val)[:, 1]
    pred  = (proba >= thr).astype(int)
    return {
        "roc_auc":   roc_auc_score(y_val, proba),
        "pr_auc":    average_precision_score(y_val, proba),
        "f1":        f1_score(y_val, pred, zero_division=0),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall":    recall_score(y_val, pred, zero_division=0),
        "threshold": thr,
    }


# ── 1. GridSearchCV — Random Forest ───────────────────────────────────────────

def tune_rf(X_train, y_train) -> object:
    log.info("GridSearchCV — Random Forest…")
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth":    [8, 12],
        "clf__min_samples_leaf": [10, 30],
    }
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf",   RandomForestClassifier(class_weight="balanced", n_jobs=1,
                                         random_state=RANDOM_STATE)),
    ])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs  = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=skf, n_jobs=1, verbose=0)
    gs.fit(X_train, y_train)
    log.info("  Best RF params: %s  ROC-AUC=%.4f", gs.best_params_, gs.best_score_)
    return gs.best_estimator_


# ── 2. RandomizedSearchCV — XGBoost ──────────────────────────────────────────

def tune_xgb(X_train, y_train) -> Optional[object]:
    if not XGB_AVAILABLE:
        log.warning("XGBoost not available — skipping RandomizedSearchCV")
        return None
    log.info("RandomizedSearchCV — XGBoost…")
    pos_weight = (1 - CHURN_RATE) / CHURN_RATE
    param_dist = {
        "clf__n_estimators":    [200, 400, 600],
        "clf__max_depth":       [4, 6, 8],
        "clf__learning_rate":   [0.01, 0.05, 0.1],
        "clf__subsample":       [0.7, 0.85, 1.0],
        "clf__colsample_bytree":[0.7, 0.85, 1.0],
        "clf__min_child_weight":[1, 5, 10],
    }
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf",   xgb.XGBClassifier(
            scale_pos_weight=pos_weight, eval_metric="auc",
            use_label_encoder=False, n_jobs=1,
            random_state=RANDOM_STATE, verbosity=0)),
    ])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rs  = RandomizedSearchCV(pipe, param_dist, n_iter=20, scoring="roc_auc",
                              cv=skf, n_jobs=1, random_state=RANDOM_STATE, verbose=0)
    rs.fit(X_train, y_train)
    log.info("  Best XGB params: %s  ROC-AUC=%.4f", rs.best_params_, rs.best_score_)
    return rs.best_estimator_


# ── 3. Optuna Bayesian — LightGBM ─────────────────────────────────────────────

def tune_lgbm_optuna(X_train, y_train, n_trials: int = TUNING_TRIALS) -> object:
    log.info("Optuna Bayesian — LightGBM (%d trials)…", n_trials)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 200, 800),
            "max_depth":          trial.suggest_int("max_depth", 4, 10),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 100),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }
        aucs = []
        for tr_idx, vl_idx in skf.split(X_train, y_train):
            Xtr, Xvl = X_train[tr_idx], X_train[vl_idx]
            ytr, yvl = y_train[tr_idx], y_train[vl_idx]

            sm    = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
            Xtr_s, ytr_s = sm.fit_resample(Xtr, ytr)

            model = lgb.LGBMClassifier(
                **params, is_unbalance=False,
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1
            )
            model.fit(Xtr_s, ytr_s)
            proba = model.predict_proba(Xvl)[:, 1]
            aucs.append(roc_auc_score(yvl, proba))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    log.info("  Optuna best ROC-AUC=%.4f  params=%s", study.best_value, best)

    # Retrain on full train set with best params
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    Xs, ys = sm.fit_resample(X_train, y_train)
    champion = lgb.LGBMClassifier(
        **best, is_unbalance=False,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1
    )
    champion.fit(Xs, ys)

    # Save study results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(REPORT_DIR / "optuna_trials.csv", index=False)
    log.info("  Optuna trials saved")

    return champion


# ── 4. Plot tuning results ─────────────────────────────────────────────────────

def plot_comparison(records: list[dict]) -> None:
    df = pd.DataFrame(records).sort_values("roc_auc")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Tuned Model Comparison — KKBox Churn", fontsize=13, fontweight="bold")

    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1 Score")]
    colors  = ["#78909C"] * (len(df) - 1) + ["#3949AB"]

    for ax, (col, title) in zip(axes, metrics):
        df_s = df.sort_values(col)
        ax.barh(df_s["model"], df_s[col], color=colors, alpha=0.88)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))

    plt.tight_layout()
    path = REPORT_DIR / "tuning_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


def plot_shap(model, X_sample: np.ndarray, feat_cols: list[str]) -> None:
    if not SHAP_AVAILABLE:
        log.warning("shap not installed — skipping SHAP plot")
        return
    try:
        explainer = shap.TreeExplainer(model)
        sv        = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs  = np.abs(sv).mean(axis=0)
        shap_df   = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs})
        shap_df   = shap_df.sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        top = shap_df.head(20)
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#3949AB", alpha=0.85)
        ax.set_title("SHAP Feature Importance — Top 20 (KKBox Churn)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        path = REPORT_DIR / "shap_feature_importance.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        log.info("SHAP plot saved → %s", path.name)
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)


def plot_optuna_history() -> None:
    trial_csv = REPORT_DIR / "optuna_trials.csv"
    if not trial_csv.exists():
        return
    df = pd.read_csv(trial_csv)
    if "value" not in df.columns:
        return
    df["best_so_far"] = df["value"].cummax()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df.index, df["value"], alpha=0.4, s=12, color="#78909C", label="Trial")
    ax.plot(df.index, df["best_so_far"], color="#3949AB", lw=2, label="Best so far")
    ax.set_title("Optuna Optimization History — LightGBM (KKBox Churn)", fontsize=11)
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV ROC-AUC")
    ax.legend()
    plt.tight_layout()
    path = REPORT_DIR / "optuna_history.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case D: KKBox Churn — Hyperparameter Tuning")
    print("=" * 65 + "\n")

    X_train, y_train, X_val, y_val, feat_cols = load_data()
    log.info("Loaded | train=%s val=%s features=%d", X_train.shape, X_val.shape, len(feat_cols))

    records = []

    print("[1] GridSearchCV — Random Forest…")
    rf_tuned = tune_rf(X_train, y_train)
    rf_thr   = calibrate_threshold(rf_tuned, X_val, y_val)
    rf_sc    = score_model(rf_tuned, X_val, y_val, rf_thr)
    rf_sc["model"] = "Random Forest (GridCV)"
    records.append(rf_sc)
    log.info("RF tuned → ROC-AUC=%.4f", rf_sc["roc_auc"])

    if XGB_AVAILABLE:
        print("[2] RandomizedSearchCV — XGBoost…")
        xgb_tuned = tune_xgb(X_train, y_train)
        if xgb_tuned:
            xgb_thr  = calibrate_threshold(xgb_tuned, X_val, y_val)
            xgb_sc   = score_model(xgb_tuned, X_val, y_val, xgb_thr)
            xgb_sc["model"] = "XGBoost (RandomCV)"
            records.append(xgb_sc)
            log.info("XGB tuned → ROC-AUC=%.4f", xgb_sc["roc_auc"])

    print(f"[3] Optuna Bayesian — LightGBM ({TUNING_TRIALS} trials)…")
    lgbm_tuned = tune_lgbm_optuna(X_train, y_train, n_trials=TUNING_TRIALS)
    lgbm_thr   = calibrate_threshold(lgbm_tuned, X_val, y_val)
    lgbm_sc    = score_model(lgbm_tuned, X_val, y_val, lgbm_thr)
    lgbm_sc["model"] = "LightGBM (Optuna)"
    records.append(lgbm_sc)
    log.info("LGBM tuned → ROC-AUC=%.4f", lgbm_sc["roc_auc"])

    # Champion = best ROC-AUC among all tuned models
    best_rec = max(records, key=lambda r: r["roc_auc"])
    log.info("Overall champion: %s  ROC-AUC=%.4f", best_rec["model"], best_rec["roc_auc"])

    print("[4] Plotting comparison…")
    plot_comparison(records)
    plot_optuna_history()

    # SHAP on LightGBM champion
    print("[5] SHAP feature importance…")
    rng      = np.random.default_rng(RANDOM_STATE)
    idx      = rng.choice(len(X_val), size=min(2000, len(X_val)), replace=False)
    plot_shap(lgbm_tuned, X_val[idx], feat_cols)

    # Persist champion
    joblib.dump(lgbm_tuned, MODEL_DIR / "lgbm_optuna_champion.pkl")
    joblib.dump(feat_cols,  MODEL_DIR / "feature_cols.pkl")
    pd.DataFrame(records).to_csv(REPORT_DIR / "tuned_model_comparison.csv", index=False)
    pd.DataFrame([lgbm_sc]).to_csv(REPORT_DIR / "champion_val_metrics.csv", index=False)
    log.info("Champion saved → models/use_case_D/lgbm_optuna_champion.pkl")

    print("\n" + "=" * 65)
    print(f"  Step 5 complete. Champion: LightGBM (Optuna)")
    print(f"  Val ROC-AUC: {lgbm_sc['roc_auc']:.4f}  "
          f"PR-AUC: {lgbm_sc['pr_auc']:.4f}  "
          f"F1: {lgbm_sc['f1']:.4f}")
    print("  Ready for ethics review (06_ethics_explainability.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
