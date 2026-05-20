"""
use_case_F_esg/05_hyperparameter_tuning.py
===========================================
DSF504 — Use Case F: ESG & Greenwashing Risk Scoring
Step 5: Hyperparameter Tuning (Optuna — XGBoost)

Search space
------------
  max_depth        : 3–10
  learning_rate    : 1e-3 – 0.3 (log-uniform)
  n_estimators     : 100–500
  subsample        : 0.5–1.0
  colsample_bytree : 0.5–1.0
  min_child_weight : 1–10
  gamma            : 0.0–5.0
  reg_alpha        : 1e-4–10.0 (L1)
  reg_lambda       : 1e-4–10.0 (L2)

Objective: maximise macro-F1 on stratified 3-fold CV (fast)
Outputs
-------
  models/use_case_F/final_model.pkl          — tuned XGBoost champion
  reports/use_case_F/tuning_results.csv      — all trial results
  reports/use_case_F/tuning_convergence.png  — best score vs trial
  reports/use_case_F/hp_importance.png       — fANOVA parameter importance

NOTE: Run from terminal for long tuning:
  python use_case_F_esg/05_hyperparameter_tuning.py
"""

from __future__ import annotations

import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE
from utils.encoding_guard import ensure_utf8

ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR    / "sec_esg"
REPORT_DIR  = REPORTS_DIR / "use_case_F"
MODEL_DIR   = MODELS_DIR  / "use_case_F"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "greenwashing_risk"
RISK_MAP = {"Low": 0, "Medium": 1, "High": 2}
SEED     = RANDOM_STATE
N_TRIALS = 8    # reduced for sandbox; increase to 40+ for full tuning


def _load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(DATA_SUBDIR / f"{name}_fe.parquet")
    drop = [TARGET, "company_id", "env_claim_label"]
    X = df.drop(columns=[c for c in drop if c in df.columns]).values.astype(np.float32)
    y = df[TARGET].map(RISK_MAP).values
    return X, y


def main() -> None:
    log.info("Step 5 — ESG & Greenwashing: Hyperparameter Tuning")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.error("optuna not installed. Run: pip install optuna --break-system-packages")
        sys.exit(1)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        log.error("xgboost not installed. Run: pip install xgboost --break-system-packages")
        sys.exit(1)

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import f1_score

    X_train, y_train = _load_split("train")
    X_val,   y_val   = _load_split("val")
    X_test,  y_test  = _load_split("test")
    log.info("Loaded splits — Train: %s, Val: %s, Test: %s",
             X_train.shape, X_val.shape, X_test.shape)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    trial_records: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth":          trial.suggest_int("max_depth", 3, 10),
            "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators":       trial.suggest_int("n_estimators", 80, 200),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "objective":          "multi:softmax",
            "num_class":          3,
            "eval_metric":        "mlogloss",
            "use_label_encoder":  False,
            "random_state":       SEED,
            "n_jobs":             1,
            "verbosity":          0,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=skf, scoring="f1_macro", n_jobs=1)
        score = scores.mean()
        trial_records.append({**trial.params, "cv_f1_macro": round(score, 5),
                               "trial": trial.number})
        return score

    log.info("Running Optuna — %d trials...", N_TRIALS)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    log.info("Best trial: %d — CV F1 = %.4f", study.best_trial.number, study.best_value)
    log.info("Best params: %s", study.best_params)

    # ── Retrain on full train set with best params ─────────────────────────────
    best_params = {**study.best_params,
                   "objective": "multi:softmax", "num_class": 3,
                   "eval_metric": "mlogloss", "use_label_encoder": False,
                   "random_state": SEED, "n_jobs": 1, "verbosity": 0}
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    val_f1  = f1_score(y_val,  final_model.predict(X_val),  average="macro")
    test_f1 = f1_score(y_test, final_model.predict(X_test), average="macro")
    val_acc  = accuracy_score(y_val,  final_model.predict(X_val))
    test_acc = accuracy_score(y_test, final_model.predict(X_test))
    log.info("Final model — Val F1: %.4f | Test F1: %.4f | Val Acc: %.4f | Test Acc: %.4f",
             val_f1, test_f1, val_acc, test_acc)

    with open(MODEL_DIR / "final_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    log.info("Saved final_model.pkl")

    # ── Save trial results ─────────────────────────────────────────────────────
    df_trials = pd.DataFrame(trial_records).sort_values("cv_f1_macro", ascending=False)
    df_trials.to_csv(REPORT_DIR / "tuning_results.csv", index=False)
    log.info("Saved tuning_results.csv (%d trials)", len(df_trials))

    # ── Convergence plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    trial_order = df_trials.sort_values("trial")
    best_so_far = trial_order["cv_f1_macro"].cummax()
    axes[0].plot(trial_order["trial"], trial_order["cv_f1_macro"],
                 "o", alpha=0.4, markersize=4, color="#42A5F5", label="Trial score")
    axes[0].plot(trial_order["trial"], best_so_far,
                 "-", color="#FFA726", linewidth=2, label="Best so far")
    axes[0].set_title("Optuna Convergence — XGBoost Macro-F1", fontweight="bold")
    axes[0].set_xlabel("Trial #"); axes[0].set_ylabel("CV Macro-F1")
    axes[0].legend()

    # HP importance (fANOVA approximation via correlation with objective)
    hp_cols = [c for c in df_trials.columns if c not in ["cv_f1_macro", "trial"]]
    hp_corr = {}
    for col in hp_cols:
        try:
            corr = abs(df_trials[col].corr(df_trials["cv_f1_macro"]))
            hp_corr[col] = corr if not np.isnan(corr) else 0.0
        except Exception:
            hp_corr[col] = 0.0
    hp_series = pd.Series(hp_corr).sort_values(ascending=True)
    axes[1].barh(hp_series.index, hp_series.values, color="#AB47BC", edgecolor="white")
    axes[1].set_title("Hyperparameter Importance\n(|correlation with CV F1|)", fontweight="bold")
    axes[1].set_xlabel("Absolute Correlation")

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "tuning_convergence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved tuning_convergence.png")

    log.info("Step 5 complete ✓")


if __name__ == "__main__":
    main()
