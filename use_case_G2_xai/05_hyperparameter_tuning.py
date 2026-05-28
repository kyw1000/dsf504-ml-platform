"""
use_case_G2_xai/05_hyperparameter_tuning.py
============================================
Use Case G2 — Explainable AI for Analysts & Managers
Phase 3, Step 5: Hyperparameter Tuning

Tunes LightGBM binary classifier using Optuna (TPE sampler) maximising
AUC-ROC on the held-out val set. Falls back to grid search if Optuna
is not installed.

Saves the tuned champion as:
  models/use_case_G2/lgbm_optuna_champion.pkl
  models/use_case_G2/final_model.pkl   (alias)
"""

from __future__ import annotations

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "sec_edgar"
REPORT_DIR  = REPORTS_DIR / "use_case_G2"
MODEL_DIR   = MODELS_DIR / "use_case_G2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "outperform"
DROP_COLS  = ["ticker", "fiscal_year", "sector", "forward_return_12m", LABEL_COL]


def _load():
    train = pd.read_parquet(DATA_SUBDIR / "train_fe.parquet")
    val   = pd.read_parquet(DATA_SUBDIR / "val_fe.parquet")
    feat_cols = [c for c in train.columns if c not in DROP_COLS]
    X_tr = train[feat_cols].fillna(0).values
    y_tr = train[LABEL_COL].values
    X_va = val[feat_cols].fillna(0).values
    y_va = val[LABEL_COL].values
    return X_tr, y_tr, X_va, y_va, feat_cols


def make_objective(X_tr, y_tr, X_va, y_va):
    def objective(trial):
        params = dict(
            objective         = "binary",
            n_estimators      = trial.suggest_int("n_estimators", 200, 600),
            num_leaves        = trial.suggest_int("num_leaves", 16, 128),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            class_weight      = "balanced",
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            verbose           = -1,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(20, verbose=False),
                             lgb.log_evaluation(period=-1)])
        scores = model.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, scores)
    return objective


def main() -> None:
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G2: Explainable AI for Analysts & Managers")
    print("  Step 5: Hyperparameter Tuning (LightGBM)")
    print("=" * 65 + "\n")

    X_tr, y_tr, X_va, y_va, feat_cols = _load()
    print(f"[1] Loaded: Train {X_tr.shape}  |  Val {X_va.shape}")

    best_params = None
    best_score  = 0.0
    tuner_used  = "grid"

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        objective = make_objective(X_tr, y_tr, X_va, y_va)
        study.optimize(objective, n_trials=15, timeout=40, show_progress_bar=False)
        best_params = study.best_params
        best_score  = study.best_value
        tuner_used  = "optuna"
        print(f"  Optuna: {len(study.trials)} trials | best AUC-ROC={best_score:.4f}")
        print(f"  Best params: {best_params}")

        # Trial history plot
        scores = [t.value for t in study.trials if t.value is not None]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, len(scores) + 1), scores, "o-", color="#1565C0", linewidth=1.5)
        ax.plot(range(1, len(scores) + 1),
                [max(scores[:i+1]) for i in range(len(scores))],
                "--", color="#388E3C", linewidth=1.5, label="Best so far")
        ax.set_title("Optuna Trial History — AUC-ROC (G2 XAI)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Trial")
        ax.set_ylabel("AUC-ROC (Val)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "optuna_history.png", dpi=120, bbox_inches="tight")
        plt.close()
        log.info("Saved optuna_history.png")

    except Exception as e:
        log.warning(f"Optuna unavailable ({e}). Running grid search.")
        grid = [
            {"n_estimators":300, "num_leaves":63, "learning_rate":0.05,
             "subsample":0.8, "colsample_bytree":0.8, "min_child_samples":20,
             "reg_lambda":1.0, "reg_alpha":0.01},
            {"n_estimators":400, "num_leaves":31, "learning_rate":0.03,
             "subsample":0.7, "colsample_bytree":0.7, "min_child_samples":30,
             "reg_lambda":5.0, "reg_alpha":0.1},
            {"n_estimators":200, "num_leaves":127, "learning_rate":0.08,
             "subsample":0.9, "colsample_bytree":0.9, "min_child_samples":10,
             "reg_lambda":0.1, "reg_alpha":0.001},
        ]
        for cfg in grid:
            m = lgb.LGBMClassifier(objective="binary", class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **cfg)
            m.fit(X_tr, y_tr)
            s = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
            print(f"  Grid: AUC-ROC={s:.4f}  {cfg}")
            if s > best_score:
                best_score  = s
                best_params = cfg

    # ── Train final model with best params ───────────────────────────────────
    print(f"\n[2] Training final model (tuner={tuner_used})…")
    final = lgb.LGBMClassifier(
        objective="binary", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        **best_params
    )
    final.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(period=-1)])
    final_score = roc_auc_score(y_va, final.predict_proba(X_va)[:, 1])
    print(f"  Final model AUC-ROC (val): {final_score:.4f}")

    # ── Calibration: param importance plot (Optuna only) ─────────────────────
    if tuner_used == "optuna":
        try:
            import optuna.visualization as ov
            param_imp = {p: 0.0 for p in best_params}
            for t in study.trials:
                if t.value and t.value > study.best_value * 0.98:
                    for p, v in t.params.items():
                        param_imp[p] = param_imp.get(p, 0) + 1
            if param_imp:
                pi_df = pd.Series(param_imp).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 4))
                pi_df.plot(kind="barh", ax=ax, color="#1565C0")
                ax.set_title("Parameter Frequency in Top Trials",
                             fontsize=11, fontweight="bold")
                ax.set_xlabel("Appearances in top-2% trials")
                plt.tight_layout()
                plt.savefig(REPORT_DIR / "param_importance.png", dpi=120, bbox_inches="tight")
                plt.close()
        except Exception:
            pass

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\n[3] Saving outputs…")
    joblib.dump(final, MODEL_DIR / "lgbm_optuna_champion.pkl")
    joblib.dump(final, MODEL_DIR / "final_model.pkl")
    joblib.dump(feat_cols, MODEL_DIR / "feat_cols.pkl")

    pd.DataFrame([{"tuner": tuner_used, "best_auc_roc": best_score, **best_params}]).to_csv(
        REPORT_DIR / "tuning_log.csv", index=False)

    log.info("Saved lgbm_optuna_champion.pkl, final_model.pkl, tuning_log.csv")
    print(f"\n  All outputs → {MODEL_DIR}")
    print("=" * 65)
    print("  Step 5 complete. Ready for Ethics & Explainability (06_)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
