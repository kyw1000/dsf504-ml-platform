"""
use_case_G1_robo/05_hyperparameter_tuning.py
=============================================
Use Case G1 — Robo-Advisory Portfolio Recommendation
Phase 3, Step 5: Hyperparameter Tuning

Tunes LightGBM LambdaRank using Optuna (TPE sampler) maximising NDCG@10.
Falls back to grid search if Optuna is not installed.

Run
---
    cd C:\\DSF504
    python use_case_G1_robo/05_hyperparameter_tuning.py
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "far_trans"
REPORT_DIR  = REPORTS_DIR / "use_case_G1"
MODEL_DIR   = MODELS_DIR / "use_case_G1"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

import lightgbm as lgb


# ─────────────────────────────────────────────────────────────────────────────
# Metric
# ─────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(y_score)[::-1][:k]
    gains = y_true[order]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg  = (gains / discounts).sum()
    ideal = np.sort(y_true)[::-1][:k]
    idcg = (ideal / discounts[:len(ideal)]).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_ndcg(model, X: np.ndarray, df: pd.DataFrame, k: int = 10) -> float:
    scores = model.predict(X)
    df = df.copy()
    df["_score"] = scores
    ndcgs = []
    for _, grp in df.groupby("query_id"):
        if grp["label"].sum() == 0:
            continue
        ndcgs.append(ndcg_at_k(grp["label"].values, grp["_score"].values, k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(X_tr, y_tr, q_tr, X_va, val_df):
    def objective(trial):
        params = dict(
            objective         = "lambdarank",
            n_estimators      = trial.suggest_int("n_estimators", 100, 500),
            num_leaves        = trial.suggest_int("num_leaves", 16, 128),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            random_state      = RANDOM_STATE,
            n_jobs            = 1,
            verbose           = -1,
        )
        model = lgb.LGBMRanker(**params)
        model.fit(X_tr, y_tr, group=q_tr)
        return evaluate_ndcg(model, X_va, val_df)
    return objective


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G1: Robo-Advisory Portfolio Recommendation")
    print("  Step 5: Hyperparameter Tuning (LGB LambdaRank)")
    print("=" * 65 + "\n")

    train_df = pd.read_parquet(DATA_SUBDIR / "train_pairs.parquet")
    val_df   = pd.read_parquet(DATA_SUBDIR / "val_pairs.parquet")

    feat_cols = [c for c in train_df.columns
                 if c not in ["customer_id", "isin", "label", "query_id", "_count"]]
    X_tr = train_df[feat_cols].fillna(0).values
    y_tr = train_df["label"].values
    q_tr = train_df.groupby("query_id").size().values
    X_va = val_df[feat_cols].fillna(0).values

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
        objective = make_objective(X_tr, y_tr, q_tr, X_va, val_df)
        study.optimize(objective, n_trials=15, timeout=40, show_progress_bar=False)
        best_params = study.best_params
        best_score  = study.best_value
        tuner_used  = "optuna"
        print(f"  Optuna: {len(study.trials)} trials | best NDCG@10={best_score:.4f}")
        print(f"  Best params: {best_params}")

        # Save history plot
        scores = [t.value for t in study.trials if t.value is not None]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, len(scores) + 1), scores, "o-", color="#1976D2")
        ax.plot(range(1, len(scores) + 1),
                [max(scores[:i+1]) for i in range(len(scores))],
                "--", color="#388E3C", label="Best so far")
        ax.set_title("Optuna Trial History — NDCG@10", fontsize=12, fontweight="bold")
        ax.set_xlabel("Trial")
        ax.set_ylabel("NDCG@10")
        ax.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "optuna_history.png", dpi=120, bbox_inches="tight")
        plt.close()

    except Exception as e:
        log.warning(f"Optuna failed ({e}). Running grid search.")
        grid = [
            {"n_estimators":200,"num_leaves":31,"learning_rate":0.05,"subsample":0.8,
             "colsample_bytree":0.8,"min_child_samples":10,"reg_lambda":0.1},
            {"n_estimators":300,"num_leaves":63,"learning_rate":0.03,"subsample":0.7,
             "colsample_bytree":0.7,"min_child_samples":20,"reg_lambda":1.0},
            {"n_estimators":150,"num_leaves":15,"learning_rate":0.10,"subsample":0.9,
             "colsample_bytree":0.9,"min_child_samples":5,"reg_lambda":0.01},
        ]
        for cfg in grid:
            m = lgb.LGBMRanker(objective="lambdarank", n_jobs=1, verbose=-1,
                                random_state=RANDOM_STATE, **cfg)
            m.fit(X_tr, y_tr, group=q_tr)
            s = evaluate_ndcg(m, X_va, val_df)
            print(f"  Grid: NDCG@10={s:.4f}  {cfg}")
            if s > best_score:
                best_score  = s
                best_params = cfg

    # Train final model
    final_params = dict(
        objective="lambdarank", n_jobs=1, verbose=-1, random_state=RANDOM_STATE,
        **best_params
    )
    final = lgb.LGBMRanker(**final_params)
    final.fit(X_tr, y_tr, group=q_tr)

    final_ndcg = evaluate_ndcg(final, X_va, val_df)
    print(f"\n  Final model NDCG@10 (val): {final_ndcg:.4f}")

    joblib.dump(final, MODEL_DIR / "lgbm_optuna_champion.pkl")
    joblib.dump(final, MODEL_DIR / "final_model.pkl")
    pd.DataFrame([{"tuner": tuner_used, "best_ndcg_at_10": best_score, **best_params}]).to_csv(
        REPORT_DIR / "tuning_log.csv", index=False)

    log.info("Saved lgbm_optuna_champion.pkl, final_model.pkl, tuning_log.csv")
    print("=" * 65)
    print("  Step 5 complete. Ready for Ethics & Explainability (06_)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
