"""
use_case_G2_xai/04_model_training.py
======================================
Use Case G2 — Explainable AI for Analysts & Managers
Phase 3, Step 4: Model Training

Models Compared
---------------
  BASELINE  : Logistic Regression (interpretable benchmark)
  TREE      : Random Forest (ensemble benchmark)
  CHAMPION  : LightGBM binary classifier (main model)

Evaluation Metrics
------------------
  AUC-ROC   : discrimination ability
  AUC-PR    : precision–recall trade-off (better for imbalanced targets)
  F1        : harmonic mean of precision and recall at threshold 0.5
  Precision : positive predictive value
  Recall    : sensitivity / hit rate

All models scored on the held-out val set (fiscal year 2022).
Champion saved to models/use_case_G2/.
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              roc_curve, precision_recall_curve)
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
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "outperform"
DROP_COLS  = ["ticker", "fiscal_year", "sector", "forward_return_12m", LABEL_COL]


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load():
    train = pd.read_parquet(DATA_SUBDIR / "train_fe.parquet")
    val   = pd.read_parquet(DATA_SUBDIR / "val_fe.parquet")
    return train, val


def _Xy(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].fillna(0).values
    y = df[LABEL_COL].values
    return X, y, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_score: np.ndarray, name: str) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "model":     name,
        "auc_roc":   round(roc_auc_score(y_true, y_score), 4),
        "auc_pr":    round(average_precision_score(y_true, y_score), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic(X_tr, y_tr, X_va):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE,
                                class_weight="balanced", n_jobs=-1)
    model.fit(X_tr_s, y_tr)
    return model, scaler, model.predict_proba(X_va_s)[:, 1]


def train_random_forest(X_tr, y_tr, X_va):
    model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    min_samples_leaf=10, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_tr, y_tr)
    return model, model.predict_proba(X_va)[:, 1]


def train_lightgbm(X_tr, y_tr, X_va, y_va):
    params = dict(
        objective         = "binary",
        n_estimators      = 400,
        num_leaves        = 63,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_samples = 20,
        reg_lambda        = 1.0,
        class_weight      = "balanced",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbose           = -1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(period=-1)])
    return model, model.predict_proba(X_va)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_pr(results_dict: dict, y_va: np.ndarray) -> None:
    """Overlay ROC and PR curves for all models."""
    colors = {"Logistic Regression": "#F57C00",
              "Random Forest":       "#388E3C",
              "LightGBM":            "#1565C0"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Comparison — ROC & Precision-Recall Curves",
                 fontsize=13, fontweight="bold")

    for name, y_score in results_dict.items():
        fpr, tpr, _ = roc_curve(y_va, y_score)
        auc = roc_auc_score(y_va, y_score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                     color=colors.get(name, "gray"), linewidth=2)

        prec, rec, _ = precision_recall_curve(y_va, y_score)
        ap = average_precision_score(y_va, y_score)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                     color=colors.get(name, "gray"), linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=9)

    baseline = y_va.mean()
    axes[1].axhline(baseline, color="k", linestyle="--", linewidth=0.8,
                    label=f"Baseline (P={baseline:.2f})")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roc_pr_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved roc_pr_curves.png")


def plot_model_comparison(results: list[dict]) -> None:
    df = pd.DataFrame(results)
    metrics = ["auc_roc", "auc_pr", "f1", "precision", "recall"]
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#F57C00", "#388E3C", "#1565C0"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["model"],
               color=colors[i], alpha=0.85)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").upper() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison — Validation Metrics (G2 XAI)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved model_comparison.png")


def plot_lgb_feature_importance(model, feat_cols: list[str]) -> None:
    imp = pd.DataFrame({"feature": feat_cols,
                         "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(25)

    colors = ["#1565C0" if "__rank" in f
              else "#388E3C" if f in ["peg_ratio", "interest_burden", "quality_spread",
                                       "value_composite", "growth_composite",
                                       "profitability_composite", "leverage_risk"]
              else "#7B1FA2" if f in ["macro_regime", "is_crisis_year",
                                       "is_bull_year", "sector_enc"]
              else "#F57C00"
              for f in imp["feature"]]

    fig, ax = plt.subplots(figsize=(10, 9))
    imp.sort_values("importance").plot(kind="barh", x="feature", y="importance",
                                        ax=ax, color=colors[::-1], legend=False)
    ax.set_title("LightGBM Feature Importance — Top 25 (G2 XAI)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance (split gain)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1565C0", label="Rank features"),
        Patch(facecolor="#388E3C", label="Derived / Composite"),
        Patch(facecolor="#7B1FA2", label="Macro / Sector"),
        Patch(facecolor="#F57C00", label="Raw ratios"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "lgb_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved lgb_feature_importance.png")


def plot_score_distribution(y_va: np.ndarray, lgb_scores: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(lgb_scores[y_va == 0], bins=40, alpha=0.6, color="#F44336",
            label="Non-outperform (0)", density=True)
    ax.hist(lgb_scores[y_va == 1], bins=40, alpha=0.6, color="#4CAF50",
            label="Outperform (1)", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="Threshold = 0.5")
    ax.set_title("LightGBM Score Distribution by Class", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Probability (Outperform)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "score_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved score_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G2: Explainable AI for Analysts & Managers")
    print("  Step 4: Model Training")
    print("=" * 65 + "\n")

    train, val = _load()
    X_tr, y_tr, feat_cols = _Xy(train)
    X_va, y_va, _         = _Xy(val)
    print(f"[1] Train: {X_tr.shape}  |  Val: {X_va.shape}")
    print(f"    Label balance — Train: {y_tr.mean():.3f}  |  Val: {y_va.mean():.3f}\n")

    all_results   = []
    scores_dict   = {}

    # ── Logistic Regression ──────────────────────────────────────────────────
    print("[2] Training Logistic Regression…")
    lr_model, lr_scaler, lr_scores = train_logistic(X_tr, y_tr, X_va)
    res = evaluate(y_va, lr_scores, "Logistic Regression")
    all_results.append(res)
    scores_dict["Logistic Regression"] = lr_scores
    print(f"    AUC-ROC={res['auc_roc']}  AUC-PR={res['auc_pr']}  F1={res['f1']}")

    # ── Random Forest ────────────────────────────────────────────────────────
    print("[3] Training Random Forest…")
    rf_model, rf_scores = train_random_forest(X_tr, y_tr, X_va)
    res = evaluate(y_va, rf_scores, "Random Forest")
    all_results.append(res)
    scores_dict["Random Forest"] = rf_scores
    print(f"    AUC-ROC={res['auc_roc']}  AUC-PR={res['auc_pr']}  F1={res['f1']}")

    # ── LightGBM ─────────────────────────────────────────────────────────────
    print("[4] Training LightGBM classifier…")
    lgb_model, lgb_scores = train_lightgbm(X_tr, y_tr, X_va, y_va)
    res = evaluate(y_va, lgb_scores, "LightGBM")
    all_results.append(res)
    scores_dict["LightGBM"] = lgb_scores
    print(f"    AUC-ROC={res['auc_roc']}  AUC-PR={res['auc_pr']}  F1={res['f1']}")

    # ── Summary ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    champion_row = results_df.loc[results_df["auc_roc"].idxmax()]
    print(f"\n  Champion: {champion_row['model']}  "
          f"(AUC-ROC={champion_row['auc_roc']}  AUC-PR={champion_row['auc_pr']})")

    print("\n[5] Saving outputs…")
    joblib.dump(lgb_model, MODEL_DIR / "champion.pkl")
    joblib.dump(lr_scaler, MODEL_DIR / "lr_scaler.pkl")
    joblib.dump(feat_cols, MODEL_DIR / "feat_cols.pkl")
    with open(MODEL_DIR / "champion_name.txt", "w") as f:
        f.write("LightGBM")
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    log.info("Saved champion.pkl, feat_cols.pkl, model_comparison.csv")

    print("[6] Plotting…")
    plot_roc_pr(scores_dict, y_va)
    plot_model_comparison(all_results)
    plot_lgb_feature_importance(lgb_model, feat_cols)
    plot_score_distribution(y_va, lgb_scores)

    print(f"\n  All outputs → {MODEL_DIR}  /  {REPORT_DIR}")
    print("=" * 65)
    print("  Step 4 complete. Ready for Hyperparameter Tuning (05_)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
