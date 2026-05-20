"""
use_case_F_esg/04_model_training.py
=====================================
DSF504 — Use Case F: ESG & Greenwashing Risk Scoring
Step 4: Model Training & Comparison

Algorithms compared
--------------------
  LogisticRegression   — L2 regularised, multinomial, balanced class weights
  RandomForestClassifier — ensemble of decision trees
  XGBoostClassifier    — gradient boosting, softmax multiclass (champion)
  LGBMClassifier       — LightGBM fast boosting

Primary metric: macro-F1 (weights all risk tiers equally regardless of frequency)
Secondary: accuracy, weighted-F1, per-class F1 (Low / Medium / High)

Outputs
-------
  models/use_case_F/champion.pkl         — best model (XGBoost)
  models/use_case_F/model_comparison.csv — all model metrics
  reports/use_case_F/model_comparison.png
  reports/use_case_F/confusion_matrix.png
  reports/use_case_F/feature_importance.png
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

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import (f1_score, accuracy_score, classification_report,
                                   confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE, CV_FOLDS
from utils.encoding_guard import ensure_utf8

ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR   / "sec_esg"
REPORT_DIR  = REPORTS_DIR / "use_case_F"
MODEL_DIR   = MODELS_DIR  / "use_case_F"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "greenwashing_risk"
RISK_MAP = {"Low": 0, "Medium": 1, "High": 2}
CLASSES  = ["Low", "Medium", "High"]
SEED     = RANDOM_STATE
PALETTE  = {"Low": "#66BB6A", "Medium": "#FFA726", "High": "#EF5350"}


def _load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(DATA_SUBDIR / f"{name}_fe.parquet")
    drop = [TARGET, "company_id", "env_claim_label"]
    X = df.drop(columns=[c for c in drop if c in df.columns]).values.astype(np.float32)
    y = df[TARGET].map(RISK_MAP).values
    return X, y


def _eval(model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    y_pred = model.predict(X_val)
    return {
        "accuracy":    round(accuracy_score(y_val, y_pred), 4),
        "f1_macro":    round(f1_score(y_val, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_val, y_pred, average="weighted"), 4),
        "f1_low":      round(f1_score(y_val, y_pred, average=None, labels=[0,1,2])[0], 4),
        "f1_medium":   round(f1_score(y_val, y_pred, average=None, labels=[0,1,2])[1], 4),
        "f1_high":     round(f1_score(y_val, y_pred, average=None, labels=[0,1,2])[2], 4),
    }


def main() -> None:
    log.info("Step 4 — ESG & Greenwashing: Model Training")

    X_train, y_train = _load_split("train")
    X_val,   y_val   = _load_split("val")
    X_test,  y_test  = _load_split("test")
    log.info("Shapes — Train: %s | Val: %s | Test: %s",
             X_train.shape, X_val.shape, X_test.shape)

    # ── Define models ──────────────────────────────────────────────────────────
    models: dict = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=300, solver="saga", class_weight="balanced", random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced",
            random_state=SEED, n_jobs=1,
        ),
    }

    # XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.10,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softmax", num_class=3,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=SEED, n_jobs=1, verbosity=0,
        )
    except ImportError:
        log.warning("xgboost not installed — skipping XGBoost")

    # LightGBM
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=80, max_depth=5, learning_rate=0.10,
            subsample=0.8, colsample_bytree=0.8,
            num_class=3, objective="multiclass",
            class_weight="balanced",
            random_state=SEED, n_jobs=1, verbose=-1,
        )
    except ImportError:
        log.warning("lightgbm not installed — skipping LightGBM")

    # ── CV + Val evaluation ───────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    results = []
    trained = {}

    for name, model in models.items():
        log.info("Training %s...", name)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=skf, scoring="f1_macro", n_jobs=1)
        model.fit(X_train, y_train)
        val_metrics = _eval(model, X_val, y_val)
        trained[name] = model

        row = {"model": name,
               "cv_f1_macro_mean": round(cv_scores.mean(), 4),
               "cv_f1_macro_std":  round(cv_scores.std(), 4),
               **val_metrics}
        results.append(row)
        log.info(
            "  %s — CV F1: %.3f±%.3f | Val F1: %.3f | Val Acc: %.3f",
            name, cv_scores.mean(), cv_scores.std(),
            val_metrics["f1_macro"], val_metrics["accuracy"],
        )

    # ── Select champion ────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    champion_name = df_results.iloc[0]["model"]
    champion = trained[champion_name]
    log.info("Champion: %s (val macro-F1 = %.4f)", champion_name, df_results.iloc[0]["f1_macro"])

    # Test set evaluation
    test_metrics = _eval(champion, X_test, y_test)
    log.info("Test metrics — %s", test_metrics)
    log.info("\nClassification Report (test):\n%s",
             classification_report(y_test, champion.predict(X_test),
                                   target_names=CLASSES, digits=3))

    # ── Save artefacts ─────────────────────────────────────────────────────────
    with open(MODEL_DIR / "champion.pkl", "wb") as f:
        pickle.dump(champion, f)
    df_results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    log.info("Saved champion.pkl and model_comparison.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Model comparison bar chart
    models_list = df_results["model"].tolist()
    f1_vals     = df_results["f1_macro"].tolist()
    bar_colors  = ["#42A5F5"] * len(models_list)
    bar_colors[0] = "#FFA726"  # highlight champion
    axes[0].barh(models_list[::-1], f1_vals[::-1], color=bar_colors[::-1], edgecolor="white")
    axes[0].set_title("Model Comparison — Validation Macro-F1", fontweight="bold")
    axes[0].set_xlabel("Macro-F1 Score")
    axes[0].axvline(df_results.iloc[0]["f1_macro"], color="red", linestyle="--",
                    linewidth=0.8, label=f"Champion: {champion_name}")
    for i, (m, v) in enumerate(zip(models_list[::-1], f1_vals[::-1])):
        axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    axes[0].legend()

    # Per-class F1 breakdown
    class_f1_data = {
        m: [row[f"f1_{c.lower()}"] for c in CLASSES]
        for m, row in zip(df_results["model"], df_results.to_dict("records"))
    }
    x = np.arange(len(CLASSES))
    width = 0.8 / len(models_list)
    for i, (m, vals) in enumerate(class_f1_data.items()):
        offset = (i - len(models_list)/2 + 0.5) * width
        axes[1].bar(x + offset, vals, width, label=m, alpha=0.85)
    axes[1].set_title("Per-Class F1 by Model", fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(CLASSES)
    axes[1].set_ylabel("F1 Score"); axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix
    y_pred_test = champion.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {champion_name} (Test Set)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Feature importance
    if hasattr(champion, "feature_importances_"):
        df_fe = pd.read_parquet(DATA_SUBDIR / "train_fe.parquet")
        drop  = [TARGET, "company_id", "env_claim_label"]
        feat_names = [c for c in df_fe.columns if c not in drop]
        importances = champion.feature_importances_
        top_n = min(20, len(feat_names))
        idx = np.argsort(importances)[-top_n:]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(np.array(feat_names)[idx], importances[idx], color="#42A5F5", edgecolor="white")
        ax.set_title(f"Top {top_n} Feature Importances — {champion_name}", fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "feature_importance.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved feature_importance.png")

    log.info("Step 4 complete ✓")


if __name__ == "__main__":
    main()
