"""
use_case_C_nlp/06_ethics_explainability.py
===========================================
DSF504 Use Case C_nlp — NLP Sentiment Analysis
ML Framework Phase 6: Ethics, Bias Audit & Model Explainability

Uses TF-IDF token weights (or SHAP LinearExplainer) as feature importance proxy.
Outputs :
  reports/use_case_C_nlp/shap_feature_importance.csv
  reports/use_case_C_nlp/shap_bar_importance.png
  reports/use_case_C_nlp/shap_tfidf_per_class.png
  reports/use_case_C_nlp/ethics_bias_report.csv

Run:
    cd C:\\DSF504
    python use_case_C_nlp/06_ethics_explainability.py
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import joblib
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

MODEL_DIR  = MODELS_DIR  / "use_case_C_nlp"
REPORT_DIR = REPORTS_DIR / "use_case_C_nlp"
DATA_PATH  = DATA_DIR    / "nlp_sentiment"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE    = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC"]
CLASSES    = ["Negative", "Neutral", "Positive"]


def load_champion():
    for fname in ["lgbm_optuna_champion.pkl", "champion.pkl"]:
        p = MODEL_DIR / fname
        if p.exists():
            obj = joblib.load(p)
            return (obj["model"], fname) if isinstance(obj, dict) and "model" in obj else (obj, fname)
    return None, None


def main():
    log.info("=" * 62)
    log.info("  Step 6: Ethics & Explainability — C_nlp (NLP Sentiment)")
    log.info("=" * 62)

    model, mname = load_champion()
    if model is None:
        raise FileNotFoundError("No champion pkl. Run Steps 4-5 first.")
    log.info("  Loaded %s → %s", mname, type(model).__name__)

    # Load TF-IDF vectoriser
    tfidf_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    tfidf = joblib.load(tfidf_path) if tfidf_path.exists() else None
    if tfidf is not None:
        vocab = tfidf.get_feature_names_out()
        log.info("  TF-IDF vocab size: %d", len(vocab))

    # Load validation data
    val_path = DATA_PATH / "val_fe.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"val_fe.parquet not found at {val_path}")
    df_val = pd.read_parquet(val_path)
    log.info("  val shape: %s", df_val.shape)

    # Feature columns (non-text engineered features)
    fc_path = MODEL_DIR / "feature_cols.pkl"
    if fc_path.exists():
        fe_cols = joblib.load(fc_path)
        X_val   = df_val[fe_cols].fillna(0).values
    else:
        # fallback: all numeric cols except target
        num_cols = [c for c in df_val.columns
                    if df_val[c].dtype in [np.float64, np.float32, np.int64, np.int32]
                    and c != "label"]
        fe_cols = num_cols
        X_val   = df_val[fe_cols].fillna(0).values

    y_val = df_val["label"].values if "label" in df_val.columns else np.zeros(len(df_val))

    # ── Feature importance from TF-IDF or model ───────────────────────────────
    if tfidf is not None and hasattr(model, "coef_"):
        # Linear model: coef_ has shape (n_classes, n_features)
        coef = np.abs(model.coef_)
        mean_abs = coef.mean(axis=0)
        feature_names = list(vocab)
        log.info("  Using linear coef_ magnitudes")
    elif hasattr(model, "feature_importances_"):
        mean_abs     = model.feature_importances_
        feature_names = fe_cols
    else:
        mean_abs      = np.ones(len(fe_cols))
        feature_names = fe_cols

    df_fi = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df_fi = df_fi.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df_fi.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
    log.info("Saved shap_feature_importance.csv (%d features)", len(df_fi))

    # ── Bar chart — top tokens ─────────────────────────────────────────────────
    top = df_fi.head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
            color=PALETTE[0], edgecolor="none")
    ax.set_xlabel("Mean |Coefficient| / Feature Importance")
    ax.set_title("Top 20 Token Importance — C_nlp Sentiment")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_bar_importance.png")

    # ── Per-class top tokens ───────────────────────────────────────────────────
    if tfidf is not None and hasattr(model, "coef_") and model.coef_.shape[0] >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
        for ci, (cls_name, coef_row) in enumerate(zip(CLASSES, model.coef_)):
            top_idx = np.argsort(np.abs(coef_row))[-15:][::-1]
            top_tokens = [vocab[i] for i in top_idx]
            top_vals   = coef_row[top_idx]
            axes[ci].barh(top_tokens[::-1], top_vals[::-1],
                          color=[PALETTE[0] if v >= 0 else PALETTE[3] for v in top_vals[::-1]])
            axes[ci].set_title(f"{cls_name}")
            axes[ci].axvline(0, color="gray", linewidth=0.8)
        fig.suptitle("Top Token Weights per Sentiment Class — C_nlp", fontsize=13)
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "shap_tfidf_per_class.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved shap_tfidf_per_class.png")
    else:
        # Proxy beeswarm
        std = X_val.std(axis=0)
        imp = getattr(model, "feature_importances_", np.ones(len(fe_cols)))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(std, imp, alpha=0.6, color=PALETTE[0], s=40)
        ax.set_xlabel("Feature Std Dev"); ax.set_ylabel("Importance")
        ax.set_title("Variability vs Importance — C_nlp")
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved shap_beeswarm.png (proxy)")

    # ── Bias audit — text length bands ────────────────────────────────────────
    if "text" in df_val.columns:
        df_val["text_len"] = df_val["text"].str.split().str.len().fillna(0)
        df_val["len_band"] = pd.qcut(df_val["text_len"], q=3,
                                     labels=["Short", "Medium", "Long"],
                                     duplicates="drop")
    else:
        df_val["len_band"] = "all"

    preds = model.predict(X_val)
    df_val["y_pred"] = preds
    df_val["y_true"] = y_val

    grp = (df_val.groupby("len_band")
           .apply(lambda g: pd.Series({
               "count":    len(g),
               "accuracy": float((g["y_pred"] == g["y_true"]).mean()),
           }))
           .reset_index())
    grp.to_csv(REPORT_DIR / "ethics_bias_report.csv", index=False)
    log.info("Saved ethics_bias_report.csv")

    overall_acc = float((preds == y_val).mean())
    log.info("  Overall accuracy: %.3f", overall_acc)

    log.info("=" * 62)
    log.info("  Step 6 complete — outputs in %s", REPORT_DIR)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
