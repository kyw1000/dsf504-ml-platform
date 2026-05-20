"""
use_case_A_fraud/06_ethics_explainability.py
=============================================
DSF504 Use Case A — Fraud Detection
ML Framework Phase 6: Ethics, Bias Audit & Model Explainability

Outputs → reports/use_case_A/
  shap_feature_importance.csv     shap_bar_importance.png
  shap_beeswarm.png               shap_dependence_top3.png
  calibration_curve.png           threshold_sensitivity.png
  probability_distribution.png    ethics_confusion_matrix.png
  ethics_fairness_bars.png        ethics_bias_report.csv
  ethics_insights.txt
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
from utils.ethics_viz import (
    plot_calibration_curve, plot_threshold_sensitivity,
    plot_probability_distribution, plot_shap_dependence,
    plot_confusion_matrix_eth, plot_fairness_bars, save_insights_txt,
)

ensure_utf8()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODEL_DIR  = MODELS_DIR  / "use_case_A"
REPORT_DIR = REPORTS_DIR / "use_case_A"
DATA_PATH  = DATA_DIR    / "ieee_fraud"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC"]
TARGET  = "isFraud"


def load_champion():
    for fname in ["lgbm_optuna_champion.pkl", "champion.pkl", "lgbm_tuned.pkl",
                  "xgb_tuned.pkl", "rf_tuned.pkl", "LightGBM.pkl", "XGBoost.pkl"]:
        p = MODEL_DIR / fname
        if p.exists():
            obj = joblib.load(p)
            if isinstance(obj, dict) and "model" in obj:
                obj = obj["model"]
            try:
                from sklearn.pipeline import Pipeline
                if isinstance(obj, Pipeline):
                    obj = obj.steps[-1][1]
            except ImportError:
                pass
            log.info("  Using %s as champion", fname)
            return obj, fname
    return None, None


def load_data():
    log.info("Loading champion model ...")
    model, mname = load_champion()
    if model is None:
        raise FileNotFoundError("No champion pkl found. Run Steps 4-5 first.")
    log.info("  %s → %s", mname, type(model).__name__)

    fc_path = MODEL_DIR / "feature_cols.pkl"
    if fc_path.exists():
        fe_cols = joblib.load(fc_path)
    elif hasattr(model, "feature_name_"):
        fe_cols = model.feature_name_()
    elif hasattr(model, "feature_names_in_"):
        fe_cols = list(model.feature_names_in_)
    else:
        raise FileNotFoundError("feature_cols.pkl not found.")

    val_path = DATA_PATH / "val_fe.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"val_fe.parquet not found at {val_path}")
    df_val = pd.read_parquet(val_path)
    log.info("  val shape: %s", df_val.shape)

    for c in fe_cols:
        if c not in df_val.columns:
            df_val[c] = 0.0
    X_val = df_val[fe_cols].fillna(0).values
    y_val = df_val[TARGET].values.astype(int)
    return model, fe_cols, df_val, X_val, y_val


def compute_shap_importance(model, fe_cols, X_val):
    try:
        import shap
        log.info("Computing SHAP TreeExplainer values ...")
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_val), size=min(300, len(X_val)), replace=False)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val[idx])
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
    except ImportError:
        log.warning("shap not installed — using feature_importances_")
        mean_abs = (model.feature_importances_ if hasattr(model, "feature_importances_")
                    else np.ones(len(fe_cols)))
    df_fi = pd.DataFrame({"feature": fe_cols, "mean_abs_shap": mean_abs})
    return df_fi.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_bar(df_fi, top_n=20):
    df_top = df_fi.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_top["feature"][::-1], df_top["mean_abs_shap"][::-1],
            color=PALETTE[0], edgecolor="none")
    ax.set_xlabel("Mean |SHAP| / Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importance — UC A (Fraud Detection)")
    top_feat = df_fi.iloc[0]["feature"]
    ax.text(0.01, 0.02, f"[i] Top driver: '{top_feat}' — highest mean absolute SHAP impact across all samples.",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7", edgecolor="#F9A825", alpha=0.9))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(REPORT_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_bar_importance.png")
    return f"Top SHAP driver: '{top_feat}'. Feature importance shows which signals most influence the model's fraud decision."


def plot_beeswarm(model, fe_cols, X_val):
    try:
        import shap
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_val), size=min(200, len(X_val)), replace=False)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val[idx])
        if isinstance(sv, list):
            sv = sv[1]
        shap.summary_plot(sv, X_val[idx], feature_names=fe_cols, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("Saved shap_beeswarm.png")
    except ImportError:
        imp = (model.feature_importances_ if hasattr(model, "feature_importances_")
               else np.ones(len(fe_cols)))
        std = X_val.std(axis=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(std, imp, alpha=0.6, color=PALETTE[0], s=40)
        for i, n in enumerate(fe_cols):
            if imp[i] > np.percentile(imp, 90):
                ax.annotate(n, (std[i], imp[i]), fontsize=7)
        ax.set_xlabel("Feature Std Dev"); ax.set_ylabel("Feature Importance")
        ax.set_title("Variability vs Importance — UC A Fraud")
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved beeswarm proxy")


def bias_audit(model, fe_cols, df_val, X_val, y_val):
    """FPR/FNR across transaction amount quartile bands."""
    log.info("Running bias audit ...")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:, 1]
    else:
        probs = model.predict(X_val)

    thr_path = MODEL_DIR / "lgbm_optimal_threshold.txt"
    threshold = float(thr_path.read_text().strip()) if thr_path.exists() else 0.5
    preds = (probs >= threshold).astype(int)

    df_a = pd.DataFrame({"y_true": y_val, "y_pred": preds})
    if "TransactionAmt" in df_val.columns:
        df_a["TransactionAmt"] = df_val["TransactionAmt"].values
        df_a["amt_band"] = pd.qcut(df_a["TransactionAmt"], q=4,
                                   labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                                   duplicates="drop")
        grp = df_a.groupby("amt_band").apply(
            lambda g: pd.Series({
                "count":      len(g),
                "fraud_rate": float(g["y_true"].mean()),
                "fpr":        float(((g["y_pred"] == 1) & (g["y_true"] == 0)).mean()),
                "fnr":        float(((g["y_pred"] == 0) & (g["y_true"] == 1)).mean()),
            })
        ).reset_index()
        grp.rename(columns={"amt_band": "group"}, inplace=True)
        grp["attribute"] = "TransactionAmt_band"
    else:
        grp = pd.DataFrame([{
            "group": "all", "attribute": "all",
            "count": len(df_a),
            "fraud_rate": float(y_val.mean()),
            "fpr": float(((preds == 1) & (y_val == 0)).mean()),
            "fnr": float(((preds == 0) & (y_val == 1)).mean()),
        }])

    grp.to_csv(REPORT_DIR / "ethics_bias_report.csv", index=False)
    log.info("Saved ethics_bias_report.csv  (threshold=%.3f)", threshold)
    return grp, threshold


def main():
    log.info("=" * 62)
    log.info("  Phase 6: Ethics & Explainability — Use Case A (Fraud Detection)")
    log.info("=" * 62)

    model, fe_cols, df_val, X_val, y_val = load_data()
    insights = {}

    # SHAP importance
    df_fi = compute_shap_importance(model, fe_cols, X_val)
    df_fi.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
    log.info("Saved shap_feature_importance.csv")
    insights["shap_bar_importance"]    = plot_bar(df_fi)
    plot_beeswarm(model, fe_cols, X_val)
    insights["shap_beeswarm"]          = "SHAP beeswarm: color = feature value (red=high, blue=low). Wide horizontal spread = high impact on predictions."

    # SHAP dependence
    insights["shap_dependence_top3"]   = plot_shap_dependence(model, fe_cols, X_val, df_fi, REPORT_DIR, " — UC A Fraud")

    # Calibration
    insights["calibration_curve"]      = plot_calibration_curve(model, X_val, y_val, REPORT_DIR, " — UC A Fraud")

    # Threshold sensitivity
    insights["threshold_sensitivity"]  = plot_threshold_sensitivity(model, X_val, y_val, REPORT_DIR, " — UC A Fraud")

    # Probability distribution
    insights["probability_distribution"] = plot_probability_distribution(
        model, X_val, y_val, REPORT_DIR, " — UC A Fraud",
        class_labels=("Legitimate", "Fraud"))

    # Confusion matrix (ethics-framed)
    thr_path = MODEL_DIR / "lgbm_optimal_threshold.txt"
    thr = float(thr_path.read_text().strip()) if thr_path.exists() else 0.5
    insights["ethics_confusion_matrix"] = plot_confusion_matrix_eth(
        model, X_val, y_val, REPORT_DIR, " — UC A Fraud",
        threshold=thr, class_labels=("Legitimate", "Fraud"))

    # Bias audit + fairness bars
    bias_df, thr = bias_audit(model, fe_cols, df_val, X_val, y_val)
    insights["ethics_fairness_bars"]   = plot_fairness_bars(
        bias_df, REPORT_DIR, " — UC A Fraud",
        rate_col="fpr", rate_label="False Positive Rate")

    # Insights summary
    save_insights_txt(insights, REPORT_DIR, "Use Case A — Fraud Detection")

    log.info("=" * 62)
    log.info("  Phase 6 complete — 11 outputs in %s", REPORT_DIR)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
