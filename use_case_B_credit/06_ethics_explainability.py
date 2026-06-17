"""
use_case_B_credit/06_ethics_explainability.py
==============================================
DSF504 Use Case B — Credit Risk Scoring
ML Framework Phase 6: Ethics, Bias Audit & Model Explainability

Outputs → reports/use_case_B/
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

MODEL_DIR  = MODELS_DIR  / "use_case_B"
REPORT_DIR = REPORTS_DIR / "use_case_B"
DATA_PATH  = DATA_DIR    / "gmsc_credit"
TARGET     = "SeriousDlqin2yrs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC"]


def load_champion():
    for fname in ["lgbm_optuna_champion.pkl", "champion.pkl"]:
        p = MODEL_DIR / fname
        if p.exists():
            obj = joblib.load(p)
            return (obj["model"], fname) if isinstance(obj, dict) and "model" in obj else (obj, fname)
    return None, None


def compute_shap_importance(model, fe_cols, X_val):
    try:
        import shap
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_val), size=min(300, len(X_val)), replace=False)
        sv = shap.TreeExplainer(model).shap_values(X_val[idx])
        if isinstance(sv, list): sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
    except ImportError:
        mean_abs = getattr(model, "feature_importances_", np.ones(len(fe_cols)))
    df_fi = pd.DataFrame({"feature": fe_cols, "mean_abs_shap": mean_abs})
    return df_fi.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_bar(df_fi):
    top = df_fi.head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color=PALETTE[0], edgecolor="none")
    ax.set_xlabel("Mean |SHAP| / Feature Importance")
    ax.set_title("Top 20 Feature Importance — UC B (Credit Risk Scoring)")
    top_feat = df_fi.iloc[0]["feature"]
    ax.text(0.01, 0.02, f"[i] Top driver: '{top_feat}' — most influential variable in credit default prediction.",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7", edgecolor="#F9A825", alpha=0.9))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(REPORT_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_bar_importance.png")
    return f"Top SHAP driver: '{top_feat}'. Reveals which financial behaviours most drive default risk scores."


def plot_beeswarm(model, fe_cols, X_val):
    try:
        import shap
        rng = np.random.default_rng(RANDOM_STATE + 1)
        idx = rng.choice(len(X_val), size=min(200, len(X_val)), replace=False)
        sv = shap.TreeExplainer(model).shap_values(X_val[idx])
        if isinstance(sv, list): sv = sv[1]
        shap.summary_plot(sv, X_val[idx], feature_names=fe_cols, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    except ImportError:
        std = X_val.std(axis=0)
        imp = getattr(model, "feature_importances_", np.ones(len(fe_cols)))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(std, imp, alpha=0.6, color=PALETTE[0], s=40)
        ax.set_xlabel("Feature Std Dev"); ax.set_ylabel("Feature Importance")
        ax.set_title("Variability vs Importance — UC B Credit")
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved shap_beeswarm.png")


def bias_audit(model, X_val, y_val, df_val):
    """FPR/FNR by debt-to-income ratio quartile — a protected-adjacent attribute in credit scoring."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:, 1]
    else:
        probs = model.predict(X_val)
    thr_path = MODEL_DIR / "lgbm_optimal_threshold.txt"
    thr = float(thr_path.read_text().strip()) if thr_path.exists() else 0.5
    preds = (probs >= thr).astype(int)

    df_a = pd.DataFrame({"y_true": y_val, "y_pred": preds, "proba": probs})
    dti_col = next((c for c in ["DebtRatio", "debt_ratio", "debt_to_income"] if c in df_val.columns), None)
    if dti_col:
        df_a["dti_band"] = pd.qcut(df_val[dti_col].values, q=4,
                                   labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                                   duplicates="drop")
        grp = df_a.groupby("dti_band").apply(
            lambda g: pd.Series({
                "count":       len(g),
                "default_rate": float(g["y_true"].mean()),
                "fpr":         float(((g["y_pred"] == 1) & (g["y_true"] == 0)).mean()),
                "fnr":         float(((g["y_pred"] == 0) & (g["y_true"] == 1)).mean()),
            })
        ).reset_index()
        grp.rename(columns={"dti_band": "group"}, inplace=True)
        grp["attribute"] = "DebtRatio_band"
    else:
        grp = pd.DataFrame([{
            "group": "all", "attribute": "all",
            "count": len(df_a),
            "default_rate": float(y_val.mean()),
            "fpr":  float(((preds == 1) & (y_val == 0)).mean()),
            "fnr":  float(((preds == 0) & (y_val == 1)).mean()),
        }])

    # overall metrics for ethics_bias_report
    pd.DataFrame({
        "metric": ["threshold", "accuracy", "precision", "recall", "fpr", "fnr"],
        "value": [
            thr,
            float((preds == y_val).mean()),
            float((preds & y_val).sum() / max(preds.sum(), 1)),
            float((preds & y_val).sum() / max(y_val.sum(), 1)),
            float(((preds == 1) & (y_val == 0)).mean()),
            float(((preds == 0) & (y_val == 1)).mean()),
        ]
    }).to_csv(REPORT_DIR / "ethics_bias_report.csv", index=False)
    log.info("Saved ethics_bias_report.csv  (threshold=%.3f)", thr)
    return grp, thr


def main():
    log.info("=" * 62)
    log.info("  Phase 6: Ethics & Explainability — UC B (Credit Risk)")
    log.info("=" * 62)

    model, mname = load_champion()
    if model is None:
        raise FileNotFoundError("No champion pkl. Run Steps 4-5 first.")
    log.info("  Loaded %s → %s", mname, type(model).__name__)

    fc_path = MODEL_DIR / "feature_cols.pkl"
    fe_cols = (joblib.load(fc_path) if fc_path.exists()
               else (list(model.feature_name_) if hasattr(model, "feature_name_")
                     else list(model.feature_names_in_)))

    val_path = DATA_PATH / "val_fe.parquet"
    df_val = pd.read_parquet(val_path)
    for c in fe_cols:
        if c not in df_val.columns:
            df_val[c] = 0.0
    X_val = df_val[fe_cols].fillna(0).values
    y_val = df_val[TARGET].values

    insights = {}

    # SHAP importance
    df_fi = compute_shap_importance(model, fe_cols, X_val)
    df_fi.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
    insights["shap_bar_importance"]      = plot_bar(df_fi)
    plot_beeswarm(model, fe_cols, X_val)
    insights["shap_beeswarm"]            = "SHAP beeswarm: red = high feature value, blue = low. Wide spread = strong influence on credit score."
    insights["shap_dependence_top3"]     = plot_shap_dependence(model, fe_cols, X_val, df_fi, REPORT_DIR, " — UC B Credit")

    # Distribution & calibration plots
    insights["calibration_curve"]        = plot_calibration_curve(model, X_val, y_val, REPORT_DIR, " — UC B Credit")
    insights["threshold_sensitivity"]    = plot_threshold_sensitivity(model, X_val, y_val, REPORT_DIR, " — UC B Credit")
    insights["probability_distribution"] = plot_probability_distribution(
        model, X_val, y_val, REPORT_DIR, " — UC B Credit",
        class_labels=("No Default", "Default"))

    # Confusion matrix
    thr_path = MODEL_DIR / "lgbm_optimal_threshold.txt"
    thr = float(thr_path.read_text().strip()) if thr_path.exists() else 0.5
    insights["ethics_confusion_matrix"]  = plot_confusion_matrix_eth(
        model, X_val, y_val, REPORT_DIR, " — UC B Credit",
        threshold=thr, class_labels=("No Default", "Default"))

    # Bias audit + fairness
    bias_df, thr = bias_audit(model, X_val, y_val, df_val)
    insights["ethics_fairness_bars"]     = plot_fairness_bars(
        bias_df, REPORT_DIR, " — UC B Credit",
        rate_col="fpr", rate_label="False Positive Rate (Unfair Rejections)")

    save_insights_txt(insights, REPORT_DIR, "Use Case B — Credit Risk Scoring")

    log.info("=" * 62)
    log.info("  Phase 6 complete — 11 outputs in %s", REPORT_DIR)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
