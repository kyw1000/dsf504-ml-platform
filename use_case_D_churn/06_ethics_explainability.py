"""
use_case_D_churn/06_ethics_explainability.py
==============================================
DSF504 Use Case D — Customer Churn (KKBox)
ML Framework Phase 6: Ethics, Bias Audit & Model Explainability

Outputs → reports/use_case_D/
  shap_feature_importance.csv     shap_bar_importance.png (renamed from shap_bar.png)
  shap_beeswarm.png               shap_dependence_top3.png
  calibration_curve.png           threshold_sensitivity.png
  probability_distribution.png    ethics_confusion_matrix.png
  fairness_audit.png              ethics_bias_report.csv
  ethics_insights.txt
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

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

MODEL_DIR  = MODELS_DIR  / "use_case_D"
REPORT_DIR = REPORTS_DIR / "use_case_D"
DATA_PATH  = DATA_DIR    / "kkbox_churn"
TARGET     = "is_churn"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC"]

FAIRNESS_ATTRS = {
    "fe_age_bucket_young":  ("Age < 25",  "Age ≥ 25"),
    "fe_age_bucket_senior": ("Age ≥ 45",  "Age < 45"),
    "fe_is_male":           ("Male",       "Female / Unknown"),
    "fe_is_female":         ("Female",     "Male / Unknown"),
}


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
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#3949AB", alpha=0.85)
    ax.set_title("SHAP Feature Importance — Top 20\nKKBox Customer Churn", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    top_feat = df_fi.iloc[0]["feature"]
    ax.text(0.01, 0.02, f"[i] Top driver: '{top_feat}' — strongest signal for predicting subscriber churn.",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7", edgecolor="#F9A825", alpha=0.9))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(REPORT_DIR / "shap_bar_importance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_bar_importance.png")
    return f"Top SHAP driver: '{top_feat}'. Reveals which subscriber behaviours most predict churn."


def plot_beeswarm(model, fe_cols, X_val):
    try:
        import shap
        rng = np.random.default_rng(RANDOM_STATE + 1)
        idx = rng.choice(len(X_val), size=min(200, len(X_val)), replace=False)
        sv = shap.TreeExplainer(model).shap_values(X_val[idx])
        if isinstance(sv, list): sv = sv[1]
        shap.summary_plot(sv, X_val[idx], feature_names=fe_cols, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=130, bbox_inches="tight")
        plt.close("all")
    except ImportError:
        imp = getattr(model, "feature_importances_", np.ones(len(fe_cols)))
        std = X_val.std(axis=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(std, imp, alpha=0.6, color=PALETTE[0], s=40)
        ax.set_xlabel("Feature Std Dev"); ax.set_ylabel("Feature Importance")
        ax.set_title("Variability vs Importance — UC D Churn")
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved shap_beeswarm.png")


def build_bias_df(model, X_val, y_val, df_val):
    proba = model.predict_proba(X_val)[:, 1]
    df_audit = df_val.copy()
    df_audit["proba"] = proba
    df_audit["pred"]  = (proba >= 0.5).astype(int)

    bias_rows = []
    for attr, (pos_label, neg_label) in FAIRNESS_ATTRS.items():
        if attr not in df_audit.columns:
            continue
        for gval, glbl in [(1, pos_label), (0, neg_label)]:
            mask = df_audit[attr] == gval
            if mask.sum() < 10:
                continue
            g_y   = y_val[mask.values]
            g_p   = proba[mask.values]
            g_pr  = df_audit.loc[mask, "pred"].values
            from sklearn.metrics import roc_auc_score
            try:
                auc = float(roc_auc_score(g_y, g_p)) if len(np.unique(g_y)) > 1 else np.nan
            except Exception:
                auc = np.nan
            bias_rows.append({
                "attribute":    attr,
                "group":        glbl,
                "n":            int(mask.sum()),
                "churn_rate":   float(g_y.mean()),
                "mean_score":   float(g_p.mean()),
                "roc_auc":      auc,
                "fpr":          float(((g_pr == 1) & (g_y == 0)).mean()),
                "fnr":          float(((g_pr == 0) & (g_y == 1)).mean()),
            })
    return pd.DataFrame(bias_rows)


def main():
    log.info("=" * 62)
    log.info("  Phase 6: Ethics & Explainability — UC D (KKBox Churn)")
    log.info("=" * 62)

    model, mname = load_champion()
    if model is None:
        raise FileNotFoundError("No champion pkl. Run Steps 4-5 first.")
    log.info("  Loaded %s → %s", mname, type(model).__name__)

    fc_path = MODEL_DIR / "feature_cols.pkl"
    if fc_path.exists():
        fe_cols = joblib.load(fc_path)
    elif hasattr(model, "feature_name_"):
        fe_cols = list(model.feature_name_)
    else:
        fe_cols = list(model.feature_names_in_)

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
    insights["shap_beeswarm"]            = "SHAP beeswarm: direction and magnitude of each feature's effect on individual churn predictions."
    insights["shap_dependence_top3"]     = plot_shap_dependence(model, fe_cols, X_val, df_fi, REPORT_DIR, " — UC D Churn")

    # New visualizations
    insights["calibration_curve"]        = plot_calibration_curve(model, X_val, y_val, REPORT_DIR, " — UC D Churn")
    insights["threshold_sensitivity"]    = plot_threshold_sensitivity(model, X_val, y_val, REPORT_DIR, " — UC D Churn")
    insights["probability_distribution"] = plot_probability_distribution(
        model, X_val, y_val, REPORT_DIR, " — UC D Churn",
        class_labels=("Retained", "Churned"))
    insights["ethics_confusion_matrix"]  = plot_confusion_matrix_eth(
        model, X_val, y_val, REPORT_DIR, " — UC D Churn",
        class_labels=("Retained", "Churned"),
        filename="confusion_matrix.png")

    # Fairness
    bias_df = build_bias_df(model, X_val, y_val, df_val)
    bias_df.to_csv(REPORT_DIR / "ethics_bias_report.csv", index=False)
    log.info("Saved ethics_bias_report.csv (%d groups)", len(bias_df))
    insights["fairness_audit"]           = plot_fairness_bars(
        bias_df, REPORT_DIR, " — UC D Churn",
        rate_col="fpr", rate_label="False Positive Rate",
        filename="fairness_audit.png")

    save_insights_txt(insights, REPORT_DIR, "Use Case D — KKBox Customer Churn")

    log.info("=" * 62)
    log.info("  Phase 6 complete — 11 outputs in %s", REPORT_DIR)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
