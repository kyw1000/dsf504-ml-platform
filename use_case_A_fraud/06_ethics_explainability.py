"""
use_case_A_fraud/06_ethics_explainability.py
=============================================
DSF504 Use Case A — Fraud Detection
ML Framework Phase 6: Ethics, Bias Audit & Model Explainability

Dataset : IEEE-CIS Fraud Detection
Model   : LightGBM champion (lgbm_optuna_champion.pkl)
Outputs :
  reports/use_case_A/shap_feature_importance.csv
  reports/use_case_A/shap_bar_importance.png
  reports/use_case_A/shap_beeswarm.png
  reports/use_case_A/ethics_bias_report.csv

Run:
    cd C:\\DSF504
    python use_case_A_fraud/06_ethics_explainability.py
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

MODEL_DIR  = MODELS_DIR  / "use_case_A"
REPORT_DIR = REPORTS_DIR / "use_case_A"
DATA_PATH  = DATA_DIR    / "ieee_fraud"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC"]

TARGET = "isFraud"


def load_champion():
    # Preferred order: Optuna-tuned LGBM first, then any tuned/trained model.
    # Step 5 saves lgbm_optuna_champion.pkl only when LGB tuning succeeds;
    # if it failed (exception caught), fall back to other saved models.
    candidates = [
        "lgbm_optuna_champion.pkl",
        "champion.pkl",
        "lgbm_tuned.pkl",
        "xgb_tuned.pkl",
        "rf_tuned.pkl",
        "LightGBM.pkl",
        "XGBoost.pkl",
        "Random_Forest.pkl",
    ]
    for fname in candidates:
        p = MODEL_DIR / fname
        if p.exists():
            obj = joblib.load(p)
            # Unwrap dict-wrapped models
            if isinstance(obj, dict) and "model" in obj:
                obj = obj["model"]
            # Unwrap sklearn Pipelines — extract the final estimator
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


def compute_feature_importance(model, fe_cols, X_val):
    try:
        import shap
        log.info("Computing SHAP TreeExplainer values ...")
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_val), size=min(1000, len(X_val)), replace=False)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val[idx])
        if isinstance(sv, list):  # binary → take class-1
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        method = "SHAP"
    except ImportError:
        log.warning("shap not installed — using feature_importances_")
        mean_abs = (model.feature_importances_
                    if hasattr(model, "feature_importances_")
                    else np.ones(len(fe_cols)))
        method = "native"
    df_fi = pd.DataFrame({"feature": fe_cols, "mean_abs_shap": mean_abs})
    df_fi = df_fi.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    log.info("  Method: %s  Top: %s", method, df_fi.iloc[0]["feature"])
    return df_fi


def plot_bar(df_fi, top_n=20):
    df_top = df_fi.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_top["feature"][::-1], df_top["mean_abs_shap"][::-1],
            color=PALETTE[0], edgecolor="none")
    ax.set_xlabel("Mean |SHAP| / Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importance — UC A (Fraud Detection)")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_bar_importance.png")


def plot_beeswarm(model, fe_cols, X_val):
    try:
        import shap
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_val), size=min(500, len(X_val)), replace=False)
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
        # Proxy scatter
        imp = (model.feature_importances_ if hasattr(model, "feature_importances_")
               else np.ones(len(fe_cols)))
        std = X_val.std(axis=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(std, imp, alpha=0.6, color=PALETTE[0], s=40)
        for i, n in enumerate(fe_cols):
            if imp[i] > np.percentile(imp, 90):
                ax.annotate(n, (std[i], imp[i]), fontsize=7)
        ax.set_xlabel("Feature Std Dev")
        ax.set_ylabel("Feature Importance")
        ax.set_title("Variability vs Importance — UC A Fraud")
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved beeswarm proxy: shap_beeswarm.png")


def bias_audit(model, fe_cols, df_val, X_val, y_val):
    """Check false positive / negative rates across transaction amount bands."""
    log.info("Running bias audit ...")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:, 1]
    else:
        probs = model.predict(X_val)

    thr_path = MODEL_DIR / "lgbm_optimal_threshold.txt"
    threshold = float(thr_path.read_text().strip()) if thr_path.exists() else 0.5
    preds = (probs >= threshold).astype(int)

    df_a = df_val[["TransactionAmt"]].copy() if "TransactionAmt" in df_val.columns else pd.DataFrame()
    df_a["y_true"] = y_val
    df_a["y_pred"] = preds

    if "TransactionAmt" in df_a.columns:
        df_a["amt_band"] = pd.qcut(df_a["TransactionAmt"], q=4,
                                   labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                                   duplicates="drop")
        grp = df_a.groupby("amt_band").apply(
            lambda g: pd.Series({
                "count":     len(g),
                "fraud_rate": g["y_true"].mean(),
                "fpr":       ((g["y_pred"] == 1) & (g["y_true"] == 0)).mean(),
                "fnr":       ((g["y_pred"] == 0) & (g["y_true"] == 1)).mean(),
            })
        ).reset_index()
    else:
        # Fallback: overall stats only
        grp = pd.DataFrame([{
            "amt_band": "all",
            "count": len(df_a),
            "fraud_rate": float(y_val.mean()),
            "fpr": float(((preds == 1) & (y_val == 0)).mean()),
            "fnr": float(((preds == 0) & (y_val == 1)).mean()),
        }])

    grp.to_csv(REPORT_DIR / "ethics_bias_report.csv", index=False)
    log.info("Saved ethics_bias_report.csv")
    log.info("  Threshold: %.3f  Overall FPR: %.3f  FNR: %.3f",
             threshold,
             float(((preds == 1) & (y_val == 0)).mean()),
             float(((preds == 0) & (y_val == 1)).mean()))


def main():
    log.info("=" * 62)
    log.info("  Step 6: Ethics & Explainability — Use Case A (Fraud)")
    log.info("=" * 62)
    model, fe_cols, df_val, X_val, y_val = load_data()
    df_fi = compute_feature_importance(model, fe_cols, X_val)
    df_fi.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
    log.info("Saved shap_feature_importance.csv")
    plot_bar(df_fi)
    plot_beeswarm(model, fe_cols, X_val)
    bias_audit(model, fe_cols, df_val, X_val, y_val)
    log.info("=" * 62)
    log.info("  Step 6 complete — outputs in %s", REPORT_DIR)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
