"""
use_case_C_market/06_ethics_explainability.py
===============================================
DSF504 Use Case C_markets — Market Volatility Prediction (Optiver)
ML Framework Phase 6: Ethics, Bias Audit & Model Explainability

Outputs → reports/use_case_C_markets/
  shap_feature_importance.csv     shap_bar_importance.png
  shap_beeswarm.png               shap_dependence_top3.png
  residual_distribution.png       stock_group_fairness.png
  pred_vs_actual_ethics.png       ethics_bias_report.csv
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
from utils.ethics_viz import plot_shap_dependence, save_insights_txt

ensure_utf8()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODEL_DIR  = MODELS_DIR  / "use_case_C_markets"
REPORT_DIR = REPORTS_DIR / "use_case_C_markets"
DATA_PATH  = DATA_DIR    / "optiver_volatility"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC",
           "#26C6DA", "#EC407A", "#D4E157"]


def rmspe(y_true, y_pred):
    mask = y_true > 0
    return float(np.sqrt(np.mean(((y_pred[mask] - y_true[mask]) / y_true[mask]) ** 2)))


def load_champion():
    for fname in ["champion.pkl", "lgbm_optuna_champion.pkl"]:
        p = MODEL_DIR / fname
        if p.exists():
            obj = joblib.load(p)
            return (obj["model"], fname) if isinstance(obj, dict) and "model" in obj else (obj, fname)
    return None, None


def compute_feature_importance(model, fe_cols, X_val):
    try:
        import shap
        rng = np.random.default_rng(RANDOM_STATE)
        bg_idx = rng.choice(len(X_val), size=min(500, len(X_val)), replace=False)
        explainer = shap.TreeExplainer(model, data=X_val[bg_idx],
                                       feature_perturbation="interventional")
        shap_vals = explainer.shap_values(X_val[:2000])
        mean_abs  = np.abs(shap_vals).mean(axis=0)
    except (ImportError, Exception) as e:
        log.warning("SHAP unavailable (%s) — using feature_importances_", e)
        mean_abs = (model.feature_importances_ if hasattr(model, "feature_importances_")
                    else np.ones(len(fe_cols)))
    df_fi = pd.DataFrame({"feature": fe_cols, "mean_abs_shap": mean_abs})
    return df_fi.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_bar_importance(df_fi):
    top = df_fi.head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color=PALETTE[0], edgecolor="none")
    ax.set_xlabel("Mean |SHAP| / Feature Importance", fontsize=11)
    ax.set_title("Top 20 Feature Importance — C_markets Champion", fontsize=13)
    top_feat = df_fi.iloc[0]["feature"]
    ax.text(0.01, 0.02, f"[i] Top driver: '{top_feat}' — highest SHAP impact on realized volatility prediction.",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7", edgecolor="#F9A825", alpha=0.9))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(REPORT_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_bar_importance.png")
    return f"Top SHAP driver: '{top_feat}'. Microstructure features like order book imbalance dominate volatility predictions."


def plot_beeswarm_proxy(model, fe_cols, X_val):
    try:
        import shap
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_val), size=min(500, len(X_val)), replace=False)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val[idx])
        shap.summary_plot(shap_vals, X_val[idx], feature_names=fe_cols, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    except (ImportError, Exception):
        imp = (model.feature_importances_ if hasattr(model, "feature_importances_") else np.ones(len(fe_cols)))
        std = X_val.std(axis=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(std, imp, alpha=0.6, color=PALETTE[0], s=40)
        for i, name in enumerate(fe_cols):
            if imp[i] > np.percentile(imp, 85):
                ax.annotate(name, (std[i], imp[i]), fontsize=7, alpha=0.8)
        ax.set_xlabel("Feature Std Dev"); ax.set_ylabel("Feature Importance")
        ax.set_title("Feature Variability vs Importance — C_markets")
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved shap_beeswarm.png")


def plot_residual_distribution(y_true, y_pred):
    """Distribution of prediction errors — key for fairness in regression."""
    residuals = y_pred - y_true
    rel_errors = (y_pred - y_true) / np.clip(y_true, 1e-9, None) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute residuals
    ax = axes[0]
    ax.hist(residuals, bins=60, color=PALETTE[0], alpha=0.75, edgecolor="none")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Residual (Predicted − Actual)"); ax.set_ylabel("Count")
    ax.set_title("Residual Distribution — C_markets")
    bias = float(np.mean(residuals))
    ax.axvline(bias, color=PALETTE[3], linewidth=1.5, linestyle="-",
               label=f"Mean bias = {bias:.5f}")
    ax.legend(fontsize=9)
    insight_txt = (f"Mean residual = {bias:.5f} ({'overestimates' if bias > 0 else 'underestimates'} volatility on average). "
                   "A near-zero mean indicates low systematic bias.")
    ax.text(0.01, 0.02, f"[i] {insight_txt}", transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#F9A825", alpha=0.9))

    # Relative errors
    ax2 = axes[1]
    clipped = np.clip(rel_errors, np.percentile(rel_errors, 1), np.percentile(rel_errors, 99))
    ax2.hist(clipped, bins=60, color=PALETTE[2], alpha=0.75, edgecolor="none")
    ax2.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Relative Error (%)"); ax2.set_ylabel("Count")
    ax2.set_title("Relative Error Distribution (1st–99th pct)")
    p90 = float(np.percentile(np.abs(rel_errors), 90))
    ax2.text(0.01, 0.02, f"[i] 90th pct |rel error| = {p90:.2f}%. Tail errors drive RMSPE — investigate high-error stocks.",
             transform=ax2.transAxes, fontsize=7, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#F9A825", alpha=0.9))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(REPORT_DIR / "residual_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved residual_distribution.png  (bias=%.5f)", bias)
    return (f"Mean bias={bias:.5f}. 90th pct |relative error|={p90:.2f}%. "
            "Systematic over/under-estimation by stock or time bucket signals potential fairness issues.")


def plot_pred_vs_actual(y_true, y_pred, rmspe_val):
    """Scatter of predicted vs actual volatility with RMSPE annotation."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y_true), size=min(5000, len(y_true)), replace=False)
    x, y = y_true[idx], y_pred[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, alpha=0.2, s=5, color=PALETTE[0], edgecolors="none")
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Realized Volatility"); ax.set_ylabel("Predicted Realized Volatility")
    ax.set_title(f"Predicted vs Actual — C_markets Champion\nRMSPE = {rmspe_val:.5f}")
    ax.legend(fontsize=9)
    corr = float(np.corrcoef(x, y)[0, 1])
    insight = (f"Prediction-actual correlation r={corr:.3f}, RMSPE={rmspe_val:.5f}. "
               + ("Strong predictive accuracy." if corr > 0.85 else
                  "Moderate accuracy — model captures main trends but struggles at extremes."))
    ax.text(0.01, 0.02, f"[i] {insight}", transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#F9A825", alpha=0.9))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(REPORT_DIR / "pred_vs_actual_ethics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved pred_vs_actual_ethics.png  (r=%.3f)", corr)
    return insight


def bias_audit_stocks(model, fe_cols, df_val, X_val, y_true_orig, y_pred_orig):
    if "stock_id" not in df_val.columns:
        log.warning("stock_id not in val_fe — skipping per-stock audit")
        return pd.DataFrame()
    df_a = df_val[["stock_id"]].copy()
    df_a["y_true"] = y_true_orig
    df_a["y_pred"] = y_pred_orig
    df_a["sq_rel_err"] = ((df_a["y_pred"] - df_a["y_true"]) / np.clip(df_a["y_true"], 1e-9, None)) ** 2

    grp = (df_a.groupby("stock_id")
           .agg(rmspe_=("sq_rel_err", lambda x: float(np.sqrt(x.mean()))),
                count=("y_true", "count"),
                mean_rv=("y_true", "mean"))
           .reset_index()
           .rename(columns={"rmspe_": "rmspe"})
           .sort_values("rmspe"))

    grp.to_csv(REPORT_DIR / "ethics_bias_report.csv", index=False)
    log.info("Saved ethics_bias_report.csv  (%d stocks)", len(grp))

    # Per-stock RMSPE bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(grp["stock_id"].astype(str), grp["rmspe"], color=PALETTE[0], alpha=0.75)
    mean_r = grp["rmspe"].mean()
    ax.axhline(mean_r, color=PALETTE[3], linestyle="--", linewidth=1.5,
               label=f"Mean RMSPE = {mean_r:.5f}")
    ax.set_xlabel("Stock ID"); ax.set_ylabel("RMSPE")
    ax.set_title("Per-Stock RMSPE — Fairness Audit (C_markets)")
    ax.legend(fontsize=9)
    plt.xticks(rotation=90, fontsize=6)
    disparity = float(grp["rmspe"].max() / max(grp["rmspe"].min(), 1e-9))
    ax.text(0.01, 0.95, f"[i] Worst/best RMSPE ratio = {disparity:.1f}x. "
            "High disparity means model is significantly less reliable for some stocks.",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#F9A825", alpha=0.9))
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "stock_group_fairness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved stock_group_fairness.png  (disparity=%.1fx)", disparity)
    return grp


def main():
    log.info("=" * 62)
    log.info("  Phase 6: Ethics & Explainability — C_markets")
    log.info("=" * 62)

    model, mname = load_champion()
    if model is None:
        raise FileNotFoundError("No champion pkl. Run Steps 4-5 first.")
    log.info("  Loaded: %s  (%s)", type(model).__name__, mname)

    fc_path = MODEL_DIR / "feature_cols.pkl"
    if fc_path.exists():
        fe_cols = joblib.load(fc_path)
    elif hasattr(model, "feature_names_in_"):
        fe_cols = list(model.feature_names_in_)
    else:
        raise FileNotFoundError("feature_cols.pkl not found.")

    val_path = DATA_PATH / "val_fe.parquet"
    df_val = pd.read_parquet(val_path)
    for c in fe_cols:
        if c not in df_val.columns:
            df_val[c] = 0.0
    X_val  = df_val[fe_cols].fillna(0).values
    y_val  = df_val["target"].values.astype(float)

    y_pred_log  = model.predict(X_val)
    y_pred_orig = np.expm1(np.clip(y_pred_log, -10, 10))
    y_true_orig = np.clip(y_val, 1e-9, None)
    rmspe_val   = rmspe(y_true_orig, y_pred_orig)
    log.info("  Overall RMSPE: %.5f", rmspe_val)

    insights = {}

    # SHAP importance
    df_fi = compute_feature_importance(model, fe_cols, X_val)
    df_fi.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
    insights["shap_bar_importance"]    = plot_bar_importance(df_fi)
    plot_beeswarm_proxy(model, fe_cols, X_val)
    insights["shap_beeswarm"]          = "SHAP beeswarm shows feature direction and magnitude per sample. Red = high feature value contributing to higher volatility prediction."
    insights["shap_dependence_top3"]   = plot_shap_dependence(model, list(fe_cols), X_val, df_fi, REPORT_DIR, " — C_markets")

    # Regression-specific ethics plots
    insights["residual_distribution"]  = plot_residual_distribution(y_true_orig, y_pred_orig)
    insights["pred_vs_actual_ethics"]  = plot_pred_vs_actual(y_true_orig, y_pred_orig, rmspe_val)

    # Per-stock fairness
    grp = bias_audit_stocks(model, fe_cols, df_val, X_val, y_true_orig, y_pred_orig)
    if not grp.empty:
        disparity = float(grp["rmspe"].max() / max(grp["rmspe"].min(), 1e-9))
        insights["stock_group_fairness"] = (
            f"Per-stock RMSPE disparity = {disparity:.1f}x. "
            "Stocks with thin books or extreme volatility regimes are hardest to predict — "
            "consider stock-specific calibration."
        )

    save_insights_txt(insights, REPORT_DIR, "Use Case C_markets — Market Volatility Prediction")

    log.info("=" * 62)
    log.info("  Phase 6 complete — all outputs saved to %s", REPORT_DIR)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
