"""
use_case_C_market/06_ethics_explainability.py
===============================================
DSF504 Use Case C_markets — Market Intelligence: Realized Volatility Prediction
ML Framework Phase 6: Ethics, Bias Audit & Model Explainability

Dataset : Optiver Realized Volatility Prediction (Kaggle)
Model   : RandomForestRegressor champion (champion.pkl)
Outputs :
  reports/use_case_C_markets/shap_feature_importance.csv
  reports/use_case_C_markets/shap_bar_importance.png
  reports/use_case_C_markets/shap_beeswarm.png         (if shap installed)
  reports/use_case_C_markets/ethics_bias_report.csv
  reports/use_case_C_markets/stock_group_fairness.png

Run:
    cd C:\\DSF504
    python use_case_C_market/06_ethics_explainability.py
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

MODEL_DIR  = MODELS_DIR  / "use_case_C_markets"
REPORT_DIR = REPORTS_DIR / "use_case_C_markets"
DATA_PATH  = DATA_DIR    / "optiver_volatility"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC",
           "#26C6DA", "#EC407A", "#D4E157"]


# ── helpers ───────────────────────────────────────────────────────────────────

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    return float(np.sqrt(np.mean(((y_pred[mask] - y_true[mask]) / y_true[mask]) ** 2)))


def load_champion():
    for fname in ["champion.pkl", "lgbm_optuna_champion.pkl"]:
        p = MODEL_DIR / fname
        if p.exists():
            obj = joblib.load(p)
            if isinstance(obj, dict) and "model" in obj:
                return obj["model"], fname
            return obj, fname
    return None, None


# ── 1. Load model + validation data ──────────────────────────────────────────

def load_data():
    log.info("Loading champion model ...")
    model, mname = load_champion()
    if model is None:
        raise FileNotFoundError("No champion pkl found in models/use_case_C_markets/. "
                                "Run Steps 4-5 first.")
    log.info("  Loaded: %s  (%s)", type(model).__name__, mname)

    # Feature columns
    fc_path = MODEL_DIR / "feature_cols.pkl"
    if fc_path.exists():
        fe_cols = joblib.load(fc_path)
    elif hasattr(model, "feature_names_in_"):
        fe_cols = list(model.feature_names_in_)
    else:
        raise FileNotFoundError("feature_cols.pkl not found. Re-run rescue script.")
    log.info("  Features: %d", len(fe_cols))

    log.info("Loading validation data ...")
    val_path = DATA_PATH / "val_fe.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"val_fe.parquet not found at {val_path}")
    df_val = pd.read_parquet(val_path)
    log.info("  val_fe shape: %s", df_val.shape)

    for c in fe_cols:
        if c not in df_val.columns:
            df_val[c] = 0.0

    X_val   = df_val[fe_cols].fillna(0).values
    y_val   = df_val["target"].values.astype(float)

    return model, fe_cols, df_val, X_val, y_val


# ── 2. Feature importance (SHAP preferred, native fallback) ──────────────────

def compute_feature_importance(model, fe_cols: list, X_val: np.ndarray,
                                 n_background: int = 500) -> pd.DataFrame:
    """Return DataFrame with columns [feature, mean_abs_shap]."""
    try:
        import shap
        log.info("SHAP available — computing TreeExplainer values ...")
        rng = np.random.default_rng(RANDOM_STATE)
        bg_idx = rng.choice(len(X_val), size=min(n_background, len(X_val)), replace=False)
        bg = X_val[bg_idx]
        explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        shap_vals = explainer.shap_values(X_val[:2000])
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        method    = "SHAP"
    except ImportError:
        log.warning("shap not installed — using native feature_importances_ as proxy.")
        if hasattr(model, "feature_importances_"):
            mean_abs = model.feature_importances_
        else:
            mean_abs = np.ones(len(fe_cols)) / len(fe_cols)
        method = "native_importance"

    df_fi = pd.DataFrame({"feature": fe_cols, "mean_abs_shap": mean_abs})
    df_fi = df_fi.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    log.info("  Method: %s  Top feature: %s (%.4f)",
             method, df_fi.iloc[0]["feature"], df_fi.iloc[0]["mean_abs_shap"])
    return df_fi


# ── 3. Plots ──────────────────────────────────────────────────────────────────

def plot_bar_importance(df_fi: pd.DataFrame, top_n: int = 20) -> None:
    df_top = df_fi.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_top["feature"][::-1], df_top["mean_abs_shap"][::-1],
            color=PALETTE[0], edgecolor="none")
    ax.set_xlabel("Mean |SHAP| / Feature Importance", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importance — C_markets Champion", fontsize=13)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    out = REPORT_DIR / "shap_bar_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_beeswarm_proxy(model, fe_cols: list, X_val: np.ndarray) -> None:
    """Beeswarm if SHAP installed; else scatter of importance vs feature value range."""
    try:
        import shap
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_val), size=min(500, len(X_val)), replace=False)
        explainer = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_val[idx])
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_vals, X_val[idx],
                          feature_names=fe_cols, show=False, max_display=20)
        plt.tight_layout()
        out = REPORT_DIR / "shap_beeswarm.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("Saved SHAP beeswarm: %s", out)
    except ImportError:
        # Proxy: scatter of std(feature) vs feature importance
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:
            imp = np.ones(len(fe_cols))
        std_vals = X_val.std(axis=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(std_vals, imp, alpha=0.6, color=PALETTE[0], s=40)
        for i, name in enumerate(fe_cols):
            if imp[i] > np.percentile(imp, 85):
                ax.annotate(name, (std_vals[i], imp[i]), fontsize=7, alpha=0.8)
        ax.set_xlabel("Feature Std Dev (proxy for variability)", fontsize=10)
        ax.set_ylabel("Feature Importance", fontsize=10)
        ax.set_title("Feature Variability vs Importance — C_markets", fontsize=12)
        plt.tight_layout()
        out = REPORT_DIR / "shap_beeswarm.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved importance scatter (shap proxy): %s", out)


# ── 4. Group fairness / bias audit (stock-level) ─────────────────────────────

def bias_audit(model, fe_cols: list, df_val: pd.DataFrame,
               X_val: np.ndarray, y_val: np.ndarray) -> None:
    """Per-stock-group RMSPE to check prediction bias across stocks."""
    log.info("Running stock-group bias audit ...")
    if "stock_id" not in df_val.columns:
        log.warning("stock_id not in val_fe — skipping bias audit.")
        return

    y_pred_log  = model.predict(X_val)
    y_pred_orig = np.expm1(np.clip(y_pred_log, -10, 10))
    y_true_orig = np.clip(y_val, 1e-9, None)

    df_audit = df_val[["stock_id"]].copy()
    df_audit["y_true"] = y_true_orig
    df_audit["y_pred"] = y_pred_orig
    df_audit["sq_rel_err"] = ((df_audit["y_pred"] - df_audit["y_true"]) /
                               df_audit["y_true"]) ** 2

    grp = (df_audit.groupby("stock_id")
           .agg(rmspe_=("sq_rel_err", lambda x: float(np.sqrt(x.mean()))),
                count=("y_true", "count"),
                mean_rv=("y_true", "mean"))
           .reset_index()
           .rename(columns={"rmspe_": "rmspe"})
           .sort_values("rmspe"))

    out_csv = REPORT_DIR / "ethics_bias_report.csv"
    grp.to_csv(out_csv, index=False)
    log.info("Saved bias report: %s  (%d stocks)", out_csv, len(grp))

    # Plot: per-stock RMSPE distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(grp["stock_id"].astype(str), grp["rmspe"], color=PALETTE[0], alpha=0.75)
    ax.axhline(grp["rmspe"].mean(), color=PALETTE[3], linestyle="--",
               linewidth=1.5, label=f"Mean RMSPE = {grp['rmspe'].mean():.5f}")
    ax.set_xlabel("Stock ID", fontsize=10)
    ax.set_ylabel("RMSPE", fontsize=10)
    ax.set_title("Per-Stock RMSPE — Fairness Audit (C_markets)", fontsize=12)
    ax.legend(fontsize=9)
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    out_png = REPORT_DIR / "stock_group_fairness.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved fairness plot: %s", out_png)

    # Summary stats
    rmspe_all = rmspe(y_true_orig, y_pred_orig)
    log.info("  Overall RMSPE: %.5f", rmspe_all)
    log.info("  Best  stock RMSPE: %.5f (stock %s)",
             grp["rmspe"].min(), grp.iloc[0]["stock_id"])
    log.info("  Worst stock RMSPE: %.5f (stock %s)",
             grp["rmspe"].max(), grp.iloc[-1]["stock_id"])
    log.info("  Disparity ratio (worst/best): %.2fx",
             grp["rmspe"].max() / max(grp["rmspe"].min(), 1e-9))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 62)
    log.info("  Step 6: Ethics & Explainability — C_markets")
    log.info("=" * 62)

    model, fe_cols, df_val, X_val, y_val = load_data()

    # Feature importance
    df_fi = compute_feature_importance(model, fe_cols, X_val)
    out_fi = REPORT_DIR / "shap_feature_importance.csv"
    df_fi.to_csv(out_fi, index=False)
    log.info("Saved shap_feature_importance.csv  (%d features)", len(df_fi))

    # Plots
    plot_bar_importance(df_fi)
    plot_beeswarm_proxy(model, fe_cols, X_val)

    # Bias audit
    bias_audit(model, fe_cols, df_val, X_val, y_val)

    log.info("=" * 62)
    log.info("  Step 6 complete — all outputs saved to %s", REPORT_DIR)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
