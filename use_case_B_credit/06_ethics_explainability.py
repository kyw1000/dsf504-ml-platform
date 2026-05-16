"""
use_case_B_credit/06_ethics_explainability.py
===========================================
DSF504 Use Case B — Credit Risk Scoring
ML Framework Phase 6: Ethics, Bias Audit & Model Explainability

Outputs saved to reports/use_case_B/
Run:
    cd C:\\DSF504
    python use_case_B_credit/06_ethics_explainability.py
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODEL_DIR  = MODELS_DIR  / "use_case_B"
REPORT_DIR = REPORTS_DIR / "use_case_B"
DATA_PATH  = DATA_DIR    / "gmsc_credit"
TARGET     = "SeriousDlqin2yrs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = ["#42A5F5","#66BB6A","#FFA726","#EF5350","#AB47BC"]


def load_champion():
    for fname in ["lgbm_optuna_champion.pkl", "champion.pkl"]:
        p = MODEL_DIR / fname
        if p.exists():
            obj = joblib.load(p)
            return (obj["model"], fname) if isinstance(obj, dict) and "model" in obj else (obj, fname)
    return None, None


def main():
    log.info("Step 6: Ethics & Explainability — B — Credit Risk Scoring")

    model, mname = load_champion()
    if model is None:
        raise FileNotFoundError("No champion pkl. Run Steps 4-5 first.")
    log.info("  Loaded %s → %s", mname, type(model).__name__)

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
    df_val = pd.read_parquet(val_path)
    for c in fe_cols:
        if c not in df_val.columns:
            df_val[c] = 0.0
    X_val = df_val[fe_cols].fillna(0).values
    y_val = df_val[TARGET].values

    # ── Feature importance ────────────────────────────────────────────────────
    try:
        import shap
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_val), size=min(1000, len(X_val)), replace=False)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val[idx])
        if isinstance(sv, list): sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        log.info("  SHAP computed")
    except ImportError:
        log.warning("shap not installed — using feature_importances_")
        mean_abs = getattr(model, "feature_importances_", np.ones(len(fe_cols)))

    df_fi = pd.DataFrame({"feature": fe_cols, "mean_abs_shap": mean_abs})
    df_fi = df_fi.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df_fi.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
    log.info("Saved shap_feature_importance.csv (%d features)", len(df_fi))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    top = df_fi.head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color=PALETTE[0], edgecolor="none")
    ax.set_xlabel("Mean |SHAP| / Feature Importance")
    ax.set_title("Top 20 Feature Importance — B — Credit Risk Scoring")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_bar_importance.png")

    # ── Beeswarm / proxy ──────────────────────────────────────────────────────
    try:
        import shap
        rng2 = np.random.default_rng(RANDOM_STATE + 1)
        idx2 = rng2.choice(len(X_val), size=min(500, len(X_val)), replace=False)
        sv2 = shap.TreeExplainer(model).shap_values(X_val[idx2])
        if isinstance(sv2, list): sv2 = sv2[1]
        shap.summary_plot(sv2, X_val[idx2], feature_names=fe_cols, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    except ImportError:
        std = X_val.std(axis=0)
        imp = getattr(model, "feature_importances_", np.ones(len(fe_cols)))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(std, imp, alpha=0.6, color=PALETTE[0], s=40)
        ax.set_xlabel("Feature Std Dev"); ax.set_ylabel("Feature Importance")
        ax.set_title("Variability vs Importance — B — Credit Risk Scoring")
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved shap_beeswarm.png")

    # ── Bias audit ────────────────────────────────────────────────────────────
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:, 1]
    else:
        probs = model.predict(X_val)
    thr_path = MODEL_DIR / "lgbm_optimal_threshold.txt"
    thr = float(thr_path.read_text().strip()) if thr_path.exists() else 0.5
    preds = (probs >= thr).astype(int)
    pd.DataFrame({
        "metric": ["threshold","accuracy","precision","recall","fpr","fnr"],
        "value": [
            thr,
            float((preds == y_val).mean()),
            float((preds & y_val).sum() / max(preds.sum(), 1)),
            float((preds & y_val).sum() / max(y_val.sum(), 1)),
            float(((preds==1) & (y_val==0)).mean()),
            float(((preds==0) & (y_val==1)).mean()),
        ]
    }).to_csv(REPORT_DIR / "ethics_bias_report.csv", index=False)
    log.info("Saved ethics_bias_report.csv  (threshold=%.3f)", thr)
    log.info("Step 6 complete — outputs in %s", REPORT_DIR)


if __name__ == "__main__":
    main()
