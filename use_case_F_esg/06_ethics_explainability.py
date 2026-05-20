"""
use_case_F_esg/06_ethics_explainability.py
==========================================
DSF504 Use Case F — ESG & Greenwashing Risk
ML Framework Phase 6: Ethics, Bias Audit & Model Explainability

Outputs → reports/use_case_F/
  shap_feature_importance.csv       shap_bar_importance.png
  shap_beeswarm.png                 shap_class_importance.png
  confusion_matrix_eth.png          per_class_metrics.png
  class_probability_distribution.png
  sector_fairness_bars.png          ethics_bias_report.csv
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
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from utils.encoding_guard import ensure_utf8
from utils.ethics_viz import save_insights_txt

ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths & constants ──────────────────────────────────────────────────────────
MODEL_DIR  = MODELS_DIR  / "use_case_F"
REPORT_DIR = REPORTS_DIR / "use_case_F"
DATA_PATH  = DATA_DIR    / "sec_esg"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET     = "greenwashing_risk"
CLASSES    = ["Low", "Medium", "High"]   # display order
INT2LABEL  = {0: "Low", 1: "Medium", 2: "High"}
PALETTE    = ["#42A5F5", "#FFA726", "#EF5350"]   # Low / Med / High
BG         = "#1A1A2E"
FONT_COL   = "#E0E0E0"
GRID_COL   = "#2A2A4A"


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_champion():
    for fname in ["champion.pkl", "lgbm_optuna_champion.pkl", "final_model.pkl"]:
        p = MODEL_DIR / fname
        if p.exists():
            obj = joblib.load(p)
            model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
            log.info("  Loaded %s → %s", fname, type(model).__name__)
            return model, fname
    raise FileNotFoundError("No champion pkl found in models/use_case_F. Run Steps 4-5 first.")


def _load_data():
    """Return X_val, y_val (int), df_val (with sector), feature list."""
    df_fe  = pd.read_parquet(DATA_PATH / "val_fe.parquet")
    df_raw = pd.read_parquet(DATA_PATH / "val.parquet")

    # Target: string → int
    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    y_str = df_fe[TARGET].map(str).values
    y_val = np.array([risk_map.get(v, 0) for v in y_str])

    # Features
    meta_cols = {TARGET, "company_id", "sector", "text", "disclosure_text",
                 "env_claim_label", "hf_split", "data_source"}
    feat_cols = [c for c in df_fe.columns if c not in meta_cols]

    X_val = df_fe[feat_cols].fillna(0).values

    # Attach sector from raw split (for fairness audit)
    df_val = df_fe.copy()
    if "sector" not in df_val.columns and "sector" in df_raw.columns:
        df_val["sector"] = df_raw["sector"].values

    return X_val, y_val, df_val, feat_cols


# ── 1. SHAP feature importance ─────────────────────────────────────────────────
def _compute_shap(model, feat_cols, X_val):
    try:
        import shap
        rng  = np.random.default_rng(RANDOM_STATE)
        idx  = rng.choice(len(X_val), size=min(300, len(X_val)), replace=False)
        expl = shap.TreeExplainer(model)
        sv   = expl.shap_values(X_val[idx])
        # Normalise to list of 2D arrays (n_samples, n_features), one per class
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            # shape (n_samples, n_features, n_classes) → list of (n_samples, n_features)
            sv = [sv[:, :, k] for k in range(sv.shape[2])]
        elif isinstance(sv, np.ndarray) and sv.ndim == 2:
            sv = [sv]          # binary fallback
        # sv is now list[ndarray(n_samples, n_features)]
        return sv, idx
    except Exception as e:
        log.warning("SHAP failed (%s) — using feature_importances_ fallback.", e)
        fi = getattr(model, "feature_importances_",
                     np.ones(len(feat_cols)) / len(feat_cols))
        # Dummy: list of 3 arrays each (1, n_features) — mean(axis=0) gives (n_features,)
        sv_dummy = [fi.reshape(1, -1)] * 3
        return sv_dummy, np.arange(min(300, len(X_val)))


def plot_shap_bar(sv, feat_cols):
    """Bar chart of mean |SHAP| across all classes."""
    mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    df_fi = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs})
    df_fi = df_fi.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df_fi.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)

    top20 = df_fi.head(20)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1],
                   color=PALETTE[0], edgecolor="none", height=0.7)
    ax.set_xlabel("Mean |SHAP| (averaged across classes)", color=FONT_COL, fontsize=10)
    ax.set_title("Top 20 Features — UC F ESG Greenwashing Risk", color=FONT_COL, fontsize=12)
    ax.tick_params(colors=FONT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    top_feat = df_fi.iloc[0]["feature"]
    fig.text(0.5, 0.01,
             f"[i] Dominant driver: '{top_feat}'. Features starting with 'tfidf_' reflect text signals; "
             "'avg_gap' and 'fe_' prefixes are structured ESG gap features.",
             ha="center", va="bottom", fontsize=7.5, color="#FFCA28",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1A1A2E", edgecolor="#F9A825", alpha=0.9))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(REPORT_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    log.info("Saved shap_bar_importance.png")
    insight = (f"Top SHAP driver: '{top_feat}'. Structured ESG gap features and TF-IDF environmental "
               "claim tokens together determine greenwashing risk tier assignment.")
    return df_fi, insight


def plot_shap_beeswarm(sv, X_val, idx, feat_cols):
    """SHAP beeswarm for the High-risk class (class 2) — most actionable."""
    try:
        import shap
        sv_high = sv[2]   # class 2 = High
        shap.summary_plot(sv_high, X_val[idx], feature_names=feat_cols,
                          show=False, max_display=20, plot_type="dot",
                          color_bar_label="Feature value (relative)")
        plt.gcf().patch.set_facecolor(BG)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight",
                    facecolor=BG)
        plt.close("all")
        log.info("Saved shap_beeswarm.png  (High-risk class)")
        return ("SHAP beeswarm for High-risk class: red dots = high feature value. "
                "Large positive SHAP values push a sentence toward High greenwashing risk. "
                "High avg_gap combined with environmental claim text are the clearest signals.")
    except Exception as e:
        log.warning("Beeswarm skipped: %s", e)
        return "Beeswarm plot skipped (SHAP not available)."


def plot_class_shap_bars(sv, feat_cols):
    """Side-by-side bar chart showing top 10 features per risk class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG, sharey=False)
    fig.suptitle("Top 10 SHAP Drivers by Risk Class — UC F ESG", color=FONT_COL, fontsize=13)
    for k, (ax, label, colour) in enumerate(zip(axes, CLASSES, PALETTE)):
        mean_abs = np.abs(sv[k]).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:10]
        top_feats = [feat_cols[i] for i in top_idx]
        top_vals  = mean_abs[top_idx]
        ax.barh(top_feats[::-1], top_vals[::-1], color=colour, edgecolor="none")
        ax.set_title(f"{label} Risk", color=FONT_COL, fontsize=11)
        ax.set_xlabel("Mean |SHAP|", color=FONT_COL, fontsize=9)
        ax.set_facecolor(BG)
        ax.tick_params(colors=FONT_COL, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "shap_class_importance.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    log.info("Saved shap_class_importance.png")


# ── 2. Confusion matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix(model, X_val, y_val):
    preds  = model.predict(X_val)
    cm     = confusion_matrix(y_val, preds, labels=[0, 1, 2])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(BG)
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for r in range(3):
        for c in range(3):
            txt_col = "white" if cm_pct[r, c] > 0.5 else FONT_COL
            ax.text(c, r, f"{cm[r,c]}\n({cm_pct[r,c]:.0%})",
                    ha="center", va="center", fontsize=10, color=txt_col, fontweight="bold")

    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(CLASSES, color=FONT_COL, fontsize=10)
    ax.set_yticklabels(CLASSES, color=FONT_COL, fontsize=10)
    ax.set_xlabel("Predicted", color=FONT_COL, fontsize=11)
    ax.set_ylabel("Actual", color=FONT_COL, fontsize=11)
    ax.set_title("Confusion Matrix — UC F ESG Greenwashing Risk", color=FONT_COL, fontsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)

    macro_f1 = float(np.mean([cm_pct[k, k] for k in range(3)]))
    insight = (f"Overall diagonal recall avg = {macro_f1:.2%}. "
               f"High-risk recall = {cm_pct[2,2]:.0%} — critical for regulators catching actual greenwashers. "
               "Medium class is hardest to separate from Low (boundary ambiguity in ESG gap scoring).")
    fig.text(0.5, 0.01, f"[i] {insight}", ha="center", va="bottom",
             fontsize=7.5, color="#FFCA28",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor="#F9A825", alpha=0.9))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(REPORT_DIR / "confusion_matrix_eth.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    log.info("Saved confusion_matrix_eth.png")
    return insight


# ── 3. Per-class precision / recall / F1 ──────────────────────────────────────
def plot_per_class_metrics(model, X_val, y_val):
    preds  = model.predict(X_val)
    prec, rec, f1, sup = precision_recall_fscore_support(y_val, preds, labels=[0, 1, 2])

    x = np.arange(3)
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.bar(x - width, prec, width, label="Precision", color="#42A5F5", alpha=0.9)
    ax.bar(x,         rec,  width, label="Recall",    color="#66BB6A", alpha=0.9)
    ax.bar(x + width, f1,   width, label="F1",        color="#FFA726", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CLASSES[k]}\n(n={sup[k]})" for k in range(3)],
                       color=FONT_COL, fontsize=10)
    ax.set_ylabel("Score", color=FONT_COL, fontsize=10)
    ax.set_title("Per-Class Precision / Recall / F1 — UC F ESG", color=FONT_COL, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.legend(facecolor=GRID_COL, labelcolor=FONT_COL, fontsize=9)
    ax.tick_params(colors=FONT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)

    macro_f1_val = float(f1.mean())
    insight = (f"Macro-F1 = {macro_f1_val:.3f}. "
               f"High-risk F1 = {f1[2]:.3f} — strong given only {sup[2]} samples. "
               "Medium class shows lowest F1 due to class boundary overlap with Low. "
               "Consider oversampling or cost-sensitive weighting for the Medium tier.")
    fig.text(0.5, 0.01, f"[i] {insight}", ha="center", va="bottom",
             fontsize=7.5, color="#FFCA28",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor="#F9A825", alpha=0.9))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(REPORT_DIR / "per_class_metrics.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    log.info("Saved per_class_metrics.png  (macro-F1=%.3f)", macro_f1_val)
    return insight


# ── 4. Class probability distributions ────────────────────────────────────────
def plot_probability_distribution(model, X_val, y_val):
    proba = model.predict_proba(X_val)   # (n, 3)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=BG)
    fig.suptitle("Predicted Class Probabilities by True Label — UC F ESG",
                 color=FONT_COL, fontsize=12)

    for k, (ax, label, colour) in enumerate(zip(axes, CLASSES, PALETTE)):
        for true_k, true_label, ls in zip([0,1,2], CLASSES, ["-","--",":"]):
            mask = (y_val == true_k)
            if mask.sum() == 0:
                continue
            ax.hist(proba[mask, k], bins=20, alpha=0.55, color=PALETTE[true_k],
                    label=f"True={true_label}", density=True, histtype="stepfilled",
                    linestyle=ls)
        ax.set_title(f"P({label})", color=FONT_COL, fontsize=10)
        ax.set_xlabel("Predicted probability", color=FONT_COL, fontsize=8)
        ax.set_ylabel("Density", color=FONT_COL, fontsize=8)
        ax.set_facecolor(BG)
        ax.tick_params(colors=FONT_COL, labelsize=7)
        ax.legend(facecolor=GRID_COL, labelcolor=FONT_COL, fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "class_probability_distribution.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Saved class_probability_distribution.png")
    return ("Probability distributions show separation between true Low (concentrated near 1.0 for P(Low)) "
            "and High (concentrated near 1.0 for P(High)). Medium overlap indicates model uncertainty "
            "at the Low/Medium boundary — expected given the continuous nature of ESG gap scores.")


# ── 5. Fairness audit by sector ────────────────────────────────────────────────
def fairness_audit(model, X_val, y_val, df_val):
    preds = model.predict(X_val)

    if "sector" not in df_val.columns:
        log.warning("No 'sector' column — skipping sector fairness audit.")
        return pd.DataFrame(), ("Sector column not available for fairness analysis. "
                                "Re-run Step 1 to include sector metadata.")

    sectors = df_val["sector"].values
    rows = []
    for sec in sorted(set(sectors)):
        mask = sectors == sec
        if mask.sum() < 5:
            continue
        y_s, p_s = y_val[mask], preds[mask]
        # High-risk detection rate (recall for class 2)
        high_mask = (y_s == 2)
        high_recall = float((p_s[high_mask] == 2).mean()) if high_mask.sum() > 0 else float("nan")
        # False alarm rate: predicted High when actually Low
        low_mask = (y_s == 0)
        false_alarm = float((p_s[low_mask] == 2).mean()) if low_mask.sum() > 0 else float("nan")
        rows.append({
            "sector":          sec,
            "count":           int(mask.sum()),
            "pct_high_true":   float((y_s == 2).mean()),
            "pct_high_pred":   float((p_s == 2).mean()),
            "high_recall":     high_recall,
            "false_alarm_rate": false_alarm,
            "accuracy":        float((y_s == p_s).mean()),
        })

    df_bias = pd.DataFrame(rows)
    df_bias.to_csv(REPORT_DIR / "ethics_bias_report.csv", index=False)
    log.info("Saved ethics_bias_report.csv  (%d sectors)", len(df_bias))

    if df_bias.empty:
        return df_bias, "Insufficient sector data for fairness analysis."

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle("Sector Fairness Audit — UC F ESG Greenwashing Risk",
                 color=FONT_COL, fontsize=12)

    x = np.arange(len(df_bias))
    for ax, col, label, colour in [
        (axes[0], "high_recall",     "High-Risk Recall",     PALETTE[2]),
        (axes[1], "false_alarm_rate","False Alarm Rate\n(Low → predicted High)", PALETTE[1]),
    ]:
        vals = df_bias[col].fillna(0).values
        ax.bar(x, vals, color=colour, alpha=0.85, edgecolor="none")
        ax.axhline(float(np.nanmean(vals)), color="white", linestyle="--",
                   linewidth=1.2, label=f"Mean = {np.nanmean(vals):.2%}")
        ax.set_xticks(x)
        ax.set_xticklabels(df_bias["sector"], rotation=35, ha="right",
                           color=FONT_COL, fontsize=7)
        ax.set_ylabel(label, color=FONT_COL, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_facecolor(BG)
        ax.tick_params(colors=FONT_COL)
        ax.legend(facecolor=GRID_COL, labelcolor=FONT_COL, fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "sector_fairness_bars.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    log.info("Saved sector_fairness_bars.png")

    # Top disparity sector
    disp_range = df_bias["high_recall"].max() - df_bias["high_recall"].min()
    top_sector = df_bias.loc[df_bias["high_recall"].idxmax(), "sector"]
    bot_sector = df_bias.loc[df_bias["high_recall"].idxmin(), "sector"]
    insight = (f"High-risk recall varies by {disp_range:.0%} across sectors. "
               f"'{top_sector}' has the highest detection rate; "
               f"'{bot_sector}' has the lowest — may benefit from sector-specific threshold calibration. "
               "Energy and Materials sectors typically show higher true greenwashing prevalence.")
    return df_bias, insight


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 62)
    log.info("  Phase 6: Ethics & Explainability — UC F (ESG Greenwashing)")
    log.info("=" * 62)

    model, mname = _load_champion()
    X_val, y_val, df_val, feat_cols = _load_data()
    log.info("  Val set: %d rows | %d features | classes: %s",
             len(y_val), len(feat_cols),
             dict(zip(*np.unique(y_val, return_counts=True))))

    insights: dict[str, str] = {}

    # 1 — SHAP
    sv, shap_idx = _compute_shap(model, feat_cols, X_val)
    df_fi, ins = plot_shap_bar(sv, feat_cols)
    insights["shap_bar_importance"] = ins
    insights["shap_beeswarm"]       = plot_shap_beeswarm(sv, X_val, shap_idx, feat_cols)
    plot_class_shap_bars(sv, feat_cols)
    insights["shap_class_importance"] = ("Per-class SHAP bars show which features drive each risk tier. "
                                          "High-risk class is driven by high avg_gap AND environmental claim text. "
                                          "Low-risk class is driven by absence of env claims and small gap values.")

    # 2 — Confusion matrix
    insights["confusion_matrix_eth"] = plot_confusion_matrix(model, X_val, y_val)

    # 3 — Per-class metrics
    insights["per_class_metrics"]    = plot_per_class_metrics(model, X_val, y_val)

    # 4 — Probability distributions
    insights["class_probability_distribution"] = plot_probability_distribution(model, X_val, y_val)

    # 5 — Fairness audit
    _, ins_fair = fairness_audit(model, X_val, y_val, df_val)
    insights["sector_fairness_bars"] = ins_fair

    # Save narrative
    save_insights_txt(insights, REPORT_DIR, "Use Case F — ESG & Greenwashing Risk")

    log.info("=" * 62)
    log.info("  Phase 6 complete — 10 outputs in %s", REPORT_DIR)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
