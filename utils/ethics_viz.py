"""
utils/ethics_viz.py
===================
Shared visualization helpers for Phase 6: Ethics, Bias Audit & Explainability.

All plot functions:
  - save a PNG to report_dir
  - return a one-sentence insight string
  - annotate the chart with that insight in a text box

Usage:
    from utils.ethics_viz import (
        plot_calibration_curve,
        plot_threshold_sensitivity,
        plot_probability_distribution,
        plot_shap_dependence,
        plot_confusion_matrix_eth,
        plot_fairness_bars,
        save_insights_txt,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

log = logging.getLogger(__name__)

PALETTE = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC",
           "#26C6DA", "#EC407A", "#D4E157"]

# ── internal helper ───────────────────────────────────────────────────────────

def _insight_box(ax, text: str, fontsize: int = 8) -> None:
    """Overlay a light-yellow insight text box at the bottom of the axes."""
    ax.text(
        0.01, 0.02, f"[Insight] {text}",
        transform=ax.transAxes,
        fontsize=fontsize, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7",
                  edgecolor="#F9A825", alpha=0.9),
        wrap=True,
        zorder=10,
    )


def _fig_insight_box(fig, text: str, fontsize: int = 8) -> None:
    """Overlay a figure-level insight box (for multi-axes figures)."""
    fig.text(
        0.01, 0.01, f"[Insight] {text}",
        fontsize=fontsize, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7",
                  edgecolor="#F9A825", alpha=0.9),
    )


# ── 1. Calibration curve ──────────────────────────────────────────────────────

def plot_calibration_curve(
    model, X_val: np.ndarray, y_val: np.ndarray,
    report_dir: Path, title_suffix: str = "",
    n_bins: int = 10,
) -> str:
    """Reliability diagram: predicted probability vs fraction of positives."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:, 1]
    else:
        probs = np.clip(model.predict(X_val), 0.0, 1.0)

    fraction_pos, mean_pred = calibration_curve(y_val, probs, n_bins=n_bins)
    brier = brier_score_loss(y_val, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax.plot(mean_pred, fraction_pos, "o-", color=PALETTE[0], linewidth=2,
            markersize=6, label=f"Model (Brier={brier:.4f})")
    ax.fill_between(mean_pred, fraction_pos, mean_pred,
                    alpha=0.12, color=PALETTE[3], label="Calibration gap")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title(f"Calibration Curve (Reliability Diagram){title_suffix}", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Insight
    gap = float(np.abs(fraction_pos - mean_pred).mean())
    if brier < 0.05:
        insight = (f"Brier score {brier:.4f} — excellent calibration; "
                   "predicted probabilities closely match observed positive rates.")
    elif brier < 0.15:
        insight = (f"Brier score {brier:.4f} — good calibration (avg gap {gap:.3f}); "
                   "model probabilities are reasonably reliable for threshold decisions.")
    else:
        insight = (f"Brier score {brier:.4f} — moderate calibration gap ({gap:.3f}); "
                   "consider Platt scaling or isotonic regression to improve probability reliability.")

    _insight_box(ax, insight)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = report_dir / "calibration_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved calibration_curve.png  (Brier=%.4f)", brier)
    return insight


# ── 2. Threshold sensitivity ──────────────────────────────────────────────────

def plot_threshold_sensitivity(
    model, X_val: np.ndarray, y_val: np.ndarray,
    report_dir: Path, title_suffix: str = "",
) -> str:
    """Precision, Recall, F1, FPR vs classification threshold."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:, 1]
    else:
        probs = np.clip(model.predict(X_val), 0.0, 1.0)

    thresholds = np.linspace(0.01, 0.99, 99)
    precision_arr, recall_arr, f1_arr, fpr_arr = [], [], [], []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        tp = int(((preds == 1) & (y_val == 1)).sum())
        fp = int(((preds == 1) & (y_val == 0)).sum())
        fn = int(((preds == 0) & (y_val == 1)).sum())
        tn = int(((preds == 0) & (y_val == 0)).sum())
        prec  = tp / max(tp + fp, 1)
        rec   = tp / max(tp + fn, 1)
        f1    = 2 * prec * rec / max(prec + rec, 1e-9)
        fpr   = fp / max(fp + tn, 1)
        precision_arr.append(prec); recall_arr.append(rec)
        f1_arr.append(f1); fpr_arr.append(fpr)

    best_idx = int(np.argmax(f1_arr))
    best_thr = thresholds[best_idx]
    best_f1  = f1_arr[best_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precision_arr, label="Precision", color=PALETTE[0], linewidth=2)
    ax.plot(thresholds, recall_arr,    label="Recall",    color=PALETTE[1], linewidth=2)
    ax.plot(thresholds, f1_arr,        label="F1 Score",  color=PALETTE[2], linewidth=2)
    ax.plot(thresholds, fpr_arr,       label="FPR",       color=PALETTE[3], linewidth=2, linestyle="--")
    ax.axvline(best_thr, color="black", linestyle=":", linewidth=1.5,
               label=f"Best F1 threshold = {best_thr:.2f}")
    ax.set_xlabel("Classification Threshold", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title(f"Threshold Sensitivity{title_suffix}", fontsize=12)
    ax.legend(fontsize=9, loc="center right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    insight = (f"Best F1 = {best_f1:.3f} at threshold {best_thr:.2f}. "
               "Increasing the threshold reduces false alarms (FPR) but misses more positives (Recall drops). "
               "Choose threshold based on business cost of FP vs FN.")

    _insight_box(ax, insight, fontsize=7)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    out = report_dir / "threshold_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved threshold_sensitivity.png  (best F1=%.3f @ thr=%.2f)", best_f1, best_thr)
    return insight


# ── 3. Predicted probability distribution ────────────────────────────────────

def plot_probability_distribution(
    model, X_val: np.ndarray, y_val: np.ndarray,
    report_dir: Path, title_suffix: str = "",
    class_labels: tuple[str, str] = ("Negative", "Positive"),
) -> str:
    """Histogram of predicted probabilities split by true class."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:, 1]
    else:
        probs = np.clip(model.predict(X_val), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, (label, color) in enumerate(zip(class_labels, [PALETTE[0], PALETTE[3]])):
        mask = y_val == cls
        ax.hist(probs[mask], bins=40, alpha=0.6, color=color,
                label=f"{label} (n={mask.sum():,})", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Default threshold (0.5)")
    ax.set_xlabel("Predicted Probability (positive class)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Predicted Probability Distribution by True Class{title_suffix}", fontsize=12)
    ax.legend(fontsize=9)

    # Insight: overlap / separation quality
    pos_probs = probs[y_val == 1]
    neg_probs = probs[y_val == 0]
    sep = float(pos_probs.mean() - neg_probs.mean()) if len(pos_probs) > 0 else 0.0
    insight = (f"Mean predicted probability: positives={pos_probs.mean():.3f}, "
               f"negatives={neg_probs.mean():.3f}. "
               f"Separation gap = {sep:.3f}. "
               + ("Strong class separation — model discriminates well."
                  if sep > 0.3 else
                  "Moderate separation — some overlap between classes near the decision boundary."))

    _insight_box(ax, insight, fontsize=7)
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    out = report_dir / "probability_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved probability_distribution.png  (sep=%.3f)", sep)
    return insight


# ── 4. SHAP dependence plots ──────────────────────────────────────────────────

def plot_shap_dependence(
    model, fe_cols: list[str], X_val: np.ndarray,
    df_fi: pd.DataFrame, report_dir: Path,
    title_suffix: str = "", n_features: int = 3,
) -> str:
    """Partial-dependence-style SHAP dependence for top N features."""
    top_feats = df_fi["feature"].head(n_features).tolist()
    top_idxs  = [list(fe_cols).index(f) for f in top_feats if f in fe_cols]
    if not top_idxs:
        log.warning("No matching SHAP features found — skipping dependence plot")
        return "SHAP dependence plot skipped (no matching features)."

    try:
        import shap
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_val), size=min(300, len(X_val)), replace=False)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val[idx])
        if isinstance(sv, list):
            sv = sv[1]
        use_shap = True
    except (ImportError, Exception):
        sv = None
        use_shap = False

    n = len(top_idxs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    insights = []
    for ax, fi in zip(axes, top_idxs):
        feat_name = fe_cols[fi]
        feat_vals = X_val[idx, fi] if use_shap else X_val[:, fi]
        if use_shap:
            shap_vals = sv[:, fi]
            ax.scatter(feat_vals, shap_vals, alpha=0.3, s=8,
                       c=feat_vals, cmap="coolwarm", edgecolors="none")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_ylabel("SHAP value", fontsize=9)
            corr = float(np.corrcoef(feat_vals, shap_vals)[0, 1])
            direction = "positive" if corr > 0.1 else ("negative" if corr < -0.1 else "non-linear")
            insights.append(f"{feat_name}: {direction} SHAP trend (r={corr:.2f})")
        else:
            # Fallback: feature distribution
            ax.hist(feat_vals, bins=30, color=PALETTE[0], alpha=0.7)
            ax.set_ylabel("Count", fontsize=9)
            insights.append(f"{feat_name}: distribution shown (SHAP unavailable)")
        ax.set_xlabel(feat_name, fontsize=9)
        ax.set_title(f"SHAP Dependence\n{feat_name}", fontsize=10)

    insight_str = " | ".join(insights)
    fig.suptitle(f"SHAP Dependence — Top {n} Features{title_suffix}", fontsize=12)
    _fig_insight_box(fig, insight_str, fontsize=7)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    out = report_dir / "shap_dependence_top3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_dependence_top3.png")
    return insight_str


# ── 5. Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix_eth(
    model, X_val: np.ndarray, y_val: np.ndarray,
    report_dir: Path, title_suffix: str = "",
    threshold: float = 0.5,
    class_labels: tuple[str, str] = ("Negative", "Positive"),
    filename: str = "ethics_confusion_matrix.png",
) -> str:
    """Annotated confusion matrix with ethics framing."""
    from sklearn.metrics import confusion_matrix

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= threshold).astype(int)
    else:
        preds = model.predict(X_val).astype(int)

    cm = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    acc = (tp + tn) / max(len(y_val), 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks); ax.set_yticks(tick_marks)
    ax.set_xticklabels([f"Pred {l}" for l in class_labels])
    ax.set_yticklabels([f"True {l}" for l in class_labels])
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Confusion Matrix{title_suffix}\n"
                 f"Acc={acc:.3f}  FPR={fpr:.3f}  FNR={fnr:.3f}", fontsize=11)

    thresh_color = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh_color else "black")

    insight = (f"FPR={fpr:.3f} (false alarms), FNR={fnr:.3f} (missed positives). "
               + ("FNR dominates — model misses many true positives; consider lowering threshold."
                  if fnr > fpr * 1.5 else
                  "FPR dominates — model raises too many false alarms; consider raising threshold."
                  if fpr > fnr * 1.5 else
                  "FPR and FNR are balanced — adjust threshold based on business priorities."))

    _insight_box(ax, insight, fontsize=7)
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    out = report_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s  (FPR=%.3f, FNR=%.3f)", filename, fpr, fnr)
    return insight


# ── 6. Fairness bar chart ─────────────────────────────────────────────────────

def plot_fairness_bars(
    bias_df: pd.DataFrame, report_dir: Path,
    title_suffix: str = "",
    rate_col: str = "fpr",
    rate_label: str = "False Positive Rate",
    filename: str = "ethics_fairness_bars.png",
) -> str:
    """Bar chart of FPR / positive rate across demographic or group segments."""
    if bias_df.empty or rate_col not in bias_df.columns:
        log.warning("Bias DF empty or missing column '%s' — skipping fairness bar chart", rate_col)
        return "Fairness chart skipped — insufficient data."

    attrs = bias_df["attribute"].unique() if "attribute" in bias_df.columns else ["group"]
    n = min(len(attrs), 4)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    disparities = []
    for ax, attr in zip(axes, attrs[:4]):
        sub = bias_df[bias_df["attribute"] == attr] if "attribute" in bias_df.columns else bias_df
        groups = sub["group"].tolist() if "group" in sub.columns else sub.index.tolist()
        rates  = sub[rate_col].tolist()
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(groups))]
        ax.bar(groups, rates, color=colors, alpha=0.85)
        ax.set_title(str(attr).replace("fe_", "").replace("_", " ").title(), fontsize=10)
        ax.set_ylabel(rate_label)
        ax.tick_params(axis="x", rotation=15)
        if len(rates) >= 2:
            disp = max(rates) - min(rates)
            disparities.append(disp)
            ax.annotate(f"Δ={disp:.3f}", xy=(0.98, 0.95),
                        xycoords="axes fraction", ha="right", fontsize=9,
                        color=PALETTE[3] if disp > 0.05 else PALETTE[1])

    avg_disp = float(np.mean(disparities)) if disparities else 0.0
    fig.suptitle(f"Fairness Audit — {rate_label} by Group{title_suffix}", fontsize=12, fontweight="bold")

    insight = (f"Average group disparity in {rate_label}: {avg_disp:.3f}. "
               + ("Disparity > 0.05 — fairness intervention may be warranted (e.g. re-weighting, threshold adjustment per group)."
                  if avg_disp > 0.05 else
                  "Disparity within acceptable range (< 0.05) — model treats groups comparably."))

    _fig_insight_box(fig, insight, fontsize=7)
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    out = report_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s  (avg disparity=%.3f)", filename, avg_disp)
    return insight


# ── 7. Insights summary text ──────────────────────────────────────────────────

def save_insights_txt(
    insights: dict[str, str], report_dir: Path, uc_name: str = ""
) -> None:
    """Write a plain-text insights summary file."""
    lines = [
        f"=" * 70,
        f"  Ethics & Explainability Insights — {uc_name}",
        f"=" * 70,
        "",
    ]
    for chart, text in insights.items():
        lines.append(f"[{chart}]")
        # Wrap long lines
        words = text.split()
        line, wrapped = [], []
        for w in words:
            if len(" ".join(line + [w])) > 78:
                wrapped.append("  " + " ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            wrapped.append("  " + " ".join(line))
        lines.extend(wrapped)
        lines.append("")

    lines += ["=" * 70, ""]
    out = report_dir / "ethics_insights.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved ethics_insights.txt  (%d charts)", len(insights))
