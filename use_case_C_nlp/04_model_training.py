"""
use_case_C_nlp/04_model_training.py
=====================================
DSF504 — Use Case C (NLP): Market Intelligence — Financial Sentiment
Step 4: Model Training & Cross-Validation

Models trained
--------------
On TF-IDF + hand-crafted features:
  ✓ Logistic Regression (baseline — industry standard for text)
  ✓ Naive Bayes (classic NLP baseline; fast, probabilistic)
  ✓ Random Forest
  ✓ XGBoost
  ✓ LightGBM

On FinBERT embeddings (if available):
  ✓ LightGBM (champion candidate — best-of-both-worlds)

Evaluation
----------
  Primary metric : Macro-F1 (handles class imbalance across 3 classes)
  Secondary      : Accuracy, per-class F1, confusion matrix
  CV             : Stratified K-Fold (k=5)

Academic context
----------------
Macro-F1 is preferred over accuracy for imbalanced multi-class problems
(Sokolova & Lapalme, 2009). In a 3-class sentiment problem with ~60%
neutral, accuracy is trivially maximised by predicting neutral always.
Macro-F1 equally weights each class's F1 score.
"""

from __future__ import annotations

import sys
import time
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE, CV_FOLDS,
)

# ── UTF-8 encoding guard (fixes garbled output on Windows) ─────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPORT_DIR = REPORTS_DIR / "use_case_C_nlp"
MODEL_DIR  = MODELS_DIR  / "use_case_C_nlp"
DATA_DIR_C = DATA_DIR    / "financial_phrasebank"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP    = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_NAMES  = ["negative", "neutral", "positive"]
N_CLASSES    = 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load feature arrays
# ─────────────────────────────────────────────────────────────────────────────

def load_features() -> dict:
    """
    Load all feature arrays saved by Step 3.

    Returns a dict with keys:
        X_tfidf_train, X_tfidf_val,
        X_hc_train, X_hc_val,
        X_finbert_train, X_finbert_val  (None if not available),
        X_combined_train, X_combined_val,
        y_train, y_val
    """
    required = [
        "X_tfidf_train.npy", "X_tfidf_val.npy",
        "X_hc_train.npy",    "X_hc_val.npy",
        "y_train.npy",       "y_val.npy",
    ]
    for f in required:
        if not (DATA_DIR_C / f).exists():
            raise FileNotFoundError(
                f"Missing {f}. Run 03_feature_engineering.py first."
            )

    data = {
        "X_tfidf_train":    np.load(DATA_DIR_C / "X_tfidf_train.npy"),
        "X_tfidf_val":      np.load(DATA_DIR_C / "X_tfidf_val.npy"),
        "X_hc_train":       np.load(DATA_DIR_C / "X_hc_train.npy"),
        "X_hc_val":         np.load(DATA_DIR_C / "X_hc_val.npy"),
        "X_combined_train": np.load(DATA_DIR_C / "X_combined_train.npy"),
        "X_combined_val":   np.load(DATA_DIR_C / "X_combined_val.npy"),
        "y_train":          np.load(DATA_DIR_C / "y_train.npy").astype(int),
        "y_val":            np.load(DATA_DIR_C / "y_val.npy").astype(int),
    }

    for key in ["X_finbert_train.npy", "X_finbert_val.npy"]:
        path = DATA_DIR_C / key
        feat_key = key.replace(".npy", "")
        data[feat_key] = np.load(path) if path.exists() else None

    # Clip TF-IDF to [0, ∞) (required for MultinomialNB)
    data["X_tfidf_train_pos"] = np.clip(data["X_tfidf_train"], 0, None)
    data["X_tfidf_val_pos"]   = np.clip(data["X_tfidf_val"], 0, None)

    log.info(f"TF-IDF train  : {data['X_tfidf_train'].shape}")
    log.info(f"Combined train: {data['X_combined_train'].shape}")
    log.info(f"FinBERT available: {data['X_finbert_train'] is not None}")
    log.info(f"y_train classes: {np.unique(data['y_train'])}")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_models() -> dict:
    """
    Return models keyed by name.  Each value is a dict:
        {"model": estimator, "features": "tfidf" | "combined" | "finbert"}

    Feature track "tfidf"    → X_tfidf_train/val
    Feature track "combined" → X_combined_train/val  (TF-IDF + HC)
    Feature track "finbert"  → FinBERT embeddings + HC (best semantic features)

    Model rationale
    ---------------
    Logistic Regression  : Linear text baseline; widely used in finance NLP;
                           coefficients directly interpretable as term weights.
    Complement NB        : Improved version of MultinomialNB for imbalanced
                           text data (Rennie et al., 2003). Strong on short
                           financial sentences.
    Random Forest        : Captures non-linear TF-IDF interactions; feature
                           importance shows which terms drive each class.
    XGBoost              : Gradient boosting on sparse TF-IDF representation.
    LightGBM (TF-IDF)    : Fast boosting; matches XGBoost at lower cost.
    LightGBM (FinBERT)   : Expected champion — combines domain semantics
                           (FinBERT) with gradient boosting efficiency.
    """
    pos_weight = N_CLASSES  # approximate for balanced loss
    models = {}

    # ── Baseline 1: Logistic Regression ──────────────────────────────────────
    models["Logistic Regression (baseline)"] = {
        "model": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=1.0,
            solver="saga",
            random_state=RANDOM_STATE,
        ),
        "features": "combined",
    }

    # ── Baseline 2: Complement Naive Bayes ───────────────────────────────────
    models["Complement NB (baseline)"] = {
        "model": ComplementNB(alpha=0.1),
        "features": "tfidf_pos",      # NB requires non-negative features
    }

    # ── Advanced 1: Random Forest ─────────────────────────────────────────────
    models["Random Forest"] = {
        "model": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "features": "combined",
    }

    # ── Advanced 2: XGBoost ───────────────────────────────────────────────────
    if XGB_AVAILABLE:
        models["XGBoost"] = {
            "model": xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=N_CLASSES,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                verbosity=0,
            ),
            "features": "combined",
        }

    # ── Advanced 3: LightGBM (TF-IDF) ─────────────────────────────────────────
    if LGB_AVAILABLE:
        models["LightGBM (TF-IDF)"] = {
            "model": lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            "features": "combined",
        }

    # ── Advanced 4: LightGBM (FinBERT) ─────────────────────────────────────────
    if LGB_AVAILABLE:
        models["LightGBM (FinBERT)"] = {
            "model": lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            "features": "finbert",   # uses FinBERT embeddings if available
        }

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Full multi-class evaluation on validation set."""
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)  # (N, 3)

    # ROC-AUC OvR (macro)
    y_bin = label_binarize(y_val, classes=[0, 1, 2])
    try:
        roc_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")

    per_class_f1 = f1_score(y_val, y_pred, average=None, zero_division=0)

    return {
        "accuracy":   round(accuracy_score(y_val, y_pred), 4),
        "macro_f1":   round(f1_score(y_val, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1":round(f1_score(y_val, y_pred, average="weighted", zero_division=0), 4),
        "roc_auc_ovr":round(roc_auc, 4),
        "f1_negative":round(per_class_f1[0], 4),
        "f1_neutral":  round(per_class_f1[1], 4),
        "f1_positive": round(per_class_f1[2], 4),
        "conf_matrix": confusion_matrix(y_val, y_pred),
        "y_pred":      y_pred,
        "y_prob":      y_prob,
    }


def cross_validate_model(
    name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = CV_FOLDS,
) -> dict:
    """Stratified K-Fold CV returning macro-F1 stats."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_f1s = []

    log.info(f"\n  [{name}] — {n_splits}-fold Stratified CV…")
    t0 = time.time()

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_vl)
        f1 = f1_score(y_vl, y_pred, average="macro", zero_division=0)
        fold_f1s.append(f1)
        log.info(f"    Fold {fold}/{n_splits}: macro-F1={f1:.4f}")

    elapsed = time.time() - t0
    result = {
        "cv_macro_f1_mean": round(float(np.mean(fold_f1s)), 4),
        "cv_macro_f1_std":  round(float(np.std(fold_f1s)), 4),
        "cv_time_sec":      round(elapsed, 1),
    }
    log.info(
        f"    → Mean macro-F1: {result['cv_macro_f1_mean']:.4f} "
        f"± {result['cv_macro_f1_std']:.4f}  ({elapsed:.0f}s)"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(all_results: dict, y_val: np.ndarray) -> None:
    n = len(all_results)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes = np.array(axes).flatten()

    for ax, (name, res) in zip(axes, all_results.items()):
        cm = res["val"]["conf_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=ax, cbar=False,
        )
        ax.set_title(
            f"{name}\nmacro-F1={res['val']['macro_f1']:.3f}  "
            f"Acc={res['val']['accuracy']:.3f}",
            fontsize=9,
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(
        "Confusion Matrices — Financial Sentiment (Validation Set)\n"
        "Classes: negative / neutral / positive",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = REPORT_DIR / "confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    """Grouped bar: macro-F1, accuracy, ROC-AUC per model."""
    metrics = ["macro_f1", "accuracy", "roc_auc_ovr"]
    labels  = ["Macro-F1", "Accuracy", "ROC-AUC (OvR)"]
    colours = ["#1E88E5", "#43A047", "#E53935"]

    x = np.arange(len(comparison_df))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(comparison_df) * 2), 6))
    for i, (metric, label, colour) in enumerate(zip(metrics, labels, colours)):
        if metric not in comparison_df.columns:
            continue
        bars = ax.bar(x + i * w, comparison_df[metric], w,
                      label=label, color=colour, alpha=0.85)
        for bar, val in zip(bars, comparison_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels(comparison_df["model"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Financial Sentiment — Model Comparison ({CV_FOLDS}-fold CV)\n"
        "Primary metric: Macro-F1 (handles class imbalance)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = REPORT_DIR / "model_cv_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


def plot_per_class_f1(comparison_df: pd.DataFrame) -> None:
    """Heatmap of per-class F1 across all models."""
    cols = ["f1_negative", "f1_neutral", "f1_positive"]
    if not all(c in comparison_df.columns for c in cols):
        return

    data = comparison_df[cols].values
    fig, ax = plt.subplots(figsize=(6, max(4, len(comparison_df) * 0.6)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Negative", "Neutral", "Positive"], fontsize=11)
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df["model"], fontsize=9)
    for i in range(len(comparison_df)):
        for j in range(3):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if data[i, j] < 0.8 else "white")
    plt.colorbar(im, ax=ax, label="F1 Score")
    ax.set_title("Per-Class F1 Scores — Validation Set", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = REPORT_DIR / "per_class_f1_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_all(data: dict) -> tuple[dict, pd.DataFrame]:
    models_def = build_models()
    all_results = {}
    summary_rows = []

    # Feature map: track name → (train_array, val_array)
    feat_map = {
        "tfidf":     (data["X_tfidf_train"],     data["X_tfidf_val"]),
        "tfidf_pos": (data["X_tfidf_train_pos"],  data["X_tfidf_val_pos"]),
        "combined":  (data["X_combined_train"],   data["X_combined_val"]),
        "finbert":   (data["X_finbert_train"],    data["X_finbert_val"]),
    }

    print("\n" + "=" * 65)
    print(f"  Training {len(models_def)} models with {CV_FOLDS}-fold Stratified CV")
    print(f"  Train: {len(data['y_train']):,}  |  Val: {len(data['y_val']):,}")
    print(f"  Classes: {dict(zip(*np.unique(data['y_train'], return_counts=True)))}")
    print("=" * 65)

    for name, spec in models_def.items():
        model      = spec["model"]
        feat_track = spec["features"]

        X_train_feat, X_val_feat = feat_map[feat_track]

        # Skip FinBERT-dependent models if embeddings not available
        if X_train_feat is None:
            log.warning(f"  [{name}] Skipping — FinBERT embeddings not available.")
            continue

        # ── Cross-validation ──────────────────────────────────────────────
        cv_results = cross_validate_model(name, model, X_train_feat, data["y_train"])

        # ── Final fit on full training set ────────────────────────────────
        log.info(f"  [{name}] Final fit…")
        model.fit(X_train_feat, data["y_train"])

        # ── Validation evaluation ─────────────────────────────────────────
        val_metrics = evaluate(model, X_val_feat, data["y_val"])
        all_results[name] = {
            "cv":      cv_results,
            "val":     val_metrics,
            "model":   model,
            "features":feat_track,
        }

        # ── Save model ────────────────────────────────────────────────────
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, MODEL_DIR / f"{safe}.pkl")

        # ── Classification report ─────────────────────────────────────────
        print(f"\n  ✓ {name}")
        print(f"    Feature track : {feat_track}")
        print(f"    CV macro-F1   : {cv_results['cv_macro_f1_mean']:.4f} "
              f"± {cv_results['cv_macro_f1_std']:.4f}")
        print(f"    Val macro-F1  : {val_metrics['macro_f1']:.4f}")
        print(f"    Val accuracy  : {val_metrics['accuracy']:.4f}")
        print(f"    Val ROC-AUC   : {val_metrics['roc_auc_ovr']:.4f}")
        print(f"    Per-class F1  : neg={val_metrics['f1_negative']:.3f}  "
              f"neu={val_metrics['f1_neutral']:.3f}  "
              f"pos={val_metrics['f1_positive']:.3f}")

        row = {"model": name, "features": feat_track}
        row.update(cv_results)
        row.update({k: v for k, v in val_metrics.items()
                    if not isinstance(v, np.ndarray)})
        summary_rows.append(row)

    comparison_df = pd.DataFrame(summary_rows)
    return all_results, comparison_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Business translation
# ─────────────────────────────────────────────────────────────────────────────

def print_business_summary(comparison_df: pd.DataFrame) -> None:
    """
    Translate macro-F1 improvement into trading signal quality.

    A sentiment classifier's value in event-driven trading comes from its
    ability to correctly separate positive/negative signals from neutral noise.
    Each misclassification (positive → neutral, negative → positive) can
    lead to a missed trade or incorrect trade direction.
    """
    print("\n" + "=" * 65)
    print("  BUSINESS IMPACT — Sentiment-Driven Trading Signals")
    print("=" * 65)
    print(
        "  Assumptions:\n"
        "    Neutral sentences correctly predicted → no trade (correct)\n"
        "    Positive predicted → long signal generated\n"
        "    Negative predicted → short signal generated\n"
        "    Misclassification rate reduces signal quality (Sharpe ratio)\n"
    )

    best = comparison_df.sort_values("macro_f1", ascending=False).iloc[0]
    base = comparison_df[comparison_df["model"].str.contains("baseline")
                         ].sort_values("macro_f1", ascending=False).iloc[0]

    for label, row in [("Best Model", best), ("Baseline", base)]:
        print(f"  [{label}: {row['model']}]")
        f1_neg = row.get("f1_negative", 0)
        f1_neu = row.get("f1_neutral",  0)
        f1_pos = row.get("f1_positive", 0)
        print(f"    Macro-F1         : {row['macro_f1']:.3f}")
        print(f"    Negative recall  : {f1_neg:.3f}  (downside risk detection)")
        print(f"    Neutral precision: {f1_neu:.3f}  (noise filtering)")
        print(f"    Positive recall  : {f1_pos:.3f}  (upside opportunity capture)")
        print()

    f1_gain = best["macro_f1"] - base["macro_f1"]
    print(f"  Macro-F1 improvement: +{f1_gain:.3f} over baseline")
    print(f"  → {f1_gain*100:.1f} pp better signal accuracy from ML champion")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case C (NLP): Model Training")
    print("  Financial PhraseBank — Sentiment Classification")
    print("=" * 65 + "\n")

    print("[1] Loading features…")
    data = load_features()

    print("[2] Training all models…")
    all_results, comparison_df = train_all(data)

    # Save comparison
    comp_path = REPORT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comp_path, index=False)
    log.info(f"Model comparison saved → {comp_path}")

    # Display table
    display_cols = [c for c in [
        "model", "features",
        "cv_macro_f1_mean", "cv_macro_f1_std",
        "macro_f1", "accuracy", "roc_auc_ovr",
        "f1_negative", "f1_neutral", "f1_positive",
    ] if c in comparison_df.columns]
    print("\n--- Model Comparison Table ---")
    print(comparison_df[display_cols].to_string(index=False))

    print("\n[3] Generating plots…")
    plot_confusion_matrices(all_results, data["y_val"])
    plot_model_comparison(comparison_df)
    plot_per_class_f1(comparison_df)

    print_business_summary(comparison_df)

    # Save feature column info for dashboard
    joblib.dump(
        {"n_tfidf": data["X_tfidf_train"].shape[1],
         "n_hc":    data["X_hc_train"].shape[1],
         "n_finbert": data["X_finbert_train"].shape[1] if data["X_finbert_train"] is not None else 0,
         "n_combined": data["X_combined_train"].shape[1]},
        MODEL_DIR / "feature_meta.pkl",
    )

    best_model_name = comparison_df.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    print(f"  Best model (by macro-F1): {best_model_name}")
    print("  → Ready for hyperparameter tuning (05_hyperparameter_tuning.py)")
    print(f"\n  Reports : {REPORT_DIR}")
    print(f"  Models  : {MODEL_DIR}\n")

    return all_results, comparison_df


if __name__ == "__main__":
    main()
