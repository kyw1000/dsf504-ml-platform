"""
use_case_A_fraud/04_model_training.py
=======================================
Use Case A — Financial Crime & Fraud Analytics
Phase 3: Model Development + Cross-Validation

Implements the full DSF504 minimum model requirement:
  ✓ Baseline model    : Logistic Regression (class_weight='balanced')
  ✓ Advanced model 1  : Random Forest
  ✓ Advanced model 2  : XGBoost
  ✓ Advanced model 3  : LightGBM
  ✓ Advanced model 4  : MLP Neural Network
  ✓ Cross-validation  : Stratified K-Fold (k=5)
  ✓ Imbalance handling: SMOTE (training fold only — no leakage)
  ✓ Metrics           : ROC-AUC, PR-AUC, F1, Precision, Recall, Threshold

All models are saved to data/ieee_fraud/models/ for use in the dashboard
and hyperparameter tuning module.

ML Framework Phase: Algorithm Selection → Training → Cross-Validation

Run
---
    cd DSF504_ML_Platform
    python use_case_A_fraud/04_model_training.py
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
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[!] xgboost not installed — run: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("[!] lightgbm not installed — run: pip install lightgbm")

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE,
    CV_FOLDS, IMBALANCE_THRESHOLD,
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

REPORT_DIR = REPORTS_DIR / "use_case_A"
MODEL_DIR  = MODELS_DIR / "use_case_A"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature / target preparation
# ─────────────────────────────────────────────────────────────────────────────

# Columns to always exclude from features
EXCLUDE_COLS = {
    "TransactionID", "isFraud", "TransactionDT",
    # Raw string columns replaced by engineered features
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "id_01", "id_02", "id_03", "id_04", "id_05",
    "id_06", "id_07", "id_08", "id_09", "id_10", "id_11",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return model-ready feature columns: strictly numeric dtypes only.

    Uses select_dtypes(include=[np.number]) so that object, category,
    StringDtype, BooleanDtype, and any other non-numeric pandas extension
    types are all excluded — regardless of how the parquet reader chose to
    represent them on load.
    """
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    return [c for c in df.columns if c in numeric_cols and c not in EXCLUDE_COLS]


def prepare_Xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X (features) and y (target) with NaN / non-numeric safety."""
    X = df[feature_cols].copy()
    y = df["isFraud"].astype(np.int8)

    # Coerce any column that still slipped through as non-numeric
    # (e.g. StringDtype or category columns with unexpected string values)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Replace infinities and fill remaining NaN with column medians
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_models(random_state: int = RANDOM_STATE) -> dict:
    """
    Return a dictionary of model name → sklearn-compatible estimator.

    Each model is wrapped in an ImbPipeline with SMOTE applied
    inside the cross-validation loop (preventing data leakage into
    the validation fold).

    Model rationale
    ---------------
    Logistic Regression  : Linear baseline; interpretable coefficients;
                           class_weight='balanced' handles imbalance without SMOTE.
    Decision Tree        : Naïve non-linear baseline; prone to overfitting
                           but establishes a low-performance anchor.
    Random Forest        : Ensemble of decorrelated trees; robust to outliers;
                           natural feature importance; handles mixed dtypes.
    XGBoost              : Gradient boosting with L1/L2 regularisation;
                           state-of-the-art for tabular fraud data.
    LightGBM             : Leaf-wise boosting; faster than XGBoost on
                           high-cardinality features; excellent on imbalanced data
                           via scale_pos_weight.
    MLP Neural Network   : Captures non-linear feature interactions;
                           complement to tree-based models.
    """
    # Fraud : Legit ratio for scale_pos_weight
    pos_weight = round((1 - 0.035) / 0.035)  # ≈ 27.6

    models = {}

    # ── Baseline 1: Logistic Regression ─────────────────────────────────────
    models["Logistic Regression (baseline)"] = ImbPipeline([
        ("smote",  SMOTE(random_state=random_state, sampling_strategy=0.10)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            solver="saga",
            random_state=random_state,
        )),
    ])

    # ── Baseline 2: Decision Tree ────────────────────────────────────────────
    models["Decision Tree (baseline)"] = ImbPipeline([
        ("smote", SMOTE(random_state=random_state, sampling_strategy=0.10)),
        ("clf",   DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])

    # ── Advanced 1: Random Forest ─────────────────────────────────────────────
    models["Random Forest"] = ImbPipeline([
        ("smote", SMOTE(random_state=random_state, sampling_strategy=0.10)),
        ("clf",   RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )),
    ])

    # ── Advanced 2: XGBoost ───────────────────────────────────────────────────
    if XGB_AVAILABLE:
        models["XGBoost"] = ImbPipeline([
            ("smote", SMOTE(random_state=random_state, sampling_strategy=0.10)),
            ("clf",   xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                use_label_encoder=False,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=random_state,
                verbosity=0,
            )),
        ])

    # ── Advanced 3: LightGBM ──────────────────────────────────────────────────
    if LGB_AVAILABLE:
        models["LightGBM"] = ImbPipeline([
            ("smote", SMOTE(random_state=random_state, sampling_strategy=0.10)),
            ("clf",   lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                n_jobs=-1,
                random_state=random_state,
                verbose=-1,
            )),
        ])

    # ── Advanced 4: MLP Neural Network ───────────────────────────────────────
    models["MLP Neural Network"] = ImbPipeline([
        ("smote",  SMOTE(random_state=random_state, sampling_strategy=0.10)),
        ("scaler", StandardScaler()),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
        )),
    ])

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_val(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.50,
) -> dict:
    """
    Compute full evaluation metrics on a hold-out validation set.

    Returns ROC-AUC, PR-AUC (average precision), F1, Precision, Recall,
    and the confusion matrix at the given threshold.
    """
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc":   round(roc_auc_score(y_val, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_val, y_prob), 4),
        "f1":        round(f1_score(y_val, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_val, y_pred, zero_division=0), 4),
        "conf_matrix": confusion_matrix(y_val, y_pred),
        "y_prob":    y_prob,
        "threshold": threshold,
    }


def find_optimal_threshold(y_val: pd.Series, y_prob: np.ndarray) -> float:
    """
    Find threshold that maximises F1 score on the validation set.

    For fraud detection, F1 is preferred over accuracy because it balances
    precision (false alarm rate) and recall (fraud caught rate).
    The optimal threshold is often lower than 0.5 given the extreme class
    imbalance — the model is uncertain about many fraud cases.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores  = [
        f1_score(y_val, (y_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_t = thresholds[np.argmax(f1_scores)]
    log.info(
        f"  Optimal threshold: {best_t:.2f} "
        f"(F1={max(f1_scores):.4f})"
    )
    return float(best_t)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-validation runner
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(
    name: str,
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_FOLDS,
) -> dict:
    """
    Run Stratified K-Fold CV and return per-fold metrics.

    SMOTE is applied inside each training fold only — never touching the
    validation fold. This is enforced by wrapping SMOTE in the ImbPipeline.

    Parameters
    ----------
    name     : Model name for logging
    pipeline : ImbPipeline with SMOTE + model
    X, y     : Full feature matrix and labels
    n_splits : Number of CV folds (default=5)

    Returns
    -------
    cv_results : dict with mean/std for each metric
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = {m: [] for m in ["roc_auc", "pr_auc", "f1", "precision", "recall"]}

    log.info(f"\n  [{name}] — {n_splits}-fold Stratified CV…")
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_prob = pipeline.predict_proba(X_vl)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fold_metrics["roc_auc"].append(roc_auc_score(y_vl, y_prob))
        fold_metrics["pr_auc"].append(average_precision_score(y_vl, y_prob))
        fold_metrics["f1"].append(f1_score(y_vl, y_pred, zero_division=0))
        fold_metrics["precision"].append(precision_score(y_vl, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_vl, y_pred, zero_division=0))

        log.info(
            f"    Fold {fold}/{n_splits}: "
            f"ROC-AUC={fold_metrics['roc_auc'][-1]:.4f} "
            f"PR-AUC={fold_metrics['pr_auc'][-1]:.4f} "
            f"F1={fold_metrics['f1'][-1]:.4f}"
        )

    elapsed = time.time() - t0
    cv_results = {
        f"cv_{m}_mean": round(float(np.mean(v)), 4)
        for m, v in fold_metrics.items()
    }
    cv_results.update({
        f"cv_{m}_std": round(float(np.std(v)), 4)
        for m, v in fold_metrics.items()
    })
    cv_results["cv_time_sec"] = round(elapsed, 1)

    log.info(
        f"    → Mean ROC-AUC: {cv_results['cv_roc_auc_mean']:.4f} "
        f"± {cv_results['cv_roc_auc_std']:.4f}  "
        f"PR-AUC: {cv_results['cv_pr_auc_mean']:.4f}  "
        f"({elapsed:.0f}s)"
    )
    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_pr_curves(
    results: dict,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    save: bool = True,
) -> None:
    """
    ROC and Precision-Recall curves for all models on the validation set.

    PR-AUC is the primary metric for imbalanced fraud detection:
    it directly measures the precision-recall trade-off without being
    inflated by the large number of true negatives.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, res), color in zip(results.items(), colors):
        y_prob = res["val"]["y_prob"]

        # ROC curve
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        axes[0].plot(fpr, tpr, lw=2, color=color,
                     label=f"{name}  (AUC={res['val']['roc_auc']:.4f})")

        # PR curve
        prec, rec, _ = precision_recall_curve(y_val, y_prob)
        axes[1].plot(rec, prec, lw=2, color=color,
                     label=f"{name}  (PR-AUC={res['val']['pr_auc']:.4f})")

    # ROC panel
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title("ROC Curves — Validation Set", fontsize=13)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR panel
    baseline_pr = y_val.mean()
    axes[1].axhline(y=baseline_pr, color="gray", linestyle="--",
                    alpha=0.6, label=f"Random ({baseline_pr:.3f})")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title("Precision-Recall Curves — Validation Set\n"
                       "(Primary metric for imbalanced fraud detection)", fontsize=13)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Model Comparison — ROC & PR Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "model_roc_pr_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


def plot_confusion_matrices(
    results: dict,
    y_val: pd.Series,
    save: bool = True,
) -> None:
    """Confusion matrix grid for all models at their optimal threshold."""
    n = len(results)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for ax, (name, res) in zip(axes, results.items()):
        cm    = res["val"]["conf_matrix"]
        thresh = res["val"]["threshold"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
            ax=ax, cbar=False,
        )
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(
            f"{name}\n"
            f"Threshold={thresh:.2f} | "
            f"Recall={res['val']['recall']:.3f} | "
            f"Prec={res['val']['precision']:.3f}",
            fontsize=9,
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(
        "Confusion Matrices — Validation Set (Optimal Threshold per Model)\n"
        "Key: Minimise FN (missed fraud) while controlling FP (false alerts)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "confusion_matrices.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


def plot_cv_comparison(comparison_df: pd.DataFrame, save: bool = True) -> None:
    """
    Grouped bar chart: CV ROC-AUC and PR-AUC for all models.
    Error bars show ± 1 standard deviation across folds.
    """
    df = comparison_df.copy()
    x  = np.arange(len(df))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - w/2, df["cv_roc_auc_mean"], w,
                   yerr=df["cv_roc_auc_std"], capsize=4,
                   color="#1976D2", alpha=0.8, label="ROC-AUC")
    bars2 = ax.bar(x + w/2, df["cv_pr_auc_mean"], w,
                   yerr=df["cv_pr_auc_std"], capsize=4,
                   color="#D32F2F", alpha=0.8, label="PR-AUC")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Model Comparison — {CV_FOLDS}-Fold Stratified CV (with SMOTE)\n"
        "Error bars = ±1 std across folds",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save:
        path = REPORT_DIR / "model_cv_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


def plot_feature_importance(
    pipeline,
    feature_cols: list[str],
    model_name: str,
    top_n: int = 30,
    save: bool = True,
) -> None:
    """
    Bar chart of top-N feature importances for tree-based models.
    For logistic regression, uses |coefficient| as importance proxy.
    """
    clf = pipeline.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return

    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    top_feats = feat_imp.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#D32F2F" if "fe_" in f else "#1976D2" for f in top_feats.index]
    ax.barh(top_feats.index[::-1], top_feats.values[::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(
        f"Top-{top_n} Feature Importances — {model_name}\n"
        f"Red = engineered features (fe_*), Blue = original features",
        fontsize=11,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D32F2F", label="Engineered (fe_*)"),
        Patch(facecolor="#1976D2", label="Original"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    if save:
        path = REPORT_DIR / f"feature_importance_{safe_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    feature_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    Train all models with CV, evaluate on validation set, return results.

    Returns
    -------
    all_results  : dict  {model_name: {cv: {...}, val: {...}, pipeline: ...}}
    comparison_df: DataFrame model comparison table (for report)
    """
    models       = build_models()
    all_results  = {}
    summary_rows = []

    print("\n" + "=" * 65)
    print(f"  Training {len(models)} models with {CV_FOLDS}-fold Stratified CV")
    print(f"  Training set: {len(X_train):,} rows  |  "
          f"Validation: {len(X_val):,} rows")
    print(f"  Fraud rate (train): {y_train.mean():.3%}")
    print("=" * 65)

    for name, pipeline in models.items():

        # ── Cross-validation ──────────────────────────────────────────────
        cv_results = cross_validate_model(name, pipeline, X_train, y_train)

        # ── Final fit on full training set ───────────────────────────────
        log.info(f"  [{name}] Final fit on full training set…")
        pipeline.fit(X_train, y_train)

        # ── Validation evaluation ─────────────────────────────────────────
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        opt_thresh = find_optimal_threshold(y_val, y_prob)
        val_metrics = evaluate_on_val(pipeline, X_val, y_val, threshold=opt_thresh)

        all_results[name] = {
            "cv":       cv_results,
            "val":      val_metrics,
            "pipeline": pipeline,
        }

        # ── Save model ────────────────────────────────────────────────────
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        model_path = MODEL_DIR / f"{safe_name}.pkl"
        joblib.dump(pipeline, model_path)
        log.info(f"  [{name}] Saved → {model_path}")

        # ── Feature importance plot (tree-based only) ─────────────────────
        plot_feature_importance(pipeline, feature_cols, name)

        # ── Summary row ───────────────────────────────────────────────────
        row = {"model": name}
        row.update(cv_results)
        row.update({
            "val_roc_auc":   val_metrics["roc_auc"],
            "val_pr_auc":    val_metrics["pr_auc"],
            "val_f1":        val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall":    val_metrics["recall"],
            "val_threshold": val_metrics["threshold"],
        })
        summary_rows.append(row)

        print(
            f"\n  ✓ {name}\n"
            f"    CV  ROC-AUC : {cv_results['cv_roc_auc_mean']:.4f} "
            f"± {cv_results['cv_roc_auc_std']:.4f}\n"
            f"    CV  PR-AUC  : {cv_results['cv_pr_auc_mean']:.4f} "
            f"± {cv_results['cv_pr_auc_std']:.4f}\n"
            f"    Val ROC-AUC : {val_metrics['roc_auc']:.4f}\n"
            f"    Val PR-AUC  : {val_metrics['pr_auc']:.4f}\n"
            f"    Val F1      : {val_metrics['f1']:.4f}  "
            f"(Precision={val_metrics['precision']:.4f}, "
            f"Recall={val_metrics['recall']:.4f})\n"
            f"    Threshold   : {val_metrics['threshold']:.2f}"
        )

    comparison_df = pd.DataFrame(summary_rows)
    return all_results, comparison_df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Business translation
# ─────────────────────────────────────────────────────────────────────────────

def print_business_summary(
    comparison_df: pd.DataFrame,
    total_transactions: int = 590_540,
    avg_fraud_amount: float = 130.0,
) -> None:
    """
    Translate model metrics into business impact estimates.

    DSF504 requirement: metrics must be translated into business value,
    not just reported as abstract percentages.

    Assumptions (illustrative):
    - Dataset represents 6 months of transactions
    - Average fraudulent transaction: $130 (industry estimate)
    - Cost of a false alert investigation: $8 per alert
    """
    n_fraud_total = int(total_transactions * 0.035)
    print("\n" + "=" * 65)
    print("  BUSINESS IMPACT TRANSLATION")
    print(f"  (Based on {total_transactions:,} transactions, "
          f"{n_fraud_total:,} fraud cases)")
    print("=" * 65)

    best_row = comparison_df.sort_values("val_roc_auc", ascending=False).iloc[0]
    worst_row = comparison_df[
        comparison_df["model"].str.contains("baseline")
    ].sort_values("val_roc_auc", ascending=False).iloc[0]

    for label, row in [("Best Model", best_row), ("Baseline", worst_row)]:
        recall    = row["val_recall"]
        precision = row["val_precision"]

        fraud_caught   = int(n_fraud_total * recall)
        fraud_missed   = n_fraud_total - fraud_caught
        false_alerts   = int(fraud_caught / (precision + 1e-9)) - fraud_caught

        fraud_loss_saved = fraud_caught * avg_fraud_amount
        false_alert_cost = false_alerts * 8
        net_benefit      = fraud_loss_saved - false_alert_cost

        print(f"\n  [{label}: {row['model']}]")
        print(f"    Recall={recall:.2%} | Precision={precision:.2%}")
        print(f"    Fraud caught        : {fraud_caught:>8,} / {n_fraud_total:,}")
        print(f"    Fraud missed        : {fraud_missed:>8,}  "
              f"(lost: ${fraud_missed * avg_fraud_amount:,.0f})")
        print(f"    False alerts        : {false_alerts:>8,}  "
              f"(cost: ${false_alert_cost:,.0f})")
        print(f"    Fraud losses saved  : ${fraud_loss_saved:>12,.0f}")
        print(f"    Net benefit         : ${net_benefit:>12,.0f}")

    improvement = (
        (best_row["val_roc_auc"] - worst_row["val_roc_auc"])
        / worst_row["val_roc_auc"] * 100
    )
    print(f"\n  ROC-AUC improvement over baseline: +{improvement:.1f}%")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case A: Model Training & Cross-Validation")
    print("=" * 65 + "\n")

    # ── Load feature-engineered data ──────────────────────────────────────
    cache_dir = DATA_DIR / "ieee_fraud"
    train_fe  = cache_dir / "train_fe.parquet"
    val_fe    = cache_dir / "val_fe.parquet"

    if not train_fe.exists():
        print(
            "[!] Engineered data not found.\n"
            "    Run first: python use_case_A_fraud/03_feature_engineering.py"
        )
        return

    log.info("Loading feature-engineered datasets…")
    df_train = pd.read_parquet(train_fe)
    df_val   = pd.read_parquet(val_fe)

    # ── Prepare X, y ──────────────────────────────────────────────────────
    feature_cols = get_feature_columns(df_train)
    log.info(f"Feature columns: {len(feature_cols)}")

    X_train, y_train = prepare_Xy(df_train, feature_cols)
    X_val,   y_val   = prepare_Xy(df_val,   feature_cols)

    # ── Train all models ──────────────────────────────────────────────────
    all_results, comparison_df = train_and_evaluate_all(
        X_train, y_train, X_val, y_val, feature_cols
    )

    # ── Save comparison table ─────────────────────────────────────────────
    comp_path = REPORT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comp_path, index=False)
    log.info(f"\nModel comparison saved → {comp_path}")

    # ── Pretty-print comparison ───────────────────────────────────────────
    display_cols = [
        "model",
        "cv_roc_auc_mean", "cv_roc_auc_std",
        "cv_pr_auc_mean",
        "val_roc_auc", "val_pr_auc",
        "val_f1", "val_precision", "val_recall",
        "val_threshold",
    ]
    print("\n--- Model Comparison Table ---")
    print(comparison_df[display_cols].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    plot_roc_pr_curves(all_results, X_val, y_val)
    plot_confusion_matrices(all_results, y_val)
    plot_cv_comparison(comparison_df)

    # ── Business impact translation ───────────────────────────────────────
    print_business_summary(comparison_df)

    # ── Save feature list for next step ──────────────────────────────────
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

    # ── Identify best model for tuning ────────────────────────────────────
    best_model_name = comparison_df.sort_values(
        "val_roc_auc", ascending=False
    ).iloc[0]["model"]
    print(f"\n  Best model (by Val ROC-AUC): {best_model_name}")
    print("  → Ready for hyperparameter tuning (05_hyperparameter_tuning.py)")
    print(f"\n  Reports: {REPORT_DIR}")
    print(f"  Models : {MODEL_DIR}\n")

    return all_results, comparison_df


if __name__ == "__main__":
    main()
