"""
use_case_E_insurance/04_model_training.py
==========================================
Use Case E — Insurance Risk & Claims Analytics
Phase 3: Model Development + Cross-Validation

Implements the full DSF504 minimum model requirement:
  ✓ Baseline model    : Logistic Regression (class_weight='balanced')
  ✓ Advanced model 1  : Random Forest
  ✓ Advanced model 2  : XGBoost
  ✓ Advanced model 3  : LightGBM
  ✓ Advanced model 4  : MLP Neural Network
  ✓ Cross-validation  : Stratified K-Fold (k=5)
  ✓ Imbalance handling: class_weight='balanced' (all applicable models)
  ✓ Metrics           : ROC-AUC, Normalized Gini, PR-AUC, F1

Primary metric: Normalized Gini = 2 × ROC-AUC − 1
  (the official Porto Seguro competition metric)

All models saved to models/use_case_E/ for hyperparameter tuning.

ML Framework Phase: Algorithm Selection → Training → Cross-Validation

Run
---
    cd C:\\DSF504
    python use_case_E_insurance/04_model_training.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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
from config import DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE, CV_FOLDS

# ── UTF-8 encoding guard ─────────────────────────────────────────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "porto_seguro"
REPORT_DIR  = REPORTS_DIR / "use_case_E"
MODEL_DIR   = MODELS_DIR / "use_case_E"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL  = "target"
PALETTE     = {"no_claim": "#1976D2", "claim": "#D32F2F"}


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalized_gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Normalized Gini Coefficient = 2 × ROC-AUC − 1
    Official Porto Seguro competition metric.
    Range [−1, 1]; a random model scores 0, a perfect model scores 1.
    """
    return 2 * roc_auc_score(y_true, y_score) - 1


def evaluate_model(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
) -> dict:
    """
    Full evaluation suite on the held-out validation set.
    Returns a dict of all metrics for comparison table.
    """
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    roc_auc  = roc_auc_score(y_val, y_proba)
    gini     = normalized_gini(y_val, y_proba)
    pr_auc   = average_precision_score(y_val, y_proba)
    f1       = f1_score(y_val, y_pred, zero_division=0)
    prec     = precision_score(y_val, y_pred, zero_division=0)
    rec      = recall_score(y_val, y_pred, zero_division=0)

    log.info(
        f"{model_name:<22} ROC-AUC={roc_auc:.4f}  Gini={gini:.4f}  "
        f"PR-AUC={pr_auc:.4f}  F1={f1:.4f}"
    )
    return {
        "model":      model_name,
        "roc_auc":    round(roc_auc, 4),
        "gini":       round(gini, 4),
        "pr_auc":     round(pr_auc, 4),
        "f1":         round(f1, 4),
        "precision":  round(prec, 4),
        "recall":     round(rec, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline | ImbPipeline]:
    """
    Return all candidate models as named pipelines.
    LR and MLP use SMOTE to handle the 3.6% class imbalance.
    Tree-based models use class_weight='balanced'.
    """
    models: dict = {}

    # ── Baseline: Logistic Regression ────────────────────────────────────────
    models["Logistic Regression"] = ImbPipeline([
        ("smote",  SMOTE(random_state=random_state, sampling_strategy=0.1)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            random_state=random_state,
            solver="saga",
            n_jobs=1,
        )),
    ])

    # ── Random Forest ─────────────────────────────────────────────────────────
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if XGB_AVAILABLE:
        # scale_pos_weight handles imbalance for XGBoost
        # ~26.6:1 ratio (no-claim:claim)
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=26.6,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if LGB_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    # ── MLP Neural Network ────────────────────────────────────────────────────
    models["MLP Neural Network"] = ImbPipeline([
        ("smote",  SMOTE(random_state=random_state, sampling_strategy=0.1)),
        ("scaler", StandardScaler()),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=200,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )),
    ])

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 5-fold Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = CV_FOLDS,
) -> dict[str, dict]:
    """
    Stratified K-Fold cross-validation on the training set.
    Reports mean ± std ROC-AUC (primary) and Gini for each model.

    Note: SMOTE is applied inside the pipeline per fold — no leakage.
    """
    skf     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results: dict[str, dict] = {}

    for name, model in models.items():
        log.info(f"\n--- CV: {name} ({cv_folds} folds) ---")
        t0 = time.time()
        try:
            res = cross_validate(
                model, X, y,
                cv=skf,
                scoring={"roc_auc": "roc_auc"},
                return_train_score=False,
                n_jobs=1,  # inner SMOTE pipelines are not thread-safe
            )
            roc_scores = res["test_roc_auc"]
            gini_scores = 2 * roc_scores - 1
            elapsed = time.time() - t0
            cv_results[name] = {
                "mean_roc_auc":  round(float(roc_scores.mean()), 4),
                "std_roc_auc":   round(float(roc_scores.std()), 4),
                "mean_gini":     round(float(gini_scores.mean()), 4),
                "std_gini":      round(float(gini_scores.std()), 4),
                "elapsed_s":     round(elapsed, 1),
                "all_roc_auc":   roc_scores.tolist(),
            }
            log.info(
                f"  ROC-AUC: {roc_scores.mean():.4f} ± {roc_scores.std():.4f}  "
                f"  Gini: {gini_scores.mean():.4f}  [{elapsed:.1f}s]"
            )
        except Exception as exc:
            log.warning(f"  CV failed for {name}: {exc}")
            cv_results[name] = {"error": str(exc)}

    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# Full training + evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv_results: dict,
) -> tuple[dict, list[dict], str]:
    """
    Train each model on the full training set and evaluate on the val set.
    Saves each trained model for downstream hyperparameter tuning.

    Returns
    -------
    trained_models   : {name: fitted model}
    comparison_rows  : list of metric dicts for comparison table
    champion_name    : model with highest Gini on val set
    """
    trained_models: dict  = {}
    comparison_rows: list = []

    for name, model in models.items():
        log.info(f"\n[Training] {name}")
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            elapsed = time.time() - t0

            metrics = evaluate_model(model, X_val, y_val, name)
            metrics["cv_roc_auc"]  = cv_results.get(name, {}).get("mean_roc_auc", None)
            metrics["cv_gini"]     = cv_results.get(name, {}).get("mean_gini", None)
            metrics["train_time_s"]= round(elapsed, 1)
            comparison_rows.append(metrics)

            trained_models[name] = model

            # Save individual model
            safe_name = name.lower().replace(" ", "_")
            model_path = MODEL_DIR / f"{safe_name}.pkl"
            joblib.dump(model, model_path)
            log.info(f"  Saved → {model_path}")

        except Exception as exc:
            log.error(f"  Training failed for {name}: {exc}")

    # Determine champion by validation Gini
    comp_df = pd.DataFrame(comparison_rows).sort_values("gini", ascending=False)
    champion_name = comp_df.iloc[0]["model"] if len(comp_df) > 0 else None

    log.info(f"\n🏆 Champion model: {champion_name}")

    # Save comparison table
    comp_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    log.info(f"Saved model_comparison.csv")

    return trained_models, comparison_rows, champion_name


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(comparison_rows: list[dict]) -> None:
    """Horizontal bar chart of val Gini and ROC-AUC for all models."""
    df = pd.DataFrame(comparison_rows).sort_values("gini", ascending=True)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df) * 0.7)))
    colors = ["#1976D2"] * len(df)
    if len(df) > 0:
        colors[-1] = "#388E3C"  # highlight champion

    axes[0].barh(df["model"], df["gini"], color=colors)
    axes[0].set_xlabel("Normalized Gini")
    axes[0].set_title("Model Comparison — Normalized Gini (val set)")
    for i, (v, m) in enumerate(zip(df["gini"], df["model"])):
        axes[0].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

    axes[1].barh(df["model"], df["roc_auc"], color=colors)
    axes[1].set_xlabel("ROC-AUC")
    axes[1].set_title("Model Comparison — ROC-AUC (val set)")
    for i, v in enumerate(df["roc_auc"]):
        axes[1].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'model_comparison.png'}")


def plot_roc_curves(
    trained_models: dict,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """ROC curves for all trained models on the validation set."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.50)")

    cmap = matplotlib.colormaps["tab10"]
    for i, (name, model) in enumerate(trained_models.items()):
        try:
            y_proba = model.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            auc = roc_auc_score(y_val, y_proba)
            ax.plot(fpr, tpr, color=cmap(i), linewidth=1.5,
                    label=f"{name} (AUC={auc:.3f})")
        except Exception as e:
            log.warning(f"ROC curve failed for {name}: {e}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Porto Seguro Insurance Risk Scoring")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'roc_curves.png'}")


def plot_champion_evaluation(
    champion_model,
    champion_name: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """2×2 evaluation panel for the champion model."""
    y_proba = champion_model.predict_proba(X_val)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # (0,0) Confusion matrix
    ax0 = fig.add_subplot(gs[0, 0])
    cm  = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax0,
                xticklabels=["No Claim", "Claim"],
                yticklabels=["No Claim", "Claim"])
    ax0.set_title(f"{champion_name}\nConfusion Matrix (thresh=0.5)")
    ax0.set_xlabel("Predicted")
    ax0.set_ylabel("Actual")

    # (0,1) Score distribution
    ax1 = fig.add_subplot(gs[0, 1])
    claim    = y_proba[y_val == 1]
    no_claim = y_proba[y_val == 0]
    ax1.hist(no_claim, bins=60, alpha=0.6, color="#1976D2", label="No Claim", density=True)
    ax1.hist(claim,    bins=60, alpha=0.7, color="#D32F2F", label="Claim",    density=True)
    ax1.set_xlabel("Predicted Probability of Claim")
    ax1.set_ylabel("Density")
    ax1.set_title("Score Distribution by Class")
    ax1.legend()

    # (1,0) ROC curve
    ax2 = fig.add_subplot(gs[1, 0])
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc = roc_auc_score(y_val, y_proba)
    gini = 2 * auc - 1
    ax2.plot(fpr, tpr, color="#388E3C", linewidth=2,
             label=f"ROC (AUC={auc:.4f}, Gini={gini:.4f})")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.set_title("ROC Curve")
    ax2.legend()

    # (1,1) PR curve
    ax3 = fig.add_subplot(gs[1, 1])
    prec, rec, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)
    ax3.plot(rec, prec, color="#8E24AA", linewidth=2, label=f"PR-AUC={pr_auc:.4f}")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision-Recall Curve")
    ax3.legend()

    fig.suptitle(
        f"Champion Model Evaluation — {champion_name}\n"
        f"Porto Seguro Insurance Risk Scoring",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "champion_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'champion_evaluation.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case E: Insurance Risk & Claims Analytics")
    print("  Step 4: Algorithm Selection + Cross-Validation")
    print("=" * 65 + "\n")

    # Load feature-engineered splits
    train_path = DATA_SUBDIR / "train_fe.parquet"
    val_path   = DATA_SUBDIR / "val_fe.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            "train_fe.parquet not found. Run 03_feature_engineering.py first."
        )

    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)

    feat_cols = [c for c in df_train.columns
                 if c not in ("id", TARGET_COL)
                 and df_train[c].dtype != object]

    X_train, y_train = df_train[feat_cols], df_train[TARGET_COL]
    X_val,   y_val   = df_val[feat_cols],   df_val[TARGET_COL]

    # Drop any remaining NaN columns (safety check)
    null_mask = X_train.isna().mean() > 0.5
    if null_mask.any():
        drop = null_mask[null_mask].index.tolist()
        X_train = X_train.drop(columns=drop)
        X_val   = X_val.drop(columns=drop, errors="ignore")
        log.info(f"Dropped {len(drop)} high-null columns: {drop[:5]}")

    X_train = X_train.fillna(0)
    X_val   = X_val.fillna(0)

    print(f"[1] X_train: {X_train.shape}  |  Claim rate: {y_train.mean():.3%}")
    print(f"    X_val  : {X_val.shape}    |  Claim rate: {y_val.mean():.3%}")

    # Get models
    models = get_models()
    print(f"\n[2] Models to evaluate: {list(models.keys())}")

    # Cross-validation
    print(f"\n[3] Running {CV_FOLDS}-fold stratified cross-validation…")
    cv_results = run_cross_validation(models, X_train, y_train)

    # Full training + val evaluation
    print("\n[4] Training on full train set + evaluating on val set…")
    trained_models, comparison_rows, champion_name = train_and_evaluate(
        models, X_train, y_train, X_val, y_val, cv_results
    )

    # Visualisations
    print("\n[5] Generating evaluation plots…")
    plot_model_comparison(comparison_rows)
    plot_roc_curves(trained_models, X_val, y_val)

    if champion_name and champion_name in trained_models:
        champion = trained_models[champion_name]
        plot_champion_evaluation(champion, champion_name, X_val, y_val)

        # Save champion model
        joblib.dump(champion, MODEL_DIR / "champion.pkl")
        log.info(f"Champion saved → {MODEL_DIR / 'champion.pkl'}")

        # Save champion name
        (MODEL_DIR / "champion_name.txt").write_text(champion_name, encoding="utf-8")

    # Print summary table
    print("\n--- Model Comparison Summary ---")
    comp_df = pd.DataFrame(comparison_rows).sort_values("gini", ascending=False)
    print(comp_df[["model", "roc_auc", "gini", "pr_auc", "f1"]].to_string(index=False))

    # Save feature list used
    feat_list_df = pd.DataFrame({"feature": feat_cols})
    feat_list_df.to_csv(REPORT_DIR / "model_features.csv", index=False)

    print("\n" + "=" * 65)
    print(f"  Step 4 complete. Champion: {champion_name}")
    print("  Ready for Hyperparameter Tuning (05_hyperparameter_tuning.py)")
    print("=" * 65 + "\n")

    return trained_models, champion_name


if __name__ == "__main__":
    trained_models, champion_name = main()
