"""
use_case_D_churn/04_model_training.py
=======================================
DSF504 — Use Case D: Customer Churn Prediction (KKBox)
Step 4: Model Training & Cross-Validation

Implements the DSF504 minimum model requirement:
  ✓ Baseline model   : Logistic Regression (class_weight='balanced')
  ✓ Baseline model 2 : Decision Tree
  ✓ Advanced model 1 : Random Forest
  ✓ Advanced model 2 : XGBoost
  ✓ Advanced model 3 : LightGBM  ← expected champion
  ✓ Advanced model 4 : MLP Neural Network
  ✓ Cross-validation : Stratified K-Fold (k=5)
  ✓ Imbalance        : SMOTE on training fold only (no leakage)
  ✓ Metrics          : ROC-AUC, PR-AUC, F1, Precision, Recall

Dataset: KKBox Churn Prediction (~900K subscribers, ~8.4% churn)

Academic references
-------------------
- Verbeke et al. (2012): ensemble methods outperform LR for churn prediction
- Burez & Van den Poel (2009): SMOTE + gradient boosting for imbalanced churn
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
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    log_warn = "[!] xgboost not installed"

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE,
    CV_FOLDS, IMBALANCE_THRESHOLD,
)

from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPORT_DIR  = REPORTS_DIR / "use_case_D"
MODEL_DIR   = MODELS_DIR  / "use_case_D"
DATA_DIR_D  = DATA_DIR    / "kkbox_churn"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET      = "is_churn"
CHURN_RATE  = 0.084   # ~8.4% churn rate
EXCLUDE_COLS = {TARGET}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE_COLS]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_DIR_D / "train_fe.parquet")
    val   = pd.read_parquet(DATA_DIR_D / "val_fe.parquet")
    return train, val


def build_candidates(churn_rate: float) -> list[tuple[str, object]]:
    """Return (name, estimator) pairs — order matches DSF504 framework."""
    pos_weight = (1 - churn_rate) / churn_rate  # ~10.9

    candidates = [
        ("Logistic Regression",
         ImbPipeline([
             ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
             ("scaler", StandardScaler()),
             ("clf",    LogisticRegression(
                 C=1.0, max_iter=1000, class_weight="balanced",
                 solver="lbfgs", random_state=RANDOM_STATE)),
         ])),
        ("Decision Tree",
         ImbPipeline([
             ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
             ("clf",   DecisionTreeClassifier(
                 max_depth=8, class_weight="balanced",
                 random_state=RANDOM_STATE)),
         ])),
        ("Random Forest",
         ImbPipeline([
             ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
             ("clf",   RandomForestClassifier(
                 n_estimators=200, max_depth=10, class_weight="balanced",
                 n_jobs=1, random_state=RANDOM_STATE)),
         ])),
        ("MLP Neural Network",
         ImbPipeline([
             ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
             ("scaler", StandardScaler()),
             ("clf",    MLPClassifier(
                 hidden_layer_sizes=(128, 64), max_iter=200,
                 early_stopping=True, random_state=RANDOM_STATE)),
         ])),
    ]

    if XGB_AVAILABLE:
        candidates.append(
            ("XGBoost",
             ImbPipeline([
                 ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
                 ("clf",   xgb.XGBClassifier(
                     n_estimators=300, max_depth=6, learning_rate=0.05,
                     scale_pos_weight=pos_weight, eval_metric="auc",
                     use_label_encoder=False, n_jobs=1,
                     random_state=RANDOM_STATE, verbosity=0)),
             ]))
        )

    if LGB_AVAILABLE:
        candidates.append(
            ("LightGBM",
             ImbPipeline([
                 ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
                 ("clf",   lgb.LGBMClassifier(
                     n_estimators=400, max_depth=6, learning_rate=0.05,
                     is_unbalance=True, n_jobs=-1,
                     random_state=RANDOM_STATE, verbose=-1)),
             ]))
        )

    return candidates


def run_cv(candidates, X: np.ndarray, y: np.ndarray) -> list[dict]:
    skf     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, pipeline in candidates:
        log.info("  CV — %s", name)
        fold_aucs, fold_prauc, fold_f1 = [], [], []
        t0 = time.time()

        for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_vl = X[tr_idx], X[vl_idx]
            y_tr, y_vl = y[tr_idx], y[vl_idx]

            pipeline.fit(X_tr, y_tr)
            proba = pipeline.predict_proba(X_vl)[:, 1]
            pred  = pipeline.predict(X_vl)

            fold_aucs.append(roc_auc_score(y_vl, proba))
            fold_prauc.append(average_precision_score(y_vl, proba))
            fold_f1.append(f1_score(y_vl, pred, zero_division=0))
            log.info("    Fold %d: ROC-AUC=%.4f", fold, fold_aucs[-1])

        elapsed = time.time() - t0
        row = {
            "model":         name,
            "roc_auc_mean":  np.mean(fold_aucs),
            "roc_auc_std":   np.std(fold_aucs),
            "pr_auc_mean":   np.mean(fold_prauc),
            "pr_auc_std":    np.std(fold_prauc),
            "f1_mean":       np.mean(fold_f1),
            "f1_std":        np.std(fold_f1),
            "time_s":        round(elapsed, 1),
        }
        results.append(row)
        log.info("  %s → ROC-AUC=%.4f ± %.4f  [%.0fs]",
                 name, row["roc_auc_mean"], row["roc_auc_std"], elapsed)

    return results


def train_champion(champion_name: str, candidates, X_train: np.ndarray,
                   y_train: np.ndarray) -> object:
    for name, pipeline in candidates:
        if name == champion_name:
            log.info("Training champion on full train set: %s", name)
            pipeline.fit(X_train, y_train)
            return pipeline
    raise ValueError(f"Champion '{champion_name}' not found in candidates.")


def evaluate_on_val(pipeline, X_val: np.ndarray, y_val: np.ndarray,
                    feature_cols: list[str]) -> dict:
    proba = pipeline.predict_proba(X_val)[:, 1]

    # Threshold optimisation (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_val, proba)
    j_scores  = tpr - fpr
    best_thr  = float(thresholds[np.argmax(j_scores)])
    pred_opt  = (proba >= best_thr).astype(int)

    metrics = {
        "roc_auc":    roc_auc_score(y_val, proba),
        "pr_auc":     average_precision_score(y_val, proba),
        "f1":         f1_score(y_val, pred_opt, zero_division=0),
        "precision":  precision_score(y_val, pred_opt, zero_division=0),
        "recall":     recall_score(y_val, pred_opt, zero_division=0),
        "threshold":  best_thr,
    }
    log.info("Val metrics: ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f  thr=%.3f",
             metrics["roc_auc"], metrics["pr_auc"], metrics["f1"], metrics["threshold"])
    return metrics


def plot_cv_comparison(cv_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("CV Model Comparison — KKBox Churn", fontsize=13, fontweight="bold")
    metrics = [("roc_auc_mean", "roc_auc_std", "ROC-AUC"),
               ("pr_auc_mean",  "pr_auc_std",  "PR-AUC"),
               ("f1_mean",      "f1_std",       "F1 Score")]

    for ax, (col, std_col, title) in zip(axes, metrics):
        df_s = cv_df.sort_values(col, ascending=True)
        colors = ["#3949AB" if i == len(df_s) - 1 else "#78909C" for i in range(len(df_s))]
        ax.barh(df_s["model"], df_s[col], xerr=df_s[std_col],
                color=colors, alpha=0.88, capsize=4)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
        ax.axvline(df_s[col].max(), color="gold", linestyle="--", linewidth=1.2, alpha=0.7)

    plt.tight_layout()
    path = REPORT_DIR / "cv_model_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


def plot_roc_pr(pipeline, X_val: np.ndarray, y_val: np.ndarray) -> None:
    proba = pipeline.predict_proba(X_val)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Champion Model — ROC & PR Curves (KKBox Churn)", fontsize=12, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_val, proba)
    auc_val = roc_auc_score(y_val, proba)
    axes[0].plot(fpr, tpr, color="#3949AB", lw=2, label=f"ROC-AUC = {auc_val:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)
    axes[1].plot(rec, prec, color="#43A047", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    axes[1].axhline(y_val.mean(), color="red", linestyle="--", lw=1, label=f"Baseline ({y_val.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    path = REPORT_DIR / "roc_pr_curves.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case D: KKBox Churn — Model Training")
    print("=" * 65 + "\n")

    df_train, df_val = load_data()
    feat_cols = get_feature_columns(df_train)
    log.info("%d features | Train=%s  Val=%s", len(feat_cols), df_train.shape, df_val.shape)

    X_train = df_train[feat_cols].fillna(0).values.astype(np.float32)
    y_train = df_train[TARGET].values
    X_val   = df_val[feat_cols].fillna(0).values.astype(np.float32)
    y_val   = df_val[TARGET].values

    churn_rate = float(y_train.mean())
    log.info("Train churn rate: %.3%", churn_rate)

    print("[1] Building candidate models…")
    candidates = build_candidates(churn_rate)

    print(f"[2] Running {CV_FOLDS}-fold stratified CV ({len(candidates)} models)…")
    cv_results = run_cv(candidates, X_train, y_train)

    cv_df = pd.DataFrame(cv_results).sort_values("roc_auc_mean", ascending=False)
    cv_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print("\n" + cv_df[["model","roc_auc_mean","roc_auc_std","pr_auc_mean","f1_mean"]].to_string(index=False))

    champion_name = cv_df.iloc[0]["model"]
    log.info("Champion: %s (ROC-AUC=%.4f)", champion_name, cv_df.iloc[0]["roc_auc_mean"])

    print(f"\n[3] Training champion ({champion_name}) on full train set…")
    champion = train_champion(champion_name, candidates, X_train, y_train)

    print("[4] Evaluating on validation set…")
    val_metrics = evaluate_on_val(champion, X_val, y_val, feat_cols)
    plot_roc_pr(champion, X_val, y_val)

    # Persist
    joblib.dump(champion, MODEL_DIR / "champion.pkl")
    joblib.dump(feat_cols, MODEL_DIR / "feature_cols.pkl")
    pd.DataFrame([val_metrics]).to_csv(REPORT_DIR / "champion_val_metrics.csv", index=False)
    log.info("Champion saved → models/use_case_D/champion.pkl")

    print("[5] Plotting CV comparison…")
    plot_cv_comparison(cv_df)

    print("\n" + "=" * 65)
    print(f"  Step 4 complete. Champion: {champion_name}")
    print(f"  Val ROC-AUC: {val_metrics['roc_auc']:.4f}  PR-AUC: {val_metrics['pr_auc']:.4f}")
    print("  Ready for hyperparameter tuning (05_hyperparameter_tuning.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
