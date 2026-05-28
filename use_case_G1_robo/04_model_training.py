"""
use_case_G1_robo/04_model_training.py
=======================================
Use Case G1 — Robo-Advisory Portfolio Recommendation
Phase 3, Step 4: Algorithm Selection & Cross-Validation

Models evaluated
----------------
  Baseline  : Popularity-based recommender (non-personalised)
  CF-SVD    : Collaborative Filtering via Truncated SVD on interaction matrix
  LGB-Rank  : LightGBM LambdaRank (Learning-to-Rank; primary model)

Evaluation metrics (recommendation-specific)
---------------------------------------------
  NDCG@10   : Normalized Discounted Cumulative Gain at top-10
  Precision@10 : Fraction of top-10 that are relevant
  Recall@10    : Fraction of relevant items captured in top-10
  MRR          : Mean Reciprocal Rank (position of first hit)

Run
---
    cd C:\\DSF504
    python use_case_G1_robo/04_model_training.py
"""

from __future__ import annotations

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR / "far_trans"
REPORT_DIR  = REPORTS_DIR / "use_case_G1"
MODEL_DIR   = MODELS_DIR / "use_case_G1"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    log.warning("LightGBM not available.")

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """NDCG@k for a single query."""
    order = np.argsort(y_score)[::-1][:k]
    gains  = y_true[order]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg  = (gains / discounts).sum()
    ideal_gains = np.sort(y_true)[::-1][:k]
    idcg = (ideal_gains / discounts[:len(ideal_gains)]).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(y_score)[::-1][:k]
    return float(y_true[order].sum() / k)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(y_score)[::-1][:k]
    total = y_true.sum()
    return float(y_true[order].sum() / total) if total > 0 else 0.0


def mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    for rank, idx in enumerate(order, 1):
        if y_true[idx] == 1:
            return 1.0 / rank
    return 0.0


def evaluate_recommendations(df: pd.DataFrame, score_col: str, k: int = 10) -> dict:
    """Evaluate a recommendation model across all queries (customers)."""
    metrics = {"ndcg": [], "precision": [], "recall": [], "mrr": []}
    for _, grp in df.groupby("query_id"):
        yt = grp["label"].values
        ys = grp[score_col].values
        if yt.sum() == 0:
            continue
        metrics["ndcg"].append(ndcg_at_k(yt, ys, k))
        metrics["precision"].append(precision_at_k(yt, ys, k))
        metrics["recall"].append(recall_at_k(yt, ys, k))
        metrics["mrr"].append(mrr(yt, ys))
    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def popularity_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Rank all items by global popularity (non-personalised)."""
    pop_scores = (train_df[train_df["label"] == 1]
                  .groupby("isin")["label"].count()
                  .reset_index(name="pop_score"))
    val_scored = val_df.merge(pop_scores, on="isin", how="left").fillna(0)
    metrics = evaluate_recommendations(val_scored, "pop_score")
    log.info(f"Popularity Baseline: NDCG@10={metrics['ndcg']:.4f}")
    return metrics


def svd_collaborative_filter(train_df: pd.DataFrame, val_df: pd.DataFrame,
                              n_components: int = 50) -> tuple[dict, TruncatedSVD]:
    """Matrix factorisation via Truncated SVD."""
    # Build interaction matrix from train positives
    pos = train_df[train_df["label"] == 1][["customer_id", "isin"]].drop_duplicates()
    all_customers = sorted(pos["customer_id"].unique())
    all_items     = sorted(pos["isin"].unique())
    c_idx = {c: i for i, c in enumerate(all_customers)}
    i_idx = {v: i for i, v in enumerate(all_items)}

    rows = pos["customer_id"].map(c_idx).values
    cols = pos["isin"].map(i_idx).values
    from scipy.sparse import csr_matrix
    R = csr_matrix((np.ones(len(rows)), (rows, cols)),
                   shape=(len(all_customers), len(all_items)))

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    U   = svd.fit_transform(R)  # (n_customers, n_components)
    V   = svd.components_        # (n_components, n_items)
    R_hat = U @ V                # reconstructed scores

    # Score val pairs
    def get_score(row):
        ci = c_idx.get(row["customer_id"], -1)
        ii = i_idx.get(row["isin"], -1)
        if ci < 0 or ii < 0:
            return 0.0
        return float(R_hat[ci, ii])

    val_scored = val_df.copy()
    val_scored["svd_score"] = [get_score(r) for r in val_scored.to_dict("records")]
    metrics = evaluate_recommendations(val_scored, "svd_score")
    log.info(f"SVD CF: NDCG@10={metrics['ndcg']:.4f}")
    return metrics, svd


def lgb_lambdarank(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict, object]:
    """LightGBM LambdaRank — Learning-to-Rank per customer query."""
    feat_cols = [c for c in train_df.columns
                 if c not in ["customer_id", "isin", "label", "query_id", "_count"]]
    X_tr = train_df[feat_cols].fillna(0).values
    y_tr = train_df["label"].values
    q_tr = train_df.groupby("query_id").size().values  # group sizes for LambdaRank

    X_va = val_df[feat_cols].fillna(0).values
    y_va = val_df["label"].values

    model = lgb.LGBMRanker(
        objective       = "lambdarank",
        n_estimators    = 200,
        num_leaves      = 31,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        min_child_samples=10,
        random_state    = RANDOM_STATE,
        n_jobs          = 1,
        verbose         = -1,
    )
    model.fit(X_tr, y_tr, group=q_tr)

    val_scored = val_df.copy()
    val_scored["lgb_score"] = model.predict(X_va)
    metrics = evaluate_recommendations(val_scored, "lgb_score")
    log.info(f"LGB LambdaRank: NDCG@10={metrics['ndcg']:.4f}")
    return metrics, model


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(results: dict) -> None:
    metrics = ["ndcg", "precision", "recall", "mrr"]
    models  = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.25
    colors  = ["#78909C", "#1976D2", "#388E3C"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [results[model].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["NDCG@10", "Precision@10", "Recall@10", "MRR"])
    ax.set_ylabel("Score")
    ax.set_title("Recommendation Model Comparison (@k=10)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.1 * max(v for r in results.values() for v in r.values()) + 0.05)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved model_comparison.png")


def plot_lgb_feature_importance(model, feat_cols: list) -> None:
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    top = imp.head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    top.sort_values().plot(kind="barh", ax=ax, color="#1565C0")
    ax.set_title("LightGBM Feature Importance (Top 20)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (split)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "lgb_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved lgb_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G1: Robo-Advisory Portfolio Recommendation")
    print("  Step 4: Algorithm Selection & Evaluation")
    print("=" * 65 + "\n")

    train_df = pd.read_parquet(DATA_SUBDIR / "train_pairs.parquet")
    val_df   = pd.read_parquet(DATA_SUBDIR / "val_pairs.parquet")
    print(f"[1] Train pairs: {len(train_df):,}  |  Val pairs: {len(val_df):,}")
    print(f"    Label balance (train): {train_df['label'].mean():.3f} positive")

    results = {}

    print("\n[2] Popularity Baseline…")
    results["Popularity"] = popularity_baseline(train_df, val_df)

    print("[3] SVD Collaborative Filtering…")
    try:
        results["SVD-CF"], svd_model = svd_collaborative_filter(train_df, val_df, n_components=30)
    except Exception as e:
        log.warning(f"SVD failed: {e}")
        results["SVD-CF"] = {"ndcg": 0, "precision": 0, "recall": 0, "mrr": 0}
        svd_model = None

    print("[4] LightGBM LambdaRank…")
    if LGB_AVAILABLE:
        results["LGB-LambdaRank"], lgb_model = lgb_lambdarank(train_df, val_df)
    else:
        log.warning("LightGBM not available — skipping.")
        results["LGB-LambdaRank"] = {"ndcg": 0, "precision": 0, "recall": 0, "mrr": 0}
        lgb_model = None

    print("\n[5] Results summary:")
    print(f"  {'Model':<20} {'NDCG@10':>8} {'P@10':>8} {'R@10':>8} {'MRR':>8}")
    print("  " + "-" * 56)
    for model_name, m in results.items():
        print(f"  {model_name:<20} {m['ndcg']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['mrr']:>8.4f}")

    # Champion = LGB if available, else SVD
    if LGB_AVAILABLE and lgb_model is not None:
        champion = lgb_model
        champion_name = "LGB-LambdaRank"
    elif svd_model is not None:
        champion = svd_model
        champion_name = "SVD-CF"
    else:
        champion = None
        champion_name = "Popularity"

    print(f"\n  Champion: {champion_name}")

    print("\n[6] Saving champion model…")
    if champion is not None:
        joblib.dump(champion, MODEL_DIR / "champion.pkl")
    (MODEL_DIR / "champion_name.txt").write_text(champion_name)

    comparison_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    comparison_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    print("[7] Generating comparison plot…")
    plot_model_comparison(results)
    if LGB_AVAILABLE and lgb_model is not None:
        feat_cols = [c for c in train_df.columns
                     if c not in ["customer_id", "isin", "label", "query_id", "_count"]]
        plot_lgb_feature_importance(lgb_model, feat_cols)

    print("=" * 65)
    print("  Step 4 complete. Ready for Hyperparameter Tuning (05_)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
