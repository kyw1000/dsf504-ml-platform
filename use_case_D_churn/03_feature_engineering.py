"""
use_case_D_churn/03_feature_engineering.py
============================================
DSF504 — Use Case D: Customer Churn Prediction (KKBox)
Step 3: Feature Engineering

Engineering strategy
--------------------
1. Listening engagement ratios
   - completion_rate   : num_100 / total songs played (engagement quality)
   - skip_rate         : num_25  / total songs (users who skip often)
   - deep_listen_rate  : (num_75 + num_985 + num_100) / total songs
   - variety_ratio     : num_unq / (total songs + 1) (diversity of listening)

2. Subscription features
   - is_discounted     : actual_paid < plan_list_price
   - discount_depth    : (plan_price - actual_paid) / plan_price
   - days_to_expiry    : membership_expire_date - last_txn_date (relative tenure)
   - is_long_plan      : plan_days_mean >= 30

3. Demographic features
   - age_clean         : bd clipped to [7, 80], NaN-filled with median
   - age_bucket        : young (<25), adult (25-45), senior (45+)
   - is_male / is_female / gender_unknown
   - city_churn_rate   : target-encoded city (mean churn rate — train-only)
   - reg_channel_risk   : registered_via target-encoded

4. Activity recency
   - log_days_log      : log1p(log_days) — capture engagement depth
   - secs_per_song     : total_secs_mean / (num_unq_mean + 1)

5. Missing value imputation
   - Numeric → median of train set
   - Categorical → mode of train set

Academic grounding
------------------
- Verbeke et al. (2012): engagement depth is the strongest churn predictor
- Buckinx & Van den Poel (2005): RFM-derived features dominate churn models
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE
from utils.data_loader import smart_split

from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR    / "kkbox_churn"
REPORT_DIR  = REPORTS_DIR / "use_case_D"
MODEL_DIR   = MODELS_DIR  / "use_case_D"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET      = "is_churn"
DROP_COLS   = {"msno", "gender", "registration_init_time", "last_expire_date", "last_txn_date"}


# ── Feature group 1: Listening engagement ratios ───────────────────────────────

def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_songs = (
        df.get("num_25_mean", 0) + df.get("num_50_mean", 0) +
        df.get("num_75_mean", 0) + df.get("num_985_mean", 0) +
        df.get("num_100_mean", 0)
    )
    total_songs = total_songs.replace(0, np.nan)

    if "num_100_mean" in df.columns:
        df["fe_completion_rate"]  = (df["num_100_mean"] / total_songs).fillna(0).clip(0, 1)
    if "num_25_mean" in df.columns:
        df["fe_skip_rate"]        = (df["num_25_mean"]  / total_songs).fillna(0).clip(0, 1)
    for col in ["num_75_mean", "num_985_mean", "num_100_mean"]:
        if col not in df.columns:
            df[col] = 0.0
    df["fe_deep_listen_rate"] = (
        (df["num_75_mean"] + df["num_985_mean"] + df["num_100_mean"]) / total_songs
    ).fillna(0).clip(0, 1)
    if "num_unq_mean" in df.columns:
        df["fe_variety_ratio"] = (df["num_unq_mean"] / (total_songs + 1)).fillna(0)
    if "total_secs_mean" in df.columns:
        df["fe_log_secs"]      = np.log1p(df["total_secs_mean"].fillna(0))
    if "log_days" in df.columns:
        df["fe_log_days_log"]  = np.log1p(df["log_days"].fillna(0))
    if "total_secs_mean" in df.columns and "num_unq_mean" in df.columns:
        df["fe_secs_per_song"] = (
            df["total_secs_mean"] / (df["num_unq_mean"].fillna(0) + 1)
        ).fillna(0)

    log.info("✓ Engagement features added")
    return df


# ── Feature group 2: Subscription plan features ────────────────────────────────

def add_subscription_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "plan_list_price" in df.columns and "actual_paid_mean" in df.columns:
        df["fe_is_discounted"]  = (df["actual_paid_mean"] < df["plan_list_price"]).astype(np.int8)
        denom = df["plan_list_price"].replace(0, np.nan)
        df["fe_discount_depth"] = ((df["plan_list_price"] - df["actual_paid_mean"]) / denom).fillna(0).clip(0, 1)
    if "plan_days_mean" in df.columns:
        df["fe_is_long_plan"]   = (df["plan_days_mean"] >= 30).astype(np.int8)
        df["fe_plan_days_log"]  = np.log1p(df["plan_days_mean"].fillna(0))
    if "txn_count" in df.columns:
        df["fe_txn_count_log"]  = np.log1p(df["txn_count"].fillna(0))
    if "auto_renew_rate" in df.columns:
        df["fe_auto_renew"]     = df["auto_renew_rate"].fillna(df["auto_renew_rate"].median())
    if "cancel_rate" in df.columns:
        df["fe_cancel_rate"]    = df["cancel_rate"].fillna(0)

    log.info("✓ Subscription features added")
    return df


# ── Feature group 3: Demographic features ─────────────────────────────────────

def add_demographic_features(df: pd.DataFrame, train_churn_rates: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    if "bd" in df.columns:
        age_median = df["bd"].median() if df["bd"].notna().any() else 30
        df["fe_age"] = df["bd"].clip(7, 80).fillna(age_median).astype(np.float32)
        df["fe_age_bucket_young"]  = (df["fe_age"] < 25).astype(np.int8)
        df["fe_age_bucket_senior"] = (df["fe_age"] >= 45).astype(np.int8)

    if "gender" in df.columns:
        df["fe_is_male"]    = (df["gender"] == "male").astype(np.int8)
        df["fe_is_female"]  = (df["gender"] == "female").astype(np.int8)

    # Target-encode city using train-set churn rates
    if "city" in df.columns:
        if train_churn_rates and "city" in train_churn_rates:
            df["fe_city_risk"] = df["city"].map(train_churn_rates["city"]).fillna(
                train_churn_rates.get("city_global", 0.084)
            ).astype(np.float32)
        else:
            df["fe_city_risk"] = df["city"].fillna(-1).astype(np.float32)

    if "registered_via" in df.columns:
        if train_churn_rates and "registered_via" in train_churn_rates:
            df["fe_reg_channel_risk"] = df["registered_via"].map(
                train_churn_rates["registered_via"]
            ).fillna(train_churn_rates.get("city_global", 0.084)).astype(np.float32)
        else:
            df["fe_reg_channel_risk"] = df["registered_via"].fillna(-1).astype(np.float32)

    log.info("✓ Demographic features added")
    return df


# ── Feature group 4: Imputation ────────────────────────────────────────────────

def impute_and_drop(df: pd.DataFrame, medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=to_drop)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET]

    if medians is None:
        medians = {c: float(df[c].median()) for c in num_cols}

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(medians.get(col, 0)).astype(np.float32)

    # Drop remaining non-numeric columns except target
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    obj_cols = [c for c in obj_cols if c != TARGET]
    if obj_cols:
        log.info("Dropping remaining categorical: %s", obj_cols)
        df = df.drop(columns=obj_cols)

    log.info("✓ Imputation done. Shape: %s", df.shape)
    return df, medians


# ── Main pipeline ──────────────────────────────────────────────────────────────

def engineer(df: pd.DataFrame, train_churn_rates: dict | None = None,
             medians: dict | None = None, fit: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    df = add_engagement_features(df)
    df = add_subscription_features(df)
    df = add_demographic_features(df, train_churn_rates)
    df, medians = impute_and_drop(df, medians)
    return df, train_churn_rates or {}, medians


def plot_engineered_features(df_raw: pd.DataFrame, df_fe: pd.DataFrame) -> None:
    """Compare raw vs engineered feature distributions."""
    new_cols = [c for c in df_fe.columns if c.startswith("fe_") and c in df_fe.columns][:8]
    if not new_cols:
        return
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle("Engineered Features — Distribution by Churn", fontsize=13, fontweight="bold")
    for i, col in enumerate(new_cols[:8]):
        ax = axes[i]
        for label, clr, lbl in [(0, "#43A047", "Retained"), (1, "#E53935", "Churned")]:
            vals = df_fe[df_fe[TARGET] == label][col].dropna()
            p99  = vals.quantile(0.99) if len(vals) else 1
            ax.hist(vals.clip(upper=p99), bins=40, alpha=0.6, color=clr, label=lbl, density=True)
        ax.set_title(col.replace("fe_", "").replace("_", " ").title(), fontsize=9)
        ax.legend(fontsize=7)
    for j in range(len(new_cols), 8):
        axes[j].set_visible(False)
    plt.tight_layout()
    path = REPORT_DIR / "engineered_feature_summary.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path.name)


def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case D: KKBox Churn — Feature Engineering")
    print("=" * 65 + "\n")

    # Load raw splits
    train_path = DATA_SUBDIR / "train_raw.parquet"
    val_path   = DATA_SUBDIR / "val_raw.parquet"
    if not train_path.exists():
        raise FileNotFoundError("Run 01_data_loading.py first.")
    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)

    log.info("Train: %s  Val: %s", df_train.shape, df_val.shape)

    # Compute target-encoding rates from TRAIN only (prevent leakage)
    train_churn_rates: dict = {
        "city_global": float(df_train[TARGET].mean()),
    }
    for cat_col in ["city", "registered_via"]:
        if cat_col in df_train.columns:
            train_churn_rates[cat_col] = df_train.groupby(cat_col)[TARGET].mean().to_dict()

    print("[1] Engineering train set…")
    df_train_fe, _, medians = engineer(df_train, train_churn_rates, fit=True)

    print("[2] Engineering validation set (using train medians)…")
    df_val_fe, _, _ = engineer(df_val, train_churn_rates, medians=medians, fit=False)

    # Save feature list
    feat_cols = [c for c in df_train_fe.columns if c != TARGET]
    feat_df   = pd.DataFrame({"feature": feat_cols, "dtype": [str(df_train_fe[c].dtype) for c in feat_cols]})
    feat_df.to_csv(REPORT_DIR / "engineered_features_list.csv", index=False)
    log.info("%d features saved to engineered_features_list.csv", len(feat_cols))

    # Save artefacts
    df_train_fe.to_parquet(DATA_SUBDIR / "train_fe.parquet", index=False)
    df_val_fe.to_parquet(DATA_SUBDIR   / "val_fe.parquet",   index=False)
    joblib.dump(medians, MODEL_DIR / "medians.pkl")
    joblib.dump(train_churn_rates, MODEL_DIR / "churn_rates.pkl")
    log.info("FE parquets + artefacts saved")

    print("[3] Plotting engineered features…")
    plot_engineered_features(df_train, df_train_fe)

    print("\n" + "=" * 65)
    print("  Step 3 complete. Ready for model training (04_model_training.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
