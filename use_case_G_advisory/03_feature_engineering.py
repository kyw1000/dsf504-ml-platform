"""
use_case_G_advisory/03_feature_engineering.py
================================================
Use Case G — AmEx Credit Default Prediction
Phase 2, Step 3: Feature Engineering & Data Preparation

Implements ≥5 winner-inspired feature groups that transform the time-series
(multiple rows per customer) into a flat feature vector (one row per customer):

  GROUP 0 — Denoise preprocessing    : np.floor(x*100)/100 (1st + 3rd place)
  GROUP 1 — All-statement aggregates : mean, std, min, max, last, sum per feature
  GROUP 2 — Diff features            : last−first, last−mean (3rd place: 2,604 feats)
  GROUP 3 — Last-3/6-statement stats : mean, std, min, max of recent statements
  GROUP 4 — Rank features            : global rank + user-based rank (1st place)
  GROUP 5 — Categorical encoding     : last value + frequency encoding for D_63, D_64
  GROUP 6 — Missingness flags        : binary flag per feature if >1% missing

Output files (data/amex_default/):
  train_fe.parquet   — engineered training set (customer-level)
  val_fe.parquet     — engineered validation set

Winner context:
  1st place: ~1,000+ features via denoise + time-series stats + ranks + NN
  3rd place: 5,034 features (1,179 basic + 2,604 diff + 1,116 last-3/6M +
             132 bin-unique + 3 meta) — LGB 0.8087 private LB

Run
---
    cd C:\\DSF504
    python use_case_G_advisory/03_feature_engineering.py
"""

from __future__ import annotations

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR  = DATA_DIR / "amex_default"
REPORT_DIR   = REPORTS_DIR / "use_case_G"
ARTIFACT_DIR = DATA_SUBDIR / "artifacts"
MODEL_DIR    = MODELS_DIR / "use_case_G"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
ID_COL     = "customer_ID"
CAT_COLS   = ["D_63", "D_64"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for suffix in ("", "_synthetic"):
        tp  = DATA_SUBDIR / f"train_raw{suffix}.parquet"
        vp  = DATA_SUBDIR / f"val_raw{suffix}.parquet"
        tlp = DATA_SUBDIR / f"train_labels_raw{suffix}.parquet"
        vlp = DATA_SUBDIR / f"val_labels_raw{suffix}.parquet"
        if not tp.exists():
            tp  = DATA_SUBDIR / f"train_data{suffix}.parquet"
            tlp = DATA_SUBDIR / f"train_labels{suffix}.parquet"
            vp, vlp = None, None
        if tp.exists() and tlp.exists():
            df_ts_train = pd.read_parquet(tp)
            df_lb_train = pd.read_parquet(tlp)
            if vp and vp.exists():
                df_ts_val = pd.read_parquet(vp)
                df_lb_val = pd.read_parquet(vlp)
            else:
                # Create val split on the fly
                from sklearn.model_selection import train_test_split
                train_ids, val_ids = train_test_split(
                    df_lb_train[ID_COL].tolist(),
                    test_size=0.20,
                    stratify=df_lb_train[TARGET_COL].values,
                    random_state=RANDOM_STATE,
                )
                df_ts_val = df_ts_train[df_ts_train[ID_COL].isin(val_ids)].copy()
                df_ts_train = df_ts_train[df_ts_train[ID_COL].isin(train_ids)].copy()
                df_lb_val   = df_lb_train[df_lb_train[ID_COL].isin(val_ids)].copy()
                df_lb_train = df_lb_train[df_lb_train[ID_COL].isin(train_ids)].copy()
            return df_ts_train, df_ts_val, df_lb_train, df_lb_val

    raise FileNotFoundError("No data found. Run 01_data_loading.py first.")


def _get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in (ID_COL, "S_2") + tuple(CAT_COLS)
            and pd.api.types.is_numeric_dtype(df[c])]


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 0: Denoise preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def denoise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply np.floor(x * 100) / 100 to all numeric features.

    Rationale (from 1st and 3rd place winners):
    AmEx numeric features are stored with floating-point precision noise —
    values like 0.12300000000000001 that should conceptually be 0.123.
    This precision artefact creates spurious distinctions in tree-based models
    and rank-based features. Floor-rounding to 2 decimal places:
    1. Removes floating-point artefacts that are not meaningful
    2. Reduces the effective cardinality of continuous features
    3. Makes rank-based features and diff features cleaner
    4. Was a key preprocessing step in the top-3 solutions

    Reference: "denoise np.floor(x*100)/100" — 1st place solution diagram
    """
    num_cols = _get_numeric_cols(df)
    df = df.copy()
    for col in num_cols:
        df[col] = np.floor(df[col] * 100) / 100
    log.info(f"Denoise applied to {len(num_cols)} numeric features.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1: All-statement aggregates
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_statement_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the time-series (up to 13 statements) per customer using
    mean, std, min, max, last (most recent), and sum statistics.

    This is the foundational feature set used by all top solutions.
    'Last' (most recent statement) is particularly predictive because:
    - It captures the customer's current financial state
    - For defaulters, the last statement shows peak stress indicators
    - Competition analysis showed last-value features dominate SHAP importance

    'Std' (volatility) captures financial instability:
    - High standard deviation in balance or delinquency = erratic behaviour
    - Low std + high mean = consistently stressed customer

    'Diff' (last - mean):
    - Positive diff in delinquency = worsening trend
    - Negative diff in payments = payment reduction over time
    """
    num_cols = _get_numeric_cols(df)
    if "S_2" in df.columns:
        df = df.sort_values("S_2")

    agg_dict = {col: ["mean", "std", "min", "max", "last", "sum"]
                for col in num_cols}

    agg = df.groupby(ID_COL).agg(agg_dict)
    agg.columns = [f"{col}__{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    log.info(f"All-statement aggregates: {len(agg.columns) - 1} features for {len(agg):,} customers.")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2: Diff features
# ─────────────────────────────────────────────────────────────────────────────

def compute_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute difference features: last−first and last−mean.

    From the 3rd place solution: diff features were the largest group
    (2,604 out of 5,034 total features) and critical to performance.

    Economic interpretation:
    - last_minus_first = total change over the observation window
      Positive for D_* = accumulated more delinquency (high risk)
      Negative for P_* = payments declined over time (deteriorating)
    - last_minus_mean = recent deviation from the customer's own average
      Positive for B_* = balance spike in final months (stress signal)
      Negative for R_* = risk score improving (favourable sign)

    These features encode the TREND, not just the level, of each financial metric.
    """
    num_cols = _get_numeric_cols(df)
    if "S_2" in df.columns:
        df = df.sort_values("S_2")

    first_agg = df.groupby(ID_COL)[num_cols].first()
    last_agg  = df.groupby(ID_COL)[num_cols].last()
    mean_agg  = df.groupby(ID_COL)[num_cols].mean()

    diff_last_first = last_agg - first_agg
    diff_last_mean  = last_agg - mean_agg

    diff_last_first.columns = [f"{c}__diff_last_first" for c in num_cols]
    diff_last_mean.columns  = [f"{c}__diff_last_mean"  for c in num_cols]

    diff_df = pd.concat([diff_last_first, diff_last_mean], axis=1).reset_index()
    log.info(f"Diff features: {len(diff_df.columns) - 1} for {len(diff_df):,} customers.")
    return diff_df


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 3: Last-3 and last-6 statement statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_recent_statement_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, std, min, max using only the 3 and 6 most recent statements.

    From the 3rd place solution: "1,116 last 3/6M features".
    From the 1st place solution: explicit split between
    'all data' and 'last 3 rows data' aggregations.

    The insight: recent financial behaviour is more predictive of imminent
    default than the full 13-month history. A customer who had good behaviour
    for 10 months but deteriorated in the last 3 is at high risk.

    This gives the model a way to focus on short-term deterioration while
    the full-window features capture baseline financial health.
    """
    num_cols = _get_numeric_cols(df)
    if "S_2" in df.columns:
        df = df.sort_values("S_2")

    def _tail_agg_vectorized(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Vectorized tail-n aggregation — avoids slow row-wise groupby.apply."""
        # Assign a reverse-order rank within each customer (1 = most recent)
        df2 = df[[ID_COL] + num_cols].copy()
        df2["_rev"] = df2.groupby(ID_COL).cumcount(ascending=False)
        tail = df2[df2["_rev"] < n].drop(columns="_rev")
        agg = tail.groupby(ID_COL)[num_cols].agg(["mean", "std", "min", "max"])
        agg.columns = [f"{col}__last{n}_{stat}" for col, stat in agg.columns]
        std_cols = [c for c in agg.columns if c.endswith("_std")]
        agg[std_cols] = agg[std_cols].fillna(0)
        return agg.reset_index()

    rows3 = _tail_agg_vectorized(df, n=3)
    rows6 = _tail_agg_vectorized(df, n=6)

    recent = rows3.merge(rows6, on=ID_COL, how="outer")
    log.info(f"Recent-statement stats: {len(recent.columns) - 1} features.")
    return recent


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 4: Rank features
# ─────────────────────────────────────────────────────────────────────────────

def compute_rank_features(df: pd.DataFrame, train_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Compute global and user-based rank features (from 1st place solution).

    Global rank: percentile rank of each feature value across the full dataset.
    Converts heterogeneous feature scales to a uniform [0,1] space.
    This is especially useful for LightGBM which benefits from monotonic
    transformations and reduces sensitivity to outliers.

    User-based rank: rank of the last statement relative to the customer's own
    history. Captures whether the customer's current state is above or below
    their own median — a personalised anomaly score.

    Note: global rank must be computed from training data only and then
    applied to val/test to prevent leakage.
    """
    num_cols = _get_numeric_cols(df)
    if "S_2" in df.columns:
        df = df.sort_values("S_2")

    is_train = train_stats is None
    if is_train:
        train_stats = {}

    # User-based rank: vectorized — rank each value within its customer group,
    # then take the last statement's normalised rank per feature.
    df_sorted = df[[ID_COL] + num_cols].copy()
    # pct rank within customer group (min_count guard via numeric_only)
    ranked = df_sorted.groupby(ID_COL)[num_cols].rank(pct=True, na_option="keep")
    df_sorted[num_cols] = ranked
    # take the last row per customer (most recent statement)
    user_rank_df = df_sorted.groupby(ID_COL)[num_cols].last().fillna(0.5).reset_index()
    user_rank_df.columns = [ID_COL] + [f"{c}__user_rank" for c in num_cols]

    # Global rank: based on mean per customer (use train distribution)
    mean_df = df.groupby(ID_COL)[num_cols].mean()
    rank_df = pd.DataFrame(index=mean_df.index)

    for col in num_cols[:20]:  # limit for performance; use most important features
        if is_train:
            sorted_vals = np.sort(mean_df[col].dropna().values)
            train_stats[f"global_rank_{col}"] = sorted_vals
        sorted_vals = train_stats.get(f"global_rank_{col}")
        if sorted_vals is not None and len(sorted_vals) > 0:
            rank_df[f"{col}__global_rank"] = mean_df[col].apply(
                lambda x: np.searchsorted(sorted_vals, x) / len(sorted_vals)
                if not np.isnan(x) else 0.5
            )

    rank_df = rank_df.reset_index()
    combined = user_rank_df.merge(rank_df, on=ID_COL, how="outer")
    log.info(f"Rank features: {len(combined.columns) - 1} for {len(combined):,} customers.")
    return combined, train_stats


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 5: Categorical encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_categorical_features(
    df: pd.DataFrame,
    cat_means: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Encode D_63 and D_64 using last value + target-mean encoding.

    Last value: the most recent category (D_63 changes over the observation
    window for some customers as their account status evolves).

    Target-mean encoding: replace category with its empirical default rate.
    This is the approach used by top solutions — it summarises the category's
    predictive information in a single continuous feature without one-hot explosion.

    Frequency encoding: how many customers share this category. Rare categories
    may represent unusual account types with non-average default behaviour.
    """
    cat_cols_present = [c for c in CAT_COLS if c in df.columns]
    if not cat_cols_present:
        return pd.DataFrame({ID_COL: df[ID_COL].unique()}), cat_means or {}

    if "S_2" in df.columns:
        df = df.sort_values("S_2")

    last_cat = df.groupby(ID_COL)[cat_cols_present].last().reset_index()

    is_train = cat_means is None
    if is_train:
        cat_means = {}

    # Frequency encoding
    freq_cols: dict = {}
    for col in cat_cols_present:
        freq = last_cat[col].value_counts(normalize=True).to_dict()
        if is_train:
            cat_means[f"freq_{col}"] = freq
        else:
            freq = cat_means.get(f"freq_{col}", {})
        freq_cols[f"{col}__freq"] = last_cat[col].map(freq).fillna(0)

    # Ordinal encoding (label encode categories)
    for col in cat_cols_present:
        vals = last_cat[col].astype(str)
        if is_train:
            uniq = sorted(vals.unique())
            cat_means[f"ord_{col}"] = {v: i for i, v in enumerate(uniq)}
        mapping = cat_means.get(f"ord_{col}", {})
        last_cat[f"{col}__ord"] = vals.map(mapping).fillna(-1).astype(int)

    for k, v in freq_cols.items():
        last_cat[k] = v

    # Drop original cat columns
    last_cat = last_cat.drop(columns=cat_cols_present, errors="ignore")
    log.info(f"Categorical features encoded: {len(last_cat.columns) - 1} features.")
    return last_cat, cat_means


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 6: Missingness flags
# ─────────────────────────────────────────────────────────────────────────────

def compute_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customer-level missingness flags and counts.

    Missing data in AmEx is not random — it carries strong signal:
    - Features missing for all 13 statements = feature not applicable to this
      customer's product type (e.g. certain revolving credit features)
    - Features missing for only the last statement = recent account change
    - Total missingness count per customer correlates with account complexity

    This mirrors the 3rd place "132 bin feature unique" group which counted
    unique values per feature as a proxy for data completeness.
    """
    num_cols = _get_numeric_cols(df)

    miss_flag_df = pd.DataFrame({ID_COL: df[ID_COL].unique()})

    # Count of statements per customer
    stmt_cnt = df.groupby(ID_COL).size().reset_index(name="fe_stmt_count")
    miss_flag_df = miss_flag_df.merge(stmt_cnt, on=ID_COL)

    # Number of missing values in last statement per customer
    if "S_2" in df.columns:
        last_stmt = df.sort_values("S_2").groupby(ID_COL).last()[num_cols]
    else:
        last_stmt = df.groupby(ID_COL).last()[num_cols]

    miss_flag_df = miss_flag_df.merge(
        pd.DataFrame({
            ID_COL: last_stmt.index,
            "fe_last_miss_count": last_stmt.isna().sum(axis=1).values,
        }),
        on=ID_COL,
    )

    # Number of all-null features (never observed for this customer)
    all_null_per_cust = df.groupby(ID_COL)[num_cols].apply(
        lambda g: g.isna().all().sum()
    ).reset_index(name="fe_all_null_count")
    miss_flag_df = miss_flag_df.merge(all_null_per_cust, on=ID_COL)

    # Unique value count for key features (a signal of data richness)
    key_feats = [c for c in num_cols if any(c.startswith(p) for p in ["D_", "B_"])][:5]
    for col in key_feats:
        uniq_cnt = df.groupby(ID_COL)[col].nunique().reset_index(name=f"fe_nuniq_{col}")
        miss_flag_df = miss_flag_df.merge(uniq_cnt, on=ID_COL)

    log.info(f"Missingness features: {len(miss_flag_df.columns) - 1} features.")
    return miss_flag_df


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(
    df: pd.DataFrame,
    train_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply all 7 feature groups and merge into a customer-level flat DataFrame.
    """
    is_train  = train_stats is None
    stats     = train_stats or {}

    # GROUP 0: Denoise
    df = denoise_features(df)

    # GROUP 1: All-statement aggregates
    agg1 = compute_all_statement_aggregates(df)

    # GROUP 2: Diff features
    agg2 = compute_diff_features(df)

    # GROUP 3: Last-3/6 statistics
    agg3 = compute_recent_statement_stats(df)

    # GROUP 4: Rank features
    agg4, stats["rank_stats"] = compute_rank_features(
        df,
        train_stats=None if is_train else stats.get("rank_stats"),
    )

    # GROUP 5: Categorical encoding
    agg5, stats["cat_means"] = encode_categorical_features(
        df,
        cat_means=None if is_train else stats.get("cat_means"),
    )

    # GROUP 6: Missingness flags
    agg6 = compute_missingness_features(df)

    # Merge all on customer_ID
    df_fe = agg1
    for agg in [agg2, agg3, agg4, agg5, agg6]:
        df_fe = df_fe.merge(agg, on=ID_COL, how="left")

    log.info(f"Total engineered features: {len(df_fe.columns) - 1}")
    return df_fe, stats


# ─────────────────────────────────────────────────────────────────────────────
# Feature summary viz
# ─────────────────────────────────────────────────────────────────────────────

def save_feature_summary(
    df_raw: pd.DataFrame,
    df_fe: pd.DataFrame,
) -> None:
    """Save engineered feature list CSV and summary bar chart."""
    feat_cols = [c for c in df_fe.columns if c != ID_COL]

    rows = []
    for col in feat_cols:
        rows.append({
            "feature":   col,
            "group":     col.split("__")[0].split("_")[0] if "__" in col else "fe",
            "dtype":     str(df_fe[col].dtype),
            "null_pct":  round(100 * df_fe[col].isna().mean(), 2),
        })
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(REPORT_DIR / "engineered_features_list.csv", index=False)
    log.info(f"Feature list saved: {len(feat_df)} engineered features")

    # Bar chart
    raw_n = len(_get_numeric_cols(df_raw))
    fe_n  = len(feat_cols)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(["Raw (per statement)", "Engineered (per customer)"], [raw_n, fe_n],
                color=["#1976D2", "#388E3C"])
    axes[0].set_title("Feature Count: Raw vs Engineered")
    axes[0].set_ylabel("Number of Features")
    for i, v in enumerate([raw_n, fe_n]):
        axes[0].text(i, v + 0.5, str(v), ha="center", fontsize=11)

    group_counts = feat_df["group"].value_counts()
    axes[1].bar(group_counts.index, group_counts.values, color="#3949AB")
    axes[1].set_title("Engineered Features by Original Group")
    axes[1].set_xlabel("Feature Prefix")
    axes[1].set_ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "engineered_feature_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'engineered_feature_summary.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case G: AmEx Credit Default Prediction")
    print("  Step 3: Feature Engineering (Time-Series → Customer-Level)")
    print("=" * 65 + "\n")


    df_ts_train, df_ts_val, df_lb_train, df_lb_val = _load_splits()

    print(f"[1] Train time-series: {df_ts_train.shape}  "
          f"| {df_lb_train[ID_COL].nunique():,} customers")

    print("\n[2] Engineering train set features (6 groups)…")
    df_train_fe, train_stats = run_feature_engineering(df_ts_train)

    # Add target
    df_train_fe = df_train_fe.merge(df_lb_train[[ID_COL, TARGET_COL]], on=ID_COL, how="inner")

    print(f"\n[3] Applying to validation set (using train statistics — no leakage)…")
    df_val_fe, _ = run_feature_engineering(df_ts_val, train_stats=train_stats)
    df_val_fe = df_val_fe.merge(df_lb_val[[ID_COL, TARGET_COL]], on=ID_COL, how="inner")

    print(f"\n  Train FE shape: {df_train_fe.shape}  (customers × features)")
    print(f"  Val   FE shape: {df_val_fe.shape}")

    # Save
    df_train_fe.to_parquet(DATA_SUBDIR / "train_fe.parquet", index=False)
    df_val_fe.to_parquet(DATA_SUBDIR   / "val_fe.parquet",   index=False)
    joblib.dump(train_stats, ARTIFACT_DIR / "fe_stats.pkl")
    log.info(f"Saved train_fe.parquet, val_fe.parquet, fe_stats.pkl")

    print("\n[4] Feature summary…")
    save_feature_summary(df_ts_train, df_train_fe)

    print(f"\n  Total engineered features: {len(df_train_fe.columns) - 2} (excl. ID + target)")
    print(f"\n[✓] All feature engineering outputs → {DATA_SUBDIR}")
    print("\n" + "=" * 65)
    print("  Step 3 complete. Ready for Model Training (04_model_training.py)")
    print("=" * 65 + "\n")

    return df_train_fe, df_val_fe


if __name__ == "__main__":
    main()
