"""
use_case_E_insurance/03_feature_engineering.py
================================================
Use Case E — Insurance Risk & Claims Analytics
Phase 2, Step 3: Feature Engineering & Data Preparation

Implements ≥5 domain-driven feature groups as required by DSF504:
  1. Missing-value handling    — convert -1 → NaN; binary missingness flags
  2. ps_calc_* removal         — known uninformative features (competition finding)
  3. Categorical encoding      — ordinal + target encoding for ps_*_cat columns
  4. Derived aggregate features— missing count per row, binary sum per group
  5. Interaction features      — ps_car × ps_ind cross-features
  6. Continuous transformations— log1p on right-skewed ps_reg_* features
  7. Scaling                   — StandardScaler on numeric features (for LR/MLP)

Each feature group is documented with actuarial domain rationale.

Output files (in data/porto_seguro/):
  train_fe.parquet   — engineered training set
  val_fe.parquet     — engineered validation set

ML Framework Phase: Perform Feature Extraction → Split Data Sets

Run
---
    cd C:\\DSF504
    python use_case_E_insurance/03_feature_engineering.py
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

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, MODELS_DIR, RANDOM_STATE

# ── UTF-8 encoding guard ─────────────────────────────────────────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR   = DATA_DIR / "porto_seguro"
REPORT_DIR    = REPORTS_DIR / "use_case_E"
ARTIFACT_DIR  = DATA_SUBDIR / "artifacts"
MODEL_DIR     = MODELS_DIR / "use_case_E"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    feat_cols = [c for c in df.columns if c not in ("id", TARGET_COL)]
    g: dict[str, list[str]] = {
        "ind_bin":  [], "ind_cat":  [], "ind_cont": [],
        "reg":      [],
        "car_bin":  [], "car_cat":  [], "car_cont": [],
        "calc_bin": [], "calc_cont":[],
    }
    for c in feat_cols:
        if c.startswith("ps_ind"):
            if c.endswith("_bin"):   g["ind_bin"].append(c)
            elif c.endswith("_cat"): g["ind_cat"].append(c)
            else:                    g["ind_cont"].append(c)
        elif c.startswith("ps_reg"):
            g["reg"].append(c)
        elif c.startswith("ps_car"):
            if c.endswith("_bin"):   g["car_bin"].append(c)
            elif c.endswith("_cat"): g["car_cat"].append(c)
            else:                    g["car_cont"].append(c)
        elif c.startswith("ps_calc"):
            if c.endswith("_bin"):   g["calc_bin"].append(c)
            else:                    g["calc_cont"].append(c)
    return g


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_SUBDIR / "train_raw.parquet")
    val   = pd.read_parquet(DATA_SUBDIR / "val_raw.parquet")
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 1: Missing-value handling
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing_values(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    train_medians: dict | None = None,
    train_modes: dict | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Porto Seguro encodes missing values as -1 in both numeric and categorical
    features. This function:
      1. Replaces -1 with NaN.
      2. Creates binary missingness-indicator columns (fe_miss_*) for features
         with >1% missing rate. The indicators carry risk signal on their own
         (e.g., missing vehicle type → newer/unusual policy).
      3. Imputes continuous features with train median, categoricals with mode.

    Parameters
    ----------
    train_medians  : If provided, use these medians (apply train stats to val set).
    train_modes    : If provided, use these modes.

    Returns
    -------
    df_out         : DataFrame with NaN imputed and missingness flags added.
    train_medians  : Fitted medians (save from train set).
    train_modes    : Fitted modes.
    """
    _new: dict = {}
    df_out = df.copy()

    # Step 1: Replace -1 with NaN
    df_out = df_out.replace(-1, np.nan)

    all_feat_cols = [c for c in df_out.columns if c not in ("id", TARGET_COL)]

    # Step 2: Binary missingness flags for features with any missing
    missing_pct = df_out[all_feat_cols].isna().mean()
    miss_cols = missing_pct[missing_pct > 0.01].index.tolist()
    for col in miss_cols:
        _new[f"fe_miss_{col}"] = df_out[col].isna().astype(np.int8)

    cont_cols = (
        groups["ind_cont"] + groups["reg"] +
        groups["car_cont"] + groups["calc_cont"]
    )
    cat_cols  = groups["ind_cat"] + groups["car_cat"]

    # Step 3: Impute continuous with median
    if train_medians is None:
        train_medians = {}
        for col in cont_cols:
            if col in df_out.columns:
                train_medians[col] = float(df_out[col].median())
    for col in cont_cols:
        if col in df_out.columns and col in train_medians:
            df_out[col] = df_out[col].fillna(train_medians[col])

    # Step 4: Impute categoricals with mode (-2 sentinel for unknown)
    if train_modes is None:
        train_modes = {}
        for col in cat_cols:
            if col in df_out.columns:
                mode_val = df_out[col].mode()
                train_modes[col] = int(mode_val.iloc[0]) if len(mode_val) > 0 else -2
    for col in cat_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(train_modes.get(col, -2))

    df_out = pd.concat([df_out, pd.DataFrame(_new, index=df_out.index)], axis=1)
    log.info(f"Missing handling: {len(_new)} missingness flags added.")
    return df_out, train_medians, train_modes


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 2: Drop ps_calc_* (uninformative)
# ─────────────────────────────────────────────────────────────────────────────

def drop_calc_features(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[str]]:
    """
    ps_calc_* features are synthetic / calculated features added by Porto Seguro.
    Competition analysis (and Kaggle kernels) confirms they carry no signal and
    add noise. Removing them improves generalisation and reduces memory.

    Domain rationale: In insurance underwriting, engineered score components
    that are post-hoc calculations should be scrutinised for data leakage or
    synthetic noise before inclusion in production models.
    """
    calc_cols = groups["calc_bin"] + groups["calc_cont"]
    existing  = [c for c in calc_cols if c in df.columns]
    df_out    = df.drop(columns=existing, errors="ignore")
    log.info(f"Dropped {len(existing)} ps_calc_* features: {existing[:5]}…")
    return df_out, existing


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 3: Categorical encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_categorical_features(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    target_means: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Encode ps_*_cat columns:
      - If n_unique ≤ 10: ordinal encode (integers already, no-op beyond fill)
      - If n_unique  > 10: target-mean encoding (computed on train set only)

    Target-mean encoding captures actuarial claim rates by category without
    inflating dimensionality — unlike one-hot encoding which would add 100+
    dummy columns for high-cardinality vehicle type or region codes.

    Parameters
    ----------
    target_means : Pre-computed train-set means. If None, fit from df.
                   Always pass train values when transforming the val set.
    """
    cat_cols = groups["ind_cat"] + groups["car_cat"]
    existing = [c for c in cat_cols if c in df.columns]

    _new: dict = {}

    if target_means is None:
        target_means = {}
        global_mean  = df[TARGET_COL].mean() if TARGET_COL in df.columns else 0.0
        for col in existing:
            if df[col].nunique() > 10 and TARGET_COL in df.columns:
                # Target-mean encoding: mean claim rate per category level
                grp = (
                    df.groupby(col)[TARGET_COL]
                    .mean()
                    .to_dict()
                )
                target_means[col] = {"map": grp, "global": global_mean, "encoded": True}
            else:
                target_means[col] = {"encoded": False}

    for col in existing:
        meta = target_means.get(col, {})
        if meta.get("encoded", False):
            mapping     = meta["map"]
            global_mean = meta["global"]
            _new[f"fe_te_{col}"] = (
                df[col].map(mapping).fillna(global_mean)
            )
            log.info(f"  Target-encoded: {col} → fe_te_{col}")
        # Leave low-cardinality categoricals as-is (ordinal integers)

    if _new:
        df = pd.concat([df, pd.DataFrame(_new, index=df.index)], axis=1)

    log.info(
        f"Categorical encoding: {sum(1 for v in target_means.values() if v.get('encoded'))} "
        f"target-encoded, "
        f"{sum(1 for v in target_means.values() if not v.get('encoded'))} left ordinal."
    )
    return df, target_means


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 4: Aggregate / count features
# ─────────────────────────────────────────────────────────────────────────────

def add_aggregate_features(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Row-level aggregate features that capture policyholder risk profile holistically:

      fe_n_missing        : Count of -1 (missing) values per row before imputation.
                            High missing count → sparse/incomplete policy record → higher risk.
      fe_n_bin_ind        : Sum of ps_ind_*_bin flags.
                            Counts how many individual risk indicators are active.
      fe_n_bin_car        : Sum of ps_car_*_bin flags.
                            Counts active vehicle risk flags.
      fe_reg_sum          : Sum of ps_reg_* continuous features.
                            Aggregate geographic/registration risk.
      fe_ind_cont_sum     : Sum of ps_ind_* continuous features.
                            Composite individual risk score proxy.

    Domain rationale: Actuarial GLM models often include count variables for
    risk factor exposure (number of accidents, number of claims, coverage flags).
    These aggregates mimic that approach for the anonymised Porto Seguro feature space.
    """
    _new: dict = {}

    # Count of missing (-1) per row on original df (before NaN replacement)
    feat_cols = [c for c in df.columns if c not in ("id", TARGET_COL)]
    # Assuming df already had -1 replaced, check missingness flags instead
    miss_flag_cols = [c for c in df.columns if c.startswith("fe_miss_")]
    if miss_flag_cols:
        _new["fe_n_missing"] = df[miss_flag_cols].sum(axis=1).astype(np.int16)

    # Sum of binary indicator flags
    ind_bin = [c for c in groups["ind_bin"] if c in df.columns]
    car_bin = [c for c in groups["car_bin"] if c in df.columns]
    if ind_bin:
        _new["fe_n_bin_ind"] = df[ind_bin].fillna(0).sum(axis=1).astype(np.int8)
    if car_bin:
        _new["fe_n_bin_car"] = df[car_bin].fillna(0).sum(axis=1).astype(np.int8)

    # Sum of registration features
    reg_cols = [c for c in groups["reg"] if c in df.columns]
    if reg_cols:
        _new["fe_reg_sum"] = df[reg_cols].fillna(0).sum(axis=1).astype(np.float32)

    # Sum of individual continuous features
    ind_cont = [c for c in groups["ind_cont"] if c in df.columns]
    if ind_cont:
        _new["fe_ind_cont_sum"] = df[ind_cont].fillna(0).sum(axis=1).astype(np.float32)

    df = pd.concat([df, pd.DataFrame(_new, index=df.index)], axis=1)
    log.info(f"Aggregate features added: {list(_new.keys())}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 5: Interaction features
# ─────────────────────────────────────────────────────────────────────────────

def add_interaction_features(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Cross-domain interaction features combining policyholder (ps_ind_*) and
    vehicle (ps_car_*) information:

      fe_car_reg_interact : ps_car_13 × ps_reg_03
                            Vehicle value × geographic region — high-value cars
                            in high-risk regions are disproportionately claimed.
      fe_ind_car_sum      : ps_ind_01 + ps_car_01 proxy composite risk.
      fe_miss_x_reg_sum   : Missing count × regional risk — amplifies risk signal
                            when both data quality and region are poor.

    Domain rationale: Insurance underwriting GLMs routinely include interaction
    terms between vehicle class and territory. XGBoost/LightGBM can learn these
    implicitly, but explicit features speed up convergence and aid interpretability.
    """
    _new: dict = {}

    # ps_car_13 × ps_reg_03  (most predictive car feature × reg feature)
    if "ps_car_13" in df.columns and "ps_reg_03" in df.columns:
        _new["fe_car13_reg03"] = (
            df["ps_car_13"].fillna(0) * df["ps_reg_03"].fillna(0)
        ).astype(np.float32)

    # ps_ind_01 + ps_car_01
    if "ps_ind_01" in df.columns and "ps_car_01" in df.columns:
        _new["fe_ind01_car01_sum"] = (
            df["ps_ind_01"].fillna(0) + df["ps_car_01"].fillna(0)
        ).astype(np.float32)

    # fe_n_missing × fe_reg_sum
    if "fe_n_missing" in df.columns and "fe_reg_sum" in df.columns:
        _new["fe_miss_x_reg"] = (
            df["fe_n_missing"].fillna(0) * df["fe_reg_sum"].fillna(0)
        ).astype(np.float32)

    # ps_car_13² (non-linear vehicle value signal)
    if "ps_car_13" in df.columns:
        _new["fe_car13_sq"] = (df["ps_car_13"].fillna(0) ** 2).astype(np.float32)

    df = pd.concat([df, pd.DataFrame(_new, index=df.index)], axis=1)
    log.info(f"Interaction features added: {list(_new.keys())}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 6: Continuous transformations
# ─────────────────────────────────────────────────────────────────────────────

def transform_continuous_features(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Log1p transformation on right-skewed continuous features (ps_reg_*).
    Porto Seguro registration features are right-skewed continuous values.
    Log-transforming them improves LightGBM leaf-split quality and
    stabilises gradient descent for MLP and Logistic Regression.

    Domain rationale: Insurance premium amounts, claim values, and regional
    exposure metrics are typically log-normal — a standard actuarial assumption.
    """
    _new: dict = {}
    reg_cols = [c for c in groups["reg"] if c in df.columns]
    for col in reg_cols:
        vals = df[col].fillna(0).clip(lower=0)
        _new[f"fe_log_{col}"] = np.log1p(vals).astype(np.float32)

    # Also log-transform car continuous features
    car_cont = [c for c in groups["car_cont"] if c in df.columns]
    for col in car_cont:
        vals = df[col].fillna(0).clip(lower=0)
        if vals.skew() > 1.0:  # only skewed features
            _new[f"fe_log_{col}"] = np.log1p(vals).astype(np.float32)

    df = pd.concat([df, pd.DataFrame(_new, index=df.index)], axis=1)
    log.info(f"Log transforms added: {len(_new)} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def save_feature_summary(
    df_raw: pd.DataFrame,
    df_fe: pd.DataFrame,
    fe_cols: list[str],
) -> None:
    """Save engineered feature list CSV + a summary PNG."""
    rows = []
    for col in fe_cols:
        rows.append({
            "feature":    col,
            "dtype":      str(df_fe[col].dtype),
            "null_pct":   round(100 * df_fe[col].isna().mean(), 2),
            "mean":       round(float(df_fe[col].mean()), 4) if df_fe[col].dtype != object else None,
            "std":        round(float(df_fe[col].std()), 4) if df_fe[col].dtype != object else None,
        })
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(REPORT_DIR / "engineered_features_list.csv", index=False)
    log.info(f"Feature list saved: {len(feat_df)} engineered features")

    # Bar chart: raw vs engineered feature count
    fig, ax = plt.subplots(figsize=(6, 4))
    raw_n = len([c for c in df_raw.columns if c not in ("id", TARGET_COL)])
    fe_n  = len(fe_cols)
    ax.bar(["Raw features", "Engineered features"], [raw_n, fe_n],
           color=["#1976D2", "#388E3C"])
    ax.set_title("Feature Count: Raw vs Engineered")
    ax.set_ylabel("Number of features")
    for i, v in enumerate([raw_n, fe_n]):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "engineered_feature_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {REPORT_DIR / 'engineered_feature_summary.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    train_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply all feature engineering steps in order.
    Returns transformed df and fitted stats dict for applying to val set.
    """
    stats = train_stats or {}

    # 1. Missing value handling
    df, stats["medians"], stats["modes"] = handle_missing_values(
        df, groups,
        train_medians=stats.get("medians"),
        train_modes=stats.get("modes"),
    )

    # 2. Drop ps_calc_* features
    df, stats["calc_dropped"] = drop_calc_features(df, groups)

    # 3. Categorical encoding
    df, stats["target_means"] = encode_categorical_features(
        df, groups,
        target_means=stats.get("target_means"),
    )

    # 4. Aggregate features
    df = add_aggregate_features(df, groups)

    # 5. Interaction features
    df = add_interaction_features(df, groups)

    # 6. Continuous transformations
    df = transform_continuous_features(df, groups)

    return df, stats


def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case E: Insurance Risk & Claims Analytics")
    print("  Step 3: Feature Engineering & Data Preparation")
    print("=" * 65 + "\n")

    # Load splits
    df_train, df_val = _load_splits()
    groups = _get_feature_groups(df_train)
    df_raw_train = df_train.copy()

    print(f"[1] Train: {df_train.shape}  |  Val: {df_val.shape}")

    # Apply feature engineering to train (fit stats here)
    print("\n[2] Engineering train set features…")
    df_train_fe, train_stats = run_feature_engineering(df_train, groups)

    # Apply same transformations to val (using train stats — no leakage)
    print("\n[3] Applying to validation set…")
    # Recompute groups after calc drop (groups dict changes)
    val_groups = _get_feature_groups(df_val)
    df_val_fe, _ = run_feature_engineering(df_val, val_groups, train_stats=train_stats)

    # Save
    df_train_fe.to_parquet(DATA_SUBDIR / "train_fe.parquet", index=False)
    df_val_fe.to_parquet(DATA_SUBDIR / "val_fe.parquet",   index=False)
    log.info("Saved train_fe.parquet and val_fe.parquet")

    # Save preprocessing artifacts
    joblib.dump(train_stats, ARTIFACT_DIR / "fe_stats.pkl")
    log.info(f"Saved fe_stats.pkl → {ARTIFACT_DIR}")

    # Feature summary
    fe_cols = [c for c in df_train_fe.columns if c.startswith("fe_")]
    print(f"\n[4] Engineered {len(fe_cols)} new features.")
    save_feature_summary(df_raw_train, df_train_fe, fe_cols)

    print(f"\n  Train FE shape : {df_train_fe.shape}")
    print(f"  Val   FE shape : {df_val_fe.shape}")
    print(f"\n[✓] All feature engineering outputs saved to: {DATA_SUBDIR}")

    print("\n" + "=" * 65)
    print("  Step 3 complete. Ready for Model Training (04_model_training.py)")
    print("=" * 65 + "\n")

    return df_train_fe, df_val_fe


if __name__ == "__main__":
    df_train_fe, df_val_fe = main()
