"""
use_case_B_credit/03_feature_engineering.py
=============================================
DSF504 — Use Case B: Credit Risk Modelling
Step 3: Feature Engineering

Engineering strategy
--------------------
1. Error-code cleaning  — replace 96/98 in DPD columns with NaN; cap > 10
2. Winsorisation        — RevolvingUtil capped at [0,1]; DebtRatio at [0, 5000]
3. Log transforms       — MonthlyIncome, DebtRatio (right-skewed)
4. Delinquency rollup   — total past-due events, weighted severity score
5. Debt-to-income       — interaction of DebtRatio × MonthlyIncome
6. Age engineering      — age², age bucket, prime-earning-years flag
7. Missing indicators   — binary flags for MonthlyIncome and NumDependents
8. Imputation           — median for income; 0 for dependents
9. Scaling artifacts    — final dtypes in float32 / int8

Academic grounding
------------------
- Siddiqi (2012): WOE binning motivates monotonic feature construction
- Baesens et al. (2016): delinquency history is the #1 scorecard predictor
- Lessmann et al. (2015): feature interaction terms improve tree & LR models
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE
from utils.data_loader import smart_split

# ── UTF-8 encoding guard (fixes garbled output on Windows) ─────────────────
from utils.encoding_guard import ensure_utf8
ensure_utf8()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR    / "gmsc_credit"
REPORT_DIR  = REPORTS_DIR / "use_case_B"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET      = "SeriousDlqin2yrs"
DELINQ      = ["DPD_30_59", "DPD_60_89", "DPD_90plus"]
ERROR_CODES = {96, 98}


# ── Feature group 1: Clean error codes ────────────────────────────────────────

def clean_error_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 96/98 error codes in delinquency columns with NaN, then cap at 10."""
    df = df.copy()
    for col in DELINQ:
        if col not in df.columns:
            continue
        df[col] = df[col].where(~df[col].isin(ERROR_CODES), other=np.nan)
        df[col] = df[col].clip(upper=10)
        # Fill NaN with 0 (no delinquency info → assume performing)
        df[col] = df[col].fillna(0).astype(np.float32)
    log.info("✓ Error codes cleaned (96/98 → NaN → 0)")
    return df


# ── Feature group 2: Winsorise extreme values ─────────────────────────────────

def winsorise(df: pd.DataFrame) -> pd.DataFrame:
    """Cap extreme values to domain-valid or 99th-percentile ranges."""
    df = df.copy()
    if "RevolvingUtil" in df.columns:
        df["RevolvingUtil"] = df["RevolvingUtil"].clip(0, 1).astype(np.float32)
    if "DebtRatio" in df.columns:
        p99 = df["DebtRatio"].quantile(0.999)
        df["DebtRatio"] = df["DebtRatio"].clip(0, p99).astype(np.float32)
    if "MonthlyIncome" in df.columns:
        p99 = df["MonthlyIncome"].quantile(0.999)
        df["MonthlyIncome"] = df["MonthlyIncome"].clip(0, p99).astype(np.float32)
    log.info("✓ Winsorisation applied")
    return df


# ── Feature group 3: Missing indicators + imputation ─────────────────────────

def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Flag missingness then impute."""
    df = df.copy()
    indicators = {}

    if "MonthlyIncome" in df.columns:
        indicators["fe_miss_income"] = df["MonthlyIncome"].isna().astype(np.int8)
        median_income = df["MonthlyIncome"].median()
        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(median_income).astype(np.float32)

    if "NumDependents" in df.columns:
        indicators["fe_miss_dependents"] = df["NumDependents"].isna().astype(np.int8)
        df["NumDependents"] = df["NumDependents"].fillna(0).astype(np.float32)

    if indicators:
        df = pd.concat([df, pd.DataFrame(indicators, index=df.index)], axis=1)

    log.info(f"✓ Missing indicators added ({len(indicators)} flags)")
    return df


# ── Feature group 4: Log transforms ──────────────────────────────────────────

def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Log(1+x) for right-skewed financial variables."""
    df = df.copy()
    new_cols = {}
    for col in ["MonthlyIncome", "DebtRatio"]:
        if col in df.columns:
            new_cols[f"fe_log_{col}"] = np.log1p(df[col]).astype(np.float32)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    log.info(f"✓ Log transforms added ({len(new_cols)} columns)")
    return df


# ── Feature group 5: Delinquency features ────────────────────────────────────

def add_delinquency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate delinquency signals:
    - fe_total_dpd        : sum of all past-due counts
    - fe_dpd_severity     : weighted sum (30d×1 + 60d×2 + 90d×3)
    - fe_any_delinquency  : binary flag for any past-due event
    - fe_chronic_default  : 90+ days late ≥ 2 times
    """
    df   = df.copy()
    new  = {}
    cols = [c for c in DELINQ if c in df.columns]

    if cols:
        vals = df[cols].fillna(0)
        new["fe_total_dpd"]       = vals.sum(axis=1).astype(np.float32)
        new["fe_dpd_severity"]    = (
            vals.get("DPD_30_59", 0) * 1 +
            vals.get("DPD_60_89", 0) * 2 +
            vals.get("DPD_90plus",0) * 3
        ).astype(np.float32)
        new["fe_any_delinquency"] = (new["fe_total_dpd"] > 0).astype(np.int8)

    if "DPD_90plus" in df.columns:
        new["fe_chronic_default"] = (df["DPD_90plus"] >= 2).astype(np.int8)

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    log.info(f"✓ Delinquency features added ({len(new)} columns)")
    return df


# ── Feature group 6: Age features ────────────────────────────────────────────

def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Non-linear and categorical age encodings."""
    if "age" not in df.columns:
        return df
    df   = df.copy()
    new  = {}
    age  = df["age"].clip(18, 100).astype(np.float32)

    new["fe_age_sq"]       = (age ** 2).astype(np.float32)
    new["fe_prime_earner"] = ((age >= 35) & (age <= 55)).astype(np.int8)
    new["fe_senior"]       = (age >= 65).astype(np.int8)
    new["fe_young"]        = (age < 30).astype(np.int8)

    # Age bucket ordinal (0–5)
    bins   = [0, 30, 40, 50, 60, 70, 200]
    labels = [0, 1, 2, 3, 4, 5]
    new["fe_age_bucket"] = pd.cut(age, bins=bins, labels=labels,
                                  right=False).astype(np.int8)

    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    log.info(f"✓ Age features added ({len(new)} columns)")
    return df


# ── Feature group 7: Credit utilisation & debt features ──────────────────────

def add_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interactions and derived ratios:
    - fe_util_sq          : squared utilisation (penalises near-max usage)
    - fe_dti_proxy        : DebtRatio × MonthlyIncome (dollar debt estimate)
    - fe_income_per_dep   : MonthlyIncome / (NumDependents + 1)
    - fe_loans_per_estate : NumOpenLoans / (NumRealEstate + 1)
    - fe_util_x_dpd       : utilisation × total delinquency (risk interaction)
    - fe_high_util        : flag: utilisation > 0.75
    - fe_zero_util        : flag: utilisation == 0 (dormant account)
    """
    df  = df.copy()
    new = {}

    if "RevolvingUtil" in df.columns:
        new["fe_util_sq"]    = (df["RevolvingUtil"] ** 2).astype(np.float32)
        new["fe_high_util"]  = (df["RevolvingUtil"] > 0.75).astype(np.int8)
        new["fe_zero_util"]  = (df["RevolvingUtil"] == 0).astype(np.int8)

    if "DebtRatio" in df.columns and "MonthlyIncome" in df.columns:
        new["fe_dti_proxy"] = (
            (df["DebtRatio"] * df["MonthlyIncome"])
            .clip(upper=df["DebtRatio"].quantile(0.999) *
                       df["MonthlyIncome"].quantile(0.999))
            .astype(np.float32)
        )

    if "MonthlyIncome" in df.columns and "NumDependents" in df.columns:
        new["fe_income_per_dep"] = (
            df["MonthlyIncome"] / (df["NumDependents"].fillna(0) + 1)
        ).astype(np.float32)

    if "NumOpenLoans" in df.columns and "NumRealEstate" in df.columns:
        new["fe_loans_per_estate"] = (
            df["NumOpenLoans"] / (df["NumRealEstate"] + 1)
        ).astype(np.float32)

    if "RevolvingUtil" in df.columns and "fe_total_dpd" in df.columns:
        new["fe_util_x_dpd"] = (
            df["RevolvingUtil"] * df["fe_total_dpd"]
        ).astype(np.float32)

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    log.info(f"✓ Credit interaction features added ({len(new)} columns)")
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_error_codes(df)
    df = winsorise(df)
    df = add_missing_indicators(df)
    df = add_log_transforms(df)
    df = add_delinquency_features(df)
    df = add_age_features(df)
    df = add_credit_features(df)
    return df


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_engineered_features(df_orig: pd.DataFrame, df_fe: pd.DataFrame) -> None:
    """Compare raw vs engineered key features."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Feature Engineering — Before vs After", fontsize=13, fontweight="bold")

    pairs = [
        ("MonthlyIncome",   "fe_log_MonthlyIncome",  "Monthly Income",   "log(1+Income)"),
        ("DebtRatio",       "fe_log_DebtRatio",       "Debt Ratio",       "log(1+DebtRatio)"),
        ("RevolvingUtil",   "fe_util_sq",             "Revolving Util",   "Util²"),
    ]

    for i, (raw, eng, label_raw, label_eng) in enumerate(pairs):
        ax_raw = axes[0, i]
        ax_eng = axes[1, i]

        if raw in df_orig.columns:
            vals = df_orig[raw].dropna().clip(
                df_orig[raw].quantile(0.01), df_orig[raw].quantile(0.99))
            ax_raw.hist(vals, bins=50, color="#1E88E5", alpha=0.8)
            ax_raw.set_title(f"Raw: {label_raw}", fontsize=9)

        if eng in df_fe.columns:
            vals2 = df_fe[eng].dropna().clip(
                df_fe[eng].quantile(0.01), df_fe[eng].quantile(0.99))
            ax_eng.hist(vals2, bins=50, color="#43A047", alpha=0.8)
            ax_eng.set_title(f"Engineered: {label_eng}", fontsize=9)

    plt.tight_layout()
    path = REPORT_DIR / "engineered_features.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


def print_feature_summary(df_fe: pd.DataFrame) -> None:
    fe_cols = [c for c in df_fe.columns if c.startswith("fe_")]
    print(f"\n  Engineered features : {len(fe_cols)}")
    for col in fe_cols:
        dtype = df_fe[col].dtype
        n_null = df_fe[col].isna().sum()
        print(f"    {col:35s} dtype={dtype}  nulls={n_null}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case B: Feature Engineering")
    print("=" * 65 + "\n")

    # Load raw parquet splits from Step 1
    train_path = DATA_SUBDIR / "train_raw.parquet"
    val_path   = DATA_SUBDIR / "val_raw.parquet"

    if not train_path.exists():
        # Fallback: use full dataset
        full_path = DATA_SUBDIR / "cs-training.parquet"
        if not full_path.exists():
            raise FileNotFoundError("Run 01_data_loading.py first.")
        log.info("Loading full dataset (no raw splits found)…")
        df_full  = pd.read_parquet(full_path)
        df_train, df_val = smart_split(
            df_full, target_col=TARGET,
            task_type="binary_classification",
            val_size=0.20, random_state=RANDOM_STATE,
        )
    else:
        log.info("Loading raw train/val splits…")
        df_train = pd.read_parquet(train_path)
        df_val   = pd.read_parquet(val_path)

    print(f"  Train: {len(df_train):,} rows | Val: {len(df_val):,} rows")

    print("\n[1] Running feature engineering pipeline on train…")
    df_train_raw = df_train.copy()
    df_train_fe  = run_feature_pipeline(df_train)

    print("[2] Applying same pipeline to validation…")
    df_val_fe = run_feature_pipeline(df_val)

    print("[3] Feature summary…")
    print_feature_summary(df_train_fe)

    print("[4] Generating visualisations…")
    plot_engineered_features(df_train_raw, df_train_fe)

    print("[5] Saving engineered datasets…")
    out_train = DATA_SUBDIR / "train_fe.parquet"
    out_val   = DATA_SUBDIR / "val_fe.parquet"
    df_train_fe.to_parquet(out_train, index=False)
    df_val_fe.to_parquet(out_val,   index=False)
    log.info(f"Saved: {out_train}  ({len(df_train_fe.columns)} columns)")
    log.info(f"Saved: {out_val}")

    print("\n" + "=" * 65)
    print("  Step 3 complete. Ready for model training (04_model_training.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
