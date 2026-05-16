"""
utils/data_loader.py
====================
Generic data-loading utilities shared across all use cases.

Responsibilities
----------------
- Kaggle dataset download (requires ~/.kaggle/kaggle.json API token)
- Cached CSV / Parquet loading with memory optimisation
- DataFrame profiling helpers (dtypes, missingness, cardinality)
- Train / validation / test split helpers

Usage
-----
    from utils.data_loader import KaggleLoader, DataProfiler, smart_split

    loader = KaggleLoader(use_case_key="A")
    df_train, df_test = loader.load()
"""

from __future__ import annotations

import os
import sys
import zipfile
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Resolve project root for config import regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR, RANDOM_STATE, DATASET_REGISTRY,
    IMBALANCE_THRESHOLD, STRATIFIED_CV,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory optimisation
# ---------------------------------------------------------------------------

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest type that fits their range.
    Typically reduces memory footprint by 50-70% on large financial datasets.

    Parameters
    ----------
    df      : Input DataFrame
    verbose : Print memory reduction summary

    Returns
    -------
    df : DataFrame with downcasted dtypes (in-place copy)
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == object:
            # Convert to category if cardinality is low enough to save memory
            n_unique = df[col].nunique()
            if n_unique / len(df) < 0.5:
                df[col] = df[col].astype("category")

        elif col_type != bool:
            c_min, c_max = df[col].min(), df[col].max()

            if str(col_type).startswith("int"):
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if (np.iinfo(dtype).min <= c_min and
                            c_max <= np.iinfo(dtype).max):
                        df[col] = df[col].astype(dtype)
                        break
            elif str(col_type).startswith("float"):
                # Skip float16 — pandas does not support it as a groupby key
                # or index, which causes NotImplementedError at runtime.
                for dtype in [np.float32, np.float64]:
                    if (np.finfo(dtype).min <= c_min and
                            c_max <= np.finfo(dtype).max):
                        df[col] = df[col].astype(dtype)
                        break

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        log.info(
            f"Memory reduced: {start_mem:.1f} MB → {end_mem:.1f} MB "
            f"({reduction:.1f}% reduction)"
        )
    return df


# ---------------------------------------------------------------------------
# Kaggle downloader
# ---------------------------------------------------------------------------

class KaggleLoader:
    """
    Downloads and loads a dataset registered in config.DATASET_REGISTRY.

    Parameters
    ----------
    use_case_key : Key in DATASET_REGISTRY, e.g. "A", "B", "G"
    force_download : Re-download even if files already exist locally
    optimize_memory : Apply reduce_mem_usage after loading

    Example
    -------
        loader = KaggleLoader("A")
        df_train, df_test = loader.load()
    """

    def __init__(
        self,
        use_case_key: str,
        force_download: bool = False,
        optimize_memory: bool = True,
    ):
        if use_case_key not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown use_case_key '{use_case_key}'. "
                f"Valid keys: {list(DATASET_REGISTRY.keys())}"
            )
        self.cfg             = DATASET_REGISTRY[use_case_key]
        self.force_download  = force_download
        self.optimize_memory = optimize_memory
        self.data_dir        = DATA_DIR / self.cfg["data_subdir"]
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _kaggle_available(self) -> bool:
        """Check if kaggle package and API credentials are available."""
        try:
            import kaggle  # noqa: F401
            cred_path = Path.home() / ".kaggle" / "kaggle.json"
            return cred_path.exists()
        except (ImportError, OSError, Exception):
            return False

    # ------------------------------------------------------------------
    def download(self) -> None:
        """Download dataset from Kaggle if not already present."""
        slug = self.cfg["kaggle_slug"]

        # Check if primary files already exist
        files = self.cfg.get("files", {})
        expected = list(files.values()) if files else []
        all_present = all(
            (self.data_dir / f).exists() for f in expected
        ) if expected else False

        if all_present and not self.force_download:
            log.info(f"Data already present in {self.data_dir} — skipping download.")
            return

        if not self._kaggle_available():
            log.warning(
                "Kaggle API not configured. "
                "Place your kaggle.json at ~/.kaggle/kaggle.json and re-run, "
                "or manually download the dataset from:\n"
                f"  https://www.kaggle.com/c/{slug}/data\n"
                f"  and extract into: {self.data_dir}"
            )
            return

        import kaggle
        log.info(f"Downloading Kaggle competition: {slug} → {self.data_dir}")

        # Try as a competition first, then as a dataset
        try:
            kaggle.api.competition_download_files(
                slug, path=str(self.data_dir), quiet=False
            )
        except Exception:
            # Fall back to dataset download (e.g. for non-competition datasets)
            owner, dataset = slug.split("/") if "/" in slug else (None, slug)
            if owner:
                kaggle.api.dataset_download_files(
                    f"{owner}/{dataset}", path=str(self.data_dir),
                    unzip=True, quiet=False
                )
                return

        # Unzip any downloaded archives
        for zf in self.data_dir.glob("*.zip"):
            log.info(f"Extracting {zf.name}…")
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(self.data_dir)
            zf.unlink()

        log.info("Download complete.")

    # ------------------------------------------------------------------
    def _load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a single CSV (or Parquet cache) from data_dir."""
        csv_path     = self.data_dir / filename
        parquet_path = self.data_dir / (Path(filename).stem + ".parquet")

        # Prefer cached Parquet for speed
        if parquet_path.exists() and not self.force_download:
            log.info(f"Loading cached Parquet: {parquet_path.name}")
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            log.info(f"Loading CSV: {csv_path.name}  (this may take a moment…)")
            df = pd.read_csv(csv_path)
            if self.optimize_memory:
                df = reduce_mem_usage(df)
            # Cache as Parquet for subsequent runs
            df.to_parquet(parquet_path, index=False)
            log.info(f"Cached as Parquet: {parquet_path.name}")
        else:
            log.warning(f"File not found: {csv_path}")
            return None

        return df

    # ------------------------------------------------------------------
    def load(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load and return (df_train, df_test).

        For Use Case A the transaction and identity files are loaded and
        left-joined on TransactionID before returning.
        """
        self.download()

        files      = self.cfg.get("files", {})
        merge_key  = self.cfg.get("merge_key")

        # --- Use Case A: two-file merge pattern ---
        if merge_key and set(files.keys()) >= {"train_transaction", "train_identity"}:
            df_trn = self._load_csv(files["train_transaction"])
            df_idn = self._load_csv(files["train_identity"])
            if df_trn is not None and df_idn is not None:
                log.info(f"Merging on '{merge_key}'…")
                df_train = df_trn.merge(df_idn, on=merge_key, how="left")
                log.info(
                    f"Merged train shape: {df_train.shape} "
                    f"(transactions: {len(df_trn):,}, identity matched: "
                    f"{df_idn[merge_key].isin(df_trn[merge_key]).sum():,})"
                )
            else:
                df_train = df_trn

            df_test = None
            if "test_transaction" in files and "test_identity" in files:
                df_tst  = self._load_csv(files["test_transaction"])
                df_tidn = self._load_csv(files["test_identity"])
                if df_tst is not None and df_tidn is not None:
                    df_test = df_tst.merge(df_tidn, on=merge_key, how="left")

        # --- Single-file pattern ---
        elif "train" in files:
            df_train = self._load_csv(files["train"])
            df_test  = self._load_csv(files.get("test")) if "test" in files else None

        else:
            raise RuntimeError(
                f"Cannot determine load strategy for files: {list(files.keys())}"
            )

        return df_train, df_test


# ---------------------------------------------------------------------------
# Data profiler
# ---------------------------------------------------------------------------

class DataProfiler:
    """
    Generates a structured profile of a DataFrame: shape, dtypes,
    missingness, cardinality, target distribution, and basic stats.

    Parameters
    ----------
    df         : The DataFrame to profile
    target_col : Name of the target/label column (optional)
    """

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.df         = df
        self.target_col = target_col

    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """
        Column-level summary table:
        dtype | n_missing | pct_missing | n_unique | cardinality_type | sample_values
        """
        rows = []
        for col in self.df.columns:
            n_miss     = int(self.df[col].isna().sum())
            pct_miss   = round(100 * n_miss / len(self.df), 2)
            n_unique   = int(self.df[col].nunique())
            dtype      = str(self.df[col].dtype)

            if n_unique == 1:
                card_type = "constant"
            elif n_unique == 2:
                card_type = "binary"
            elif n_unique <= 10:
                card_type = "low-cardinality"
            elif n_unique <= 50:
                card_type = "medium-cardinality"
            else:
                card_type = "high-cardinality"

            sample = self.df[col].dropna().head(3).tolist()
            rows.append({
                "column":           col,
                "dtype":            dtype,
                "n_missing":        n_miss,
                "pct_missing":      pct_miss,
                "n_unique":         n_unique,
                "cardinality_type": card_type,
                "sample_values":    str(sample),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def target_distribution(self) -> pd.Series:
        """Value counts and rates for the target column."""
        if self.target_col is None or self.target_col not in self.df.columns:
            log.warning("No target column specified or found.")
            return pd.Series(dtype=float)

        counts = self.df[self.target_col].value_counts()
        rates  = (counts / len(self.df) * 100).round(3)
        result = pd.concat(
            [counts.rename("count"), rates.rename("pct")], axis=1
        )
        log.info(
            f"\nTarget distribution ({self.target_col}):\n{result.to_string()}"
        )
        return result

    # ------------------------------------------------------------------
    def missing_heatmap_data(self) -> pd.DataFrame:
        """Return % missing by column, sorted descending, for visualisation."""
        missing = (
            self.df.isna().mean() * 100
        ).sort_values(ascending=False)
        return missing[missing > 0].rename("pct_missing").to_frame()

    # ------------------------------------------------------------------
    def print_report(self) -> None:
        """Print a human-readable summary to stdout."""
        df      = self.df
        n_rows, n_cols = df.shape
        mem_mb  = df.memory_usage(deep=True).sum() / 1024 ** 2

        print("=" * 60)
        print("DATASET PROFILE REPORT")
        print("=" * 60)
        print(f"  Rows       : {n_rows:,}")
        print(f"  Columns    : {n_cols:,}")
        print(f"  Memory     : {mem_mb:.1f} MB")
        print(f"  Dtypes     : {dict(df.dtypes.value_counts())}")
        print()

        miss_cols = df.isna().sum()
        miss_cols = miss_cols[miss_cols > 0]
        print(f"  Columns with missing values : {len(miss_cols)}")
        print(f"  Max missing in one column   : "
              f"{miss_cols.max() if len(miss_cols) else 0:,} "
              f"({100 * miss_cols.max() / n_rows:.1f}%)" if len(miss_cols) else "")
        print()

        if self.target_col and self.target_col in df.columns:
            self.target_distribution()

        print("=" * 60)


# ---------------------------------------------------------------------------
# Train / validation split helpers
# ---------------------------------------------------------------------------

def smart_split(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "binary_classification",
    val_size: float = 0.20,
    test_size: float = 0.00,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, ...]:
    """
    Stratified train/validation (and optionally test) split.

    Parameters
    ----------
    df           : Full DataFrame including target
    target_col   : Name of the target column
    task_type    : "binary_classification" | "multiclass_classification"
                   | "regression"
    val_size     : Fraction of data for validation
    test_size    : Fraction for hold-out test (0 = no separate test split)
    random_state : Seed for reproducibility

    Returns
    -------
    (df_train, df_val)              if test_size == 0
    (df_train, df_val, df_test)     if test_size > 0
    """
    y = df[target_col]
    stratify = y if task_type in ("binary_classification",
                                   "multiclass_classification") else None

    # First split: train vs (val + test)
    hold_size = val_size + test_size
    df_train, df_hold = train_test_split(
        df, test_size=hold_size, random_state=random_state,
        stratify=stratify,
    )

    if test_size == 0:
        return df_train, df_hold

    # Second split: val vs test (from the hold set)
    relative_test = test_size / hold_size
    stratify_hold = df_hold[target_col] if stratify is not None else None
    df_val, df_test = train_test_split(
        df_hold, test_size=relative_test, random_state=random_state,
        stratify=stratify_hold,
    )

    log.info(
        f"Split sizes — train: {len(df_train):,} | "
        f"val: {len(df_val):,} | test: {len(df_test):,}"
    )
    return df_train, df_val, df_test
