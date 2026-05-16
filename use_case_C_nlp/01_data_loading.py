"""
use_case_C_nlp/01_data_loading.py
===================================
DSF504 — Use Case C (NLP): Market Intelligence — Financial Sentiment
Step 1: Data Loading, Profiling, and Train/Validation Split

Dataset
-------
Financial PhraseBank (Malo et al., 2014)
  ~4,840 sentences annotated by 16 financial domain experts
  3-class sentiment: negative (0) · neutral (1) · positive (2)
  Configuration used: "sentences_allagree" — only sentences where
  ALL annotators agreed (2,264 sentences, highest quality).

Download
--------
Loaded automatically via HuggingFace `datasets` library.
No Kaggle token required.
    pip install datasets

Academic references
-------------------
- Malo P. et al. (2014). Good Debt or Bad Debt: Detecting Semantic
  Orientations in Economic Texts. JASIST.
- Huang A. et al. (2023). FinBERT: A Large Language Model for Extracting
  Information from Financial Text. Contemporary Accounting Research.
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

DATA_SUBDIR = DATA_DIR    / "financial_phrasebank"
REPORT_DIR  = REPORTS_DIR / "use_case_C_nlp"
DATA_SUBDIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET    = "label"
TEXT_COL  = "sentence"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_phrasebank(config: str = "sentences_allagree") -> pd.DataFrame:
    """
    Load Financial PhraseBank from HuggingFace datasets.

    Parameters
    ----------
    config : annotation agreement level
        "sentences_50agree"  — ≥50% annotators agree  (4,846 sentences)
        "sentences_66agree"  — ≥66% agree             (4,217 sentences)
        "sentences_75agree"  — ≥75% agree             (3,453 sentences)
        "sentences_allagree" — 100% agree              (2,264 sentences)
    """
    parquet_path = DATA_SUBDIR / f"phrasebank_{config}.parquet"

    if parquet_path.exists():
        log.info(f"Loading cached parquet: {parquet_path.name}")
        return pd.read_parquet(parquet_path)

    log.info(f"Downloading Financial PhraseBank ({config}) via HuggingFace…")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace `datasets` not installed.\n"
            "Run: pip install datasets --break-system-packages"
        )

    # Use the Parquet-native mirror (HuggingFace deprecated loading scripts in 2024)
    ds = load_dataset("takala/financial_phrasebank", config)

    # The dataset only has a 'train' split — we'll do our own stratified split
    df = ds["train"].to_pandas()
    df = df.rename(columns={"sentence": TEXT_COL, "label": TARGET})
    df[TARGET] = df[TARGET].astype(np.int8)
    df["label_name"] = df[TARGET].map(LABEL_MAP)

    # Basic text cleaning
    df[TEXT_COL] = df[TEXT_COL].str.strip()
    df = df.dropna(subset=[TEXT_COL, TARGET])
    df = df[df[TEXT_COL].str.len() > 5].reset_index(drop=True)

    df.to_parquet(parquet_path, index=False)
    log.info(f"Cached as: {parquet_path.name}  ({len(df):,} rows)")
    return df


# ── 2. Profile ─────────────────────────────────────────────────────────────────

def profile_data(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  DATASET PROFILE — Financial PhraseBank")
    print("=" * 60)
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {df.shape[1]}")
    print(f"  Memory       : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print()

    # Class distribution
    dist = df[TARGET].value_counts().sort_index()
    print("  Class distribution:")
    for label_id, count in dist.items():
        name = LABEL_MAP[label_id]
        pct  = 100 * count / len(df)
        bar  = "█" * int(pct / 2)
        print(f"    {label_id} ({name:8s}): {count:>5,}  ({pct:5.1f}%)  {bar}")

    # Text length stats
    df["_len"] = df[TEXT_COL].str.split().str.len()
    print(f"\n  Text length (words):")
    print(f"    Min    : {df['_len'].min()}")
    print(f"    Median : {df['_len'].median():.0f}")
    print(f"    Mean   : {df['_len'].mean():.1f}")
    print(f"    Max    : {df['_len'].max()}")
    df.drop(columns=["_len"], inplace=True)
    print("=" * 60 + "\n")


# ── 3. Visualisations ──────────────────────────────────────────────────────────

def plot_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Financial PhraseBank — Dataset Overview",
                 fontsize=14, fontweight="bold")

    colours = {"negative": "#E53935", "neutral": "#FB8C00", "positive": "#43A047"}

    # Class distribution bar
    ax = axes[0]
    dist = df["label_name"].value_counts()
    bars = ax.bar(dist.index, dist.values,
                  color=[colours[l] for l in dist.index], alpha=0.85)
    for bar, count in zip(bars, dist.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10, f"{count:,}",
                ha="center", fontsize=10)
    ax.set_title("Sentiment Class Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Sentiment")

    # Sentence length distribution by class
    ax = axes[1]
    df["word_count"] = df[TEXT_COL].str.split().str.len()
    for label, colour in colours.items():
        subset = df[df["label_name"] == label]["word_count"]
        ax.hist(subset, bins=30, alpha=0.6, color=colour, label=label)
    ax.set_title("Sentence Length by Sentiment")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    df.drop(columns=["word_count"], inplace=True)

    # Pie chart
    ax = axes[2]
    dist2 = df["label_name"].value_counts()
    ax.pie(dist2.values,
           labels=dist2.index,
           colors=[colours[l] for l in dist2.index],
           autopct="%1.1f%%",
           startangle=90)
    ax.set_title("Sentiment Distribution (%)")

    plt.tight_layout()
    path = REPORT_DIR / "overview.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


def plot_sentence_lengths(df: pd.DataFrame) -> None:
    """Box plot of sentence length per class."""
    df = df.copy()
    df["word_count"] = df[TEXT_COL].str.split().str.len()
    df["char_count"] = df[TEXT_COL].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sentence Length Analysis", fontsize=12, fontweight="bold")

    colours = {"negative": "#E53935", "neutral": "#FB8C00", "positive": "#43A047"}
    order   = ["negative", "neutral", "positive"]

    for ax, col, title in zip(
        axes,
        ["word_count", "char_count"],
        ["Word Count by Sentiment", "Character Count by Sentiment"],
    ):
        data_by_class = [df[df["label_name"] == cls][col].values for cls in order]
        bp = ax.boxplot(data_by_class, labels=order, patch_artist=True, notch=False)
        for patch, cls in zip(bp["boxes"], order):
            patch.set_facecolor(colours[cls])
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_ylabel(col.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = REPORT_DIR / "sentence_lengths.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ── 4. Split ───────────────────────────────────────────────────────────────────

def split_and_save(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split preserving class proportions."""
    df_train, df_val = smart_split(
        df, target_col=TARGET, task_type="multiclass_classification",
        val_size=0.20, random_state=RANDOM_STATE,
    )

    print(f"\n  Train: {len(df_train):,} rows  |  Val: {len(df_val):,} rows")
    print("  Class distribution (train):")
    for label_id, count in df_train[TARGET].value_counts().sort_index().items():
        print(f"    {LABEL_MAP[label_id]:8s}: {count:>4,}  ({100*count/len(df_train):.1f}%)")

    df_train.to_parquet(DATA_SUBDIR / "train_raw.parquet", index=False)
    df_val.to_parquet(DATA_SUBDIR / "val_raw.parquet",     index=False)
    log.info(f"Splits saved to {DATA_SUBDIR}")
    return df_train, df_val


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case C (NLP): Financial Sentiment")
    print("  Step 1: Data Loading")
    print("=" * 65 + "\n")

    print("[1] Loading Financial PhraseBank…")
    df = load_phrasebank(config="sentences_allagree")

    print("[2] Profiling dataset…")
    profile_data(df)

    print("[3] Generating overview plots…")
    plot_overview(df)
    plot_sentence_lengths(df)

    print("[4] Splitting train / validation…")
    split_and_save(df)

    # Save column summary CSV
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype":  [str(df[c].dtype) for c in df.columns],
        "n_unique": [df[c].nunique() for c in df.columns],
        "n_missing": [df[c].isna().sum() for c in df.columns],
    })
    summary.to_csv(REPORT_DIR / "column_summary.csv", index=False)

    print("\n" + "=" * 65)
    print("  Step 1 complete. Ready for EDA (02_eda_analysis.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
