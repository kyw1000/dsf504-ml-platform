"""
use_case_C_nlp/01_data_loading.py
===================================
DSF504 -- Use Case C (NLP): Market Intelligence -- Financial Sentiment
Step 1: Data Loading, Profiling, and Train/Validation Split

Dataset: Financial PhraseBank (Malo et al., 2014)
  ~4,840 sentences annotated by 16 financial domain experts
  3-class sentiment: negative (0), neutral (1), positive (2)
  Configuration: "sentences_allagree" -- 100% annotator agreement (2,264 sentences)

Academic references
- Malo P. et al. (2014). Good Debt or Bad Debt: Detecting Semantic
  Orientations in Economic Texts. JASIST.
- Huang A. et al. (2023). FinBERT: A Large Language Model for Extracting
  Information from Financial Text. Contemporary Accounting Research.
"""

from __future__ import annotations

import sys
import logging
import urllib.request
import json as _json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE
from utils.data_loader import smart_split

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
_LABEL_STR = {"negative": 0, "neutral": 1, "positive": 2}
_FILE_MAP  = {
    "sentences_allagree": "Sentences_AllAgree.txt",
    "sentences_75agree":  "Sentences_75Agree.txt",
    "sentences_66agree":  "Sentences_66Agree.txt",
    "sentences_50agree":  "Sentences_50Agree.txt",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _fetch(url, timeout=20):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (DSF504/1.0)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def _decode(b):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("latin-1", errors="replace")

def _parse_at(text):
    """Parse 'sentence@label' format."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "@" not in line:
            continue
        i = line.rfind("@")
        s, l = line[:i].strip(), line[i+1:].strip().lower()
        if l in _LABEL_STR:
            rows.append({"sentence": s, "label": _LABEL_STR[l]})
    return rows

def _parse_csv(text):
    """Parse 'label,sentence' CSV format."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        l = parts[0].strip().strip('"').lower()
        s = parts[1].strip().strip('"')
        if l in _LABEL_STR and len(s) > 5:
            rows.append({"sentence": s, "label": _LABEL_STR[l]})
    return rows


# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_phrasebank(config="sentences_allagree"):
    """
    Load Financial PhraseBank. Tries 6 methods in order:
    1. Direct HuggingFace convert parquet
    2. hf_hub_download
    3. load_dataset (datasets < 3.0)
    4. datasets-server API (discover parquet URLs)
    5. Raw GitHub text mirrors
    6. Local file placed manually by user
    """
    parquet_path = DATA_SUBDIR / ("phrasebank_" + config + ".parquet")
    if parquet_path.exists():
        log.info("Loading cached parquet: %s", parquet_path.name)
        return pd.read_parquet(parquet_path)

    log.info("Downloading Financial PhraseBank (%s)...", config)
    df = None

    # Attempt 1: HuggingFace convert parquet
    _url1 = (
        "https://huggingface.co/datasets/takala/financial_phrasebank"
        "/resolve/refs%2Fconvert%2Fparquet/" + config + "/train/0000.parquet"
    )
    try:
        log.info("  Attempt 1: HuggingFace convert parquet...")
        df = pd.read_parquet(_url1, storage_options={"anon": True})
        log.info("  Attempt 1 succeeded.")
    except Exception as e:
        log.warning("  Attempt 1 failed: %s", e)

    # Attempt 2: hf_hub_download
    if df is None:
        try:
            from huggingface_hub import hf_hub_download
            _lf = hf_hub_download(
                repo_id="takala/financial_phrasebank",
                filename="data/" + config + "-train.parquet",
                repo_type="dataset",
            )
            df = pd.read_parquet(_lf)
            log.info("  Attempt 2 (hf_hub_download) succeeded.")
        except Exception as e:
            log.warning("  Attempt 2 failed: %s", e)

    # Attempt 3: load_dataset (datasets < 3.0)
    if df is None:
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "takala/financial_phrasebank", config,
                trust_remote_code=True,
            )
            df = ds["train"].to_pandas()
            log.info("  Attempt 3 (load_dataset) succeeded.")
        except Exception as e:
            log.warning("  Attempt 3 failed: %s", e)

    # Attempt 4: datasets-server API
    if df is None:
        try:
            log.info("  Attempt 4: datasets-server API...")
            meta = _json.loads(_fetch(
                "https://datasets-server.huggingface.co/parquet"
                "?dataset=takala/financial_phrasebank", timeout=15))
            urls = [f["url"] for f in meta.get("parquet_files", [])
                    if config in f.get("config", "").lower()]
            if not urls:
                urls = [f["url"] for f in meta.get("parquet_files", [])]
            for u in urls[:3]:
                try:
                    df = pd.read_parquet(u)
                    if len(df) > 0:
                        log.info("  Attempt 4 succeeded via: %s", u)
                        break
                    df = None
                except Exception:
                    pass
        except Exception as e:
            log.warning("  Attempt 4 failed: %s", e)

    # Attempt 5: raw text from GitHub mirrors
    if df is None:
        fname = _FILE_MAP.get(config, "Sentences_AllAgree.txt")
        mirrors = [
            "https://raw.githubusercontent.com/financial-phrasebank/financial-phrasebank/master/FinancialPhraseBank-v1.0/" + fname,
            "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/FinancialPhraseBank-v1.0/" + fname,
            "https://raw.githubusercontent.com/duynht/financial-statement-sentiment/master/data/" + fname,
            "https://raw.githubusercontent.com/PhraseBank/financial_phrasebank/master/" + fname,
            "https://raw.githubusercontent.com/ankurzing/sentiment-analysis-for-financial-news/master/data/all-data.csv",
        ]
        for url in mirrors:
            try:
                log.info("  Attempt 5: %s", url)
                text = _decode(_fetch(url, timeout=20))
                rows = _parse_at(text)
                if len(rows) < 50:
                    rows = _parse_csv(text)
                if len(rows) > 100:
                    df = pd.DataFrame(rows)
                    log.info("  Attempt 5 succeeded: %d rows", len(df))
                    break
            except Exception as e:
                log.warning("  Attempt 5 failed (%s): %s", url, e)

    # Attempt 6: local file
    if df is None:
        for lf in [DATA_SUBDIR / "Sentences_AllAgree.txt",
                   DATA_SUBDIR / "all-data.csv",
                   DATA_SUBDIR / "financial_phrasebank.csv"]:
            if lf.exists():
                log.info("  Attempt 6: local file %s", lf.name)
                text = _decode(lf.read_bytes())
                rows = _parse_at(text)
                if len(rows) < 50:
                    rows = _parse_csv(text)
                if rows:
                    df = pd.DataFrame(rows)
                    log.info("  Attempt 6 succeeded: %d rows", len(df))
                    break

    if df is None:
        raise RuntimeError(
            "\n\nAll download attempts for Financial PhraseBank failed.\n\n"
            "MANUAL FIX:\n"
            "  1. Go to: https://www.kaggle.com/datasets/ankurzing/"
            "sentiment-analysis-for-financial-news\n"
            "     Download all-data.csv (415 KB)\n"
            "  2. Copy the file to:\n"
            "     " + str(DATA_SUBDIR / "all-data.csv") + "\n"
            "  3. Re-run this script.\n"
        )

    # Normalise columns
    if "sentence" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "sentence"})
    df = df.rename(columns={"sentence": TEXT_COL, "label": TARGET}, errors="ignore")
    df[TARGET] = df[TARGET].astype(np.int8)
    df["label_name"] = df[TARGET].map(LABEL_MAP)

    # Clean
    df[TEXT_COL] = df[TEXT_COL].str.strip()
    df = df.dropna(subset=[TEXT_COL, TARGET])
    df = df[df[TEXT_COL].str.len() > 5].reset_index(drop=True)

    df.to_parquet(parquet_path, index=False)
    log.info("Cached: %s  (%d rows)", parquet_path.name, len(df))
    return df


# ── 2. Profile ─────────────────────────────────────────────────────────────────

def profile_data(df):
    print("\n" + "=" * 60)
    print("  DATASET PROFILE -- Financial PhraseBank")
    print("=" * 60)
    print("  Rows   : {:,}".format(len(df)))
    print("  Cols   : {}".format(df.shape[1]))
    print("  Memory : {:.2f} MB".format(df.memory_usage(deep=True).sum() / 1e6))
    print()
    dist = df[TARGET].value_counts().sort_index()
    print("  Class distribution:")
    for lid, cnt in dist.items():
        pct = 100 * cnt / len(df)
        bar = chr(9608) * int(pct / 2)
        print("    {} ({:8s}): {:>5,}  ({:5.1f}%)  {}".format(
            lid, LABEL_MAP[lid], cnt, pct, bar))
    df["_len"] = df[TEXT_COL].str.split().str.len()
    print("\n  Text length (words):")
    print("    Min    : {}".format(df["_len"].min()))
    print("    Median : {:.0f}".format(df["_len"].median()))
    print("    Mean   : {:.1f}".format(df["_len"].mean()))
    print("    Max    : {}".format(df["_len"].max()))
    df.drop(columns=["_len"], inplace=True)
    print("=" * 60 + "\n")


# ── 3. Visualisations ──────────────────────────────────────────────────────────

def plot_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Financial PhraseBank -- Dataset Overview",
                 fontsize=14, fontweight="bold")
    colours = {"negative": "#E53935", "neutral": "#FB8C00", "positive": "#43A047"}

    ax = axes[0]
    dist = df["label_name"].value_counts()
    bars = ax.bar(dist.index, dist.values,
                  color=[colours[l] for l in dist.index], alpha=0.85)
    for b, c in zip(bars, dist.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 10,
                "{:,}".format(c), ha="center", fontsize=10)
    ax.set_title("Sentiment Class Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Sentiment")

    ax = axes[1]
    df["word_count"] = df[TEXT_COL].str.split().str.len()
    for lbl, col in colours.items():
        ax.hist(df[df["label_name"] == lbl]["word_count"],
                bins=30, alpha=0.6, color=col, label=lbl)
    ax.set_title("Sentence Length by Sentiment")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    df.drop(columns=["word_count"], inplace=True)

    ax = axes[2]
    dist2 = df["label_name"].value_counts()
    ax.pie(dist2.values, labels=dist2.index,
           colors=[colours[l] for l in dist2.index],
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Sentiment Distribution (%)")

    plt.tight_layout()
    path = REPORT_DIR / "overview.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", path)


def plot_sentence_lengths(df):
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
        data_by_class = [df[df["label_name"] == c][col].values for c in order]
        bp = ax.boxplot(data_by_class, tick_labels=order, patch_artist=True)
        for patch, c in zip(bp["boxes"], order):
            patch.set_facecolor(colours[c])
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_ylabel(col.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = REPORT_DIR / "sentence_lengths.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", path)


# ── 4. Split ───────────────────────────────────────────────────────────────────

def split_and_save(df):
    df_train, df_val = smart_split(
        df, target_col=TARGET, task_type="multiclass_classification",
        val_size=0.20, random_state=RANDOM_STATE,
    )
    print("\n  Train: {:,} rows  |  Val: {:,} rows".format(
        len(df_train), len(df_val)))
    print("  Class distribution (train):")
    for lid, cnt in df_train[TARGET].value_counts().sort_index().items():
        print("    {:8s}: {:>4,}  ({:.1f}%)".format(
            LABEL_MAP[lid], cnt, 100 * cnt / len(df_train)))
    df_train.to_parquet(DATA_SUBDIR / "train_raw.parquet", index=False)
    df_val.to_parquet(DATA_SUBDIR / "val_raw.parquet",     index=False)
    log.info("Splits saved to %s", DATA_SUBDIR)
    return df_train, df_val


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 -- Use Case C (NLP): Financial Sentiment")
    print("  Step 1: Data Loading")
    print("=" * 65 + "\n")

    print("[1] Loading Financial PhraseBank...")
    df = load_phrasebank(config="sentences_allagree")

    print("[2] Profiling dataset...")
    profile_data(df)

    print("[3] Generating overview plots...")
    plot_overview(df)
    plot_sentence_lengths(df)

    print("[4] Splitting train / validation...")
    split_and_save(df)

    summary = pd.DataFrame({
        "column":    df.columns,
        "dtype":     [str(df[c].dtype)  for c in df.columns],
        "n_unique":  [df[c].nunique()   for c in df.columns],
        "n_missing": [df[c].isna().sum() for c in df.columns],
    })
    summary.to_csv(REPORT_DIR / "column_summary.csv", index=False)

    print("\n" + "=" * 65)
    print("  Step 1 complete. Ready for EDA (02_eda_analysis.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
