"""
use_case_C_nlp/02_eda_analysis.py
===================================
DSF504 — Use Case C (NLP): Market Intelligence — Financial Sentiment
Step 2: Exploratory Data Analysis

Key findings targeted
---------------------
1. Class imbalance — neutral dominates (~60%), drives macro-F1 choice
2. Vocabulary richness per sentiment class
3. Top discriminative unigrams / bigrams per class
4. Sentence complexity metrics (Flesch readability)
5. Lexical overlap between classes — what makes sentiment ambiguous

Academic references
-------------------
- Loughran & McDonald (2011): finance-specific sentiment word lists
- Malo et al. (2014): annotation methodology and inter-rater agreement
"""

from __future__ import annotations

import sys
import re
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR

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
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "label"
TEXT_COL = "sentence"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
COLOURS   = {"negative": "#E53935", "neutral": "#FB8C00", "positive": "#43A047"}

# Common English stopwords (minimal — we want financial terms to appear)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "will", "would", "could", "should", "may", "might",
    "its", "it", "this", "that", "these", "those", "as", "than", "also",
    "more", "not", "no", "s", "than", "which", "while", "about", "after",
    "over", "up", "during", "into", "through", "between", "each", "both",
}


def load() -> pd.DataFrame:
    path = DATA_SUBDIR / "train_raw.parquet"
    if not path.exists():
        path = DATA_SUBDIR / "phrasebank_sentences_allagree.parquet"
    if not path.exists():
        raise FileNotFoundError("Run 01_data_loading.py first.")
    log.info(f"Loading from {path.name}…")
    df = pd.read_parquet(path)
    if "label_name" not in df.columns:
        df["label_name"] = df[TARGET].map(LABEL_MAP)
    return df


# ── 1. Class imbalance ─────────────────────────────────────────────────────────

def analyse_imbalance(df: pd.DataFrame) -> None:
    print("\n--- Class Imbalance ---")
    dist = df[TARGET].value_counts().sort_index()
    for label_id, count in dist.items():
        name = LABEL_MAP[label_id]
        pct  = 100 * count / len(df)
        print(f"  {label_id} ({name:8s}): {count:>5,}  ({pct:.1f}%)")
    print(f"\n  Imbalance → macro-F1 chosen over accuracy")
    print(f"  Class weights applied in all classifiers")


# ── 2. Vocabulary analysis ─────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser (lowercase)."""
    tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def get_top_ngrams(
    df: pd.DataFrame,
    label_name: str,
    n: int = 1,
    top_k: int = 20,
) -> list[tuple[str, int]]:
    """Return top-k n-grams for a given sentiment class."""
    texts  = df[df["label_name"] == label_name][TEXT_COL].tolist()
    tokens = []
    for text in texts:
        words = tokenise(text)
        if n == 1:
            tokens.extend(words)
        else:
            tokens.extend([" ".join(words[i:i+n]) for i in range(len(words)-n+1)])
    return Counter(tokens).most_common(top_k)


def analyse_vocabulary(df: pd.DataFrame) -> None:
    print("\n--- Vocabulary Analysis ---")
    for label_id in sorted(LABEL_MAP.keys()):
        name  = LABEL_MAP[label_id]
        texts = df[df["label_name"] == name][TEXT_COL].tolist()
        vocab = set()
        for text in texts:
            vocab.update(tokenise(text))
        avg_len = np.mean([len(tokenise(t)) for t in texts])
        print(f"  {name:8s}: unique vocab={len(vocab):,}  avg content words/sentence={avg_len:.1f}")


# ── 3. Top n-gram plots ────────────────────────────────────────────────────────

def plot_top_ngrams(df: pd.DataFrame, n: int = 1, top_k: int = 15) -> None:
    """Bar chart of top unigrams (n=1) or bigrams (n=2) per sentiment class."""
    title  = "Unigrams" if n == 1 else "Bigrams"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Top {top_k} {title} per Sentiment Class",
                 fontsize=13, fontweight="bold")

    for ax, label_name in zip(axes, ["negative", "neutral", "positive"]):
        ngrams = get_top_ngrams(df, label_name, n=n, top_k=top_k)
        if not ngrams:
            continue
        terms, counts = zip(*ngrams)
        colour = COLOURS[label_name]
        ax.barh(list(reversed(terms)), list(reversed(counts)),
                color=colour, alpha=0.85)
        ax.set_title(f"{label_name.capitalize()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frequency")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = REPORT_DIR / f"top_{title.lower()}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ── 4. Readability & complexity ────────────────────────────────────────────────

def flesch_reading_ease(text: str) -> float:
    """
    Approximate Flesch Reading Ease score.
    Higher = easier to read. Financial sentences typically score 20–50.
    """
    sentences = max(len(re.split(r"[.!?]+", text)), 1)
    words     = text.split()
    if not words:
        return 0.0
    syllables = sum(max(1, len(re.findall(r"[aeiouAEIOU]", w))) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


def plot_readability(df: pd.DataFrame) -> None:
    df = df.copy()
    df["flesch"] = df[TEXT_COL].apply(flesch_reading_ease)
    df["word_count"] = df[TEXT_COL].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Text Complexity by Sentiment Class",
                 fontsize=12, fontweight="bold")

    for label_name, colour in COLOURS.items():
        subset = df[df["label_name"] == label_name]
        axes[0].hist(subset["flesch"],    bins=20, alpha=0.6, color=colour, label=label_name)
        axes[1].hist(subset["word_count"],bins=20, alpha=0.6, color=colour, label=label_name)

    axes[0].set_title("Flesch Reading Ease Score")
    axes[0].set_xlabel("Score (higher = easier)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Sentence Word Count")
    axes[1].set_xlabel("Words")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = REPORT_DIR / "readability_complexity.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ── 5. Discriminative power heatmap ───────────────────────────────────────────

def plot_term_heatmap(df: pd.DataFrame, top_terms: int = 20) -> None:
    """
    Heatmap: how often do the top discriminative terms appear per class?
    Uses term frequency ratio to pick class-specific terms.
    """
    all_ngrams = {}
    for label_name in LABEL_MAP.values():
        freq = dict(get_top_ngrams(df, label_name, n=1, top_k=40))
        all_ngrams[label_name] = freq

    # Score terms by max-to-sum ratio (discriminativeness)
    all_terms = set()
    for freq in all_ngrams.values():
        all_terms.update(freq.keys())

    rows = []
    for term in all_terms:
        counts = [all_ngrams[cls].get(term, 0) for cls in LABEL_MAP.values()]
        if sum(counts) < 3:
            continue
        max_count = max(counts)
        total     = sum(counts) + 1
        disc      = max_count / total          # high → one class dominates
        rows.append({"term": term, "disc": disc, **dict(zip(LABEL_MAP.values(), counts))})

    disc_df = pd.DataFrame(rows).sort_values("disc", ascending=False).head(top_terms)
    disc_df = disc_df.set_index("term")[list(LABEL_MAP.values())]

    # Normalise each row to 0-1
    row_sums = disc_df.sum(axis=1).replace(0, 1)
    heat_df  = disc_df.div(row_sums, axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heat_df.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(heat_df.columns)))
    ax.set_xticklabels(heat_df.columns, fontsize=11)
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Relative frequency (row-normalised)")
    ax.set_title("Discriminative Term Heatmap\n"
                 "(High value in one column → class-specific term)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = REPORT_DIR / "term_heatmap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ── 6. EDA summary ────────────────────────────────────────────────────────────

def print_eda_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  EDA SUMMARY — KEY FINDINGS")
    print("=" * 65)
    dist = df[TARGET].value_counts().sort_index()
    for label_id, count in dist.items():
        print(f"  {LABEL_MAP[label_id]:8s}: {100*count/len(df):.1f}% → class_weight='balanced'")
    print()
    print("  Feature Engineering Priorities:")
    print("    1. TF-IDF (1+2-grams, max 5000) — fast, interpretable baseline")
    print("    2. FinBERT CLS embeddings (768-dim) — finance-domain BERT")
    print("    3. Sentence length features (word count, char count)")
    print("    4. Loughran-McDonald financial word counts (positive/negative/uncertain)")
    print()
    print("  Model Strategy:")
    print("    Baseline : Logistic Regression + TF-IDF (industry standard)")
    print("    Advanced : LightGBM + TF-IDF")
    print("    Champion : LightGBM + FinBERT embeddings")
    print("    Metric   : Macro-F1 (handles class imbalance)")
    print("=" * 65 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case C (NLP): EDA Analysis")
    print("=" * 65 + "\n")

    df = load()

    print("[1] Class imbalance…")
    analyse_imbalance(df)

    print("\n[2] Vocabulary analysis…")
    analyse_vocabulary(df)

    print("[3] Top unigrams per class…")
    plot_top_ngrams(df, n=1, top_k=15)

    print("[4] Top bigrams per class…")
    plot_top_ngrams(df, n=2, top_k=12)

    print("[5] Readability & complexity…")
    plot_readability(df)

    print("[6] Discriminative term heatmap…")
    plot_term_heatmap(df, top_terms=20)

    print_eda_summary(df)

    # ── Supplemental standardised EDA plots ──────────────────────────────
    print("[7] Supplemental standardised visualizations…")
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.eda_viz import plot_target_distribution, plot_overview_panel
        plot_target_distribution(df, TARGET, REPORT_DIR, " — UC C_nlp Sentiment",
                                 label_map={0: "Negative", 1: "Neutral", 2: "Positive"})
        plot_overview_panel(df, TARGET, REPORT_DIR, " — UC C_nlp Sentiment")
        print("    Saved: target_distribution.png, overview.png")
    except Exception as _e:
        print(f"    [warn] Supplemental plots skipped: {_e}")

    print(f"  All EDA outputs saved to: {REPORT_DIR}")
    print("  Ready for feature engineering (03_feature_engineering.py)")


if __name__ == "__main__":
    main()
