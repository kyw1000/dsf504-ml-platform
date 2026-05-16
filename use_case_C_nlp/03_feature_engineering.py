"""
use_case_C_nlp/03_feature_engineering.py
==========================================
DSF504 — Use Case C (NLP): Market Intelligence — Financial Sentiment
Step 3: Feature Engineering

Two feature tracks
------------------
Track 1 — TF-IDF (fast, interpretable, always runs)
    - Unigrams + bigrams, max 5,000 features
    - sublinear_tf=True (dampens high-frequency dominance)
    - min_df=2 (exclude hapax legomena)
    - Financial stopword extension (adds domain noise words)
    - Saved as dense numpy arrays (.npy)

Track 2 — FinBERT Embeddings (optional, requires torch + transformers)
    - ProsusAI/finbert: finance-domain BERT pretrained on 10K Reuters,
      FT, and Bloomberg news (Huang et al., 2023)
    - Extract [CLS] token embedding (768-dim) for each sentence
    - Batch inference with GPU/MPS detection
    - Saved as .npy arrays

Additional hand-crafted features
    - word_count, char_count (sentence complexity)
    - Loughran-McDonald (LM) word list counts:
        positive, negative, uncertainty, litigious, modal-strong

Academic references
-------------------
- Huang A. et al. (2023). FinBERT. Contemporary Accounting Research.
- Loughran T. & McDonald B. (2011). When is a Liability not a Liability?
  Journal of Finance.
"""

from __future__ import annotations

import sys
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE

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

TEXT_COL  = "sentence"
TARGET    = "label"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# ── Loughran-McDonald seed word lists (abbreviated — key terms) ────────────────
LM_POSITIVE = {
    "profitable", "profitability", "profit", "gain", "gains", "growth",
    "improvement", "improved", "increasing", "increase", "higher", "strong",
    "strength", "record", "robust", "exceeded", "exceed", "growth", "benefit",
    "efficient", "recovery", "advance", "positive", "good", "better", "best",
    "outstanding", "excellent", "success", "successful", "opportunity", "optimistic",
}

LM_NEGATIVE = {
    "loss", "losses", "decline", "declined", "decrease", "decreased", "lower",
    "weak", "weakness", "deficit", "debt", "risk", "risks", "concern", "concerns",
    "uncertainty", "uncertain", "difficult", "difficulties", "challenging", "challenge",
    "impairment", "write-off", "writeoff", "downturn", "recession", "default",
    "bankrupt", "bankruptcy", "restructuring", "layoff", "layoffs", "cut", "cuts",
    "reduction", "reduced", "miss", "missed", "adverse", "deterioration",
}

LM_UNCERTAINTY = {
    "approximately", "roughly", "may", "might", "could", "possibly", "potential",
    "uncertain", "uncertainty", "estimate", "estimated", "expect", "expected",
    "anticipate", "anticipated", "pending", "projected", "projection", "forecast",
}

LM_MODAL_STRONG = {
    "will", "shall", "must", "require", "required", "commit", "committed",
    "definite", "definitely", "certainly", "certain",
}

# Financial domain stopwords to add to sklearn's english list
FIN_STOPWORDS_EXTRA = {
    "company", "companies", "corp", "inc", "ltd", "plc", "group", "said",
    "year", "quarter", "annual", "fiscal", "report", "reported", "says",
    "million", "billion", "per", "cent", "percent",
}


# ── Track 1: TF-IDF ───────────────────────────────────────────────────────────

def build_tfidf(
    train_texts: list[str],
    val_texts:   list[str],
    max_features: int = 5000,
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Fit TF-IDF on training corpus, transform both splits.

    Design choices
    --------------
    - sublinear_tf=True: log(1+tf) dampens very frequent terms in long sentences
    - ngram_range=(1,2): bigrams capture "operating loss", "net income", etc.
    - min_df=2: remove vocabulary items appearing in only one document
    - max_df=0.95: ignore terms appearing in >95% of documents (near-stopwords)
    - analyzer="word": word-level tokenisation (not char n-grams)
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop_words = list(ENGLISH_STOP_WORDS | FIN_STOPWORDS_EXTRA)

    log.info(f"Fitting TF-IDF vectorizer (max_features={max_features})…")
    vectorizer = TfidfVectorizer(
        ngram_range  = (1, 2),
        max_features = max_features,
        min_df       = 2,
        max_df       = 0.95,
        sublinear_tf = True,
        analyzer     = "word",
        token_pattern= r"(?u)\b[a-zA-Z][a-zA-Z-]{1,}\b",  # min 2 chars
        stop_words   = stop_words,
    )
    X_train = vectorizer.fit_transform(train_texts).toarray().astype(np.float32)
    X_val   = vectorizer.transform(val_texts).toarray().astype(np.float32)

    log.info(f"TF-IDF shape: train={X_train.shape}  val={X_val.shape}")
    return X_train, X_val, vectorizer


# ── Track 2: FinBERT Embeddings ───────────────────────────────────────────────

def build_finbert_embeddings(
    train_texts: list[str],
    val_texts:   list[str],
    model_name:  str = "ProsusAI/finbert",
    batch_size:  int = 32,
    max_length:  int = 128,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Generate [CLS] token embeddings from FinBERT for each sentence.

    FinBERT (Huang et al., 2023) is a BERT model fine-tuned on:
    - 10,000 Reuters financial news articles
    - Bloomberg financial news
    - Financial Times articles

    The [CLS] embedding is a 768-dimensional dense representation that
    encodes semantic, syntactic, and financial domain knowledge.

    Returns
    -------
    train_emb, val_emb : (N, 768) float32 arrays, or (None, None) if
                         transformers / torch is not available.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        log.warning(
            "transformers or torch not installed. Skipping FinBERT embeddings.\n"
            "Run: pip install transformers torch --break-system-packages"
        )
        return None, None

    device = (
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info(f"Loading FinBERT from HuggingFace: {model_name}  (device={device})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device).eval()

    def embed_batch(texts: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc   = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS]
            all_embeddings.append(cls_emb)
            if (i // batch_size) % 5 == 0:
                log.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} sentences")
        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    log.info("Generating FinBERT embeddings for training set…")
    t0 = time.time()
    train_emb = embed_batch(train_texts)
    log.info(f"Train embeddings: {train_emb.shape}  ({time.time()-t0:.0f}s)")

    log.info("Generating FinBERT embeddings for validation set…")
    t1 = time.time()
    val_emb = embed_batch(val_texts)
    log.info(f"Val embeddings  : {val_emb.shape}  ({time.time()-t1:.0f}s)")

    return train_emb, val_emb


# ── Track 3: Hand-crafted features ───────────────────────────────────────────

def build_handcrafted(texts: list[str]) -> np.ndarray:
    """
    Lightweight lexical features:
    0: word_count
    1: char_count
    2: avg_word_length
    3: lm_positive_count
    4: lm_negative_count
    5: lm_uncertainty_count
    6: lm_modal_strong_count
    7: positive_negative_ratio  (lm_pos / (lm_neg + 1))
    """
    rows = []
    for text in texts:
        words     = text.lower().split()
        word_set  = set(words)
        n_words   = len(words)
        n_chars   = len(text)
        avg_wlen  = np.mean([len(w) for w in words]) if words else 0.0
        n_pos     = len(word_set & LM_POSITIVE)
        n_neg     = len(word_set & LM_NEGATIVE)
        n_unc     = len(word_set & LM_UNCERTAINTY)
        n_modal   = len(word_set & LM_MODAL_STRONG)
        pn_ratio  = n_pos / (n_neg + 1)
        rows.append([n_words, n_chars, avg_wlen, n_pos, n_neg, n_unc, n_modal, pn_ratio])
    return np.array(rows, dtype=np.float32)


HC_FEATURE_NAMES = [
    "word_count", "char_count", "avg_word_length",
    "lm_positive", "lm_negative", "lm_uncertainty", "lm_modal_strong",
    "lm_pos_neg_ratio",
]


# ── Combine and save ──────────────────────────────────────────────────────────

def combine_features(
    tfidf: np.ndarray,
    hc:    np.ndarray,
    finbert: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate TF-IDF + hand-crafted (+ optional FinBERT) features."""
    parts = [tfidf, hc]
    if finbert is not None:
        parts.append(finbert)
    return np.concatenate(parts, axis=1)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_tfidf_top_terms(
    vectorizer: TfidfVectorizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    top_n: int = 20,
) -> None:
    """Mean TF-IDF weight per class for top discriminative terms."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    colours = {"negative": "#E53935", "neutral": "#FB8C00", "positive": "#43A047"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f"Top {top_n} TF-IDF Terms by Mean Weight per Sentiment Class",
                 fontsize=12, fontweight="bold")

    for ax, (label_id, label_name) in zip(axes, LABEL_MAP.items()):
        mask        = y_train == label_id
        mean_tfidf  = X_train[mask].mean(axis=0)
        top_idx     = np.argsort(mean_tfidf)[-top_n:]
        top_terms   = feature_names[top_idx]
        top_weights = mean_tfidf[top_idx]

        ax.barh(top_terms, top_weights, color=colours[label_name], alpha=0.85)
        ax.set_title(f"{label_name.capitalize()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean TF-IDF weight")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = REPORT_DIR / "tfidf_top_terms.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


def plot_handcrafted_dist(
    hc_train: np.ndarray,
    y_train:  np.ndarray,
) -> None:
    """Distribution of hand-crafted features by sentiment class."""
    colours = {0: "#E53935", 1: "#FB8C00", 2: "#43A047"}
    show    = ["word_count", "lm_positive", "lm_negative", "lm_pos_neg_ratio"]
    idx     = [HC_FEATURE_NAMES.index(f) for f in show]

    fig, axes = plt.subplots(1, len(show), figsize=(16, 4))
    fig.suptitle("Hand-crafted Feature Distributions by Sentiment",
                 fontsize=12, fontweight="bold")

    for ax, feat_idx, feat_name in zip(axes, idx, show):
        for label_id, colour in colours.items():
            mask   = y_train == label_id
            values = hc_train[mask, feat_idx]
            ax.hist(values, bins=20, alpha=0.6, color=colour,
                    label=LABEL_MAP[label_id])
        ax.set_title(feat_name.replace("_", " ").title(), fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = REPORT_DIR / "handcrafted_features.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  DSF504 — Use Case C (NLP): Feature Engineering")
    print("=" * 65 + "\n")

    # Load splits
    train_path = DATA_SUBDIR / "train_raw.parquet"
    val_path   = DATA_SUBDIR / "val_raw.parquet"

    if not train_path.exists():
        raise FileNotFoundError("Run 01_data_loading.py first.")

    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)
    print(f"  Train: {len(df_train):,}  |  Val: {len(df_val):,}")

    train_texts = df_train[TEXT_COL].tolist()
    val_texts   = df_val[TEXT_COL].tolist()
    y_train     = df_train[TARGET].values.astype(int)
    y_val       = df_val[TARGET].values.astype(int)

    # ── Track 1: TF-IDF ──────────────────────────────────────────────────────
    print("\n[1] Building TF-IDF features…")
    X_tfidf_train, X_tfidf_val, vectorizer = build_tfidf(train_texts, val_texts)
    joblib.dump(vectorizer, DATA_SUBDIR / "tfidf_vectorizer.pkl")
    np.save(DATA_SUBDIR / "X_tfidf_train.npy", X_tfidf_train)
    np.save(DATA_SUBDIR / "X_tfidf_val.npy",   X_tfidf_val)
    print(f"  TF-IDF: {X_tfidf_train.shape[1]} features")

    # ── Track 2: Hand-crafted ─────────────────────────────────────────────────
    print("\n[2] Building hand-crafted features (LM word lists)…")
    hc_train = build_handcrafted(train_texts)
    hc_val   = build_handcrafted(val_texts)
    np.save(DATA_SUBDIR / "X_hc_train.npy", hc_train)
    np.save(DATA_SUBDIR / "X_hc_val.npy",   hc_val)
    print(f"  Hand-crafted: {hc_train.shape[1]} features")

    # ── Track 3: FinBERT embeddings ───────────────────────────────────────────
    print("\n[3] Generating FinBERT embeddings…")
    finbert_train, finbert_val = build_finbert_embeddings(train_texts, val_texts)

    if finbert_train is not None:
        np.save(DATA_SUBDIR / "X_finbert_train.npy", finbert_train)
        np.save(DATA_SUBDIR / "X_finbert_val.npy",   finbert_val)
        print(f"  FinBERT: {finbert_train.shape[1]} embedding dimensions")
    else:
        print("  FinBERT skipped — models will use TF-IDF + hand-crafted only")

    # ── Combined feature matrices ─────────────────────────────────────────────
    print("\n[4] Combining features…")
    X_combined_train = combine_features(X_tfidf_train, hc_train,
                                         finbert_train if finbert_train is not None else None)
    X_combined_val   = combine_features(X_tfidf_val,   hc_val,
                                         finbert_val   if finbert_val   is not None else None)
    np.save(DATA_SUBDIR / "X_combined_train.npy", X_combined_train)
    np.save(DATA_SUBDIR / "X_combined_val.npy",   X_combined_val)
    np.save(DATA_SUBDIR / "y_train.npy", y_train)
    np.save(DATA_SUBDIR / "y_val.npy",   y_val)

    print(f"  Combined train: {X_combined_train.shape}")
    print(f"  Combined val  : {X_combined_val.shape}")

    # ── Feature summary table ─────────────────────────────────────────────────
    summary = {
        "Feature track":   ["TF-IDF", "Hand-crafted (LM)", "FinBERT (optional)", "Combined"],
        "Dimensions":      [
            X_tfidf_train.shape[1],
            hc_train.shape[1],
            finbert_train.shape[1] if finbert_train is not None else "N/A",
            X_combined_train.shape[1],
        ],
        "Train rows":      [X_tfidf_train.shape[0]] * 4,
        "Saved":           ["X_tfidf_train/val.npy", "X_hc_train/val.npy",
                            "X_finbert_train/val.npy", "X_combined_train/val.npy"],
    }
    pd.DataFrame(summary).to_csv(REPORT_DIR / "feature_summary.csv", index=False)

    # ── Visualisations ────────────────────────────────────────────────────────
    print("\n[5] Generating visualisations…")
    plot_tfidf_top_terms(vectorizer, X_tfidf_train, y_train)
    plot_handcrafted_dist(hc_train, y_train)

    print("\n" + "=" * 65)
    print("  Step 3 complete. Ready for model training (04_model_training.py)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
