"""
utils/eda_viz.py
================
Shared EDA, Feature Engineering & Post-Processing visualization helpers.
Each function saves a PNG and returns a one-sentence insight string.
All insight annotations appear as a light-yellow text box on the chart.

Functions
---------
plot_target_distribution     -- class / label / target histogram
plot_overview_panel          -- 4-panel dataset overview (shape, missing, balance, dtypes)
plot_missing_heatmap         -- missingno-style missing values heatmap
plot_correlation_heatmap     -- Pearson correlation heatmap (top N features)
plot_engineered_feature_summary -- bar chart of new vs original feature counts + types
plot_raw_vs_processed        -- distribution comparison before/after feature engineering
plot_numeric_distributions   -- grid of histograms for top numeric features
plot_class_balance_bar       -- grouped bar of class proportions (train/val/test)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

log = logging.getLogger(__name__)

PALETTE = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC",
           "#26C6DA", "#EC407A", "#D4E157", "#8D6E63", "#78909C"]


def _insight_box(ax, text: str, fontsize: int = 8, loc: str = "bottom") -> None:
    y_pos = 0.02 if loc == "bottom" else 0.96
    va    = "bottom" if loc == "bottom" else "top"
    ax.text(0.01, y_pos, f"[Insight] {text}",
            transform=ax.transAxes, fontsize=fontsize, va=va, ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7",
                      edgecolor="#F9A825", alpha=0.92),
            wrap=True, zorder=10)


def _fig_insight(fig, text: str, fontsize: int = 8) -> None:
    fig.text(0.01, 0.01, f"[Insight] {text}",
             fontsize=fontsize, va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7",
                       edgecolor="#F9A825", alpha=0.92))


# ── 1. Target distribution ────────────────────────────────────────────────────

def plot_target_distribution(
    df: pd.DataFrame, target_col: str,
    report_dir: Path, title_suffix: str = "",
    label_map: Optional[dict] = None,
) -> str:
    vc = df[target_col].value_counts().sort_index()
    labels = [label_map.get(k, str(k)) for k in vc.index] if label_map else [str(k) for k in vc.index]
    pcts   = vc.values / vc.values.sum() * 100
    imb    = float(pcts.max() / max(pcts.min(), 0.01))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vc.values, color=PALETTE[:len(vc)], alpha=0.85, edgecolor="none")
    ax.set_xlabel("Class / Label"); ax.set_ylabel("Count")
    ax.set_title(f"Target Distribution{title_suffix}", fontsize=13)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + vc.values.max() * 0.01,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    if imb > 5:
        insight = (f"Imbalance ratio {imb:.1f}x — minority class will be underrepresented. "
                   "Use stratified splits, class weights, or resampling to compensate.")
    else:
        insight = (f"Class distribution is relatively balanced (ratio {imb:.1f}x). "
                   "Standard splits are appropriate.")

    _insight_box(ax, insight)
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    out = report_dir / "target_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved target_distribution.png")
    return insight


# ── 2. Overview panel ─────────────────────────────────────────────────────────

def plot_overview_panel(
    df: pd.DataFrame, target_col: str,
    report_dir: Path, title_suffix: str = "",
    split_name: str = "train",
) -> str:
    n_rows, n_cols = df.shape
    miss_pct    = df.isnull().mean() * 100
    miss_feats  = int((miss_pct > 0).sum())
    num_cols    = df.select_dtypes(include=[np.number]).columns
    cat_cols    = df.select_dtypes(include=["object", "category"]).columns
    target_vc   = df[target_col].value_counts(normalize=True) if target_col in df.columns else pd.Series()

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: shape summary text
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    summary = (
        f"Rows      : {n_rows:,}\n"
        f"Columns   : {n_cols:,}\n"
        f"Numeric   : {len(num_cols):,}\n"
        f"Categorical: {len(cat_cols):,}\n"
        f"Missing   : {miss_feats} cols ({miss_pct.mean():.2f}% avg)\n"
        f"Target    : {target_col}"
    )
    ax0.text(0.05, 0.95, summary, transform=ax0.transAxes,
             fontsize=11, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))
    ax0.set_title("Dataset Summary", fontsize=11, fontweight="bold")

    # Panel 2: target distribution
    ax1 = fig.add_subplot(gs[0, 1])
    if len(target_vc) <= 12:
        bars = ax1.bar(target_vc.index.astype(str), target_vc.values * 100,
                       color=PALETTE[:len(target_vc)], alpha=0.85)
        ax1.set_ylabel("% of rows"); ax1.set_xlabel(target_col)
        ax1.set_title("Target Distribution", fontsize=11, fontweight="bold")
        for b, v in zip(bars, target_vc.values * 100):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%",
                     ha="center", va="bottom", fontsize=8)
    else:
        ax1.hist(df[target_col].dropna(), bins=40, color=PALETTE[0], alpha=0.75)
        ax1.set_xlabel(target_col); ax1.set_ylabel("Count")
        ax1.set_title("Target Distribution", fontsize=11, fontweight="bold")

    # Panel 3: missing values bar (top 15 cols)
    ax2 = fig.add_subplot(gs[0, 2])
    top_miss = miss_pct[miss_pct > 0].sort_values(ascending=False).head(15)
    if len(top_miss) > 0:
        ax2.barh(top_miss.index, top_miss.values, color=PALETTE[3], alpha=0.75)
        ax2.set_xlabel("Missing %"); ax2.set_title("Top Missing Columns", fontsize=11, fontweight="bold")
        ax2.axvline(5, color="black", ls="--", lw=1, label=">5% threshold")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No missing values", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="green")
        ax2.set_title("Missing Values", fontsize=11, fontweight="bold"); ax2.axis("off")

    # Panel 4: numeric feature value range (box-style min/mean/max)
    ax3 = fig.add_subplot(gs[1, 0])
    if len(num_cols) > 1:
        top_num = num_cols[:12]
        stats = df[top_num].describe().T[["mean", "std"]].reset_index()
        ax3.barh(stats["index"], stats["std"], color=PALETTE[0], alpha=0.7, label="Std Dev")
        ax3.set_xlabel("Std Dev"); ax3.set_title("Feature Variability (Std Dev)", fontsize=11, fontweight="bold")
        ax3.tick_params(axis="y", labelsize=7)

    # Panel 5: dtype breakdown pie
    ax4 = fig.add_subplot(gs[1, 1])
    dtype_counts = df.dtypes.value_counts()
    ax4.pie(dtype_counts.values, labels=[str(d) for d in dtype_counts.index],
            autopct="%1.0f%%", colors=PALETTE[:len(dtype_counts)], startangle=90)
    ax4.set_title("Data Type Breakdown", fontsize=11, fontweight="bold")

    # Panel 6: duplicate / memory
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    mem_mb = df.memory_usage(deep=True).sum() / 1_048_576
    dups   = int(df.duplicated().sum())
    cardinality = {c: int(df[c].nunique()) for c in cat_cols[:5]}
    info = (
        f"Memory     : {mem_mb:.1f} MB\n"
        f"Duplicates : {dups:,}\n"
        f"Split      : {split_name}\n"
        f"\nCat cardinality (top 5):\n" +
        "\n".join(f"  {k}: {v}" for k, v in cardinality.items())
    )
    ax5.text(0.05, 0.95, info, transform=ax5.transAxes,
             fontsize=10, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#F3E5F5", alpha=0.8))
    ax5.set_title("Additional Info", fontsize=11, fontweight="bold")

    imb  = float(target_vc.max() / max(target_vc.min(), 1e-6)) if len(target_vc) >= 2 else 1.0
    insight = (f"{n_rows:,} rows, {n_cols:,} cols, {miss_feats} columns with missing values. "
               f"Target imbalance ratio: {imb:.1f}x. "
               + ("Severe imbalance — use stratified sampling." if imb > 5 else "Balanced dataset."))

    fig.suptitle(f"Dataset Overview{title_suffix}", fontsize=14, fontweight="bold", y=1.01)
    _fig_insight(fig, insight)
    plt.savefig(report_dir / "overview.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved overview.png")
    return insight


# ── 3. Missing values heatmap ─────────────────────────────────────────────────

def plot_missing_heatmap(
    df: pd.DataFrame, report_dir: Path, title_suffix: str = "",
    max_cols: int = 50,
) -> str:
    miss_pct = df.isnull().mean() * 100
    miss_cols = miss_pct[miss_pct > 0].sort_values(ascending=False)
    if len(miss_cols) == 0:
        log.info("No missing values — skipping missing heatmap")
        return "No missing values found in this dataset."

    top_cols = miss_cols.head(max_cols).index.tolist()
    sample   = df[top_cols].sample(min(300, len(df)), random_state=42)
    mask     = sample.isnull().astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Left: heatmap
    ax = axes[0]
    im = ax.imshow(mask.T.values, aspect="auto", cmap="RdYlGn_r",
                   interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(top_cols)))
    ax.set_yticklabels(top_cols, fontsize=max(5, 8 - len(top_cols) // 10))
    ax.set_xlabel("Sample index (random 300 rows)")
    ax.set_title(f"Missing Values Heatmap{title_suffix}\n(red=missing, green=present)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.02)

    # Right: bar chart of miss%
    ax2 = axes[1]
    miss_vals = miss_cols.head(max_cols).values
    ax2.barh(range(len(top_cols)), miss_vals[::-1], color=PALETTE[3], alpha=0.75)
    ax2.set_yticks(range(len(top_cols)))
    ax2.set_yticklabels(top_cols[::-1], fontsize=max(5, 8 - len(top_cols) // 10))
    ax2.set_xlabel("% Missing")
    ax2.set_title("Missing Rate", fontsize=11)
    ax2.axvline(5, color="black", ls="--", lw=1, label="5% line")
    ax2.legend(fontsize=8)

    max_miss = float(miss_cols.max())
    n_high   = int((miss_cols > 20).sum())
    insight  = (f"{len(miss_cols)} features have missing data (max {max_miss:.1f}%). "
                f"{n_high} features exceed 20% missing — consider imputation or dropping.")
    _insight_box(axes[0], insight, fontsize=7)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = report_dir / "missing_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved missing_heatmap.png  (%d cols with missing)", len(miss_cols))
    return insight


# ── 4. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame, report_dir: Path, title_suffix: str = "",
    top_n: int = 25, target_col: Optional[str] = None,
    method: str = "pearson",
) -> str:
    num_df = df.select_dtypes(include=[np.number])
    if target_col and target_col in num_df.columns:
        # Sort by correlation with target
        target_corr = num_df.corr(method=method)[target_col].abs().sort_values(ascending=False)
        top_feats = target_corr.head(top_n).index.tolist()
    else:
        top_feats = num_df.columns[:top_n].tolist()

    sub  = num_df[top_feats].dropna()
    corr = sub.corr(method=method)

    fig, ax = plt.subplots(figsize=(max(10, len(top_feats) * 0.5),
                                    max(9,  len(top_feats) * 0.45)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(top_feats))); ax.set_yticks(range(len(top_feats)))
    ax.set_xticklabels(top_feats, rotation=90, fontsize=max(5, 8 - len(top_feats) // 8))
    ax.set_yticklabels(top_feats, fontsize=max(5, 8 - len(top_feats) // 8))
    ax.set_title(f"{method.capitalize()} Correlation Heatmap{title_suffix}\n(top {len(top_feats)} features)", fontsize=12)

    # Annotate cells if small enough
    if len(top_feats) <= 15:
        for i in range(len(top_feats)):
            for j in range(len(top_feats)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(corr.values[i, j]) < 0.6 else "white")

    # High-correlation pairs
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_pairs = [(c, r, float(upper.at[r, c]))
                  for c in upper.columns for r in upper.index
                  if pd.notna(upper.at[r, c]) and abs(upper.at[r, c]) > 0.85]
    n_high = len(high_pairs)
    insight = (f"{n_high} feature pairs have |correlation| > 0.85 — potential multicollinearity. "
               "Consider dropping redundant features or using PCA."
               if n_high > 0 else
               "No highly correlated feature pairs (|r|>0.85) found — low multicollinearity risk.")

    _insight_box(ax, insight, fontsize=7)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = report_dir / "correlation_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved correlation_heatmap.png  (%d high-corr pairs)", n_high)
    return insight


# ── 5. Engineered feature summary ─────────────────────────────────────────────

def plot_engineered_feature_summary(
    df_raw: pd.DataFrame, df_fe: pd.DataFrame,
    report_dir: Path, title_suffix: str = "",
    target_col: Optional[str] = None,
) -> str:
    raw_cols = set(df_raw.columns) - ({target_col} if target_col else set())
    fe_cols  = set(df_fe.columns)  - ({target_col} if target_col else set())
    new_cols  = fe_cols - raw_cols
    kept_cols = fe_cols & raw_cols
    drop_cols = raw_cols - fe_cols

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: column counts
    ax = axes[0]
    bars = ax.bar(["Original", "Engineered", "New", "Dropped"],
                  [len(raw_cols), len(fe_cols), len(new_cols), len(drop_cols)],
                  color=[PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]], alpha=0.85)
    ax.set_ylabel("# Features")
    ax.set_title("Feature Count: Before vs After FE", fontsize=11, fontweight="bold")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{int(b.get_height())}", ha="center", va="bottom", fontsize=9)

    # Panel 2: dtype breakdown of new features
    ax2 = axes[1]
    if new_cols:
        new_dtypes = df_fe[list(new_cols)].dtypes.astype(str).value_counts()
        ax2.bar(new_dtypes.index, new_dtypes.values, color=PALETTE[:len(new_dtypes)], alpha=0.85)
        ax2.set_ylabel("Count"); ax2.set_title("New Feature Data Types", fontsize=11, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No new features detected", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11)
        ax2.axis("off")

    # Panel 3: target correlation of top new features
    ax3 = axes[2]
    if new_cols and target_col and target_col in df_fe.columns:
        new_num = [c for c in new_cols if df_fe[c].dtype in [np.float64, np.float32, np.int64, np.int32]][:15]
        if new_num:
            corrs = df_fe[new_num + [target_col]].corr()[target_col].drop(target_col).abs().sort_values(ascending=False).head(12)
            ax3.barh(corrs.index, corrs.values, color=PALETTE[2], alpha=0.85)
            ax3.set_xlabel("|Correlation with target|")
            ax3.set_title("Top New Features\nvs Target", fontsize=11, fontweight="bold")
            ax3.tick_params(axis="y", labelsize=8)
        else:
            ax3.text(0.5, 0.5, "No numeric new features", ha="center", va="center",
                     transform=ax3.transAxes, fontsize=11); ax3.axis("off")
    else:
        ax3.text(0.5, 0.5, "No target for correlation", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=11); ax3.axis("off")

    growth = len(new_cols)
    insight = (f"{len(raw_cols)} original → {len(fe_cols)} engineered features. "
               f"{growth} new features added, {len(drop_cols)} dropped. "
               + (f"FE expanded feature space by {growth/max(len(raw_cols),1)*100:.0f}%."
                  if growth > 0 else "Feature space unchanged."))

    fig.suptitle(f"Feature Engineering Summary{title_suffix}", fontsize=13, fontweight="bold")
    _fig_insight(fig, insight)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = report_dir / "engineered_feature_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved engineered_feature_summary.png")
    return insight


# ── 6. Raw vs Processed distribution comparison ───────────────────────────────

def plot_raw_vs_processed(
    df_raw: pd.DataFrame, df_fe: pd.DataFrame,
    report_dir: Path, title_suffix: str = "",
    target_col: Optional[str] = None,
    n_features: int = 6,
) -> str:
    # Find common numeric features
    raw_num = set(df_raw.select_dtypes(include=[np.number]).columns) - ({target_col} if target_col else set())
    fe_num  = set(df_fe.select_dtypes(include=[np.number]).columns)  - ({target_col} if target_col else set())
    common  = list(raw_num & fe_num)[:n_features]
    if not common:
        log.warning("No common numeric features for raw vs processed comparison")
        return "No common numeric features to compare raw vs processed."

    fig, axes = plt.subplots(2, len(common), figsize=(3.5 * len(common), 7))
    if len(common) == 1:
        axes = axes.reshape(2, 1)

    for ci, col in enumerate(common):
        for ri, (df, label, color) in enumerate([(df_raw, "Raw", PALETTE[3]), (df_fe, "Processed", PALETTE[0])]):
            ax = axes[ri, ci]
            vals = df[col].dropna()
            ax.hist(vals, bins=30, color=color, alpha=0.75, edgecolor="none")
            ax.set_title(f"{col}\n({label})", fontsize=8)
            ax.set_xlabel("")
            ax.tick_params(labelsize=7)
            if ci == 0:
                ax.set_ylabel(f"{label}\nCount", fontsize=8)
            # Skewness
            from scipy.stats import skew as scipy_skew
            try:
                sk = float(scipy_skew(vals))
                ax.text(0.97, 0.95, f"skew={sk:.2f}", transform=ax.transAxes,
                        fontsize=6, va="top", ha="right", color="gray")
            except Exception:
                pass

    insight = (f"Comparing distributions of {len(common)} shared features before and after feature engineering. "
               "Check for normalisation, outlier removal, and skewness reduction applied during processing.")

    fig.suptitle(f"Raw vs Processed Feature Distributions{title_suffix}", fontsize=12, fontweight="bold")
    _fig_insight(fig, insight)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = report_dir / "raw_vs_processed_distributions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved raw_vs_processed_distributions.png")
    return insight


# ── 7. Numeric feature distributions grid ────────────────────────────────────

def plot_numeric_distributions(
    df: pd.DataFrame, report_dir: Path, title_suffix: str = "",
    target_col: Optional[str] = None, n_features: int = 12,
) -> str:
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col][:n_features]
    if not num_cols:
        return "No numeric features to plot."
    ncols = min(4, len(num_cols)); nrows = (len(num_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    skews = []
    for ci, col in enumerate(num_cols):
        ax = axes[ci]
        vals = df[col].dropna()
        if target_col and target_col in df.columns and df[target_col].nunique() <= 5:
            for cls, color in zip(sorted(df[target_col].unique()), PALETTE):
                mask = df[target_col] == cls
                ax.hist(df.loc[mask, col].dropna(), bins=25, alpha=0.55, color=color,
                        label=str(cls), density=True)
            ax.legend(fontsize=6)
        else:
            ax.hist(vals, bins=30, color=PALETTE[0], alpha=0.75)
        from scipy.stats import skew as scipy_skew
        try:
            sk = float(scipy_skew(vals))
        except Exception:
            sk = 0.0
        skews.append(abs(sk))
        ax.set_title(f"{col}\nskew={sk:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for ci in range(len(num_cols), len(axes)):
        axes[ci].axis("off")

    high_skew = int(sum(1 for s in skews if s > 1.5))
    insight   = (f"{len(num_cols)} numeric features shown. "
                 f"{high_skew} have |skewness| > 1.5 — consider log/sqrt transforms.")

    fig.suptitle(f"Numeric Feature Distributions{title_suffix}", fontsize=13, fontweight="bold")
    _fig_insight(fig, insight)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = report_dir / "numeric_distributions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved numeric_distributions.png")
    return insight
