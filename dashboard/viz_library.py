"""
dashboard/viz_library.py
========================
DSF504 ML Platform — Interactive Visualization Library

Market Intelligence & Price Prediction Charts
All functions return Plotly figures (dark-themed, ready for st.plotly_chart).
Use width='stretch' when rendering: st.plotly_chart(fig, width='stretch')

Chart inventory
---------------
  kpi_cards()           — metric summary cards row (rendered directly to Streamlit)
  candlestick_chart()   — simulated OHLC from WAP aggregates + rangeslider
  volatility_timeseries()  — mean RV + min/max band + rolling mean + anomaly zone
  rv_heatmap()          — stocks × time-bucket heatmap
  scatter_bubble()      — scatter / bubble with optional OLS trendline
  seasonal_subseries()  — box plots by time period (seasonality patterns)
  forecast_ribbon()     — actual line + forecast line + confidence interval shading
  waterfall_chart()     — cumulative factor-impact waterfall
  indexed_chart()       — normalize series to 100 at baseline for relative comparison
  orderflow_chart()     — buy/sell pressure bars from volume imbalance + spread line
  actual_vs_predicted() — time-series overlay + scatter diagonal side-by-side
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── House colours ──────────────────────────────────────────────────────────────
BG    = "#1A1A2E"
FG    = "#E0E0E0"
BLUE  = "#42A5F5"
GRN   = "#66BB6A"
ORG   = "#FFA726"
RED   = "#EF5350"
PURP  = "#AB47BC"
TEL   = "#26C6DA"
PINK  = "#EC407A"
LIME  = "#D4E157"
GRID  = "#2A2A4A"

PALETTE = [BLUE, ORG, GRN, RED, PURP, TEL, PINK, LIME,
           "#FF8A65", "#78909C", "#FFEE58", "#80DEEA"]


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a 6-digit hex color + float alpha to 'rgba(r,g,b,a)'.

    Plotly candlestick fillcolor does NOT accept 8-digit hex (#RRGGBBAA).
    Use this helper whenever an alpha-blended fill is needed.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Layout helper ──────────────────────────────────────────────────────────────

def _L(fig: go.Figure, h: int = 420, t: int = 40, **kw) -> go.Figure:
    """Apply dark theme + standard margins to a Plotly figure."""
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_color=FG,
        height=h, margin=dict(t=t, b=30, l=50, r=20),
        **kw,
    )
    return fig


# ── KPI cards ──────────────────────────────────────────────────────────────────

def kpi_cards(
    metrics: Dict[str, Any],
    border_colors: Optional[List[str]] = None,
) -> None:
    """
    Render a horizontal row of KPI metric cards directly into Streamlit.

    Parameters
    ----------
    metrics : dict
        Keys = label strings.
        Values = scalar  OR  (value, delta_str, icon_emoji).
    border_colors : list, optional
        Per-card accent colour. Defaults to PALETTE.

    Example
    -------
    kpi_cards({
        "Avg RV":       ("0.00285", "+3.2%", "📈"),
        "Peak RV":      ("0.01420", None,    "🚨"),
        "Anomaly rate": ("4.7%",    None,    "⚠️"),
    })
    """
    colors = border_colors or PALETTE
    cols   = st.columns(len(metrics))
    for i, (label, val) in enumerate(metrics.items()):
        color = colors[i % len(colors)]
        if isinstance(val, (tuple, list)):
            parts = list(val) + [None, None, None]
            value, delta, icon = parts[0], parts[1], parts[2] or "📊"
        else:
            value, delta, icon = val, None, "📊"

        dhtml = ""
        if delta is not None:
            dcol = GRN if (str(delta).startswith("+") or (
                isinstance(delta, (int, float)) and delta >= 0)) else RED
            dhtml = (f"<div style='font-size:0.70rem;color:{dcol};"
                     f"margin-top:2px;font-weight:600;'>{delta}</div>")

        cols[i].markdown(
            f"<div style='background:{color}18;border:1.5px solid {color};"
            f"border-radius:10px;padding:12px 14px;text-align:center;"
            f"min-height:96px;'>"
            f"<div style='font-size:1.5rem;line-height:1;'>{icon}</div>"
            f"<div style='font-size:1.05rem;font-weight:700;color:{color};"
            f"margin-top:4px;'>{value}</div>"
            f"<div style='font-size:0.70rem;color:{FG};opacity:0.75;"
            f"margin-top:2px;'>{label}</div>"
            f"{dhtml}</div>",
            unsafe_allow_html=True,
        )


# ── Candlestick chart ──────────────────────────────────────────────────────────

def candlestick_chart(
    df: pd.DataFrame,
    time_col: str,
    wap_mean_col: str,
    wap_std_col:   Optional[str] = None,
    wap_range_col: Optional[str] = None,
    stock_id: Optional[int] = None,
    title: str = "Price Action — Simulated OHLC from WAP Aggregates",
) -> go.Figure:
    """
    Candlestick chart built from WAP summary statistics.

    OHLC simulation
    ---------------
      Open  ≈ wap_mean − wap_std × 0.5
      Close ≈ wap_mean + wap_std × 0.5
      High  ≈ wap_mean + wap_range × 0.5   (or mean + std if range absent)
      Low   ≈ wap_mean − wap_range × 0.5
    """
    df = df.sort_values(time_col).copy()
    mean = df[wap_mean_col]
    std  = df[wap_std_col].fillna(0)   if (wap_std_col   and wap_std_col   in df.columns) else mean * 0.002
    rng  = df[wap_range_col].fillna(0) if (wap_range_col and wap_range_col in df.columns) else std * 2

    opens  = mean - std * 0.5
    closes = mean + std * 0.5
    highs  = (mean + rng * 0.5).combine(closes, max)
    lows   = (mean - rng * 0.5).combine(opens,  min)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df[time_col],
        open=opens, high=highs, low=lows, close=closes,
        increasing=dict(line=dict(color=GRN), fillcolor=_rgba(GRN, 0.6)),
        decreasing=dict(line=dict(color=RED), fillcolor=_rgba(RED, 0.6)),
        name=f"Stock {stock_id}" if stock_id is not None else "OHLC",
    ))
    fig.add_trace(go.Scatter(
        x=df[time_col], y=mean, mode="lines",
        line=dict(color=BLUE, width=1.2, dash="dot"),
        name="WAP Mean", opacity=0.65,
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Time ID",
            rangeslider=dict(visible=True, thickness=0.05),
            showgrid=True, gridcolor=GRID,
        ),
        yaxis=dict(title="Price (WAP)", showgrid=True, gridcolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return _L(fig, h=520, t=50)


# ── Volatility time series ─────────────────────────────────────────────────────

def volatility_timeseries(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    rolling_n: int = 25,
    stock_overlays: Optional[List[int]] = None,
    df_full: Optional[pd.DataFrame] = None,
    stock_col: str = "stock_id",
    title: str = "Realized Volatility — Time Series with Anomaly Detection",
) -> go.Figure:
    """
    Aggregated time-series with:
    • Min–Max range fill
    • Rolling-mean dotted line
    • 2σ anomaly zone (red shaded)
    • Optional per-stock overlays (up to 5)
    """
    ts = (df.groupby(time_col)[value_col]
           .agg(["mean", "std", "min", "max"])
           .reset_index()
           .sort_values(time_col))

    roll_mean = ts["mean"].rolling(rolling_n, min_periods=1).mean()
    roll_std  = ts["mean"].rolling(rolling_n, min_periods=1).std().fillna(0)
    upper     = roll_mean + 2 * roll_std

    fig = go.Figure()

    # Min–Max band
    fig.add_trace(go.Scatter(
        x=list(ts[time_col]) + list(ts[time_col])[::-1],
        y=list(ts["max"])    + list(ts["min"])[::-1],
        fill="toself", fillcolor="rgba(66,165,245,0.08)",
        line=dict(width=0), name="Min–Max range", hoverinfo="skip",
    ))
    # Mean RV
    fig.add_trace(go.Scatter(
        x=ts[time_col], y=ts["mean"], mode="lines",
        line=dict(color=BLUE, width=1.8), name="Mean RV",
    ))
    # Rolling mean
    fig.add_trace(go.Scatter(
        x=ts[time_col], y=roll_mean, mode="lines",
        line=dict(color=GRN, width=1.5, dash="dot"),
        name=f"Rolling mean (n={rolling_n})",
    ))
    # Anomaly zone
    fig.add_trace(go.Scatter(
        x=list(ts[time_col]) + list(ts[time_col])[::-1],
        y=list(upper)         + list(roll_mean)[::-1],
        fill="toself", fillcolor="rgba(239,83,80,0.12)",
        line=dict(width=0), name="Anomaly zone (>2σ)", hoverinfo="skip",
    ))

    # Per-stock overlays
    src = df_full if df_full is not None else df
    if stock_overlays:
        for i, sid in enumerate(stock_overlays[:5]):
            s = src[src[stock_col] == sid].sort_values(time_col)[[time_col, value_col]]
            fig.add_trace(go.Scatter(
                x=s[time_col], y=s[value_col], mode="lines", opacity=0.75,
                line=dict(width=1.2, color=PALETTE[i + 2]),
                name=f"Stock {sid}",
            ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time ID", rangeslider=dict(visible=True, thickness=0.06),
                   showgrid=True, gridcolor=GRID),
        yaxis=dict(title="Realized Volatility", showgrid=True, gridcolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="x unified",
    )
    return _L(fig, h=500, t=50)


# ── Heatmap ────────────────────────────────────────────────────────────────────

def rv_heatmap(
    df: pd.DataFrame,
    stock_col: str,
    time_col:  str,
    value_col: str,
    n_buckets:  int = 30,
    max_stocks: int = 30,
    title: str = "Stock × Time Volatility Heatmap",
) -> go.Figure:
    """
    Heatmap: stocks (rows) × time buckets (columns) — coloured by mean RV.
    Top-N stocks selected by intra-period variance.
    """
    df = df.copy()
    df["_tb"] = pd.cut(df[time_col], bins=n_buckets,
                       labels=[f"T{i+1}" for i in range(n_buckets)]).astype(str)
    top_ids = (df.groupby(stock_col)[value_col].std()
                .sort_values(ascending=False)
                .head(max_stocks).index.tolist())
    pivot = (df[df[stock_col].isin(top_ids)]
              .groupby([stock_col, "_tb"])[value_col].mean()
              .unstack("_tb").fillna(0))
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[f"Stock {s}" for s in pivot.index],
        colorscale="RdYlBu_r",
        colorbar=dict(title="Realized<br>Volatility", tickfont=dict(color=FG)),
        hovertemplate="Stock %{y}<br>Period %{x}<br>RV: %{z:.6f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Time Bucket", tickangle=-45, showgrid=False, tickfont=dict(size=8)),
        yaxis=dict(title="Stock ID",    showgrid=False, tickfont=dict(size=8)),
    )
    return _L(fig, h=min(620, 240 + max_stocks * 10), t=50)


# ── Scatter / Bubble ───────────────────────────────────────────────────────────

def scatter_bubble(
    df: pd.DataFrame,
    x_col:     str,
    y_col:     str,
    color_col: Optional[str] = None,
    size_col:  Optional[str] = None,
    log_x:     bool = True,
    log_y:     bool = False,
    sample_n:  int  = 5_000,
    title:     str  = "",
) -> go.Figure:
    """
    Scatter / bubble chart with optional OLS trendline.
    Bubble size driven by size_col (e.g., trade volume).
    """
    df = df.dropna(subset=[x_col, y_col]).copy()
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)

    x_vals  = np.log1p(df[x_col].clip(lower=0)) if log_x else df[x_col]
    y_vals  = np.log1p(df[y_col].clip(lower=0)) if log_y else df[y_col]
    xl = f"log1p({x_col})" if log_x else x_col
    yl = f"log1p({y_col})" if log_y else y_col

    plot = df.copy()
    plot[xl] = x_vals
    plot[yl] = y_vals

    kw: dict = dict(
        data_frame=plot, x=xl, y=yl, opacity=0.45,
        color_discrete_sequence=PALETTE,
        labels={xl: xl, yl: yl}, title=title,
    )
    if color_col and color_col in plot.columns:
        kw["color"] = color_col
    if size_col and size_col in plot.columns:
        kw["size"]     = size_col
        kw["size_max"] = 18
    else:
        # OLS trendline requires statsmodels — add only if available
        try:
            import statsmodels  # noqa: F401
            kw["trendline"] = "ols"
        except ImportError:
            pass   # skip trendline silently; no hard dependency

    fig = px.scatter(**kw)
    fig.update_traces(marker=dict(size=4 if not size_col else None))
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor=GRID),
        yaxis=dict(showgrid=True, gridcolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return _L(fig, h=460, t=50 if title else 20)


# ── Seasonal subseries (box plots) ────────────────────────────────────────────

def seasonal_subseries(
    df: pd.DataFrame,
    value_col:  str,
    time_col:   str,
    n_periods:  int = 6,
    period_labels: Optional[List[str]] = None,
    title: str = "Seasonal Subseries — Volatility by Time Period",
) -> go.Figure:
    """
    Box plots of value_col for each equally-spaced time segment.
    Reveals distributional shifts (seasonality) across the dataset.
    """
    df = df.copy()
    labels = period_labels or [f"P{i+1}" for i in range(n_periods)]
    df["_period"] = pd.cut(df[time_col], bins=n_periods, labels=labels).astype(str)

    fig = go.Figure()
    for i, p in enumerate(labels):
        vals = df[df["_period"] == p][value_col].dropna()
        fig.add_trace(go.Box(
            y=vals, name=p,
            marker_color=PALETTE[i % len(PALETTE)],
            boxmean="sd",
            hovertemplate=f"Period: {p}<br>RV: %{{y:.6f}}<extra></extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis=dict(title=value_col, showgrid=True, gridcolor=GRID),
        showlegend=False,
    )
    return _L(fig, h=380, t=50)


# ── Forecast ribbon ────────────────────────────────────────────────────────────

def forecast_ribbon(
    df: pd.DataFrame,
    time_col:      str,
    actual_col:    str,
    pred_col:      Optional[str] = None,
    ci_lower_col:  Optional[str] = None,
    ci_upper_col:  Optional[str] = None,
    rolling_ci_n:  int = 20,
    ci_sigma:      float = 1.5,
    title: str = "Forecast Ribbon — Actual vs Predicted with Confidence Interval",
) -> go.Figure:
    """
    Line chart with:
    • Actual line (blue)
    • Predicted line (orange dashed) — if pred_col provided
    • Confidence interval ribbon (shaded)
      – If ci_lower/upper cols provided: use them directly
      – Otherwise: compute rolling mean ± ci_sigma × std as proxy
    """
    df = df.sort_values(time_col).copy()
    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df[actual_col], mode="lines",
        line=dict(color=BLUE, width=2), name="Actual",
    ))

    # Predicted
    if pred_col and pred_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[time_col], y=df[pred_col], mode="lines",
            line=dict(color=ORG, width=2, dash="dash"), name="Predicted",
        ))

    # CI ribbon
    if ci_upper_col and ci_lower_col and ci_upper_col in df.columns and ci_lower_col in df.columns:
        hi, lo = df[ci_upper_col], df[ci_lower_col]
        ribbon_name = "95% CI"
        ribbon_color = "rgba(255,167,38,0.15)"
    else:
        roll  = df[actual_col].rolling(rolling_ci_n, min_periods=1)
        hi    = roll.mean() + ci_sigma * roll.std().fillna(0)
        lo    = roll.mean() - ci_sigma * roll.std().fillna(0)
        ribbon_name  = f"Rolling ±{ci_sigma}σ band"
        ribbon_color = "rgba(66,165,245,0.12)"

    fig.add_trace(go.Scatter(
        x=list(df[time_col]) + list(df[time_col])[::-1],
        y=list(hi) + list(lo)[::-1],
        fill="toself", fillcolor=ribbon_color,
        line=dict(width=0), name=ribbon_name, hoverinfo="skip",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time ID", rangeslider=dict(visible=True, thickness=0.05),
                   showgrid=True, gridcolor=GRID),
        yaxis=dict(title=actual_col, showgrid=True, gridcolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return _L(fig, h=480, t=50)


# ── Waterfall chart ────────────────────────────────────────────────────────────

def waterfall_chart(
    factors:    List[str],
    values:     List[float],
    base_value: float = 0.0,
    title:      str   = "Factor Impact — Waterfall Chart",
    y_label:    str   = "Impact",
) -> go.Figure:
    """
    Waterfall showing cumulative impact of each factor.
    Positive = upward step (green), negative = downward step (red).
    Typically used for feature importance or price decomposition.
    """
    x_labels = ["Baseline"] + factors + ["Total"]
    measures = ["absolute"] + ["relative"] * len(factors) + ["total"]
    y_values = [base_value] + list(values) + [base_value + sum(values)]

    text_vals = []
    for i, v in enumerate(y_values):
        if i == 0 or i == len(y_values) - 1:
            text_vals.append(f"{v:.5f}")
        else:
            text_vals.append(f"{v:+.5f}")

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        text=text_vals,
        textposition="auto",
        connector=dict(line=dict(color=FG, width=0.5, dash="dot")),
        increasing=dict(marker=dict(color=GRN)),
        decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=ORG)),
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Factor", tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(title=y_label, showgrid=True, gridcolor=GRID),
        showlegend=False,
    )
    return _L(fig, h=440, t=50)


# ── Indexed comparison chart ───────────────────────────────────────────────────

def indexed_chart(
    df: pd.DataFrame,
    time_col:     str,
    value_col:    str,
    id_col:       str,
    base_time_id: Optional[int] = None,
    max_series:   int = 10,
    title: str = "Indexed Volatility — Relative Performance (Base = 100)",
) -> go.Figure:
    """
    Normalize each series to 100 at base_time_id (or first available period)
    so multiple stocks / products can be compared on the same scale.
    """
    df = df.sort_values([id_col, time_col]).copy()
    top_ids = (df.groupby(id_col)[value_col].std()
                .sort_values(ascending=False)
                .head(max_series).index.tolist())
    df = df[df[id_col].isin(top_ids)]
    base_t = base_time_id if base_time_id is not None else df[time_col].min()
    base_vals = df[df[time_col] == base_t].set_index(id_col)[value_col]

    fig = go.Figure()
    for i, sid in enumerate(top_ids):
        s  = df[df[id_col] == sid].sort_values(time_col)
        bv = base_vals.get(sid, s[value_col].iloc[0] if len(s) else None)
        if bv is None or bv == 0:
            continue
        idx = s[value_col] / bv * 100
        fig.add_trace(go.Scatter(
            x=s[time_col], y=idx, mode="lines", opacity=0.80,
            line=dict(width=1.3, color=PALETTE[i % len(PALETTE)]),
            name=f"Stock {sid}",
            hovertemplate=f"Stock {sid}<br>T: %{{x}}<br>Index: %{{y:.1f}}<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dash", line_color=FG, opacity=0.35,
                  annotation_text="Baseline 100", annotation_font_color=FG)
    fig.update_layout(
        title=title,
        xaxis=dict(title="Time ID", showgrid=True, gridcolor=GRID),
        yaxis=dict(title="Index Value", showgrid=True, gridcolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hovermode="x unified",
    )
    return _L(fig, h=460, t=50)


# ── Order flow chart ───────────────────────────────────────────────────────────

def orderflow_chart(
    df: pd.DataFrame,
    time_col:      str,
    imbalance_col: str,
    spread_col:    Optional[str] = None,
    title: str = "Order Flow — Buy / Sell Pressure & Bid-Ask Spread",
) -> go.Figure:
    """
    Bar chart of buy (positive) and sell (negative) pressure derived from
    volume imbalance aggregated over time.  Optionally shows bid-ask spread
    in a linked subplot below.
    """
    ts = (df.groupby(time_col)
           .agg(imb_mean=(imbalance_col, "mean"))
           .reset_index()
           .sort_values(time_col))
    ts["imb_mean"] = ts["imb_mean"].fillna(0)

    buy  = ts["imb_mean"].clip(lower=0)
    sell = ts["imb_mean"].clip(upper=0).abs()

    has_spread = spread_col and spread_col in df.columns
    rows = 2 if has_spread else 1
    row_h = [0.65, 0.35] if has_spread else [1.0]
    subtitles = ["Buy / Sell Pressure (Volume Imbalance)"] + (["Bid-Ask Spread"] if has_spread else [])

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        subplot_titles=subtitles,
                        vertical_spacing=0.08,
                        row_heights=row_h)

    fig.add_trace(go.Bar(x=ts[time_col], y=buy,  name="Buy pressure",
                         marker_color=GRN, opacity=0.80), row=1, col=1)
    fig.add_trace(go.Bar(x=ts[time_col], y=-sell, name="Sell pressure",
                         marker_color=RED, opacity=0.80), row=1, col=1)
    fig.add_hline(y=0, line_color=FG, line_width=0.5, opacity=0.4, row=1, col=1)

    if has_spread:
        sp = (df.groupby(time_col)[spread_col].mean()
               .reset_index().sort_values(time_col))
        fig.add_trace(go.Scatter(
            x=sp[time_col], y=sp[spread_col], mode="lines",
            line=dict(color=PURP, width=1.3), name="Spread",
            fill="tozeroy", fillcolor=_rgba(PURP, 0.15),
        ), row=2, col=1)

    fig.update_layout(
        title=title,
        barmode="relative",
        xaxis=dict(showgrid=True, gridcolor=GRID),
        yaxis=dict(title="Imbalance", showgrid=True, gridcolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return _L(fig, h=480 if not has_spread else 560, t=50)


# ── Actual vs Predicted ────────────────────────────────────────────────────────

def actual_vs_predicted(
    df: pd.DataFrame,
    actual_col: str,
    pred_col:   str,
    time_col:   Optional[str] = None,
    title: str = "Actual vs Predicted — Model Accuracy Over Time",
) -> go.Figure:
    """
    Two-panel layout:
    Left  — time-series overlay of actual and predicted
    Right — scatter against the perfect-fit diagonal (y = x)
    """
    df = df.dropna(subset=[actual_col, pred_col]).copy()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Over Time", "Scatter vs Perfect Fit (y = x)"],
        horizontal_spacing=0.08,
    )

    x_ax = df[time_col] if (time_col and time_col in df.columns) else df.index

    # Time series
    fig.add_trace(go.Scatter(x=x_ax, y=df[actual_col], mode="lines",
                             line=dict(color=BLUE, width=1.6), name="Actual"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x_ax, y=df[pred_col], mode="lines",
                             line=dict(color=ORG, width=1.6, dash="dash"), name="Predicted"),
                  row=1, col=1)

    # Scatter diagonal
    mn = min(df[actual_col].min(), df[pred_col].min())
    mx = max(df[actual_col].max(), df[pred_col].max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(color=FG, dash="dot", width=1),
                             showlegend=False, opacity=0.4),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=df[actual_col], y=df[pred_col], mode="markers",
                             marker=dict(color=BLUE, size=4, opacity=0.5),
                             name="Actual vs Predicted"),
                  row=1, col=2)

    fig.update_xaxes(title_text="Time" if time_col else "Index", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Actual", row=1, col=2)
    fig.update_yaxes(title_text="Predicted", row=1, col=2)
    return _L(fig, h=400, t=50)
