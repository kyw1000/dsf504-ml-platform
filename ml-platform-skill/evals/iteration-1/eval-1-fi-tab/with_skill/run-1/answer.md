# Feature Importance Tab — Model Evaluation Page

## Changes required in `dashboard/app.py`

### 1. Updated `st.tabs()` call (line ~4662)

Replace the existing three-tab declaration:

```python
# BEFORE
tab_metrics, tab_curves, tab_thresh = st.tabs([
    "🏆 Champion Metrics",
    "📈 ROC / PR Curves",
    "📏 Threshold Calibration",
])
```

with:

```python
# AFTER
tab_metrics, tab_curves, tab_thresh, tab_fi = st.tabs([
    "🏆 Champion Metrics",
    "📈 ROC / PR Curves",
    "📏 Threshold Calibration",
    "📊 Feature Importance",
])
```

---

### 2. New `tab_fi` block (paste immediately after the `with tab_thresh:` block)

```python
with tab_fi:
    st.markdown(
        "<p style='color:#888;font-size:0.88rem;margin-bottom:12px;'>"
        "Top 20 feature importances from the Optuna-tuned LightGBM champion model "
        "(<code>final_model.pkl</code>). Importance is the total gain split across "
        "all trees.</p>",
        unsafe_allow_html=True,
    )

    _BG, _FONT = "#1A1A2E", "#E0E0E0"

    _fi_model_path = str(ROOT / "models" / "use_case_A" / "final_model.pkl")
    _fi_model = load_model(_fi_model_path)

    if _fi_model is None:
        st.warning(
            "Champion model not found at `models/use_case_A/final_model.pkl`. "
            "Run **Step 5 — HP Tuning + Final Training** to generate it."
        )
    else:
        try:
            # LightGBM Booster (native) or sklearn wrapper — handle both
            import lightgbm as lgb

            if hasattr(_fi_model, "booster_"):
                # sklearn API (LGBMClassifier / LGBMRegressor)
                _booster = _fi_model.booster_
                _fi_names = _booster.feature_name()
                _fi_scores = _booster.feature_importance(importance_type="gain")
            elif hasattr(_fi_model, "feature_importance"):
                # Native Booster
                _fi_names = _fi_model.feature_name()
                _fi_scores = _fi_model.feature_importance(importance_type="gain")
            elif hasattr(_fi_model, "feature_importances_"):
                # Generic sklearn fallback (e.g. Pipeline wrapper)
                _fi_scores = _fi_model.feature_importances_
                _fi_names = [f"feature_{i}" for i in range(len(_fi_scores))]
            else:
                raise AttributeError("Cannot extract feature importances from model object.")

            import numpy as np
            import pandas as pd
            import plotly.graph_objects as go

            _fi_df = (
                pd.DataFrame({"feature": _fi_names, "importance": _fi_scores})
                .sort_values("importance", ascending=False)
                .head(20)
                .sort_values("importance", ascending=True)   # horizontal bar: highest at top
                .reset_index(drop=True)
            )

            # Normalise to 0-100 for readability
            _fi_max = _fi_df["importance"].max()
            _fi_df["importance_norm"] = (
                (_fi_df["importance"] / _fi_max * 100) if _fi_max > 0 else _fi_df["importance"]
            )

            fig_fi = go.Figure(
                go.Bar(
                    x=_fi_df["importance_norm"],
                    y=_fi_df["feature"],
                    orientation="h",
                    marker_color=PALETTE["primary"],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Normalised importance: %{x:.1f}<br>"
                        "Raw gain: %{customdata:,.0f}<extra></extra>"
                    ),
                    customdata=_fi_df["importance"],
                )
            )
            fig_fi.update_layout(
                plot_bgcolor=_BG,
                paper_bgcolor=_BG,
                font_color=_FONT,
                height=560,
                margin=dict(l=0, r=20, t=30, b=40),
                xaxis=dict(
                    title="Feature importance (gain, normalised 0–100)",
                    gridcolor="#2A2A4A",
                    zeroline=False,
                ),
                yaxis=dict(
                    title="",
                    tickfont=dict(size=12),
                    automargin=True,
                ),
            )
            st.plotly_chart(fig_fi, width="stretch")

            with st.expander("Raw importance values"):
                st.dataframe(
                    _fi_df[["feature", "importance"]]
                    .sort_values("importance", ascending=False)
                    .rename(columns={"feature": "Feature", "importance": "Gain (raw)"}),
                    width="stretch",
                    hide_index=True,
                )

        except Exception as _fi_err:
            st.error(f"Could not render feature importances: {_fi_err}")
```

---

## Notes

- `load_model()` is already defined in `app.py` (line ~310) as a `@st.cache_resource` function — it returns `None` when the file is absent, so no extra try/except is needed around the load call itself.
- `ROOT`, `PALETTE`, and `load_model` are all module-level names already available in the function scope.
- The chart uses `width="stretch"` (not the deprecated `use_container_width=True`) and the house-style colours `BG="#1A1A2E"` / `FONT="#E0E0E0"` with `PALETTE["primary"]` for the bars.
- Both the native LightGBM `Booster` and the sklearn `LGBMClassifier` wrapper are handled; a generic `feature_importances_` fallback covers pipelines.
- The tab is named `"📊 Feature Importance"` — no registration is required; Streamlit tabs are plain Python variables.
