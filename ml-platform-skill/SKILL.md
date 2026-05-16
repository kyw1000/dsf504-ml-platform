---
name: ml-platform
description: >
  Expert assistant for the DSF504 ML Platform — a Streamlit multi-use-case
  machine learning dashboard with five-step pipeline scripts.

  Use this skill whenever the user asks about, works on, or debugs anything
  in the DSF504 project, including: the Streamlit dashboard (app.py), pipeline
  scripts (01–05 per use case), config.py, utils/, or any page/tab/chart
  inside the platform. Also trigger for requests like "add a new use case",
  "add a page / tab", "fix this dashboard error", "improve the correlation
  chart", or any question about the ML Framework phases used in this project.

  Trigger even when the user says something casual like "the dashboard is
  broken" or "can you add something to the model page" — if it's about
  DSF504, always invoke this skill.
---

# DSF504 ML Platform — Expert Skill

## Project at a Glance

| Item | Value |
|------|-------|
| Root | `C:\DSF504\` (bash: `/sessions/.../mnt/DSF504/`) |
| Dashboard entry | `dashboard/app.py` (~5 300 lines) |
| Run command | `streamlit run dashboard/app.py` (from `C:\DSF504\`) |
| Python | 3.13, Windows; use `pd`, `np`, `plotly`, `go`, `px`, `make_subplots` |
| Active use cases | **A** (fraud), **B** (credit), **C\_nlp** (NLP sentiment) |
| Scaffolded | C\_market, D\_churn, E\_insurance, F\_esg, G\_advisory |

## ML Framework — 8 Dashboard Pages

The sidebar nav mirrors the ML workflow exactly (top → bottom = first → last):

```
▶️  Run Pipeline          ← execute any of the 5 steps
🔬 Data Studio            ← raw data browser + statistical profiling + correlation matrix
🔧 Data Preparation       ← feature engineering guidance + run Step 3
📈 Post-Processing EDA    ← raw vs processed distribution comparison
🤖 Model Development      ← algorithm comparison (Step 4) + CV explorer + HP guide
📊 Model Evaluation       ← champion model metrics + threshold calibration
🔍 Ethics & Explainability← SHAP + fairness & bias audit
🎯 Prediction Demo        ← live inference on the tuned champion
```

The PAGES dict (bottom of app.py) and `_NAV_SECTIONS` inside `render_sidebar()`
**must stay byte-identical** — any label mismatch causes a nav button that never
highlights the active page.

## Five-Step Pipeline (per use case)

```
Step 1  01_data_loading.py       → Parquet splits (train/val/test)
Step 2  02_eda_analysis.py       → CSV/PNG reports in reports/<uc>/
Step 3  03_feature_engineering.py→ train_fe.parquet, feature list CSV
Step 4  04_model_training.py     → champion.pkl, model_comparison.csv
Step 5  05_hyperparameter_tuning.py → final_model.pkl, tuning logs
```

All scripts live in `use_case_<X>/`. They follow an identical header pattern —
see `references/architecture.md` for the full import / logging / ensure_utf8
boilerplate every script must have.

## Key Patterns — Quick Reference

### Adding a new page

1. Write `def page_my_page(uc_key: str) -> None:` anywhere in app.py.
2. Add the entry to **both** PAGES dict and `_NAV_SECTIONS` inside
   `render_sidebar()` using the **same label string** (copy-paste to be safe).
3. If the page needs a "Run Step X" button, call
   `_run_step_action(step, uc_key, "label  (goes to Run Pipeline)")`.
   Add `suffix="unique_id"` if the same (step, uc_key) pair appears more than
   once in the same render pass — duplicate keys crash Streamlit.

### Adding a tab to an existing page

```python
tab_a, tab_b, tab_c = st.tabs(["🔤 Tab A", "🔤 Tab B", "🔤 Tab C"])
with tab_a:
    ...
```

Tabs are just Python variables — no registration needed.

### Adding a new use case

1. Create `use_case_X/01…05_*.py` (copy Use Case A as a template, patch
   `ensure_utf8()` before `logging.basicConfig()`).
2. Add the key to `USE_CASE_SCRIPTS` and `USE_CASE_META` at the top of app.py.
3. Add per-use-case config dicts: `_PROFILING_SRC`, `_FE_EDA_SRC`,
   `_FE_GUIDANCE`, `_DATASET_ANALYSIS_CONFIG`.
4. Scaffolded use cases (no scripts yet) show a warning on Run Pipeline —
   that's intentional.

**Status field rule** — always provide `"status": "scaffolded"` when scaffolding
a new use case (i.e. when the pipeline scripts don't exist yet or haven't been run
end-to-end). Only set `"status": "complete"` after all five scripts run successfully.
This distinction controls whether the Run Pipeline page shows a scaffold warning.

```python
# ✗ Wrong — don't set complete just because you're writing the dict entry
"status": "complete",   # scripts don't exist yet!

# ✓ Right — use scaffolded until the scripts actually run
"status": "scaffolded",
```

### Charts — house style

```python
# Always use plotly; never matplotlib in dashboard functions
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots   # already imported at top

BG, FONT = "#1A1A2E", "#E0E0E0"

fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT, height=340)
# Use width='stretch' (NOT use_container_width=True — deprecated after 2025-12-31)
st.plotly_chart(fig, width='stretch')
st.dataframe(df, width='stretch', hide_index=True)
st.image(path, width='stretch')
```

### Side-by-side comparison charts

```python
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["<b>RAW</b> — col", "<b>PROCESSED</b> — col"],
                    horizontal_spacing=0.06)
fig.add_trace(go.Histogram(x=raw_vals, ...), row=1, col=1)
fig.add_trace(go.Histogram(x=fe_vals,  ...), row=1, col=2)
```

### Session-state navigation

```python
st.session_state.nav_page = "🤖 Model Development"  # must match PAGES key exactly
st.rerun()
```

### Auto-run a pipeline step from any page

```python
st.session_state["_auto_run_step"] = 4   # page_run_pipeline pops this on load
st.session_state.nav_page = "▶️  Run Pipeline"
st.rerun()
```

## Known Pitfalls (fix before asking why it's broken)

See `references/pitfalls.md` for full detail. Quick list:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `StreamlitDuplicateElementKey` | Two `_run_step_action(N, uc_key)` calls on same page | Add `suffix="unique"` to second call |
| `NameError: name 'plt' is not defined` | Matplotlib not imported in dashboard | Replace with plotly (`go.Histogram`, `go.Bar`) |
| Garbled `â€¦` / `â€"` in pipeline output | Windows cp1252 encoding on subprocess pipe | `ensure_utf8()` is called before `logging.basicConfig()` in every script; `Popen` needs both `encoding="utf-8"` AND `env={"PYTHONIOENCODING":"utf-8",...}` |
| `PerformanceWarning: DataFrame is highly fragmented` | Many individual `df["col"] = …` assignments | Collect in `_new: dict`, then `pd.concat([df, pd.DataFrame(_new)], axis=1)` |
| `Cannot perform reduction 'mean' with string dtype` | Column is numeric in FE parquet but string in raw parquet | Filter `shared_num` against `df_raw.select_dtypes(include="number")` too |
| `use_container_width` deprecation warning | Old Streamlit kwarg | Replace every occurrence with `width='stretch'` |
| `labels=` in `ax.boxplot()` warning | Matplotlib 3.9+ renamed it | Change to `tick_labels=` |
| Pandas4Warning on `select_dtypes(include=["object"])` | pandas 3 separates `str` from `object` | Use `include=["object","category","str"]` |

## Helper Functions (app.py)

| Function | Purpose |
|----------|---------|
| `section_header(title, subtitle)` | Renders the blue title bar at the top of every page |
| `load_parquet(path)` | Cached parquet loader with error handling |
| `load_csv(path)` | Cached CSV loader |
| `_run_step_action(step, uc_key, label, suffix="")` | Renders a primary button that navigates to Run Pipeline and auto-triggers the step |
| `_render_correlation_matrix(uc_key)` | Interactive Pearson/Spearman heatmap + high-corr pairs table |
| `render_sidebar()` | Builds nav sidebar; returns `uc_key` string |
| `metric_card(label, value, delta, colour)` | Coloured metric tile |
| `fmt_pct(v)` / `fmt_num(v, d)` | Formatting helpers |

## Reference Files

Read these when you need more detail:

- `references/architecture.md` — full file tree, pipeline script boilerplate,
  per-use-case config dict schemas, `_PROFILING_SRC` / `_FE_EDA_SRC` keys
- `references/pitfalls.md` — extended root-cause analysis for every known bug

## Workflow for Making Changes

1. **Read the relevant section of app.py first** (use `sed -n 'N,Mp'` or
   `grep -n` to find line numbers — never guess).
2. **Make changes via Python script** that reads the file, does a string
   replacement, checks `ast.parse()`, and writes back. Never use `sed -i` on
   a file with emoji — it corrupts bytes.
3. **Verify** with `ast.parse()` + `grep -n` to confirm anchors are in place.
4. For pipeline scripts: patch `ensure_utf8()` before `logging.basicConfig()`
   and use `encoding="utf-8"` on every `subprocess.Popen()` call.
