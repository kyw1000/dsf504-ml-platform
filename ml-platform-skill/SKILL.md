---
name: ml-platform
description: >
  Expert assistant for the DSF504 ML Platform -- a Streamlit multi-use-case
  machine learning dashboard with six-step pipeline scripts.
  Use this skill whenever the user asks about, works on, or debugs anything
  in the DSF504 project, including: the Streamlit dashboard (app.py), pipeline
  scripts (01-06 per use case), config.py, utils/, viz_library.py,
  use_case_G1_robo/, use_case_G2_xai/, or any page/tab/chart inside the
  platform. Also trigger for requests like "add a new use case", "add a page /
  tab", "fix this dashboard error", "improve the correlation chart", or any
  question about the ML Framework phases used in this project. Trigger even
  when the user says something casual like "the dashboard is broken" or "can
  you add something to the model page" -- if it is about DSF504, always invoke
  this skill.
---

# DSF504 ML Platform -- Expert Skill

## Project at a Glance

| Item | Value |
|------|-------|
| Root | C:\\DSF504\\ (bash: /sessions/.../mnt/DSF504/) |
| Dashboard | dashboard/app.py (~4,241 lines) |
| Viz library | dashboard/viz_library.py (Plotly chart helpers for C_markets) |
| Run | streamlit run dashboard/app.py (from C:\\DSF504\\) |
| Python | 3.13, Windows |
| Complete | A (fraud), B (credit), C_nlp (NLP sentiment), C_markets (volatility), D (churn), E (insurance), F (ESG greenwashing), G (AmEx default) |
| Active | G1 (robo-advisory / LambdaRank), G2 (XAI for analysts) |

## ML Framework -- 8 Dashboard Pages

Top nav uses st.radio(horizontal=True) styled as pill tabs. PAGES dict and
_NAV_SECTIONS must use byte-identical label strings -- any mismatch causes a
nav button that never activates. _NAV_SECTIONS is now just list(PAGES.keys()).

```python
PAGES: dict = {
    "\u25b6\ufe0f  Run Pipeline":           page_run_pipeline,
    "\U0001f52c Data Studio":             page_data_profiling,
    "\U0001f527 Data Preparation":        page_feature_engineering,
    "\U0001f4c8 Post-Processing EDA":     page_post_processing_eda,
    "\U0001f916 Model Development":       page_model_development,
    "\U0001f4ca Model Evaluation":        page_model_performance,
    "\U0001f3af Prediction Demo":         page_prediction_demo,
    "\U0001f50d Ethics & Explainability": page_explainability,
}
_NAV_SECTIONS: list[str] = list(PAGES.keys())
```

Note: "Run Pipeline" has two spaces after the play emoji -- copy-paste exactly;
any whitespace difference breaks active-page highlighting.
Debug mismatches with .encode('utf-8').hex().

## Six-Step Pipeline (all use cases)

All use cases have all six pipeline steps registered in USE_CASE_SCRIPTS.

```
Step 1  01_data_loading.py          -> Parquet splits (train/val/test)
Step 2  02_eda_analysis.py          -> CSV/PNG reports in reports/<uc>/
Step 3  03_feature_engineering.py   -> train_fe.parquet, feature list CSV
Step 4  04_model_training.py        -> champion.pkl, model_comparison.csv
Step 5  05_hyperparameter_tuning.py -> lgbm_optuna_champion.pkl (or champion.pkl)
Step 6  06_ethics_explainability.py -> SHAP PNGs, fairness CSVs, ethics_insights.txt
```

STEP_NAMES (registered in app.py):
```python
{
    1: "Data Loading",
    2: "EDA & Data Understanding",
    3: "Data Preparation",
    4: "Algorithm Selection + Cross-Validation",
    5: "Hyperparameter Tuning + Final Training",
    6: "Ethics & Explainability",
}
```

### UC-F multiclass SHAP note
UC-F (3-class ESG) LightGBM SHAP returns a 3-D ndarray (n_samples, n_features,
n_classes); slice to a list of 2-D arrays before plotting:
```python
[sv[:, :, k] for k in range(sv.shape[2])]
```

### C_markets naming trap
The USE_CASE_SCRIPTS key is "C_markets" but the script folder is
use_case_C_market (no trailing 's'). The model_dir in USE_CASE_META is
"use_case_C_markets" (with 's'). Keep these separate:

```python
# USE_CASE_SCRIPTS -- folder WITHOUT trailing s
"C_markets": { 1: "use_case_C_market/01_data_loading.py", ... }
# USE_CASE_META -- model_dir WITH trailing s
"C_markets": { "model_dir": "use_case_C_markets", ... }
```

UC-C_markets also has fast alternatives:
use_case_C_market/_run_step4_fast.py and _run_step5_fast.py.

### G1 / G2 -- new active use cases

| Key | Title | Folder | Task | Metric | Data |
|-----|-------|--------|------|--------|------|
| G1 | Robo-Advisory Portfolio Recommendation | use_case_G1_robo/ | Learning to Rank | NDCG@10 | data/far-trans/ |
| G2 | Explainable AI for Analysts & Managers | use_case_G2_xai/ | Binary Classification | AUC-ROC | data/sec_edgar/ |

Both have all 6 pipeline scripts registered (status: "active"). Run Steps 1-6
and verify artefacts before promoting to "complete" in USE_CASE_META.

## CRITICAL: File Tool Size Limit and Safe Writing

The Edit and Write tools have a hard buffer cap of ~12 KB. Files larger than
this are silently truncated with no error or warning. The tool also sometimes
pads the tail with null bytes (0x00) when shrinking a file, which causes
Python AST parse failures. This has caused serious damage to app.py and
pipeline scripts in previous sessions.

### Before touching any file: check its size
```bash
wc -c /sessions/.../mnt/DSF504/dashboard/app.py
# If output > 10000 bytes, use bash Python -- not the Edit/Write tools.
```

### Safe write pattern (bypasses the buffer limit)
```python
content = open('/sessions/.../mnt/DSF504/target.py', encoding='utf-8').read()
# ... make your changes to content ...
with open('/sessions/.../mnt/DSF504/target.py', 'w', encoding='utf-8') as f:
    f.write(content)
import ast; ast.parse(content); print("AST OK,", len(content.splitlines()), "lines")
```

### Strip null bytes after any Edit on a large file
```python
path = '/sessions/.../mnt/DSF504/dashboard/app.py'
raw = open(path, 'rb').read()
if b'\x00' in raw:
    open(path, 'wb').write(raw.rstrip(b'\x00'))
    print("Stripped null bytes")
import ast; ast.parse(open(path, encoding='utf-8').read()); print("AST OK")
```

### Use utils/file_guard.py
```bash
python utils/file_guard.py dashboard/app.py      # check + strip nulls
```

Never chain multiple Edit calls on the same large file in one turn -- each
Edit re-reads the truncated state and compounds the damage. Write the complete
final content in one bash Python call instead.

## Key Patterns -- Quick Reference

### Adding a new page
1. Write def page_my_page(uc_key: str) -> None: anywhere in app.py.
2. Add to PAGES dict; _NAV_SECTIONS = list(PAGES.keys()) updates automatically.
3. Use suffix="unique_id" on _run_step_action() if the same (step, uc_key)
   pair appears more than once -- duplicate widget keys crash Streamlit.

### Navigation tab CSS -- the cascade trap
color on label does NOT cascade to the inner p in Streamlit radio buttons.
Always target both elements in CSS:

```css
div[data-testid="stRadio"] div[role="radiogroup"] > label { color: #1A237E !important; }
div[data-testid="stRadio"] div[role="radiogroup"] > label p { color: #1A237E !important; }
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) { background: #3949AB !important; color: #FFFFFF !important; }
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p { color: #FFFFFF !important; }
```

### Dark background -- always use light text

The dashboard uses a dark background (BG = "#1A1A2E"). Any inline HTML,
custom CSS, or st.markdown content must use light colours. Always use:
    st.markdown(f"<p style='color:{FONT};'>...</p>", unsafe_allow_html=True)
Not: st.markdown("<p>...</p>", unsafe_allow_html=True)  # invisible black text

When writing CSS, always target both the container AND inner p tags.

House colour palette:

| Var    | Hex       | Use                          |
|--------|-----------|------------------------------|
| BG     | #1A1A2E   | Main background (dark navy)  |
| FONT   | #E0E0E0   | Primary text (light grey)    |
| ACCENT | #3949AB   | Indigo accent / borders      |
| BLUE   | #42A5F5   | Info / links                 |
| GRN    | #66BB6A   | Success / positive           |
| ORG    | #FFA726   | Warning / neutral            |
| RED    | #EF5350   | Danger / negative            |
| PURP   | #AB47BC   | Purple highlights            |
| GRID   | #2A2A4A   | Subtle grid lines            |

### Charts & widgets -- house style
```python
fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT, height=340)
st.plotly_chart(fig, width='stretch')   # use_container_width removed in Streamlit >=1.44
st.dataframe(df, width='stretch', hide_index=True)
st.image(img_path, width='stretch')
```

width='stretch' replaces use_container_width=True.
width='content' replaces use_container_width=False.
Applies to all Streamlit widgets that previously accepted use_container_width.

### viz_library.py -- C_markets chart helpers

dashboard/viz_library.py provides pre-built dark-themed Plotly charts used
in the Market Intelligence tab (_render_market_analytics_tab()). All
functions return a go.Figure ready for st.plotly_chart(fig, width='stretch').

Chart inventory: kpi_cards, candlestick_chart, volatility_timeseries,
rv_heatmap, scatter_bubble, seasonal_subseries, forecast_ribbon,
waterfall_chart, indexed_chart, orderflow_chart, actual_vs_predicted.

### eda_viz.py -- shared EDA visualization helpers

utils/eda_viz.py provides matplotlib-based helpers used by pipeline scripts
(not the dashboard). Each function saves a PNG and returns a one-sentence
insight string. Import: from utils.eda_viz import plot_target_distribution, ...

Functions: plot_target_distribution, plot_overview_panel, plot_missing_heatmap,
plot_correlation_heatmap, plot_engineered_feature_summary, plot_raw_vs_processed,
plot_numeric_distributions, plot_class_balance_bar.

### Data Preparation page -- dynamic last tab

page_feature_engineering computes _fe_done (True when train_fe.parquet exists)
and uses it to switch the last tab label and content:
- _fe_done = False  ->  tab label "Run Step 3", button runs Step 3
- _fe_done = True   ->  tab label "Run Step 4", primary CTA runs Step 4,
  with a collapsible expander to re-run Step 3 if needed

### Post-Processing EDA -- colour constants scope

_RAW_CLR and _FE_CLR are defined at the top of page_post_processing_eda
(before the with tab_compare: block) so that tab_new_feats can always
reference them even when df_raw is None. Do NOT move these back inside the
if shared_num: block or the UnboundLocalError will return.

### Adding a new use case
1. Create use_case_X/01...06_*.py (copy UC-G as the standard template;
   use UC-G1/use_case_G1_robo for ranking tasks, UC-G2/use_case_G2_xai
   for XAI-focused workflows).
2. Add key to USE_CASE_SCRIPTS (steps 1-6) and USE_CASE_META in app.py.
3. Add dicts: _PROFILING_SRC, _FE_EDA_SRC, _FE_GUIDANCE (with stage_notes
   per stage), _EDA_INSIGHTS (4 keys: target/correlation/missing/outlier),
   and _DATASET_INFO (url, label, intro).
4. _PROFILING_SRC["X"]["raw"] must point to the actual parquet filename on
   disk (verify with ls data/<dir>/). Wrong path -> Data Studio shows no raw
   samples.
5. Use "status": "active" for new use cases. Promote to "status": "complete"
   only after Steps 1-6 all produce their artefacts.

### USE_CASE_META -- champion field

Each entry includes a "champion" key with the saved model filename:
```python
"A": { ..., "champion": "lgbm_optuna_champion.pkl" }
"C_markets": { ..., "champion": "champion.pkl" }
```
Use USE_CASE_META[uc_key].get("champion", "lgbm_optuna_champion.pkl") when
constructing the model path in page_model_performance and page_prediction_demo.

### UC-G raw data paths
UC-G (AmEx Default) raw file is data/amex_default/train_raw.parquet --
NOT train_data_synthetic.parquet. Both _PROFILING_SRC["G"]["raw"] and
_FE_EDA_SRC["G"]["raw"] must use train_raw.parquet.

### n_jobs in sandbox pipeline scripts
The bash sandbox hangs with n_jobs=-1. All pipeline scripts must use n_jobs=1.
Check with: grep -rn "n_jobs=-1" use_case_*/

### Git -- must run from Windows, not bash sandbox
The bash sandbox mounts the Windows filesystem read-only for git operations.
Run all git commands from Windows PowerShell (cd C:\\DSF504; git add -A; git push).

## Known Pitfalls -- Quick Lookup

See references/pitfalls.md for full root-cause analysis.

| Symptom | Cause | Fix |
|---------|-------|-----|
| File silently truncated after Edit/Write | Tool buffer cap ~12 KB | Use bash Python for files > 10 KB |
| Null bytes / SyntaxError after Edit | Tool pads with 0x00 when shrinking | raw.rstrip(b'\x00') then rewrite |
| StreamlitDuplicateElementKey | Two _run_step_action(N, uc_key) on same page | Add suffix="unique" to second call |
| NameError: plt is not defined | matplotlib not imported in dashboard | Replace with plotly go.Histogram etc |
| Garbled output on Windows | cp1252 vs UTF-8 pipe encoding | ensure_utf8() before logging.basicConfig() |
| TypeError: multi_class unexpected kwarg | Removed in scikit-learn 1.5 | Delete multi_class= from LogisticRegression |
| Dataset scripts no longer supported | HuggingFace datasets v3 change | Use parquet fallbacks + raw text mirrors |
| Nav tab text invisible (dark on dark) | CSS color on label does not cascade to p | Also target > label p { color: !important } |
| git init fails from bash sandbox | Mounted Windows filesystem | Run git from Windows PowerShell |
| PerformanceWarning: DataFrame fragmented | Many df["col"] = assignments | Collect in dict, pd.concat once |
| use_container_width deprecation warning | Removed in Streamlit >=1.44 | width='stretch' or width='content' |
| Pandas4Warning on select_dtypes("object") | pandas 3 separates str from object | Use include=["object","category","str"] |
| PAGES / _NAV_SECTIONS key mismatch | Label strings differ | Copy-paste labels; debug with .encode('utf-8').hex() |
| LightGBM multiclass SHAP is 3-D ndarray | shap_values() returns (n,p,k) not a list | Slice: [sv[:,:,k] for k in range(sv.shape[2])] |
| model.feature_name_() TypeError | LightGBM feature_name_ is a property | Remove () -- use model.feature_name_ |
| Beta generator yields no High-risk class | beta(1.5,4.0) rarely exceeds threshold | Use explicit 3-tier (60/25/15% Low/Med/High) |
| UnboundLocalError: _FE_CLR not associated | _FE_CLR defined inside if shared_num block | Define _RAW_CLR/_FE_CLR before with tab_compare: |
| UC-G Data Studio shows no raw samples | _PROFILING_SRC["G"]["raw"] -> wrong file | Change to train_raw.parquet |
| UC-G step 04 hangs / only 1 model | n_jobs=-1 in all 4 model definitions | Change all to n_jobs=1 in 04_model_training.py |
| UC-G ROC-AUC = 1.000 | Synthetic data -- target derived from features | Expected; not a leakage bug |
| C_markets scripts not found | Key vs folder name mismatch | Key "C_markets" -> folder use_case_C_market (no 's'); model_dir use_case_C_markets (with 's') |
| sed corrupts emoji in app.py | sed treats file as byte stream | Always use Python str.replace() for in-place edits |
| Auto-run step silently does nothing | Step not in available_steps for that UC | Expected for use cases missing that script |

## Helper Functions (app.py)

| Function | Purpose |
|----------|---------|
| section_header(title, subtitle) | Blue title bar at top of every page |
| load_parquet(path, nrows=0) | Cached parquet loader with error handling |
| load_csv(path) | Cached CSV loader |
| load_model(path) | Cached pickle loader with error handling |
| _dark_fig(h=380) | Returns a pre-styled dark go.Figure (BG/FONT applied) |
| _run_step_action(step, uc_key, label, suffix="") | Button that navigates to Run Pipeline and auto-triggers step |
| _prereq_warning(page, uc_key) | Returns warning string if required artefacts are missing, else None |
| _render_step_badges(steps, status) | Renders coloured step-done/run/wait badges |
| _render_correlation_matrix(uc_key) | Interactive Pearson/Spearman heatmap |
| _render_market_analytics_tab() | Full C_markets analytics tab using viz_library.py |
| _HP_GUIDE_DATA() | Returns the hyperparameter guide reference dict |
| render_sidebar() | Builds nav sidebar; returns uc_key string |
| metric_card(label, value, delta, colour) | Coloured metric tile HTML |
| fmt_pct(v) / fmt_num(v, d) | Formatting helpers |

## Workflow -- Approach for Every Task

1. Read before writing. Check file size. Over 10 KB means bash Python only.
2. Targeted edits on large files: use Edit tool for small changes, then
   immediately strip null bytes and verify AST.
3. Verify after every write: ast.parse or python utils/file_guard.py.
4. Commit after stable milestones from Windows PowerShell.

## Reference Files

- references/architecture.md -- full file tree, pipeline boilerplate, config dict schemas
- references/pitfalls.md -- extended root-cause analysis for every known bug
