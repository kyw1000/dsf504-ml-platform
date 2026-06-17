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
| Dashboard | dashboard/app.py (~5,370 lines, ~304 KB) |
| Viz library | dashboard/viz_library.py (Plotly chart helpers for C_markets) |
| Run | streamlit run dashboard/app.py (from C:\\DSF504\\) |
| Python | 3.13, Windows |
| Complete | A (fraud), B (credit), C_nlp (NLP sentiment), C_markets (volatility), D (churn), E (insurance), F (ESG greenwashing), B3 (AmEx default), G1 (robo-advisory / LambdaRank), G2 (XAI for analysts) |

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
| G1 | Robo-Advisory Portfolio Recommendation | use_case_G1_robo/ | Learning to Rank | NDCG@10 | data/far_trans/ |
| G2 | Explainable AI for Analysts & Managers | use_case_G2_xai/ | Binary Classification | AUC-ROC | data/sec_edgar/ |

Both have all 6 pipeline scripts registered and all Steps 1-6 verified.
Status in USE_CASE_META is "complete".

#### G1 feature builder -- is_train detection pattern
G1's 03_feature_engineering.py calls three builder functions. Each must detect
whether it is in training mode (compute + save statistics) or inference mode
(apply saved statistics). Use **key-based** detection, not None-based:

```python
# WRONG -- breaks once main() passes accumulated train_stats to later builders
is_train = train_stats is None

# CORRECT -- check for the specific key this function populates
if train_stats is None:
    train_stats = {}
is_train = "pref_cat_map" not in train_stats   # use function-specific sentinel
```

Also, main() must pass the accumulated dict into every builder call, or the
interaction builder returns an empty {} and overwrites earlier keys:
```python
# WRONG -- discards accumulated user/item keys
train_inter, train_stats = build_interaction_features(train_pairs, train_tx, customers, assets)

# CORRECT
train_inter, train_stats = build_interaction_features(train_pairs, train_tx, customers, assets, train_stats)
```

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

### Patch scripts -- line-range replacement when string matching fails

app.py stores some characters as Python escape sequences in the source file
(e.g. the Delta symbol stored as \\u0394, the arrow as \\u2192). When a patch
script does content.replace(OLD, NEW), the actual bytes in the file won't match
the literal characters in the script. Use line-range replacement instead:

```python
lines = content.splitlines(keepends=True)
START, END = 3854, 3965   # 0-based, exclusive END
new_block = "... replacement lines ...\n"
lines[START:END] = [new_block]
content = "".join(lines)
```

Find the correct START/END by grepping for a stable nearby string:
```bash
grep -n "some_stable_nearby_string" /sessions/.../mnt/DSF504/dashboard/app.py
```

Also avoid writing patch scripts via bash heredoc when the replacement block
contains Unicode characters like the arrow (U+2192). The heredoc truncates at
multi-byte sequences. Write patch scripts with the Python Write tool to the
outputs directory, then execute with bash.

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

### Data Preparation -- Feature List tab routing

The Feature List tab (tab_feats) uses _FE_EDA_SRC[uc_key] to route:

```
feat_list (CSV path)  -> load CSV -> if cols include "table"+"column" (G1):
                             show schema table + SHAP CSV as engineered features
                         else (standard):
                             _enrich_feat_df() + _feat_search_widget()
train_fe (parquet)    -> load 5 rows -> filter out target/ID cols
                         show all remaining columns with descriptions
neither               -> C_nlp: explain TF-IDF pipeline (no tabular features)
                         others: "No feature list configured"
```

G1 special case: data_dictionary.csv has columns
[table, column, dtype, nunique, null_pct] -- the first column is NOT a
feature name. Detect with "table" in df.columns and "column" in df.columns
and render as a schema browser. Then look for
reports/use_case_G1/shap_feature_importance.csv to show the engineered
features with importance scores.

UC-B: _FE_EDA_SRC["B"]["feat_list"] points to a CSV whose first column
is "feature". Some use cases write ALL columns including non-fe_ ones. The
tab shows all of them -- this is correct. Don't filter to fe_ prefix only.

_enrich_feat_df(df_in): Detects the feature name column (tries "feature"
then "column"), calls _describe_feature(feat, uc_key) for each row, and
inserts a "description" column immediately after the feature name column.

_feat_search_widget(df_in, total): Renders a text_input search box
(searches both feature name and description), then an st.dataframe with the
filtered results and a row count caption.

### Feature Glossary -- _FEATURE_GLOSSARY and _describe_feature

A module-level dict _FEATURE_GLOSSARY maps feature name patterns to
human-readable descriptions. _describe_feature(feat, uc_key) checks:
1. Exact match in _FEATURE_GLOSSARY
2. Prefix/suffix patterns (fe_, log_, missing_, ratio_, etc.)
3. Use-case specific overrides for well-known column names
4. Falls back to the raw feature name if no match found

Add new patterns to _FEATURE_GLOSSARY at the module level (not inside a
function) so they persist across all tab renders.

### Post-Processing EDA -- colour constants scope

_RAW_CLR and _FE_CLR are defined at the top of page_post_processing_eda
(before the with tab_compare: block) so that tab_new_feats can always
reference them even when df_raw is None. Do NOT move these back inside the
if shared_num: block or the UnboundLocalError will return.

### Post-Processing EDA -- Raw vs Processed tab correctness

The Raw vs Processed comparison (tab_compare) has four correctness rules:

1. Per-dataframe null% denominator: _stats(s, n_total) takes n_total
   explicitly -- the total row count of the source dataframe (df_raw or
   df_fe), not a shared variable. Previously both views used len(df_raw),
   producing false null% for the processed view when train splits differ.

```python
def _stats(s, n_total):
    null_count = n_total - len(s)
    return {
        "total rows": n_total,
        "non-null":   len(s),
        "null count": null_count,
        "null%":      round(null_count / max(n_total, 1) * 100, 2),
        # ... mean, std, min, median, max, skew ...
    }
rs = _stats(_rv, _n_raw)   # _n_raw = len(df_raw)
fs = _stats(_fv, _n_fe)    # _n_fe  = len(df_fe)
```

2. ID/timestamp column exclusion -- shared_num now filters out columns whose
   names look like identifiers or timestamps:

```python
_ID_ENDSWITH  = ("ID", "_id", "DT", "_dt", "_ts", "_key")
_ID_STARTSWITH = ("id_", "ID_")
shared_num = [
    c for c in df_fe.select_dtypes(include="number").columns
    if c in _raw_num and c != target
    and not any(c.endswith(s)   for s in _ID_ENDSWITH)
    and not any(c.startswith(s) for s in _ID_STARTSWITH)
]
```

3. Row-count caption: when raw and processed row counts differ, display a
   caption so the user understands the n mismatch (e.g. 590K raw vs 472K
   train for UC-A).

4. Delta logic: Delta is computed only for distribution metrics (null%, mean,
   std, min, median, max, skew). Row-count metrics (total rows, non-null,
   null count) show "---" since comparing absolute counts across
   different-sized splits is misleading.

### Adding a new use case
1. Create use_case_X/01...06_*.py (copy UC-G as the standard template;
   use UC-G1/use_case_G1_robo for ranking tasks, UC-G2/use_case_G2_xai
   for XAI-focused workflows).
2. Add key to ALL EIGHT locations in app.py: USE_CASE_SCRIPTS, USE_CASE_META,
   _DATASET_INFO, _PROFILING_SRC, _FE_EDA_SRC, _EDA_RECOMMENDATIONS,
   _FE_GUIDANCE, _EDA_INSIGHTS. Missing any one causes silent "not found"
   errors on the corresponding dashboard tab.
3. _PROFILING_SRC and _FE_EDA_SRC paths must match what the pipeline scripts
   ACTUALLY WRITE to disk. Run the path-audit script (see below) after every
   registration to catch mismatches before the dashboard launch.
4. For non-LightGBM use cases (e.g. C_nlp uses Naive Bayes), ensure
   USE_CASE_META["X"]["champion"] names the actual saved model file, not the
   default "lgbm_optuna_champion.pkl".
5. Set optional paths (corr_csv, missing_png, outlier_csv) to None rather
   than a non-existent string when the pipeline does not generate them.
6. Use "status": "active" for new use cases. Promote to "status": "complete"
   only after Steps 1-6 all produce their artefacts.

### Path-audit script (run after any registration)
Paste this into a bash Python heredoc to check every registered file path:

```python
import ast, os
APP = "dashboard/app.py"
DSF = "."   # run from project root
NON_PATH_KEYS = {"target", "target_labels", "report_dir"}
src = open(APP, encoding="utf-8").read()
issues = []
for dict_name, start_marker, end_marker in [
    ("_PROFILING_SRC", "\n_PROFILING_SRC", "\n_FE_EDA_SRC"),
    ("_FE_EDA_SRC",    "\n_FE_EDA_SRC",    "\n\n\n"),
]:
    idx = src.find(start_marker)
    block = src[idx:src.find(end_marker, idx)]
    data = ast.literal_eval(ast.parse(block.strip()).body[0].value)
    for uc, d in data.items():
        for key, val in d.items():
            if key in NON_PATH_KEYS or not val or not isinstance(val, str):
                continue
            if not os.path.exists(os.path.join(DSF, val)):
                issues.append(f"MISSING [{dict_name}.{uc}.{key}] {val}")
for i in issues or ["All paths OK"]:
    print(i)
```

### USE_CASE_META -- champion field

Each entry includes a "champion" key with the saved model filename:
```python
"A": { ..., "champion": "lgbm_optuna_champion.pkl" }
"C_markets": { ..., "champion": "champion.pkl" }
```
Use USE_CASE_META[uc_key].get("champion", "lgbm_optuna_champion.pkl") when
constructing the model path in page_model_performance and page_prediction_demo.

### UC-B3 raw data paths
UC-B3 (AmEx Default) raw file is data/amex_default/train_raw.parquet --
NOT train_data_synthetic.parquet. Both _PROFILING_SRC["B3"]["raw"] and
_FE_EDA_SRC["B3"]["raw"] must use train_raw.parquet.

### n_jobs in sandbox pipeline scripts
The bash sandbox hangs with n_jobs=-1 for scikit-learn estimators (RandomForest,
LogisticRegression, etc.). LightGBM tolerates n_jobs=-1 (manages its own thread
pool independently). All sklearn estimators must use n_jobs=1.
Check with: grep -rn "n_jobs=-1" use_case_*/

### Git -- must run from Windows, not bash sandbox
The bash sandbox mounts the Windows filesystem read-only for git operations.
Run all git commands from Windows PowerShell (cd C:\\DSF504; git add -A; git push).

## Known Pitfalls -- Quick Lookup

See references/pitfalls.md for full root-cause analysis.

| Symptom | Cause | Fix |
|---------|-------|-----|
| File silently truncated after Edit/Write | Tool buffer cap ~12 KB | Use bash Python for files > 10 KB |
| Null bytes / SyntaxError after Edit | Tool pads with 0x00 when shrinking | raw.rstrip(b'\\x00') then rewrite |
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
| UC-B3 Data Studio shows no raw samples | _PROFILING_SRC["B3"]["raw"] -> wrong file | Change to train_raw.parquet |
| UC-B3 step 04 hangs / only 1 model | n_jobs=-1 in sklearn model definitions | Use n_jobs=1 for sklearn; LGB may use n_jobs=-1 |
| UC-B3 ROC-AUC = 1.000 | Synthetic data -- target derived from features | Expected; not a leakage bug |
| C_markets scripts not found | Key vs folder name mismatch | Key "C_markets" -> folder use_case_C_market (no 's'); model_dir use_case_C_markets (with 's') |
| sed corrupts emoji in app.py | sed treats file as byte stream | Always use Python str.replace() for in-place edits |
| Auto-run step silently does nothing | Step not in available_steps for that UC | Expected for use cases missing that script |
| is_train KeyError in G1/G2 Step 3 | Feature builder uses is_train = train_stats is None; once main() passes a non-None dict, subsequent builders skip map creation entirely | Key-based detection: if train_stats is None: train_stats = {}; is_train = "pref_cat_map" not in train_stats |
| interaction builder overwrites train_stats | main() calls build_interaction_features() without passing accumulated train_stats; returns empty {} and wipes prior keys | Always pass the dict: inter, train_stats = build_interaction_features(..., train_stats) |
| Pipeline script silent RC=0 / no stdout | ensure_utf8() replaces sys.stdout; subprocess sees new TextIOWrapper but bash captures nothing | Use exec(compile(open(fname).read(), fname, 'exec'), {"__file__": fname}) in a Python heredoc to share stdout |
| File truncated mid-line (SyntaxError) | Multi-byte Unicode in bash heredoc body causes silent truncation at that byte | Write patch scripts via Python Write tool to outputs dir; use ASCII -> not arrow char |
| ImportError: libscipy_openblas-...so | System scipy has a broken BLAS shared library | pip install --no-cache-dir --target=/tmp/pylibs scikit-learn scipy pyarrow then export PYTHONPATH=/tmp/pylibs |
| ImportError: no usable engine (parquet) | pyarrow absent in sandbox | pip install --no-cache-dir --target=/tmp/pylibs pyarrow |
| Optuna / SHAP not available in sandbox | Neither installed by default | G1 and G2 pipelines have grid-search and native-importance fallbacks; all 6 steps complete without them |
| UC-C_nlp champion points to lgbm_optuna_champion.pkl | C_nlp uses TF-IDF + Naive Bayes, not LightGBM; that file never exists | Set USE_CASE_META["C_nlp"]["champion"] to Complement_NB_baseline.pkl |
| _PROFILING_SRC or _FE_EDA_SRC missing a UC key | UC was added to USE_CASE_SCRIPTS/META but the two data-source dicts were not updated | Any tab that calls _PROFILING_SRC.get(uc_key, {}) returns {} and shows "Raw data not found"; always add F-style entries to BOTH dicts when registering a new use case |
| _DATASET_INFO missing a UC key | UC added late without touching _DATASET_INFO | Dataset card on the sidebar shows blank; add url/label/intro entry alongside USE_CASE_META |
| Registered path does not exist on disk | Path copied from a template for files the pipeline never generated | Run the path-audit script after every registration; set missing optional paths to None rather than a non-existent string |
| ValueError: invalid literal for int() in Post-Processing EDA | Target-split lambda does int(x) unconditionally; breaks for string-keyed targets like UC-F ('Low'/'Medium'/'High') | Fix: target_labels.get(x, target_labels.get(int(x) if str(x).lstrip('-').isdigit() else x, str(x))) -- tries x as-is first, then int(x) only when numeric |
| G1 Feature List shows table names instead of features | data_dictionary.csv first column is "table", not feature names | Detect with "table" in df.columns and "column" in df.columns; show as schema browser, then load shap_feature_importance.csv for engineered features |
| Feature List tab shows nothing for C_nlp | C_nlp has no tabular features (TF-IDF tokens); feat_list=None and train_fe=None | Show explanatory st.info() describing the TF-IDF + Complement NB pipeline instead |
| null% wrong in Raw vs Processed | Both raw and processed used len(df_raw) as denominator | Pass n_total per-df: _stats(_rv, _n_raw) and _stats(_fv, _n_fe) |
| ID/timestamp columns appear in Raw vs Processed comparison | shared_num filter only excluded target, not identifier columns | Add endswith/startswith filters: _ID_ENDSWITH=("ID","_id","DT","_dt","_ts","_key") |
| content.replace(OLD, NEW) fails silently in patch scripts | app.py stores escape sequences as literals (\\u0394, \\u2192); bytes do not match the actual Unicode chars in the script | Use line-range replacement: lines[START:END] = [new_block]; find START/END with grep -n |
| Patch script heredoc truncated | Multi-byte Unicode in heredoc body causes bash to cut at that byte | Write patch scripts via Python Write tool to outputs dir, then run with bash |

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
| _describe_feature(feat, uc_key) | Returns human-readable description for a feature name using _FEATURE_GLOSSARY patterns |
| _enrich_feat_df(df_in) | Detects feature name column, adds "description" column via _describe_feature |
| _feat_search_widget(df_in, total) | Search box + filtered dataframe for Feature List tab |

## Workflow -- Approach for Every Task

1. Read before writing. Check file size. Over 10 KB means bash Python only.
2. Targeted edits on large files: use Edit tool for small changes, then
   immediately strip null bytes and verify AST.
3. Verify after every write: ast.parse or python utils/file_guard.py.
4. Commit after stable milestones from Windows PowerShell.
5. When string matching fails in patch scripts (Unicode escapes, quote style
   mismatches), switch immediately to line-range replacement.
6. Write patch scripts via Python Write tool (not bash heredoc) when the
   replacement block contains non-ASCII characters.

## Reference Files

- references/architecture.md -- full file tree, pipeline boilerplate, config dict schemas
- references/pitfalls.md -- extended root-cause analysis for every known bug

---

## UC-B Credit Risk -- Pipeline Improvement Patterns

These patterns were implemented in UC-B (Give Me Some Credit dataset) to fix
a PR-AUC regression and improve overall model quality. Apply them whenever
working on Steps 03–05 of UC-B or any highly imbalanced binary classifier.

### Dataset characteristics
- 150,000 borrowers, 6.7% default rate (93.3 / 6.7 class split)
- 10 raw features; 19% missing MonthlyIncome, 2.5% missing NumDependents
- 96/98 error codes in DPD (Days Past Due) columns mean "unknown", not zero
- After engineering: ~25 features including fe_ prefixed engineered flags

### Metric choice: PR-AUC not ROC-AUC
At 6.7% base rate, ROC-AUC misleads -- all models cluster at 0.84–0.87 with
only 0.021 probability gap between classes. Use PR-AUC (average_precision_score)
as the primary metric and Optuna objective. Typical UC-B values: 0.38–0.43.

FNR (False Negative Rate) at default threshold ~0.58 is ~45% -- nearly 1 in 2
defaults approved. Use cost-optimal threshold with FN:FP = 10:1 ratio.

### Step 3 -- Feature Engineering changes

#### KNN imputation for MonthlyIncome
```python
from sklearn.impute import KNNImputer
from typing import Optional, Tuple

def add_missing_indicators(
    df: pd.DataFrame,
    knn_imputer: Optional[KNNImputer] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, Optional[KNNImputer]]:
    """Flag missingness then impute. Returns (df, fitted_knn_imputer)."""
    df = df.copy()
    indicators = {}
    if "MonthlyIncome" in df.columns:
        indicators["fe_miss_income"] = df["MonthlyIncome"].isna().astype(np.int8)
        knn_features = [c for c in ["DebtRatio","NumOpenLoans","NumRealEstate","age","NumDependents"]
                        if c in df.columns]
        knn_input = df[["MonthlyIncome"] + knn_features].copy()
        if fit or knn_imputer is None:
            knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
            imputed = knn_imputer.fit_transform(knn_input)
        else:
            imputed = knn_imputer.transform(knn_input)
        df["MonthlyIncome"] = imputed[:, 0].astype(np.float32)
    if "NumDependents" in df.columns:
        indicators["fe_miss_dependents"] = df["NumDependents"].isna().astype(np.int8)
        df["NumDependents"] = df["NumDependents"].fillna(0).astype(np.float32)
    if indicators:
        df = pd.concat([df, pd.DataFrame(indicators, index=df.index)], axis=1)
    return df, knn_imputer
```
Fit only on training data; pass fitted imputer for val/test to prevent leakage.

#### fe_over_limit flag (capture BEFORE winsorisation)
RevolvingUtil > 1.0 means debt exceeds credit limit -- a distinct high-risk tier.
winsorise() clips RevolvingUtil to [0,1], destroying this signal. Capture it first:
```python
def winsorise(df: pd.DataFrame) -> pd.DataFrame:
    if "RevolvingUtil" in df.columns:
        df["fe_over_limit"] = (df["RevolvingUtil"] > 1.0).astype(np.int8)  # BEFORE clip
        df["RevolvingUtil"] = df["RevolvingUtil"].clip(0, 1).astype(np.float32)
    ...
```

#### fe_dpd_unknown flag (96/98 error codes)
96/98 in DPD columns means "unknown history", not "no delinquency". Borrowers with
unknown DPD default at higher rates than confirmed-clean borrowers:
```python
def clean_error_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    any_unknown = pd.Series(False, index=df.index)
    for col in DELINQ:
        if col not in df.columns:
            continue
        unknown_mask = df[col].isin(ERROR_CODES)   # ERROR_CODES = {96, 98}
        any_unknown = any_unknown | unknown_mask
        df[col] = df[col].where(~unknown_mask, other=np.nan)
        df[col] = df[col].clip(upper=10).fillna(0).astype(np.float32)
    df["fe_dpd_unknown"] = any_unknown.astype(np.int8)
    return df
```

#### run_feature_pipeline() signature change
After the KNN imputation refactor, the function returns a tuple:
```python
def run_feature_pipeline(
    df: pd.DataFrame,
    knn_imputer: Optional[KNNImputer] = None,
    fit_imputer: bool = True,
) -> Tuple[pd.DataFrame, Optional[KNNImputer]]:
    df = clean_error_codes(df)
    df = winsorise(df)
    df, knn_imputer = add_missing_indicators(df, knn_imputer=knn_imputer, fit=fit_imputer)
    df = add_log_transforms(df)
    df = add_delinquency_features(df)
    df = add_age_features(df)
    df = add_credit_features(df)
    return df, knn_imputer
```
Callers must unpack: `df_fe, knn_imputer = run_feature_pipeline(df_train, fit_imputer=True)`

### Step 4 -- Model Training changes

XGBoost and LightGBM ImbPipelines switch from SMOTE to ADASYN:
```python
from imblearn.over_sampling import SMOTE, ADASYN

models["XGBoost"] = ImbPipeline([
    ("sampler", ADASYN(random_state=random_state, sampling_strategy=0.20)),
    ("clf", XGBClassifier(...)),
])
models["LightGBM"] = ImbPipeline([
    ("sampler", ADASYN(random_state=random_state, sampling_strategy=0.20)),
    ("clf", LGBMClassifier(...)),
])
# Baselines (LR, DT, RF, MLP) keep SMOTE -- they are for comparison, not deployment
```
ADASYN focuses synthetic generation on borderline defaulters near the decision
boundary rather than uniformly; produces better-calibrated minority-class scores.

### Step 5 -- Hyperparameter Tuning changes

#### 4 root causes of Optuna PR-AUC regression
1. Wrong objective: Optuna used roc_auc_score → switch to average_precision_score
2. Search space too wide: num_leaves up to 255, lr up to 0.2 → complex models overfit
3. No TPE warm-start: first 10 trials are random noise → n_startup_trials=20
4. SMOTE distributional mismatch in CV folds → switch to ADASYN with fallback

#### Tightened Optuna search space
```python
"n_estimators":      trial.suggest_int("n_estimators", 300, 1000, step=100),
"num_leaves":        trial.suggest_int("num_leaves", 31, 127),          # was 31-255
"max_depth":         trial.suggest_int("max_depth", 6, 12),
"learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.05, log=True),  # was 0.01-0.2
"min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
"feature_fraction":  trial.suggest_float("feature_fraction", 0.6, 0.9),  # was 0.4-1.0
"bagging_fraction":  trial.suggest_float("bagging_fraction", 0.6, 0.9),  # was 0.4-1.0
"bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),
"lambda_l1":         trial.suggest_float("lambda_l1", 1e-3, 1.0, log=True),  # floor raised
"lambda_l2":         trial.suggest_float("lambda_l2", 1e-3, 1.0, log=True),
# removed: min_split_gain, max_bin
```

#### PR-AUC CV objective
```python
from sklearn.metrics import average_precision_score
# In CV fold scoring loop -- replaces roc_auc_score:
pr_score = average_precision_score(y_fold_val, y_prob)
fold_scores.append(pr_score)
```

#### ADASYN in CV folds (with SMOTE fallback)
```python
adasyn = ADASYN(sampling_strategy=0.20, random_state=RANDOM_STATE)
try:
    X_fold_tr_res, y_fold_tr_res = adasyn.fit_resample(X_fold_tr, y_fold_tr)
except Exception:
    smote = SMOTE(sampling_strategy=0.20, random_state=RANDOM_STATE)
    X_fold_tr_res, y_fold_tr_res = smote.fit_resample(X_fold_tr, y_fold_tr)
```
ADASYN can fail on very small folds (too few minority samples for k-neighbours);
always wrap with try/except SMOTE fallback.

#### TPE warm-start
```python
sampler = TPESampler(seed=RANDOM_STATE, n_startup_trials=20)  # was no n_startup_trials
```

#### Monotonic constraints (regulatory defensibility)
Required for ECOA adverse-action explanations and SR 26-2 conceptual soundness.
Apply to the final champion LightGBM before retraining:
```python
feature_names = getattr(X_train, 'columns', None)
if feature_names is not None:
    MONO_MAP = {
        'RevolvingUtil': 1, 'fe_util_sq': 1, 'fe_high_util': 1, 'fe_over_limit': 1,
        'fe_total_dpd': 1, 'fe_dpd_severity': 1, 'fe_any_delinquency': 1,
        'fe_chronic_default': 1, 'fe_dpd_unknown': 1,
        'DebtRatio': 1, 'NumOpenLoans': 1,
        'MonthlyIncome': -1, 'fe_log_MonthlyIncome': -1, 'fe_income_per_dep': -1,
        'age': 0,
    }
    mono = [MONO_MAP.get(c, 0) for c in feature_names]
    best_params['monotone_constraints'] = mono
    best_params['monotone_constraints_method'] = 'advanced'
```
+1 = higher value → higher default risk; -1 = higher value → lower risk; 0 = unconstrained.

#### Cost-optimal threshold function
Add after find_optimal_threshold():
```python
def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
) -> float:
    """Minimise total business cost. Default 1:10 ratio: missed default ~10x false alarm."""
    best_cost, best_thr = float("inf"), 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        total_cost = cost_fp * fp + cost_fn * fn
        if total_cost < best_cost:
            best_cost, best_thr = total_cost, float(thr)
    return best_thr
```

### UC-B Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| run_feature_pipeline returns one value | Signature updated to return tuple | Unpack: df_fe, knn_imputer = run_feature_pipeline(...) |
| fe_over_limit always 0 | Flag added after winsorise clips RevolvingUtil to [0,1] | Must compute BEFORE clip — inside winsorise() |
| fe_dpd_unknown always 0 | Flag computed after error codes replaced with NaN/0 | Compute unknown_mask BEFORE df[col].where(~unknown_mask) |
| Optuna PR-AUC drops after tuning | roc_auc_score used as objective; search space too wide | Switch to average_precision_score; tighten search space |
| ADASYN fails on small fold | Too few minority samples for k-NN neighbourhood | Wrap with try/except SMOTE fallback |
| Monotonic constraints ignored | best_params updated after LGBMClassifier constructed | Update best_params dict BEFORE constructing LGBMClassifier |
| KNN imputer leaks val/test stats | fit_imputer=True passed for val/test splits | Pass fit_imputer=False and knn_imputer=fitted_imputer for val/test |

