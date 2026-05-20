---
name: ml-platform
description: >
  Expert assistant for the DSF504 ML Platform -- a Streamlit multi-use-case
  machine learning dashboard with five-step pipeline scripts.
  Use this skill whenever the user asks about, works on, or debugs anything
  in the DSF504 project, including: the Streamlit dashboard (app.py), pipeline
  scripts (01-05 per use case), config.py, utils/, or any page/tab/chart
  inside the platform. Also trigger for requests like "add a new use case",
  "add a page / tab", "fix this dashboard error", "improve the correlation
  chart", or any question about the ML Framework phases used in this project.
  Trigger even when the user says something casual like "the dashboard is
  broken" or "can you add something to the model page" -- if it is about
  DSF504, always invoke this skill.
---

# DSF504 ML Platform -- Expert Skill

## Project at a Glance

| Item | Value |
|------|-------|
| Root | C:\\DSF504\\ (bash: /sessions/.../mnt/DSF504/) |
| Dashboard | dashboard/app.py (~2,700 lines) |
| Run | streamlit run dashboard/app.py (from C:\\DSF504\\) |
| Python | 3.13, Windows |
| Active use cases | A (fraud), B (credit), C_nlp (NLP sentiment), C_markets (volatility), D (churn), E (insurance) |

## ML Framework -- 8 Dashboard Pages

Top nav uses st.radio(horizontal=True) styled as a pill tab strip (no `---` separator between nav and content):

```
▶️  Run Pipeline           -- execute any of the 5 pipeline steps
🔬 Data Studio            -- raw data browser + profiling + correlation matrix
🔧 Data Preparation       -- feature engineering guidance + run Step 3
📈 Post-Processing EDA    -- raw vs processed + Report Figures (PNG gallery)
🤖 Model Development      -- algorithm comparison + CV explorer + HP tuning guide
📊 Model Evaluation       -- champion metrics + threshold calibration
🎯 Prediction Demo        -- live inference on the tuned champion
🔍 Ethics & Explainability -- SHAP + fairness audit
```

PAGES dict and _NAV_SECTIONS must use byte-identical label strings --
any mismatch causes a nav button that never activates.

## Five-Step Pipeline (per use case)

```
Step 1  01_data_loading.py          -> Parquet splits (train/val/test)
Step 2  02_eda_analysis.py          -> CSV/PNG reports in reports/<uc>/
Step 3  03_feature_engineering.py   -> train_fe.parquet, feature list CSV
Step 4  04_model_training.py        -> champion.pkl, model_comparison.csv
Step 5  05_hyperparameter_tuning.py -> final_model.pkl, tuning logs
```

See references/architecture.md for the full import/logging/ensure_utf8
boilerplate every script must include.

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
```bash
python3 - <<'EOF'
content = open('/sessions/.../mnt/DSF504/target.py', encoding='utf-8').read()
# ... make your changes to content ...
with open('/sessions/.../mnt/DSF504/target.py', 'w', encoding='utf-8') as f:
    f.write(content)
import ast; ast.parse(content); print("AST OK,", len(content.splitlines()), "lines")
EOF
```

### Strip null bytes after any Edit on a large file
```bash
python3 - <<'EOF'
path = '/sessions/.../mnt/DSF504/dashboard/app.py'
raw = open(path, 'rb').read()
if b'\x00' in raw:
    open(path, 'wb').write(raw.rstrip(b'\x00'))
    print("Stripped null bytes")
import ast; ast.parse(open(path, encoding='utf-8').read()); print("AST OK")
EOF
```

### Use utils/file_guard.py
```bash
python utils/file_guard.py dashboard/app.py      # check + strip nulls
```
```python
from utils.file_guard import check_file, safe_write
check_file("dashboard/app.py")       # after any Edit
safe_write("big_file.py", content)   # instead of Write tool
```

Never chain multiple Edit calls on the same large file in one turn -- each
Edit re-reads the truncated state and compounds the damage. Write the complete
final content in one bash Python call instead.

## Key Patterns -- Quick Reference

### Adding a new page
1. Write def page_my_page(uc_key: str) -> None: anywhere in app.py.
2. Add to BOTH PAGES dict and _NAV_SECTIONS with the same label string.
3. Use suffix="unique_id" on _run_step_action() if the same (step, uc_key)
   pair appears more than once -- duplicate widget keys crash Streamlit.

### Navigation tab CSS -- the cascade trap
color on <label> does NOT cascade to the inner <p> in Streamlit radio buttons.
Always target both elements:

```css
div[data-testid="stRadio"] div[role="radiogroup"] > label {
    background: #FFFFFF; color: #1A237E !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label p {
    color: #1A237E !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: #3949AB !important; color: #FFFFFF !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #FFFFFF !important;
}
```

### Dark background — always use light text

The dashboard uses a dark background (`BG = "#1A1A2E"`). Any inline HTML,
custom CSS, or st.markdown content must use light colours or it will be
invisible against the dark canvas.

```python
# ✅ Correct — light text on dark background
st.markdown(f"<p style='color:{FONT};'> ... </p>", unsafe_allow_html=True)

# ❌ Wrong — default browser black text, invisible on dark bg
st.markdown("<p> ... </p>", unsafe_allow_html=True)
```

When writing CSS selectors that set text colour, always specify both the
container element AND any inner `<p>` tags, because Streamlit wraps markdown
text in `<p>` which resets colour:

```css
/* target the label AND its inner <p> */
div.my-class { color: #E0E0E0 !important; }
div.my-class p { color: #E0E0E0 !important; }
```

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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

BG, FONT = "#1A1A2E", "#E0E0E0"
fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=FONT, height=340)
st.plotly_chart(fig, width='stretch')    # not use_container_width (removed in Streamlit ≥1.44)
st.dataframe(df, width='stretch', hide_index=True)
st.image(img_path, width='stretch')      # same — use_container_width= is gone
```

`use_container_width` is fully removed. Always use:
- `width='stretch'`  (replaces use_container_width=True)
- `width='content'`  (replaces use_container_width=False)

Applies to: st.plotly_chart, st.dataframe, st.image, st.button, and all
other Streamlit widgets that previously accepted use_container_width.

### Adding a new use case
1. Create use_case_X/01...05_*.py (copy Use Case A as template).
2. Add key to USE_CASE_SCRIPTS and USE_CASE_META in app.py.
3. Add dicts: _PROFILING_SRC, _FE_EDA_SRC, _FE_GUIDANCE, _DATASET_ANALYSIS_CONFIG.
4. Use "status": "scaffolded" until all five scripts run end-to-end successfully.

### Git -- must run from Windows, not bash sandbox
The bash sandbox mounts the Windows filesystem read-only for git operations.
Run all git commands from Windows PowerShell:
```powershell
cd C:\DSF504
git add -A
git commit -m "fix: describe change"
git push
```

## Known Pitfalls -- Quick Lookup

See references/pitfalls.md for full root-cause analysis.

| Symptom | Cause | Fix |
|---------|-------|-----|
| File silently truncated after Edit/Write | Tool buffer cap ~12 KB | Use bash Python for files > 10 KB |
| Null bytes / SyntaxError after Edit | Tool pads with 0x00 when shrinking | raw.rstrip(b'\\x00') then rewrite |
| StreamlitDuplicateElementKey | Two _run_step_action(N, uc_key) on same page | Add suffix="unique" to second call |
| NameError: plt is not defined | matplotlib.pyplot not imported in dashboard | Replace with plotly go.Histogram etc |
| Garbled output on Windows | cp1252 vs UTF-8 pipe encoding | ensure_utf8() before logging.basicConfig() |
| TypeError: multi_class unexpected kwarg | Removed in scikit-learn 1.5 | Delete multi_class= from LogisticRegression |
| Dataset scripts are no longer supported | HuggingFace datasets v3 dropped script loading | Use parquet fallbacks + raw text mirrors (pitfalls #13) |
| Nav tab text invisible (dark on dark) | CSS color on label does not cascade to p | Also target > label p { color: !important } |
| git init fails from bash sandbox | Mounted Windows filesystem | Run git from Windows PowerShell |
| PerformanceWarning: DataFrame fragmented | Many df["col"] = ... assignments | Collect in dict, pd.concat once |
| use_container_width deprecation warning | Removed in Streamlit ≥1.44 | width='stretch' (True) or width='content' (False) — all widgets incl. st.image |
| Pandas4Warning on select_dtypes("object") | pandas 3 separates str from object | Use include=["object","category","str"] |
| PAGES / _NAV_SECTIONS key mismatch | Label strings differ | Copy-paste labels; debug with .encode('utf-8').hex() |

## Helper Functions (app.py)

| Function | Purpose |
|----------|---------|
| section_header(title, subtitle) | Blue title bar at top of every page |
| load_parquet(path) | Cached parquet loader with error handling |
| load_csv(path) | Cached CSV loader |
| _run_step_action(step, uc_key, label, suffix="") | Button that navigates to Run Pipeline and auto-triggers step |
| _render_correlation_matrix(uc_key) | Interactive Pearson/Spearman heatmap |
| render_sidebar() | Builds nav sidebar; returns uc_key string |
| metric_card(label, value, delta, colour) | Coloured metric tile |
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
