# DSF504 Known Pitfalls & Fixes

## 1. StreamlitDuplicateElementKey

**Error:** `StreamlitDuplicateElementKey: There are multiple elements with the same key='_page_run_4_A'`

**Cause:** `_run_step_action(step, uc_key)` generates key `f"_page_run_{step}_{uc_key}"`.
If the same `(step, uc_key)` pair is called more than once in the same render pass
(e.g., Step 4 appears in both the Algorithm Comparison tab and the CV Results tab
of `page_model_development`), Streamlit sees two widgets with identical keys.

**Fix:** Pass a unique `suffix` to each duplicate callsite:
```python
# tab_algo
_run_step_action(4, uc_key, "▶ Run Step 4 …", suffix="algo")
# tab_cv
_run_step_action(4, uc_key, "▶ Run Step 4 …", suffix="cv")
```

---

## 2. NameError: name 'plt' is not defined

**Cause:** `matplotlib.pyplot` is not imported in `dashboard/app.py`.
The dashboard imports `matplotlib` (for `matplotlib.use("Agg")`) but
never imports `pyplot`.

**Fix:** Replace every `plt.subplots / st.pyplot / plt.close` block with plotly:
```python
# Instead of:
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(raw_vals, bins=40)
st.pyplot(fig); plt.close(fig)

# Use:
fig = make_subplots(rows=1, cols=2, subplot_titles=["Raw", "Processed"])
fig.add_trace(go.Histogram(x=raw_vals, nbinsx=40), row=1, col=1)
fig.add_trace(go.Histogram(x=fe_vals,  nbinsx=40), row=1, col=2)
st.plotly_chart(fig, width='stretch')
```

---

## 3. Garbled Output — â€¦ â€" ✓ in pipeline logs

**Cause:** Windows uses cp1252 encoding on console stdout/stderr by default.
When a subprocess writes UTF-8 bytes (emoji, en-dash, check mark) to a pipe
and the pipe reads them as cp1252, multi-byte sequences are decoded incorrectly.

**Two-sided fix:**

_Script side_ — add before `logging.basicConfig()` in every `01–05_*.py`:
```python
from utils.encoding_guard import ensure_utf8
ensure_utf8()
```

_Dashboard side_ — both `subprocess.Popen()` calls in `_run_step()`:
```python
proc = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    cwd=str(ROOT),
    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
)
```

---

## 4. PerformanceWarning: DataFrame is highly fragmented

**Error:** `PerformanceWarning: DataFrame is highly fragmented. This is usually the result of calling frame.insert many times.`

**Cause:** Feature engineering functions add columns one by one to a 400-column
DataFrame:
```python
df["fe_card1_txn_count"] = ...
df["fe_card1_cum_amount"] = ...
# × 20 more assignments
```

**Fix:** Collect new columns in a dict, then concat once:
```python
_new: dict = {}
_new["fe_card1_txn_count"]  = df.groupby("card1").cumcount()
_new["fe_card1_cum_amount"] = df.groupby("card1")["TransactionAmt"].cumsum()
# ...
df = pd.concat([df, pd.DataFrame(_new, index=df.index)], axis=1)
```

---

## 5. TypeError: Cannot perform reduction 'mean' with string dtype

**Error:** `TypeError: Cannot perform reduction 'mean' with string dtype`
in `page_post_processing_eda` when computing stats on a column from `df_raw`.

**Cause:** `shared_num` is built from `df_fe.select_dtypes(include="number")`,
which captures columns that were label-encoded to integers in the processed
dataset. Those same columns may still be strings (ArrowDtype or object) in
the raw parquet.

**Fix:** Intersect with numeric columns from `df_raw` too:
```python
_raw_num_cols = set(df_raw.select_dtypes(include="number").columns)
shared_num = [
    c for c in df_fe.select_dtypes(include="number").columns
    if c in _raw_num_cols and c != target
]
```

---

## 6. use_container_width Deprecation

**Warning:** `Please replace use_container_width with width. use_container_width will be removed after 2025-12-31.`

**Fix (global replace):**
```python
import re
src = re.sub(r'\buse_container_width=True\b', "width='stretch'", src)
```
Applies to `st.plotly_chart`, `st.dataframe`, `st.image`, etc.

---

## 7. MatplotlibDeprecationWarning: 'labels' parameter renamed

**Warning:** `The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9`

**Fix:**
```python
# Old:
ax.boxplot([legit_vals, fraud_vals], labels=["Legit", "Fraud"], ...)
# New:
ax.boxplot([legit_vals, fraud_vals], tick_labels=["Legit", "Fraud"], ...)
```

---

## 8. Pandas4Warning: select_dtypes with 'object' won't include 'str' in pandas 3

**Warning:** `For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated.`

**Fix:**
```python
# Old:
cat_cols = df.select_dtypes(include=["object", "category"]).columns
# New (works in both pandas 2 and 3):
cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns
```

---

## 9. File Surgery — Never use sed on emoji-containing files

`sed -i` reads bytes; emoji are multi-byte UTF-8 sequences. On some systems
`sed` processes the file as a byte stream and can corrupt emoji or cut a
multi-byte sequence mid-character, silently truncating content.

**Always use Python for in-place edits:**
```python
src = open("app.py", encoding="utf-8").read()
new = src.replace(OLD, NEW)
assert new != src, "Pattern not found — check whitespace"
import ast; ast.parse(new)   # syntax check before writing
open("app.py", "w", encoding="utf-8").write(new)
```

---

## 10. PAGES / _NAV_SECTIONS Key Mismatch

If a page nav button never turns orange (active), the label in `_NAV_SECTIONS`
and `PAGES` differ by at least one character (invisible whitespace, different
emoji code point, trailing space).

**Debug:**
```python
for k in PAGES:
    print(k.encode('utf-8').hex(), repr(k))
```
Compare hex output against the `_NAV_SECTIONS` label string.

---

## 11. Auto-run Step Not Firing

`_run_step_action()` sets `st.session_state["_auto_run_step"] = step`.
`page_run_pipeline()` pops this key at the **top** of the function with:
```python
_auto_step = st.session_state.pop("_auto_run_step", None)
step_triggered = [_auto_step] if _auto_step in available_steps else []
```
If the step is not in `available_steps` (i.e. the use case has no registered
scripts for that step), the auto-run silently does nothing — correct behaviour
for scaffolded use cases.
