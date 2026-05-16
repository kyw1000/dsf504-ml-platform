# Diagnosis & Fix — TypeError: Cannot perform reduction 'mean' with string dtype

## Error Location

**Page:** Post-Processing EDA (`page_post_processing_eda`)
**Tab:** Raw vs Processed
**Crash site:** `_stats(s)` called on line ~4875 of `dashboard/app.py`, which calls `s.mean()` on a pandas Series that has string/object dtype.

---

## Root Cause

`shared_num` is built by selecting numeric columns from the **processed** parquet (`df_fe`):

```python
# BUGGY — only checks df_fe dtype
shared_num = [
    c for c in df_fe.select_dtypes(include="number").columns
    if c != target
]
```

Feature engineering (Step 3) label-encodes categorical columns to integers in the processed dataset. Those same columns remain as strings (`object` or `ArrowDtype str`) in the raw parquet (`df_raw`).

When the loop reaches one of those columns and calls `_rv = df_raw[col].dropna()` followed by `_stats(_rv)`, the Series has string dtype. `s.mean()` then raises:

```
TypeError: Cannot perform reduction 'mean' with string dtype
```

---

## Fix

Intersect the candidate list against numeric columns from **both** DataFrames. Columns that are numeric only in the processed dataset are excluded from the comparison (they may still appear in the "New Features" list below).

### Code change in `dashboard/app.py` (inside `page_post_processing_eda`, Raw vs Processed tab)

```python
# BEFORE
shared_num = [
    c for c in df_fe.select_dtypes(include="number").columns
    if c != target
]
```

```python
# AFTER
_raw_num_cols = set(df_raw.select_dtypes(include="number").columns)
shared_num = [
    c for c in df_fe.select_dtypes(include="number").columns
    if c in _raw_num_cols and c != target
]
```

No other changes are required. The `_stats()` helper itself is correct — the bug is purely in the column-selection guard upstream of it.

---

## Why This Works

- `df_fe.select_dtypes(include="number")` returns every column that is numeric after encoding — including formerly-categorical columns that were integer-encoded.
- Adding `if c in _raw_num_cols` ensures only columns that were **already numeric in the raw data** are passed to `_stats()`.
- Label-encoded columns (string in raw, int in processed) fall out of `shared_num` automatically and do not cause a dtype crash.
- The existing `new_feats` list (columns present only in `df_fe`) is unaffected; it is built independently and is never passed to `_stats()`.

---

## Reference

See `references/pitfalls.md` — Pitfall #5 ("TypeError: Cannot perform reduction 'mean' with string dtype") for the full root-cause analysis and the canonical fix snippet.
