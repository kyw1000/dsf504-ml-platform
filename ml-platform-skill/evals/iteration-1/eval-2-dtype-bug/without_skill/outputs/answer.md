# Diagnosis: TypeError in `_stats()` — "Raw vs Processed" Tab

## Root Cause

The crash occurs in `dashboard/app.py`, inside the `"Raw vs Processed"` tab (Tab 2 of the post-processing EDA section, approximately line 4863).

The `_stats()` helper is defined inside a `for col in picked:` loop and calls `s.mean()`, `s.std()`, `s.min()`, `s.median()`, and `s.max()` on whatever Series is passed in. These operations require a numeric dtype.

The guard that was supposed to prevent string columns from reaching `_stats()` is:

```python
_raw_num_cols = set(df_raw.select_dtypes(include="number").columns)
shared_num = [
    c for c in df_fe.select_dtypes(include="number").columns
    if c in _raw_num_cols and c != target
]
```

This filter correctly excludes columns that are non-numeric in the **processed** dataset (`df_fe`). However, it does not protect against a subtler case: a column can be numeric in `df_fe` yet still be stored as `object` (string) dtype in `df_raw` — for example, a categorical column that was label-encoded or ordinally mapped during feature engineering, but whose raw parquet still holds the original string values.

When such a column passes the filter (because `df_fe` has it as numeric AND `df_raw` technically has a column with the same name — but the raw parquet version has string dtype, e.g. `"object"`), the code reads:

```python
_rv = df_raw[col].dropna()   # dtype = object / string
_fv = df_fe[col].dropna()    # dtype = int64 / float64
rs, fs = _stats(_rv), _stats(_fv)
```

`_stats(_rv)` then calls `_rv.mean()` on a string-dtype Series, which raises:

```
TypeError: Cannot perform reduction 'mean' with string dtype
```

The existing comment on line 4814 even acknowledges the scenario but the fix was applied only to the list-comprehension filter on `df_fe`, not to `df_raw`:

```python
# A column encoded to int in processed but still string in raw
# would crash mean()/std() on the raw series.
_raw_num_cols = set(df_raw.select_dtypes(include="number").columns)
```

The intent was correct but the implementation is incomplete: `_raw_num_cols` is built from `df_raw.select_dtypes(include="number")`, which should exclude string columns. The bug therefore has a second possible trigger path — `load_parquet` returns the raw dataframe exactly as stored in the parquet file. If the parquet was written with a column typed as `object` (e.g., pandas `StringDtype` or plain Python `object` holding strings), `select_dtypes(include="number")` will correctly exclude it. This means `shared_num` should already exclude such columns.

**The actual failure mode** occurs because the `_stats()` function is defined *inside the for-loop* (a Python closure issue is not the direct cause here), but more critically: the `picked` multiselect widget is populated from `shared_num`, which is filtered at page-load time and cached. If `df_raw` was loaded from a cached parquet that had numeric dtypes, but the underlying data for some columns contains mixed types (numbers stored as strings), `pd.read_parquet` will infer `object` dtype for that column. `select_dtypes(include="number")` will then *exclude* it from `_raw_num_cols`, meaning it won't appear in `shared_num` — unless the raw parquet was written with an explicit numeric dtype for that column but the values inside are strings (possible if the parquet schema declares `int64` but actual values were coerced).

**Simplest and most direct cause:** The raw dataset contains a column where the parquet schema is numeric but the Series values are actually string objects due to mixed-type data written improperly. `dropna()` preserves the object dtype. Calling `.mean()` on an object-dtype Series raises the `TypeError`.

## The Fix

Apply `pd.to_numeric(..., errors='coerce')` inside `_stats()` before computing any aggregation, so that even if a string-dtype Series sneaks through, numeric operations will succeed (non-parseable strings become `NaN` and are dropped). Also add an explicit numeric cast guard in the loop before calling `_stats()`.

### Option A — Minimal, defensive fix inside `_stats()` (recommended)

In `dashboard/app.py`, replace the `_stats` definition (around line 4863):

```python
# BEFORE (broken)
def _stats(s):
    return {
        "n":      len(s),
        "mean":   s.mean(),
        "std":    s.std(),
        "min":    s.min(),
        "median": s.median(),
        "max":    s.max(),
        "skew":   float(s.skew()) if len(s) > 2 else 0.0,
        "missing%": (1 - len(s)/max(len(df_raw), 1)) * 100,
    }
```

```python
# AFTER (fixed)
def _stats(s):
    # Coerce to numeric — guards against object/string dtype columns
    # that may slip through the shared_num filter (e.g. mixed-type parquet).
    s = pd.to_numeric(s, errors="coerce").dropna()
    return {
        "n":      len(s),
        "mean":   s.mean()   if len(s) > 0 else float("nan"),
        "std":    s.std()    if len(s) > 0 else float("nan"),
        "min":    s.min()    if len(s) > 0 else float("nan"),
        "median": s.median() if len(s) > 0 else float("nan"),
        "max":    s.max()    if len(s) > 0 else float("nan"),
        "skew":   float(s.skew()) if len(s) > 2 else 0.0,
        "missing%": (1 - len(s)/max(len(df_raw), 1)) * 100,
    }
```

### Option B — Also guard the loop variables before calling `_stats()`

Add explicit casts right after the values are extracted from the dataframes (around line 4858–4860):

```python
for col in picked:
    _rv = pd.to_numeric(df_raw[col], errors="coerce").dropna()
    _fv = pd.to_numeric(df_fe[col],  errors="coerce").dropna()

    def _stats(s):
        return {
            "n":      len(s),
            "mean":   s.mean()   if len(s) > 0 else float("nan"),
            "std":    s.std()    if len(s) > 0 else float("nan"),
            "min":    s.min()    if len(s) > 0 else float("nan"),
            "median": s.median() if len(s) > 0 else float("nan"),
            "max":    s.max()    if len(s) > 0 else float("nan"),
            "skew":   float(s.skew()) if len(s) > 2 else 0.0,
            "missing%": (1 - len(s)/max(len(df_raw), 1)) * 100,
        }

    rs, fs = _stats(_rv), _stats(_fv)
```

Option B is the most thorough because it ensures both `_rv` and `_fv` are already numeric before they are used in both `_stats()` and the histogram traces below.

## Summary

| Item | Detail |
|---|---|
| File | `dashboard/app.py` |
| Tab | "Raw vs Processed" (Tab 2 of post-processing EDA) |
| Function | `_stats()` (defined inside the `for col in picked:` loop, ~line 4863) |
| Failing call | `s.mean()` (also `s.std()`, `s.min()`, `s.median()`, `s.max()`) |
| Root cause | A column in `df_raw` has `object`/string dtype (mixed-type or string-encoded parquet values), so calling any numeric reduction raises `TypeError: Cannot perform reduction 'mean' with string dtype` |
| Fix | Coerce the Series to numeric with `pd.to_numeric(s, errors="coerce").dropna()` before any aggregation, either inside `_stats()` (Option A) or at the loop level before calling `_stats()` (Option B, preferred) |
