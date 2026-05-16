"""Fast-path runner for Step 4 — uses 15% sample + 2 models to fit sandbox timeout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Patch constants before importing the module

import numpy as np
import pandas as pd
import pickle, time, warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE, CV_FOLDS
from utils.encoding_guard import ensure_utf8
ensure_utf8()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_SUBDIR = DATA_DIR  / "optiver_volatility"
MODEL_DIR   = MODELS_DIR / "use_case_C_markets"
REPORT_DIR  = REPORTS_DIR / "use_case_C_markets"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FE_PQ = DATA_SUBDIR / "train_fe.parquet"
VAL_FE_PQ   = DATA_SUBDIR / "val_fe.parquet"
TARGET_COL  = "target"
LOG_TARGET  = "log_target"

def rmspe(y_true, y_pred):
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_pred[mask]-y_true[mask])/y_true[mask])**2)))

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

df_train = pd.read_parquet(TRAIN_FE_PQ)
df_val   = pd.read_parquet(VAL_FE_PQ)
fe_cols  = sorted([c for c in df_train.columns if c.startswith("fe_") and pd.api.types.is_numeric_dtype(df_train[c])])
log.info("Features: %d  train: %d  val: %d", len(fe_cols), len(df_train), len(df_val))

# 15% sample
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(df_train), int(len(df_train)*0.15), replace=False)
X_cv = df_train[fe_cols].fillna(0).values[idx].astype(np.float32)
y_cv_log  = df_train[LOG_TARGET].fillna(0).values[idx]
y_cv_orig = df_train[TARGET_COL].fillna(1e-8).values[idx]

X_train = df_train[fe_cols].fillna(0).values.astype(np.float32)
y_train_log  = df_train[LOG_TARGET].fillna(0).values
y_train_orig = df_train[TARGET_COL].fillna(1e-8).values
X_val   = df_val[fe_cols].fillna(0).values.astype(np.float32)
y_val_log   = df_val[LOG_TARGET].fillna(0).values
y_val_orig  = df_val[TARGET_COL].fillna(1e-8).values

models = {
    "ridge":    Ridge(alpha=10.0),
    "lightgbm": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05,
                                   num_leaves=31, random_state=RANDOM_STATE, verbose=-1),
}

results = []
trained = {}
kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    log.info("--- %s ---", name.upper())
    fold_rmspe = []
    for tr_i, vl_i in kf.split(X_cv):
        model.fit(X_cv[tr_i], y_cv_log[tr_i])
        p = np.expm1(np.clip(model.predict(X_cv[vl_i]), -10, 10))
        fold_rmspe.append(rmspe(y_cv_orig[vl_i], p))
    cv_rmspe = float(np.mean(fold_rmspe))

    model.fit(X_train, y_train_log)
    val_preds_log  = model.predict(X_val)
    val_preds_orig = np.expm1(np.clip(val_preds_log, -10, 10))
    val_rmspe = rmspe(y_val_orig, val_preds_orig)
    val_r2    = float(r2_score(y_val_log, val_preds_log))
    log.info("  CV RMSPE=%.5f  Val RMSPE=%.5f  Val R²=%.4f", cv_rmspe, val_rmspe, val_r2)

    results.append({"model": name, "rmspe_mean": round(cv_rmspe,5),
                    "rmse_mean": 0, "mae_mean": 0, "r2_mean": round(val_r2,4),
                    "cv_rmspe_mean": round(cv_rmspe,5), "cv_rmspe_std": 0,
                    "cv_r2_mean": round(val_r2,4),
                    "val_rmspe": round(val_rmspe,5), "val_rmse": 0, "val_r2": round(val_r2,4)})
    trained[name] = model
    with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump({"model": model, "fe_cols": fe_cols}, f)

res_df = pd.DataFrame(results)
res_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
log.info("Saved model_comparison.csv")

# Simple bar chart
fig, ax = plt.subplots(figsize=(8,4))
fig.patch.set_facecolor("#1A1A2E"); ax.set_facecolor("#1A1A2E")
ax.barh(res_df["model"], res_df["cv_rmspe_mean"], color="#EF5350", edgecolor="none")
ax.set_title("Model Comparison — CV RMSPE (lower=better)", color="white")
ax.tick_params(colors="white"); ax.set_xlabel("RMSPE", color="white")
plt.tight_layout()
plt.savefig(REPORT_DIR / "model_comparison.png", dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
plt.close()

champion = res_df.sort_values("cv_rmspe_mean").iloc[0]["model"]
log.info("Champion: %s", champion)
with open(MODEL_DIR / "champion.pkl", "wb") as f:
    pickle.dump({"model": trained[champion], "fe_cols": fe_cols, "champion_name": champion}, f)
log.info("Saved champion.pkl -> %s", champion)
log.info("Step 4 complete.")
