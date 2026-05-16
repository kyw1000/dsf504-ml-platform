"""Fast-path Step 5 — 10 Optuna trials on 10% sample, then retrain full."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd, pickle, warnings, time
warnings.filterwarnings("ignore")
from config import DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from utils.encoding_guard import ensure_utf8
ensure_utf8()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

import lightgbm as lgb, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_SUBDIR = DATA_DIR  / "optiver_volatility"
MODEL_DIR   = MODELS_DIR / "use_case_C_markets"
REPORT_DIR  = REPORTS_DIR / "use_case_C_markets"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FE = DATA_SUBDIR / "train_fe.parquet"
VAL_FE   = DATA_SUBDIR / "val_fe.parquet"
TEST_FE  = DATA_SUBDIR / "test_fe.parquet"

def rmspe(y_true, y_pred):
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_pred[mask]-y_true[mask])/y_true[mask])**2)))

df_tr = pd.read_parquet(TRAIN_FE)
df_vl = pd.read_parquet(VAL_FE)
df_te = pd.read_parquet(TEST_FE)
fe_cols = sorted([c for c in df_tr.columns if c.startswith("fe_") and pd.api.types.is_numeric_dtype(df_tr[c])])

X_tr = df_tr[fe_cols].fillna(0).values.astype(np.float32)
y_tr_log  = df_tr["log_target"].fillna(0).values
y_tr_orig = df_tr["target"].fillna(1e-8).values
X_vl = df_vl[fe_cols].fillna(0).values.astype(np.float32)
y_vl_log  = df_vl["log_target"].fillna(0).values
y_vl_orig = df_vl["target"].fillna(1e-8).values
X_te = df_te[fe_cols].fillna(0).values.astype(np.float32)
y_te_orig = df_te["target"].fillna(1e-8).values

# 10% sample for tuning
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(X_tr), int(len(X_tr)*0.10), replace=False)
Xc = X_tr[idx]; yc_log = y_tr_log[idx]; yc_orig = y_tr_orig[idx]

from sklearn.model_selection import KFold

def objective(trial):
    p = {
        "n_estimators":     trial.suggest_int("n_estimators", 80, 300),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 16, 63),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 30),
        "objective": "regression", "metric": "rmse",
        "random_state": RANDOM_STATE, "verbose": -1, "n_jobs": -1,
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for ti, vi in kf.split(Xc):
        m = lgb.LGBMRegressor(**p)
        m.fit(Xc[ti], yc_log[ti])
        scores.append(rmspe(yc_orig[vi], np.expm1(np.clip(m.predict(Xc[vi]),-10,10))))
    return float(np.mean(scores))

log.info("Optuna search: 10 trials …")
study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=10, show_progress_bar=False)
log.info("Best trial RMSPE=%.5f  params=%s", study.best_value, study.best_params)

# Retrain on full training set
best = {**study.best_params, "objective":"regression","metric":"rmse",
        "random_state":RANDOM_STATE,"verbose":-1,"n_jobs":-1}
final = lgb.LGBMRegressor(**best)
final.fit(X_tr, y_tr_log)

def eval_split(X, y_orig, name):
    p_log  = final.predict(X)
    p_orig = np.expm1(np.clip(p_log, -10, 10))
    rp = rmspe(y_orig, p_orig)
    log.info("  %s RMSPE=%.5f", name, rp)
    return rp

val_rmspe  = eval_split(X_vl, y_vl_orig, "val")
test_rmspe = eval_split(X_te, y_te_orig, "test")

# Feature importance
imp = pd.Series(final.feature_importances_, index=fe_cols).sort_values(ascending=False)
imp.reset_index().rename(columns={"index":"feature",0:"importance"}).to_csv(
    REPORT_DIR / "feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(10,6))
fig.patch.set_facecolor("#1A1A2E"); ax.set_facecolor("#1A1A2E")
top = imp.head(15)
ax.barh(top.index[::-1], top.values[::-1], color="#AB47BC", edgecolor="none")
ax.set_title("LightGBM Feature Importance — Top 15", color="white")
ax.tick_params(colors="white"); ax.set_xlabel("Importance", color="white")
plt.tight_layout()
plt.savefig(REPORT_DIR/"feature_importance.png", dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
plt.close()

# Tuning history plot
vals = [t.value for t in study.trials if t.value]
fig, ax = plt.subplots(figsize=(10,4))
fig.patch.set_facecolor("#1A1A2E"); ax.set_facecolor("#1A1A2E")
ax.plot(vals, color="#42A5F5", marker="o", ms=4)
ax.plot(np.minimum.accumulate(vals), color="#EF5350", linewidth=2, linestyle="--", label="Best")
ax.set_title("Optuna Trial RMSPE", color="white"); ax.tick_params(colors="white")
ax.set_xlabel("Trial", color="white"); ax.set_ylabel("RMSPE", color="white")
ax.legend(facecolor="#1A1A2E", labelcolor="white")
plt.tight_layout()
plt.savefig(REPORT_DIR/"tuning_history.png", dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
plt.close()

# Save final model
payload = {"model": final, "fe_cols": fe_cols, "best_params": study.best_params,
           "val_rmspe": val_rmspe, "test_rmspe": test_rmspe}
with open(MODEL_DIR / "lgbm_optuna_champion.pkl", "wb") as f:
    pickle.dump(payload, f)
log.info("Saved lgbm_optuna_champion.pkl")

pd.DataFrame([{"model":"lgbm_optuna","val_rmspe":round(val_rmspe,6),
               "test_rmspe":round(test_rmspe,6),"n_trials":10}]).to_csv(
    REPORT_DIR/"final_model_metrics.csv", index=False)
log.info("Saved final_model_metrics.csv")
log.info("Step 5 complete — Val RMSPE=%.5f  Test RMSPE=%.5f", val_rmspe, test_rmspe)
