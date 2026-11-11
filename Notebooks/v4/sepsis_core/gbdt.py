"""LightGBM reference on causal per-hour features (v3 cell 14, same features and parameters), plus the two additions agreed for v4:
a TIME-BLIND variant and a SMOTE (FIT-only) variant. Natural class distribution, no class weights, unless SMOTE is requested.

ICULOS note. "no_ICULOS" in v3 was not time-free: for a feature never measured so far, time-since-measurement equals hours since
admission, so the trees could read the ICU hour (v3's top feature was FiO2_hrs_since). `time_blind=True` sets that value to the cap
for never-measured features. Rolling 6 h windows still shorten in the first hours of a stay, so a little time information remains.
"""
import os, json
import numpy as np

from . import config as C
from . import balancing as BAL

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7,
              lambda_l2=1.0, bagging_fraction=1.0, max_bin=255, verbose=-1, device_type="cpu")
HRS_CAP = 240.0


def stay_rows(store, stay_idx):
    return np.concatenate([np.arange(store.starts[i], store.starts[i] + store.lens[i]) for i in stay_idx]) if len(stay_idx) else np.zeros(0, np.int64)


def never_measured(store, rows):
    """(len(rows), F) bool: True where the feature has not been measured at or before this hour of the stay."""
    start_row = np.repeat(store.starts, store.lens); out = np.zeros((len(rows), store.M.shape[1]), bool)
    for j in range(store.M.shape[1]):
        m = store.M[:, j].astype(np.int32); cm = np.cumsum(m)
        out[:, j] = (cm - (cm[start_row] - m[start_row]))[rows] == 0
    return out


def row_matrix(store, pre, stay_idx, with_iculos=False, time_blind=False):
    if with_iculos and time_blind: raise ValueError("time_blind and with_iculos are opposites")
    rows = stay_rows(store, stay_idx)
    hrs = np.log1p(np.minimum(store.D[rows], HRS_CAP)).astype(np.float32)
    if time_blind: hrs[never_measured(store, rows)] = np.float32(np.log1p(HRS_CAP))
    parts = [store.XL[rows], store.M[rows], hrs, store.E[rows].astype(np.float32), store.S[rows], store.G[rows].astype(np.float32)[:, None]]
    names = ([f"{c}_ffill" for c in pre.feat] + [f"{c}_obs" for c in pre.feat] + [f"{c}_hrs_since" for c in pre.feat]
             + list(pre.state["eng_names"]) + ["age", "hospadm", "unit1", "unit2", "gender"])
    if with_iculos:
        parts.append(np.log1p(store.hours[rows])[:, None].astype(np.float32)); names.append("log_iculos")
    X = np.concatenate(parts, 1)
    C.assert_no_forbidden([n for n in names if n != "log_iculos"]); assert X.shape[1] == len(names)
    return rows, X, names


def train(store, pre, fit_idx, val_idx, with_iculos=False, time_blind=False, smote_ratio=None, rounds=3000, seed=None):
    """-> dict(rows, scores (probabilities, (n,H)), boosters, names, best_iter, top_features, n_synthetic). FIT rows train, VAL rows
    early-stop and are scored. SMOTE, when requested, is applied to FIT rows only (balancing.smote_rows checks the folds)."""
    import lightgbm as lgb
    seed = C.CFG.SEED if seed is None else seed
    r_fit, X_fit, names = row_matrix(store, pre, fit_idx, with_iculos, time_blind); r_val, X_val, _ = row_matrix(store, pre, val_idx, with_iculos, time_blind)
    Hn = store.Y.shape[1]; scores = np.zeros((len(r_val), Hn), np.float32); boosters, best_iter, n_syn, top = [], [], [], None
    stay_of = np.searchsorted(store.starts, r_fit, side="right") - 1; folds = np.asarray(store.stay_fold)[stay_of]
    params = dict(PARAMS, seed=seed, num_threads=os.cpu_count())
    for k in range(Hn):
        mf = store.V[r_fit, k].astype(bool); mv = store.V[r_val, k].astype(bool)
        Xtr, ytr, n_new = X_fit[mf], store.Y[r_fit[mf], k].astype(np.float32), 0
        if smote_ratio: Xtr, ytr, n_new = BAL.smote_rows(Xtr, ytr > 0.5, folds[mf], target_ratio=float(smote_ratio), seed=seed + k)
        dtr = lgb.Dataset(Xtr, ytr, feature_name=names, free_raw_data=True)
        dva = lgb.Dataset(X_val[mv], store.Y[r_val[mv], k].astype(np.float32), reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        scores[:, k] = bst.predict(X_val, num_iteration=bst.best_iteration)
        boosters.append(bst); best_iter.append(int(bst.best_iteration or rounds)); n_syn.append(int(n_new))
        if k == 0:
            gain = bst.feature_importance("gain"); top = [names[i] for i in np.argsort(-gain)[:12]]
        del Xtr, ytr, dtr, dva
    return dict(rows=r_val, scores=scores, boosters=boosters, names=names, best_iter=best_iter, top_features=top, n_synthetic=n_syn,
                with_iculos=with_iculos, time_blind=time_blind, smote_ratio=smote_ratio)


def predict(g, store, pre, stay_idx):
    """Score another set of stays (e.g. an external cohort) with trained boosters. -> rows, probabilities (n,H)."""
    rows, X, names = row_matrix(store, pre, stay_idx, g["with_iculos"], g["time_blind"])
    assert names == g["names"], "feature layout differs from the trained model"
    return rows, np.stack([b.predict(X, num_iteration=it) for b, it in zip(g["boosters"], g["best_iter"])], 1).astype(np.float32)


def fit_or_load(dirpath, store, pre, fit_idx, val_idx, **kw):
    """Resumable wrapper: boosters are saved as text files, VAL scores as npz."""
    import lightgbm as lgb
    meta_p = os.path.join(dirpath, "gbdt.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p)); z = np.load(os.path.join(dirpath, "val_scores.npz"))
        boosters = [lgb.Booster(model_file=os.path.join(dirpath, f"h{k}.txt")) for k in range(len(meta["best_iter"]))]
        print(f"[gbdt] loaded {dirpath}")
        return dict(meta, rows=z["rows"], scores=z["scores"], boosters=boosters)
    if C.CFG.SMOKE_TEST: kw.setdefault("rounds", 50)
    g = train(store, pre, fit_idx, val_idx, **kw); os.makedirs(dirpath, exist_ok=True)
    for k, b in enumerate(g["boosters"]): b.save_model(os.path.join(dirpath, f"h{k}.txt"))
    np.savez_compressed(os.path.join(dirpath, "val_scores.npz"), rows=g["rows"], scores=g["scores"])
    json.dump({k: v for k, v in g.items() if k not in ("rows", "scores", "boosters")}, open(meta_p, "w"), default=str)
    return g


def to_logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7); return np.log(p / (1 - p)).astype(np.float32)
