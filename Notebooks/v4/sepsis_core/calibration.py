"""Post-hoc probability calibration (v3 cell 12, logic unchanged): raw / prior shift / Platt / isotonic, compared out-of-fold by
stay inside the calibration set, lowest mean log loss wins. Fit on natural-distribution VAL only: never on balanced data, never on
external cohorts (PhysioNet gets the MIMIC calibrator applied as is)."""
import numpy as np
from scipy import optimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import config as C

_sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
_lgt = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
METHODS = ["raw", "prior", "platt", "isotonic"]


def fit_cal(method, z, y):
    z = np.asarray(z, np.float64); y = np.asarray(y, np.float64)
    if method == "raw": return {}
    if method == "prior":
        f = lambda b: log_loss(y, np.clip(_sig(z + b), 1e-7, 1 - 1e-7), labels=[0, 1])
        return {"b": float(optimize.minimize_scalar(f, bounds=(-30, 30), method="bounded").x)}
    if method == "platt":
        def nll(ab): s = ab[0] * z + ab[1]; return float(np.mean(np.logaddexp(0, s) - y * s))
        def grad(ab): r_ = _sig(ab[0] * z + ab[1]) - y; return np.array([np.mean(r_ * z), np.mean(r_)])
        r = optimize.minimize(nll, x0=np.array([1.0, _lgt(np.array([y.mean()]))[0] - float(np.median(z))]), jac=grad, method="L-BFGS-B")
        return {"a": float(r.x[0]), "b": float(r.x[1])}
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(z, y)
        return {"x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist()}
    raise ValueError(method)


def apply_cal_1d(method, p, z):
    z = np.asarray(z, np.float64)
    if method == "raw": out = _sig(z)
    elif method == "prior": out = _sig(z + p["b"])
    elif method == "platt": out = _sig(p["a"] * z + p["b"])
    elif method == "isotonic": out = np.interp(z, p["x"], p["y"])
    else: raise ValueError(method)
    return np.clip(out, 1e-6, 1 - 1e-6)


def cal_report(y, p, n_bins=15):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1)); b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, n_bins - 1)
    cnt = np.bincount(b, minlength=n_bins); mp = np.bincount(b, p, n_bins) / np.maximum(cnt, 1); my = np.bincount(b, y, n_bins) / np.maximum(cnt, 1)
    ece = float(np.sum(cnt * np.abs(mp - my)) / cnt.sum())
    lp = _lgt(p); A = np.c_[np.ones_like(lp), lp]
    def nll(w): s = A @ w; return float(np.mean(np.logaddexp(0, s) - y * s))
    w = optimize.minimize(nll, x0=np.array([0.0, 1.0]), method="L-BFGS-B").x
    return dict(brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])), ece=ece, cal_slope=float(w[1]),
                cal_in_the_large=float(w[0]), mean_pred=float(p.mean()), obs_rate=float(y.mean()), max_pred=float(p.max()),
                auroc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")), (mp, my, cnt)


def calibrate(store, rows, logits, cfg=None, fold_label=None):
    """-> (calibrator dict, table of out-of-fold reports, reliability curves). `fold_label` (fold of each row's stay), when given,
    must be all 'VAL': calibration is never fitted on FIT, balanced or external data."""
    cfg = cfg or C.CFG
    if fold_label is not None and not (np.asarray(fold_label) == "VAL").all():
        raise PermissionError("calibration must be fitted on natural-distribution VAL rows only")
    stays = store.ids[rows]; u = np.unique(stays)
    perm = np.random.default_rng(cfg.SEED + 5).permutation(len(u)); fold_of = dict(zip(u[perm].tolist(), (np.arange(len(u)) % cfg.CAL_FOLDS).tolist()))
    fold = np.array([fold_of[s] for s in stays.tolist()]); recs, rel = [], {}
    for k, h in enumerate(C.HORIZONS):
        v = store.V[rows, k]; z = logits[v, k].astype(np.float64); y = store.Y[rows, k][v].astype(np.float64); f = fold[v]
        for m in METHODS:
            p_oof = np.empty_like(z)
            for fo in range(cfg.CAL_FOLDS):
                tr, te = f != fo, f == fo
                p_oof[te] = apply_cal_1d(m, fit_cal(m, z[tr], y[tr]), z[te])
            rep, r = cal_report(y, p_oof, cfg.CAL_BINS); rel[(m, h)] = r; recs.append(dict(horizon=f"t+{h}", method=m, **rep))
    chosen = cfg.CAL_METHOD
    if chosen == "auto":
        means = {m: np.mean([r["logloss"] for r in recs if r["method"] == m]) for m in METHODS if m != "raw"}; chosen = min(means, key=means.get)
    cal = dict(method=chosen, per_h=[], horizons=list(C.HORIZONS), fitted_on="VAL (all stays)")
    for k, h in enumerate(C.HORIZONS):
        v = store.V[rows, k]; cal["per_h"].append(fit_cal(chosen, logits[v, k], store.Y[rows, k][v]))
    return cal, recs, rel


def apply_calibrator(cal, logits):
    lg = np.asarray(logits, np.float64); out = np.empty_like(lg)
    for k in range(lg.shape[1]): out[:, k] = apply_cal_1d(cal["method"], cal["per_h"][k], lg[:, k])
    return out
