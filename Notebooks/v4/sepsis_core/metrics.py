"""Metrics. Everything here returns aggregates only. Unit of inference for every interval is the stay (stay-level bootstrap)."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from . import config as C

ICU_HOUR_BUCKETS = [(1, 6), (7, 12), (13, 24), (25, 48), (49, 168), (169, 10 ** 9)]


def compute_metrics(store, rows, scores, horizons=None):
    horizons = horizons or C.HORIZONS
    y, v = store.Y[rows], store.V[rows]; out = {}
    for k, h in enumerate(horizons):
        m = v[:, k]; yy = y[m, k]; pp = scores[m, k]
        ok = yy.size > 0 and yy.min() != yy.max()
        out[f"auroc_t{h}"] = float(roc_auc_score(yy, pp)) if ok else float("nan")
        out[f"auprc_t{h}"] = float(average_precision_score(yy, pp)) if ok else float("nan")
        out[f"pos_rate_t{h}"] = float(yy.mean()) if yy.size else float("nan")
        out[f"auprc_lift_t{h}"] = out[f"auprc_t{h}"] / max(out[f"pos_rate_t{h}"], 1e-12) if ok else float("nan")
    out["auroc_mean"] = float(np.nanmean([out[f"auroc_t{h}"] for h in horizons])) if any(np.isfinite(out[f"auroc_t{h}"]) for h in horizons) else float("nan")
    out["auprc_mean"] = float(np.nanmean([out[f"auprc_t{h}"] for h in horizons])) if any(np.isfinite(out[f"auprc_t{h}"]) for h in horizons) else float("nan")
    return out


def row_onset(store, stay_onset_series):
    """Onset hour broadcast to rows (NaN for non-septic stays)."""
    import pandas as pd
    return np.repeat(stay_onset_series.reindex(pd.Index(store.ids[store.starts].astype(np.int64))).to_numpy(np.float64), store.lens)


def v3_view_mask(store, rows, hours=None):
    """True for rows v3 would have scored: the most recent `V3_VIEW_HOURS` rows of each stay (the bridge to the old numbers)."""
    hours = hours or C.CFG.V3_VIEW_HOURS
    stay_end = np.repeat(store.starts + store.lens, store.lens)            # row index one past each row's stay end
    return (stay_end[rows] - rows) <= hours


def subgroup_masks(store, rows, onset_rows, cfg=None):
    cfg = cfg or C.CFG
    early = onset_rows[rows] < cfg.EARLY_ONSET_HOURS                         # NaN (non-septic) -> False
    return {"late_rows": store.hours[rows] > cfg.EARLY_ONSET_HOURS, "no_early_onset": ~early,
            "early_onset_stays": early, "late_onset_or_nonseptic": ~early, "v3_view": v3_view_mask(store, rows)}


def by_icu_hour(store, rows, scores, onset_rows=None, views=("all",), cfg=None):
    """AUROC / AUPRC per ICU-hour bucket: the 'how early can it tell' table. views: 'all', 'no_early_onset' (stays with onset < 12 h removed),
    'early_onset' / 'late_onset' (septic stays of that kind against ALL non-septic stays).
    Hours 1-6 of early-onset stays are recognition of sepsis already under way, not prediction: report the views side by side."""
    cfg = cfg or C.CFG; out = []
    hrs = store.hours[rows]
    onset = onset_rows[rows] if onset_rows is not None else np.full(len(rows), np.nan)
    septic = np.isfinite(onset); early = septic & (onset < cfg.EARLY_ONSET_HOURS)
    sel = {"all": np.ones(len(rows), bool), "no_early_onset": ~early, "early_onset": early | ~septic, "late_onset": (septic & ~early) | ~septic}
    for view in views:
        for lo, hi in ICU_HOUR_BUCKETS:
            m = sel[view] & (hrs >= lo) & (hrs <= hi)
            met = compute_metrics(store, rows[m], scores[m]) if m.sum() else {}
            out.append(dict(view=view, bucket=f"{lo}-{hi if hi < 10 ** 8 else 'end'}", n_rows=int(m.sum()), **{k: met.get(k, float("nan")) for k in
                       [f"auroc_t{h}" for h in C.HORIZONS] + [f"auprc_t{h}" for h in C.HORIZONS] + [f"pos_rate_t{h}" for h in C.HORIZONS]}))
    return out


# ---------------------------------------------------------------- stay-level bootstrap (v3's fast weighted AUROC) ----------------------------------------------------------------
def auc_prep(y, s):
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    grp = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])])
    return o, grp, y[o].astype(bool)


def auc_weighted(prep, w):
    o, grp, ys = prep; ws = w[o]
    gn = np.bincount(grp, weights=ws * (~ys)); gp = np.bincount(grp, weights=ws * ys)
    below = np.cumsum(gn) - gn; den = gp.sum() * gn.sum()
    return float((gp * (below + 0.5 * gn)).sum() / den) if den > 0 else float("nan")


def _parts(store, rows, scores_list):
    y, v = store.Y[rows], store.V[rows]
    _, inv = np.unique(store.ids[rows], return_inverse=True); ns = int(inv.max()) + 1
    parts = []
    for k in range(y.shape[1]):
        m = v[:, k]; parts.append((inv[m], [auc_prep(y[m, k], s[m, k]) for s in scores_list]))
    return parts, ns


def stay_bootstrap_auroc(store, rows, scores, n_boot=300, seed=0, per_horizon=False):
    """AUROC with a stay-level percentile bootstrap CI. -> (point, lo, hi) for the mean over horizons, or a list of such per horizon."""
    parts, ns = _parts(store, rows, [scores]); rng = np.random.default_rng(seed)
    def one(ws): return [auc_weighted(p[0], ws[ii]) for ii, p in parts]
    point = one(np.ones(ns)); draws = np.array([one(np.bincount(rng.integers(0, ns, ns), minlength=ns).astype(np.float64)) for _ in range(n_boot)]) if n_boot > 0 else None
    if per_horizon:
        return [(point[k],) + (tuple(np.nanpercentile(draws[:, k], [2.5, 97.5])) if draws is not None else (float("nan"),) * 2) for k in range(len(point))]
    pm = float(np.mean(point))
    if draws is None: return pm, float("nan"), float("nan")
    lo, hi = np.nanpercentile(draws.mean(1), [2.5, 97.5]); return pm, float(lo), float(hi)


def stay_bootstrap_dauroc(store, rows, sa, sb, n_boot=300, seed=0, per_horizon=False):
    """Paired, stay-level bootstrap of AUROC(sb) - AUROC(sa) on the same rows. -> (point, lo, hi) (mean over horizons) or per horizon."""
    parts, ns = _parts(store, rows, [sa, sb]); rng = np.random.default_rng(seed)
    def one(ws): return [auc_weighted(p[1], ws[ii]) - auc_weighted(p[0], ws[ii]) for ii, p in parts]
    point = one(np.ones(ns)); draws = np.array([one(np.bincount(rng.integers(0, ns, ns), minlength=ns).astype(np.float64)) for _ in range(n_boot)]) if n_boot > 0 else None
    if per_horizon:
        return [(point[k],) + (tuple(np.nanpercentile(draws[:, k], [2.5, 97.5])) if draws is not None else (float("nan"),) * 2) for k in range(len(point))]
    pm = float(np.mean(point))
    if draws is None: return pm, float("nan"), float("nan")
    lo, hi = np.nanpercentile(draws.mean(1), [2.5, 97.5]); return pm, float(lo), float(hi)


# ---------------------------------------------------------------- the gate (adopted 2026-11-11, Docs/DEVIATIONS.md D1) ----------------------------------------------------------------
def gate(store, rows, model_scores, gbdt_scores, n_boot=300, seed=0, min_lift=3.0):
    """Pass iff, at EVERY horizon, (i) the lower 95% bound of AUROC(model) - AUROC(LightGBM reference) is > 0 on the same rows
    (stay-level paired bootstrap) and (ii) AUPRC >= min_lift x base rate. Returns the per-horizon evidence, not just the verdict."""
    met = compute_metrics(store, rows, model_scores); d = stay_bootstrap_dauroc(store, rows, gbdt_scores, model_scores, n_boot, seed, per_horizon=True)
    per = []
    for k, h in enumerate(C.HORIZONS):
        per.append(dict(horizon=h, d_auroc=d[k][0], d_lo=d[k][1], d_hi=d[k][2], auprc_lift=met[f"auprc_lift_t{h}"],
                        ok_auroc=bool(d[k][1] > 0), ok_lift=bool(met[f"auprc_lift_t{h}"] >= min_lift)))
    return dict(passed=bool(all(p["ok_auroc"] and p["ok_lift"] for p in per)), per_horizon=per)


# ---------------------------------------------------------------- threshold metrics (lecturer's table) ----------------------------------------------------------------
def threshold_metrics(store, rows, probs, n_folds=2, seed=0):
    """Sensitivity / specificity / F1 / accuracy at an F1-optimal threshold that is CROSS-FITTED inside VAL by stay (chosen on one half,
    scored on the other), so the numbers are not optimistic. Accuracy is reported because the KA paper reports it; at ~1% prevalence
    'always negative' already scores ~99%, so it carries no information."""
    out = []; stays = store.ids[rows]; u = np.unique(stays)
    fold_of = dict(zip(u[np.random.default_rng(seed).permutation(len(u))].tolist(), (np.arange(len(u)) % n_folds).tolist()))
    fold = np.array([fold_of[s] for s in stays.tolist()])
    for k, h in enumerate(C.HORIZONS):
        v = store.V[rows, k]; y = store.Y[rows, k][v].astype(bool); p = probs[v, k]; f = fold[v]
        tp = fp = tn = fn = 0; thr = []
        for fo in range(n_folds):
            tr, te = f != fo, f == fo
            if y[tr].sum() == 0 or te.sum() == 0: continue
            cand = np.unique(np.quantile(p[tr], np.linspace(0.5, 0.999, 200)))
            f1 = [2 * (y[tr] & (p[tr] >= c)).sum() / max(1, y[tr].sum() + (p[tr] >= c).sum()) for c in cand]
            c = float(cand[int(np.argmax(f1))]); thr.append(c); pred = p[te] >= c
            tp += int((pred & y[te]).sum()); fp += int((pred & ~y[te]).sum()); tn += int((~pred & ~y[te]).sum()); fn += int((~pred & y[te]).sum())
        n = max(1, tp + fp + tn + fn)
        out.append(dict(horizon=h, threshold_mean=float(np.mean(thr)) if thr else float("nan"), sensitivity=tp / max(1, tp + fn), specificity=tn / max(1, tn + fp),
                        precision=tp / max(1, tp + fp), f1=2 * tp / max(1, 2 * tp + fp + fn), accuracy=(tp + tn) / n, accuracy_always_negative=(tn + fp) / n))
    return out
