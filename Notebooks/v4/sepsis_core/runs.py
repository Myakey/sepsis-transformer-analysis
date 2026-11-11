"""High-level helpers shared by the training notebooks (01, 01b, 02). The notebooks stay thin; everything that must be identical
across tracks (how a model is trained, scored, calibrated, registered and reported) lives here.

Every function prints or returns aggregates only. Per-row model scores are written to Drive as .npz, never displayed.
"""
import gc, json, os, time
import numpy as np
import torch

from . import config as C
from . import calibration as CAL
from . import controls as CT
from . import data as D
from . import engine as E
from . import gbdt as GB
from . import metrics as MET
from . import models as M
from . import tracking as T
from .dataset import make_loader

STATE = dict(n_boot=300, n_boot_auprc=100)
CORE_COMMIT = os.environ.get("SEPSIS_CORE_COMMIT", "unknown")


# ---------------------------------------------------------------- setup / paths ----------------------------------------------------------------
def setup(smoke=False, experiment=None, need_data=True):
    """Call once per notebook, right after the bootstrap cell."""
    if smoke: C.CFG.apply_smoke()
    C.CFG.NUM_WORKERS = 0 if E.RT.device.type == "cpu" else 2
    C.require_mounted(need_data=need_data)
    T.setup_tracking(experiment or C.CFG.MLFLOW_EXPERIMENT)
    STATE.update(n_boot=30 if smoke else 300, n_boot_auprc=20 if smoke else 100)
    print(f"run={C.CFG.RUN_NAME} | smoke={C.CFG.SMOKE_TEST} | device={E.RT.device} amp={E.RT.amp} | core commit={CORE_COMMIT[:12]} | "
          f"artifacts={run_root()} | mlflow={'ON ' + str(T.TRACKING['uri']) if T.TRACKING['enabled'] else 'OFF'}")


def run_root(): return os.path.join(C.PATHS.WORK_DIR, C.CFG.RUN_NAME)
def model_dir(name): return os.path.join(run_root(), "models", name)
def report_path(name):
    d = os.path.join(run_root(), "reports"); os.makedirs(d, exist_ok=True); return os.path.join(d, name)
def figure_path(name):
    d = os.path.join(run_root(), "figures"); os.makedirs(d, exist_ok=True); return os.path.join(d, name)
def registry(): return E.Registry(run_root())
def save_json(obj, name):
    p = report_path(name); json.dump(obj, open(p, "w"), indent=1, default=_js); print("saved:", p); return p
def _js(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


# ---------------------------------------------------------------- data ----------------------------------------------------------------
def make_bundle(df, pre, static_real, stats, onset, tag, split_sha="n/a", use_cache=True):
    """df (with a 'fold' column holding FIT / VAL) -> the bundle every helper below takes."""
    folds = sorted(df["fold"].unique())
    key = D.store_cache_key(split_sha, pre, stats, folds)
    path = os.path.join(C.PATHS.WORK_DIR, "cache", f"store_{tag}_{key}.npz") if use_cache else None
    store, cached = D.build_store(df, pre, path)
    fit_idx = np.flatnonzero(np.asarray(store.stay_fold) == "FIT"); val_idx = np.flatnonzero(np.asarray(store.stay_fold) == "VAL")
    pr, alpha, pw = D.prevalence_alpha(store, fit_idx)
    b = dict(store=store, pre=pre, fit_idx=fit_idx, val_idx=val_idx, onset=onset, onset_rows=MET.row_onset(store, onset), alpha=alpha, pw=pw,
             prevalence=pr, data_hash=key, stats=stats, split_sha=split_sha, static_real=static_real, tag=tag)
    print(f"[{tag}] stays FIT={len(fit_idx):,} VAL={len(val_idx):,} | rows={store.n_rows:,} | features={len(pre.feat)} | hour-gap violations={store.gap_violations:,} | "
          f"FIT prevalence/horizon={[round(p, 4) for p in pr]} | focal alpha(prevalence)={alpha:.4f} | data hash={key} | store {'loaded from cache' if cached else 'built'}")
    return b


def prepare_mimic(features=None, tag="full36", stage="train", use_cache=True, keep_df=False):
    """MIMIC-IV FIT/VAL through the real path: label fix -> FROZEN split (fingerprint checked) -> fold guard -> FIT-only preprocessing."""
    cfg = C.CFG; feats = list(features or C.DECAY_FEATURES)
    df, static_real, stats = D.load_hourly(features=feats)
    onset = D.stay_onset(df)
    split, sha = D.load_frozen_split()
    df, counts = D.attach_folds(df, split, ("FIT", "VAL"), D.FoldGuard(stage))
    print(f"[{tag}] cohort: rows={stats['rows']:,} stays={stats['stays']:,} patients={stats['patients']:,} pos_rate={stats['pos_rate']:.4%} | split sha1={sha} | "
          f"septic stays={int(onset.notna().sum()):,} early-onset={(onset < cfg.EARLY_ONSET_HOURS).sum() / max(1, onset.notna().sum()):.1%}")
    if cfg.SMOKE_TEST: df = D.smoke_subsample(df)
    pre = D.Preprocessor(features=feats, static_cols=static_real).fit(df, (df["fold"] == "FIT").to_numpy(), fold=df["fold"].to_numpy())
    b = make_bundle(df, pre, static_real, stats, onset, tag, sha, use_cache)
    if keep_df: b["df"] = df
    else: del df; gc.collect()
    return b


def stay_of_rows(store, rows): return np.searchsorted(store.starts, rows, side="right") - 1


# ---------------------------------------------------------------- models ----------------------------------------------------------------
def rebuild(hp, store):
    """The single model factory: our Transformer in any arm (optionally with LayerScale) or the KA-Transformer."""
    nF = store.X.shape[1]
    if hp.get("arm") == "ka":
        return M.build_ka(nF, use_iculos=bool(hp.get("use_iculos", False)), use_mask=bool(hp.get("ka_use_mask", True)), **(hp.get("ka") or {}))
    m = M.build_model(hp, store.x_mean)
    if float(hp.get("layerscale", 0.0)) > 0: CT.add_layerscale(m, float(hp["layerscale"]))
    return m


def predict_store(model, store, idx=None, mode="eval", max_len=None):
    """Full-coverage scoring. -> rows (ascending), logits (n,3). Every model scored on the same stays returns the same `rows`,
    which is what makes the paired stay-level bootstrap valid."""
    idx = np.arange(len(store.starts)) if idx is None else idx
    loader, _ = make_loader(store, idx, C.CFG.EVAL_BATCH, False, max_len=max_len, mode=mode, pin=E.RT.device.type == "cuda")
    rows, lg, _ = E.predict_loader(model, loader, as_logits=True); o = np.argsort(rows, kind="stable")
    return rows[o], lg[o]


def train_model(bundle, name, hp, *, seed, role, store=None, fit_idx=None, max_epochs=None, patience=None, fixed_epochs=False,
                stop_after_epochs=None, soup_k=None, wrapper=None, force=False):
    """Train (or resume, or skip if finished) one model; score VAL with the soup and with the best single epoch; register it.
    `store` overrides the TRAINING labels only (shuffled-label controls); VAL scoring always uses the bundle's real labels."""
    cfg = C.CFG; mdir = model_dir(name); done = os.path.join(mdir, "done.json")
    if os.path.exists(done) and not force:
        rec = json.load(open(done)); print(f"[{name}] finished earlier -> VAL AUROC {rec['metrics']['auroc_mean']:.4f} (soup), {rec['metrics_single']['auroc_mean']:.4f} (single epoch)")
        return rec
    hp = dict(hp)
    if cfg.SMOKE_TEST:
        if hp.get("arm") == "ka": hp.update(ka=dict(d_model=64, num_heads=8, num_layers=2, ffn_dim=128), batch_size=16)
        else: hp.update(C.SMOKE_HP_OVERRIDE)
    st = store if store is not None else bundle["store"]; fi = bundle["fit_idx"] if fit_idx is None else fit_idx
    with T.safe_run(name, tags=dict(role=role, arm=hp.get("arm", "softmax"), seed=seed, run=cfg.RUN_NAME, data=bundle["tag"], core_commit=CORE_COMMIT)):
        T.m_params({**{k: v for k, v in hp.items() if k != "ka"}, "seed": seed, "role": role, "data_hash": bundle["data_hash"], "core_commit": CORE_COMMIT,
                    "max_seq_len": cfg.MAX_SEQ_LEN, "fit_stays": len(fi), "val_stays": len(bundle["val_idx"])})
        res = E.fit(hp, st, fi, bundle["val_idx"], max_epochs=max_epochs or cfg.FINAL_MAX_EPOCHS, patience=patience or cfg.FINAL_PATIENCE, seed=seed,
                    alpha=bundle["alpha"], pw=bundle["pw"], pre=bundle["pre"], ckpt_dir=mdir, resume=True, log_prefix=f"[{name}] ", soup_k=soup_k,
                    fixed_epochs=fixed_epochs, stop_after_epochs=stop_after_epochs, train_dataset_wrapper=wrapper, model_factory=rebuild)
        rows, lg = predict_store(res["model"], bundle["store"], bundle["val_idx"])
        res["model"].load_state_dict(res["single_state"]); rows_s, lg_s = predict_store(res["model"], bundle["store"], bundle["val_idx"])
        assert np.array_equal(rows, rows_s)
        met = MET.compute_metrics(bundle["store"], rows, lg); met_s = MET.compute_metrics(bundle["store"], rows, lg_s)
        np.savez_compressed(os.path.join(mdir, "val_scores.npz"), rows=rows, logits=lg.astype(np.float32), logits_single=lg_s.astype(np.float32))
        hist = res["history"]
        rec = dict(name=name, arm=res["hp"].get("arm", "softmax"), seed=int(seed), role=role, hp=res["hp"], metrics=met, metrics_single=met_s,
                   best_epoch=int(res["best"]["epoch"]), soup_epochs=[int(e) for e in res["soup_epochs"]], epochs_run=len(hist), n_params=int(res["n_params"]),
                   mass_per_epoch=[h.get("mass_mean") for h in hist], epoch_sec=float(np.mean([h["epoch_sec"] for h in hist])) if hist else None,
                   dir=mdir, path=os.path.join(mdir, "best.pt"), data_hash=bundle["data_hash"], core_commit=CORE_COMMIT)
        T.m_metrics({**{f"val_{k}": v for k, v in met.items()}, **{f"val_single_{k}": v for k, v in met_s.items()}, "epochs_run": len(hist)})
    registry().register(name, arm=rec["arm"], seed=seed, role=role, data_hash=bundle["data_hash"], metrics=met, path=rec["path"], hp=res["hp"],
                        extra=dict(metrics_single=met_s, core_commit=CORE_COMMIT, epochs_run=len(hist), soup_epochs=rec["soup_epochs"]))
    json.dump(rec, open(done, "w"), indent=1, default=_js)
    print(f"[{name}] VAL AUROC soup {met['auroc_mean']:.4f} | single epoch {met_s['auroc_mean']:.4f} | epochs {len(hist)} | soup of {rec['soup_epochs']}")
    del res; gc.collect()
    if E.RT.device.type == "cuda": torch.cuda.empty_cache()
    return rec


def val_scores(name, which="soup"):
    z = np.load(os.path.join(model_dir(name), "val_scores.npz")); return z["rows"], z["logits" if which == "soup" else "logits_single"]


def load_trained(name, store, which="best.pt"):
    """Rebuild a trained model (any family) and load its weights. `which`: best.pt (soup), best_epoch.pt (single) or init.pt (step 0)."""
    ck = torch.load(os.path.join(model_dir(name), which), map_location="cpu", weights_only=False)
    m = rebuild(ck["hp"], store); m.load_state_dict(ck["model"]); return m.to(E.RT.device).eval()


# ---------------------------------------------------------------- evaluation ----------------------------------------------------------------
def stay_bootstrap_auprc(store, rows, scores, n_boot=None, seed=0):
    """AUPRC per horizon with a stay-level percentile bootstrap CI. -> [(point, lo, hi)] per horizon."""
    from sklearn.metrics import average_precision_score
    n_boot = STATE["n_boot_auprc"] if n_boot is None else n_boot
    _, inv = np.unique(store.ids[rows], return_inverse=True); ns = int(inv.max()) + 1; rng = np.random.default_rng(seed); out = []
    for k in range(store.Y.shape[1]):
        m = store.V[rows, k].astype(bool); y = store.Y[rows, k][m].astype(np.int8); s = scores[m, k]; ii = inv[m]
        if y.size == 0 or y.min() == y.max(): out.append((float("nan"),) * 3); continue
        pt = float(average_precision_score(y, s)); draws = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, ns, ns), minlength=ns)[ii].astype(np.float64)
            if (w * y).sum() > 0: draws.append(average_precision_score(y, s, sample_weight=w))
        lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws) > 5 else (float("nan"), float("nan")))
        out.append((pt, float(lo), float(hi)))
    return out


def calibrate_val(bundle, rows, logits):
    """Calibrate on natural-distribution VAL (cross-fitted). -> (calibrator, calibrated probs, out-of-fold report of the chosen method)."""
    st = bundle["store"]; fl = np.asarray(st.stay_fold)[stay_of_rows(st, rows)]
    cal, recs, _ = CAL.calibrate(st, rows, logits, fold_label=fl)
    return cal, CAL.apply_calibrator(cal, logits), [r for r in recs if r["method"] == cal["method"]], [r for r in recs if r["method"] == "raw"]


def report(store, rows, scores, onset_rows, probs=None, hours=True, mask=None):
    """The standard evaluation block (aggregates only): AUROC/AUPRC per horizon with stay-level CIs, lift, strata, ICU-hour buckets,
    and (if calibrated `probs` are given) threshold metrics and calibration-in-the-large / slope / ECE of those probabilities."""
    if mask is not None: rows, scores, probs = rows[mask], scores[mask], (probs[mask] if probs is not None else None)
    met = MET.compute_metrics(store, rows, scores); nb = STATE["n_boot"]
    ci = MET.stay_bootstrap_auroc(store, rows, scores, nb, per_horizon=True); mean_ci = MET.stay_bootstrap_auroc(store, rows, scores, nb)
    ap = stay_bootstrap_auprc(store, rows, scores)
    out = dict(n_rows=int(len(rows)), n_stays=int(len(np.unique(store.ids[rows]))), metrics=met, auroc_mean_ci=list(mean_ci),
               auroc_ci={f"t{h}": list(ci[k]) for k, h in enumerate(C.HORIZONS)}, auprc_ci={f"t{h}": list(ap[k]) for k, h in enumerate(C.HORIZONS)}, strata={})
    for nm, m in MET.subgroup_masks(store, rows, onset_rows).items():
        if m.sum() > 0:
            sm = MET.compute_metrics(store, rows[m], scores[m]); out["strata"][nm] = dict(n_rows=int(m.sum()), auroc_mean=sm["auroc_mean"], auprc_mean=sm["auprc_mean"])
    if hours: out["by_icu_hour"] = MET.by_icu_hour(store, rows, scores, onset_rows, views=("all", "no_early_onset", "early_onset", "late_onset"))
    if probs is not None:
        out["thresholds"] = MET.threshold_metrics(store, rows, probs); out["calibration"] = cal_block(store, rows, probs)
    return out


def cal_block(store, rows, probs):
    """Calibration of a given set of probabilities, per horizon: slope, calibration-in-the-large, ECE, Brier, mean predicted vs observed."""
    out = []
    for k, h in enumerate(C.HORIZONS):
        v = store.V[rows, k].astype(bool); y = store.Y[rows, k][v]
        if y.size and y.min() != y.max():
            rep, _ = CAL.cal_report(y, np.clip(probs[v, k], 1e-6, 1 - 1e-6), C.CFG.CAL_BINS)
            out.append(dict(horizon=h, **{kk: rep[kk] for kk in ("cal_slope", "cal_in_the_large", "ece", "brier", "mean_pred", "obs_rate")}))
    return out


def evaluate(bundle, rows, logits, mask=None):
    """VAL evaluation of one model: calibrate on VAL (cross-fitted), then the standard report on calibrated probabilities."""
    cal, probs, oof, raw = calibrate_val(bundle, rows, logits)
    rep = report(bundle["store"], rows, logits, bundle["onset_rows"], probs=probs, mask=mask)
    rep["calibrator"] = cal; rep["calibration_oof"] = oof; rep["calibration_raw"] = raw
    return rep


def paired(store, rows, a, b):
    """AUROC(b) - AUROC(a), stay-level paired bootstrap: overall (mean of horizons) and per horizon."""
    nb = STATE["n_boot"]; d = MET.stay_bootstrap_dauroc(store, rows, a, b, nb); per = MET.stay_bootstrap_dauroc(store, rows, a, b, nb, per_horizon=True)
    return dict(d=d[0], lo=d[1], hi=d[2], per_horizon={f"t{h}": list(per[k]) for k, h in enumerate(C.HORIZONS)},
                verdict=("b better" if d[1] > 0 else "a better" if d[2] < 0 else "no clear difference"))


def line(name, rep):
    m = rep["metrics"]; ci = rep["auroc_mean_ci"]; cal = rep.get("calibration") or []
    s = (f"{name:<28s} AUROC {m['auroc_mean']:.4f} [{ci[1]:.4f}, {ci[2]:.4f}] (t1/t3/t6 {m['auroc_t1']:.3f}/{m['auroc_t3']:.3f}/{m['auroc_t6']:.3f}) | "
         f"AUPRC lift x{np.mean([m[f'auprc_lift_t{h}'] for h in C.HORIZONS]):.1f}")
    st = rep.get("strata", {})
    if "late_rows" in st: s += f" | hour>12: {st['late_rows']['auroc_mean']:.4f} | no early-onset: {st['no_early_onset']['auroc_mean']:.4f} | v3 view: {st['v3_view']['auroc_mean']:.4f}"
    if cal: s += f" | cal slope {np.mean([c['cal_slope'] for c in cal]):.2f} ECE {np.mean([c['ece'] for c in cal]):.4f}"
    return s


def seed_summary(recs, key="metrics"):
    """Mean, SD and range of VAL AUROC over seeds. A difference smaller than the seed SD is 'not resolved'."""
    v = np.array([r[key]["auroc_mean"] for r in recs], float)
    return dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan"), min=float(v.min()), max=float(v.max()), values=[float(x) for x in v])


# ---------------------------------------------------------------- external cohorts ----------------------------------------------------------------
def recal_external(store, rows, logits, fraction=0.1, seed=0):
    """SECONDARY result, labelled as such: Platt recalibration fitted on a small random share of the external stays, scored on the rest.
    The primary external numbers never use external data for normalisation, calibration or thresholds."""
    stays = store.ids[rows]; u = np.unique(stays); rng = np.random.default_rng(seed)
    fit_set = set(rng.choice(u, size=max(2, int(fraction * len(u))), replace=False).tolist()); is_fit = np.array([s in fit_set for s in stays.tolist()])
    probs = np.full(logits.shape, np.nan, np.float64)
    for k in range(logits.shape[1]):
        v = store.V[rows, k].astype(bool); tr = v & is_fit; y = store.Y[rows, k][tr]
        if y.size == 0 or y.min() == y.max(): return dict(skipped="no positives in the external recalibration split")
        probs[:, k] = CAL.apply_cal_1d("platt", CAL.fit_cal("platt", logits[tr, k], y), logits[:, k])
    te = ~is_fit
    return dict(fraction=fraction, n_stays_fit=len(fit_set), calibration=cal_block(store, rows[te], probs[te]))


def external_report(src_bundle, name, store_t, onset_rows_t, mask_t=None, gb=None, hours=True):
    """Score a model trained on `src_bundle` on another cohort's store (already transformed with the SOURCE preprocessor, i.e. source
    FIT statistics applied as they are). Returns the report on all stays and on the matched stratum, the calibration of raw and of
    source-calibrated probabilities, and the labelled secondary external recalibration."""
    if gb is not None:
        rows, p = GB.predict(gb, store_t, src_bundle["pre"], np.arange(len(store_t.starts))); lg = GB.to_logit(p)
        rows_v, lg_v = gb["rows"], GB.to_logit(gb["scores"])
    else:
        model = load_trained(name, src_bundle["store"]); rows, lg = predict_store(model, store_t); del model
        rows_v, lg_v = val_scores(name)
    cal, _, _, _ = calibrate_val(src_bundle, rows_v, lg_v); probs = CAL.apply_calibrator(cal, lg)
    out = dict(name=name, source_calibrator=cal["method"], all=report(store_t, rows, lg, onset_rows_t, probs=probs, hours=False))
    if mask_t is not None: out["matched"] = report(store_t, rows, lg, onset_rows_t, probs=probs, mask=mask_t[rows], hours=hours)
    out["calibration_raw"] = cal_block(store_t, rows, 1.0 / (1.0 + np.exp(-lg.astype(np.float64))))
    out["calibration_recal_external_SECONDARY"] = recal_external(store_t, rows, lg)
    if E.RT.device.type == "cuda": torch.cuda.empty_cache()
    return out


def internal_report(bundle, name, mask=None, gb=None):
    """The same model on its own VAL fold (optionally restricted to the matched stratum): the 'internal' side of the external drop."""
    rows, lg = (gb["rows"], GB.to_logit(gb["scores"])) if gb is not None else val_scores(name)
    return report(bundle["store"], rows, lg, bundle["onset_rows"], mask=(mask[rows] if mask is not None else None), hours=False)
