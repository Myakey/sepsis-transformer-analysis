"""Correlational battery, PDF points P1-P4 (Docs/PREREGISTRATION.md section 5 and 6). One pipeline, run unchanged on every model.

    stage 1  per stay and head: DV_tj = b0 + b1*severity_j + b2*log(t-j+1) + b3*log(t+1) + b4*density_j + b5*deviation_j  (v3 design)
    stage 2  DerSimonian-Laird random-effects pooling of b1 across stays (v3 estimator: the primary decision rule)
    band     95th percentile of |pooled b1| over all heads of five control sets; threshold = max(band, SESOI)
    heads    pass on DISCOVERY, confirm on SCREEN (v3 rule); seed rule on counts (section 5.6)

Everything that decides something is ported from v3 notebook cell 13 without change (same random streams, so the same (t, j) pairs).
Additions are reported beside it: cluster-robust stage-1 errors, prediction intervals, total mass, knockouts, sensitivity families.
Governance: nothing here prints or returns row-level patient data; per-stay coefficients stay in memory or in Drive artifacts.
"""
import os, json
import numpy as np
import pandas as pd
import torch
from scipy import stats as sps
from scipy.special import logsumexp

from . import config as C
from . import data as D
from . import attention as A
from . import metrics as MET
from .dataset import make_loader
from .engine import to_dev
from .models import attention_modules

COV_NAMES = ["severity", "log_recency", "log_pool", "density", "deviation_resid"]
DVS = ("clr", "loga", "normw")          # clr = row-centred logit, computed the same way in every arm: the cross-arm quantity of notebook 03
PRIMARY_DV = {"softmax": "clr", "kernel": "clr", "sigmoid": "loga", "kernel_raw": "loga"}
BAND_SURROGATES = ("circ_shift", "donor", "rand_proj")
ALT_SEVERITY = ("SOFA_resp", "SOFA_coag", "SOFA_liver", "SOFA_cv", "SOFA_cns", "SOFA_renal", "SOFA_delta")


# ---------------------------------------------------------------- stays, severity, splits ----------------------------------------------------------------
def window(store, i, cfg=None):
    """(first row, length) of the hours a stay contributes: its most recent MAX_SEQ_LEN hours (v3's `win`)."""
    cfg = cfg or C.CFG; l = int(store.lens[i]); n = min(l, cfg.MAX_SEQ_LEN); return int(store.starts[i]) + l - n, n


def load_severity(store, stay_idx, columns=None, path=None, cfg=None):
    """ANALYSIS-ONLY covariates (never model inputs) for the rows of the given stays, built exactly as in v3 cell 13: forward-filled within
    stay, then the stay median, then 0. -> ({column: float32 array over all store rows, NaN outside the requested stays}, {stay_id: subject_id})."""
    import pyarrow.parquet as pq
    cfg = cfg or C.CFG; path = path or C.PATHS.DATA_PATH; names = pq.read_schema(path).names; ID, HR = cfg.ID_COL, cfg.HOUR_COL
    have = []
    for c in list(columns or [cfg.CORR_SEVERITY_COL]):
        try: have.append(D._resolve(c, names))
        except KeyError: print(f"[severity] column '{c}' is not in the parquet -> skipped")
    if not have: raise KeyError(f"none of the severity columns {columns} is in the parquet; SOFA-like columns present: {[c for c in names if 'SOFA' in c.upper()]}")
    stay_idx = np.asarray(stay_idx); ids = np.unique(store.ids[store.starts[stay_idx]])
    aux = pd.read_parquet(path, columns=[ID, cfg.SUBJECT_COL, HR, "HoursToOnset"] + have, filters=[(ID, "in", [int(i) for i in ids])])
    aux = aux[aux["HoursToOnset"].isna() | (aux["HoursToOnset"] >= 0)].copy()
    aux[HR] = aux[HR].astype(np.int64)
    aux = aux.sort_values([ID, HR]).drop_duplicates([ID, HR])
    rows = np.concatenate([np.arange(store.starts[i], store.starts[i] + store.lens[i]) for i in stay_idx])
    key = pd.DataFrame({ID: store.ids[rows], HR: store.hours[rows].astype(np.int64), "row": rows})
    out = {}
    for c in have:
        aux["_sev"] = aux.groupby(ID)[c].ffill()
        aux["_sev"] = aux["_sev"].fillna(aux.groupby(ID)["_sev"].transform("median")).fillna(0.0)
        mrg = key.merge(aux[[ID, HR, "_sev"]], on=[ID, HR], how="left")
        arr = np.full(store.n_rows, np.nan, np.float32); arr[mrg["row"].to_numpy()] = mrg["_sev"].to_numpy(np.float32); out[c] = arr
    return out, {int(k): int(v) for k, v in aux.groupby(ID)[cfg.SUBJECT_COL].first().to_dict().items()}


def eligible(store, sev, i, cfg=None):
    cfg = cfg or C.CFG; a, n = window(store, i, cfg); s = sev[a:a + n]
    return bool(n >= cfg.CORR_MIN_LEN and np.isfinite(s).all() and np.std(s) > 1e-6)


def v3_splits(store, path=None):
    """DISCOVERY / SCREEN exactly as v3 drew them: the stay IDs are LOADED from v3's correlational_summary.json, never re-drawn.
    -> ({'DISCOVERY': sorted stay indices, 'SCREEN': ...}, info). Stops if the file or any listed stay is missing."""
    cands = [path or C.PATHS.V3_CORR_SUMMARY, os.path.join(C.PATHS.V3_FINAL_DIR, "correlation", "correlational_summary.json"),
             os.path.join(C.PATHS.V3_FINAL_DIR, "correlational_summary.json")]
    src = next((p for p in cands if p and os.path.exists(p)), None)
    if src is None: raise FileNotFoundError(f"v3's correlational_summary.json was not found in any of {cands}. It holds the pre-registered DISCOVERY / SCREEN stay IDs; "
                                            "they are never re-drawn. Edit Paths.V3_CORR_SUMMARY in sepsis_core/config.py if the file moved.")
    js = json.load(open(src))
    if "discovery_stays" not in js or "screen_stays" not in js: raise KeyError(f"{src} has no 'discovery_stays' / 'screen_stays'")
    id2idx = {int(s): i for i, s in enumerate(store.ids[store.starts])}; out = {}
    for nm, k in (("DISCOVERY", "discovery_stays"), ("SCREEN", "screen_stays")):
        miss = [s for s in js[k] if int(s) not in id2idx]
        if miss: raise RuntimeError(f"{len(miss)} of v3's {len(js[k])} {nm} stays are not in this store (FIT/VAL of the frozen split). The split or the cohort differs from v3.")
        out[nm] = np.sort(np.array([id2idx[int(s)] for s in js[k]], np.int64))
        if hasattr(store, "stay_fold") and not (np.asarray(store.stay_fold)[out[nm]] == "VAL").all(): raise RuntimeError(f"a v3 {nm} stay is not in VAL")
    if set(out["DISCOVERY"].tolist()) & set(out["SCREEN"].tolist()): raise RuntimeError("DISCOVERY and SCREEN overlap")
    return out, dict(source=src, n_disc=len(out["DISCOVERY"]), n_screen=len(out["SCREEN"]), v3_band_clr=js.get("band_clr"), v3_band_normw=js.get("band_normw"))


def draw_splits(store, sev, pool_idx, subj_of_stay, n=None, seed=None, cfg=None):
    """v3's own drawing procedure (cell 13, section 2). Used ONLY where no v3 list exists: smoke runs and the exploratory PhysioNet screen."""
    cfg = cfg or C.CFG; n = n or cfg.CORR_N_STAYS; rng = np.random.default_rng(cfg.SEED + 13 if seed is None else seed)
    elig = np.array([int(i) for i in pool_idx if eligible(store, sev, int(i), cfg)], np.int64)
    subj = np.array([subj_of_stay.get(int(store.ids[store.starts[i]]), -1) for i in elig])
    us = rng.permutation(np.unique(subj)); disc_subj = set(us[: len(us) // 2].tolist()); is_disc = np.array([s in disc_subj for s in subj], bool)
    out = {}
    for nm, msk in (("DISCOVERY", is_disc), ("SCREEN", ~is_disc)):
        pool = elig[msk]; out[nm] = np.sort(rng.choice(pool, size=min(len(pool), n), replace=False)) if len(pool) else pool
    if min(len(v) for v in out.values()) < 3: raise RuntimeError(f"too few eligible stays for the two-stage regression: {[len(v) for v in out.values()]}")
    return out, dict(source="drawn (v3 procedure)", n_eligible=int(len(elig)), n_disc=len(out["DISCOVERY"]), n_screen=len(out["SCREEN"]))


def partial_sofa(store, pre):
    """EXPLORATORY severity proxy for cohorts without SOFA columns (PhysioNet): the four SOFA components that can be derived from shared
    inputs, with the thresholds of the SOFA table in Singer et al. 2016 (cardiovascular: MAP < 70 only, no vasopressor data).
    Uses the forward-filled value; a component is 0 until its input has been measured. The same code runs on both cohorts."""
    feats = list(pre.feat); mu = np.asarray(pre.state["feat_mean"], np.float64); sd = np.asarray(pre.state["feat_std"], np.float64)
    def raw(name):
        j = feats.index(name); cs = np.cumsum(store.M[:, j].astype(np.int64)); before = np.repeat(np.r_[0, cs][store.starts], store.lens)
        return np.where(cs - before > 0, store.XL[:, j].astype(np.float64) * sd[j] + mu[j], np.nan)
    mp, pl, bi, cr = raw("MAP"), raw("Platelets"), raw("Bilirubin_total"), raw("Creatinine")
    with np.errstate(invalid="ignore"):
        s = (mp < 70).astype(np.float32)
        s += (pl < 150).astype(np.float32) + (pl < 100) + (pl < 50) + (pl < 20)
        s += (bi >= 1.2).astype(np.float32) + (bi >= 2.0) + (bi >= 6.0) + (bi >= 12.0)
        s += (cr >= 1.2).astype(np.float32) + (cr >= 2.0) + (cr >= 3.5) + (cr >= 5.0)
    return s.astype(np.float32)


# ---------------------------------------------------------------- stage 1 (v3 cell 13, sections 3-4) ----------------------------------------------------------------
def _resid(y, x):
    Am = np.c_[np.ones_like(x), x]; return y - Am @ np.linalg.lstsq(Am, y, rcond=None)[0]


def stay_info(store, sev, i, donor_pool, w_rp, core_j, cfg=None, wlen=None):
    """Per-stay covariates, severity controls and the analysed (t, j) pairs. Same random stream as v3, so the same pairs for every model and arm."""
    cfg = cfg or C.CFG; a, n = window(store, i, cfg); sid = int(store.ids[store.starts[i]])
    r = np.random.default_rng((cfg.SEED * 1_000_003 + sid) % (2 ** 32))
    s = sev[a:a + n].astype(np.float64)
    dens = store.M[a:a + n].mean(1).astype(np.float64)
    dev = np.abs(store.XL[a:a + n][:, core_j]).mean(1).astype(np.float64)
    var = {"real": s, "circ_shift": np.roll(s, int(r.integers(n // 4, 3 * n // 4 + 1)))}
    cands = [d for d in donor_pool if d != i and (wlen[d] if wlen is not None else window(store, d, cfg)[1]) >= n]
    if cands:
        d = int(r.choice(cands)); da, dn = window(store, d, cfg); var["donor"] = sev[da + dn - n: da + dn].astype(np.float64)
    rp = store.XL[a:a + n].astype(np.float64) @ w_rp
    var["rand_proj"] = np.sort(s)[np.argsort(np.argsort(rp, kind="stable"), kind="stable")]      # rank-matched to this stay's severity marginal
    ts = np.arange(cfg.CORR_MIN_T, n)
    ts = np.sort(r.choice(ts, size=min(len(ts), cfg.CORR_Q_PER_STAY), replace=False))
    tt = np.repeat(ts, ts + 1); jj = np.concatenate([np.arange(t + 1) for t in ts])
    return dict(a=a, n=n, sid=sid, var=var, dens=dens, dev=dev, ts=ts, tt=tt, jj=jj)


def design(inf, variant):
    """variant = a key of inf['var'], optionally followed by '~rev' (EXPLORATORY: severity residualised on deviation instead of the reverse)."""
    name = variant.split("@")[0]; rev = name.endswith("~rev"); name = name.replace("~rev", "")
    s = inf["var"].get(name)
    if s is None: return None
    tt, jj = inf["tt"], inf["jj"]
    if rev: return np.c_[_resid(s, inf["dev"])[jj], np.log(tt - jj + 1.0), np.log(tt + 1.0), inf["dens"][jj], inf["dev"][jj]]
    devr = _resid(inf["dev"], s)                     # pre-registered rule: deviation residualised on the analysed severity, within stay
    return np.c_[s[jj], np.log(tt - jj + 1.0), np.log(tt + 1.0), inf["dens"][jj], devr[jj]]


def stay_ols(Xc, Y, cluster=None):
    """Xc (P,p) covariates, Y (P,K) DVs, both standardised within stay -> standardised b1 (K,), its OLS SE (K,) [v3], and, if `cluster`
    (P,) is given, a cluster-robust (CR1) SE (K,) for the same coefficient."""
    K = Y.shape[1]; nanv = np.full(K, np.nan)
    mu, sd = Xc.mean(0), Xc.std(0)
    if len(Xc) <= Xc.shape[1] + 2 or sd[0] < 1e-8: return nanv, nanv.copy(), (nanv.copy() if cluster is not None else None)
    Xs = np.where(sd > 1e-8, (Xc - mu) / np.where(sd > 1e-8, sd, 1.0), 0.0)
    ysd = Y.std(0); Ys = (Y - Y.mean(0)) / np.where(ysd > 1e-8, ysd, np.nan)
    Am = np.c_[np.ones(len(Xs)), Xs]; Gi = np.linalg.pinv(Am.T @ Am)
    Ysf = np.nan_to_num(Ys); Bc = Gi @ (Am.T @ Ysf); R = Ysf - Am @ Bc
    dof = max(1, len(Am) - np.linalg.matrix_rank(Am)); se = np.sqrt((R ** 2).sum(0) / dof * Gi[1, 1])
    ok = ysd > 1e-8; b1 = np.where(ok, Bc[1], np.nan); se = np.where(ok, se, np.nan); rob = None
    if cluster is not None:
        g = Gi[1] @ Am.T                                              # the row of (A'A)^-1 A' that produces b1
        u = np.unique(cluster, return_inverse=True)[1]; G = int(u.max()) + 1
        S = np.zeros((G, K)); np.add.at(S, u, g[:, None] * R)         # per-cluster score sums
        corr = (G / max(G - 1, 1)) * ((len(Am) - 1) / max(len(Am) - Am.shape[1], 1))
        rob = np.where(ok, np.sqrt(corr * (S ** 2).sum(0)), np.nan)
    return b1, se, rob


def log_weight(zf, tri, cnt, arm):
    """log attention weight on the lower triangle, from the float64 logits (no floor): softmax / kernel: log-softmax over visible keys;
    sigmoid: log sigmoid(z - log T_t); kernel_raw: z itself (weight = exp(z))."""
    if arm == "sigmoid": lw = -np.logaddexp(0.0, -(zf - np.log(cnt)[:, None]))
    elif arm == "kernel_raw": lw = zf
    else: lw = zf - logsumexp(np.where(tri, zf, -np.inf), axis=-1, keepdims=True)
    return np.where(tri, lw, 0.0)


def dvs_for_stay(z, vn, inf, arm):
    """z (L,h,n,n) pre-normaliser logits, vn (L,h,n) = ||v_j W_O^(h)||. -> {dv: (P, L*h)} at the analysed pairs.
    clr: z_tj - mean_i z_ti (the normaliser cancels; identical computation in every arm).  loga: log weight, uncentred (sigmoid screening DV).
    normw: log(weight * ||v W_O||), row-centred (P2)."""
    L_, nh_, n, _ = z.shape; tt, jj = inf["tt"], inf["jj"]
    tri = np.tril(np.ones((n, n), bool)); cnt = np.arange(1, n + 1, dtype=np.float64)
    zf = np.where(tri, z, 0.0).astype(np.float64); rowmean = zf.sum(-1) / cnt
    clr = zf[:, :, tt, jj] - rowmean[:, :, tt]                                                    # (L,h,P)
    lw = log_weight(zf, tri, cnt, arm)
    lwc = clr if arm in ("softmax", "kernel") else lw[:, :, tt, jj] - (lw.sum(-1) / cnt)[:, :, tt]  # row-centred log weight (= clr under a softmax)
    lv = np.log(np.maximum(vn.astype(np.float64), 1e-12)); cml = np.cumsum(lv, -1) / cnt
    normw = lwc + (lv[:, :, jj] - cml[:, :, tt])
    f = lambda x: x.reshape(L_ * nh_, -1).T
    return {"clr": f(clr), "loga": f(lw[:, :, tt, jj]), "normw": f(normw)}


@torch.no_grad()
def capture_stays(model, store, stay_idx, bs=None, cfg=None):
    """Yield (stay index, z (L,h,n,n), weights (L,h,n,n), vn (L,h,n)) as numpy, one stay at a time. fp32, dropout off."""
    cfg = cfg or C.CFG; loader, _ = make_loader(store, stay_idx, bs or cfg.CORR_CAPTURE_BATCH, False, cfg=cfg, num_workers=0)
    a2i = {window(store, int(i), cfg)[0]: int(i) for i in stay_idx}
    for b in loader:
        b = to_dev(b); cap = A.capture(model, b)
        n_real = (~b["PAD"]).sum(1).tolist(); first = b["ROW"][:, 0].tolist()
        for bi, n in enumerate(n_real):
            yield (a2i[int(first[bi])], cap["Z"][bi, :, :, :n, :n].cpu().numpy(), cap["A"][bi, :, :, :n, :n].cpu().numpy(), cap["VN"][bi, :, :, :n].cpu().numpy())


class Screen:
    """The analysis stays, their covariates and severity controls. Built once; every model is run through the same object."""
    def __init__(self, store, sev, subj_of_stay, splits, features, alt_sev=None, cfg=None, check_eligible=True):
        self.cfg = cfg or C.CFG; self.store, self.splits = store, {k: np.asarray(v, np.int64) for k, v in splits.items()}
        feats = list(features); self.core_j = [feats.index(c) for c in C.ENG_CORE if c in feats]
        self.w_rp = np.random.default_rng(self.cfg.SEED + 99).standard_normal(store.X.shape[1])       # one fixed random projection for every stay (v3)
        wlen = np.minimum(store.lens, self.cfg.MAX_SEQ_LEN)
        if check_eligible:
            bad = [int(i) for nm in self.splits for i in self.splits[nm] if not eligible(store, sev, int(i), self.cfg)]
            if bad: raise RuntimeError(f"{len(bad)} analysis stays are not eligible (>= {self.cfg.CORR_MIN_LEN} h, severity present and varying). "
                                       "With v3's stay list this means the cohort or the severity column differs from v3.")
        self.info = {nm: {int(i): stay_info(store, sev, int(i), self.splits[nm], self.w_rp, self.core_j, self.cfg, wlen) for i in self.splits[nm]} for nm in self.splits}
        for col, arr in (alt_sev or {}).items():
            for nm in self.info:
                for inf in self.info[nm].values(): inf["var"][f"sev:{col}"] = arr[inf["a"]:inf["a"] + inf["n"]].astype(np.float64)
        self.subject = {int(i): int(subj_of_stay.get(int(store.ids[store.starts[i]]), -1)) for nm in self.splits for i in self.splits[nm]}
        self.truncated = {int(i): bool(store.lens[i] > self.cfg.MAX_SEQ_LEN) for nm in self.splits for i in self.splits[nm]}

    def run(self, model, arm, plan, clusters=True):
        """plan: {split: [variants]}. A variant is a severity key ('real', 'circ_shift', 'donor', 'rand_proj', 'sev:<column>'), optionally with
        '~rev' (exploratory reversed residualisation) and / or '@t24' (queries at position >= 24 only).
        -> dict(fits={(split, variant): dict(b1 (S, nDV*K), se, rob, stays, groups)}, dvs, K, L, H, mass)."""
        cfg = self.cfg; mh = attention_modules(model); L_, H_ = len(mh), mh[0].num_heads; K = L_ * H_; Tm = cfg.MAX_SEQ_LEN
        fits = {}; pos_sum = np.zeros((L_, Tm)); pos_cnt = np.zeros(Tm); head_sum = np.zeros((L_, H_)); head_cnt = 0.0
        floor_sum = np.zeros((L_, H_)); n_stays = 0; mfit = ([], [], [])
        for split, variants in plan.items():
            acc = {v: ([], [], [], []) for v in variants}
            for i, z, w, vn in capture_stays(model, self.store, self.splits[split], cfg=cfg):
                inf = self.info[split][i]; Dv = dvs_for_stay(z, vn, inf, arm); Y = np.concatenate([Dv[d] for d in DVS], 1)
                for v in variants:
                    Xc = design(inf, v)
                    if Xc is None: continue
                    keep = inf["tt"] >= 24 if "@t24" in v else None
                    if keep is not None: b1, se, rob = stay_ols(Xc[keep], Y[keep], inf["jj"][keep] if clusters else None)
                    else: b1, se, rob = stay_ols(Xc, Y, inf["jj"] if clusters else None)
                    acc[v][0].append(b1); acc[v][1].append(se); acc[v][2].append(rob if rob is not None else np.full(len(b1), np.nan)); acc[v][3].append(i)
                m = w.sum(-1).astype(np.float64); n = m.shape[-1]                                 # total mass M_t per head (L,h,n)
                pos_sum[:, :n] += m.mean(1); pos_cnt[:n] += 1
                head_sum += m[:, :, cfg.CORR_MIN_T:].sum(-1); head_cnt += max(0, n - cfg.CORR_MIN_T)
                floor_sum += (w[:, :, inf["tt"], inf["jj"]] < 1e-6).mean(-1); n_stays += 1
                if arm in ("sigmoid", "kernel_raw") and n - cfg.CORR_MIN_T >= 8:                  # 6.4: does total mass depend on what is in the window?
                    t = np.arange(cfg.CORR_MIN_T, n); cnt = np.arange(1, n + 1.0); s = inf["var"]["real"]
                    Xm = np.c_[(np.cumsum(s) / cnt)[t], np.log(t + 1.0), (np.cumsum(inf["dens"]) / cnt)[t]]
                    b1, se, _ = stay_ols(Xm, m[:, :, t].reshape(K, -1).T); mfit[0].append(b1); mfit[1].append(se); mfit[2].append(i)
            for v in variants:
                if acc[v][0]:
                    st_ = np.array(acc[v][3], np.int64)
                    fits[(split, v)] = dict(b1=np.array(acc[v][0]), se=np.array(acc[v][1]), rob=np.array(acc[v][2]), stays=st_,
                                            groups=np.array([self.subject[int(s)] for s in st_]), truncated=np.array([self.truncated[int(s)] for s in st_]))
        mass = dict(by_layer=(head_sum / max(head_cnt, 1)).mean(1), by_head=head_sum / max(head_cnt, 1),
                    by_pos=np.where(pos_cnt > 0, pos_sum / np.maximum(pos_cnt, 1), np.nan), pos_count=pos_cnt, floor=floor_sum / max(n_stays, 1))
        if mfit[0]: mass["fit"] = dict(b1=np.array(mfit[0]), se=np.array(mfit[1]), groups=np.array([self.subject[int(s)] for s in mfit[2]]))
        return dict(fits=fits, dvs=list(DVS), K=K, L=L_, H=H_, mass=mass, arm=arm)


# ---------------------------------------------------------------- stage 2 and decisions ----------------------------------------------------------------
def dl_meta(b, se, groups=None, n_boot=1000):
    """DerSimonian-Laird random-effects pooling across stays, per column (v3 cell 13, section 6, unchanged), plus: a 95% prediction interval
    and, if `groups` (subject per stay) is given, a subject-level bootstrap CI of the plain mean. -> dict of (K,) arrays."""
    K = b.shape[1]
    out = {k_: np.full(K, np.nan) for k_ in ("est", "se", "lo", "hi", "tau2", "I2", "k", "mean", "boot_lo", "boot_hi", "pi_lo", "pi_hi", "subj_lo", "subj_hi")}
    for k in range(K):
        ok = np.isfinite(b[:, k]) & np.isfinite(se[:, k]) & (se[:, k] > 0); bb, ss = b[ok, k], se[ok, k]
        if len(bb) < 3: continue
        w = 1 / ss ** 2; fe = np.sum(w * bb) / w.sum(); Q = np.sum(w * (bb - fe) ** 2); dfq = len(bb) - 1
        tau2 = max(0.0, (Q - dfq) / (w.sum() - np.sum(w ** 2) / w.sum())); ws = 1 / (ss ** 2 + tau2)
        est = np.sum(ws * bb) / ws.sum(); sei = np.sqrt(1 / ws.sum())
        out["est"][k], out["se"][k], out["lo"][k], out["hi"][k] = est, sei, est - 1.96 * sei, est + 1.96 * sei
        out["tau2"][k], out["I2"][k], out["k"][k] = tau2, (max(0.0, (Q - dfq) / Q) if Q > 0 else 0.0), len(bb)
        bm = np.random.default_rng(k).integers(0, len(bb), (n_boot, len(bb)))            # v3: plain mean with a stay-bootstrap CI
        out["mean"][k] = bb.mean(); out["boot_lo"][k], out["boot_hi"][k] = np.percentile(bb[bm].mean(1), [2.5, 97.5])
        half = sps.t.ppf(0.975, max(len(bb) - 2, 1)) * np.sqrt(tau2 + sei ** 2); out["pi_lo"][k], out["pi_hi"][k] = est - half, est + half
        if groups is not None:
            _, inv = np.unique(np.asarray(groups)[ok], return_inverse=True); G = int(inv.max()) + 1
            sm = np.bincount(inv, weights=bb, minlength=G); cn = np.bincount(inv, minlength=G).astype(float)
            dr = np.random.default_rng(10_000 + k).integers(0, G, (n_boot, G)); out["subj_lo"][k], out["subj_hi"][k] = np.percentile(sm[dr].sum(1) / cn[dr].sum(1), [2.5, 97.5])
    return out


def tost_p(est, se, delta):
    """Two one-sided tests against +/- delta (v3)."""
    return np.maximum(1 - sps.norm.cdf((est + delta) / se), sps.norm.cdf((est - delta) / se))


def two_sided_p(est, se): return 2 * (1 - sps.norm.cdf(np.abs(est / se)))


def bh_fdr(p, q=0.05):
    """Benjamini-Hochberg: boolean array, True where the hypothesis is rejected at FDR q. NaN p-values are never rejected."""
    p = np.asarray(p, float); out = np.zeros(p.shape, bool); idx = np.flatnonzero(np.isfinite(p.ravel())); m = len(idx)
    if m == 0: return out
    o = idx[np.argsort(p.ravel()[idx])]; below = p.ravel()[o] <= q * np.arange(1, m + 1) / m
    if below.any(): out.ravel()[o[: int(np.max(np.flatnonzero(below))) + 1]] = True
    return out


def pass_confirm(disc, scr, thr, sesoi):
    """v3 head-level rule. pass on DISCOVERY: |b1| > threshold and both CIs exclude 0. Confirmed on SCREEN: same sign, |b1| > SESOI, CI excludes 0."""
    p = (np.abs(disc["est"]) > thr) & ((disc["lo"] > 0) | (disc["hi"] < 0)) & ((disc["boot_lo"] > 0) | (disc["boot_hi"] < 0))
    c = (np.sign(scr["est"]) == np.sign(disc["est"])) & (np.abs(scr["est"]) > sesoi) & ((scr["lo"] > 0) | (scr["hi"] < 0))
    return p, c


def seed_verdict(E, n_required=None):
    """Section 5.6. E: per-seed excess (or any per-seed number whose sign is the evidence).
    PRESENT: E > 0 in every one of the 5 seeds (exact one-sided sign test, p = 1/32). ABSENT: E <= 0 in at least 4 of 5. Otherwise SEED-DEPENDENT."""
    n_required = n_required or C.CFG.SEED_RULE_N
    E = [float(e) for e in E if e is not None and np.isfinite(e)]; n = len(E); pos = sum(e > 0 for e in E)
    if n < n_required: return f"INCOMPLETE ({n} of {n_required} seeds)"
    if pos == n: return "PRESENT"
    if n - pos >= n_required - 1: return "ABSENT"
    return "SEED-DEPENDENT"


def all_seeds(flags, n_required=None):
    """Model-level quantities: the statement holds only if it holds in 5 of 5 seeds."""
    n_required = n_required or C.CFG.SEED_RULE_N; flags = [f for f in flags if f is not None]
    if len(flags) < n_required: return f"INCOMPLETE ({len(flags)} of {n_required} seeds)"
    return "YES" if all(flags) else f"NO ({sum(bool(f) for f in flags)} of {len(flags)} seeds)"


def _meta(run, split, variant, cols, robust=False, keep=None):
    f = run["fits"].get((split, variant))
    if f is None: return None
    b, s, g = f["b1"][:, cols], (f["rob"] if robust else f["se"])[:, cols], f["groups"]
    if keep is not None: b, s, g = b[keep(f)], s[keep(f)], g[keep(f)]
    return dl_meta(b, s, groups=g)


def _l(a, nd=6): return [None if not np.isfinite(x) else round(float(x), nd) for x in np.asarray(a, float).ravel()]


def analyse(runs_by_role, arm, cfg=None):
    """runs_by_role: outputs of Screen.run for 'trained' (DISCOVERY: real + the three surrogate severities; SCREEN: real), 'init' and 'shuf_v3'
    (band members; DISCOVERY and SCREEN, real severity) and optionally 'shuf_budget', 'shuf_budget_noaux' (beside the band, never in it).
    -> JSON-ready record: per DV the band, the threshold, the head table, the counts and the excess E that feeds the seed rule."""
    cfg = cfg or C.CFG; tr = runs_by_role["trained"]; K, L_, H_ = tr["K"], tr["L"], tr["H"]
    rec = dict(arm=arm, primary_dv=PRIMARY_DV.get(arm, "clr"), L=L_, H=H_, band_complete=all(r in runs_by_role for r in ("init", "shuf_v3")), by_dv={})
    for di, dv in enumerate(tr["dvs"]):
        cols = slice(di * K, (di + 1) * K)
        disc, scr = _meta(tr, "DISCOVERY", "real", cols), _meta(tr, "SCREEN", "real", cols)
        if disc is None or scr is None: continue
        ctrl = {}
        for role in ("init", "shuf_v3"):
            m = _meta(runs_by_role[role], "DISCOVERY", "real", cols) if role in runs_by_role else None
            if m is not None: ctrl[f"{role}:real"] = m["est"]
        for v in BAND_SURROGATES:
            m = _meta(tr, "DISCOVERY", v, cols)
            if m is not None: ctrl[f"trained:{v}"] = m["est"]
        band = float(np.nanpercentile(np.abs(np.concatenate(list(ctrl.values()))), cfg.CORR_BAND_Q)) if ctrl else float("nan")
        thr = max(band, cfg.CORR_SESOI) if np.isfinite(band) else cfg.CORR_SESOI
        p, c = pass_confirm(disc, scr, thr, cfg.CORR_SESOI)
        counts = dict(trained=int((p & c).sum()), trained_pass_discovery=int(p.sum())); ctrl_tables = {}
        for role in ("init", "shuf_v3", "shuf_budget", "shuf_budget_noaux"):
            if role not in runs_by_role: continue
            dd, ss = _meta(runs_by_role[role], "DISCOVERY", "real", cols), _meta(runs_by_role[role], "SCREEN", "real", cols)
            if dd is None or ss is None: continue
            pp, cc = pass_confirm(dd, ss, thr, cfg.CORR_SESOI); counts[role] = int((pp & cc).sum()); ctrl_tables[role] = dict(b1_disc=_l(dd["est"]), max_abs=float(np.nanmax(np.abs(dd["est"]))))
        def excess(other): return (counts["trained"] - max(counts["init"], counts[other])) if ("init" in counts and other in counts) else None
        rob = _meta(tr, "DISCOVERY", "real", cols, robust=True)
        tost = tost_p(disc["est"], disc["se"], cfg.CORR_SESOI) < 0.05
        order = np.argsort(-np.nan_to_num(np.abs(disc["est"]), nan=-1.0))[: cfg.CORR_TOP_K]
        top = [dict(index=int(i), layer=int(i // H_) + 1, head=int(i % H_) + 1, b1_disc=float(disc["est"][i]), b1_screen=float(scr["est"][i]), passed=bool(p[i]),
                    confirmed=bool(c[i]), layer1=bool(i // H_ == 0), label=("passed the screen" if p[i] and c[i] else "carried forward without passing the screen")) for i in order]
        rec["by_dv"][dv] = dict(
            band=band, threshold=thr, band_members={k_: float(np.nanmax(np.abs(v_))) for k_, v_ in ctrl.items()}, counts=counts,
            E=excess("shuf_v3"), E_budget=excess("shuf_budget"), E_noaux=excess("shuf_budget_noaux"),
            n_tost_equivalent=int(tost.sum()), max_abs_trained=float(np.nanmax(np.abs(disc["est"]))), n_stays_disc=int(np.nanmax(disc["k"])), top=top,
            heads=dict(b1_disc=_l(disc["est"]), lo_disc=_l(disc["lo"]), hi_disc=_l(disc["hi"]), boot_lo=_l(disc["boot_lo"]), boot_hi=_l(disc["boot_hi"]),
                       subj_lo=_l(disc["subj_lo"]), subj_hi=_l(disc["subj_hi"]), b1_screen=_l(scr["est"]), lo_screen=_l(scr["lo"]), hi_screen=_l(scr["hi"]),
                       pass_disc=[bool(x) for x in p], confirmed=[bool(x) for x in c], tost_equiv=[bool(x) for x in tost], I2_v3=_l(disc["I2"]),
                       I2_robust=_l(rob["I2"]), tau_robust=_l(np.sqrt(rob["tau2"])), b1_robust=_l(rob["est"]), pi_lo=_l(rob["pi_lo"]), pi_hi=_l(rob["pi_hi"]),
                       ci_excludes_0_robust=[bool(x) for x in ((rob["lo"] > 0) | (rob["hi"] < 0))],
                       **{f"ctrl_{k_}": _l(v_) for k_, v_ in ctrl.items()}),
            controls=ctrl_tables)
    m = tr["mass"]; rec["mass"] = dict(by_layer=_l(m["by_layer"]), by_head=_l(m["by_head"]), by_pos=_l(m["by_pos"]), floor_fraction=_l(m["floor"]),
                                       floor_heads_above_5pct=int((m["floor"] > 0.05).sum()))
    if "fit" in m:                                                                       # 6.4: total mass against the mean severity of the visible keys
        mm = dl_meta(m["fit"]["b1"], m["fit"]["se"], groups=m["fit"]["groups"]); eq = tost_p(mm["est"], mm["se"], cfg.CORR_SESOI) < 0.05
        rec["mass"]["composition"] = dict(b1=_l(mm["est"]), lo=_l(mm["lo"]), hi=_l(mm["hi"]), tost_equivalent=[bool(x) for x in eq],
                                          all_heads_equivalent_layers_ge2=bool(eq[H_:].all()), share_equivalent_layers_ge2=float(eq[H_:].mean()))
    return rec


def sensitivity(tr, arm, cfg=None, threshold=None):
    """Pre-specified sensitivity analyses on the trained model (5.10). None of them changes a primary verdict; they are reported beside it.
    -> (table rows, p-values per severity definition for the FDR family)."""
    cfg = cfg or C.CFG; K = tr["K"]; di = tr["dvs"].index(PRIMARY_DV.get(arm, "clr")); cols = slice(di * K, (di + 1) * K); rows = []; pvals = {}
    prim = _meta(tr, "DISCOVERY", "real", cols)
    def row(name, m):
        r = np.corrcoef(np.nan_to_num(prim["est"]), np.nan_to_num(m["est"]))[0, 1] if prim is not None else float("nan")
        rows.append(dict(analysis=name, max_abs_b1=float(np.nanmax(np.abs(m["est"]))), heads_beyond_primary_threshold=(int((np.abs(m["est"]) > threshold).sum()) if threshold else None),
                         correlation_with_primary=float(r), b1=_l(m["est"])))
    for (split, v) in sorted(tr["fits"]):
        if split != "DISCOVERY": continue
        if v.startswith("sev:"):
            m = _meta(tr, split, v, cols); row(f"severity = {v[4:]}", m); pvals[v[4:]] = two_sided_p(m["est"], m["se"])
        elif "@t24" in v or "~rev" in v:
            row({"real@t24": "queries at position >= 24 only", "real~rev": "EXPLORATORY reversed residualisation"}.get(v, v), _meta(tr, split, v, cols))
    if prim is not None:
        row("stays longer than the window excluded", _meta(tr, "DISCOVERY", "real", cols, keep=lambda f: ~f["truncated"]))
        row("cluster-robust stage-1 errors", _meta(tr, "DISCOVERY", "real", cols, robust=True))
        nw = slice(tr["dvs"].index("normw") * K, (tr["dvs"].index("normw") + 1) * K); row("norm-weighted DV", _meta(tr, "DISCOVERY", "real", nw))
    return rows, pvals


def subgroup_search(tr, arm, attrs, cfg=None):
    """S5 (EXPLORATORY). attrs: {name: {stay index: 0/1 or NaN}} two-group splits of the stays. Per head, difference of the mean per-stay b1
    between the groups (Welch test) on DISCOVERY; Benjamini-Hochberg at q = 0.10 across heads x subgroups; confirmed on SCREEN (same sign, p < 0.05)."""
    cfg = cfg or C.CFG; K = tr["K"]; di = tr["dvs"].index(PRIMARY_DV.get(arm, "clr")); cols = slice(di * K, (di + 1) * K); res = {}; P = []
    def diff(split, name):
        f = tr["fits"][(split, "real")]; g = np.array([attrs[name].get(int(s), np.nan) for s in f["stays"]], float); b = f["b1"][:, cols]
        a_, b_ = b[g == 1], b[g == 0]
        if len(a_) < 10 or len(b_) < 10: return np.full(K, np.nan), np.full(K, np.nan), (len(a_), len(b_))
        t = sps.ttest_ind(a_, b_, axis=0, equal_var=False, nan_policy="omit"); return np.nanmean(a_, 0) - np.nanmean(b_, 0), np.asarray(t.pvalue, float), (len(a_), len(b_))
    for name in attrs:
        d, p, n = diff("DISCOVERY", name); d2, p2, n2 = diff("SCREEN", name); res[name] = dict(d=d, p=p, d2=d2, p2=p2, n_disc=n, n_screen=n2); P.append(p)
    rej = bh_fdr(np.array(P), cfg.S5_FDR_Q) if P else np.zeros((0, K), bool); out = []
    for gi, name in enumerate(attrs):
        r = res[name]; conf = rej[gi] & (np.sign(r["d"]) == np.sign(r["d2"])) & (r["p2"] < 0.05)
        out.append(dict(subgroup=name, n_disc=r["n_disc"], n_screen=r["n_screen"], heads_fdr_discovery=int(rej[gi].sum()), heads_confirmed_screen=int(conf.sum()),
                        confirmed_heads=[int(i) for i in np.flatnonzero(conf)], max_abs_difference=float(np.nanmax(np.abs(r["d"]))) if np.isfinite(r["d"]).any() else None))
    return out


# ---------------------------------------------------------------- section 6.2: is attention used at all? ----------------------------------------------------------------
class knockout:
    """Context manager. mode='zero': the output of every attention sublayer is set to zero (the residual stream is kept).
    mode='uniform': the weights of every row are replaced by a uniform row with the same total mass (descriptive only)."""
    def __init__(self, model, mode): self.model, self.mode, self.h = model, mode, []
    def __enter__(self):
        fn = self._zero if self.mode == "zero" else self._uniform
        self.h = [m.register_forward_hook(fn) for m in attention_modules(self.model)]; return self
    def __exit__(self, *a):
        for h in self.h: h.remove()
        return False
    @staticmethod
    def _zero(m, inp, out): return torch.zeros_like(out)
    @staticmethod
    def _uniform(m, inp, out):
        W, V = m.attention_weights, m.Value; mask = inp[1] if len(inp) > 1 else None
        vis = torch.ones_like(W) if mask is None else (~mask).to(W.dtype).expand_as(W)
        U = vis * (W.sum(-1, keepdim=True) / vis.sum(-1, keepdim=True).clamp(min=1))
        att = torch.matmul(U, V).transpose(1, 2).contiguous().view(out.shape[0], out.shape[1], -1)
        return m.W_O(att).to(out.dtype)


def knockout_report(bundle, name, which="best.pt", n_boot=300):
    """VAL AUROC with attention intact and under both knockouts, with the stay-level paired-bootstrap CI of the drop. Aggregates only."""
    from . import runs as R
    st = bundle["store"]; model = R.load_trained(name, st, which); rows, base = R.predict_store(model, st, bundle["val_idx"])
    out = dict(name=name, checkpoint=which, auroc_intact=MET.compute_metrics(st, rows, base)["auroc_mean"])
    for mode in ("zero", "uniform"):
        with knockout(model, mode): r2, lg = R.predict_store(model, st, bundle["val_idx"])
        assert np.array_equal(rows, r2)
        d = MET.stay_bootstrap_dauroc(st, rows, lg, base, n_boot)                         # AUROC(intact) - AUROC(knockout)
        out[mode] = dict(auroc=MET.compute_metrics(st, rows, lg)["auroc_mean"], drop=float(d[0]), lo=float(d[1]), hi=float(d[2]))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def knockout_matters(k, cfg=None):
    """6.2: the knockout MATTERS iff it lowers VAL AUROC by at least 0.01 with the 95% CI excluding 0. Otherwise it costs little."""
    cfg = cfg or C.CFG; z = k["zero"]; return bool(z["drop"] >= cfg.KNOCKOUT_MIN_DROP and z["lo"] > 0)


def seed_outcome(sig_knock, soft_knock, mass_by_layer, cfg=None):
    """6.2, outcome per seed from the paired sigmoid and softmax models."""
    cfg = cfg or C.CFG; mass_ok = bool(np.nanmax(np.asarray(mass_by_layer, float)) >= cfg.MASS_FLOOR)
    sm, fm = knockout_matters(sig_knock, cfg), knockout_matters(soft_knock, cfg)
    if sm and mass_ok: return "alive"
    if sm: return "collapsed (mass floor not met)"
    return "collapsed (sigmoid only)" if fm else "attention contributes little in either arm"


def study_outcome(outcomes, n_required=None):
    """6.2, outcome for the study by count over the five seeds."""
    n_required = n_required or C.CFG.SEED_RULE_N; o = list(outcomes)
    if len(o) < n_required: return f"INCOMPLETE ({len(o)} of {n_required} seeds)"
    if sum(x == "alive" for x in o) >= n_required - 1: return "sigmoid ALIVE"
    if sum(x == "attention contributes little in either arm" for x in o) >= n_required - 1: return "ATTENTION CONTRIBUTES LITTLE IN EITHER ARM"
    return "sigmoid NOT ALIVE"


def drift_flag(mass_by_layer, mass_per_epoch):
    """6.3: flagged if mean total mass exceeds 5 in any layer, or the logged mass at the last epoch is more than twice its value three epochs earlier."""
    m = [x for x in (mass_per_epoch or []) if x is not None]
    return bool(np.nanmax(np.asarray(mass_by_layer, float)) > 5 or (len(m) >= 4 and m[-1] > 2 * m[-4]))
