"""Causal battery, PDF points P5-P8 (Docs/PREREGISTRATION.md section 7). Order of folds: LOC -> Docs/LOC_CHOICES.md committed -> CONF-A once
-> EXT once at the very end. The fold rules are enforced by data.FoldGuard, not by this module.

Notation: k = intervened hour, t = readout (query) hour, S = {j : k < j < t}. All positions are positions inside the stay's analysis window
(its most recent MAX_SEQ_LEN hours). Every effect is the paired contrast "severe splice - sham splice" on the same (stay, k, t).

Sites   S1 embedding of hour k replaced by the donor hour's embedding (PRIMARY; no other hour's input changes)
        S0 raw inputs of hour k replaced, forward-fill and time-since-measurement recomputed downstream (reported beside S1)
        S4 delta added to logit z_tk, S6 query vector scaled (model-level references); S2 / S3 / S5 exploratory, LOC only
Exact   E1 identical runs give dz = 0; E2 dz = 0 for every query before k; E3 (S1 only) layer-1 dz = 0 off row / column k;
        E4 (arousal) outputs before t unchanged. A violation is a bug: the notebook stops.
Governance: pair tables and per-pair statistics are written to Drive, never displayed; everything printed is an aggregate.
"""
import os, json, math, time, types, hashlib
import numpy as np
import torch

from . import config as C
from . import data as D
from . import attention as A
from . import metrics as MET
from . import user_classes as UC
from .dataset import collate, build_attn_mask
from .engine import RT, to_dev
from .models import attention_modules, release_attn_cache
from .correlational import seed_verdict, all_seeds, tost_p, load_severity

VITALS = ("HR", "MAP", "Resp", "O2Sat")
STATS = ("d", "c", "lam")


# ---------------------------------------------------------------- folds and FIT quantiles ----------------------------------------------------------------
def prepare_fold(fold, pre, guard=None, stage="causal", sev_col=None, cfg=None):
    """Load ONE reserved fold through the guard and transform it with the FIT-fitted preprocessor (FIT statistics applied as they are).
    -> dict(store, sev (per row, analysis only), subj (per stay), onset, counts). Prints nothing row-level."""
    cfg = cfg or C.CFG; sev_col = sev_col or cfg.CORR_SEVERITY_COL; ID, HR = cfg.ID_COL, cfg.HOUR_COL
    df, static_real, stats = D.load_hourly(features=list(pre.feat), extra_columns=[sev_col])
    onset = D.stay_onset(df)
    split, sha = D.load_frozen_split()
    df, counts = D.attach_folds(df, split, (fold,), guard or D.FoldGuard(stage))
    if cfg.SMOKE_TEST: df = D.smoke_subsample(df, min_subjects=120)
    df = df.sort_values([ID, HR], kind="mergesort").reset_index(drop=True)
    store = pre.transform(df)
    if store.n_rows != len(df) or not np.array_equal(store.ids, df[ID].to_numpy()): raise RuntimeError("row order of the store and the frame differ")
    df["_sev"] = df.groupby(ID)[sev_col].ffill()                                             # same rule as the correlational stage (v3)
    df["_sev"] = df["_sev"].fillna(df.groupby(ID)["_sev"].transform("median")).fillna(0.0)
    fb = dict(store=store, sev=df["_sev"].to_numpy(np.float32), subj=df[cfg.SUBJECT_COL].to_numpy()[store.starts], onset=onset, fold=fold,
              counts={k: int(v) for k, v in counts.items()}, split_sha=sha, n_stays=int(len(store.starts)), n_subjects=int(df[cfg.SUBJECT_COL].nunique()))
    del df
    return fb


def fit_quantiles(bundle, cfg=None):
    """Class boundaries from FIT rows only (PDF P5): severity Q50 / Q90, and the arousal index z(HR) + z(Resp) Q50 / Q90.
    The arousal index is taken where HR and Resp are both measured at that hour: there the decayed input equals the measured z-score
    for every model, so the index is 'from the decayed input' (PREREGISTRATION 7.6) and still the same for all models."""
    cfg = cfg or C.CFG; st = bundle["store"]; feats = list(bundle["pre"].feat)
    sev, _ = load_severity(st, bundle["fit_idx"], [cfg.CORR_SEVERITY_COL]); s = sev[list(sev)[0]]
    fit_rows = np.repeat(np.isin(np.arange(len(st.starts)), bundle["fit_idx"]), st.lens)
    hr, rs = feats.index("HR"), feats.index("Resp"); both = fit_rows & (st.M[:, hr] > 0) & (st.M[:, rs] > 0)
    ar = (st.XL[both, hr] + st.XL[both, rs]).astype(np.float64)
    qs = np.nanpercentile(s[fit_rows], [cfg.CAUSAL_Q_ROUTINE, cfg.CAUSAL_Q_SEVERE]); qa = np.percentile(ar, [cfg.CAUSAL_Q_ROUTINE, cfg.CAUSAL_Q_SEVERE])
    return dict(sev_routine_max=float(qs[0]), sev_severe_min=float(qs[1]), arousal_low_max=float(qa[0]), arousal_high_min=float(qa[1]),
                n_fit_rows=int(fit_rows.sum()), share_fit_rows_routine=float((s[fit_rows] <= qs[0]).mean()), share_fit_rows_severe=float((s[fit_rows] >= qs[1]).mean()))


def hour_table(fb, feats, cfg=None):
    """Per-row attributes for pair construction and matching (never printed)."""
    cfg = cfg or C.CFG; st = fb["store"]; n = st.n_rows; ns = len(st.starts); feats = list(feats)
    stay = np.repeat(np.arange(ns), st.lens)
    wlen = np.minimum(st.lens, cfg.MAX_SEQ_LEN).astype(np.int64); wstart = (st.starts + st.lens - wlen).astype(np.int64)
    vj = [feats.index(v) for v in VITALS]; prev = np.zeros((n, len(vj)), np.float32); prev[1:] = st.XL[:-1][:, vj]
    hr, rs = feats.index("HR"), feats.index("Resp")
    return dict(stay=stay, wpos=np.arange(n) - np.repeat(wstart, st.lens), wlen=np.repeat(wlen, st.lens), subj=np.asarray(fb["subj"])[stay],
                iculos=st.hours.astype(np.float64), nmeas=st.M.sum(1).astype(np.int64), unit=(st.S[:, 2] > 0.5).astype(np.int8) * 2 + (st.S[:, 3] > 0.5).astype(np.int8),
                prev=prev, sev=np.asarray(fb["sev"], np.float64), arousal=(st.XL[:, hr] + st.XL[:, rs]).astype(np.float64),
                both=(st.M[:, hr] > 0) & (st.M[:, rs] > 0), stay_wlen=wlen, stay_wstart=wstart, hr=hr, resp=rs)


def shifted_index(T, values, seed):
    """Placebo index (7.6): the same index circularly shifted within each stay's window."""
    out = np.full(len(values), np.nan); rng = np.random.default_rng(seed)
    for s in range(len(T["stay_wlen"])):
        n, a = int(T["stay_wlen"][s]), int(T["stay_wstart"][s])
        if n >= 4: out[a:a + n] = np.roll(values[a:a + n], int(rng.integers(n // 4, 3 * n // 4 + 1)))
    return out


# ---------------------------------------------------------------- pairs (7.1, 7.6) ----------------------------------------------------------------
def build_pairs(T, base_ok, treat_ok, sham_ok, kind, cfg=None, seed=0, n_bases=None):
    """kind='priority': the base row is hour k and a readout t >= max(8, k + 6) is drawn later in the same stay.
    kind='readout': the base row is the readout hour itself (k = t), used for do(Arousal_t).
    Treat donor and sham donors: another subject, ICULOS within +/-4 h, number of measured features within +/-2, same Unit1/Unit2 pattern,
    previous-hour HR / MAP / Resp / O2Sat each within 1 FIT SD. Bases without both donors are dropped. At most 3 bases per stay."""
    cfg = cfg or C.CFG; rng = np.random.default_rng(seed); n_bases = n_bases or cfg.CAUSAL_N_BASES; valid = T["wpos"] >= 1
    def pool_of(ok):
        r = np.flatnonzero(ok & valid); return dict(rows=r, subj=T["subj"][r], iculos=T["iculos"][r], nmeas=T["nmeas"][r], unit=T["unit"][r], prev=T["prev"][r])
    PT, PS = pool_of(treat_ok), pool_of(sham_ok)
    def match(Pl, r):
        m = ((Pl["subj"] != T["subj"][r]) & (np.abs(Pl["iculos"] - T["iculos"][r]) <= cfg.CAUSAL_TOL_ICULOS) & (np.abs(Pl["nmeas"] - T["nmeas"][r]) <= cfg.CAUSAL_TOL_NMEAS)
             & (Pl["unit"] == T["unit"][r]) & (np.abs(Pl["prev"] - T["prev"][r]).max(1) <= cfg.CAUSAL_TOL_VITALS_SD))
        return Pl["rows"][m]
    if kind == "priority": cand = base_ok & (T["wpos"] >= cfg.CAUSAL_MIN_K) & (np.maximum(cfg.CAUSAL_MIN_T, T["wpos"] + cfg.CAUSAL_MIN_GAP) <= T["wlen"] - 1)
    else: cand = base_ok & (T["wpos"] >= cfg.CAUSAL_MIN_T)
    cand_rows = np.flatnonzero(cand); by_stay = {}
    for r in cand_rows[rng.permutation(len(cand_rows))]: by_stay.setdefault(int(T["stay"][r]), []).append(int(r))
    stays = list(by_stay); stays = [stays[i] for i in rng.permutation(len(stays))]
    rec = {k_: [] for k_ in ("row", "stay", "a", "k", "t", "treat", "sham", "sham2", "loose")}; tried = dropped = 0
    for s in stays:
        if len(rec["row"]) >= n_bases: break
        got = 0
        for r in by_stay[s]:
            if got >= cfg.CAUSAL_MAX_PER_STAY or len(rec["row"]) >= n_bases: break
            tried += 1; mt = match(PT, r); ms = match(PS, r); ms = ms[ms != r]
            if len(mt) == 0 or len(ms) == 0: dropped += 1; continue
            k = int(T["wpos"][r]); t = k if kind == "readout" else int(rng.integers(max(cfg.CAUSAL_MIN_T, k + cfg.CAUSAL_MIN_GAP), int(T["wlen"][r])))
            sh = rng.choice(ms, size=min(2, len(ms)), replace=False); lo = PT["rows"][PT["subj"] != T["subj"][r]]
            for k_, v in (("row", r), ("stay", s), ("a", int(r) - k), ("k", k), ("t", t), ("treat", int(rng.choice(mt))), ("sham", int(sh[0])),
                          ("sham2", int(sh[1]) if len(sh) > 1 else -1), ("loose", int(rng.choice(lo)) if len(lo) else -1)): rec[k_].append(v)
            got += 1
    P = {k_: np.asarray(v, np.int64) for k_, v in rec.items()}; n = len(P["row"])
    P["subj"] = T["subj"][P["row"]] if n else np.zeros(0, np.int64)
    P["sev_base"] = T["sev"][P["row"]] if n else np.zeros(0); P["sev_treat"] = T["sev"][P["treat"]] if n else np.zeros(0); P["sev_sham"] = T["sev"][P["sham"]] if n else np.zeros(0)
    mx, mn, ar = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):                                         # severity of the other visible keys (for model M1) and the natural arousal of the readout hour
        a, k, t = int(P["a"][i]), int(P["k"][i]), int(P["t"][i]); s = np.delete(T["sev"][a:a + t + 1], k)
        if len(s): mx[i], mn[i] = s.max(), s.min()
        if T["both"][a + t]: ar[i] = T["arousal"][a + t]
    P["sev_max_others"], P["sev_min_others"], P["arousal_t"] = mx, mn, ar
    P["ar_base"] = T["arousal"][P["row"]] if n else np.zeros(0); P["ar_treat"] = T["arousal"][P["treat"]] if n else np.zeros(0)
    info = dict(kind=kind, bases=int(n), stays=int(len(np.unique(P["stay"]))) if n else 0, subjects=int(len(np.unique(P["subj"]))) if n else 0, candidates_tried=int(tried),
                dropped_no_donor=int(dropped), yield_share=float(n / max(tried, 1)), with_second_sham=int((P["sham2"] >= 0).sum()) if n else 0,
                treat_pool_rows=int(len(PT["rows"])), sham_pool_rows=int(len(PS["rows"])))
    return P, info


def all_pairs(fb, feats, q, cfg=None, seed=0):
    """Every pair set of the battery for one fold. -> (PAIRS, INFO, hour table)."""
    cfg = cfg or C.CFG; T = hour_table(fb, feats, cfg); sev = T["sev"]; rout, sevr = sev <= q["sev_routine_max"], sev >= q["sev_severe_min"]
    ar_lo = T["both"] & (T["arousal"] <= q["arousal_low_max"]); ar_hi = T["both"] & (T["arousal"] >= q["arousal_high_min"])
    sh = shifted_index(T, np.where(T["both"], T["arousal"], np.nan), seed + 5)
    with np.errstate(invalid="ignore"): pl_lo, pl_hi = T["both"] & (sh <= q["arousal_low_max"]), T["both"] & (sh >= q["arousal_high_min"])
    spec = dict(up=(rout, sevr, rout, "priority"), down=(sevr, rout, sevr, "priority"), arousal=(ar_lo, ar_hi, ar_lo, "readout"),
                placebo=(pl_lo, pl_hi, pl_lo, "readout"), sofa_t=(rout, sevr, rout, "readout"))
    PAIRS, INFO = {}, {}
    for i, (name, (b_, t_, s_, kind)) in enumerate(spec.items()): PAIRS[name], INFO[name] = build_pairs(T, b_, t_, s_, kind, cfg, seed + 101 * (i + 1))
    return PAIRS, INFO, T


def severity_surrogates(fb, T, PAIRS, cfg=None):
    """Surrogate severities per stay for the selectivity-slope band (the three controls of 5.4: circular shift, donor stay, random projection).
    -> {stay index: (3, window length) float array}."""
    cfg = cfg or C.CFG; st = fb["store"]; w_rp = np.random.default_rng(cfg.SEED + 99).standard_normal(st.X.shape[1])
    stays = np.unique(np.concatenate([PAIRS[k]["stay"] for k in ("up", "down") if len(PAIRS[k]["stay"])])) if any(len(PAIRS[k]["stay"]) for k in ("up", "down")) else np.zeros(0, np.int64)
    out = {}
    for s in stays.tolist():
        n, a = int(T["stay_wlen"][s]), int(T["stay_wstart"][s]); sv = T["sev"][a:a + n]
        r = np.random.default_rng((cfg.SEED * 1_000_003 + int(st.ids[st.starts[s]]) + 17) % (2 ** 32))
        circ = np.roll(sv, int(r.integers(n // 4, 3 * n // 4 + 1)))
        cands = stays[(T["stay_wlen"][stays] >= n) & (stays != s)]
        if len(cands):
            d = int(r.choice(cands)); da, dn = int(T["stay_wstart"][d]), int(T["stay_wlen"][d]); don = T["sev"][da + dn - n: da + dn]
        else: don = r.permutation(sv)
        rp = st.XL[a:a + n].astype(np.float64) @ w_rp
        out[int(s)] = np.stack([circ, don, np.sort(sv)[np.argsort(np.argsort(rp, kind="stable"), kind="stable")]])
    return out


# ---------------------------------------------------------------- items, splices, forward ----------------------------------------------------------------
def _item(st, a, n):
    b = a + n
    return dict(X=st.X[a:b], M=st.M[a:b], D=st.D[a:b], XL=st.XL[a:b], S=st.S[a:b], G=st.G[a:b], E=st.E[a:b], hrs=st.hours[a:b], Y=st.Y[a:b], V=st.V[a:b],
                rows=np.arange(a, b), keep_from=0)


def splice_input(item, k, drow, st, feats=None):
    """S0: the raw inputs of hour k (all features, or only `feats`) are replaced by those of donor row `drow`; the forward-filled value (XL)
    and the time since the last measurement (D) are then recomputed for every hour >= k with the Preprocessor's own rule. Hours before k
    are untouched. The engineered channels E are not recomputed: no v4 model reads them (eng = 0), and run_* asserts that."""
    assert k >= 1, "the spliced hour needs a previous hour inside the window"
    f = np.arange(item["X"].shape[1]) if feats is None else np.asarray(feats)
    X, M, XL, Dl = item["X"].copy(), item["M"].copy(), item["XL"].copy(), item["D"].copy(); h = item["hrs"]
    X[k, f] = st.X[drow, f]; M[k, f] = st.M[drow, f]
    for j in range(k, len(h)):
        obs = M[j, f] > 0
        XL[j, f] = np.where(obs, X[j, f], XL[j - 1, f]); Dl[j, f] = np.where(obs, 0.0, Dl[j - 1, f] + (h[j] - h[j - 1]))
    out = dict(item); out.update(X=X, M=M, XL=XL, D=Dl); return out


def delete_hours(item, hours):
    """Set the given hours to 'not measured' (used by the faithfulness tests of Track D): mask 0, value 0, forward-fill and time-since recomputed.
    At position 0 a deleted measurement falls back to the training mean (0 in z-space) with time-since 0: the history before the window is not kept."""
    hours = sorted(int(j) for j in hours)
    if not hours: return item
    X, M, XL, Dl = item["X"].copy(), item["M"].copy(), item["XL"].copy(), item["D"].copy(); h = item["hrs"]
    X[hours] = 0.0; M[hours] = 0.0
    for j in range(hours[0], len(h)):
        obs = M[j] > 0
        if j == 0: XL[j] = np.where(obs, X[j], 0.0); Dl[j] = np.where(obs, 0.0, Dl[j] if j not in hours else 0.0)
        else: XL[j] = np.where(obs, X[j], XL[j - 1]); Dl[j] = np.where(obs, 0.0, Dl[j - 1] + (h[j] - h[j - 1]))
    out = dict(item); out.update(X=X, M=M, XL=XL, D=Dl); return out


@torch.no_grad()
def forward(model, b, e=None, values=False):
    """One fp32 forward pass with dropout off. -> Z (B,L,h,T,T) pre-normaliser logits (-inf where masked), A (weights), logits (B,T,3) [, V]."""
    was = model.training; model.eval()
    with torch.autocast(device_type=b["X"].device.type, enabled=False):
        out = model(b["X"].float(), b["M"].float(), b["D"].float(), b["XL"].float(), b["S"].float(), b["G"], b["hrs"].float(), b["E"].float(),
                    build_attn_mask(b["PAD"]), embedding_override=e)
    mh = attention_modules(model)
    r = dict(Z=torch.stack([m.attention_logits.float() for m in mh], 1), A=torch.stack([m.attention_weights.float() for m in mh], 1),
             logits=torch.stack([out["logit_t1"], out["logit_t3"], out["logit_t6"]], -1).float())
    if values: r["V"] = torch.stack([m.Value.float() for m in mh], 1)
    release_attn_cache(model)
    if was: model.train()
    return r


@torch.no_grad()
def donor_embedding(model, st, rows):
    """S1: the embedding of each donor hour, computed from that hour's own inputs (the embedding is position-wise). -> (B, d)."""
    b = to_dev(collate([_item(st, int(r), 1) for r in rows])); return A.embed(model, b)[:, 0]


def _check_model(model):
    if getattr(model, "eng_proj", None) is not None: raise NotImplementedError("input splices do not recompute the engineered channels; this model reads them")
    if not hasattr(model, "embed"): raise NotImplementedError("the causal battery is defined for the ArmPipeline family")


# ---------------------------------------------------------------- per-pair statistics at the readout row ----------------------------------------------------------------
def kendall(dz, Sm, sev, n_perm, gen):
    """Kendall tau-b between severity and dz over the keys of each item's set, per head, for the real severity and n_perm within-set
    permutations of it (the random-alignment floor). -> (B, 1 + n_perm, L, h), NaN where the set has fewer than 3 keys or no variation."""
    B, L_, H_, T = dz.shape; out = torch.full((B, 1 + n_perm, L_, H_), float("nan"), device=dz.device)
    for i in range(B):
        idx = torch.nonzero(Sm[i]).squeeze(-1); m = int(idx.numel())
        if m < 3: continue
        x = sev[i, idx]; X = torch.stack([x] + [x[torch.randperm(m, generator=gen).to(x.device)] for _ in range(n_perm)])
        sx = torch.sign(X[:, :, None] - X[:, None, :]).reshape(1 + n_perm, -1)
        y = dz[i][:, :, idx].reshape(L_ * H_, m); sy = torch.sign(y[:, :, None] - y[:, None, :]).reshape(L_ * H_, -1)
        den = (sx.abs().sum(-1)[:, None] * sy.abs().sum(-1)[None, :]).sqrt()
        out[i] = torch.where(den > 0, (sx @ sy.T) / den.clamp(min=1e-12), torch.full_like(den, float("nan"))).reshape(1 + n_perm, L_, H_)
    return out


def _stats(base, cur, kv, tv, sev, sur, set_kind, n_perm, gen):
    """Statistics of one condition against the base run, at the readout row t of every item.
    Logit-scale quantities are divided by the SD of the base logits over the visible keys of that row, so the SESOI of 0.05 applies (7.4).
      d     dz at key k                         c     mean dz over the set                  lam   slope of dz on severity standardised within the set
      cP / lamP  the same on the keys BEFORE k (the query-side reference set, never a noise floor)
      da_k, dM   change of the weight to k and of the total mass      dH  change in row entropy      dlogit  change of the three output logits at t
      beta  slope of the BASE logits on severity over the visible keys (sign of log-priority for the high-level models)
      tau   Kendall tau between severity and dz over the set (+ permutations)      d_rand  dz at a random key of the set
    set_kind 'S': keys k < j < t (do(Priority_k)).  'before': keys j < t (do(Arousal_t), where k = t)."""
    bi = torch.arange(len(tv), device=tv.device)
    zb, zc = base["Z"][bi, :, :, tv, :], cur["Z"][bi, :, :, tv, :]; ab, ac = base["A"][bi, :, :, tv, :], cur["A"][bi, :, :, tv, :]
    B, L_, H_, T = zb.shape; pos = torch.arange(T, device=zb.device)[None, :]
    vis = pos <= tv[:, None]; Sm = ((pos > kv[:, None]) & (pos < tv[:, None])) if set_kind == "S" else (pos < tv[:, None]); Pm = pos < kv[:, None]
    v4 = vis[:, None, None, :]; zero = torch.zeros_like(zb)
    zb0 = torch.where(v4, zb, zero); dz = torch.where(v4, zc - zb, zero); da = torch.where(v4, ac - ab, zero)
    nv = vis.sum(-1).clamp(min=1)[:, None, None].float(); mu = zb0.sum(-1) / nv
    sd = ((((zb0 - mu[..., None]) ** 2) * v4).sum(-1) / nv).sqrt().clamp(min=1e-6)
    def mean_over(x, m): return (x * m[:, None, None, :]).sum(-1) / m.sum(-1).clamp(min=1)[:, None, None]
    def slope(x, m, s):
        mf = m.float(); n = mf.sum(-1).clamp(min=1); sm = (s * mf).sum(-1) / n; sv = (((s - sm[:, None]) ** 2) * mf).sum(-1) / n
        ss = (s - sm[:, None]) / sv.clamp(min=1e-12).sqrt()[:, None] * mf
        o = (x * ss[:, None, None, :]).sum(-1) / n[:, None, None]; ok = ((sv > 1e-8) & (mf.sum(-1) >= 3))[:, None, None]
        return torch.where(ok, o, torch.full_like(o, float("nan")))
    def ent(a): return -(a.clamp(min=1e-30).log() * a).sum(-1)
    o = dict(sd=sd, d=dz[bi, :, :, kv] / sd, c=mean_over(dz, Sm) / sd, lam=slope(dz, Sm, sev) / sd, cP=mean_over(dz, Pm) / sd, lamP=slope(dz, Pm, sev) / sd,
             da_k=da[bi, :, :, kv], dM=da.sum(-1), dH=ent(torch.where(v4, ac, zero)) - ent(torch.where(v4, ab, zero)),
             beta=slope(zb0, vis, sev), tau=kendall(dz, Sm, sev, n_perm, gen))
    o["c"] = torch.where((Sm.sum(-1) > 0)[:, None, None], o["c"], torch.full_like(o["c"], float("nan")))
    rk = torch.full((B,), -1, dtype=torch.long, device=zb.device)
    for i in range(B):
        idx = torch.nonzero(Sm[i]).squeeze(-1)
        if idx.numel(): rk[i] = idx[int(torch.randint(int(idx.numel()), (1,), generator=gen))]
    o["d_rand"] = torch.where((rk >= 0)[:, None, None], dz[bi, :, :, rk.clamp(min=0)] / sd, torch.full_like(sd, float("nan")))
    if sur is not None: o["lam_sur"] = torch.stack([slope(dz, Sm, sur[:, i]) / sd for i in range(sur.shape[1])], 1)
    out = {k_: v.double().cpu().numpy() for k_, v in o.items()}
    out["dlogit"] = (cur["logits"][bi, tv] - base["logits"][bi, tv]).double().cpu().numpy()
    return out


def _nan_rows(o, ok):
    """Statistics of items whose donor does not exist become NaN."""
    if ok.all(): return o
    return {k_: np.where(ok.reshape((-1,) + (1,) * (v.ndim - 1)), v, np.nan) for k_, v in o.items()}


def _batch(st, P, ix, sev_rows, sur_of_stay, dev):
    items = [_item(st, int(P["a"][i]), int(P["t"][i]) + 1) for i in ix]; b = to_dev(collate(items)); T = b["X"].shape[1]
    kv = torch.as_tensor(P["k"][ix], device=dev); tv = torch.as_tensor(P["t"][ix], device=dev)
    sev = torch.zeros((len(ix), T), device=dev); sur = torch.zeros((len(ix), 3, T), device=dev) if sur_of_stay is not None else None
    for j, i in enumerate(ix):
        a, t = int(P["a"][i]), int(P["t"][i]); sev[j, :t + 1] = torch.as_tensor(np.asarray(sev_rows[a:a + t + 1], np.float32), device=dev)
        if sur is not None: sur[j, :, :t + 1] = torch.as_tensor(np.asarray(sur_of_stay[int(P["stay"][i])][:, :t + 1], np.float32), device=dev)
    return items, b, kv, tv, sev, sur


# ---------------------------------------------------------------- do(Priority_k): S1 (primary) and S0 ----------------------------------------------------------------
@torch.no_grad()
def run_priority(model, fb, P, sur_of_stay=None, conds=("treat", "sham", "sham2"), s0=True, loose=False, cfg=None, seed=0):
    """-> {'<site>_<condition>__<statistic>': (N, ...) arrays} plus 'checks' (E1, E2, E3: maximum absolute violation, must be 0)."""
    cfg = cfg or C.CFG; _check_model(model); st = fb["store"]; dev = RT.device; n = len(P["row"]); gen = torch.Generator().manual_seed(seed)
    acc = {}; checks = dict(E1=0.0, E2=0.0, E3=0.0); bs = cfg.CAUSAL_BATCH
    def put(tag, o):
        for k_, v in o.items(): acc.setdefault(f"{tag}__{k_}", []).append(v)
    for s_ in range(0, n, bs):
        ix = np.arange(s_, min(n, s_ + bs)); items, b, kv, tv, sev, sur = _batch(st, P, ix, fb["sev"], sur_of_stay, dev)
        bi = torch.arange(len(ix), device=dev); pos = torch.arange(b["X"].shape[1], device=dev)
        base = forward(model, b); e0 = A.embed(model, b)
        if s_ < 2 * bs: checks["E1"] = max(checks["E1"], float(A.delta_z(forward(model, b)["Z"], base["Z"]).abs().max()))
        q_before = (pos[None, :] < kv[:, None])[:, None, None, :, None]
        off_k = ((pos[None, :, None] != kv[:, None, None]) & (pos[None, None, :] != kv[:, None, None]))[:, None]
        def exact(cur, layer1):
            dzz = A.delta_z(cur["Z"], base["Z"]).abs(); checks["E2"] = max(checks["E2"], float((dzz * q_before).max()))
            if layer1: checks["E3"] = max(checks["E3"], float((dzz[:, 0] * off_k).max()))
        for c in tuple(conds) + (("loose",) if loose else ()):
            rows = P[c][ix]; ok = rows >= 0
            e = e0.clone(); e[bi, kv] = donor_embedding(model, st, np.where(ok, rows, P["treat"][ix]))
            cur = forward(model, b, e=e); exact(cur, True)
            put("S1_" + c, _nan_rows(_stats(base, cur, kv, tv, sev, sur if c in ("treat", "sham") else None, "S", cfg.IIA_N_PERM if c in ("treat", "sham") else 0, gen), ok))
        if s0:
            for c in ("treat", "sham"):
                b2 = to_dev(collate([splice_input(items[j], int(P["k"][i]), int(P[c][i]), st) for j, i in enumerate(ix)]))
                cur = forward(model, b2); exact(cur, False); put("S0_" + c, _stats(base, cur, kv, tv, sev, None, "S", 0, gen))
    out = {k_: np.concatenate(v) for k_, v in acc.items()}; out["checks"] = checks; return out


# ---------------------------------------------------------------- do(Arousal_t) and its placebo ----------------------------------------------------------------
@torch.no_grad()
def run_readout(model, fb, P, feats=None, cfg=None, seed=0):
    """The inputs of the READOUT hour t (only `feats`, e.g. HR and Resp; None = all features) are replaced by a donor hour's. Keys j < t are
    untouched in every layer, so the effect is purely query-side. Check E4: outputs and attention rows before t do not change."""
    cfg = cfg or C.CFG; _check_model(model); st = fb["store"]; dev = RT.device; n = len(P["row"]); gen = torch.Generator().manual_seed(seed)
    acc = {}; checks = dict(E4=0.0); bs = cfg.CAUSAL_BATCH
    for s_ in range(0, n, bs):
        ix = np.arange(s_, min(n, s_ + bs)); items, b, kv, tv, sev, _ = _batch(st, P, ix, fb["sev"], None, dev)
        pos = torch.arange(b["X"].shape[1], device=dev); before = pos[None, :] < tv[:, None]; base = forward(model, b)
        for c in ("treat", "sham", "sham2"):
            rows = P[c][ix]; ok = rows >= 0; rows = np.where(ok, rows, P["treat"][ix])
            b2 = to_dev(collate([splice_input(items[j], int(P["t"][i]), int(rows[j]), st, feats) for j, i in enumerate(ix)])); cur = forward(model, b2)
            checks["E4"] = max(checks["E4"], float(((cur["logits"] - base["logits"]).abs() * before[:, :, None]).max()),
                               float((A.delta_z(cur["Z"], base["Z"]).abs() * before[:, None, None, :, None]).max()))
            o = _nan_rows(_stats(base, cur, kv, tv, sev, None, "before", cfg.IIA_N_PERM if c != "sham2" else 0, gen), ok)
            for k_, v in o.items(): acc.setdefault(f"A_{c}__{k_}", []).append(v)
    out = {k_: np.concatenate(v) for k_, v in acc.items()}; out["checks"] = checks; return out


# ---------------------------------------------------------------- interventions inside the attention module (S4, S5, S6) ----------------------------------------------------------------
def _attn_forward(m, X, mask=None):
    """The attention forward of the softmax and sigmoid arms, re-stated so that three interventions can be placed inside it:
    q_scale (S6: the query vector of hour t is scaled), delta (S4: added to logit z_tk before the normaliser), v_patch (S5: value vector of key k).
    With no intervention it must reproduce the module exactly; `check_patched_forward` verifies that at run time."""
    iv = m._iv; B, T = X.shape[0], X.shape[1]; bi = torch.arange(B, device=X.device)
    Q = m.W_Q(X).view(B, T, m.num_heads, m.head_dim).transpose(1, 2)
    K = m.W_K(X).view(B, T, m.num_heads, m.head_dim).transpose(1, 2)
    V = m.W_V(X).view(B, T, m.num_heads, m.head_dim).transpose(1, 2)
    if iv.get("q_scale") is not None: Q = Q.clone(); Q[bi, :, iv["t"]] = Q[bi, :, iv["t"]] * iv["q_scale"]
    if iv.get("v_patch") is not None: V = V.clone(); V[bi, :, iv["k"]] = iv["v_patch"].to(V.dtype)
    z = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(m.head_dim)
    if mask is not None: z = z.masked_fill(mask == True, float("-inf"))
    if iv.get("delta") is not None: z = z.clone(); z[bi, :, iv["t"], iv["k"]] = z[bi, :, iv["t"], iv["k"]] + iv["delta"]
    m.attention_logits = z
    if m.normalizer == "softmax": w = m.sf(z)
    elif m.normalizer == "sigmoid":
        T_t = (~mask).sum(dim=-1, keepdim=True).clamp(min=1).to(z.dtype) if mask is not None else torch.full_like(z[..., :1], float(z.shape[-1]))
        w = torch.sigmoid(z - torch.log(T_t))
    else: raise ValueError(f"unknown normalizer '{m.normalizer}'")
    m.attention_weights = w
    att = torch.matmul(m.dropout(w), V).transpose(1, 2).contiguous().view(B, T, m.embed_dim)
    att = m.W_O(att); m.Value = V
    return att


class intervene:
    """`with intervene(model, [layer indices], kv, tv, delta=...):` the listed attention modules run `_attn_forward` with that intervention."""
    def __init__(self, model, layers, kv, tv, **iv):
        mh = attention_modules(model); self.ms = [mh[l] for l in layers]; self.iv = dict(k=kv, t=tv, **iv)
        if not all(type(m) is UC.MultiHeadAttention for m in self.ms): raise NotImplementedError("in-module interventions exist for the softmax and sigmoid arms only")
    def __enter__(self):
        for m in self.ms: m._iv = self.iv; m.forward = types.MethodType(_attn_forward, m)
        return self
    def __exit__(self, *a):
        for m in self.ms:
            del m.forward; del m._iv
        return False


@torch.no_grad()
def check_patched_forward(model, fb, P, n=16):
    """Run-time proof that `_attn_forward` with no intervention equals the module's own forward (maximum absolute differences, must be 0)."""
    ix = np.arange(min(n, len(P["row"]))); items, b, kv, tv, sev, _ = _batch(fb["store"], P, ix, fb["sev"], None, RT.device); base = forward(model, b)
    with intervene(model, range(len(attention_modules(model))), kv, tv): cur = forward(model, b)
    return dict(max_abs_dz=float(A.delta_z(cur["Z"], base["Z"]).abs().max()), max_abs_dlogit=float((cur["logits"] - base["logits"]).abs().max()))


@torch.no_grad()
def run_logit_sites(model, fb, P, cfg=None, seed=0, n_max=None):
    """S4: delta added to z_tk in one layer at a time (does the output depend on the allocation?). S6: the query vector of the readout hour
    scaled in one layer >= 2 at a time (what a pure gain change looks like; reference, not evidence for M2). Model-level."""
    cfg = cfg or C.CFG; st = fb["store"]; dev = RT.device; n = min(len(P["row"]), n_max or len(P["row"])); gen = torch.Generator().manual_seed(seed)
    nL = len(attention_modules(model)); acc = {}; bs = cfg.CAUSAL_BATCH
    def put(tag, o):
        for k_ in ("dlogit", "dM", "da_k", "dH", "lam", "c"): acc.setdefault(f"{tag}__{k_}", []).append(o[k_])
    for s_ in range(0, n, bs):
        ix = np.arange(s_, min(n, s_ + bs)); items, b, kv, tv, sev, _ = _batch(st, P, ix, fb["sev"], None, dev); base = forward(model, b)
        for l in range(nL):
            with intervene(model, [l], kv, tv, delta=cfg.S4_DELTA): cur = forward(model, b)
            put(f"S4_L{l + 1}", _stats(base, cur, kv, tv, sev, None, "S", 0, gen))
        for l in range(1, nL):
            for sc in cfg.S6_SCALES:
                with intervene(model, [l], kv, tv, q_scale=sc): cur = forward(model, b)
                put(f"S6_L{l + 1}_x{sc}", _stats(base, cur, tv, tv, sev, None, "before", 0, gen))
    return {k_: np.concatenate(v) for k_, v in acc.items()}


@torch.no_grad()
def run_patch_sites(model, fb, P, cfg=None, n_max=500):
    """EXPLORATORY, LOC only (S2, S3, S5). Source run = the S1 severe splice. One activation of the source run is copied into the base run:
    S2 residual stream entering layer L at hour k; S3 attention-sublayer output and MLP output of layer L at the readout hour;
    S5 value vector of key k in layer L. -> t+1 logit at the readout for base, source and each patch (restoration is computed by `restoration`)."""
    cfg = cfg or C.CFG; _check_model(model); st = fb["store"]; dev = RT.device; n = min(len(P["row"]), n_max); layers = model.core.encoder.layers; nL = len(layers)
    acc = {}; bs = cfg.CAUSAL_BATCH
    for s_ in range(0, n, bs):
        ix = np.arange(s_, min(n, s_ + bs)); items, b, kv, tv, sev, _ = _batch(st, P, ix, fb["sev"], None, dev); bi = torch.arange(len(ix), device=dev)
        e0 = A.embed(model, b); base = forward(model, b); es = e0.clone(); es[bi, kv] = donor_embedding(model, st, P["treat"][ix])
        rec = dict(res=[None] * nL, att=[None] * nL, ffn=[None] * nL); hs = []
        for l, layer in enumerate(layers):
            hs.append(layer.register_forward_pre_hook(lambda mod, args, l=l: rec["res"].__setitem__(l, args[0].detach().clone())))
            hs.append(layer.attn_block.sublayer.register_forward_hook(lambda mod, inp, out, l=l: rec["att"].__setitem__(l, out.detach().clone())))
            hs.append(layer.ffn_block.sublayer.register_forward_hook(lambda mod, inp, out, l=l: rec["ffn"].__setitem__(l, out.detach().clone())))
        src = forward(model, b, e=es, values=True)
        for h in hs: h.remove()
        res_s = [x.clone() for x in rec["res"]]; att_s = [x.clone() for x in rec["att"]]; ffn_s = [x.clone() for x in rec["ffn"]]
        f = lambda r: r["logits"][bi, tv, 0].double().cpu().numpy()
        acc.setdefault("base", []).append(f(base)); acc.setdefault("source", []).append(f(src))
        for l in range(nL):
            def pre(mod, args, l=l):
                X = args[0].clone(); X[bi, kv] = res_s[l][bi, kv].to(X.dtype); return (X,) + tuple(args[1:])
            h = layers[l].register_forward_pre_hook(pre); acc.setdefault(f"S2_L{l + 1}", []).append(f(forward(model, b, e=e0))); h.remove()
            for route, mod, srcs in (("att", layers[l].attn_block.sublayer, att_s), ("mlp", layers[l].ffn_block.sublayer, ffn_s)):
                def post(m_, inp, out, l=l, srcs=srcs):
                    o = out.clone(); o[bi, tv] = srcs[l][bi, tv].to(o.dtype); return o
                h = mod.register_forward_hook(post); acc.setdefault(f"S3_{route}_L{l + 1}", []).append(f(forward(model, b, e=e0))); h.remove()
            if type(attention_modules(model)[l]) is UC.MultiHeadAttention:
                with intervene(model, [l], kv, tv, v_patch=src["V"][bi, l, :, kv]): acc.setdefault(f"S5_L{l + 1}", []).append(f(forward(model, b, e=e0)))
    return {k_: np.concatenate(v) for k_, v in acc.items()}


def restoration(sites, logit_sd, min_share=0.05):
    """RE = (f(patched) - f(base)) / (f(source) - f(base)); pairs with a tiny denominator (< 5% of the logit SD) are dropped."""
    den = sites["source"] - sites["base"]; ok = np.abs(den) >= min_share * logit_sd; out = dict(pairs_used=int(ok.sum()), pairs_total=int(len(den)))
    for k_, v in sites.items():
        if k_ in ("base", "source"): continue
        num = v - sites["base"]; out[k_] = dict(ratio_of_means=float(num[ok].mean() / den[ok].mean()) if ok.any() else None, median_ratio=float(np.median(num[ok] / den[ok])) if ok.any() else None)
    return out


# ---------------------------------------------------------------- bootstrap, effects, bands, decisions (7.4, 7.5) ----------------------------------------------------------------
def boot_mean(X, groups, n_boot=None, seed=0):
    """Mean over pairs (NaN ignored) with a 95% bootstrap CI that resamples SUBJECTS. X (N, ...) -> (mean, lo, hi), each of shape X.shape[1:]."""
    n_boot = n_boot or C.CFG.CAUSAL_N_BOOT; X = np.asarray(X, float); shp = X.shape[1:]; X2 = X.reshape(len(X), -1); ok = np.isfinite(X2)
    if len(X2) == 0: e = np.full(shp, np.nan); return e, e.copy(), e.copy()
    _, inv = np.unique(np.asarray(groups), return_inverse=True); G = int(inv.max()) + 1
    S = np.zeros((G, X2.shape[1])); N = np.zeros_like(S); np.add.at(S, inv, np.where(ok, X2, 0.0)); np.add.at(N, inv, ok.astype(float))
    tot = N.sum(0); mean = np.where(tot > 0, S.sum(0) / np.maximum(tot, 1), np.nan)
    W = np.random.default_rng(seed).multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)
    num, den = W @ S, W @ N
    with np.errstate(invalid="ignore", divide="ignore"): bm = np.where(den > 0, num / den, np.nan)
    with np.errstate(all="ignore"): lo, hi = np.nanpercentile(bm, [2.5, 97.5], axis=0)
    return mean.reshape(shp), lo.reshape(shp), hi.reshape(shp)


def boot_ratio(num, den, groups, n_boot=None, seed=0):
    """Ratio of means sum(num) / sum(den) over pairs with a subject-level bootstrap CI. -> (ratio, lo, hi, mean of den)."""
    n_boot = n_boot or C.CFG.CAUSAL_N_BOOT; ok = np.isfinite(num) & np.isfinite(den); num, den, g = num[ok], den[ok], np.asarray(groups)[ok]
    if len(num) == 0: return float("nan"), float("nan"), float("nan"), float("nan")
    _, inv = np.unique(g, return_inverse=True); G = int(inv.max()) + 1
    a = np.bincount(inv, weights=num, minlength=G); b = np.bincount(inv, weights=den, minlength=G)
    W = np.random.default_rng(seed).multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)
    with np.errstate(all="ignore"): r = (W @ a) / (W @ b); lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return float(a.sum() / b.sum()), float(lo), float(hi), float(den.mean())


def _ci(X, g): return dict(zip(("mean", "lo", "hi"), boot_mean(X, g)))


def effects(raw, P, part="up", site="S1"):
    """Mean 'severe - sham' contrast per head (L,h) with its subject-level CI, for every quantity of 7.4. Also: 'null_<stat>' = sham - second
    sham on the same network (the empirical noise floor), 'lam_sur' = the slope against the three surrogate severities, 'dlogit' in SD units."""
    g = P["subj"]; pre = f"{part}__{site}_"; out = {}
    if pre + "treat__d" not in raw: return out
    for s in STATS + ("cP", "lamP", "dM", "da_k", "dH"): out[s] = _ci(raw[pre + f"treat__{s}"] - raw[pre + f"sham__{s}"], g)
    if pre + "sham2__d" in raw:
        for s in STATS: out["null_" + s] = _ci(raw[pre + f"sham__{s}"] - raw[pre + f"sham2__{s}"], g)
    if pre + "treat__lam_sur" in raw: out["lam_sur"] = _ci(raw[pre + "treat__lam_sur"] - raw[pre + "sham__lam_sur"], g)
    out["dlogit"] = _ci((raw[pre + "treat__dlogit"] - raw[pre + "sham__dlogit"]) / raw["logit_sd"][None, :], g)
    return out


def thresholds(tr, init, shuf, cfg=None, stats=STATS):
    """7.5, computed on LOC: band = 95th percentile over heads in layers >= 2 of the pooled |statistic| from the untrained network, the
    v3-protocol shuffled-label network, the sham - sham contrast on the trained network and, for lam, the trained network with the three
    surrogate severities. Decision threshold = max(band, SESOI)."""
    cfg = cfg or C.CFG; out = {}
    for s in stats:
        mem = {}
        if init and s in init: mem["untrained"] = np.abs(init[s]["mean"][1:]).ravel()
        if shuf and s in shuf: mem["shuffled_v3"] = np.abs(shuf[s]["mean"][1:]).ravel()
        if "null_" + s in tr: mem["sham_minus_sham"] = np.abs(tr["null_" + s]["mean"][1:]).ravel()
        if s == "lam" and "lam_sur" in tr:
            for i, nm in enumerate(("circ_shift", "donor", "rand_proj")): mem[nm] = np.abs(tr["lam_sur"]["mean"][i][1:]).ravel()
        vals = np.concatenate(list(mem.values())) if mem else np.array([np.nan])
        band = float(np.nanpercentile(vals, cfg.CORR_BAND_Q)) if np.isfinite(vals).any() else float("nan")
        out[s] = dict(band=band, threshold=float(max(band, cfg.CORR_SESOI)) if np.isfinite(band) else cfg.CORR_SESOI,
                      members={k_: float(np.nanmax(v)) if np.isfinite(v).any() else None for k_, v in mem.items()}, complete=bool(init and shuf and "null_" + s in tr))
    return out


def passes(eff, thr):
    """Head passes on LOC: |statistic| > max(band, SESOI) with the CI excluding 0."""
    return (np.abs(eff["mean"]) > thr) & ((eff["lo"] > 0) | (eff["hi"] < 0))


def confirms(eff_loc, eff_conf, cfg=None):
    """Confirmed on CONF-A: same sign, |statistic| > SESOI, CI excluding 0."""
    cfg = cfg or C.CFG; m = eff_conf["mean"]
    return (np.sign(m) == np.sign(eff_loc["mean"])) & (np.abs(m) > cfg.CORR_SESOI) & ((eff_conf["lo"] > 0) | (eff_conf["hi"] < 0))


def count_ge2(mask): return int(np.asarray(mask)[1:].sum())


def equivalent(eff, cfg=None):
    """TOST against +/- SESOI from the bootstrap CI: equivalent iff the 90% interval lies inside (-SESOI, +SESOI); approximated with the
    normal interval implied by the 95% CI."""
    cfg = cfg or C.CFG; se = (eff["hi"] - eff["lo"]) / (2 * 1.96)
    with np.errstate(all="ignore"): return tost_p(eff["mean"], np.maximum(se, 1e-12), cfg.CORR_SESOI) < 0.05


def kappa(raw, P, part="up", site="S1", cfg=None):
    """Sigmoid budget (7.5): compensation kappa = - sum_{j != k} da_tj / da_tk, pooled over heads in layers >= 2 as a ratio of means of the
    severe - sham contrasts. 1 = full compensation (a budget), 0 = independent gates."""
    cfg = cfg or C.CFG; pre = f"{part}__{site}_"
    oth = lambda c: (raw[pre + f"{c}__dM"] - raw[pre + f"{c}__da_k"])[:, 1:].sum((1, 2)); dak = lambda c: raw[pre + f"{c}__da_k"][:, 1:].sum((1, 2))
    k, lo, hi, den = boot_ratio(-(oth("treat") - oth("sham")), dak("treat") - dak("sham"), P["subj"])
    v = ("learned budget" if (lo >= cfg.KAPPA_BUDGET[0] and hi <= cfg.KAPPA_BUDGET[1]) else
         "independent gates" if (lo >= cfg.KAPPA_INDEP[0] and hi <= cfg.KAPPA_INDEP[1]) else "partial compensation")
    return dict(kappa=k, lo=lo, hi=hi, mean_change_of_weight_to_k=den, verdict=v if np.isfinite(k) else "not computable")


def by_tertile(y, x, groups):
    """Mean of y within tertiles of x and the top - bottom difference, with subject-level CIs (interaction and dose-response checks of 7.6)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30: return dict(n=int(ok.sum()))
    y, x, g = y[ok], x[ok], np.asarray(groups)[ok]; e = np.quantile(x, [1 / 3, 2 / 3]); b = np.digitize(x, e)
    out = dict(n=int(ok.sum()), edges=[float(v) for v in e], means=[float(y[b == i].mean()) if (b == i).any() else None for i in range(3)])
    d = np.where(b == 2, y, np.nan) ; lo_ = np.where(b == 0, y, np.nan)
    rng = np.random.default_rng(0); _, inv = np.unique(g, return_inverse=True); G = int(inv.max()) + 1; diffs = []
    for _ in range(min(C.CFG.CAUSAL_N_BOOT, 1000)):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[inv].astype(float)
        with np.errstate(all="ignore"): diffs.append(np.nansum(w * d) / max((w * np.isfinite(d)).sum(), 1e-12) - np.nansum(w * lo_) / max((w * np.isfinite(lo_)).sum(), 1e-12))
    out["top_minus_bottom"] = float(np.nanmean(d) - np.nanmean(lo_)); out["ci"] = [float(v) for v in np.percentile(diffs, [2.5, 97.5])]
    return out


# ---------------------------------------------------------------- high-level models and IIA (7.7) ----------------------------------------------------------------
def beta_sign(raw, part="up"):
    """Sign of log-priority per head, fitted on LOC from the model's own base logits: sign of the mean slope of z on severity over visible keys."""
    with np.errstate(all="ignore"): return np.sign(np.nanmean(raw[f"{part}__S1_treat__beta"], 0))


def iia(raw, P, sgn, part="up", cfg=None, floor="prereg"):
    """Per pair and head: does the observed change match what each high-level model predicts?
      model predicts structure on the set  -> match iff Kendall tau(predicted, observed) >= 0.3                          (PDF)
      model predicts a constant vector     -> match iff |tau(log P, observed)| < 0.3 and d has the predicted sign       (PREREGISTRATION 7.7)
    do(Priority): M0 constant; M1 structure when k becomes the strongest item, else constant; M2 constant.
    do(Arousal) : M0 and M1 predict no change (|tau| < 0.3); M2 predicts structure.
    Floor: the same computation with log P permuted within the set ('prereg'). floor='random_key' is the PROPOSED alternative for
    constant-prediction models (d taken at a random key of the set); it may be used only after it is approved in Docs/DEVIATIONS.md.
    -> dict(match={M: (N,L,h)}, floor={M: (N,L,h)}, sham=(N,L,h) or None), float arrays with NaN where tau is undefined."""
    cfg = cfg or C.CFG; thr = cfg.IIA_TAU; s = sgn[None]; s4 = sgn[None, None]
    if part in ("up", "down"):
        pre = f"{part}__S1_"; tau = raw[pre + "treat__tau"]; t0, tp = tau[:, 0], tau[:, 1:]; ok = np.isfinite(t0)
        want = s * np.sign(P["sev_treat"] - P["sev_base"])[:, None, None]
        dsign = np.sign(raw[pre + "treat__d"]) == want
        const = (np.abs(t0) < thr) & dsign; struct = (s * t0) >= thr
        strongest = np.where(s > 0, (P["sev_treat"] > P["sev_max_others"])[:, None, None], (P["sev_treat"] < P["sev_min_others"])[:, None, None])
        match = dict(M0=const, M1=np.where(strongest, struct, const), M2=const)
        with np.errstate(invalid="ignore"):
            cf = ((np.abs(tp) < thr) & dsign[:, None]).mean(1); sf = ((s4 * tp) >= thr).mean(1)
            if floor == "random_key": cf = ((np.abs(t0) < thr) & (np.sign(raw[pre + "treat__d_rand"]) == want)).astype(float)
        fl = dict(M0=cf, M1=np.where(strongest, sf, cf), M2=cf)
        ts = raw[pre + "sham__tau"][:, 0]; sham = np.where(np.isfinite(ts), (np.abs(ts) < thr).astype(float), np.nan)
    else:
        pre = f"{part}__A_"; tau = raw[pre + "treat__tau"]; t0, tp = tau[:, 0], tau[:, 1:]; ok = np.isfinite(t0)
        const = np.abs(t0) < thr; struct = (s * t0) >= thr; match = dict(M0=const, M1=const, M2=struct)
        with np.errstate(invalid="ignore"): cf = (np.abs(tp) < thr).mean(1); sf = ((s4 * tp) >= thr).mean(1)
        fl = dict(M0=cf, M1=cf, M2=sf); ts = raw[pre + "sham__tau"][:, 0]; sham = np.where(np.isfinite(ts), (np.abs(ts) < thr).astype(float), np.nan)
    return dict(match={k_: np.where(ok, v.astype(float), np.nan) for k_, v in match.items()}, floor={k_: np.where(ok, np.asarray(v, float), np.nan) for k_, v in fl.items()}, sham=sham)


def pool_heads(x, heads):
    """(N,L,h) -> (N,): mean over the selected heads (boolean (L,h) mask), NaN ignored."""
    with np.errstate(all="ignore"): return np.nanmean(x.reshape(len(x), -1)[:, np.asarray(heads, bool).ravel()], 1)


def iia_table(I, P, heads, cfg=None):
    """IIA per model, its floor, model-minus-floor and model-minus-model differences, pooled over `heads`, with subject-level CIs."""
    g = P["subj"]; out = {}; pm = {k_: pool_heads(v, heads) for k_, v in I["match"].items()}; pf = {k_: pool_heads(v, heads) for k_, v in I["floor"].items()}
    def ci(x): m, lo, hi = boot_mean(x, g); return dict(mean=float(m), lo=float(lo), hi=float(hi))
    for k_ in pm: out[k_] = dict(iia=ci(pm[k_]), floor=ci(pf[k_]), minus_floor=ci(pm[k_] - pf[k_]))
    for a, b in (("M1", "M0"), ("M2", "M0"), ("M2", "M1")): out[f"{a}_minus_{b}"] = ci(pm[a] - pm[b])
    if I.get("sham") is not None: out["sham_pairs_no_change"] = ci(pool_heads(I["sham"], heads))
    return out


def beats(d, cfg=None):
    """Difference in IIA of at least 0.05 with the CI excluding 0."""
    cfg = cfg or C.CFG; return bool(np.isfinite(d["mean"]) and d["mean"] >= cfg.IIA_MARGIN and d["lo"] > 0)


def iia_risk(raw, P, part="up", cuts=None):
    """Secondary (P7): share of pairs whose t+1 logit moves in the predicted direction (more severe hour -> higher risk), and agreement between
    the tertile of the severity change and the tertile of the logit change. Tertile cut points come from LOC (`cuts`)."""
    pre = f"{part}__S1_"; y = (raw[pre + "treat__dlogit"] - raw[pre + "sham__dlogit"])[:, 0]; x = P["sev_treat"] - P["sev_sham"]; ok = np.isfinite(x) & np.isfinite(y)
    if cuts is None: cuts = dict(x=[float(v) for v in np.quantile(x[ok], [1 / 3, 2 / 3])], y=[float(v) for v in np.quantile(y[ok], [1 / 3, 2 / 3])])
    m, lo, hi = boot_mean((np.sign(y) == np.sign(x)).astype(float)[ok], P["subj"][ok])
    agree = float((np.digitize(x[ok], cuts["x"]) == np.digitize(y[ok], cuts["y"])).mean())
    return dict(direction=dict(mean=float(m), lo=float(lo), hi=float(hi)), tertile_agreement=agree, cuts=cuts, n=int(ok.sum()))


# ---------------------------------------------------------------- running and caching one fold ----------------------------------------------------------------
class Battery:
    """Runs the battery of one fold for any model and caches the per-pair statistics on Drive (<cache_dir>/<model>__<checkpoint>.npz),
    so a notebook can be re-run without recomputing. Nothing in the cache is ever displayed."""
    PARTS = ("up", "down", "arousal", "placebo", "sofa_t")

    def __init__(self, fb, PAIRS, sur, feats, cache_dir, cfg=None, loose=False):
        self.fb, self.P, self.sur, self.cfg, self.loose = fb, PAIRS, sur, cfg or C.CFG, loose
        feats = list(feats); self.hr_resp = [feats.index("HR"), feats.index("Resp")]; self.dir = cache_dir; os.makedirs(cache_dir, exist_ok=True); self._eff = {}

    def path(self, name, which): return os.path.join(self.dir, f"{name}__{which.replace('.pt', '')}.npz")

    def raw(self, name, which="best.pt", force=False):
        p = self.path(name, which)
        if os.path.exists(p) and not force:
            z = np.load(p, allow_pickle=False); out = {k_: z[k_] for k_ in z.files}; out["checks"] = json.load(open(p + ".checks.json")); return out
        from . import runs as R
        fb, P, cfg = self.fb, self.P, self.cfg; model = R.load_trained(name, fb["store"], which); out = {}; checks = {}; t0 = time.time()
        def add(part, r):
            checks[part] = r.pop("checks", {})
            for k_, v in r.items(): out[f"{part}__{k_}"] = v
        add("up", run_priority(model, fb, P["up"], self.sur, s0=True, loose=self.loose, cfg=cfg, seed=1))
        add("down", run_priority(model, fb, P["down"], None, conds=("treat", "sham"), s0=False, cfg=cfg, seed=2))
        add("arousal", run_readout(model, fb, P["arousal"], self.hr_resp, cfg, seed=3))
        add("placebo", run_readout(model, fb, P["placebo"], self.hr_resp, cfg, seed=4))
        add("sofa_t", run_readout(model, fb, P["sofa_t"], None, cfg, seed=5))
        rows, lg = R.predict_store(model, fb["store"]); out["logit_sd"] = lg.astype(np.float64).std(0)
        del model
        if RT.device.type == "cuda": torch.cuda.empty_cache()
        np.savez_compressed(p + ".tmp.npz", **out); os.replace(p + ".tmp.npz", p); json.dump(checks, open(p + ".checks.json", "w"), indent=1)
        print(f"  [{self.fb['fold']}] {name} ({which}): battery done in {time.time() - t0:.0f}s | exact checks, max abs violation: " +
              ", ".join(f"{part} {c}" for part, c in checks.items() if c))
        out["checks"] = checks; return out

    def eff(self, name, which="best.pt"):
        """-> {'up': S1 effects, 'up_S0': S0 effects, 'down', 'arousal', 'placebo', 'sofa_t'} (each: {quantity: dict(mean, lo, hi) of (L,h)})."""
        key = (name, which)
        if key not in self._eff:
            raw = self.raw(name, which)
            self._eff[key] = dict(up=effects(raw, self.P["up"], "up", "S1"), up_S0=effects(raw, self.P["up"], "up", "S0"), down=effects(raw, self.P["down"], "down", "S1"),
                                  arousal=effects(raw, self.P["arousal"], "arousal", "A"), placebo=effects(raw, self.P["placebo"], "placebo", "A"),
                                  sofa_t=effects(raw, self.P["sofa_t"], "sofa_t", "A"))
        return self._eff[key]


def exact_ok(checks, tol=0.0):
    """True iff every exact check of a battery run is at most `tol` (0 on CPU fp32; pass a small tolerance on GPU if cuDNN is non-deterministic)."""
    return all(float(v) <= tol for part in checks.values() for v in part.values())


def decide(loc, conf=None, part="up", stats=STATS, cfg=None):
    """One seed. loc / conf: {'trained': Battery.eff(...), 'init': ..., 'shuf': ...} on LOC and on the confirmation fold.
    Thresholds come from LOC only. A head counts iff it passes on LOC and is confirmed. Heads are counted in layers >= 2.
    -> per statistic: band, threshold, counts per network (all / positive / negative sign) and the excess E, E_pos, E_neg for the seed rule."""
    cfg = cfg or C.CFG; g = lambda d, role: (d or {}).get(role, {}).get(part, {})
    thr = thresholds(g(loc, "trained"), g(loc, "init"), g(loc, "shuf"), cfg, stats); out = {}
    for s in stats:
        t = thr[s]["threshold"]; rec = dict(band=thr[s]["band"], threshold=t, band_members=thr[s]["members"], band_complete=thr[s]["complete"]); n = {}
        for role in ("trained", "init", "shuf"):
            el = g(loc, role).get(s)
            if el is None: continue
            p = passes(el, t); ec = g(conf, role).get(s) if conf else None; ok = p & confirms(el, ec, cfg) if ec is not None else None
            use = ok if ok is not None else p; sign = np.sign((ec or el)["mean"])
            n[role] = dict(all=count_ge2(use), pos=count_ge2(use & (sign > 0)), neg=count_ge2(use & (sign < 0)))
            rec[role] = dict(pass_loc=count_ge2(p), confirmed=(count_ge2(ok) if ok is not None else None), positive=n[role]["pos"], negative=n[role]["neg"],
                             heads=[int(i) for i in np.flatnonzero(use.ravel()) if i >= use.shape[1]], max_abs=float(np.nanmax(np.abs(el["mean"][1:]))))
            rec[role]["signs"] = [int(sign.ravel()[i]) for i in rec[role]["heads"]]                 # heads are flat indices (layer * H + head), layers >= 2 only
            if role == "trained": rec["equivalent_heads_ge2"] = count_ge2(equivalent(ec or el, cfg)); rec["heads_ge2"] = int(el["mean"][1:].size)
        if all(r in n for r in ("trained", "init", "shuf")):
            for k_, nm in (("all", "E"), ("pos", "E_pos"), ("neg", "E_neg")): rec[nm] = n["trained"][k_] - max(n["init"][k_], n["shuf"][k_])
        rec["confirmed_stage"] = conf is not None; out[s] = rec
    return out


def absent_grade(recs):
    """Section 8, grading of an ABSENT verdict: 'equivalent to zero' if every analysed head is TOST-equivalent (+/- SESOI) in at least 4 of 5
    seeds; 'underpowered' if the band is below the SESOI while heads are not equivalent; otherwise 'indistinguishable from controls'."""
    eq = [r.get("equivalent_heads_ge2") == r.get("heads_ge2") for r in recs]
    if len(recs) >= C.CFG.SEED_RULE_N and sum(eq) >= C.CFG.SEED_RULE_N - 1: return "equivalent to zero"
    if all((r.get("band") is not None and np.isfinite(r["band"]) and r["band"] < C.CFG.CORR_SESOI) for r in recs): return "underpowered"
    return "indistinguishable from controls"


# ---------------------------------------------------------------- DAS (7.8): only if no hand-specified model beats the floor ----------------------------------------------------------------
def das(model, fb_train, P_train, fb_eval, P_eval, layer, cfg=None, dims=None, epochs=2, lr=1e-3, train=True, shuffle_labels=False, seed=0, n_max=None):
    """Distributed alignment search at the residual stream entering `layer` (1-based) at hour k. Linear, orthogonal R only (parametrised).
    Base = unmodified stay; source = the S1 severe splice. The first m rotated coordinates of the source replace those of the base
    (h* = h_b + R_m' R_m (h_s - h_b)); R is trained on `P_train` so that the three output logits at the readout match the source run.
    OPERATIONALISATION NOT FIXED IN THE PRE-REGISTRATION (needs approval in Docs/DEVIATIONS.md before it is used for a claim):
    a held-out pair MATCHES iff the patched t+1 logit moves in the direction of the source run and by at least half as much; pairs whose
    source effect is below 5% of the SD of the base logit are dropped.
    Controls are obtained by calling this function with: an untrained network (init.pt); shuffle_labels=True; train=False (random subspace).
    -> {m: per-pair match (N,), 'whole': the whole-component patch}."""
    cfg = cfg or C.CFG; _check_model(model); dev = RT.device; model.eval(); dims = list(dims or cfg.DAS_DIMS); bs = cfg.CAUSAL_BATCH
    for p in model.parameters(): p.requires_grad_(False)
    layers = model.core.encoder.layers; lyr = layers[layer - 1]; d_model = lyr.attn_block.normalize.normalized_shape[0]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    ntr = min(len(P_train["row"]), n_max or len(P_train["row"])); nev = min(len(P_eval["row"]), n_max or len(P_eval["row"]))
    def upper(x, mask):
        for l_ in layers[layer - 1:]: x = l_(x, mask)
        return torch.stack(model.core.head(model.core.encoder.final_norm(x)), -1)
    def tensors(fb, P, ix):
        with torch.no_grad():
            st = fb["store"]; items, b, kv, tv, sev, _ = _batch(st, P, ix, fb["sev"], None, dev); bi = torch.arange(len(ix), device=dev); rec = {}
            h = lyr.register_forward_pre_hook(lambda m_, a_: rec.__setitem__("x", a_[0].detach().clone()))
            base = forward(model, b); xb = rec["x"]; e = A.embed(model, b).clone(); e[bi, kv] = donor_embedding(model, st, P["treat"][ix])
            src = forward(model, b, e=e); xs = rec["x"]; h.remove()
        return b, bi, kv, tv, xb, xs[bi, kv] - xb[bi, kv], base["logits"][bi, tv], src["logits"][bi, tv]
    def patched(b, bi, kv, tv, xb, diff, Rm):
        x = xb.clone(); x[bi, kv] = xb[bi, kv] + (diff if Rm is None else (diff @ Rm.T) @ Rm)
        with torch.autocast(device_type=dev.type, enabled=False): y = upper(x, build_attn_mask(b["PAD"]))
        release_attn_cache(model); return y[bi, tv]
    def evaluate(Rm):
        num, den, yb_ = [], [], []
        for s_ in range(0, nev, bs):
            b, bi, kv, tv, xb, diff, yb, ys = tensors(fb_eval, P_eval, np.arange(s_, min(nev, s_ + bs)))
            with torch.no_grad(): y = patched(b, bi, kv, tv, xb, diff, Rm)
            num.append((y - yb)[:, 0].cpu().numpy()); den.append((ys - yb)[:, 0].cpu().numpy()); yb_.append(yb[:, 0].cpu().numpy())
        num, den, yb_ = np.concatenate(num), np.concatenate(den), np.concatenate(yb_); ok = np.abs(den) >= 0.05 * yb_.std()
        return np.where(ok, ((np.sign(num) == np.sign(den)) & (np.abs(num) >= 0.5 * np.abs(den))).astype(float), np.nan)
    out = {}
    for m in dims:
        lin = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(d_model, d_model, bias=False)).to(dev)
        if train:
            opt = torch.optim.Adam(lin.parameters(), lr=lr)
            for ep in range(epochs):
                order = rng.permutation(ntr)
                for s_ in range(0, ntr, bs):
                    b, bi, kv, tv, xb, diff, yb, ys = tensors(fb_train, P_train, order[s_:s_ + bs])
                    if shuffle_labels: ys = ys[torch.randperm(len(ys), device=dev)]
                    y = patched(b, bi, kv, tv, xb, diff, lin.weight[:m])
                    loss = (y - ys).pow(2).mean() / (ys - yb).pow(2).mean().clamp(min=1e-6)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad(): Rm = lin.weight[:m].detach().clone()
        out[int(m)] = evaluate(Rm); del lin
    out["whole"] = evaluate(None)
    return out


# ---------------------------------------------------------------- LOC choices: fixed and committed before CONF-A is opened ----------------------------------------------------------------
def _js(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    return str(o)


def write_loc_choices(payload, json_path, md_path):
    """Everything chosen on LOC (thresholds, heads that passed, signs for the high-level models, tertile cut points, the DAS layer, pair
    counts). The JSON stays on Drive; the Markdown file is what gets committed as Docs/LOC_CHOICES.md. Both carry the same sha256."""
    blob = json.dumps(payload, sort_keys=True, default=_js); h = hashlib.sha256(blob.encode()).hexdigest()
    os.makedirs(os.path.dirname(json_path), exist_ok=True); json.dump(dict(payload, loc_choices_id=h), open(json_path, "w"), indent=1, sort_keys=True, default=_js)
    L = ["# LOC choices (notebook 04, LOC stage)", "", f"- id (sha256 of the choices JSON): `{h}`", f"- written: {time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"- core commit: `{payload.get('core_commit')}`", f"- run: `{payload.get('run')}`", "",
         "Every choice the causal stage makes is in the JSON this id refers to (`reports/04_loc_choices.json` on Drive): FIT class boundaries, pair counts, "
         "per-model control bands and decision thresholds, the heads that passed on LOC, the sign of log-priority per head, the tertile cut points and the DAS layer.",
         "CONF-A is opened only when this file is committed as `Docs/LOC_CHOICES.md` and its id matches the JSON. Nothing below is changed after that.", "",
         "| model | d: threshold / heads passing | c: threshold / heads passing | lam: threshold / heads passing |", "|---|---|---|---|"]
    for name, rec in sorted((payload.get("models") or {}).items()):
        cell = lambda s: f"{rec[s]['threshold']:.4f} / {rec[s].get('trained', {}).get('pass_loc')}" if s in rec else "n/a"
        L.append(f"| `{name}` | {cell('d')} | {cell('c')} | {cell('lam')} |")
    open(md_path, "w").write("\n".join(L) + "\n"); return h


def verify_loc_choices(json_path, docs_dir=None):
    """-> the choices, after checking that Docs/LOC_CHOICES.md (in the repo clone, so committed) carries the id of the JSON on Drive."""
    if not os.path.exists(json_path): raise FileNotFoundError(f"{json_path} is missing: run the LOC stage of notebook 04 first")
    js = json.load(open(json_path)); md = os.path.join(docs_dir or C.PATHS.DOCS_DIR, "LOC_CHOICES.md")
    if not os.path.exists(md): raise PermissionError("Docs/LOC_CHOICES.md is not in the repo clone. Commit the file the LOC stage wrote, push, and re-run Cell 1.")
    if js["loc_choices_id"] not in open(md).read(): raise PermissionError("Docs/LOC_CHOICES.md does not carry the id of reports/04_loc_choices.json: the LOC stage was re-run after the commit. "
                                                                          "Commit the current file again, or restore the JSON that was committed.")
    return js
