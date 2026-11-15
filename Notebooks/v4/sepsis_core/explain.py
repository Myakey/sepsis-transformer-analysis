"""Shared tools for Track D (notebook 05, Docs/PREREGISTRATION.md section 10). Nothing here depends on which candidate is chosen:
the candidate fix is selected from the results of 03 / 04 and recorded in Docs/PREREG_ADDENDUM_TRACK_D.md before any 05 training.

    explanation methods   raw attention, clr, norm-weighted, rollout (softmax only), Integrated Gradients, gradient x input
    faithfulness          deletion / insertion curves and AUC, comprehensiveness and sufficiency (top 20% of hours set to 'not measured')
    stability             mean pairwise Spearman correlation of the explanation ranking across seeds, on the same readouts
    scorecard             the four pre-registered criteria, the winner rule and the CONF-B agreement rule

An explanation is one score per visible hour j <= t for one readout (stay, t) and one output (default: the t+6 logit).
Models are expected to have the ArmPipeline forward signature. A candidate whose units are not hours (e.g. variable-level tokens) passes
its own `delete_fn` and scoring function; the faithfulness, stability and scorecard code does not change.
Governance: aggregates only; per-readout scores stay in memory or in Drive artifacts.
"""
import numpy as np
import torch

from . import config as C
from . import attention as A
from . import metrics as MET
from .dataset import collate, build_attn_mask
from .engine import RT, to_dev
from .models import attention_modules, release_attn_cache
from .causal import _item, delete_hours, boot_mean, forward

ATTENTION_METHODS = ("raw_attention", "clr", "norm_weighted", "rollout")
GRADIENT_METHODS = ("integrated_gradients", "grad_x_input")
METHODS = ATTENTION_METHODS + GRADIENT_METHODS


# ---------------------------------------------------------------- readouts ----------------------------------------------------------------
def sample_points(store, stay_idx, n=1000, per_stay=3, horizon_index=2, subj_of_stay=None, seed=0, cfg=None):
    """Readouts (stay, t): window position t >= CORR_MIN_T with a valid label at the chosen horizon. The same points are used for every model,
    method and seed. -> dict(a, t, stay, subj)."""
    cfg = cfg or C.CFG; rng = np.random.default_rng(seed); rec = dict(a=[], t=[], stay=[], subj=[])
    for i in rng.permutation(np.asarray(stay_idx)):
        if len(rec["a"]) >= n: break
        l = int(store.lens[i]); w = min(l, cfg.MAX_SEQ_LEN); a = int(store.starts[i]) + l - w
        ok = np.flatnonzero(store.V[a:a + w, horizon_index].astype(bool)); ok = ok[ok >= cfg.CORR_MIN_T]
        if not len(ok): continue
        for t in rng.choice(ok, size=min(per_stay, len(ok)), replace=False):
            rec["a"].append(a); rec["t"].append(int(t)); rec["stay"].append(int(i))
            rec["subj"].append(int(subj_of_stay[i]) if subj_of_stay is not None else int(store.ids[store.starts[i]]))
    return {k_: np.asarray(v, np.int64) for k_, v in rec.items()}


def _batch(store, pts, ix, items=None):
    items = items if items is not None else [_item(store, int(pts["a"][i]), int(pts["t"][i]) + 1) for i in ix]
    b = to_dev(collate(items)); tv = torch.as_tensor(pts["t"][ix], device=RT.device); return items, b, tv


# ---------------------------------------------------------------- explanation methods ----------------------------------------------------------------
@torch.no_grad()
def attention_scores(model, b, tv, method, layer="last"):
    """(B,T) scores for the readout row t; positions after t are NaN. layer='last': mean over the heads of the last layer; 'mean': over all layers."""
    r = forward(model, b, values=(method == "norm_weighted")); B = len(tv); bi = torch.arange(B, device=tv.device)
    T = r["Z"].shape[-1]; vis = torch.arange(T, device=tv.device)[None, :] <= tv[:, None]
    if method == "rollout":
        if getattr(model, "arm", "softmax") not in ("softmax", "kernel"): raise ValueError("attention rollout needs rows that sum to 1: it is defined for softmax arms only")
        eye = torch.eye(T, device=tv.device)[None]; roll = None
        for l in range(r["A"].shape[1]):
            a = 0.5 * r["A"][:, l].mean(1) + 0.5 * eye; a = a / a.sum(-1, keepdim=True).clamp(min=1e-12)
            roll = a if roll is None else torch.matmul(a, roll)
        s = roll[bi, tv]
    else:
        Z, W = r["Z"][bi, :, :, tv, :], r["A"][bi, :, :, tv, :]
        if method == "raw_attention": x = W
        elif method == "clr":
            v4 = vis[:, None, None, :]; z0 = torch.where(v4, Z, torch.zeros_like(Z)); x = z0 - z0.sum(-1, keepdim=True) / vis.sum(-1)[:, None, None, None].clamp(min=1)
        elif method == "norm_weighted":
            vn = []
            for l, m in enumerate(attention_modules(model)):
                Wr = m.W_O.weight.float().view(m.W_O.weight.shape[0], m.num_heads, m.head_dim)
                vn.append(torch.einsum("bhtk,dhk->bhtd", r["V"][:, l], Wr).norm(dim=-1))
            x = W * torch.stack(vn, 1)
        else: raise ValueError(f"unknown attention method '{method}'")
        s = x[:, -1].mean(1) if layer == "last" else x.mean((1, 2))
    return torch.where(vis, s, torch.full_like(s, float("nan")))


def baseline_embedding(model, b):
    """Reference input for Integrated Gradients: every hour 'not measured' (mask 0, values at the training mean), hour and static features kept."""
    b0 = dict(b); b0["X"] = torch.zeros_like(b["X"]); b0["M"] = torch.zeros_like(b["M"]); b0["XL"] = torch.zeros_like(b["XL"])
    return A.embed(model, b0)


def gradient_scores(model, b, tv, method, horizon_index=2, steps=32):
    """Integrated Gradients (midpoint rule, `steps` points) or gradient x input, computed on the per-hour embedding and summed over its
    dimensions: one score per hour. -> (B,T), NaN after t."""
    model.eval(); B = len(tv); bi = torch.arange(B, device=tv.device); mask = build_attn_mask(b["PAD"]); key = f"logit_t{C.HORIZONS[horizon_index]}"
    e = A.embed(model, b).detach()
    def f(emb):
        with torch.autocast(device_type=emb.device.type, enabled=False):
            out = model(b["X"].float(), b["M"].float(), b["D"].float(), b["XL"].float(), b["S"].float(), b["G"], b["hrs"].float(), b["E"].float(), mask, embedding_override=emb)
        release_attn_cache(model); return out[key][bi, tv].sum()
    with torch.enable_grad():
        if method == "grad_x_input":
            emb = e.clone().requires_grad_(True); g, = torch.autograd.grad(f(emb), emb); s = (g * e).sum(-1)
        elif method == "integrated_gradients":
            base = baseline_embedding(model, b).detach(); tot = torch.zeros_like(e)
            for i in range(steps):
                emb = (base + (i + 0.5) / steps * (e - base)).requires_grad_(True); g, = torch.autograd.grad(f(emb), emb); tot += g
            s = ((e - base) * tot / steps).sum(-1)
        else: raise ValueError(f"unknown gradient method '{method}'")
    vis = torch.arange(e.shape[1], device=tv.device)[None, :] <= tv[:, None]
    return torch.where(vis, s.detach(), torch.full_like(s, float("nan")))


def explain(model, store, pts, method, batch=16, horizon_index=2, layer="last"):
    """One method on every readout. -> list of 1-D arrays (scores of hours 0..t)."""
    out = []
    for s_ in range(0, len(pts["a"]), batch):
        ix = np.arange(s_, min(len(pts["a"]), s_ + batch)); items, b, tv = _batch(store, pts, ix)
        sc = attention_scores(model, b, tv, method, layer) if method in ATTENTION_METHODS else gradient_scores(model, b, tv, method, horizon_index)
        sc = sc.float().cpu().numpy(); out += [sc[j, : int(pts["t"][i]) + 1] for j, i in enumerate(ix)]
    return out


def random_scores(pts, seed=0):
    """Random-order baseline for the faithfulness comparison."""
    rng = np.random.default_rng(seed); return [rng.random(int(t) + 1) for t in pts["t"]]


# ---------------------------------------------------------------- faithfulness ----------------------------------------------------------------
@torch.no_grad()
def faithfulness(model, store, pts, scores, fracs=None, horizon_index=2, batch=32, delete_fn=delete_hours, by_abs=False, cfg=None):
    """Deletion and insertion along the ranking given by `scores` (largest first; by_abs=True ranks by magnitude).
      deletion curve   f after the top p% of hours are set to 'not measured'          insertion curve   f when ONLY the top p% are kept
      comprehensiveness = f(full) - f(top 20% deleted)                                 sufficiency = f(full) - f(only top 20% kept)
    f = the chosen output logit at the readout. -> dict of per-readout arrays (aggregate them with `summary`)."""
    cfg = cfg or C.CFG; fracs = list(fracs if fracs is not None else np.round(np.arange(0, 1.01, 0.1), 2)); n = len(pts["a"]); dev = RT.device
    if cfg.DELETE_FRACTION not in fracs: fracs = sorted(fracs + [cfg.DELETE_FRACTION])
    order = []
    for s in scores:
        v = np.abs(s) if by_abs else np.asarray(s, float); order.append(np.argsort(-np.nan_to_num(v, nan=-np.inf), kind="stable"))
    def run(build):
        out = np.zeros(n)
        for s_ in range(0, n, batch):
            ix = np.arange(s_, min(n, s_ + batch)); items = [build(i) for i in ix]; _, b, tv = _batch(store, pts, ix, items)
            out[ix] = forward(model, b)["logits"][torch.arange(len(ix), device=dev), tv, horizon_index].double().cpu().numpy()
        return out
    base_item = lambda i: _item(store, int(pts["a"][i]), int(pts["t"][i]) + 1)
    def first(i, fr): return order[i][: int(round(fr * len(order[i])))]
    dele = np.stack([run(lambda i: delete_fn(base_item(i), first(i, fr))) for fr in fracs], 1)
    ins = np.stack([run(lambda i: delete_fn(base_item(i), np.setdiff1d(order[i], first(i, fr)))) for fr in fracs], 1)
    j = fracs.index(cfg.DELETE_FRACTION); full = dele[:, fracs.index(0.0)] if 0.0 in fracs else run(base_item); x = np.asarray(fracs, float)
    trap = lambda y: ((y[:, 1:] + y[:, :-1]) / 2 * np.diff(x)[None, :]).sum(1)
    return dict(fracs=fracs, full=full, deletion_curve=dele, insertion_curve=ins, comprehensiveness=full - dele[:, j], sufficiency=full - ins[:, j],
                deletion_auc=trap(dele - full[:, None]), insertion_auc=trap(ins - full[:, None]))


def summary(F, groups):
    """Means with subject-level bootstrap CIs."""
    out = {}
    for k_ in ("comprehensiveness", "sufficiency", "deletion_auc", "insertion_auc"):
        m, lo, hi = boot_mean(F[k_], groups); out[k_] = dict(mean=float(m), lo=float(lo), hi=float(hi))
    out["deletion_curve"] = [float(v) for v in F["deletion_curve"].mean(0)]; out["insertion_curve"] = [float(v) for v in F["insertion_curve"].mean(0)]; out["fracs"] = [float(v) for v in F["fracs"]]
    return out


def margin(Fa, Fb, groups, key="comprehensiveness"):
    """Paired difference a - b on the same readouts, subject-level bootstrap CI (criterion 3 compares a method with random order and with raw attention)."""
    m, lo, hi = boot_mean(Fa[key] - Fb[key], groups); return dict(mean=float(m), lo=float(lo), hi=float(hi))


# ---------------------------------------------------------------- stability across seeds ----------------------------------------------------------------
def seed_stability(scores_by_seed):
    """{seed: list of per-readout score arrays (same readouts, same order)} -> mean pairwise Spearman correlation of the rankings."""
    from scipy.stats import spearmanr
    seeds = list(scores_by_seed); n = len(scores_by_seed[seeds[0]]); per = np.full(n, np.nan)
    for i in range(n):
        rs = []
        for a in range(len(seeds)):
            for b in range(a + 1, len(seeds)):
                x, y = np.asarray(scores_by_seed[seeds[a]][i], float), np.asarray(scores_by_seed[seeds[b]][i], float); ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 3 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0: rs.append(spearmanr(x[ok], y[ok])[0])
        if rs: per[i] = float(np.mean(rs))
    return dict(mean=float(np.nanmean(per)) if np.isfinite(per).any() else float("nan"), n_readouts=int(np.isfinite(per).sum()), n_seeds=len(seeds), per_readout=per)


# ---------------------------------------------------------------- scorecard (pre-registered) ----------------------------------------------------------------
def noninferiority(store, rows, cand_logits, ref_logits, n_boot=300, cfg=None):
    """AUROC(candidate) - AUROC(matched softmax model, same seed), mean of horizons, stay-level paired bootstrap."""
    cfg = cfg or C.CFG; d = MET.stay_bootstrap_dauroc(store, rows, ref_logits, cand_logits, n_boot)
    return dict(d=float(d[0]), lo=float(d[1]), hi=float(d[2]), passed=bool(d[1] > -cfg.NONINF_MARGIN))


def scorecard(name, gate_passed, noninf_primary, noninf_by_seed, sanity_verdict, comp_vs_random, comp_vs_raw_attention, stability, stability_raw_attention, cfg=None):
    """Criterion 1 is mandatory; the score is the number of criteria 2-4 passed.
      1 performance   gate D1 passed, lower bound of the AUROC difference above -0.01 at the primary seed, mean difference over seeds above -0.01
      2 sanity        the candidate's primary explanation is PRESENT by the seed rule against its own untrained and shuffled-label controls
      3 faithfulness  comprehensiveness exceeds random-order deletion AND raw attention of the standard softmax model (both CIs exclude 0)
      4 stability     mean pairwise Spearman across the 5 seeds >= 0.5 and >= that of raw attention in the standard model"""
    cfg = cfg or C.CFG; ds = [float(x["d"]) for x in noninf_by_seed]
    c1 = bool(gate_passed and noninf_primary["lo"] > -cfg.NONINF_MARGIN and len(ds) > 0 and np.mean(ds) > -cfg.NONINF_MARGIN)
    c2 = sanity_verdict == "PRESENT"; c3 = bool(comp_vs_random["lo"] > 0 and comp_vs_raw_attention["lo"] > 0)
    c4 = bool(np.isfinite(stability) and stability >= cfg.STABILITY_MIN and stability >= stability_raw_attention)
    return dict(candidate=name, performance=c1, sanity=c2, faithfulness=c3, stability=c4, eligible=c1, score=(int(c2) + int(c3) + int(c4)) if c1 else None,
                tiebreak_comprehensiveness_margin=float(comp_vs_raw_attention["mean"]),
                detail=dict(noninferiority_primary=noninf_primary, mean_difference_over_seeds=float(np.mean(ds)) if ds else None, n_seeds=len(ds), sanity_verdict=sanity_verdict,
                            comprehensiveness_vs_random=comp_vs_random, comprehensiveness_vs_raw_attention=comp_vs_raw_attention, stability=float(stability),
                            stability_raw_attention=float(stability_raw_attention)))


def pick_winner(cards):
    """The eligible candidate with the most of criteria 2-4; ties broken by the larger comprehensiveness margin. None if no candidate is eligible."""
    el = [c for c in cards if c["eligible"]]
    return max(el, key=lambda c: (c["score"], c["tiebreak_comprehensiveness_margin"]))["candidate"] if el else None


def agrees_on_conf_b(card_val, card_conf_b):
    """A candidate passes only if criteria 1-3, recomputed once on CONF-B with everything frozen, agree with VAL."""
    return all(bool(card_val[k_]) == bool(card_conf_b[k_]) for k_ in ("performance", "sanity", "faithfulness")) and bool(card_conf_b["performance"])
