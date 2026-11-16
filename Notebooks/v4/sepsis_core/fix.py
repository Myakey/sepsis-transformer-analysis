"""Track D candidates (notebook 05). Rules: Docs/PREREGISTRATION.md section 10 and Docs/PREREG_ADDENDUM_TRACK_D.md.

    decision_table / candidates   the pre-registered table of section 10, applied mechanically to the saved results of 03 and 04
    VarTok                        the one new model: variable-level observation tokens read out by an additive attention layer
    scores / variable_table       hour-level explanations of any candidate; the per-variable summary of VarTok
    rebuild / install             lets runs.train_model and runs.load_trained build a VarTok (every other model is built as before)

VarTok in one paragraph. The trunk is the matched softmax model, unchanged (same code, same hyperparameters, same initial weights for
a given seed) and gives a state h_t per hour. Every measurement is a token u_(j,v) = variable embedding + value embedding + hour
encoding (STraTS-style triplet; Tipirneni & Reddy 2022), and every hour has one extra token "hour j exists". The prediction at hour t is
        logit_k(t) = b_k + sum over heads h and visible tokens n of  alpha^h_(t,n) * c^h_(n,k)
with alpha^h_t = softmax over the tokens of hours j <= t of (W_Q h_t . W_K u_n) and c^h_(n,k) = (W_O^h W_V^h u_n)_k, a number that depends
on the token alone. So each logit is an exact sum of contributions of named measurements; the trunk only decides the weights.
Hour-level quantities (sum over the tokens of hour j) are exposed through the same attributes as an ordinary attention layer
(attention_logits, attention_weights, Value, W_O, num_heads, head_dim), so the screen of section 5, the capture code and the
faithfulness tools run on it unchanged.
Governance: aggregates only.
"""
import contextlib, math
import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import models as M
from . import runs as R
from . import explain as EX
from . import causal as CA
from .dataset import build_attn_mask
from .models import release_attn_cache

READOUT_DIM, READOUT_HEADS = 256, 4            # fixed in the addendum; not tuned
VARTOK_METHODS = ("contribution", "raw_attention", "clr", "norm_weighted", "integrated_gradients", "grad_x_input")


# ---------------------------------------------------------------- the decision table (section 10) ----------------------------------------------------------------
CANDIDATES = {
    "vartok": dict(row=1, prefix="vartok", arm="vartok", method="contribution", dv="clr", expose="contribution", trains=True,
                   label="variable-level observation tokens, additive attention readout (STraTS-style)"),
    "kernel": dict(row=1, prefix="kernel", arm="kernel", method="raw_attention", dv="clr", expose=None, trains=True,
                   label="kernel (RBF) attention in the same trunk"),
    "sigmoid": dict(row=2, prefix="sigmoid", arm="sigmoid", method="raw_attention", dv="loga", expose=None, trains=False,
                    label="sigmoid attention as the explanation layer (the trained sigmoid arm)"),
    "softmax_dz": dict(row=3, prefix="softmax", arm="softmax", method="clr", dv="clr", expose=None, trains=False,
                       label="standard softmax model, explained by the row-centred logit instead of raw attention"),
    "softmax_normw": dict(row=3, prefix="softmax", arm="softmax", method="norm_weighted", dv="normw", expose=None, trains=False,
                          label="standard softmax model, explained by norm-weighted attention instead of raw attention"),
}


def decision_table(s03, s04, primary_seed):
    """Rows of the table in section 10, read from reports/03_summary.json and reports/04_summary.json. Nothing here is a judgement call:
    every input is a saved verdict. -> dict(rows={1..4: bool}, evidence={...})."""
    soft0 = ((s03.get("rung0_soup") or {}).get("softmax") or {}).get("primary", {}).get("verdict")
    floor_heads = ((s03.get("mass") or {}).get(f"softmax_s{primary_seed}") or {}).get("floor_heads_above_5pct")
    four = (s04 or {}).get("four_outcome") or {}; rung = ((s04 or {}).get("rungs") or {}).get("softmax") or {}
    cell_absent_present = bool(four.get("cell") is not None and four.get("softmax") == "ABSENT" and four.get("sigmoid") == "PRESENT")
    floor = bool(floor_heads is not None and floor_heads > 0)
    rows = {1: soft0 == "ABSENT", 2: bool(cell_absent_present or floor),
            3: bool(rung.get("rung1_M0") and not rung.get("rung2_M1") and not rung.get("rung3_M2")),
            4: bool((s03.get("heterogeneity") or {}).get("S5_trigger"))}
    ev = dict(softmax_rung0_verdict=soft0, four_outcome=dict(softmax=four.get("softmax"), sigmoid=four.get("sigmoid"), cell=four.get("cell")),
              cell_absent_present=cell_absent_present, softmax_primary_seed_heads_with_floor_fraction_above_5pct=floor_heads,
              softmax_rungs={k_: rung.get(k_) for k_ in ("rung1_M0", "rung2_M1", "rung3_M2")}, S5_trigger=rows[4],
              iia_floor_in_force=(s04 or {}).get("iia_floor_constant_models"), have_04=bool(s04))
    return dict(rows=rows, evidence=ev)


def candidates(table, kernel_second=True, force_all=False):
    """Names of the candidates whose row fired, in the order of the table. force_all is for smoke runs only (every code path is exercised)."""
    out = [n for n, c in CANDIDATES.items() if force_all or table["rows"].get(c["row"])]
    if not kernel_second and "kernel" in out: out.remove("kernel")
    return out


def hp_for(name, base_hp):
    c = CANDIDATES[name]
    if c["arm"] == "vartok": return {**base_hp, "arm": "vartok", "readout_dim": READOUT_DIM, "readout_heads": READOUT_HEADS}
    return {**base_hp, "arm": c["arm"]}


# ---------------------------------------------------------------- the model ----------------------------------------------------------------
class ObsReadout(nn.Module):
    """The explanation layer: attention from the trunk state h_t over observation tokens, whose weighted sum IS the three logits."""
    def __init__(self, d_model, n_features, d_tok=READOUT_DIM, num_heads=READOUT_HEADS, dropout=0.1, n_out=3):
        super().__init__()
        assert d_tok % num_heads == 0, "readout_dim must be divisible by readout_heads"
        self.num_heads, self.head_dim, self.embed_dim, self.n_features = int(num_heads), d_tok // int(num_heads), int(d_tok), int(n_features)
        self.feat = nn.Embedding(n_features + 1, d_tok)                       # one vector per named variable; the last one is the token "this hour exists"
        c = max(8, int(math.sqrt(d_tok)))
        self.cve = nn.Sequential(nn.Linear(1, c), nn.Tanh(), nn.Linear(c, d_tok, bias=False))      # continuous value embedding (STraTS)
        self.pos = M.SinusoidalHourEncoding(d_tok)
        self.W_Q = nn.Linear(d_model, d_tok); self.W_K = nn.Linear(d_tok, d_tok); self.W_V = nn.Linear(d_tok, d_tok)
        self.W_O = nn.Linear(d_tok, n_out)
        self.dropout = nn.Dropout(dropout)
        self.expose, self.horizon_index, self.keep_tokens, self.chunk = "mass", 2, False, 32
        self.attention_logits = None; self.attention_weights = None; self.Value = None; self.alpha_tokens = None; self.token_contrib = None

    def forward(self, h, X, Mk, hrs, attn_mask, gate=None):
        """h (B,T,d) trunk states; X, Mk (B,T,F) z-scored values and observed mask; attn_mask (B,1,T,T) True = blocked.
        gate (B,T) in [0,1], analysis only: 1 = hour as measured, 0 = its measurements absent (the path used by the gradient methods)."""
        B, T, F = X.shape; V1 = F + 1; N = T * V1; H, dh = self.num_heads, self.head_dim; dev = X.device
        Xv = torch.where(Mk > 0, torch.nan_to_num(X), torch.zeros_like(X))                                     # a value enters only where it was measured
        u = torch.cat([self.feat.weight[:F][None, None] + self.cve(Xv.unsqueeze(-1)).float(), self.feat.weight[F].expand(B, T, 1, -1)], 2)
        u = (u + self.pos(hrs).unsqueeze(2)).reshape(B, N, self.embed_dim)
        present = torch.cat([Mk > 0, torch.ones(B, T, 1, dtype=torch.bool, device=dev)], 2)                    # (B,T,V1)
        Q = self.W_Q(h).view(B, T, H, dh).transpose(1, 2)                                                      # (B,H,T,dh)
        K = self.W_K(u).view(B, N, H, dh).transpose(1, 2); Vv = self.W_V(u).view(B, N, H, dh).transpose(1, 2)  # (B,H,N,dh)
        blocked = (attn_mask[:, :, :, :, None] | ~present[:, None, None, :, :]).reshape(B, 1, T, N)
        bias = None
        if gate is not None:
            g = torch.cat([gate[:, :, None].expand(B, T, F), torch.ones(B, T, 1, device=dev, dtype=gate.dtype)], 2).reshape(B, 1, 1, N)
            bias = torch.log(g.clamp_min(1e-12))
        extras = not self.training; ctxs, masses, zs, toks = [], [], [], []
        Wk = self.W_O.weight.float().view(-1, H, dh)                                                           # (3,H,dh)
        for s in range(0, B, self.chunk):
            sl = slice(s, s + self.chunk)
            sc = torch.matmul(Q[sl], K[sl].transpose(-2, -1)) / math.sqrt(dh)                                  # (b,H,T,N)
            if bias is not None: sc = sc + bias[sl].to(sc.dtype)
            al = torch.softmax(sc.float().masked_fill(blocked[sl], float("-inf")), dim=-1); del sc
            ctxs.append(torch.matmul(self.dropout(al).to(Vv.dtype), Vv[sl]))                                   # (b,H,T,dh)
            if extras:
                a5 = al.view(-1, H, T, T, V1); mass = a5.sum(-1); masses.append(mass)                          # (b,H,T,T): weight on the tokens of hour j
                if self.expose == "contribution":
                    c = torch.einsum("bhnd,hd->bhn", Vv[sl].float(), Wk[self.horizon_index])                   # contribution of a token if it had weight 1
                    zs.append((a5 * c.view(-1, H, 1, T, V1)).sum(-1))
                else: zs.append(torch.log(mass.clamp_min(1e-30)))
                if self.keep_tokens: toks.append(al)
        out = self.W_O(torch.cat(ctxs, 0).transpose(1, 2).reshape(B, T, H * dh))                               # (B,T,3)
        if extras:
            self.attention_weights = torch.cat(masses, 0)
            self.attention_logits = torch.cat(zs, 0).masked_fill(attn_mask, float("-inf"))
            pm = present[:, None, :, :, None].to(Vv.dtype)
            self.Value = (Vv.reshape(B, H, T, V1, dh) * pm).sum(3) / present.sum(-1).clamp(min=1)[:, None, :, None].to(Vv.dtype)   # mean value vector of the hour's tokens
            if self.keep_tokens:
                self.alpha_tokens = torch.cat(toks, 0); self.token_contrib = torch.einsum("bhnd,khd->bhnk", Vv.float(), Wk)
        return out


class _Holder(nn.Module):
    """Gives the readout the address the capture code looks for: model.layers[i].MultiHeadAttention."""
    def __init__(self, mha):
        super().__init__(); self.MultiHeadAttention = mha


class VarTok(nn.Module):
    """Matched softmax trunk (an ArmPipeline, unchanged) + ObsReadout. Same forward signature as every other model in the project.
    models.attention_modules(model) returns the readout only: it is the layer whose weights are offered as the explanation."""
    def __init__(self, base, n_features, d_tok=READOUT_DIM, num_heads=READOUT_HEADS, dropout=0.1):
        super().__init__()
        self.base = base; base.core.head = nn.Identity()                       # the trunk's own linear heads are not used: the readout produces the logits
        d = int(base.core.encoder.final_norm.normalized_shape[0])
        self.layers = nn.ModuleList([_Holder(ObsReadout(d, n_features, d_tok, num_heads, dropout))])
        self.arm = "vartok"

    @property
    def readout(self): return self.layers[0].MultiHeadAttention
    @property
    def aux_head(self): return self.base.aux_head

    def embed(self, X, Mk, D, XL, S, G, hrs, E): return self.base.embed(X, Mk, D, XL, S, G, hrs, E)

    def forward(self, X, Mk, D, XL, S, G, hrs, E, attn_mask, return_aux=False, embedding_override=None, gate=None):
        e = self.base.embed(X, Mk, D, XL, S, G, hrs, E) if embedding_override is None else embedding_override
        if gate is not None:
            e0 = self.base.embed(torch.zeros_like(X), torch.zeros_like(Mk), D, torch.zeros_like(XL), S, G, hrs, E)   # every hour "not measured"
            e = e0 + gate.unsqueeze(-1).to(e.dtype) * (e - e0)
        h, _, _ = self.base.core.encoder(e, attn_mask)
        lg = self.readout(h, X, Mk, hrs, attn_mask, gate)
        out = {"logit_t1": lg[..., 0], "logit_t3": lg[..., 1], "logit_t6": lg[..., 2]}
        if return_aux and self.base.aux_head is not None: out["aux"] = self.base.aux_head(h)
        return out


_BUILD_DEFAULT = R.rebuild


def rebuild(hp, store):
    """runs.rebuild plus the VarTok family. The trunk is built first, with the softmax arm, so for a given seed it starts from exactly the
    weights of the matched softmax model; the readout is initialised afterwards."""
    if hp.get("arm") != "vartok": return _BUILD_DEFAULT(hp, store)
    base = M.build_model({**hp, "arm": "softmax"}, store.x_mean)
    d = int(hp["d_model"]); heads = int(hp.get("readout_heads", READOUT_HEADS)); d_tok = int(min(int(hp.get("readout_dim", READOUT_DIM)), d))
    return VarTok(base, store.X.shape[1], d_tok, heads, float(hp["dropout"]))


def install():
    """Call once per notebook, after runs.setup: runs.train_model / runs.load_trained then build VarTok models too."""
    R.rebuild = rebuild


def is_vartok(model): return isinstance(model, VarTok)


def set_expose(model, mode=None, horizon_index=2):
    """What the readout exposes as `attention_logits`: 'mass' = log of the weight on each hour; 'contribution' = each head's signed
    contribution of each hour to the chosen logit. No effect on other models."""
    if is_vartok(model): model.readout.expose = mode or "mass"; model.readout.horizon_index = int(horizon_index)
    return model


@contextlib.contextmanager
def exposing(model, mode, horizon_index=2):
    ro = model.readout; old = (ro.expose, ro.horizon_index); ro.expose, ro.horizon_index = mode, int(horizon_index)
    try: yield model
    finally: ro.expose, ro.horizon_index = old


# ---------------------------------------------------------------- explanations ----------------------------------------------------------------
def _visible(sc, tv): return torch.where(torch.arange(sc.shape[1], device=tv.device)[None, :] <= tv[:, None], sc, torch.full_like(sc, float("nan")))


def gate_gradient_scores(model, b, tv, method, horizon_index=2, steps=32):
    """Integrated Gradients / gradient x input for VarTok, along the path gate: 0 (hour not measured) -> 1 (hour as measured), one path
    variable per hour. For a model whose only route is the per-hour embedding this is the same quantity as explain.gradient_scores."""
    model.eval(); B, T = b["PAD"].shape; dev = tv.device; bi = torch.arange(B, device=dev); mask = build_attn_mask(b["PAD"]); key = f"logit_t{C.HORIZONS[horizon_index]}"
    def f(g):
        with torch.autocast(device_type=dev.type, enabled=False):
            out = model(b["X"].float(), b["M"].float(), b["D"].float(), b["XL"].float(), b["S"].float(), b["G"], b["hrs"].float(), b["E"].float(), mask, gate=g)
        release_attn_cache(model); return out[key][bi, tv].sum()
    with torch.enable_grad():
        if method == "grad_x_input":
            g = torch.ones(B, T, device=dev, requires_grad=True); s, = torch.autograd.grad(f(g), g)
        elif method == "integrated_gradients":
            tot = torch.zeros(B, T, device=dev)
            for i in range(steps):
                g = torch.full((B, T), (i + 0.5) / steps, device=dev, requires_grad=True); gr, = torch.autograd.grad(f(g), g); tot += gr
            s = tot / steps
        else: raise ValueError(f"unknown gradient method '{method}'")
    return _visible(s.detach(), tv)


def methods_for(model):
    """The methods of section 10 that are defined for this model (rollout needs stacked row-normalised layers)."""
    if is_vartok(model): return list(VARTOK_METHODS)
    arm = getattr(model, "arm", "softmax")
    return [m for m in EX.METHODS if not (m == "rollout" and arm not in ("softmax", "kernel"))]


def scores(model, store, pts, method, batch=16, horizon_index=2):
    """One explanation method on every readout. -> list of 1-D arrays (one score per hour 0..t). Any model of the project."""
    if not is_vartok(model): return EX.explain(model, store, pts, method, batch=batch, horizon_index=horizon_index)
    out = []
    for s_ in range(0, len(pts["a"]), batch):
        ix = np.arange(s_, min(len(pts["a"]), s_ + batch)); _, b, tv = EX._batch(store, pts, ix); bi = torch.arange(len(ix), device=tv.device)
        if method == "contribution":
            with exposing(model, "contribution", horizon_index): z = CA.forward(model, b)["Z"][:, 0]            # (B,H,T,T), -inf where blocked
            z = z[bi, :, tv, :]; sc = _visible(torch.where(torch.isfinite(z), z, torch.zeros_like(z)).sum(1), tv)
        elif method in ("raw_attention", "clr", "norm_weighted"):
            with exposing(model, "mass", horizon_index): sc = EX.attention_scores(model, b, tv, method)
        elif method in EX.GRADIENT_METHODS: sc = gate_gradient_scores(model, b, tv, method, horizon_index)
        else: raise ValueError(f"method '{method}' is not defined for the VarTok model")
        sc = sc.float().cpu().numpy(); out += [sc[j, : int(pts["t"][i]) + 1] for j, i in enumerate(ix)]
    return out


@torch.no_grad()
def additivity_error(model, store, pts, n=16, horizon_index=2):
    """max | bias + sum of the hour contributions - logit | over a few readouts. Must be ~1e-5: the model is additive by construction."""
    ix = np.arange(min(n, len(pts["a"]))); _, b, tv = EX._batch(store, pts, ix); bi = torch.arange(len(ix), device=tv.device)
    with exposing(model, "contribution", horizon_index): r = CA.forward(model, b)
    z = r["Z"][:, 0][bi, :, tv, :]; tot = torch.where(torch.isfinite(z), z, torch.zeros_like(z)).sum((1, 2)) + model.readout.W_O.bias.float()[horizon_index]
    return float((tot - r["logits"][bi, tv, horizon_index]).abs().max())


@torch.no_grad()
def variable_table(model, store, pts, feat_names, batch=8, horizon_index=2):
    """Per named variable, over all readouts: mean share of the readout attention, mean signed contribution to the chosen logit and mean
    absolute contribution (each summed over the hours visible at the readout). Aggregates over readouts only. -> list of dicts, largest first."""
    ro = model.readout; F = ro.n_features; V1 = F + 1; acc = np.zeros((3, V1)); n = 0
    ro.keep_tokens = True
    try:
        for s_ in range(0, len(pts["a"]), batch):
            ix = np.arange(s_, min(len(pts["a"]), s_ + batch)); _, b, tv = EX._batch(store, pts, ix); bi = torch.arange(len(ix), device=tv.device)
            CA.forward(model, b); al = ro.alpha_tokens[bi, :, tv, :]; T = al.shape[-1] // V1                    # (B,H,N)
            con = (al * ro.token_contrib[..., horizon_index]).sum(1).view(len(ix), T, V1).sum(1)               # (B,V1)
            share = al.mean(1).view(len(ix), T, V1).sum(1)
            acc[0] += share.sum(0).double().cpu().numpy(); acc[1] += con.sum(0).double().cpu().numpy(); acc[2] += con.abs().sum(0).double().cpu().numpy(); n += len(ix)
            ro.alpha_tokens = None; ro.token_contrib = None
    finally:
        ro.keep_tokens = False; ro.alpha_tokens = None; ro.token_contrib = None
    names = list(feat_names)[:F] + ["(hour exists, no specific variable)"]; acc /= max(n, 1)
    rows = [dict(variable=names[v], attention_share=float(acc[0, v]), mean_contribution=float(acc[1, v]), mean_abs_contribution=float(acc[2, v])) for v in range(V1)]
    return sorted(rows, key=lambda r_: -r_["mean_abs_contribution"])
