"""Recurrent baselines for the architecture benchmark (notebook 06, lecturer track). PERFORMANCE ONLY: no interpretability claim rests
on any model in this file, and none of them enters notebooks 03-05.

    grud   GRU-D (Che et al. 2018), full: input decay toward the training mean AND hidden-state decay, mask as an input
    lstm   stacked LSTM on forward-filled values + observed mask + time since the last measurement
    rwkv   minimal RWKV-4 style network (Peng et al. 2023) in pure PyTorch, trained from scratch: time-mixing with the exact WKV
           recurrence, channel-mixing, pre-LayerNorm residual blocks. No pretrained weights, no custom CUDA kernel.

Every model has the forward signature of the project's Transformer, so the engine, the 168 h view, the loss, the schedule, early stopping,
the soup and the scoring code are shared. All are causal by construction (recurrent, or lower-triangular in time).
What each model is given: the same per-hour information the Transformer gets. Forward-filled z-scored values, the observed mask and the
time since the last measurement (GRU-D receives the raw value, the mask and the gap, and builds its own decayed input, as in the paper),
the four static scalars and gender, and the ICU hour (the Transformer knows the hour through its positional encoding).
Sizes are fixed here and are not tuned; only the learning rate is searched (three values, notebook 06).
"""
import math
import torch
import torch.nn as nn

from . import config as C
from . import models as M
from . import runs as R
from . import user_classes as UC

FAMILIES = ("grud", "lstm", "rwkv")
LABELS = dict(grud="GRU-D (input and hidden-state decay)", lstm="LSTM (2 layers)", rwkv="RWKV (minimal, from scratch)")
SIZES = dict(grud=dict(b_hidden=512), lstm=dict(b_hidden=512, b_layers=2), rwkv=dict(b_hidden=512, b_layers=4))
SMOKE_SIZES = dict(grud=dict(b_hidden=48), lstm=dict(b_hidden=48, b_layers=2), rwkv=dict(b_hidden=48, b_layers=2))
HRS_CAP = 240.0                                  # same cap as the LightGBM reference uses for "hours since the last measurement"


def _side(S, G): return torch.cat([S.float(), G.unsqueeze(-1).float()], -1)                                   # 4 static scalars + gender


def _filled(XL, Mk, D):
    """Forward-filled value, observed mask, log time since the last measurement scaled to [0, 1]: three channels per variable."""
    return torch.cat([XL.float(), Mk.float(), torch.log1p(D.float().clamp(min=0.0, max=HRS_CAP)) / math.log1p(HRS_CAP)], -1)


def _out(head, h):
    t1, t3, t6 = head(h); return {"logit_t1": t1, "logit_t3": t3, "logit_t6": t6}


# ---------------------------------------------------------------- GRU-D ----------------------------------------------------------------
class GRUD(nn.Module):
    """GRU-D. Input decay: x_hat = m*x + (1-m)*(gamma_x*x_last + (1-gamma_x)*mean), gamma_x = exp(-max(0, w*delta + b)) (the project's own
    GRUDInputDecay module, the one the Transformer uses). Hidden-state decay: h <- gamma_h * h before each update, gamma_h =
    exp(-max(0, W_h g(delta) + b_h)), with g = the gap on a log scale in [0, 1]. The GRU cell reads [x_hat, mask, statics, three scalar functions of the ICU hour]."""
    def __init__(self, n_features, x_mean, hidden=512, dropout=0.1):
        super().__init__()
        self.decay = UC.GRUDInputDecay(n_features, x_mean)
        self.hidden_decay = nn.Linear(n_features, hidden)
        self.cell = nn.GRUCell(2 * n_features + 5 + 3, hidden)
        self.drop = nn.Dropout(dropout); self.head = UC.MultiHorizonHead(hidden)
        self.layers = nn.ModuleList(); self.aux_head = None; self.arm = "grud"      # `layers` stays empty: there is no attention to capture

    def forward(self, X, Mk, D, XL, S, G, hrs, E, attn_mask, return_aux=False, embedding_override=None):
        x_hat, _ = self.decay(X, Mk, D, XL)
        inp = torch.cat([x_hat, Mk, _side(S, G), M.iculos_feats(hrs)], -1)
        gap = torch.log1p(D.float().clamp(min=0.0, max=HRS_CAP)) / math.log1p(HRS_CAP)   # the gap on a [0, 1] log scale, so the decay starts mild
        gh = torch.exp(-torch.relu(self.hidden_decay(gap)))                         # (B,T,hidden)
        h = inp.new_zeros(inp.shape[0], self.cell.hidden_size); out = []
        for t in range(inp.shape[1]):
            h = self.cell(inp[:, t], gh[:, t].to(h.dtype) * h); out.append(h)
        return _out(self.head, self.drop(torch.stack(out, 1)))


# ---------------------------------------------------------------- LSTM ----------------------------------------------------------------
class LSTMNet(nn.Module):
    def __init__(self, n_features, hidden=512, layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(3 * n_features + 5, hidden); self.pos = M.SinusoidalHourEncoding(hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, dropout=(dropout if layers > 1 else 0.0))
        self.drop = nn.Dropout(dropout); self.head = UC.MultiHorizonHead(hidden)
        self.layers = nn.ModuleList(); self.aux_head = None; self.arm = "lstm"

    def forward(self, X, Mk, D, XL, S, G, hrs, E, attn_mask, return_aux=False, embedding_override=None):
        e = self.inp(torch.cat([_filled(XL, Mk, D), _side(S, G)], -1)) + self.pos(hrs)
        h, _ = self.rnn(e)                                                          # unidirectional: hour t sees hours <= t only
        return _out(self.head, self.drop(h))


# ---------------------------------------------------------------- RWKV (RWKV-4 style) ----------------------------------------------------------------
def wkv(time_decay, time_first, k, v):
    """The RWKV-4 weighted key-value average, exact, written as one lower-triangular product over time (windows here are <= 168 h):
        wkv_t = ( sum_{i<t} e^{-(t-1-i) w + k_i} v_i + e^{u + k_t} v_t ) / ( sum_{i<t} e^{-(t-1-i) w + k_i} + e^{u + k_t} ),   w = exp(time_decay) > 0
    per channel. Computed in fp32 with k clamped, as the reference kernel clamps it. k, v: (B,T,d) -> (B,T,d)."""
    T = k.shape[1]
    with torch.autocast(device_type=k.device.type, enabled=False):
        k = k.float().clamp(min=-60.0, max=30.0); v = v.float(); w = torch.exp(time_decay.float()); u = time_first.float()
        idx = torch.arange(T, device=k.device); lag = (idx[:, None] - 1 - idx[None, :]).float()                     # (T,T): t-1-i, >= 0 for the past
        W = torch.exp(-lag.clamp(min=0.0)[None] * w[:, None, None]) * (lag >= 0)[None].float()                      # (d,T,T) past hours, decayed
        W = W + torch.eye(T, device=k.device)[None] * torch.exp(u)[:, None, None]                                   # the current hour, with its own bonus
        ek = torch.exp(k)
        num = torch.matmul(W, (ek * v).permute(2, 1, 0)); den = torch.matmul(W, ek.permute(2, 1, 0))                # (d,T,B)
        return (num / den.clamp_min(1e-30)).permute(2, 1, 0)


def _shift(x): return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)                                         # the previous hour


class RWKVTimeMix(nn.Module):
    def __init__(self, d, layer_id, n_layers):
        super().__init__()
        r01 = layer_id / max(n_layers - 1, 1); r10 = 1.0 - layer_id / n_layers; pos = torch.arange(d).float() / max(d - 1, 1)
        self.time_decay = nn.Parameter(-5.0 + 8.0 * pos ** (0.7 + 1.3 * r01))       # per-channel decay rates spread from slow (long memory) to fast
        self.time_first = nn.Parameter(torch.full((d,), math.log(0.3)))
        x = torch.arange(d).float() / d
        self.mix_k = nn.Parameter(torch.pow(x, r10)[None, None]); self.mix_v = nn.Parameter(torch.pow(x, r10)[None, None] + 0.3 * r01)
        self.mix_r = nn.Parameter(torch.pow(x, 0.5 * r10)[None, None])
        self.key = nn.Linear(d, d, bias=False); self.value = nn.Linear(d, d, bias=False); self.receptance = nn.Linear(d, d, bias=False); self.output = nn.Linear(d, d, bias=False)

    def forward(self, x):
        xx = _shift(x)
        k = self.key(x * self.mix_k + xx * (1 - self.mix_k)); v = self.value(x * self.mix_v + xx * (1 - self.mix_v))
        r = torch.sigmoid(self.receptance(x * self.mix_r + xx * (1 - self.mix_r)))
        return self.output(r * wkv(self.time_decay, self.time_first, k, v).to(r.dtype))


class RWKVChannelMix(nn.Module):
    def __init__(self, d, layer_id, n_layers):
        super().__init__()
        r10 = 1.0 - layer_id / n_layers; x = torch.arange(d).float() / d
        self.mix_k = nn.Parameter(torch.pow(x, r10)[None, None]); self.mix_r = nn.Parameter(torch.pow(x, r10)[None, None])
        self.key = nn.Linear(d, 4 * d, bias=False); self.receptance = nn.Linear(d, d, bias=False); self.value = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        xx = _shift(x)
        k = torch.square(torch.relu(self.key(x * self.mix_k + xx * (1 - self.mix_k))))
        return torch.sigmoid(self.receptance(x * self.mix_r + xx * (1 - self.mix_r))) * self.value(k)


class RWKVBlock(nn.Module):
    def __init__(self, d, layer_id, n_layers, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = RWKVTimeMix(d, layer_id, n_layers); self.ffn = RWKVChannelMix(d, layer_id, n_layers); self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.att(self.ln1(x)))
        return x + self.drop(self.ffn(self.ln2(x)))


class RWKVNet(nn.Module):
    """Input projection + hour encoding -> LayerNorm -> RWKV blocks -> LayerNorm -> three horizon heads. Initial constants follow the
    pattern of the public RWKV-4 code as far as it matters (decay rates spread across channels); they are not a copy of it."""
    def __init__(self, n_features, d=512, n_layers=4, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(3 * n_features + 5, d); self.pos = M.SinusoidalHourEncoding(d); self.ln0 = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([RWKVBlock(d, i, n_layers, dropout) for i in range(n_layers)])
        self.ln_out = nn.LayerNorm(d); self.head = UC.MultiHorizonHead(d)
        self.layers = nn.ModuleList(); self.aux_head = None; self.arm = "rwkv"

    def forward(self, X, Mk, D, XL, S, G, hrs, E, attn_mask, return_aux=False, embedding_override=None):
        x = self.ln0(self.inp(torch.cat([_filled(XL, Mk, D), _side(S, G)], -1)) + self.pos(hrs))
        for blk in self.blocks: x = blk(x)
        return _out(self.head, self.ln_out(x))


# ---------------------------------------------------------------- factory ----------------------------------------------------------------
_BUILD_DEFAULT = R.rebuild


def hp_for(family, base_hp, lr):
    """The Transformer's training settings with this family's learning rate. No auxiliary head (as for the KA-Transformer)."""
    if family not in FAMILIES: raise ValueError(f"unknown baseline family '{family}' (one of {FAMILIES})")
    return {**base_hp, "arm": family, "lr": float(lr), "aux_w": 0.0, **(SMOKE_SIZES if C.CFG.SMOKE_TEST else SIZES)[family]}


def rebuild(hp, store):
    arm = hp.get("arm")
    if arm not in FAMILIES: return _BUILD_DEFAULT(hp, store)
    nF = store.X.shape[1]; p = float(hp["dropout"]); hid = int(hp["b_hidden"])
    if arm == "grud": return GRUD(nF, store.x_mean, hidden=hid, dropout=p)
    if arm == "lstm": return LSTMNet(nF, hidden=hid, layers=int(hp["b_layers"]), dropout=p)
    return RWKVNet(nF, d=hid, n_layers=int(hp["b_layers"]), dropout=p)


def install():
    """Call once per notebook, after runs.setup: runs.train_model / runs.load_trained then build these families too."""
    R.rebuild = rebuild
