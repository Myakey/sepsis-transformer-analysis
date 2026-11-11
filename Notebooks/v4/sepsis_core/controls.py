"""Control models and the pre-registered sigmoid fallback (LayerScale in BOTH arms).

Shuffled-label controls (Docs/DECISIONS.md A2):
  v3 protocol     4 epochs on 30% of FIT, aux head on, best epoch by VAL: the PRIMARY band member, reproduced exactly from v3 cell 13.
  budget-matched  all of FIT, the trained model's schedule and number of epochs, aux head on: robustness.
  aux-off         budget-matched with aux_w = 0: decides whether "nothing beyond architecture and inputs" also holds.
None of these moves the band. The untrained control is each run's own step-0 weights (init.pt).
"""
import types
import numpy as np
import torch
import torch.nn as nn

from . import config as C


def shuffled_store(store, fit_idx, seed=None):
    """Labels permuted across FIT rows, per horizon, prevalence kept; VAL rows untouched. Same random stream as v3 (seed = SEED + 3).
    -> (store copy with permuted Y, the rng, so the caller can draw v3's 30% FIT subset from the same stream)."""
    seed = C.CFG.SEED + 3 if seed is None else seed
    st = types.SimpleNamespace(**vars(store)); Yp = store.Y.copy(); r = np.random.default_rng(seed)
    fitrows = np.flatnonzero(np.repeat(np.isin(np.arange(len(store.starts)), fit_idx), store.lens))
    for k in range(Yp.shape[1]):
        vi = fitrows[store.V[fitrows, k].astype(bool)]; Yp[vi, k] = store.Y[r.permutation(vi), k]
    st.Y = Yp; pr = ((Yp == 1) & store.V.astype(bool)).any(1)
    st.pos_cs = np.concatenate([[0], np.cumsum(pr.astype(np.int64))])
    st.stay_has_pos = np.add.reduceat(pr.astype(np.int64), store.starts) > 0
    return st, r


def v3_fit_subset(rng, fit_idx, fraction=0.3):
    """v3's shuffled-label FIT subset: drawn from the SAME rng right after the permutations."""
    return np.sort(rng.choice(fit_idx, size=max(1, int(fraction * len(fit_idx))), replace=False))


class LayerScaled(nn.Module):
    """LayerScale: the sublayer's output is multiplied by a learned per-channel gain initialised small (Ramapuram et al. 2025 use it
    to stabilise sigmoid attention). Used ONLY in the pre-registered secondary pair, and then in both arms."""
    def __init__(self, inner, dim, init=1e-4):
        super().__init__(); self.inner = inner; self.gamma = nn.Parameter(torch.full((dim,), float(init)))
    def forward(self, X, mask=None):
        out = self.inner(X, mask) if mask is not None else self.inner(X)
        return out * self.gamma.to(out.dtype)


def add_layerscale(model, init=1e-4):
    """Wrap every attention and feed-forward sublayer. Call after the arm is set. `layer.MultiHeadAttention` keeps pointing at the
    attention module itself, so capture is unchanged."""
    for layer in model.core.encoder.layers:
        d = layer.attn_block.normalize.normalized_shape[0]
        layer.attn_block.sublayer = LayerScaled(layer.attn_block.sublayer, d, init)
        layer.ffn_block.sublayer = LayerScaled(layer.ffn_block.sublayer, d, init)
    model.layerscale = float(init)
    return model
