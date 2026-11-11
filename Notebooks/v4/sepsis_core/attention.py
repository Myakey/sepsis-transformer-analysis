"""Attention capture and the intervention primitives the tests need. Capture is always fp32, eval mode (dropout off), so two
identical runs give identical logits. The full P1-P4 battery and the P5-P8 causal tools are added on top of these in Phase 3,
before the pre-registration is frozen.

Quantities (PDF notation): z = pre-normaliser logit (same definition in every arm), a = attention weight, t = query hour, j = key hour.
"""
import numpy as np
import torch

from .dataset import build_attn_mask
from .models import attention_modules, release_attn_cache


@torch.no_grad()
def capture(model, b, embedding_override=None, keep_weights=True):
    """One forward pass on batch `b` (already on the model's device). -> dict(Z (B,L,h,T,T), A (same) or None, VN (B,L,h,T), logits (B,T,3)).
    Masked entries of Z are -inf, exactly as the attention module stores them."""
    was_training = model.training; model.eval()
    with torch.autocast(device_type=b["X"].device.type, enabled=False):
        out = model(b["X"].float(), b["M"].float(), b["D"].float(), b["XL"].float(), b["S"].float(), b["G"], b["hrs"].float(), b["E"].float(),
                    build_attn_mask(b["PAD"]), embedding_override=embedding_override)
    mhas = attention_modules(model)
    Z = torch.stack([m.attention_logits.float() for m in mhas], 1)
    A = torch.stack([m.attention_weights.float() for m in mhas], 1) if keep_weights else None
    VN = []
    for m in mhas:
        d_ = m.W_O.weight.shape[0]; Wr = m.W_O.weight.float().view(d_, m.num_heads, m.head_dim)
        VN.append(torch.einsum("bhtk,dhk->bhtd", m.Value.float(), Wr).norm(dim=-1))          # ||v_j W_O^(h)||, Kobayashi-style
    logits = torch.stack([out["logit_t1"], out["logit_t3"], out["logit_t6"]], -1).float()
    release_attn_cache(model)
    if was_training: model.train()
    return dict(Z=Z, A=A, VN=torch.stack(VN, 1), logits=logits)


@torch.no_grad()
def embed(model, b):
    """The per-hour embedding e (B,T,d) that enters the encoder: the S1 patch site."""
    if not hasattr(model, "embed"): raise NotImplementedError("embedding-level splices are defined for the ArmPipeline family")
    return model.embed(b["X"].float(), b["M"].float(), b["D"].float(), b["XL"].float(), b["S"].float(), b["G"], b["hrs"].float(), b["E"].float())


@torch.no_grad()
def splice_embedding(model, b, k, donor_e):
    """S1: replace the embedding of hour k (every stay in the batch) by donor_e (B,d) and capture. Other hours' embeddings are untouched."""
    e = embed(model, b).clone(); e[:, k] = donor_e
    return capture(model, b, embedding_override=e)


def delta_z(src, base):
    """Dz = z(source) - z(base), with masked (-inf) entries set to 0 so they never enter a statistic."""
    d = src - base
    return torch.where(torch.isfinite(d), d, torch.zeros_like(d))


def total_mass_by_position(A, pad):
    """Mean total mass M_t = sum_j a_tj per query position t, over real queries and all layers/heads. -> (T,) numpy, NaN where no real query."""
    real = (~pad).float()                                         # (B,T)
    m = A.sum(-1).mean(dim=(1, 2))                                # (B,T)
    num = (m * real).sum(0); den = real.sum(0)
    return (num / den.clamp(min=1)).masked_fill(den == 0, float("nan")).cpu().numpy()


def clr_from_logits(Z, pad):
    """clr(alpha)_tj = z_tj - mean_{i<=t} z_ti over visible keys: the softmax normaliser cancels exactly (PDF P1). Masked -> 0."""
    vis = torch.isfinite(Z); z0 = torch.where(vis, Z, torch.zeros_like(Z))
    mean = z0.sum(-1, keepdim=True) / vis.sum(-1, keepdim=True).clamp(min=1)
    return torch.where(vis, z0 - mean, torch.zeros_like(Z))


def log_sigmoid_weight(Z):
    """Sigmoid-arm screening DV: log a_tj = log sigmoid(z_tj - log T_t), uncentred (PDF P1 / Part D). Masked -> 0."""
    vis = torch.isfinite(Z); T = vis.sum(-1, keepdim=True).clamp(min=1).float()
    return torch.where(vis, torch.nn.functional.logsigmoid(torch.where(vis, Z, torch.zeros_like(Z)) - torch.log(T)), torch.zeros_like(Z))
