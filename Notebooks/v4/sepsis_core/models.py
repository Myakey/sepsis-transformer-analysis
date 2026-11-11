"""Model assembly. The pipeline is v3's `SoftmaxArmPipeline` (same submodule names, so v3 checkpoints load), built from the class cells
in user_classes.py. The arm is chosen AFTER construction, so all arms consume the same RNG stream and start from identical weights.

Arms
  softmax     v3 path, untouched.
  sigmoid     a_tj = sigmoid(z_tj - log T_t)                      (primary contrast, PDF Part D)
  kernel      z_tj = -beta_h * ||q_t - k_j||^2, softmax over keys  (EXPLORATORY third variant: similarity function, not normaliser)
  kernel_raw  same z, weights = exp(z), rows NOT normalised        (exploratory sensitivity run)
"""
import math
import torch
import torch.nn as nn

from . import config as C
from . import user_classes as UC


# ----------------------------------------------------------------------------------------------------------------------
# Kernel (RBF) attention: its own class, as the brief requires. It reuses an existing MultiHeadAttention's projections.
# ----------------------------------------------------------------------------------------------------------------------
class KernelMultiHeadAttention(UC.MultiHeadAttention):
    """Distance-based attention logits: z_tj = -beta_h * ||q_t - k_j||^2 with one learned beta per head (beta = exp(log_beta) > 0).

    Why this init: ||q - k||^2 = ||q||^2 + ||k||^2 - 2 q.k, and the ||q_t||^2 term is constant within a row, so under a softmax
    z is equivalent to 2*beta*(q.k) - beta*||k_j||^2. With beta = 1 / (2*sqrt(head_dim)) the first term IS v3's scaled dot product;
    the only new ingredient at step 0 is the key-norm penalty. Everything else (W_Q, W_K, W_V, W_O, dropout) is shared code and weights.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, normalizer="softmax"):
        super().__init__(embed_dim, num_heads, dropout)
        self.normalizer = normalizer
        self.log_beta = nn.Parameter(torch.full((num_heads,), -math.log(2.0 * math.sqrt(self.head_dim))))

    @classmethod
    def from_mha(cls, mha, normalizer="softmax"):
        """Wrap an already-initialised MultiHeadAttention: same parameter tensors (step-0 equality by identity), plus log_beta (no RNG used)."""
        new = cls.__new__(cls); nn.Module.__init__(new)
        new.head_dim, new.num_heads, new.embed_dim = mha.head_dim, mha.num_heads, mha.embed_dim
        new.W_Q, new.W_K, new.W_V = mha.W_Q, mha.W_K, mha.W_V
        new.dropout, new.sf, new.W_O = mha.dropout, mha.sf, mha.W_O
        new.normalizer = normalizer
        new.log_beta = nn.Parameter(torch.full((mha.num_heads,), -math.log(2.0 * math.sqrt(mha.head_dim))))
        return new

    def forward(self, X, mask=None):
        B, T = X.shape[0], X.shape[1]
        Q = self.W_Q(X).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(X).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(X).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        d2 = (Q.pow(2).sum(-1, keepdim=True) + K.pow(2).sum(-1).unsqueeze(-2) - 2.0 * torch.matmul(Q, K.transpose(-2, -1))).clamp(min=0.0)
        attention_inside = -torch.exp(self.log_beta).to(d2.dtype).view(1, -1, 1, 1) * d2
        if mask is not None:
            attention_inside = attention_inside.masked_fill(mask == True, float("-inf"))
        self.attention_logits = attention_inside
        if self.normalizer == "softmax":
            attention_weights = self.sf(attention_inside)
        elif self.normalizer == "none":
            attention_weights = torch.exp(attention_inside)              # raw RBF kernel value in (0, 1]; masked keys -> 0
        else:
            raise ValueError(f"unknown kernel normalizer '{self.normalizer}' (softmax | none)")
        self.attention_weights = attention_weights
        attention_weights = self.dropout(attention_weights)
        attention = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        attention = self.W_O(attention)
        self.Value = V
        return attention


def set_arm(model, arm):
    """Switch the attention function of an already-built pipeline. Called once, right after construction."""
    if arm not in C.ARMS: raise ValueError(f"arm must be one of {C.ARMS}, got '{arm}'")
    for layer in model.core.encoder.layers:
        mha = layer.MultiHeadAttention
        if arm in ("softmax", "sigmoid"):
            mha.normalizer = arm
        else:
            k = KernelMultiHeadAttention.from_mha(mha, normalizer="softmax" if arm == "kernel" else "none")
            layer.attn_block.sublayer = k          # the module that runs
            layer.MultiHeadAttention = k           # the handle the capture code uses (v3 keeps both names)
    model.arm = arm
    return model


# ----------------------------------------------------------------------------------------------------------------------
# v3 assembly (ported from v3 notebook cell 7; names and maths unchanged)
# ----------------------------------------------------------------------------------------------------------------------
class SinusoidalHourEncoding(nn.Module):
    """Fixed (parameter-free) sinusoidal encoding of ICULOS, added on top of the clinical embedding.
    NOTE (stated in the lecturer notebook): because of this encoding every model knows the ICU hour, with or without the explicit ICULOS input."""
    def __init__(self, d_model, max_period=10000.0):
        super().__init__()
        half = (d_model + 1) // 2
        self.register_buffer("inv_freq", torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half), persistent=False)
        self.d_model = d_model
    def forward(self, hours):
        a = hours.unsqueeze(-1).float() * self.inv_freq
        return torch.cat([a.sin(), a.cos()], -1)[..., : self.d_model]


def iculos_feats(hrs):
    h = hrs.float().clamp(min=0)
    return torch.stack([torch.log1p(h) / math.log1p(168.0), (h / 168.0).clamp(max=2.0), (h / 24.0).clamp(max=1.0)], -1)


def eng_columns(mode):
    mode = int(mode or 0); return [] if mode == 0 else (C.ENG_LOCAL_COLS if mode == 1 else C.ENG_ALL_COLS)


class ArmPipeline(nn.Module):
    """GRU-D decay -> ProposedClinicalEmbedding (+PE) (+explicit ICULOS) (+engineered channels) -> causal Transformer -> 3 horizon heads
    [+ auxiliary next-hour forecasting head, training only]. Attention capture: core.encoder.layers[i].MultiHeadAttention.{attention_logits,
    attention_weights, Value}."""
    def __init__(self, decay, core, d_model, n_features, use_pos_enc=True, use_iculos=False, eng_cols=None, aux=False, classes=UC):
        super().__init__()
        self.decay, self.core = decay, core
        self.pos = SinusoidalHourEncoding(d_model) if use_pos_enc else None
        self.iculos_proj = nn.Linear(3, d_model) if use_iculos else None
        cols = list(eng_cols) if eng_cols else []
        self.eng_proj = nn.Linear(len(cols), d_model) if cols else None
        self.register_buffer("eng_idx", torch.tensor(cols if cols else [0], dtype=torch.long), persistent=False)
        self.aux_head = classes.NextHourForecastHead(d_model, n_features) if aux else None
        self.arm = "softmax"
    def embed(self, X, M, D, XL, S, G, hrs, E):
        x_hat, gamma = self.decay(X, M, D, XL)
        e = self.core.embedding(x_hat, M, gamma, G, S)
        if self.pos is not None: e = e + self.pos(hrs)
        if self.iculos_proj is not None: e = e + self.iculos_proj(iculos_feats(hrs))
        if self.eng_proj is not None: e = e + self.eng_proj(E.index_select(-1, self.eng_idx).float())
        return e
    def forward(self, X, M, D, XL, S, G, hrs, E, attn_mask, return_aux=False, embedding_override=None):
        e = self.embed(X, M, D, XL, S, G, hrs, E) if embedding_override is None else embedding_override
        h, _, _ = self.core.encoder(e, attn_mask)
        t1, t3, t6 = self.core.head(h)
        out = {"logit_t1": t1, "logit_t3": t3, "logit_t6": t6}
        if return_aux and self.aux_head is not None: out["aux"] = self.aux_head(h)
        return out


SoftmaxArmPipeline = ArmPipeline      # v3 name


def set_dropout(model, p):
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.p = float(p)


def release_attn_cache(model):
    for m in model.modules():
        if hasattr(m, "attention_logits"):
            m.attention_logits = None; m.attention_weights = None; m.Value = None


def build_model(hp, x_mean, n_features=None, classes=UC, cfg=None):
    """hp: full hyperparameter dict (see engine.complete_hp). `classes` lets the tests build the same model from v3_reference."""
    cfg = cfg or C.CFG
    nF = int(n_features if n_features is not None else len(x_mean))
    d = int(hp["d_model"]); nh = int(hp["num_heads"]); assert d % nh == 0, "d_model must be divisible by num_heads"
    decay = classes.GRUDInputDecay(nF, x_mean)
    emb = classes.ProposedClinicalEmbedding(nF, d, rho=cfg.RHO, sigma=cfg.SIGMA)
    enc = classes.TransformerEncoder(d, nh, int(hp["ffn_mult"]) * d, num_layers=int(hp["num_layers"]))
    model = ArmPipeline(decay, classes.SepsisForecastingModel(emb, enc, d), d, nF, use_pos_enc=bool(hp.get("pos_enc", True)),
                        use_iculos=bool(hp.get("use_iculos", False)), eng_cols=eng_columns(hp.get("eng", 0)),
                        aux=float(hp.get("aux_w", 0.0)) > 0, classes=classes)
    set_dropout(model, hp["dropout"])
    arm = hp.get("arm", "softmax")
    if arm != "softmax":
        if classes is not UC: raise ValueError("only the softmax arm exists in the v3 reference classes")
        set_arm(model, arm)
    return model


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------------------------------
# KA-Transformer under OUR protocol (Track A comparison model). Architecture from the paper; protocol from this project.
# ----------------------------------------------------------------------------------------------------------------------
class _KALayer(nn.Module):
    """Post-LN encoder layer as drawn in the paper's Fig. 3: kernel attention -> add & norm -> feed-forward -> add & norm."""
    def __init__(self, d, nh, ffn_dim, dropout):
        super().__init__()
        self.MultiHeadAttention = KernelMultiHeadAttention(d, nh, dropout=dropout, normalizer="softmax")
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d))
        self.drop = nn.Dropout(dropout)
    def forward(self, X, mask=None):
        X = self.norm1(X + self.drop(self.MultiHeadAttention(X, mask)))
        return self.norm2(X + self.drop(self.ffn(X)))


class KATransformer(nn.Module):
    """Paper: linear input projection + sinusoidal position encoding + L kernel-attention encoder layers + global average pooling + FC.
    Adapted to this project's protocol (each adaptation is logged in Docs/DECISIONS.md):
      - causal + padding mask in every layer, and a per-hour readout (the last visible position) instead of global average pooling;
      - three horizon heads (t+1/t+3/t+6) on our labels; forward-filled z-scored inputs only (no backward fill);
      - ICU length of stay, an input channel in the paper, is the explicit +/-ICULOS switch here.
    Same forward signature as ArmPipeline, so the engine, masks and tests are shared."""
    def __init__(self, n_features, d_model=512, num_heads=8, num_layers=4, ffn_dim=2048, dropout=0.2, use_iculos=False, use_mask=True):
        super().__init__()
        self.use_iculos, self.use_mask = bool(use_iculos), bool(use_mask)
        # channels: forward-filled values, [observed mask: not in the paper, on by default so KA sees what our model sees], 4 static scalars + gender, [ICULOS]
        n_in = n_features * (2 if use_mask else 1) + 5 + (3 if use_iculos else 0)
        self.input_proj = nn.Linear(n_in, d_model)
        self.pos = SinusoidalHourEncoding(d_model)
        self.layers = nn.ModuleList([_KALayer(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.head = UC.MultiHorizonHead(d_model)
        self.aux_head = None
        self.arm = "ka"
    def forward(self, X, M, D, XL, S, G, hrs, E, attn_mask, return_aux=False, embedding_override=None):
        if embedding_override is None:
            parts = [XL] + ([M] if self.use_mask else []) + [S, G.unsqueeze(-1).float()] + ([iculos_feats(hrs)] if self.use_iculos else [])
            e = self.input_proj(torch.cat(parts, -1)) + self.pos(hrs)
        else:
            e = embedding_override
        for layer in self.layers: e = layer(e, attn_mask)
        t1, t3, t6 = self.head(e)
        return {"logit_t1": t1, "logit_t3": t3, "logit_t6": t6}


def build_ka(n_features, use_iculos=False, use_mask=True, **over):
    hp = {**C.KA_HP, **over}
    return KATransformer(n_features, hp["d_model"], hp["num_heads"], hp["num_layers"], hp["ffn_dim"], hp["dropout"], use_iculos=use_iculos, use_mask=use_mask)


def attention_modules(model):
    """Per-layer attention modules, for capture, whatever the model family."""
    if hasattr(model, "core"): return [layer.MultiHeadAttention for layer in model.core.encoder.layers]
    return [layer.MultiHeadAttention for layer in model.layers]
