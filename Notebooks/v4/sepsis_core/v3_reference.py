"""v3 class cells, extracted verbatim by script from Notebooks/sepsis_softmax_pipeline_v3.ipynb.
DO NOT EDIT. This file is the reference for the bitwise test and the source that user_classes.py is patched from.
Cell boundaries are marked `# >>> v3 cell [i] sha256=...`; each hash is of the cell source exactly as stored in the notebook."""
import math
import torch
import torch.nn as nn

# >>> v3 cell [5] sha256=fabf414129f588b434114536ac7847feb86ce745b70c256171f4c21cac685bb0
class GRUDInputDecay(nn.Module):
    """Decay-only GRU-D (Che et al., 2016/2018) -- x_hat only, no hidden state,
    no GRU gates, fully parallel across time."""
    def __init__(self, input_size, x_mean):
        super().__init__()
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
        self.w_dg_x = nn.Parameter(torch.zeros(input_size))
        self.b_dg_x = nn.Parameter(torch.zeros(input_size))

    def forward(self, X, M, Delta, X_last):
        # all inputs: [batch, seq_len, input_size] -- X_last is the forward-filled
        # last-observed raw value per feature, from Cell 1
        gamma_x = torch.exp(-torch.clamp(self.w_dg_x * Delta + self.b_dg_x, min=0))
        x_hat = M * X + (1 - M) * (gamma_x * X_last + (1 - gamma_x) * self.x_mean)
        return x_hat, gamma_x
# <<< v3 cell [5]

# >>> v3 cell [7] sha256=bbc46af61e9be791b7472ebbc28c846b13d7d07b17e1f8fc2523d38ade0da342
class ProposedClinicalEmbedding(nn.Module):
    """
    Three pathways, fractionally sized against d_model so capacity allocation
    stays identical across the small / BERT-sized / sigmoid arms:
        d_m (missingness) = rho * d_model
        d_s (static)      = sigma * d_model
        d_c (continuous)  = d_model - d_m - d_s   (remainder)

    `continuous` must be GRU-D's x_hat (decay-imputed), not a plain forward-fill --
    this module does no imputation of its own.
    `mask` / `gamma` are GRU-D's per-feature observed-mask and decay factor,
    each [*, num_continuous] -- concatenated, not mask alone.
    """
    def __init__(self, num_continuous, d_model, rho=0.25, sigma=0.15,
                 gender_vocab=2):
        super().__init__()
        self.num_continuous = num_continuous

        d_m = int(round(rho * d_model))
        d_s = int(round(sigma * d_model))
        d_c = d_model - d_m - d_s
        assert d_c > 0, "rho + sigma too large for this d_model"

        # --- continuous: vitals/labs only, via GRU-D's x_hat ---
        self.continuous_proj = nn.Linear(num_continuous, d_c)

        # --- missingness: [mask ; gamma], NOT mask alone ---
        self.missingness_proj = nn.Linear(2 * num_continuous, d_m)

        # --- static: age, hospadmtime, unit1, unit2 (raw scalars, no embedding
        #     table -- unit flags were already decided not to need one) +
        #     gender (the one categorical worth embedding, per that same decision) ---
        gender_embed_dim = max(2, d_s // 4)
        static_raw_dim = d_s - gender_embed_dim
        self.gender_embed = nn.Embedding(gender_vocab, gender_embed_dim)
        self.static_raw_proj = nn.Linear(4, static_raw_dim)  # age, hospadmtime, unit1, unit2

        # Full d_model x d_model mixing projection -- breaks the hard partition
        # between pathways rather than leaving three untouched concatenated blocks.
        self.merge = nn.Linear(d_c + d_m + d_s, d_model)

    def forward(self, continuous, mask, gamma, gender_idx, static_raw):
        """
        continuous : [*, num_continuous]       GRU-D x_hat
        mask       : [*, num_continuous]       GRU-D observed-mask
        gamma      : [*, num_continuous]       GRU-D decay factor
        gender_idx : [*]                       long
        static_raw : [*, 4]                    [age, hospadmtime, unit1, unit2], pre-normalized
        """
        c = self.continuous_proj(continuous)
        m = self.missingness_proj(torch.cat([mask, gamma], dim=-1))
        s = torch.cat([self.static_raw_proj(static_raw), self.gender_embed(gender_idx)], dim=-1)
        merged = torch.cat([c, m, s], dim=-1)
        return self.merge(merged)
# <<< v3 cell [7]

# >>> v3 cell [9] sha256=9a7a285a4306da927c87de6c17dbc579d20ad305e2d31066069893cdb410413f
import torch
import torch.nn as nn
import math

"""
This Architect implements the main transformer model (Vaswani et. al, 2017) while using LayerNormalization before each sub-layer making it the pre-LN Transformer Layer as used in modern transformers to stabilize the model during training (Xiong et. al, 2020).

Attention is defined from scratch in order to trace causality by interferance of the layers from the architecture itself during forward pass. Therefore, attention_weights and value vector are noted from each layer to analyze the architecture's decision later on.

This form of architecture using additional masking for each sequence so the model sees only the past and the present
"""

class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, dropout=0.1):
    super().__init__()

    self.head_dim = embed_dim // num_heads
    self.num_heads = num_heads
    self.embed_dim = embed_dim

    self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
    self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
    self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

    self.dropout = nn.Dropout(dropout)

    self.sf = nn.Softmax(dim=-1)

    # Weights to concate everything
    self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

  def forward(self, X, mask = None):
    Q = self.W_Q(X)
    K = self.W_K(X)
    V = self.W_V(X)

    # (Batch, Seq_length, Embed_dim)
    # (Batch, Seq_length, num_heads, head_dim)
    # (1, 6, 8, 64)
    Q = Q.view(X.shape[0], X.shape[1], self.num_heads, self.head_dim)
    K = K.view(X.shape[0], X.shape[1], self.num_heads, self.head_dim)
    V = V.view(X.shape[0], X.shape[1], self.num_heads, self.head_dim)

    # (Batch, num_heads, seq_length, head_dim)
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    attention_inside = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
    if mask is not None:
      attention_inside = attention_inside.masked_fill(mask == True, float('-inf'))

    # FLAW #3 FIX (chosen solution: Δz as primary DV, not clr-transformed α):
    # Store the pre-normalizer logits z, captured here — before softmax/sigmoid —
    # because Δz_j = z_j(severe) - z_j(routine) cancels softmax's shared log(Z)
    # term exactly, without needing clr at all. This is also the one quantity that
    # stays directly comparable between the softmax and sigmoid arms, since it's
    # captured upstream of whichever normalizer either arm applies.
    self.attention_logits = attention_inside

    attention_weights = self.sf(attention_inside)
    self.attention_weights = attention_weights

    attention_weights = self.dropout(attention_weights)

    # Attention weights (Batch, Num_heads, seq_length, seq_length)
    attention = torch.matmul(attention_weights, V)

    # Attention (Batch, Num_heads, seq_length, head_dim)
    attention = attention.transpose(1, 2)
    attention = attention.contiguous().view(X.shape[0], X.shape[1], self.embed_dim)
    attention = self.W_O(attention)
    self.Value = V

    return attention

class FeedForward(nn.Module):
  def __init__(self, embed_dim, ffn_dim):
    super().__init__()

    self.ffn = nn.Sequential(
        nn.Linear(embed_dim, ffn_dim, bias=True),
        nn.ReLU(),
        nn.Linear(ffn_dim, embed_dim, bias=True)
    )

  def forward(self, X):
    return self.ffn(X)

class ResidualAdd(nn.Module):
  def __init__(self, sublayer, embed_dim, dropout=0.1):
    super().__init__()

    self.sublayer = sublayer
    self.normalize = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, X, mask=None):
    if mask is not None :
      return X + self.dropout(self.sublayer(self.normalize(X), mask))
    return X + self.dropout(self.sublayer(self.normalize(X)))

class SingleLayer(nn.Module):
  def __init__(self, embed_dim, num_heads, ffn_dim):
    super().__init__()

    self.attn_block = ResidualAdd(MultiHeadAttention(embed_dim, num_heads), embed_dim)
    self.ffn_block = ResidualAdd(FeedForward(embed_dim, ffn_dim), embed_dim)
    self.MultiHeadAttention = self.attn_block.sublayer  # keep this handle for hooking into attention_weights later

  def forward(self, X, mask=None):
    X = self.attn_block(X, mask)
    X = self.ffn_block(X)
    return X

# The Encoder Layer of the Transformer
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            SingleLayer(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        # FINAL NORM (Pre-LN): the residual stream is never normalized inside the layers,
        # so its scale grows with depth. Normalize it once before it leaves the encoder
        # and reaches the multi-horizon head (Xiong et al., 2020).
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, X, mask=None):
        all_attention_weights = []
        all_attention_logits = []
        for layer in self.layers:
            X = layer(X, mask)
            all_attention_weights.append(layer.MultiHeadAttention.attention_weights)
            all_attention_logits.append(layer.MultiHeadAttention.attention_logits)
        X = self.final_norm(X)   # applied after the last layer, before the head
        return X, all_attention_weights, all_attention_logits

def make_causal_mask(seq_len, device=None):
    # True = blocked. Build once per batch, based on the padded sequence length.
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

def make_windowed_causal_mask(seq_len, window, device=None):
    # FLAW #4 FIX (chosen solution: sliding window as a secondary config, composed
    # with causal masking, not replacing it): visible(t, j) = (j <= t) AND (t - j < window).
    # True = blocked, same convention as make_causal_mask. Use this mask for the
    # W=24h / 48h secondary runs; keep make_causal_mask for the primary full-history run.
    idx = torch.arange(seq_len, device=device)
    t = idx.view(-1, 1)   # query position
    j = idx.view(1, -1)   # key position
    future = j > t
    too_old = (t - j) >= window
    return future | too_old
# <<< v3 cell [9]

# >>> v3 cell [11] sha256=0bbd211821250af532451ce9042cb907d9c1639593c2def47cf82c8233bedeac
class MultiHorizonHead(nn.Module):
    """
    Reads off the SAME final hidden state h_t (TransformerEncoder's output,
    X in your code) with three independent linear heads -- one forward pass,
    no decoder, no recursion, matching the I/O design from earlier.
    Horizons are fixed to t+1/t+3/t+6, not parameterized -- the causal
    battery's primary readout (C1) is bound specifically to t+1, and keeping
    that binding explicit in the code is better than hiding it behind a
    configurable list.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.head_t1 = nn.Linear(embed_dim, 1)
        self.head_t3 = nn.Linear(embed_dim, 1)
        self.head_t6 = nn.Linear(embed_dim, 1)

    def forward(self, h):
        # h: [batch, seq_len, embed_dim]
        # returns raw logits, not sigmoid-applied -- feed these straight into
        # BCEWithLogitsLoss / your focal loss variant, apply sigmoid only at inference
        return (
            self.head_t1(h).squeeze(-1),  # [batch, seq_len]
            self.head_t3(h).squeeze(-1),
            self.head_t6(h).squeeze(-1),
        )

class SepsisForecastingModel(nn.Module):
    """Composes embedding -> encoder -> multi-horizon head. attention_weights
    and attention_logits pass through untouched for the causal battery."""
    def __init__(self, embedding_module, encoder: TransformerEncoder, embed_dim):
        super().__init__()
        self.embedding = embedding_module
        self.encoder = encoder
        self.head = MultiHorizonHead(embed_dim)

    def forward(self, *embedding_args, mask=None):
        X = self.embedding(*embedding_args)
        h, attn_weights, attn_logits = self.encoder(X, mask)
        logit_t1, logit_t3, logit_t6 = self.head(h)
        return {
            "logit_t1": logit_t1, "logit_t3": logit_t3, "logit_t6": logit_t6,
            "attention_weights": attn_weights, "attention_logits": attn_logits,
        }
# <<< v3 cell [11]

# >>> v3 cell [13] sha256=7098b2a2d1796468e5ffdc2d5d1308bb78f24435f3aa7e63f90175fd75814e5f
class FocalLoss(nn.Module):
    # Lin et al. (2017), "Focal Loss for Dense Object Detection"
    def __init__(self, alpha=0.989, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets, valid_mask):
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt.clamp(min=1e-8))
        return (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
# <<< v3 cell [13]

# >>> v3 cell [15] sha256=f3394d53ceb0a2935872314465e3daebdae3616220f3d058fdf7a0b1f01ada98
# ============================ CLASSES v2 — alternative loss + weight EMA (nothing above is modified) ============================
class WeightedBCE(nn.Module):
    """Binary cross-entropy with an optional positive-class weight. NO resampling anywhere.
    pos_weight = 1  -> ordinary BCE, a proper scoring rule (probabilities come out roughly calibrated on their own).
    pos_weight > 1  -> more pressure on the rare positives, but probabilities inflate; Cell 12 (calibration) undoes that afterwards."""
    def __init__(self, pos_weight=1.0):
        super().__init__(); self.pw = float(pos_weight)
    def forward(self, logits, targets, valid_mask):
        w = torch.where(targets == 1, torch.full_like(logits, self.pw), torch.ones_like(logits))
        l = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none") * w
        return (l * valid_mask).sum() / valid_mask.sum().clamp(min=1)

class EMA:
    """Exponential moving average of the trainable parameters (Polyak averaging). Evaluation and checkpoints use the averaged weights:
    it smooths the noisy late-epoch updates that matter when only ~1% of rows are positive. Still ONE network, so attention capture is unaffected."""
    def __init__(self, model, decay):
        self.decay, self.step, self.backup = float(decay), 0, None
        self.shadow = [p.detach().clone().float() for p in model.parameters() if p.requires_grad]
    @torch.no_grad()
    def update(self, model):
        self.step += 1; d = min(self.decay, (1 + self.step) / (10 + self.step))      # warm-up: early steps are not dominated by the random init
        ps = [p.detach().float() for p in model.parameters() if p.requires_grad]
        torch._foreach_mul_(self.shadow, d); torch._foreach_add_(self.shadow, ps, alpha=1.0 - d)
    @torch.no_grad()
    def swap_in(self, model):
        ps = [p for p in model.parameters() if p.requires_grad]
        self.backup = [p.detach().clone() for p in ps]
        for p, s in zip(ps, self.shadow): p.copy_(s.to(p.dtype))
    @torch.no_grad()
    def swap_out(self, model):
        ps = [p for p in model.parameters() if p.requires_grad]
        for p, b in zip(ps, self.backup): p.copy_(b)
        self.backup = None
    def state_dict(self): return dict(step=self.step, shadow=[s.cpu() for s in self.shadow])
    def load_state_dict(self, d):
        dev = self.shadow[0].device; self.step = int(d["step"]); self.shadow = [s.to(dev) for s in d["shadow"]]
# <<< v3 cell [15]

# >>> v3 cell [17] sha256=59fea4a9bca0ee4cbf2daf093f3751f4dfd1f484774335a1749de7ab6ca704bc
# ============================ CLASSES v3 — auxiliary next-hour forecasting head + checkpoint soup (nothing above is modified) ============================
class NextHourForecastHead(nn.Module):
    """Self-supervised auxiliary task: from h_t (which, through the causal mask, has only seen hours <= t) predict the z-scored value of every
    one of the 36 decay features at hour t+1. The future value is used ONLY as a training target, never as an input, and the loss is computed
    only where that value was actually measured and the next row is exactly 1 h later in the same stay.
    Rationale: with ~1% positive rows the sepsis heads see very few positive gradients; forecasting the physiology gives the shared trunk a dense,
    label-free training signal (multi-task regularisation). SOFA / label columns are never targets either."""
    def __init__(self, embed_dim, n_features):
        super().__init__()
        self.proj = nn.Linear(embed_dim, n_features)
    def forward(self, h):                       # (B,T,d) -> (B,T,F)
        return self.proj(h)

def next_hour_forecast_loss(pred, X, M, PAD, hrs):
    """Masked Huber loss between pred[:, t] and X[:, t+1] (z-space). Only measured (M=1), non-padded, exactly-1-hour-apart pairs count."""
    tgt = X[:, 1:]
    m = M[:, 1:] * (~PAD[:, 1:]).unsqueeze(-1).float() * ((hrs[:, 1:] - hrs[:, :-1]) == 1).unsqueeze(-1).float()
    err = torch.nn.functional.smooth_l1_loss(pred[:, :-1].float(), tgt, reduction="none", beta=1.0)
    return (err * m).sum() / m.sum().clamp(min=1.0)

def average_states(states):
    """Uniform average of several state_dicts of the SAME network (one training run). Non-float tensors are copied from the first state.
    The result is still ONE network, so attention capture for the correlational / causal stages is unaffected (weight averaging, not an ensemble)."""
    out = {}
    for k, v in states[0].items():
        out[k] = (torch.stack([s[k].float() for s in states]).mean(0).to(v.dtype) if torch.is_floating_point(v) else v.clone())
    return out
# <<< v3 cell [17]
