"""Datasets, batching and masks. v4 change vs v3: FULL COVERAGE. Every hour of every stay is trained on once per epoch and
predicted exactly once at evaluation (v3 evaluated only each stay's most recent 168 h).

With MAX_SEQ_LEN = 336 (the longest stay) one window is the whole stay. The overlapping-window path below exists for hosts or
cohorts with longer stays and is exercised by the tests: window k starts at k*stride, and only its not-yet-predicted rows are kept,
so each row is predicted once with at least (max_len - stride) hours of history (all of its history if the stay fits).
"""
import math
import numpy as np
import torch

from . import config as C
from .user_classes import make_causal_mask


class StayDataset(torch.utils.data.Dataset):
    """mode='train': consecutive NON-overlapping windows ending at the stay's last hour (each row once per epoch; not resampling).
    mode='eval' : full coverage, each row predicted once (overlapping causal windows only if a stay exceeds max_len).
    mode='last' : v3 behaviour, the most recent max_len hours only (kept for the v3 checkpoint reproducibility check)."""
    def __init__(self, store, stay_idx, max_len, mode="eval", stride=None):
        self.st = store; self.idx = np.asarray(stay_idx); self.mode = mode
        l = store.lens[self.idx]; s = store.starts[self.idx]
        A, Nn, Kp = [], [], []
        if mode == "last":
            n = np.minimum(l, max_len); A, Nn, Kp = (s + l - n).tolist(), n.tolist(), [0] * len(n)
        elif mode == "train":
            for si, li in zip(s.tolist(), l.tolist()):
                end = si + li
                while end > si:
                    a_ = max(si, end - max_len); A.append(a_); Nn.append(end - a_); Kp.append(0); end = a_
        elif mode == "eval":
            stride = int(stride or max(1, max_len // 2))
            for si, li in zip(s.tolist(), l.tolist()):
                if li <= max_len: A.append(si); Nn.append(li); Kp.append(0); continue
                A.append(si); Nn.append(max_len); Kp.append(0); done = max_len           # first window: keep everything
                while done < li:
                    end = min(li, done + stride); a_ = end - max_len                     # window ends at `end`, full max_len of history
                    A.append(si + a_); Nn.append(max_len); Kp.append(done - a_); done = end
        else:
            raise ValueError(mode)
        self.a = np.asarray(A, np.int64); self.n = np.asarray(Nn, np.int64); self.keep_from = np.asarray(Kp, np.int64)
        self.has_pos = (store.pos_cs[self.a + self.n] - store.pos_cs[self.a]) > 0
    def __len__(self): return len(self.a)
    def __getitem__(self, i):
        a = int(self.a[i]); b = a + int(self.n[i]); st = self.st
        return dict(X=st.X[a:b], M=st.M[a:b], D=st.D[a:b], XL=st.XL[a:b], S=st.S[a:b], G=st.G[a:b], E=st.E[a:b], hrs=st.hours[a:b],
                    Y=st.Y[a:b], V=st.V[a:b], rows=np.arange(a, b), keep_from=int(self.keep_from[i]))


def collate(items):
    B = len(items); T = max(len(it["rows"]) for it in items); F = items[0]["X"].shape[1]; Hn = items[0]["Y"].shape[1]; Ed = items[0]["E"].shape[1]
    out = dict(X=np.zeros((B, T, F), np.float32), M=np.zeros((B, T, F), np.float32), D=np.zeros((B, T, F), np.float32),
               XL=np.zeros((B, T, F), np.float32), S=np.zeros((B, T, 4), np.float32), G=np.zeros((B, T), np.int64),
               E=np.zeros((B, T, Ed), np.float32), hrs=np.zeros((B, T), np.float32),
               Y=np.zeros((B, T, Hn), np.float32), V=np.zeros((B, T, Hn), np.float32),
               PAD=np.ones((B, T), bool), KEEP=np.zeros((B, T), bool), ROW=np.full((B, T), -1, np.int64))
    for i, it in enumerate(items):
        n = len(it["rows"])
        for k in ("X", "M", "D", "XL", "S", "E", "Y", "V"): out[k][i, :n] = it[k]
        out["G"][i, :n] = it["G"]; out["hrs"][i, :n] = it["hrs"]; out["PAD"][i, :n] = False; out["ROW"][i, :n] = it["rows"]
        out["KEEP"][i, it["keep_from"]:n] = True
    t = {k: torch.from_numpy(v) for k, v in out.items()}
    t["n_pos"] = int(((t["Y"] == 1) & (t["V"] > 0)).sum())
    return t


class StayBatchSampler(torch.utils.data.Sampler):
    """train: length-bucketed shuffling, every item exactly once per epoch, order a pure function of (seed, epoch) -> identical across arms.
    eval : length-sorted and deterministic; batches shrink for long sequences so B*T*T stays under `cell_budget`."""
    def __init__(self, ds, batch_size, train, seed, mega=50, cell_budget=None):
        self.ds, self.bs, self.train, self.seed, self.mega, self.epoch, self.budget = ds, batch_size, train, seed, mega, 0, cell_budget
    def set_epoch(self, e): self.epoch = e
    def _eval_batches(self):
        order = np.argsort(self.ds.n, kind="stable"); batches, i = [], 0
        while i < len(order):
            j = min(len(order), i + self.bs)
            if self.budget:
                while j > i + 1 and (j - i) * int(self.ds.n[order[j - 1]]) ** 2 > self.budget: j -= 1
            batches.append(order[i:j]); i = j
        return batches
    def __len__(self): return int(math.ceil(len(self.ds) / self.bs)) if self.train else len(self._eval_batches())
    def __iter__(self):
        n = len(self.ds); lens = self.ds.n
        if not self.train:
            batches = self._eval_batches()
        else:
            rng = np.random.default_rng(self.seed + self.epoch)
            perm = rng.permutation(n); batches = []
            for i in range(0, n, self.bs * self.mega):
                c = perm[i:i + self.bs * self.mega]; c = c[np.argsort(lens[c], kind="stable")]
                batches += [c[j:j + self.bs] for j in range(0, len(c), self.bs)]
            batches = [batches[i] for i in rng.permutation(len(batches))]
        for b in batches:
            if len(b): yield b.tolist()


def make_loader(store, stay_idx, batch_size, train, max_len=None, seed=0, mode=None, cfg=None, num_workers=None, pin=False):
    cfg = cfg or C.CFG
    mode = mode or ("train" if train else "eval")
    ds = StayDataset(store, stay_idx, max_len or cfg.MAX_SEQ_LEN, mode=mode, stride=cfg.EVAL_STRIDE)
    sampler = StayBatchSampler(ds, batch_size, train, seed, cell_budget=None if train else cfg.EVAL_CELL_BUDGET)
    nw = cfg.NUM_WORKERS if num_workers is None else num_workers
    kw = dict(num_workers=nw, pin_memory=pin)
    if nw > 0: kw.update(persistent_workers=False, prefetch_factor=4)
    return torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate, **kw), sampler


def build_attn_mask(pad):
    """(B,1,T,T) bool, True = masked: causal OR key-padding. Applied in every attention layer of every model."""
    causal = make_causal_mask(pad.shape[1], device=pad.device)
    return causal[None, None, :, :] | pad[:, None, None, :]
