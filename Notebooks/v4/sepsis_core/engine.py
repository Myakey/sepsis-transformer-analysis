"""Training / evaluation engine: v3's `fit` (auxiliary next-hour head, budget-matched cosine, greedy checkpoint soup) with the v4
additions needed for matched arms: arm-specific warmup, per-epoch total attention mass, a fixed-epoch mode for budget-matched
controls, a checkpoint registry (manifest.json), and no Optuna.

Matching rule: for a given seed every arm gets the same initial weights, the same batches in the same order, the same optimiser,
loss, clip and schedule; only the normaliser (and, for sigmoid, the warmup length) differs. tests.py asserts this.
"""
import os, gc, json, math, time
import numpy as np
import torch

from . import config as C
from . import tracking as T
from . import user_classes as UC
from .dataset import make_loader, build_attn_mask
from .models import build_model, release_attn_cache, count_params, attention_modules
from .metrics import compute_metrics


class Runtime:
    def __init__(self): self.device, self.amp, self.scaler = C.device_setup()
RT = Runtime()


def to_dev(b): return {k: (v.to(RT.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def complete_hp(hp, alpha=None, pw=None):
    """One full, self-describing dict for every run / checkpoint."""
    out = {**C.BASE_HP, **C.DEFAULT_FLAGS, **hp}
    if alpha is not None: out.setdefault("focal_alpha", float(alpha))
    if pw is not None: out.setdefault("pos_weight", float(pw))
    out["aux_w"] = float(out.get("aux_w", 0.0))
    if out["arm"] == "sigmoid" and "warmup_mult" not in hp: out["warmup_mult"] = C.CFG.SIGMOID_WARMUP_MULT
    return out


def make_criterion(hp):
    kind = hp.get("loss", "focal")
    if kind == "focal": return UC.FocalLoss(alpha=hp["focal_alpha"], gamma=hp["focal_gamma"])
    if kind == "wbce":  return UC.WeightedBCE(pos_weight=hp["pos_weight"])
    raise ValueError(f"unknown loss '{kind}'")


def forward_batch(model, b, with_aux=False, horizons=None):
    horizons = horizons or C.HORIZONS
    with torch.autocast(device_type=RT.device.type, dtype=RT.amp, enabled=RT.amp is not None):
        out = model(b["X"], b["M"], b["D"], b["XL"], b["S"], b["G"], b["hrs"], b["E"], build_attn_mask(b["PAD"]), return_aux=with_aux)
    lg = torch.stack([out[f"logit_t{h}"].float() for h in horizons], dim=-1)
    return (lg, out.get("aux")) if with_aux else lg


def multi_horizon_loss(criterion, logits, b, weight=None):
    V = b["V"] if weight is None else b["V"] * weight
    return sum(criterion(logits[..., k], b["Y"][..., k], V[..., k]) for k in range(logits.shape[-1]))


@torch.no_grad()
def predict_loader(model, loader, criterion=None, as_logits=False):
    """-> rows (n,), scores (n,H), mean loss. Exactly one score per real row (KEEP handles overlapping windows)."""
    model.eval(); R, P, tot, nb = [], [], 0.0, 0
    for b in loader:
        b = to_dev(b); lg = forward_batch(model, b); keep = (~b["PAD"]) & b["KEEP"]
        R.append(b["ROW"][keep].cpu().numpy()); P.append((lg[keep] if as_logits else torch.sigmoid(lg[keep])).cpu().numpy())
        if criterion is not None: tot += float(multi_horizon_loss(criterion, lg, b, b["KEEP"].unsqueeze(-1).float())); nb += 1
        release_attn_cache(model)
    return np.concatenate(R), np.concatenate(P), (tot / max(nb, 1) if criterion is not None else None)


@torch.no_grad()
def attention_mass(model, loader, n_batches=2):
    """Mean total attention mass M_t = sum_j a_tj over real queries, per layer. Exactly 1 for softmax arms; free for sigmoid / kernel_raw.
    Logged every epoch (PDF Part D): collapse toward 0 = attention bypassed; unbounded growth = scale drift."""
    model.eval(); sums = None; cnt = 0
    rng_state = torch.get_rng_state()             # iterating a DataLoader draws from the global generator; restore it afterwards so the
    try:                                          # training random stream (dropout, shuffling) is exactly what it would be without this probe
        batches = [b for i, b in zip(range(n_batches), loader)]
    finally:
        torch.set_rng_state(rng_state)
    for i, b in enumerate(batches):
        b = to_dev(b)
        with torch.autocast(device_type=RT.device.type, dtype=RT.amp, enabled=RT.amp is not None):
            model(b["X"], b["M"], b["D"], b["XL"], b["S"], b["G"], b["hrs"], b["E"], build_attn_mask(b["PAD"]))
        real = (~b["PAD"]).float()                                                         # (B,T)
        per_layer = [(m.attention_weights.float().sum(-1).mean(1) * real).sum() for m in attention_modules(model)]
        sums = per_layer if sums is None else [a + c for a, c in zip(sums, per_layer)]; cnt += float(real.sum())
        release_attn_cache(model)
    if not cnt: return {}
    per = [float(s) / cnt for s in sums]
    return {"mass_mean": float(np.mean(per)), **{f"mass_L{l + 1}": v for l, v in enumerate(per)}}


def _param_groups(model, wd):
    dec, nodec = [], []
    for n, p in model.named_parameters():
        if p.requires_grad: (nodec if p.ndim < 2 else dec).append(p)
    return [dict(params=dec, weight_decay=wd), dict(params=nodec, weight_decay=0.0)]


def _cpu_state(model): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path, model_state, hp, pre=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True); tmp = path + ".tmp"
    torch.save(dict(model=model_state, hp=hp, preproc=(pre.to_dict() if pre is not None else None), cfg=C.cfg_dict(), **(extra or {})), tmp)
    os.replace(tmp, path)


def load_pipeline(path, x_mean=None):
    from .data import Preprocessor
    ck = torch.load(path, map_location="cpu", weights_only=False)
    pre = Preprocessor.from_dict(ck["preproc"]) if ck.get("preproc") else None
    model = build_model(ck["hp"], pre.x_mean if pre is not None else x_mean); model.load_state_dict(ck["model"]); model.to(RT.device).eval()
    return model, pre, ck


def greedy_soup(model, topk, va_loader, store, sel, log_prefix=""):
    """Greedy checkpoint soup over the K best epochs of ONE run (still one network). Selection uses VAL `sel`, as in v3."""
    ranked = sorted(topk, key=lambda r: -r["score"])
    ingredients = [ranked[0]]; best_state, best_score, best_met = ranked[0]["state"], ranked[0]["score"], ranked[0]["metrics"]
    for cand in ranked[1:]:
        st = UC.average_states([r["state"] for r in ingredients + [cand]])
        model.load_state_dict(st); rows, lg, _ = predict_loader(model, va_loader, as_logits=True)
        met = compute_metrics(store, rows, lg); sc = met[sel] if np.isfinite(met[sel]) else 0.0
        if sc > best_score: ingredients.append(cand); best_state, best_score, best_met = st, sc, met
    used = [r["epoch"] for r in ingredients]
    print(f"{log_prefix}soup: epochs {used} -> {sel} {best_score:.4f} (best single epoch {ranked[0]['score']:.4f})")
    return best_state, best_score, best_met, used


def fit(hp, store, fit_idx, val_idx, *, max_epochs, patience, seed, alpha=None, pw=None, pre=None, ckpt_dir=None, resume=False,
        log_prefix="", soup_k=None, fixed_epochs=False, stop_after_epochs=None, train_dataset_wrapper=None, model_factory=None, cfg=None):
    """Train one model. Returns dict(model, history, best, single, single_state, soup_epochs, n_params, hp, init_state).

    fixed_epochs=True      run exactly max_epochs (budget-matched controls whose VAL AUROC is meaningless, e.g. shuffled labels).
    stop_after_epochs      stop after this many epochs while KEEPING the max_epochs learning-rate schedule (budget-matched controls:
                           same schedule and number of updates as the model they are matched to).
    train_dataset_wrapper  callable(dataset) -> dataset, the ONLY hook through which balancing enters; it is applied to the FIT
                           training set and nothing else (see balancing.py, which also checks the folds).
    model_factory          callable(hp, store) -> nn.Module for non-default families (KA-Transformer); default is build_model.
    """
    cfg = cfg or C.CFG; hp = complete_hp(hp, alpha, pw); sel = cfg.SELECT_METRIC; soup_k = cfg.SOUP_TOP_K if soup_k is None else int(soup_k)
    if float(hp.get("ema", 0.0)) > 0: raise NotImplementedError("weight EMA was not part of the v3 winner and is not carried into v4")
    C.seed_everything(seed)
    model = (model_factory(hp, store) if model_factory else build_model(hp, store.x_mean)).to(RT.device); n_params = count_params(model)
    init_state = _cpu_state(model)                                    # step-0 weights = the "untrained, same init" control
    criterion = make_criterion(hp); aux_w = float(hp["aux_w"]) if getattr(model, "aux_head", None) is not None else 0.0
    tr_loader, tr_sampler = make_loader(store, fit_idx, int(hp["batch_size"]), True, seed=seed, cfg=cfg, pin=RT.device.type == "cuda")
    if train_dataset_wrapper is not None:
        tr_loader, tr_sampler = _rewrap(tr_loader, tr_sampler, train_dataset_wrapper, int(hp["batch_size"]), seed, cfg)
    va_loader, _ = make_loader(store, val_idx, cfg.EVAL_BATCH, False, cfg=cfg, pin=RT.device.type == "cuda")
    opt = torch.optim.AdamW(_param_groups(model, hp["weight_decay"]), lr=hp["lr"])
    total = max(1, max_epochs * len(tr_sampler)); floor = cfg.LR_FLOOR
    warm = max(1, int(cfg.WARMUP_FRAC * float(hp.get("warmup_mult", 1.0)) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    scaler = torch.amp.GradScaler("cuda", enabled=RT.scaler)
    best = dict(score=-1.0, epoch=-1, metrics=None, state=None); bad = 0; history = []; start_ep = 0; topk = []
    last_path = os.path.join(ckpt_dir, "last.pt") if ckpt_dir else None; topk_path = os.path.join(ckpt_dir, "topk.pt") if ckpt_dir else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        if not os.path.exists(os.path.join(ckpt_dir, "init.pt")): save_checkpoint(os.path.join(ckpt_dir, "init.pt"), init_state, hp, pre, dict(seed=seed, role="step0"))
    if resume and cfg.ALLOW_RESUME and last_path and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        if ck.get("hp") == hp and ck.get("max_epochs") == max_epochs and ck.get("select") == sel and ck.get("seed") == seed and ck.get("run") == cfg.RUN_NAME:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
            best, bad, history, start_ep = ck["best"], ck["bad"], ck["history"], ck["epoch"] + 1
            if os.path.exists(topk_path): topk = torch.load(topk_path, map_location="cpu", weights_only=False)
            print(f"{log_prefix}[resume] from epoch {start_ep} (best {best['score']:.4f})")
        else: print(f"{log_prefix}[resume] checkpoint does not match run/hp/epochs/selection/seed -> starting fresh")
    print(f"{log_prefix}arm={hp['arm']} seed={seed} params={n_params:,} items/epoch={len(tr_sampler.ds):,} steps/epoch={len(tr_sampler)} bs={hp['batch_size']} "
          f"lr={hp['lr']:.2e} warmup_steps={warm} | focal(a={hp.get('focal_alpha', float('nan')):.3f}, g={hp['focal_gamma']}) aux_w={aux_w} | select={sel} soup_k={soup_k}")
    score, met, ep = best["score"], best["metrics"], start_ep - 1
    try:
        for ep in range(start_ep, max_epochs):
            if stop_after_epochs is not None and ep >= int(stop_after_epochs): break
            if bad >= patience and not fixed_epochs: break
            t0 = time.time(); model.train(); tr_sampler.set_epoch(ep)
            loss_sum = torch.zeros((), device=RT.device); aux_sum = torch.zeros((), device=RT.device); gn_sum = torch.zeros((), device=RT.device)
            nsteps = skipped = zero_pos = nb = 0
            for b in tr_loader:
                nb += 1; zero_pos += int(b["n_pos"] == 0); b = to_dev(b)
                if aux_w > 0:
                    lg, ap = forward_batch(model, b, with_aux=True)
                    aux_l = UC.next_hour_forecast_loss(ap, b["X"], b["M"], b["PAD"], b["hrs"])
                    loss = multi_horizon_loss(criterion, lg, b) + aux_w * aux_l
                else:
                    lg = forward_batch(model, b); aux_l = None; loss = multi_horizon_loss(criterion, lg, b)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                finite = bool(torch.isfinite(gn))
                if RT.scaler: scaler.step(opt); scaler.update()
                elif finite: opt.step()
                sched.step(); release_attn_cache(model)
                if finite:
                    loss_sum += loss.detach(); gn_sum += gn.detach(); nsteps += 1
                    if aux_l is not None: aux_sum += aux_l.detach()
                else: skipped += 1
            if nsteps == 0 or skipped > 0.2 * nb: raise RuntimeError(f"diverged: {skipped}/{nb} non-finite steps")
            rows, probs, vloss = predict_loader(model, va_loader, criterion)
            met = compute_metrics(store, rows, probs); score = met[sel] if np.isfinite(met[sel]) else 0.0
            mass = attention_mass(model, va_loader)
            if soup_k > 1 and (len(topk) < soup_k or score > min(r["score"] for r in topk)):
                topk.append(dict(score=score, epoch=ep, metrics=met, state=_cpu_state(model)))
                topk = sorted(topk, key=lambda r: -r["score"])[:soup_k]
                if topk_path: torch.save(topk, topk_path + ".tmp"); os.replace(topk_path + ".tmp", topk_path)
            if score > best["score"]:
                best.update(score=score, epoch=ep, metrics=met, state=_cpu_state(model)); bad = 0
                if ckpt_dir: save_checkpoint(os.path.join(ckpt_dir, "best_epoch.pt"), best["state"], hp, pre, dict(best_epoch=ep, best_metrics=met, seed=seed))
            else: bad += 1
            rec = dict(epoch=ep, train_loss=float(loss_sum) / nsteps, aux_loss=(float(aux_sum) / nsteps if aux_w > 0 else None), val_loss=vloss,
                       grad_norm=float(gn_sum) / nsteps, lr=sched.get_last_lr()[0], skipped_steps=skipped,
                       frac_zero_pos_batches=zero_pos / max(nb, 1), epoch_sec=time.time() - t0, **met, **mass)
            history.append(rec); T.m_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=ep)
            print(f"{log_prefix}ep{ep:02d} loss={rec['train_loss']:.4f}" + (f" aux={rec['aux_loss']:.3f}" if aux_w > 0 else "") +
                  f" val_loss={vloss:.4f} | AUROC {met['auroc_mean']:.4f} AUPRC {met['auprc_mean']:.4f} | mass {mass.get('mass_mean', float('nan')):.3f} | "
                  f"lr {rec['lr']:.2e} | {rec['epoch_sec']:.0f}s")
            if last_path:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(), best=best, bad=bad,
                                history=history, epoch=ep, hp=hp, max_epochs=max_epochs, select=sel, seed=seed, run=cfg.RUN_NAME), last_path + ".tmp")
                os.replace(last_path + ".tmp", last_path)
            if bad >= patience and not fixed_epochs: print(f"{log_prefix}early stop (no {sel} gain for {patience} epochs)"); break
        if best["state"] is None: raise RuntimeError("no epoch finished")
        single = dict(score=best["score"], epoch=best["epoch"], metrics=best["metrics"])
        final_state, final_score, final_met, soup_epochs = best["state"], best["score"], best["metrics"], [best["epoch"]]
        if fixed_epochs:                                  # a control is "whatever the last epoch is": no selection on VAL at all
            final_state, final_score, final_met, soup_epochs = _cpu_state(model), score, met, [ep]
        elif soup_k > 1 and len(topk) > 1:
            final_state, final_score, final_met, soup_epochs = greedy_soup(model, topk, va_loader, store, sel, log_prefix)
        model.load_state_dict(final_state); model.to(RT.device)
        out_best = dict(score=final_score, epoch=best["epoch"], metrics=final_met, state=final_state)
        if ckpt_dir: save_checkpoint(os.path.join(ckpt_dir, "best.pt"), final_state, hp, pre,
                                     dict(best_epoch=best["epoch"], best_metrics=final_met, seed=seed, soup_epochs=soup_epochs, single_epoch=single))
        return dict(model=model, history=history, best=out_best, single=single, single_state=best["state"], soup_epochs=soup_epochs,
                    n_params=n_params, hp=hp, init_state=init_state)
    finally:
        del opt, sched, tr_loader, va_loader; gc.collect()
        if RT.device.type == "cuda": torch.cuda.empty_cache()


def _rewrap(loader, sampler, wrapper, batch_size, seed, cfg):
    from .dataset import StayBatchSampler, collate
    ds = wrapper(sampler.ds); smp = StayBatchSampler(ds, batch_size, True, seed)
    nw = cfg.NUM_WORKERS; kw = dict(num_workers=nw, pin_memory=RT.device.type == "cuda")
    if nw > 0: kw.update(persistent_workers=False, prefetch_factor=4)
    return torch.utils.data.DataLoader(ds, batch_sampler=smp, collate_fn=collate, **kw), smp


# ---------------------------------------------------------------- checkpoint registry ----------------------------------------------------------------
class Registry:
    """manifest.json: one record per trained model (id, arm, seed, role, data hash, metrics, path). Every number in STORY.md must trace here."""
    def __init__(self, work_dir=None):
        self.path = os.path.join(work_dir or C.PATHS.WORK_DIR, "manifest.json")
    def load(self):
        return json.load(open(self.path)) if os.path.exists(self.path) else {}
    def register(self, model_id, *, arm, seed, role, data_hash, metrics, path, hp=None, extra=None):
        m = self.load()
        m[model_id] = dict(model_id=model_id, arm=arm, seed=int(seed), role=role, data_hash=data_hash, path=path, run=C.CFG.RUN_NAME,
                           metrics={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in (metrics or {}).items()},
                           hp=hp, time=time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {}))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(m, open(self.path + ".tmp", "w"), indent=1, default=str); os.replace(self.path + ".tmp", self.path); return m[model_id]
    def get(self, model_id): return self.load().get(model_id)
