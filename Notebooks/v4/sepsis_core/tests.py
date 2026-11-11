"""The v4 test-suite. `run_all()` runs on synthetic data (no patient data) and, when given a real `bundle`, on FIT/VAL in smoke mode.
Every test prints aggregates only (pass/fail, counts, maximum absolute differences).

Exactness tests (bitwise, causal, placebo) run on CPU in fp32 on purpose: on a GPU with fp16 autocast two runs of the SAME code
already differ in the last bits, so "bitwise" is only a meaningful claim on CPU.
"""
import ast, copy, hashlib, json, os, re, tempfile, traceback
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import attention as A
from . import balancing as BAL
from . import calibration as CAL
from . import data as D
from . import engine as E
from . import metrics as MET
from . import models as M
from . import synthetic as SYN
from . import tracking as TR
from . import user_classes as UC
from . import v3_reference as V3
from .dataset import StayDataset, collate, make_loader, build_attn_mask

TEST_HP = dict(d_model=64, num_heads=8, num_layers=3, ffn_mult=2, dropout=0.1, lr=1e-3, weight_decay=1e-4, batch_size=16,
               focal_gamma=1.0, focal_alpha=0.88, aux_w=0.05, pos_weight=10.0)
ARMS_ALL = ("softmax", "sigmoid", "kernel", "kernel_raw", "ka")
CPU = torch.device("cpu")
TOL = 1e-6      # CPU fp32: the expected value of every 'must not change' quantity is exactly 0.0; the printed maxima show it


# ---------------------------------------------------------------- helpers ----------------------------------------------------------------
def _model(arm, bundle, seed=0, hp=None):
    C.seed_everything(seed); nF = bundle["store"].X.shape[1]
    if arm == "ka": return M.build_ka(nF, d_model=64, num_heads=8, num_layers=3, ffn_dim=128, dropout=0.1).to(CPU).eval()
    return M.build_model({**C.DEFAULT_FLAGS, **TEST_HP, **(hp or {}), "arm": arm}, bundle["store"].x_mean).to(CPU).eval()


def _batch(bundle, n=6, min_len=30, max_len=None, which="val_idx"):
    st = bundle["store"]; idx = bundle[which]; idx = idx[st.lens[idx] >= min_len][:n]
    if len(idx) < 2: raise RuntimeError(f"need at least 2 stays of >= {min_len} h for this test, found {len(idx)}")
    ds = StayDataset(st, idx, max_len or int(st.lens[idx].max()), mode="eval")
    return collate([ds[i] for i in range(len(ds))])


def _run(model, b):
    out = model(b["X"], b["M"], b["D"], b["XL"], b["S"], b["G"], b["hrs"], b["E"], build_attn_mask(b["PAD"]), return_aux=True)
    lg = torch.stack([out["logit_t1"], out["logit_t3"], out["logit_t6"]], -1)
    return lg, out.get("aux")


def _result(name, passed, **detail): return dict(name=name, passed=bool(passed), detail=detail)


# ---------------------------------------------------------------- 1 · causal leakage ----------------------------------------------------------------
@torch.no_grad()
def test_causal_leakage(bundle):
    """Perturb every input at hours > t (values, masks, time-since-measurement, statics, hour index); outputs at hours <= t must not move.
    Also checks the Preprocessor: changing raw measurements after hour t0 must not change any tensor at hours <= t0 (no backward fill)."""
    b = _batch(bundle); T_ = b["X"].shape[1]; t = T_ // 2; g = torch.Generator().manual_seed(1); detail = {}; ok = True
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    for k in ("X", "D", "XL", "S", "E"): b2[k][:, t + 1:] += torch.randn(b2[k][:, t + 1:].shape, generator=g) * 3.0
    b2["M"][:, t + 1:] = 1.0 - b2["M"][:, t + 1:]; b2["G"][:, t + 1:] = 1 - b2["G"][:, t + 1:]; b2["hrs"][:, t + 1:] += 7.0
    for arm in ARMS_ALL:
        m = _model(arm, bundle); l0, a0 = _run(m, b); l1, a1 = _run(m, b2)
        past = float((l0[:, : t + 1] - l1[:, : t + 1]).abs().max()); fut = float((l0[:, t + 1:] - l1[:, t + 1:]).abs().max())
        if a0 is not None: past = max(past, float((a0[:, : t + 1] - a1[:, : t + 1]).abs().max()))
        detail[arm] = dict(max_abs_change_hours_le_t=past, max_abs_change_hours_gt_t=fut)
        ok &= (past <= TOL) and (fut > 1e-4)                      # zero in the past (exactly 0.0 expected on CPU); and the perturbation was not vacuous
    # preprocessing level
    raw, pre = bundle.get("df"), bundle["pre"]
    if raw is not None:
        cfg = pre.cfg; ids = raw[cfg.ID_COL].unique()[:25]; sub = raw[raw[cfg.ID_COL].isin(ids)].copy(); t0 = 10
        sub2 = sub.copy(); fut_rows = sub2[cfg.HOUR_COL] > t0; rng = np.random.default_rng(0)
        for f in pre.feat:
            v = sub2.loc[fut_rows, f].to_numpy(np.float32, copy=True)
            sub2.loc[fut_rows, f] = np.where(rng.random(len(v)) < 0.5, np.nan, np.nan_to_num(v, nan=1.0) * 1.7 + 3.0).astype(np.float32)
        s1, s2 = pre.transform(sub, with_labels=False), pre.transform(sub2, with_labels=False); past_rows = s1.hours <= t0
        pp = {k: float(np.abs(getattr(s1, k)[past_rows].astype(np.float64) - getattr(s2, k)[past_rows].astype(np.float64)).max()) for k in ("X", "M", "D", "XL", "E")}
        changed = float(np.abs(s1.XL[~past_rows] - s2.XL[~past_rows]).max())
        detail["preprocessor"] = dict(max_abs_change_rows_le_t0=pp, future_rows_changed=changed)
        # E is float16 and built from table-wide running sums: allow rounding (1e-4); a real leak would be orders of magnitude larger
        ok &= all(pp[k] <= TOL for k in ("X", "M", "D", "XL")) and pp["E"] <= 1e-4 and changed > 0
    return _result("1 causal leakage (inputs after t cannot move outputs at <= t)", ok, **detail)


# ---------------------------------------------------------------- 2 · full coverage ----------------------------------------------------------------
def test_full_coverage(bundle):
    """Every row of every stay is predicted exactly once, hence every valid (row, horizon) label gets exactly one prediction;
    every row is trained on exactly once per epoch. Checked with one window per stay AND with forced overlapping windows."""
    st = bundle["store"]; idx = bundle["val_idx"]; detail = {}; ok = True
    all_rows = np.concatenate([np.arange(st.starts[i], st.starts[i] + st.lens[i]) for i in idx])
    cfg_small = copy.copy(C.CFG); cfg_small.MAX_SEQ_LEN = 24; cfg_small.EVAL_STRIDE = None
    for tag, cfg in (("one_window_per_stay", C.CFG), ("overlapping_windows_max_len_24", cfg_small)):
        for mode, train in (("eval", False), ("train", True)):
            loader, smp = make_loader(st, idx, 16, train, seed=3, cfg=cfg, num_workers=0)
            rows, hist_ok = [], True
            for b in loader:
                keep = (~b["PAD"]) & b["KEEP"]; rows.append(b["ROW"][keep].numpy())
                if mode == "eval":                                      # history actually available to each kept row
                    pos_in_win = torch.arange(b["PAD"].shape[1])[None, :].expand_as(keep)[keep].numpy()
                    r = b["ROW"][keep].numpy(); stay_start = st.starts[np.searchsorted(st.starts, r, side="right") - 1]
                    need = np.minimum(r - stay_start, cfg.MAX_SEQ_LEN - (cfg.EVAL_STRIDE or max(1, cfg.MAX_SEQ_LEN // 2)))
                    hist_ok &= bool((pos_in_win >= need).all())
            rows = np.concatenate(rows); u, c = np.unique(rows, return_counts=True)
            exact = bool(len(rows) == len(all_rows) and np.array_equal(u, np.sort(all_rows)) and (c == 1).all())
            per_h = [int(st.V[rows, k].sum()) == int(st.V[all_rows, k].sum()) for k in range(st.V.shape[1])]
            detail[f"{tag}/{mode}"] = dict(rows_expected=int(len(all_rows)), rows_seen=int(len(rows)), each_exactly_once=exact,
                                           valid_labels_covered_per_horizon=per_h, n_windows=len(smp.ds), history_rule_ok=hist_ok)
            ok &= exact and all(per_h) and hist_ok
    v3 = MET.v3_view_mask(st, all_rows); exp = int(np.minimum(st.lens[idx], C.CFG.V3_VIEW_HOURS).sum())
    detail["v3_view_rows"] = dict(expected=exp, got=int(v3.sum())); ok &= int(v3.sum()) == exp
    return _result("2 full coverage (each row predicted once; each row trained once per epoch)", ok, **detail)


# ---------------------------------------------------------------- 3 · step-0 equality across arms ----------------------------------------------------------------
def test_step0_equality(bundle, seeds=(42, 1042)):
    """Same seed -> every arm starts from identical weights and sees identical batches in identical order. Different seeds differ."""
    st = bundle["store"]; detail = {}; ok = True
    for s in seeds:
        sd = {arm: {k: v.clone() for k, v in _model(arm, bundle, seed=s).state_dict().items()} for arm in ("softmax", "sigmoid", "kernel", "kernel_raw")}
        ref = sd["softmax"]
        for arm in ("sigmoid", "kernel", "kernel_raw"):
            extra = sorted(set(sd[arm]) - set(ref)); missing = sorted(set(ref) - set(sd[arm]))
            same = all(torch.equal(ref[k], sd[arm][k]) for k in ref if k in sd[arm])
            only_beta = all(k.endswith("log_beta") for k in extra)
            detail[f"seed{s}/{arm}"] = dict(shared_tensors_equal=same, n_shared=len(ref) - len(missing), extra_keys=len(extra), extra_are_log_beta_only=only_beta)
            ok &= same and not missing and only_beta and (len(extra) == 0 or arm.startswith("kernel"))
        orders = []
        for arm in ("softmax", "sigmoid"):
            loader, _ = make_loader(st, bundle["fit_idx"], 16, True, seed=s, num_workers=0)
            orders.append([b["ROW"][:, 0].tolist() for i, b in enumerate(loader) if i < 5])
        detail[f"seed{s}/batch_order_equal"] = orders[0] == orders[1]; ok &= orders[0] == orders[1]
    a, b_ = _model("softmax", bundle, seed=seeds[0]).state_dict(), _model("softmax", bundle, seed=seeds[1]).state_dict()
    differ = any(not torch.equal(a[k], b_[k]) for k in a); detail["different_seeds_differ"] = differ; ok &= differ
    hp_s = E.complete_hp({**TEST_HP, "arm": "softmax"}); hp_g = E.complete_hp({**TEST_HP, "arm": "sigmoid"})
    diff = sorted(k for k in hp_s if hp_s[k] != hp_g.get(k)); detail["hp_keys_that_differ_between_arms"] = diff
    ok &= diff == ["arm", "warmup_mult"]
    return _result("3 step-0 parameter equality across arms (same seed)", ok, **detail)


# ---------------------------------------------------------------- 4 · softmax path bitwise == v3 ----------------------------------------------------------------
def _defs(module):
    src = Path(module.__file__).read_text(encoding="utf-8"); tree = ast.parse(src)
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


def test_softmax_bitwise(bundle):
    """(a) every class cell except MultiHeadAttention is textually identical to the v3 notebook cell; (b) the reference file matches the
    notebook's cell hashes (when the notebook is reachable); (c) the patched MultiHeadAttention in softmax mode equals v3's bit for bit;
    (d) so does the whole model: weights, outputs (eval and train/dropout), loss and gradients."""
    detail = {}; ok = True
    d3, du = _defs(V3), _defs(UC); differ = sorted(k for k in d3 if d3[k] != du.get(k))
    detail["a_definitions_that_differ_from_v3"] = differ; ok &= differ == ["MultiHeadAttention"] and set(d3) == set(du)
    # (b) reference file vs notebook
    nbp = C.PATHS.V3_NOTEBOOK
    if os.path.exists(nbp):
        nb = json.load(open(nbp, encoding="utf-8")); src = Path(V3.__file__).read_text(encoding="utf-8")
        marks = re.findall(r"# >>> v3 cell \[(\d+)\] sha256=([0-9a-f]{64})", src)
        good = [hashlib.sha256("".join(nb["cells"][int(i)]["source"]).encode()).hexdigest() == h for i, h in marks]
        ptxt = Path(D._v3p.__file__).read_text(encoding="utf-8"); ph = re.search(r"sha256 of the class source: ([0-9a-f]{64})", ptxt).group(1)
        c25 = "".join(nb["cells"][25]["source"]); t25 = ast.parse(c25); cl = [n for n in t25.body if isinstance(n, ast.ClassDef) and n.name == "Preprocessor"][0]
        pre_ok = hashlib.sha256("".join(c25.splitlines(keepends=True)[cl.lineno - 1: cl.end_lineno]).encode()).hexdigest() == ph
        detail["b_reference_matches_notebook"] = dict(cells_checked=len(marks), all_match=all(good), preprocessor_matches=pre_ok); ok &= all(good) and len(marks) >= 5 and pre_ok
    else:
        detail["b_reference_matches_notebook"] = "skipped: v3 notebook not found at PATHS.V3_NOTEBOOK (runs in Colab)"
    # (c) attention module
    torch.manual_seed(0); m3 = V3.MultiHeadAttention(64, 8); torch.manual_seed(0); mu = UC.MultiHeadAttention(64, 8)
    same_w = all(torch.equal(a, b) for a, b in zip(m3.state_dict().values(), mu.state_dict().values())) and list(m3.state_dict()) == list(mu.state_dict())
    X = torch.randn(3, 17, 64); pad = torch.zeros(3, 17, dtype=torch.bool); pad[1, 12:] = True; pad[2, 5:] = True; mask = build_attn_mask(pad)
    res = {}
    for mode in ("eval", "train"):
        getattr(m3, mode)(); getattr(mu, mode)()
        torch.manual_seed(7); o3 = m3(X, mask); torch.manual_seed(7); ou = mu(X, mask)
        res[mode] = bool(torch.equal(o3, ou) and torch.equal(m3.attention_logits, mu.attention_logits)
                         and torch.equal(m3.attention_weights, mu.attention_weights) and torch.equal(m3.Value, mu.Value))
    detail["c_attention_module"] = dict(state_dict_identical=same_w, **{f"{k}_bitwise_equal": v for k, v in res.items()}); ok &= same_w and all(res.values())
    # (d) whole model
    hp = {**C.DEFAULT_FLAGS, **TEST_HP, "arm": "softmax"}; xm = bundle["store"].x_mean; b = _batch(bundle)
    C.seed_everything(11); f3 = M.build_model(hp, xm, classes=V3); C.seed_everything(11); fu = M.build_model(hp, xm, classes=UC)
    s3, su = f3.state_dict(), fu.state_dict()
    same_sd = list(s3) == list(su) and all(torch.equal(s3[k], su[k]) for k in s3)
    f3.eval(); fu.eval()
    with torch.no_grad(): (l3, a3), (lu, au) = _run(f3, b), _run(fu, b)
    eq_eval = bool(torch.equal(l3, lu) and torch.equal(a3, au))
    crit = UC.FocalLoss(alpha=0.88, gamma=1.0); grads = []
    for f in (f3, fu):
        f.train(); torch.manual_seed(5); lg, ap = _run(f, b)
        loss = sum(crit(lg[..., k], b["Y"][..., k], b["V"][..., k]) for k in range(3)) + 0.05 * UC.next_hour_forecast_loss(ap, b["X"], b["M"], b["PAD"], b["hrs"])
        f.zero_grad(); loss.backward(); grads.append((loss.detach().clone(), [p.grad.clone() for p in f.parameters() if p.grad is not None]))
    eq_train = bool(torch.equal(grads[0][0], grads[1][0]) and len(grads[0][1]) == len(grads[1][1]) and all(torch.equal(x, y) for x, y in zip(grads[0][1], grads[1][1])))
    detail["d_full_model"] = dict(state_dict_identical=same_sd, n_tensors=len(s3), eval_outputs_bitwise_equal=eq_eval, train_loss_and_all_gradients_bitwise_equal=eq_train)
    ok &= same_sd and eq_eval and eq_train
    return _result("4 softmax path bitwise equal to the v3 classes", ok, **detail)


# ---------------------------------------------------------------- 5 · sigmoid weights at init ----------------------------------------------------------------
@torch.no_grad()
def test_sigmoid_init(bundle):
    """a_tj = sigmoid(z_tj - log T_t) exactly; with z = 0 that is 1/(1+T_t) (NOT 1/T_t), so total mass is T/(T+1): 0.5 at the first hour,
    against exactly 1 for softmax. Reports the mass by query position, which is the by-construction arm difference at early hours."""
    b = _batch(bundle); detail = {}; ok = True
    m = _model("sigmoid", bundle); cap = A.capture(m, b); Z, Aw = cap["Z"], cap["A"]
    vis = torch.isfinite(Z); Tt = vis.sum(-1, keepdim=True).float()
    expect = torch.where(vis, torch.sigmoid(torch.where(vis, Z, torch.zeros_like(Z)) - torch.log(Tt.clamp(min=1))), torch.zeros_like(Z))
    f_err = float((Aw - expect).abs().max()); detail["formula_max_abs_error"] = f_err; ok &= f_err < 1e-6
    real = ~b["PAD"]; q = real[:, None, None, :, None].expand_as(Z) & vis
    ratio = (Aw * (1 + Tt))[q]                                         # 1.0 means exactly 1/(1+T)
    detail["init_weight_times_(1+T)"] = dict(mean=float(ratio.mean()), p05=float(ratio.quantile(0.05)), p95=float(ratio.quantile(0.95)))
    ok &= 0.8 < float(ratio.mean()) < 1.25
    mz = _model("sigmoid", bundle)                                      # exact check with z forced to 0
    for layer in mz.core.encoder.layers: layer.MultiHeadAttention.W_Q.weight.zero_()
    cz = A.capture(mz, b); vz = torch.isfinite(cz["Z"]); Tz = vz.sum(-1, keepdim=True).float()
    z_err = float((cz["A"] - torch.where(vz, 1.0 / (1.0 + Tz), torch.zeros_like(Tz))).abs().max())
    detail["z_equals_0_max_abs_error_vs_1/(1+T)"] = z_err; ok &= z_err < 1e-6
    ms = _model("softmax", bundle); mass_s = A.total_mass_by_position(A.capture(ms, b)["A"], b["PAD"]); mass_g = A.total_mass_by_position(Aw, b["PAD"])
    pos = [p for p in (0, 1, 3, 7, 15, 23) if p < len(mass_g)]
    detail["total_mass_by_query_position"] = {f"T={p + 1}": dict(sigmoid=round(float(mass_g[p]), 4), theory_T_over_T_plus_1=round((p + 1) / (p + 2), 4),
                                                                softmax=round(float(mass_s[p]), 4)) for p in pos}
    ok &= bool(np.nanmax(np.abs(mass_s - 1.0)) < 1e-5) and abs(float(mass_g[0]) - 0.5) < 0.1
    return _result("5 sigmoid weights at init: a = sigmoid(z - log T) ~ 1/(1+T)", ok, **detail)


# ---------------------------------------------------------------- 6 · layer-1 placebo ----------------------------------------------------------------
@torch.no_grad()
def test_layer1_placebo(bundle):
    """Intervene on hour k only (S1: embedding splice; S0-lite: raw-input splice). Exact facts that must hold in every arm:
      noise floor     two identical runs give Dz = 0 exactly;
      layer 1         Dz_tj = 0 for every j != k and t != k (keys and queries depend on their own hour only);
      causality       Dz_tj = 0 for every query t < k, in every layer.
    Reported, not asserted (see Docs/DECISIONS.md, open issue O1): in layers >= 2, Dz_tj for j < k at queries t > k is NOT zero,
    because the query q_t has already read hour k. The PDF's 'placebo set j < k is exactly zero' holds in layer 1 only."""
    b = _batch(bundle); T_ = b["X"].shape[1]; k = T_ // 3; detail = {}; ok = True
    for arm in ("softmax", "sigmoid", "kernel"):
        m = _model(arm, bundle); base = A.capture(m, b); again = A.capture(m, b)
        floor = float(A.delta_z(again["Z"], base["Z"]).abs().max())
        g = torch.Generator().manual_seed(2); e = A.embed(m, b); donor = e[:, k] + torch.randn(e[:, k].shape, generator=g) * e.std()
        b0 = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in b.items()}
        b0["X"][:, k] += 2.5; b0["XL"][:, k] += 2.5; b0["M"][:, k] = 1.0
        out = {}
        for site, cap in (("S1_embedding", A.splice_embedding(m, b, k, donor)), ("S0_input", A.capture(m, b0))):
            dz = A.delta_z(cap["Z"], base["Z"])                                    # (B,L,h,T,T)
            rows_not_k = torch.ones(T_, dtype=torch.bool); rows_not_k[k] = False
            l1 = dz[:, 0][:, :, rows_not_k][:, :, :, rows_not_k]
            l1_other = float(l1.abs().max()); l1_col_k = float(dz[:, 0, :, k + 1:, k].abs().max())
            before_k = float(dz[:, :, :, :k, :].abs().max())
            deep_placebo = float(dz[:, 1:, :, k + 1:, :k].abs().max()); deep_after = float(dz[:, 1:, :, k + 1:, k + 1:].abs().max())
            out[site] = dict(layer1_max_abs_dz_j_ne_k=l1_other, layer1_max_abs_dz_j_eq_k=l1_col_k, all_layers_max_abs_dz_queries_before_k=before_k,
                             REPORTED_layers_ge2_max_abs_dz_j_lt_k=deep_placebo, REPORTED_layers_ge2_max_abs_dz_j_gt_k=deep_after)
            ok &= l1_other <= TOL and before_k <= TOL and l1_col_k > 1e-6
        detail[arm] = dict(noise_floor_max_abs_dz=floor, **out); ok &= floor <= TOL
    return _result("6 layer-1 placebo: Dz_j = 0 for j != k (exact), plus causality", ok, **detail)


# ---------------------------------------------------------------- 7 · forbidden columns and reserved folds ----------------------------------------------------------------
def _raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    except Exception: return False
    return False


def test_forbidden_guard(bundle):
    """SOFA / label / onset columns can never become inputs or auxiliary targets; ICULOS only inside the explicit ablation;
    reserved folds cannot be loaded by a training stage; the frozen split is refused if its fingerprint differs."""
    detail = {}; ok = True
    per_col = {c: _raises(lambda c=c: C.assert_no_forbidden(C.DECAY_FEATURES + [c]), PermissionError) for c in sorted(C.FORBIDDEN_INPUT_COLUMNS)}
    detail["every_forbidden_column_refused"] = all(per_col.values()); ok &= all(per_col.values())
    detail["iculos_allowed_only_in_ablation"] = bool(C.assert_no_forbidden(["ICULOS"], allow_iculos=True)) and _raises(lambda: C.assert_no_forbidden(["ICULOS"]), PermissionError)
    detail["preprocessor_refuses_SOFA"] = _raises(lambda: D.Preprocessor(features=C.DECAY_FEATURES + ["SOFA_total"]), PermissionError)
    detail["preprocessor_refuses_HoursToOnset"] = _raises(lambda: D.Preprocessor(features=C.DECAY_FEATURES + ["HoursToOnset"]), PermissionError)
    pre = bundle["pre"]; inputs = set(pre.feat) | set(pre.static_cols.values())
    detail["store_inputs_free_of_forbidden"] = not (inputs & C.FORBIDDEN_INPUT_COLUMNS)
    m = _model("softmax", bundle); detail["aux_head_targets_are_input_features_only"] = m.aux_head.proj.out_features == len(pre.feat)
    with tempfile.TemporaryDirectory() as td:
        raw = SYN.make_hourly(40, seed=5, max_len=40); p = os.path.join(td, "h.parquet"); raw.to_parquet(p)
        detail["loader_refuses_forbidden_feature"] = _raises(lambda: D.load_hourly(p, features=C.DECAY_FEATURES + ["SOFA_cv"]), PermissionError)
        df, sr, stats = D.load_hourly(p); detail["label_fix_drops_post_onset_rows"] = bool((df["HoursToOnset"].isna() | (df["HoursToOnset"] >= 0)).all()) and stats["rows"] < stats["rows_raw"]
        sp = SYN.make_split(raw, 5); sp_path = os.path.join(td, "split.parquet"); sp.to_parquet(sp_path)
        detail["split_refused_on_wrong_sha"] = _raises(lambda: D.load_frozen_split([sp_path], expect_sha="000000000000"), RuntimeError)
        detail["split_never_rebuilt_when_missing"] = _raises(lambda: D.load_frozen_split([os.path.join(td, "nope.parquet")]), FileNotFoundError)
        detail["train_stage_refuses_reserved_folds"] = all(_raises(lambda f=f: D.attach_folds(df.copy(), sp, ("FIT", "VAL", f), D.FoldGuard("train")), PermissionError) for f in C.CFG.RESERVED_FOLDS)
        detail["causal_stage_refused_until_prereg_frozen"] = _raises(lambda: D.FoldGuard("causal", docs_dir=td).check({"LOC"}), PermissionError)
        open(os.path.join(td, "PREREGISTRATION.md"), "w").write("x"); open(os.path.join(td, "PREREGISTRATION.frozen"), "w").write("x")
        detail["causal_stage_allowed_after_freeze"] = bool(D.FoldGuard("causal", docs_dir=td).check({"LOC", "CONF"}))
        d2, counts = D.attach_folds(df.copy(), sp, ("FIT", "VAL"), D.FoldGuard("train")); detail["reserved_folds_dropped_from_memory"] = set(d2["fold"].unique()) <= {"FIT", "VAL"}
        detail["preprocessor_fit_refuses_non_FIT_rows"] = _raises(lambda: D.Preprocessor(static_cols=sr).fit(d2, np.ones(len(d2), bool), fold=d2["fold"].to_numpy()), PermissionError)
    ok &= all(v is True for k, v in detail.items())
    return _result("7 forbidden-column guard, fold guard, frozen split", ok, **detail)


# ---------------------------------------------------------------- 8 · balancing touches FIT only ----------------------------------------------------------------
def test_balancing_fit_only(bundle):
    st = bundle["store"]; detail = {}; ok = True
    fit_ds = StayDataset(st, bundle["fit_idx"], C.CFG.MAX_SEQ_LEN, mode="train"); val_ds = StayDataset(st, bundle["val_idx"], C.CFG.MAX_SEQ_LEN, mode="train")
    wrap = BAL.oversample_positive_windows(st)
    detail["oversampling_refuses_VAL"] = _raises(lambda: wrap(val_ds), PermissionError)
    detail["oversampling_refuses_eval_dataset"] = _raises(lambda: wrap(StayDataset(st, bundle["fit_idx"], C.CFG.MAX_SEQ_LEN, mode="eval")), PermissionError)
    o = wrap(fit_ds); added = o.map[len(fit_ds):]
    detail["added_items"] = int(o.n_added); detail["added_items_are_positive_FIT_windows"] = bool(fit_ds.has_pos[added].all()) if len(added) else True
    detail["every_original_window_still_present_once"] = bool(np.array_equal(o.map[: len(fit_ds)], np.arange(len(fit_ds))))
    Xf = np.nan_to_num(st.XL[:500]); yf = (np.arange(500) % 25 == 0)
    detail["smote_refuses_VAL_rows"] = _raises(lambda: BAL.smote_rows(Xf, yf, np.array(["FIT"] * 499 + ["VAL"])), PermissionError)
    Xb, yb, n_new = BAL.smote_rows(Xf, yf, np.array(["FIT"] * 500)); detail["smote_synthetic_rows"] = int(n_new)
    detail["smote_keeps_original_rows_first"] = bool(np.array_equal(Xb[:500], Xf.astype(np.float32)) and n_new > 0)
    rows_seen = []
    def spy(ds): w = wrap(ds); rows_seen.append(np.unique(np.concatenate([np.arange(a, a + n) for a, n in zip(ds.a, ds.n)]))); return w
    with tempfile.TemporaryDirectory() as td:
        res = E.fit({**TEST_HP, "d_model": 32, "num_layers": 1, "arm": "softmax"}, st, bundle["fit_idx"], bundle["val_idx"], max_epochs=1, patience=1, seed=1,
                    train_dataset_wrapper=spy, soup_k=0, cfg=_cfg0(), log_prefix="[test8] ")
        loader, _ = make_loader(st, bundle["val_idx"], 16, False, cfg=_cfg0()); r, lg, _ = E.predict_loader(res["model"], loader, as_logits=True)
    fit_rows = np.concatenate([np.arange(st.starts[i], st.starts[i] + st.lens[i]) for i in bundle["fit_idx"]])
    val_rows = np.concatenate([np.arange(st.starts[i], st.starts[i] + st.lens[i]) for i in bundle["val_idx"]])
    detail["balanced_training_rows_subset_of_FIT"] = bool(np.isin(rows_seen[0], fit_rows).all())
    detail["VAL_scored_once_at_natural_distribution"] = bool(np.array_equal(np.sort(r), np.sort(val_rows)))
    detail["calibration_refuses_FIT_rows"] = _raises(lambda: CAL.calibrate(st, r, lg, fold_label=np.array(["FIT"] * len(r))), PermissionError)
    ok &= all(v is True for k, v in detail.items() if isinstance(v, bool)) and detail["added_items"] > 0
    return _result("8 balancing touches FIT only", ok, **detail)


def _cfg0():
    c = copy.copy(C.CFG); c.NUM_WORKERS = 0; c.ALLOW_RESUME = True; return c


# ---------------------------------------------------------------- 9 · end-to-end smoke training ----------------------------------------------------------------
def test_smoke_train(bundle, epochs=3):
    """Both primary arms through the real engine: training, soup, checkpoints, resume, registry, calibration, gate, hour buckets, thresholds."""
    st = bundle["store"]; detail = {}; ok = True; cfg = _cfg0(); cfg.SOUP_TOP_K = 2
    hp = {**TEST_HP, "d_model": 32, "num_layers": 2}
    with tempfile.TemporaryDirectory() as td:
        reg = E.Registry(td); scores = {}
        for arm in ("softmax", "sigmoid"):
            ck = os.path.join(td, arm)
            res = E.fit({**hp, "arm": arm}, st, bundle["fit_idx"], bundle["val_idx"], max_epochs=epochs, patience=epochs, seed=42, alpha=bundle["alpha"], pw=bundle["pw"],
                        pre=bundle["pre"], ckpt_dir=ck, cfg=cfg, log_prefix=f"[smoke {arm}] ")
            loader, _ = make_loader(st, bundle["val_idx"], cfg.EVAL_BATCH, False, cfg=cfg); rows, lg, _ = E.predict_loader(res["model"], loader, as_logits=True)
            met = MET.compute_metrics(st, rows, lg); scores[arm] = (rows, lg)
            reg.register(f"{arm}_s42", arm=arm, seed=42, role="smoke", data_hash="synthetic", metrics=met, path=os.path.join(ck, "best.pt"), hp=res["hp"])
            model2, pre2, ckd = E.load_pipeline(os.path.join(ck, "best.pt")); _, lg2, _ = E.predict_loader(model2, loader, as_logits=True)
            again = E.fit({**hp, "arm": arm}, st, bundle["fit_idx"], bundle["val_idx"], max_epochs=epochs, patience=epochs, seed=42, alpha=bundle["alpha"], pw=bundle["pw"],
                          pre=bundle["pre"], ckpt_dir=ck, resume=True, cfg=cfg, log_prefix=f"[resume {arm}] ")
            mass = [h.get("mass_mean") for h in res["history"]]
            detail[arm] = dict(val_auroc=round(met["auroc_mean"], 4), epochs_run=len(res["history"]), mass_mean_per_epoch=[round(x, 4) for x in mass],
                               checkpoint_roundtrip_max_abs_diff=float(np.abs(lg - lg2).max()), files=sorted(os.listdir(ck)),
                               resume_did_not_retrain=len(again["history"]) == len(res["history"]), warmup_mult=res["hp"]["warmup_mult"],
                               init_checkpoint_equals_step0=all(torch.equal(torch.load(os.path.join(ck, "init.pt"), weights_only=False)["model"][k], v) for k, v in res["init_state"].items()))
            ok &= np.isfinite(met["auroc_mean"]) and detail[arm]["checkpoint_roundtrip_max_abs_diff"] < 1e-2 and detail[arm]["resume_did_not_retrain"] and detail[arm]["init_checkpoint_equals_step0"]
            if arm == "softmax": ok &= all(abs(x - 1.0) < 5e-3 for x in mass)      # fp16 autocast on a T4 rounds each weight
        i0 = torch.load(os.path.join(td, "softmax", "init.pt"), weights_only=False)["model"]; i1 = torch.load(os.path.join(td, "sigmoid", "init.pt"), weights_only=False)["model"]
        detail["arms_share_step0_weights_in_engine"] = all(torch.equal(i0[k], i1[k]) for k in i0); ok &= detail["arms_share_step0_weights_in_engine"]
        detail["warmup_differs_by_2x_only"] = detail["softmax"]["warmup_mult"] == 1.0 and detail["sigmoid"]["warmup_mult"] == 2.0; ok &= detail["warmup_differs_by_2x_only"]
        rows, lg = scores["softmax"]
        cal, recs, _ = CAL.calibrate(st, rows, lg, cfg=cfg, fold_label=np.array(["VAL"] * len(rows))); probs = CAL.apply_calibrator(cal, lg)
        detail["calibration_method"] = cal["method"]
        g = MET.gate(st, rows, lg, np.random.default_rng(0).random(lg.shape), n_boot=30)
        detail["gate_runs"] = isinstance(g["passed"], bool) and len(g["per_horizon"]) == 3
        onset_rows = MET.row_onset(st, bundle["onset"]); tab = MET.by_icu_hour(st, rows, lg, onset_rows, views=("all", "no_early_onset", "early_onset", "late_onset"))
        detail["hour_bucket_rows"] = len(tab); detail["threshold_table_rows"] = len(MET.threshold_metrics(st, rows, probs))
        p, lo, hi = MET.stay_bootstrap_auroc(st, rows, lg, n_boot=30); detail["bootstrap_ci_finite"] = bool(np.isfinite([p, lo, hi]).all())
        detail["registry_entries"] = sorted(reg.load()); ok &= detail["gate_runs"] and detail["bootstrap_ci_finite"] and len(detail["registry_entries"]) == 2
    return _result("9 smoke training, both arms, end to end", ok, **detail)


# ---------------------------------------------------------------- 10 · MLflow active-run bug ----------------------------------------------------------------
def test_mlflow_active_run():
    """Reproduces v3's failure mode (a run left open) against DagsHub and checks the next top-level run still starts and closes.
    Writes three tiny runs to the experiment 'sepsis-attention-v4-selftest'."""
    name = "10 MLflow 'already active run' fix (DagsHub)"
    if TR.mlflow is None: return _result(name, False, error="mlflow is not installed")
    TR.setup_tracking("sepsis-attention-v4-selftest")
    if not TR.TRACKING["enabled"]: return _result(name, False, error="DagsHub tracking is off: see the [mlflow] message above (secrets or connection)")
    detail = {"uri": TR.TRACKING["uri"]}
    try:
        TR.mlflow.start_run(run_name="left_open_by_a_crash")
        with TR.safe_run("next_run") as r:
            detail["second_run_started"] = r is not None; TR.m_metrics({"x": 1.0})
        detail["no_run_left_active"] = TR.mlflow.active_run() is None
        try:
            with TR.safe_run("failing"): raise ValueError("boom")
        except ValueError: pass
        detail["closed_after_exception"] = TR.mlflow.active_run() is None
    finally:
        TR.end_active_runs()
    return _result(name, all(v for k, v in detail.items() if k != "uri"), **detail)


# ---------------------------------------------------------------- runner ----------------------------------------------------------------
def run_all(bundle=None, smoke_train=True, verbose=True, n_subjects=500):
    """bundle=None -> synthetic data built here. Returns the list of results; prints one line per test plus the details."""
    if bundle is None: bundle = SYN.make_bundle(n_subjects=n_subjects, seed=0, max_len=120)
    tests = [lambda: test_causal_leakage(bundle), lambda: test_full_coverage(bundle), lambda: test_step0_equality(bundle), lambda: test_softmax_bitwise(bundle),
             lambda: test_sigmoid_init(bundle), lambda: test_layer1_placebo(bundle), lambda: test_forbidden_guard(bundle), lambda: test_balancing_fit_only(bundle)]
    if smoke_train: tests.append(lambda: test_smoke_train(bundle))
    tests.append(test_mlflow_active_run)
    results = []
    for t in tests:
        try: r = t()
        except Exception as e:
            r = _result(f"test raised {type(e).__name__}", False, error=str(e)[:400], trace=traceback.format_exc(limit=4)[-900:])
        results.append(r)
        if verbose:
            print(("PASS  " if r["passed"] else "FAIL  ") + r["name"])
            print("      " + json.dumps(r["detail"], default=str)[:2400])
    n_ok = sum(r["passed"] for r in results)
    if verbose: print(f"\n{n_ok}/{len(results)} tests passed" + ("" if n_ok == len(results) else "  <-- DO NOT TRAIN until every test is green"))
    return results
