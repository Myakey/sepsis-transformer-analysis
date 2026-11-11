"""v3 Preprocessor class, extracted verbatim by script from sepsis_softmax_pipeline_v3.ipynb (cell 5 / notebook index 25). DO NOT EDIT.
The names it reads from module scope (CFG, DECAY_FEATURES, STATIC_REAL, HORIZONS, H, ENG_*) are bound by sepsis_core.preprocess before use.
sha256 of the class source: 89a3a8f29f6cc906a974f936a3693ca257df2bc214ea27f25aaeccdc49f977ed"""
import types
import numpy as np

class Preprocessor:
    """fit() learns FIT-only statistics (no leakage); transform() is the SAME code path for training data and for new data at inference.
    Features are standardised with FIT-only mean/std *before* GRU-D decay, so the decay's x_mean is 0 (= the FIT mean in z-space).
    v2 adds E (N, ENG_DIM): causal engineered channels. They are always computed; the feature flag 'eng' decides whether the model reads them."""
    def __init__(self):
        self.feat = list(DECAY_FEATURES)
        self.static_cols = dict(STATIC_REAL)
        self.state = None

    # ---------- fit ----------
    def fit(self, df, fit_mask):
        st = {"feat_mean": [], "feat_std": []}
        sub = df.loc[fit_mask]
        for c in self.feat:
            s = sub[c].astype("float64").replace([np.inf, -np.inf], np.nan)
            mu, sd = float(s.mean()), float(s.std(ddof=0))
            st["feat_mean"].append(0.0 if not np.isfinite(mu) else mu); st["feat_std"].append(1.0 if (not np.isfinite(sd) or sd < 1e-6) else sd)
        for key in ("STATIC_AGE", "STATIC_HOSP"):
            s = sub[self.static_cols[key]].astype("float64").replace([np.inf, -np.inf], np.nan)
            mu, sd = float(s.mean()), float(s.std(ddof=0))
            st[key + "_mean"] = 0.0 if not np.isfinite(mu) else mu; st[key + "_std"] = 1.0 if (not np.isfinite(sd) or sd < 1e-6) else sd
        # engineered-ratio normalisers: FIT rows only, using only rows where BOTH inputs were actually measured
        def _ms(v):
            v = np.asarray(v, dtype=np.float64); v = v[np.isfinite(v)]
            if v.size < 10: return [0.0, 1.0]
            mu_, sd_ = float(v.mean()), float(v.std(ddof=0)); return [mu_, sd_ if sd_ > 1e-6 else 1.0]
        r = {c: sub[c].astype("float64").replace([np.inf, -np.inf], np.nan) for c in ("HR", "SBP", "DBP", "BUN", "Creatinine")}
        st["eng_logSI"]    = _ms(np.log(r["HR"].clip(lower=1.0) / r["SBP"].clip(lower=1.0)).to_numpy())
        st["eng_logBUNCr"] = _ms(np.log(r["BUN"].clip(lower=0.1) / r["Creatinine"].clip(lower=0.05)).to_numpy())
        st["eng_pp"]       = _ms((r["SBP"] - r["DBP"]).to_numpy())
        st.update(z_clip=CFG.Z_CLIP, horizons=HORIZONS, features=self.feat, static_cols=self.static_cols, eng_names=ENG_NAMES, version=2)
        self.state = st; return self

    def to_dict(self): return self.state
    @classmethod
    def from_dict(cls, d):
        o = cls(); o.state = d; o.feat = d["features"]; o.static_cols = d["static_cols"]; return o

    @property
    def x_mean(self): return np.zeros(len(self.feat), dtype=np.float32)

    # ---------- transform ----------
    def transform(self, df, with_labels=True):
        assert self.state is not None, "fit() or from_dict() first"
        st, ID, HR = self.state, CFG.ID_COL, CFG.HOUR_COL
        ids = df[ID].to_numpy(); hrs = df[HR].to_numpy(np.float64)
        order = np.lexsort((hrs, ids))
        if not (order == np.arange(len(order))).all():
            df = df.iloc[order].reset_index(drop=True); ids = df[ID].to_numpy(); hrs = df[HR].to_numpy(np.float64)
        N = len(df); hours = hrs.astype(np.float32)
        new_stay = np.r_[True, ids[1:] != ids[:-1]]
        starts = np.flatnonzero(new_stay); lens = np.diff(np.r_[starts, N])
        stay_start_row = np.repeat(starts, lens)

        F = df[self.feat].to_numpy(np.float32, copy=True)
        F[~np.isfinite(F)] = np.nan
        mu = np.asarray(st["feat_mean"], np.float32); sd = np.asarray(st["feat_std"], np.float32)
        Fz = (F - mu) / sd; del F
        np.clip(Fz, -st["z_clip"], st["z_clip"], out=Fz)
        obs = ~np.isnan(Fz)
        M = obs.astype(np.float32)
        X = np.where(obs, Fz, 0.0).astype(np.float32)
        XL = np.zeros_like(X); D = np.zeros_like(X)
        ar = np.arange(N); start_hour = hours[stay_start_row]
        for j in range(len(self.feat)):
            last = np.where(obs[:, j], ar, -1); np.maximum.accumulate(last, out=last)
            has_prev = last >= stay_start_row
            lc = np.where(has_prev, last, 0)
            XL[:, j] = np.where(has_prev, Fz[lc, j], 0.0)
            D[:, j] = np.where(has_prev, hours - hours[lc], hours - start_hour)
        del Fz, obs

        # ---------------- v2 engineered channels E -- strictly causal: row i uses rows <= i of the SAME stay only ----------------
        # Built from the forward-filled z values XL (raw units recovered where a ratio needs them) and the mask M.
        # NOT included on purpose: SOFA / qSOFA / SIRS composites -- they share inputs with the severity covariate and the Sepsis-3 label,
        # which would make "attention follows severity" partly true by construction in Cell 13.
        zc = st["z_clip"]; E = np.zeros((N, ENG_DIM), np.float16)
        lo1 = np.maximum(ar - 1, stay_start_row); lo5 = np.maximum(ar - 5, stay_start_row); lo6 = np.maximum(ar - 6, stay_start_row)
        cnt6 = (ar - lo5 + 1).astype(np.float64)
        def _raw(name):
            j = self.feat.index(name); return XL[:, j].astype(np.float64) * float(sd[j]) + float(mu[j])
        def _zs(v, ms): return np.clip((v - ms[0]) / ms[1], -zc, zc)
        hr_, sbp_, dbp_, bun_, cr_ = (_raw(n_) for n_ in ("HR", "SBP", "DBP", "BUN", "Creatinine"))
        frac = M.mean(1).astype(np.float64)
        E[:, 0] = _zs(np.log(np.clip(hr_, 1.0, None) / np.clip(sbp_, 1.0, None)), st.get("eng_logSI", [0.0, 1.0]))
        E[:, 1] = _zs(np.log(np.clip(bun_, 0.1, None) / np.clip(cr_, 0.05, None)), st.get("eng_logBUNCr", [0.0, 1.0]))
        E[:, 2] = _zs(sbp_ - dbp_, st.get("eng_pp", [0.0, 1.0]))
        E[:, 3] = np.clip((frac - 0.3) / 0.2, -zc, zc)
        del hr_, sbp_, dbp_, bun_, cr_
        for ci, c in enumerate(ENG_CORE):
            j = self.feat.index(c); x = XL[:, j].astype(np.float64)
            cs = np.concatenate([[0.0], np.cumsum(x)]); cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
            m6 = (cs[ar + 1] - cs[lo5]) / cnt6; v6 = np.maximum((cs2[ar + 1] - cs2[lo5]) / cnt6 - m6 ** 2, 0.0)
            base = len(ENG_LOCAL_NAMES) + 4 * ci
            E[:, base + 0] = np.clip(x - x[lo1], -zc, zc)          # change vs previous hour
            E[:, base + 1] = np.clip(x - x[lo6], -zc, zc)          # change vs 6 h ago (or stay start)
            E[:, base + 2] = np.clip(m6, -zc, zc)                  # 6 h rolling mean
            E[:, base + 3] = np.clip(np.sqrt(v6), 0.0, zc)         # 6 h rolling std
        csf = np.concatenate([[0.0], np.cumsum(frac)])
        E[:, ENG_DIM - 1] = np.clip(((csf[ar + 1] - csf[lo5]) / cnt6 - 0.3) / 0.2, -zc, zc)
        E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
        del cs, cs2, m6, v6, csf, x

        sc = self.static_cols
        age = df[sc["STATIC_AGE"]].to_numpy(np.float64); hosp = df[sc["STATIC_HOSP"]].to_numpy(np.float64)
        age_z = np.nan_to_num((age - st["STATIC_AGE_mean"]) / st["STATIC_AGE_std"], nan=0.0, posinf=0.0, neginf=0.0)
        hosp_z = np.nan_to_num((hosp - st["STATIC_HOSP_mean"]) / st["STATIC_HOSP_std"], nan=0.0, posinf=0.0, neginf=0.0)
        age_z = np.clip(age_z, -st["z_clip"], st["z_clip"]); hosp_z = np.clip(hosp_z, -st["z_clip"], st["z_clip"])
        u1 = np.nan_to_num(df[sc["STATIC_U1"]].to_numpy(np.float64), nan=0.0); u2 = np.nan_to_num(df[sc["STATIC_U2"]].to_numpy(np.float64), nan=0.0)
        S = np.stack([age_z, hosp_z, u1, u2], 1).astype(np.float32)
        G = np.clip(np.nan_to_num(df[sc["STATIC_GENDER"]].to_numpy(np.float64), nan=0.0), 0, 1).astype(np.int8)

        Y = np.zeros((N, H), np.uint8); V = np.zeros((N, H), bool); gap_viol = 0
        if with_labels:
            lab = df[CFG.LABEL_COL].to_numpy(np.int8)
            for k, h in enumerate(HORIZONS):
                idx = np.arange(N) + h; inb = idx < N; idc = np.minimum(idx, N - 1)
                same = inb & (ids[idc] == ids)
                ok = same & ((hours[idc] - hours) == h)
                gap_viol += int((same & ~ok).sum())
                Y[:, k] = np.where(ok, lab[idc], 0); V[:, k] = ok
        s = types.SimpleNamespace(
            X=X, M=M, D=D, XL=XL, S=S, G=G, E=E, hours=hours, Y=Y, V=V, ids=ids, starts=starts, lens=lens, n_rows=N,
            x_mean=self.x_mean, gap_violations=gap_viol)
        pos_row = ((Y == 1) & V).any(1)
        s.pos_cs = np.concatenate([[0], np.cumsum(pos_row.astype(np.int64))])          # prefix count of positive rows (chunk-level has_pos)
        s.stay_has_pos = (np.add.reduceat(pos_row.astype(np.int64), starts) > 0) if with_labels else np.zeros(len(starts), bool)
        if "fold" in df.columns: s.stay_fold = df["fold"].to_numpy()[starts]
        return s
