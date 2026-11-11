"""Synthetic data with the EXACT schema of the MIMIC-IV hourly parquet and of the frozen split table. No real patient data is
involved: every value is drawn from a random generator. Used by the test-suite and by SMOKE runs outside the credentialed environment.

The generator plants a weak, causal signal (a latent severity that rises before onset and drives vitals, labs and how often labs are
measured), so a smoke training run has something to learn and the plumbing can be checked end to end.
"""
import numpy as np
import pandas as pd

from . import config as C

_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
_BASE = dict(HR=(85, 15), O2Sat=(97, 2), Temp=(37, 0.7), SBP=(120, 20), MAP=(80, 14), DBP=(62, 12), Resp=(18, 5), EtCO2=(33, 6), BaseExcess=(0, 4),
             HCO3=(24, 4), FiO2=(0.45, 0.2), pH=(7.38, 0.07), PaCO2=(41, 9), SaO2=(95, 5), AST=(80, 150), BUN=(24, 18), Alkalinephos=(100, 80),
             Calcium=(8.3, 0.8), Chloride=(105, 6), Creatinine=(1.3, 1.2), Bilirubin_direct=(1.0, 2.0), Glucose=(135, 50), Lactate=(2.2, 1.8),
             Magnesium=(2.0, 0.4), Phosphate=(3.5, 1.2), Potassium=(4.1, 0.6), Bilirubin_total=(1.5, 3.0), Hct=(31, 6), Hgb=(10.3, 2), PTT=(40, 20),
             WBC=(11, 6), Fibrinogen=(300, 150), Platelets=(200, 100), GCS=(13, 3), UrineOut_mL=(90, 80), PF_ratio=(280, 110))
_SEV_EFFECT = dict(HR=0.9, Resp=0.8, Temp=0.5, SBP=-0.7, MAP=-0.8, DBP=-0.5, O2Sat=-0.5, Lactate=1.0, WBC=0.7, Creatinine=0.6, Platelets=-0.6,
                   Bilirubin_total=0.5, BUN=0.5, GCS=-0.7, PF_ratio=-0.6, UrineOut_mL=-0.5)
EXTRA_FORBIDDEN = ["SOFA_total", "SOFA_resp", "SOFA_coag", "SOFA_liver", "SOFA_cv", "SOFA_cns", "SOFA_renal", "SOFA_delta",
                   "SuspInfection", "HoursSinceSusp", "SepsisStay", "SepsisOnsetHour"]


def make_hourly(n_subjects=400, seed=0, max_len=120, septic_rate=0.2, early_onset_share=0.6, min_len=1):
    """-> DataFrame in the parquet's layout: stay_id, subject_id, ICULOS, SepsisLabel (raw, broadcast), HoursToOnset, the 36 features (NaN where
    unmeasured), Age, HospAdmTime, Unit1, Unit2, Gender, and the SOFA / onset columns that must never become inputs."""
    rng = np.random.default_rng(seed); frames = []; stay_id = 30_000_000
    for subj in range(10_000_000, 10_000_000 + n_subjects):
        age = float(np.clip(rng.normal(64, 16), 18, 95)); gender = int(rng.integers(0, 2))
        for _ in range(1 + int(rng.random() < 0.25)):
            stay_id += int(rng.integers(1, 50))
            n = int(np.clip(rng.lognormal(3.5, 0.8), min_len, max_len))
            septic = rng.random() < septic_rate
            onset = (int(rng.integers(1, 12)) if rng.random() < early_onset_share else int(rng.integers(12, max(13, n + 20)))) if septic else None
            total = n if onset is None else max(n, onset + int(rng.integers(0, 12)))          # septic stays keep some post-onset rows (dropped by the label fix)
            total = int(min(total, max_len)); hours = np.arange(1, total + 1)
            sev = np.zeros(total); x = rng.normal(0, 0.5)
            for i in range(total):
                x = 0.92 * x + rng.normal(0, 0.3); sev[i] = x
            if onset is not None: sev += 2.2 / (1 + np.exp(-(hours - (onset - 4)) / 2.5))      # rises over the hours before onset
            d = {C.CFG.ID_COL: np.full(total, stay_id, np.int64), C.CFG.SUBJECT_COL: np.full(total, subj, np.int64), "ICULOS": hours.astype(np.int64)}
            for f in C.DECAY_FEATURES:
                mu, sd = _BASE[f]; v = mu + sd * (_SEV_EFFECT.get(f, 0.0) * sev * 0.6 + rng.normal(0, 0.6, total))
                p_obs = 0.9 if f in _VITALS else float(np.clip(0.07 + 0.03 * rng.random(), 0, 1))
                obs = rng.random(total) < np.clip(p_obs + (0.0 if f in _VITALS else 0.03 * np.clip(sev, 0, 3)), 0, 1)
                if f in ("EtCO2", "Fibrinogen", "Bilirubin_direct") and rng.random() < 0.7: obs[:] = False       # never measured in most stays
                d[f] = np.where(obs, v, np.nan).astype(np.float32)
            d["Age"] = np.full(total, age, np.float32); d["Gender"] = np.full(total, gender, np.int64)
            d["HospAdmTime"] = np.full(total, -abs(rng.normal(30, 60)), np.float32)
            u = rng.integers(0, 3); d["Unit1"] = np.full(total, np.nan if u == 2 else float(u == 0)); d["Unit2"] = np.full(total, np.nan if u == 2 else float(u == 1))
            d["HoursToOnset"] = (onset - hours).astype(np.float64) if onset is not None else np.full(total, np.nan)
            d["SepsisLabel"] = np.full(total, int(septic), np.int64)                         # raw broadcast label, as in the parquet (never trained on)
            sofa = np.clip(np.round(2 + 2.0 * sev + rng.normal(0, 0.5, total)), 0, 24)
            d["SOFA_total"] = sofa.astype(np.float32)
            for c in EXTRA_FORBIDDEN[1:]: d[c] = np.zeros(total, np.float32)
            d["SepsisStay"] = np.full(total, int(septic), np.float32)
            d["SepsisOnsetHour"] = np.full(total, np.nan if onset is None else float(onset), np.float32)
            frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)


def make_split(df, seed=0, fractions=(("FIT", 0.50), ("VAL", 0.14), ("LOC", 0.10), ("CONF", 0.16), ("EXT", 0.10))):
    """A split table with the frozen split's columns. SYNTHETIC ONLY: the real split is loaded, never built."""
    rng = np.random.default_rng(seed); subj = np.sort(df[C.CFG.SUBJECT_COL].unique()); rng.shuffle(subj)
    edges = np.round(np.cumsum([f for _, f in fractions]) * len(subj)).astype(int); fold = np.empty(len(subj), object); lo = 0
    for (name, _), hi in zip(fractions, edges): fold[lo:hi] = name; lo = hi
    return pd.DataFrame({"subject_id": subj, "split": np.where(np.isin(fold, ["FIT", "VAL"]), "TRAIN", fold), "fold": fold,
                         "approx_date": pd.Timestamp("2150-01-01") + pd.to_timedelta(np.arange(len(subj)), unit="D"),
                         "hospital_expire_flag": rng.integers(0, 2, len(subj))})


def make_bundle(n_subjects=400, seed=0, max_len=120, folds=("FIT", "VAL"), cfg=None, **kw):
    """Synthetic end-to-end bundle through the REAL code path (label fix -> folds -> Preprocessor -> store).
    -> dict(store, pre, fit_idx, val_idx, onset, df_stats, alpha, pw, sofa (analysis-only array aligned to store rows))."""
    from . import data as D
    cfg = cfg or C.CFG
    raw = make_hourly(n_subjects, seed, max_len, **kw); split = make_split(raw, seed)
    static_real = D.resolve_static(list(raw.columns), cfg)
    keep = [cfg.ID_COL, cfg.SUBJECT_COL, cfg.HOUR_COL, "SepsisLabel", "HoursToOnset"] + C.DECAY_FEATURES + list(static_real.values()) + ["SOFA_total"]
    df, static_real, stats = D.fix_labels(raw[keep].copy(), list(C.DECAY_FEATURES), static_real, cfg)
    onset = D.stay_onset(df, cfg)
    guard = D.FoldGuard("train") if set(folds) <= {"FIT", "VAL"} else None
    if guard is None:                                  # synthetic-only: reserved folds may be materialised to test the guards themselves
        df["fold"] = df[cfg.SUBJECT_COL].map(split.set_index("subject_id")["fold"]); df = df[df["fold"].isin(folds)].reset_index(drop=True)
    else:
        df, _ = D.attach_folds(df, split, folds, guard, cfg)
    pre = D.Preprocessor(static_cols=static_real, cfg=cfg).fit(df, (df["fold"] == "FIT").to_numpy(), fold=df["fold"].to_numpy())
    store = pre.transform(df)
    order = np.lexsort((df[cfg.HOUR_COL].to_numpy(np.float64), df[cfg.ID_COL].to_numpy()))
    sofa = df["SOFA_total"].to_numpy(np.float32)[order]
    fit_idx = np.flatnonzero(store.stay_fold == "FIT"); val_idx = np.flatnonzero(store.stay_fold == "VAL")
    pr, alpha, pw = D.prevalence_alpha(store, fit_idx, cfg)
    return dict(store=store, pre=pre, fit_idx=fit_idx, val_idx=val_idx, onset=onset, df_stats=stats, alpha=alpha, pw=pw, sofa=sofa, split=split, raw=raw, df=df)
