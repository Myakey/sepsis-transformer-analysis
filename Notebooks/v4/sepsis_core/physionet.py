"""PhysioNet / Computing in Cardiology Challenge 2019 (sets A and B) in the SAME layout and label definition as the MIMIC-IV cohort.

Governance: PhysioNet 2019 rows are treated like MIMIC rows. Nothing here prints row-level data; tables are per-feature or per-cohort.

Label (Reyna et al. 2020): SepsisLabel = 1 for t >= t_sepsis - 6. So t_sepsis = (first positive hour) + 6, and we rebuild
HoursToOnset = t_sepsis - ICULOS, then apply the project's definition (1 iff 0 <= HoursToOnset <= 6; post-onset rows dropped).
Two cohort facts from the same paper matter for any comparison with MIMIC:
  - patients with t_sepsis < 4 h after ICU admission and records shorter than 8 h were excluded from the Challenge;
  - a record that is positive on its FIRST row has t_sepsis somewhere in hours 4-7: its onset is CENSORED (flagged `censored`).
Hence the primary cross-cohort comparison uses the matched stratum: stay >= 8 h, and non-septic or (onset >= 4 h and not censored).

Site note: set A is Beth Israel Deaconess, the hospital behind MIMIC: a temporal / same-site check with possible patient overlap.
Set B (Emory) is the clean external test.
"""
import glob, os, zipfile
import numpy as np
import pandas as pd

from . import config as C
from . import data as D

MIMIC_ONLY = ("GCS", "UrineOut_mL", "PF_ratio")          # MIMIC inputs PhysioNet does not have
PHYSIONET_ONLY = ("TroponinI",)                           # PhysioNet column the MIMIC models never used
ID_BASE = {"A": 910_000_000, "B": 920_000_000}


def reduced_features():
    """The intersection used by every cross-cohort model (33 of the 36 MIMIC inputs)."""
    return [f for f in C.DECAY_FEATURES if f not in MIMIC_ONLY]


def extract(zip_path=None, out_dir=None):
    zip_path = zip_path or C.PATHS.PHYSIONET_ZIP; out_dir = out_dir or C.PATHS.PHYSIONET_DIR
    if not os.path.isdir(out_dir):
        if not os.path.exists(zip_path): raise FileNotFoundError(f"PhysioNet zip not found at {zip_path} (edit Paths.PHYSIONET_ZIP in sepsis_core/config.py)")
        with zipfile.ZipFile(zip_path) as z: z.extractall(out_dir)
    found = {}
    for root, dirs, _ in os.walk(out_dir):
        for d in dirs:
            if d.lower() == "training_seta": found["A"] = os.path.join(root, d)
            if d.lower() == "training_setb": found["B"] = os.path.join(root, d)
    if set(found) != {"A", "B"}: raise FileNotFoundError(f"training_setA / training_setB not found under {out_dir}")
    return found


def read_set(dir_path, set_name, limit=None, cache_dir=None):
    """All .psv files of one set -> one frame with integer stay ids. Cached as parquet (on Drive, next to the other v4 artifacts)."""
    cache = os.path.join(cache_dir or os.path.join(C.PATHS.WORK_DIR, "cache"), f"physionet_set{set_name}{'' if limit is None else f'_{limit}'}.parquet")
    if os.path.exists(cache): return pd.read_parquet(cache)
    files = sorted(glob.glob(os.path.join(dir_path, "*.psv")))
    if limit: files = files[:limit]
    frames = []
    for i, f in enumerate(files):
        d = pd.read_csv(f, sep="|"); d[C.CFG.ID_COL] = ID_BASE[set_name] + i; frames.append(d)
    df = pd.concat(frames, ignore_index=True); os.makedirs(os.path.dirname(cache), exist_ok=True); df.to_parquet(cache)
    return df


def to_layout(df, set_name, features=None, unit_fixes=None, cfg=None):
    """Raw PhysioNet frame -> (frame in the MIMIC layout with the project's label, static_real, stats, per-stay meta).
    unit_fixes: {column: multiplier} applied to PhysioNet values (decided by you after reading the harmonisation table)."""
    cfg = cfg or C.CFG; feats = list(features or reduced_features()); ID, HR = cfg.ID_COL, cfg.HOUR_COL
    miss = [c for c in feats + [HR, "SepsisLabel", cfg.STATIC_AGE, cfg.STATIC_GENDER, cfg.STATIC_U1, cfg.STATIC_U2, cfg.STATIC_HOSP] if c not in df.columns]
    if miss: raise KeyError(f"PhysioNet set {set_name} lacks columns {miss}")
    df = df.sort_values([ID, HR]).reset_index(drop=True)
    for c, mult in (unit_fixes or {}).items(): df[c] = df[c] * float(mult)
    df[cfg.SUBJECT_COL] = df[ID]                                              # one record per patient in the Challenge data
    first_hour = df.groupby(ID)[HR].min(); raw_len = df.groupby(ID)[HR].size()
    first_pos = df.loc[df["SepsisLabel"] == 1].groupby(ID)[HR].min()
    onset = first_pos + cfg.LEAD_HOURS
    df["HoursToOnset"] = df[ID].map(onset) - df[HR]
    orig = df["SepsisLabel"].to_numpy()
    static_real = D.resolve_static(list(df.columns), cfg)
    keep = [ID, cfg.SUBJECT_COL, HR, "SepsisLabel", "HoursToOnset"] + feats + list(static_real.values())
    pre_onset = (df["HoursToOnset"].isna() | (df["HoursToOnset"] >= 0)).to_numpy()
    rebuilt = ((df["HoursToOnset"] >= 0) & (df["HoursToOnset"] <= cfg.LEAD_HOURS)).to_numpy()
    agree = float((rebuilt[pre_onset] == (orig[pre_onset] == 1)).mean())      # must be 1.0: the rebuilt label equals the Challenge label before onset
    out, static_real, stats = D.fix_labels(df[keep].copy(), feats, static_real, cfg)
    meta = pd.DataFrame({"raw_len": raw_len, "septic": raw_len.index.isin(first_pos.index), "onset": onset.reindex(raw_len.index),
                         "censored": (first_pos == first_hour.reindex(first_pos.index)).reindex(raw_len.index).fillna(False).astype(bool)})
    n_sep = int(meta["septic"].sum())
    stats.update(set=set_name, septic_stays=n_sep, censored_onset_stays=int(meta["censored"].sum()),
                 censored_share_of_septic=float(meta["censored"].sum() / max(1, n_sep)),
                 early_onset_share=float((meta["onset"] < cfg.EARLY_ONSET_HOURS).sum() / max(1, n_sep)),
                 label_agreement_with_challenge_before_onset=agree, first_hour_is_1=float((first_hour == 1).mean()), max_len=int(raw_len.max()), min_len=int(raw_len.min()))
    return out, static_real, stats, meta


def split_patients(df, val_fraction=0.15, seed=0, cfg=None):
    """Patient-level FIT/VAL split INSIDE PhysioNet (for models trained on PhysioNet: directions 2 and 3). Adds the 'fold' column."""
    cfg = cfg or C.CFG; subj = np.sort(df[cfg.SUBJECT_COL].unique()); rng = np.random.default_rng(seed)
    val = set(rng.choice(subj, size=int(val_fraction * len(subj)), replace=False).tolist())
    df = df.copy(); df["fold"] = np.where(df[cfg.SUBJECT_COL].isin(val), "VAL", "FIT"); return df


def matched_mask(store, onset_series, censored_ids=None, min_len=8, min_onset=4):
    """Per-ROW mask of the matched stratum, defined identically for both cohorts on the processed store:
    stay length >= min_len (after dropping post-onset rows) and (non-septic or onset >= min_onset) and onset not censored."""
    ids = store.ids[store.starts].astype(np.int64)
    onset = onset_series.reindex(pd.Index(ids)).to_numpy(np.float64)
    ok = (store.lens >= min_len) & (np.isnan(onset) | (onset >= min_onset))
    if censored_ids is not None and len(censored_ids): ok &= ~np.isin(ids, np.asarray(list(censored_ids), np.int64))
    return np.repeat(ok, store.lens), dict(stays_total=int(len(ok)), stays_matched=int(ok.sum()), septic_matched=int((ok & ~np.isnan(onset)).sum()))


def harmonisation_table(frames, features, static=("Age", "HospAdmTime", "Gender", "Unit1", "Unit2"), ref="MIMIC_FIT"):
    """frames: {cohort name: frame}. Per feature and cohort: share of hours with a measurement, median, 5th and 95th percentile.
    FLAG = 'CHECK UNITS' when a cohort's median is outside 0.67x-1.5x of the reference cohort (e.g. FiO2 as fraction vs percent),
    'CHECK DENSITY' when the measured share differs by more than 0.20. Aggregates only."""
    rows = []
    for f in list(features) + [s for s in static if all(s in d.columns for d in frames.values())]:
        rec = dict(feature=f); flags = []
        for name, d in frames.items():
            v = d[f].to_numpy(np.float64); ok = np.isfinite(v)
            rec[f"{name}_measured"] = float(ok.mean())
            q = np.percentile(v[ok], [5, 50, 95]) if ok.sum() > 10 else [np.nan] * 3
            rec[f"{name}_p05"], rec[f"{name}_median"], rec[f"{name}_p95"] = (float(x) for x in q)
        for name in frames:
            if name == ref: continue
            a, b = rec[f"{ref}_median"], rec[f"{name}_median"]
            if np.isfinite(a) and np.isfinite(b) and abs(a) > 1e-9 and not (0.67 <= b / a <= 1.5): flags.append(f"CHECK UNITS ({name})")
            if abs(rec[f"{name}_measured"] - rec[f"{ref}_measured"]) > 0.20: flags.append(f"CHECK DENSITY ({name})")
        rec["flag"] = "; ".join(flags); rows.append(rec)
    return rows


def shortcut_diagnostics(store, idx=None, horizon_index=2):
    """Covariate-shift / shortcut check on one cohort (aggregates): how well do measurement density ALONE and the ICU hour ALONE rank the
    label? If density ranks sepsis in the training cohort but not (or in the opposite direction) in the external one, a density shortcut
    will not transfer."""
    from sklearn.metrics import roc_auc_score
    idx = np.arange(len(store.starts)) if idx is None else idx
    rows = np.concatenate([np.arange(store.starts[i], store.starts[i] + store.lens[i]) for i in idx])
    v = store.V[rows, horizon_index].astype(bool); y = store.Y[rows, horizon_index][v].astype(int)
    dens = store.M[rows].mean(1)[v]; dens6 = store.E[rows, -1].astype(np.float32)[v]; hrs = store.hours[rows][v]
    ok = y.min() != y.max()
    return dict(n_rows=int(v.sum()), pos_rate=float(y.mean()), measured_share_per_hour=float(store.M[rows].mean()),
                features_measured_per_hour_label0=float(dens[y == 0].mean() * store.M.shape[1]), features_measured_per_hour_label1=float(dens[y == 1].mean() * store.M.shape[1]) if ok else float("nan"),
                auroc_density_only=float(roc_auc_score(y, dens)) if ok else float("nan"), auroc_density6h_only=float(roc_auc_score(y, dens6)) if ok else float("nan"),
                auroc_icu_hour_only=float(roc_auc_score(y, hrs)) if ok else float("nan"),
                per_feature_measured_share={f: float(m) for f, m in zip(range(store.M.shape[1]), store.M[rows].mean(0))})
