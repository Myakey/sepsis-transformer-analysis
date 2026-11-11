"""Loading, the v3 label fix, the FROZEN split (loaded, never re-derived), fold guards, the Preprocessor and the tensor cache.

Governance: nothing here prints or returns row-level data to a notebook cell; callers print the aggregate `stats` dicts only.
"""
import os, json, hashlib, types
import numpy as np
import pandas as pd

from . import config as C
from . import v3_preprocessor as _v3p


# ---------------------------------------------------------------- columns ----------------------------------------------------------------
def _resolve(name, available):
    low = {a.lower(): a for a in available}
    if name in available: return name
    if name.lower() in low: return low[name.lower()]
    raise KeyError(name)


def resolve_static(names, cfg=None):
    cfg = cfg or C.CFG; out = {}
    for k in C.STATIC_KEYS:
        try: out[k] = _resolve(getattr(cfg, k), names)
        except KeyError: raise KeyError(f"static column '{getattr(cfg, k)}' not found; edit Cfg.{k}. Columns: {list(names)}")
    return out


# ---------------------------------------------------------------- load + label fix (v3 cell 4, logic unchanged) ----------------------------------------------------------------
def load_hourly(path=None, features=None, cfg=None, extra_columns=()):
    """-> (df, static_real, stats). Label = 1 iff 0 <= HoursToOnset <= LEAD_HOURS; every post-onset row is dropped.
    Only the columns the model needs are read. `extra_columns` is for ANALYSIS-ONLY covariates (e.g. SOFA_total); they are returned
    in the frame but can never become inputs because the Preprocessor refuses forbidden columns."""
    import pyarrow.parquet as pq
    cfg = cfg or C.CFG; path = path or C.PATHS.DATA_PATH; features = list(features or C.DECAY_FEATURES)
    C.assert_no_forbidden(features)
    names = pq.read_schema(path).names
    static_real = resolve_static(names, cfg)
    need = [cfg.ID_COL, cfg.SUBJECT_COL, cfg.HOUR_COL, "SepsisLabel", "HoursToOnset"] + features + list(static_real.values()) + list(extra_columns)
    miss = [c for c in need if c not in names]
    if miss: raise KeyError(f"missing columns in parquet: {miss}")
    df = pd.read_parquet(path, columns=list(dict.fromkeys(need)))
    return fix_labels(df, features, static_real, cfg)


def fix_labels(df, features, static_real, cfg=None):
    cfg = cfg or C.CFG; n_raw = len(df)
    for c in features + [static_real["STATIC_AGE"], static_real["STATIC_HOSP"]]:
        df[c] = df[c].astype(np.float32)
    df["HoursToOnset"] = df["HoursToOnset"].astype(np.float64)
    df = df.rename(columns={"SepsisLabel": "SepsisLabel_raw_broadcast"})
    raw_pos_rate = float(df["SepsisLabel_raw_broadcast"].mean())
    df["SepsisLabel"] = ((df["HoursToOnset"] >= 0) & (df["HoursToOnset"] <= cfg.LEAD_HOURS)).astype(np.int32)
    df = df[df["HoursToOnset"].isna() | (df["HoursToOnset"] >= 0)].copy()          # drop ALL post-onset rows
    df = df.drop(columns=["SepsisLabel_raw_broadcast"]); df.reset_index(drop=True, inplace=True)
    stats = dict(rows_raw=n_raw, raw_broadcast_pos_rate=raw_pos_rate, rows=len(df), patients=int(df[cfg.SUBJECT_COL].nunique()),
                 stays=int(df[cfg.ID_COL].nunique()), pos_rate=float(df["SepsisLabel"].mean()))
    return df, static_real, stats


def stay_onset(df, cfg=None):
    """Onset hour per stay (NaN for non-septic stays): ICULOS + HoursToOnset is constant within a stay."""
    cfg = cfg or C.CFG
    s = (df[cfg.HOUR_COL] + df["HoursToOnset"]).groupby(df[cfg.ID_COL]).min(); s.index = s.index.astype(np.int64); return s


# ---------------------------------------------------------------- frozen split ----------------------------------------------------------------
def split_sha1(sp):
    return hashlib.sha1(pd.util.hash_pandas_object(sp[["subject_id", "fold"]], index=False).values.tobytes()).hexdigest()[:12]


def load_frozen_split(sources=None, expect_sha=None, cfg=None):
    """The split is a data definition. v4 only ever LOADS it and checks its fingerprint; there is deliberately no code path that rebuilds it."""
    cfg = cfg or C.CFG; sources = sources or C.PATHS.SPLIT_SOURCES
    expect_sha = cfg.SPLIT_SHA1 if expect_sha is None else expect_sha
    for src in sources:
        if src and os.path.exists(src):
            sp = pd.read_parquet(src); sha = split_sha1(sp)
            if expect_sha and sha != expect_sha:
                raise RuntimeError(f"frozen split at {src} has sha1 {sha}, expected {expect_sha}. Refusing to continue: a different split "
                                   "could move subjects into or out of the reserved LOC/CONF/EXT folds.")
            return sp, sha
    raise FileNotFoundError(f"frozen split not found in {list(sources)}. v4 never re-derives it; restore the v2/v3 subject_split.parquet.")


class FoldGuard:
    """Which folds a notebook stage may load. FIT/VAL always; LOC and EXT only in the causal stage; CONF only once the
    pre-registration is frozen (Docs/PREREGISTRATION.md plus the stamp file Docs/PREREGISTRATION.frozen)."""
    STAGE_FOLDS = {"train": {"FIT", "VAL"}, "correlational": {"FIT", "VAL"}, "causal": {"FIT", "VAL", "LOC", "CONF", "EXT"},
                   "fix": {"FIT", "VAL", "CONF"}, "final_ext": {"FIT", "VAL", "EXT"}}

    def __init__(self, stage="train", docs_dir=None):
        if stage not in self.STAGE_FOLDS: raise ValueError(f"unknown stage '{stage}'")
        self.stage, self.docs_dir = stage, docs_dir or C.PATHS.DOCS_DIR

    def prereg_frozen(self):
        return os.path.exists(os.path.join(self.docs_dir, "PREREGISTRATION.md")) and os.path.exists(os.path.join(self.docs_dir, "PREREGISTRATION.frozen"))

    def check(self, folds):
        folds = set(folds); bad = folds - self.STAGE_FOLDS[self.stage]
        if bad: raise PermissionError(f"stage '{self.stage}' may not load folds {sorted(bad)} (allowed: {sorted(self.STAGE_FOLDS[self.stage])})")
        if folds & {"CONF", "EXT", "LOC"} and not self.prereg_frozen():
            raise PermissionError("reserved folds stay untouched until Docs/PREREGISTRATION.md is frozen (PREREGISTRATION.frozen stamp missing)")
        return True


def attach_folds(df, split, folds=("FIT", "VAL"), guard=None, cfg=None):
    """Map subject -> fold, prove no subject spans folds, and DROP every other fold from memory."""
    cfg = cfg or C.CFG; guard = guard or FoldGuard("train"); guard.check(folds)
    df["fold"] = df[cfg.SUBJECT_COL].map(split.set_index("subject_id")["fold"])
    if not df["fold"].notna().all(): raise RuntimeError("some subjects in the data are absent from the frozen split table")
    if not (df.groupby(cfg.SUBJECT_COL)["fold"].nunique() == 1).all(): raise RuntimeError("a subject spans folds (leak)")
    counts = df.groupby("fold")[cfg.SUBJECT_COL].nunique().to_dict()
    df = df[df["fold"].isin(list(folds))].copy(); df.reset_index(drop=True, inplace=True)
    assert set(df["fold"].unique()) <= set(folds)
    return df, counts


def smoke_subsample(df, fraction=None, min_subjects=20, seed=1, cfg=None):
    cfg = cfg or C.CFG; fraction = cfg.SMOKE_STAY_FRACTION if fraction is None else fraction
    rng = np.random.default_rng(seed); keep = []
    for f in sorted(df["fold"].unique()):
        s = np.sort(df.loc[df["fold"] == f, cfg.SUBJECT_COL].unique())
        keep += list(rng.choice(s, size=min(len(s), max(min_subjects, int(len(s) * fraction))), replace=False))
    out = df[df[cfg.SUBJECT_COL].isin(keep)].copy(); out.reset_index(drop=True, inplace=True); return out


# ---------------------------------------------------------------- Preprocessor ----------------------------------------------------------------
class Preprocessor(_v3p.Preprocessor):
    """v3's Preprocessor, byte-for-byte (v3_preprocessor.py), with two additions: the feature list and static columns are arguments
    (reduced-feature cross-cohort models need that), and forbidden columns are refused at construction.
    fit() uses FIT rows only; transform() is forward-fill + time-since-measurement only (no backward fill anywhere)."""
    def __init__(self, features=None, static_cols=None, cfg=None):
        self.cfg = cfg or C.CFG
        feats = list(features or C.DECAY_FEATURES)
        C.assert_no_forbidden(feats)
        self._eng_core = [c for c in C.ENG_CORE if c in feats]
        for need in ("HR", "SBP", "DBP", "BUN", "Creatinine"):
            if need not in feats: raise KeyError(f"feature '{need}' is required by the engineered channels")
        self._static = dict(static_cols or {k: getattr(self.cfg, k) for k in C.STATIC_KEYS})
        self._bind(feats)
        super().__init__()

    def _bind(self, feats=None):
        """v3's class reads these names from module scope; bind this instance's values before every call."""
        feats = feats if feats is not None else self.feat
        eng_names = C.ENG_LOCAL_NAMES + [f"{c}_{s}" for c in self._eng_core for s in ("d1h", "d6h", "mean6h", "std6h")] + ["density6h"]
        _v3p.CFG, _v3p.DECAY_FEATURES, _v3p.STATIC_REAL = self.cfg, feats, self._static
        _v3p.HORIZONS, _v3p.H = list(self.cfg.HORIZONS), len(self.cfg.HORIZONS)
        _v3p.ENG_LOCAL_NAMES, _v3p.ENG_CORE, _v3p.ENG_NAMES, _v3p.ENG_DIM = C.ENG_LOCAL_NAMES, self._eng_core, eng_names, len(eng_names)

    def fit(self, df, fit_mask, fold=None):
        """`fold` (the fold label of every row) is optional but, when given, proves the statistics come from FIT rows only."""
        if fold is not None and not (np.asarray(fold)[np.asarray(fit_mask)] == "FIT").all():
            raise PermissionError("Preprocessor.fit received non-FIT rows: normalisation statistics are FIT-only")
        self._bind(); return super().fit(df, fit_mask)

    def transform(self, df, with_labels=True):
        self._bind(); return super().transform(df, with_labels=with_labels)

    @classmethod
    def from_dict(cls, d, cfg=None):
        o = cls(features=d["features"], static_cols=d["static_cols"], cfg=cfg); o.state = d; return o

    def config_hash(self):
        return hashlib.sha1(json.dumps(self.state, sort_keys=True, default=str).encode()).hexdigest()[:12]


def prevalence_alpha(store, fit_idx, cfg=None):
    """FIT prevalence per horizon -> focal alpha and the sqrt-odds positive weight (v3 cell 5)."""
    fit_rows = np.repeat(np.isin(np.arange(len(store.starts)), fit_idx), store.lens)
    pr = [float(store.Y[fit_rows, k][store.V[fit_rows, k]].mean()) for k in range(store.Y.shape[1])]
    prev = float(np.mean(pr)); return pr, 1.0 - prev, float(min(30.0, np.sqrt((1.0 - prev) / max(prev, 1e-9))))


# ---------------------------------------------------------------- tensor cache ----------------------------------------------------------------
_STORE_ARRAYS = ("X", "M", "D", "XL", "S", "G", "E", "hours", "Y", "V", "ids", "starts", "lens", "x_mean", "pos_cs", "stay_has_pos", "stay_fold")


def store_cache_key(split_sha, pre, data_stats, folds, cfg=None):
    cfg = cfg or C.CFG
    blob = json.dumps(dict(split=split_sha, pre=pre.config_hash(), stats=data_stats, folds=sorted(folds), horizons=list(cfg.HORIZONS),
                           lead=cfg.LEAD_HOURS, smoke=cfg.SMOKE_TEST), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def save_store(store, path):
    os.makedirs(os.path.dirname(path), exist_ok=True); tmp = path + ".tmp.npz"
    arrs = {k: getattr(store, k) for k in _STORE_ARRAYS if hasattr(store, k)}
    arrs = {k: (np.asarray(v).astype(str) if np.asarray(v).dtype == object else v) for k, v in arrs.items()}     # no pickles in the cache
    np.savez(tmp, n_rows=store.n_rows, gap_violations=store.gap_violations, **arrs)
    os.replace(tmp, path)


def load_store(path):
    z = np.load(path, allow_pickle=False); s = types.SimpleNamespace(**{k: z[k] for k in z.files})
    s.n_rows = int(s.n_rows); s.gap_violations = int(s.gap_violations); return s


def build_store(df, pre, cache_path=None):
    """Preprocess once and cache the tensors (key the filename with store_cache_key)."""
    if cache_path and os.path.exists(cache_path): return load_store(cache_path), True
    store = pre.transform(df)
    if cache_path: save_store(store, cache_path)
    return store, False
