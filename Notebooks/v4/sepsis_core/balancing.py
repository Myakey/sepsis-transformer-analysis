"""Class balancing for the lecturer track. HARD RULE: balancing touches FIT training data only. Every entry point below takes the
fold labels and raises PermissionError if anything other than FIT is present. Balanced models are separate models (one seed, the
unbalanced model's hyperparameters, no tuning) and support no interpretability claim in the main text. After balancing, recalibrate
on natural-distribution VAL (calibration.calibrate), because balancing distorts the base rate.

Two methods:
  oversample_positive_windows   sequence models. Whole training windows that contain a positive row are repeated. A window is a real
                                patient trajectory, so the temporal structure the Transformer sees stays valid.
  smote_rows                    SMOTE proper (Chawla et al. 2002), for per-hour tabular features (LightGBM) only.

Why not SMOTE for the Transformer: SMOTE interpolates between two feature vectors. An hourly ICU sequence is thousands of coupled
values (36 measurements, their missingness mask and time-since-measurement, per hour); interpolating two stays mixes hours that are
not aligned, produces fractional "was this measured" flags and breaks the forward-fill bookkeeping, so the synthetic stays are not
plausible patients. Repeating real positive windows changes the class balance without inventing physiology.
"""
import numpy as np
import torch


def _require_fit(folds, what):
    folds = np.asarray(folds)
    if folds.size == 0 or not (folds == "FIT").all():
        bad = sorted(set(folds.tolist()) - {"FIT"})
        raise PermissionError(f"{what}: balancing is allowed on FIT only, got folds {bad or '(empty)'}")


class OversampledDataset(torch.utils.data.Dataset):
    """Wraps a TRAIN StayDataset: positive-containing windows appear `ratio` times per epoch, negatives once."""
    def __init__(self, ds, ratio):
        self.ds = ds; pos = np.flatnonzero(ds.has_pos)
        self.map = np.concatenate([np.arange(len(ds))] + [pos] * (int(ratio) - 1)) if ratio > 1 and len(pos) else np.arange(len(ds))
        self.n = ds.n[self.map]; self.has_pos = ds.has_pos[self.map]; self.n_added = len(self.map) - len(ds)
    def __len__(self): return len(self.map)
    def __getitem__(self, i): return self.ds[int(self.map[i])]


def oversample_positive_windows(store, target_pos_fraction=0.5, max_ratio=50):
    """-> wrapper for engine.fit(train_dataset_wrapper=...). Chooses the repeat ratio so positive windows are ~target fraction of items."""
    def wrapper(ds):
        if ds.mode not in ("train", "last"): raise PermissionError("balancing wraps the training dataset only")
        stay_of_item = np.searchsorted(store.starts, ds.a, side="right") - 1
        _require_fit(np.asarray(store.stay_fold)[stay_of_item], "oversample_positive_windows")
        n_pos = int(ds.has_pos.sum()); n_neg = len(ds) - n_pos
        ratio = 1 if n_pos == 0 else int(np.clip(round(target_pos_fraction / (1 - target_pos_fraction) * n_neg / n_pos), 1, max_ratio))
        out = OversampledDataset(ds, ratio); out.mode = "train"; out.ratio = ratio
        print(f"[balancing] FIT only: {n_pos:,} positive windows x{ratio} -> {len(out):,} items/epoch (was {len(ds):,})")
        return out
    return wrapper


def smote_rows(X, y, folds, k=5, target_ratio=1.0, seed=0, max_new=2_000_000):
    """SMOTE on per-hour feature rows (GBDT only). X must already be imputed (finite). -> (X_balanced, y_balanced, n_synthetic)."""
    from sklearn.neighbors import NearestNeighbors
    _require_fit(folds, "smote_rows")
    X = np.asarray(X, np.float32); y = np.asarray(y).astype(bool)
    if not np.isfinite(X).all(): raise ValueError("SMOTE needs finite features (use the forward-filled matrix)")
    P = X[y]; n_new = int(min(max_new, max(0, round(target_ratio * (~y).sum()) - y.sum())))
    if len(P) < 2 or n_new == 0: return X, y.astype(np.float32), 0
    rng = np.random.default_rng(seed); k_ = min(k, len(P) - 1)
    nn_idx = NearestNeighbors(n_neighbors=k_ + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
    base = rng.integers(0, len(P), n_new); nb = nn_idx[base, rng.integers(0, k_, n_new)]
    lam = rng.random((n_new, 1), dtype=np.float32)
    S = P[base] + lam * (P[nb] - P[base])
    return np.concatenate([X, S]), np.concatenate([y, np.ones(n_new, bool)]).astype(np.float32), n_new
