"""sepsis_core.config: the single place every v4 notebook reads paths, constants and hyperparameters from.

To run on another host, change only `Paths` (or set the environment variables named below). Nothing else holds a path.
"""
import os, json, random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

# <project>/Notebooks/v4/sepsis_core/config.py -> the project root is inferred from where the package lives,
# so the only place that names the Drive folder is the bootstrap cell that puts Notebooks/v4 on sys.path.
PROJECT_ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
DRIVE = os.environ.get("SEPSIS_DRIVE_ROOT", "/content/drive/MyDrive")


@dataclass
class Paths:
    PROJECT_ROOT: str = str(PROJECT_ROOT)
    DOCS_DIR: str = str(PROJECT_ROOT / "Docs")
    V3_NOTEBOOK: str = str(PROJECT_ROOT / "Notebooks" / "sepsis_softmax_pipeline_v3.ipynb")
    # --- read-only inputs (same locations v3 used) ---
    DATA_PATH: str = f"{DRIVE}/datasets/sepsis_physionet/physionet_style_hourly.parquet"
    SPLIT_SOURCES: tuple = (f"{DRIVE}/sepsis_softmax_pass3/subject_split.parquet", f"{DRIVE}/sepsis_softmax_pass2/subject_split.parquet")
    V3_FINAL_DIR: str = f"{DRIVE}/sepsis_softmax_pass3/v3/final_i0pe1e0ema0ah0"       # v3 checkpoints + correlational_summary.json (read-only)
    PHYSIONET_ZIP: str = f"{DRIVE}/Datasets/sepsis_training_data.zip"
    PHYSIONET_DIR: str = "/content/sepsis_data"
    # --- everything v4 writes (checkpoints, caches, manifest). Never a v1-v3 folder. ---
    WORK_DIR: str = os.environ.get("SEPSIS_WORK_DIR", f"{DRIVE}/sepsis_v4")
    LOCAL_DIR: str = os.environ.get("SEPSIS_LOCAL_DIR", "/content/sepsis_work_v4")


PATHS = Paths()


def require_mounted(paths=None, need_data=True):
    """Fail early, with the fix in the message, instead of deep inside a run."""
    p = paths or PATHS
    if not Path(p.PROJECT_ROOT, "Notebooks", "v4", "sepsis_core").is_dir():
        raise FileNotFoundError(f"Project folder not found at '{p.PROJECT_ROOT}'. Mount Google Drive and set PROJECT_ROOT in the notebook's first cell "
                                "(or the SEPSIS_PROJECT_ROOT environment variable) to the folder that contains Notebooks/ and Docs/.")
    if need_data and not os.path.exists(p.DATA_PATH):
        raise FileNotFoundError(f"Hourly parquet not found at '{p.DATA_PATH}'. Is Drive mounted? Edit sepsis_core/config.py -> Paths.DATA_PATH if it moved.")
    for d in (p.WORK_DIR, p.LOCAL_DIR):
        os.makedirs(d, exist_ok=True)
    return p


@dataclass
class Cfg:
    SMOKE_TEST: bool = False
    SEED: int = 42
    SEEDS: tuple = (42, 1042, 2042, 3042, 4042)          # 5 paired seeds per arm; the first three are v3's
    RUN_NAME: str = "v4"
    ALLOW_RESUME: bool = True
    # ---- columns ----
    ID_COL: str = "stay_id"; SUBJECT_COL: str = "subject_id"; HOUR_COL: str = "ICULOS"; LABEL_COL: str = "SepsisLabel"
    STATIC_AGE: str = "Age"; STATIC_HOSP: str = "HospAdmTime"; STATIC_U1: str = "Unit1"; STATIC_U2: str = "Unit2"; STATIC_GENDER: str = "Gender"
    # ---- label / horizons (unchanged from v3) ----
    LEAD_HOURS: int = 6
    HORIZONS: tuple = (1, 3, 6)
    # ---- model constraints ----
    RHO: float = 0.25
    SIGMA: float = 0.15
    MAX_SEQ_LEN: int = 336            # v4: the longest stay is 336 h, so one window covers every stay (v3: 168, most-recent window only)
    V3_VIEW_HOURS: int = 168          # bridge to v3 numbers: metrics restricted to each stay's most recent 168 h
    EVAL_STRIDE: int = None           # only used if a stay is longer than MAX_SEQ_LEN (never on MIMIC/PhysioNet); None -> MAX_SEQ_LEN // 2
    Z_CLIP: float = 8.0
    EARLY_ONSET_HOURS: int = 12
    # ---- training (v3 values) ----
    WARMUP_FRAC: float = 0.05
    SIGMOID_WARMUP_MULT: float = 2.0  # PDF Part D: about 2x warmup for the sigmoid arm
    LR_FLOOR: float = 0.02
    GRAD_CLIP: float = 1.0
    EVAL_BATCH: int = 128
    EVAL_CELL_BUDGET: int = 128 * 168 * 168    # eval batches shrink so B*T*T never exceeds v3's peak (T is 336 now)
    NUM_WORKERS: int = 2
    SELECT_METRIC: str = "auroc_mean"
    SOUP_TOP_K: int = 5
    FINAL_MAX_EPOCHS: int = 24
    FINAL_PATIENCE: int = 8
    # ---- frozen split ----
    SPLIT_SHA1: str = "65441cff1a47"
    RESERVED_FOLDS: tuple = ("LOC", "CONF", "EXT")
    # ---- calibration ----
    CAL_METHOD: str = "auto"; CAL_FOLDS: int = 2; CAL_BINS: int = 15
    # ---- correlational constants carried from v3 (pre-registered there; changing them is a deviation) ----
    CORR_SEVERITY_COL: str = "SOFA_total"
    CORR_N_STAYS: int = 800; CORR_MIN_LEN: int = 24; CORR_MIN_T: int = 8; CORR_Q_PER_STAY: int = 24
    CORR_SESOI: float = 0.05; CORR_BAND_Q: float = 95.0; CORR_TOP_K: int = 6
    # ---- MLflow ----
    USE_DAGSHUB: bool = True
    MLFLOW_EXPERIMENT: str = "sepsis-attention-v4"
    # ---- smoke ----
    SMOKE_STAY_FRACTION: float = 0.03

    def apply_smoke(self):
        self.SMOKE_TEST = True
        self.FINAL_MAX_EPOCHS, self.FINAL_PATIENCE, self.SOUP_TOP_K = 2, 2, 2
        if not self.RUN_NAME.endswith("_smoke"): self.RUN_NAME += "_smoke"
        return self


CFG = Cfg()

# ---- feature roles (identical to v3) ----
DECAY_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3",
    "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
    "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
    "GCS", "UrineOut_mL", "PF_ratio",
]
assert len(DECAY_FEATURES) == 36
FORBIDDEN_INPUT_COLUMNS = frozenset({
    "ICULOS", "SepsisLabel", "SepsisLabel_raw_broadcast",
    "SOFA_total", "SOFA_resp", "SOFA_coag", "SOFA_liver", "SOFA_cv", "SOFA_cns", "SOFA_renal", "SOFA_delta",
    "SuspInfection", "HoursSinceSusp", "SepsisStay", "SepsisOnsetHour", "HoursToOnset",
})
STATIC_KEYS = ("STATIC_AGE", "STATIC_HOSP", "STATIC_U1", "STATIC_U2", "STATIC_GENDER")
ENG_LOCAL_NAMES = ["log_shock_index", "log_bun_cr", "pulse_pressure", "frac_measured"]
ENG_CORE = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Lactate", "WBC", "Creatinine", "Platelets", "Bilirubin_total", "BUN", "GCS"]
ENG_NAMES = ENG_LOCAL_NAMES + [f"{c}_{s}" for c in ENG_CORE for s in ("d1h", "d6h", "mean6h", "std6h")] + ["density6h"]
ENG_DIM = len(ENG_NAMES)
ENG_LOCAL_COLS = list(range(len(ENG_LOCAL_NAMES))) + [ENG_DIM - 1]
ENG_ALL_COLS = list(range(ENG_DIM))
HORIZONS = list(CFG.HORIZONS); H = len(HORIZONS)

ARMS = ("softmax", "sigmoid", "kernel", "kernel_raw")      # primary contrast: softmax vs sigmoid. kernel* are exploratory.
DEFAULT_FLAGS = dict(use_iculos=False, eng=0, loss="focal", pos_enc=True, ema=0.0, train_all_hours=True, aux_w=0.0, arm="softmax", warmup_mult=1.0)
BASE_HP = dict(d_model=256, num_heads=8, num_layers=6, ffn_mult=4, dropout=0.1, lr=5e-4, weight_decay=1e-2, batch_size=64, focal_gamma=2.0)

# v3's winning hyperparameters, read from a file that a script extracted from the v3 notebook output (never retyped).
_hp_file = Path(__file__).with_name("v3_best_hp.json")
V3_BEST_HP = json.load(open(_hp_file))["hp"]
SMOKE_HP_OVERRIDE = dict(d_model=64, num_layers=2, batch_size=16)      # CPU-sized dry runs only
# KA-Transformer under OUR protocol (architecture from the paper, sec. 2.7; 512/6 heads is not valid -> 8 heads x 64).
KA_HP = dict(d_model=512, num_heads=8, num_layers=4, ffn_dim=2048, dropout=0.2)


def assert_no_forbidden(columns, *, allow_iculos=False, what="model input"):
    """The forbidden-column guard. ICULOS passes only inside an explicit +/-ICULOS ablation (allow_iculos=True)."""
    bad = set(columns) & FORBIDDEN_INPUT_COLUMNS
    if allow_iculos: bad.discard("ICULOS")
    if bad:
        raise PermissionError(f"forbidden column(s) configured as {what}: {sorted(bad)}. "
                              "SOFA / label / onset columns are never inputs or auxiliary targets; ICULOS only inside the +/-ICULOS ablation.")
    return True


assert_no_forbidden(set(DECAY_FEATURES) | set(ENG_CORE) | {CFG.STATIC_AGE, CFG.STATIC_HOSP, CFG.STATIC_U1, CFG.STATIC_U2, CFG.STATIC_GENDER})


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def device_setup():
    """-> (device, amp_dtype, use_grad_scaler), same policy as v3 (bf16 on A100/L4/H100, fp16 + GradScaler on T4, fp32 on CPU)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        if torch.cuda.get_device_capability()[0] >= 8: return dev, torch.bfloat16, False
        return dev, torch.float16, True
    return dev, None, False


def cfg_dict(cfg=None): return asdict(cfg or CFG)
