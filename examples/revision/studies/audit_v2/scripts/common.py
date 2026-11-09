"""Paths, constants and small helpers shared by the calibration / electrode pipeline."""
from __future__ import annotations
from revision_runtime import resolve_path

import hashlib
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(resolve_path('repo', ''))
OUT = Path(resolve_path("audit_v2"))
PREV = Path(resolve_path("audit_v1"))
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
DATA = Path(resolve_path("raw", "sub-01"))
NWB = {s: DATA / f"sub-01_ses-0{s}.nwb" for s in (1, 2, 3)}
TAB = OUT / "tables"
FIG = OUT / "figures"
CK = OUT / "checkpoints"
CAL = OUT / "calibration"
ELE = OUT / "electrodes"
WQC = OUT / "waveform_qc"
REAL = OUT / "realdata"
REP_DIR = OUT / "reports"
LOGS = OUT / "logs"
for _d in (TAB, FIG, FIG / "source_data", CK, CAL, ELE, WQC, REAL, REP_DIR, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 20261031
FS = 32000
TRIAL_LEN = 256000
W, H = 1001, 500
MAX_SHIFT = 500
MS_PER_SAMPLE = 1000.0 / FS
NEAR_ZERO = 1e-8
N_WORKERS = 6
REF_DRAWS = 4_000_000
REF_CHUNK = 100_000
CAL_NSIM = 2000
VAL_NSIM_ANA = 100_000
VAL_NSIM_BOOT = 10_000
VAL_NSIM_BOOT_FALLBACK = 5_000
BOOT_B = 1999
ANA_BATCH = 5000
BOOT_BATCH = 50
REAL_B = 50_000
ELEC_B = 20_000
DESCRIPTOR_URL = "https://www.nature.com/articles/s41597-020-0364-3"
DESCRIPTOR_CITE = ("Boran E, Fedele T, Klaver P, Hilfiker P, Stieglitz L, Grunwald T, Sarnthein J. "
                   "Dataset of human medial temporal lobe neurons, scalp and intracranial EEG during a verbal "
                   "working memory task. Sci Data 7, 30 (2020). doi:10.1038/s41597-020-0364-3")
RUN_STATE = OUT / "run_state.json"


def sha256_file(path: Path, on_bytes=None, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if on_bytes:
                on_bytes(done)
    return h.hexdigest()


def load_state() -> dict:
    return json.loads(RUN_STATE.read_text()) if RUN_STATE.exists() else {}


def save_state(**kw):
    st = load_state()
    st.update(kw)
    RUN_STATE.write_text(json.dumps(st, indent=2, ensure_ascii=False, default=str))


def code_hash() -> str:
    h = hashlib.sha256()
    for name in ("infer.py", "genmodels.py"):
        h.update((OUT / "scripts" / name).read_bytes())
    return h.hexdigest()[:16]


def dec(v):
    return v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
