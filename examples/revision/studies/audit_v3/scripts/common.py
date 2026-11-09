"""Shared paths and constants for the real-data comprehensive audit."""
from __future__ import annotations
from revision_runtime import resolve_path

import hashlib
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(resolve_path('repo', ''))
OUT = Path(resolve_path("audit_v3"))
PREV1 = Path(resolve_path("audit_v1"))
PREV2 = Path(resolve_path("audit_v2"))
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
DATA = Path(resolve_path("raw"))
NWB = {s: DATA / "sub-01" / f"sub-01_ses-0{s}.nwb" for s in (1, 2, 3)}
SEED = 20261031
FS = 32000
FS_MACRO = 2000
TRIAL_LEN = 256000
W, H, MAX_SHIFT = 1001, 500, 500
MS = 1000.0 / FS
NEAR_ZERO = 1e-8
N_WORKERS = 6
# Time-scaled macro parameters (same milliseconds as micro W/H/search)
W_MACRO = int(round(W / FS * FS_MACRO))       # 63
H_MACRO = int(round(H / FS * FS_MACRO))       # 31
MAX_SHIFT_MACRO = int(round(MAX_SHIFT / FS * FS_MACRO))  # 31
BUNDLES = ["AHL", "AL", "ECL", "PHR"]
REGIONS4 = ["Amygdala", "Hippocampus", "Middle Temporal Gyrus", "Superior Temporal Gyrus"]
# Task epochs relative to NWB trial start (Boran et al. 2019 Sci Adv; 2020 Sci Data)
# Fixation 1 s, encoding 2 s, maintenance 3 s, probe thereafter; probe at 6 s (matches NeuroTD start+6).
EPOCHS = {
    "fixation": (0.0, 1.0),
    "encoding": (1.0, 3.0),
    "maintenance": (3.0, 6.0),
    "probe": (6.0, 8.0),
    "full": (0.0, 8.0),
}
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


def bundle_of(label: str) -> str:
    i = 0
    while i < len(label) and label[i].isalpha():
        i += 1
    return label[:i]
