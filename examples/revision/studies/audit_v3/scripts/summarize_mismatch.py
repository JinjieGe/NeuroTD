#!/usr/bin/env python -u
"""Full mismatch/timeshift summaries after s5 checkpoints exist."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, MS
import numpy as np
import pandas as pd

CK = OUT / "checkpoints/mismatch"


def load_shifts(path):
    z = np.load(path)
    return z["shifts"], z["mse0"], z["mse_opt"]


def main():
    bp = pd.read_csv(OUT / "electrodes/bundle_pairs_112.csv")
    sh = np.load(OUT / "checkpoints/bundle_shifts_150x112x508.npz")["shifts"]
    rows = []
    for r in bp.itertuples():
        matched = sh[:, r.pair_index, :]
        rec = dict(pair=f"{r.label_i}–{r.label_j}", bundle=r.bundle, contact_gap=int(r.contact_gap),
                   matched_mean_abs=float(np.abs(matched).mean() * MS),
                   matched_zero=float((matched == 0).mean()),
                   matched_boundary=float((np.abs(matched) == 500).mean()))
        for mi in (0, 1):
            arr, m0, mo = [], [], []
            missing = 0
            for t in range(150):
                p = CK / f"b_m{mi}_p{r.pair_index:03d}_t{t:03d}.npz"
                if not p.exists():
                    missing += 1
                    continue
                s, a, b = load_shifts(p)
                arr.append(np.abs(s) * MS)
                m0.append(a)
                mo.append(b)
            if not arr:
                rec[f"m{mi}_missing"] = 150
                continue
            mm = np.stack(arr)
            rec[f"m{mi}_mean_abs"] = float(mm.mean())
            rec[f"m{mi}_zero"] = float((mm == 0).mean())
            rec[f"m{mi}_delta_vs_matched"] = rec["matched_mean_abs"] - rec[f"m{mi}_mean_abs"]
            rec[f"m{mi}_improve_mean"] = float(np.mean(np.concatenate(m0) - np.concatenate(mo)))
            rec[f"m{mi}_missing"] = missing
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "mismatch/bundle_matched_vs_mismatch_maps.csv", index=False)
    print("wrote mismatch maps summary", len(rows), flush=True)

    # timeshifts if present
    sh_rows = []
    for D in (32000, 128000):
        for r in bp.itertuples():
            vals = []
            for t in range(150):
                p = CK / f"sh{D}_p{int(r.pair_index):03d}_t{t:03d}.npz"
                if p.exists():
                    s, _, _ = load_shifts(p)
                    vals.append(np.abs(s).mean() * MS)
            if vals:
                sh_rows.append(dict(pair=f"{r.label_i}–{r.label_j}", D=D, n=len(vals),
                                    mean_abs=float(np.mean(vals))))
    if sh_rows:
        pd.DataFrame(sh_rows).to_csv(OUT / "mismatch/timeshift_descriptives.csv", index=False)
        print("wrote timeshift", len(sh_rows), flush=True)


if __name__ == "__main__":
    main()
