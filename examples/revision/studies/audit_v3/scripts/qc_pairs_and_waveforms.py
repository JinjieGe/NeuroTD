#!/usr/bin/env python -u
"""Pair-level QC and representative traces. Run after s3; does not recompute delays."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, NWB, TRIAL_LEN, FS, PREV2
import numpy as np
import pandas as pd

FOCUS = [
    ("AL1", "AL2"), ("AHL6", "AHL7"), ("AHL6", "AHL8"), ("AL6", "AL7"),
    ("PHR6", "PHR7"), ("ECL5", "ECL6"), ("AHL1", "AHL2"), ("AHL1", "PHR1"),
    ("AHL2", "PHR1"), ("AHL1", "AHL8"), ("AL1", "AL8"), ("ECL1", "PHR1"),
]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pynwb import NWBHDF5IO

    ch = pd.read_csv(PREV2 / "electrodes/channel_verification_table.csv")
    c1 = ch[ch.session == "ses-01"].sort_values("waveform_column")
    lab2col = dict(zip(c1.label, c1.waveform_column.astype(int)))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "qc").mkdir(parents=True, exist_ok=True)

    rows = []
    trials_plot = [(1, 0), (2, 0), (3, 0)]
    nwbs = {s: NWBHDF5IO(str(p), "r").read() for s, p in NWB.items()}

    def wav(ses, trial):
        sl = slice(trial * TRIAL_LEN, (trial + 1) * TRIAL_LEN)
        return np.asarray(nwbs[ses].acquisition["ecephys.lfp"].data[sl, :], dtype=np.float64)

    sample_trials = [(s, t) for s in (1, 2, 3) for t in (0, 24, 49)]
    for a, b in FOCUS:
        ia, ib = lab2col[a], lab2col[b]
        for ses, trial in sample_trials:
            x, y = wav(ses, trial)[:, ia], wav(ses, trial)[:, ib]
            ident = float(np.mean(x == y))
            xc = x - x.mean()
            yc = y - y.mean()
            r0 = float(np.dot(xc, yc) / (np.linalg.norm(xc) * np.linalg.norm(yc) + 1e-18))
            drms = float(np.sqrt(np.mean((x - y) ** 2)))
            rows.append(dict(pair=f"{a}–{b}", session=f"ses-0{ses}", trial=trial,
                             identical_frac=ident, lag0_corr=r0, diff_rms=drms,
                             rms_x=float(np.sqrt(np.mean(x * x))),
                             rms_y=float(np.sqrt(np.mean(y * y)))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "qc/focus_pair_waveform_metrics.csv", index=False)
    summ = df.groupby("pair").agg(
        identical_median=("identical_frac", "median"),
        lag0_corr_median=("lag0_corr", "median"),
        diff_rms_median=("diff_rms", "median"),
    ).reset_index()
    summ.to_csv(OUT / "qc/focus_pair_summary.csv", index=False)

    # representative 200 ms snippets
    fig, axes = plt.subplots(4, 3, figsize=(11, 8), sharex=True)
    t = np.arange(int(0.2 * FS)) / FS * 1000
    for ax, (a, b) in zip(axes.ravel(), FOCUS[:12]):
        x, y = wav(1, 0)[:, lab2col[a]], wav(1, 0)[:, lab2col[b]]
        s0 = 32000
        ax.plot(t, x[s0:s0 + len(t)], lw=0.6, label=a)
        ax.plot(t, y[s0:s0 + len(t)], lw=0.6, label=b, alpha=0.8)
        ax.set_title(f"{a}–{b}", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
    axes[-1, 1].set_xlabel("ms (ses-01 trial 0, t=1.0–1.2 s)")
    fig.tight_layout()
    fig.savefig(OUT / "figures/fig_focus_pair_snippets.png", dpi=140)
    fig.savefig(OUT / "figures/fig_focus_pair_snippets.pdf")
    plt.close()
    print("qc_pairs_and_waveforms done", len(df), "rows", flush=True)


if __name__ == "__main__":
    main()
