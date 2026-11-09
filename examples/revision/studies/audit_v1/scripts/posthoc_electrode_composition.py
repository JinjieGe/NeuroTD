"""POST-HOC, descriptive only (not in the pre-specified plan, no p-values).

Added after variant D (pair-equal weighting) reversed the sign of Amygdala - STG.
Shows how the pooled region differences depend on which channel pairs / electrodes enter
each region: pair-level table, leave-one-electrode-out, hemisphere-label category,
and search-boundary (|shift| = 500 samples) windows.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import stats_core as sc  # noqa: E402

t0 = time.time()
print(f"[事后描述性分析] 开始 {time.strftime('%H:%M:%S')}", flush=True)
TAB, FIG = OUT / "tables", OUT / "figures"
z = np.load(OUT / "checkpoints/windows_all_trials.npz")
Wv = z["shifts_samples"].astype(np.int32)
absms = np.abs(Wv) * (1000 / 32000)
nz = absms > 1e-8
pairs = pd.read_csv(TAB / "channel_pair_mapping.csv")
pairs = pairs[pairs.session == "ses-01"].reset_index(drop=True)
reg = pairs.region.to_numpy()

S_p = (absms * nz).sum(axis=(0, 2))
N_p = nz.sum(axis=(0, 2))
pt = pairs[["pair_index", "region", "channel_i", "channel_j", "label_i", "label_j", "lateral_i", "lateral_j",
            "distance_mm"]].copy()
pt["hemisphere_label_category"] = np.where(pt.lateral_i == pt.lateral_j, "same_label", "different_label")
pt["N_nonzero"] = N_p
pt["zero_fraction"] = 1 - N_p / (Wv.shape[0] * Wv.shape[2])
pt["mean_nonzero_ms"] = S_p / N_p
pt["frac_shift_plus500"] = (Wv == 500).sum(axis=(0, 2)) / (Wv.shape[0] * Wv.shape[2])
pt["frac_shift_minus500"] = (Wv == -500).sum(axis=(0, 2)) / (Wv.shape[0] * Wv.shape[2])
pt["share_of_region_N"] = pt.N_nonzero / pt.groupby("region").N_nonzero.transform("sum")
pt.to_csv(TAB / "posthoc_pair_level_heterogeneity.csv", index=False)


def pooled_diffs(mask):
    mu = []
    for r in sc.REGIONS:
        m = mask & (reg == r)
        mu.append(S_p[m].sum() / N_p[m].sum() if N_p[m].sum() else np.nan)
    mu = np.array(mu)
    return mu, np.array([mu[a] - mu[b] for a, b in sc.COMPARISONS])


mu_all, d_all = pooled_diffs(np.ones(len(pairs), bool))
rows = [dict(dropped="none", region_of_dropped="", **{f"mu_{sc.REGION_SHORT[r]}": mu_all[j] for j, r in enumerate(sc.REGIONS)},
             **{f"d_{sc.REGION_SHORT[sc.REGIONS[a]]}-{sc.REGION_SHORT[sc.REGIONS[b]]}": d_all[k]
                for k, (a, b) in enumerate(sc.COMPARISONS)})]
for lab in sorted(set(pairs.label_i) | set(pairs.label_j)):
    m = ~((pairs.label_i == lab) | (pairs.label_j == lab)).to_numpy()
    r_of = pairs.region[(pairs.label_i == lab) | (pairs.label_j == lab)].iloc[0]
    mu, d = pooled_diffs(m)
    rows.append(dict(dropped=lab, region_of_dropped=r_of, **{f"mu_{sc.REGION_SHORT[r]}": mu[j] for j, r in enumerate(sc.REGIONS)},
                     **{f"d_{sc.REGION_SHORT[sc.REGIONS[a]]}-{sc.REGION_SHORT[sc.REGIONS[b]]}": d[k]
                        for k, (a, b) in enumerate(sc.COMPARISONS)}))
loo = pd.DataFrame(rows)
loo.to_csv(TAB / "posthoc_leave_one_electrode_out_pooled_differences.csv", index=False)
dcols = [c for c in loo.columns if c.startswith("d_")]
summ = []
for c in dcols:
    v = loo[c].iloc[1:]
    summ.append(dict(comparison=c[2:], all_pairs=loo[c].iloc[0], loo_min=v.min(), loo_max=v.max(),
                     n_sign_flips=int((np.sign(v) != np.sign(loo[c].iloc[0])).sum()),
                     electrode_giving_min=loo.dropped.iloc[1:][v.idxmin()], electrode_giving_max=loo.dropped.iloc[1:][v.idxmax()]))
summ = pd.DataFrame(summ)
summ.to_csv(TAB / "posthoc_leave_one_electrode_out_summary.csv", index=False)

cat_rows = []
for cat in ["same_label", "different_label"]:
    m = (pt.hemisphere_label_category == cat).to_numpy()
    mu, d = pooled_diffs(m)
    cat_rows.append(dict(pairs=cat, **{f"n_pairs_{sc.REGION_SHORT[r]}": int((m & (reg == r)).sum()) for r in sc.REGIONS},
                         **{f"mu_{sc.REGION_SHORT[r]}": mu[j] for j, r in enumerate(sc.REGIONS)},
                         **{f"d_{sc.REGION_SHORT[sc.REGIONS[a]]}-{sc.REGION_SHORT[sc.REGIONS[b]]}": d[k]
                            for k, (a, b) in enumerate(sc.COMPARISONS)}))
pd.DataFrame(cat_rows).to_csv(TAB / "posthoc_hemisphere_label_category.csv", index=False)

bnd = []
for r in sc.REGIONS:
    m = reg == r
    w = Wv[:, m]
    bnd.append(dict(region=r, n_windows=w.size, frac_plus500=float((w == 500).mean()), frac_minus500=float((w == -500).mean()),
                    frac_zero=float((w == 0).mean())))
pd.DataFrame(bnd).to_csv(TAB / "posthoc_search_boundary_windows_by_region.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
for j, r in enumerate(sc.REGIONS):
    g = pt[pt.region == r]
    ax.scatter(np.full(len(g), j) + np.random.default_rng(j).uniform(-0.2, 0.2, len(g)), g.mean_nonzero_ms,
               s=20 + 400 * g.share_of_region_N, alpha=0.7,
               c=np.where(g.hemisphere_label_category == "same_label", "tab:blue", "tab:orange"))
    ax.hlines(mu_all[j], j - 0.3, j + 0.3, color="crimson", lw=2)
ax.set_xticks(range(4)); ax.set_xticklabels([sc.REGION_SHORT[r] for r in sc.REGIONS])
ax.set_ylabel("pair mean nonzero |delay| (ms), 150 trials")
ax.set_title("Pair-level means (size = share of region's nonzero windows;\nblue = same hemisphere label, orange = different; red = pooled mean)", fontsize=9)
ax = axes[1]
for k, c in enumerate(dcols):
    v = loo[c].iloc[1:]
    ax.scatter(np.full(len(v), k), v, s=10, color="0.4")
    ax.scatter([k], [loo[c].iloc[0]], color="crimson", s=40, zorder=3)
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(range(len(dcols))); ax.set_xticklabels([c[2:] for c in dcols], rotation=30)
ax.set_ylabel("pooled Δ (ms)")
ax.set_title("Leave-one-electrode-out pooled differences (grey) vs all pairs (red)\nPOST-HOC, descriptive, no p-values", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig7_posthoc_electrode_composition.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig7_posthoc_electrode_composition.pdf", bbox_inches="tight")
pt.to_csv(FIG / "source_data/fig7_pair_level.csv", index=False)
loo.to_csv(FIG / "source_data/fig7_leave_one_electrode_out.csv", index=False)
print(summ.to_string(index=False), flush=True)
print(pd.DataFrame(cat_rows).to_string(index=False), flush=True)
print(pd.DataFrame(bnd).to_string(index=False), flush=True)
print(pt.sort_values(["region", "mean_nonzero_ms"]).to_string(index=False), flush=True)
print(f"[事后描述性分析] 完成，耗时 {time.time() - t0:.1f} s", flush=True)
