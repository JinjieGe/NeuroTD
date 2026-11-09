from pathlib import Path
from collections import Counter
from itertools import combinations
import json
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P / 'independent_checks'
O.mkdir(exist_ok=True)
bp = pd.read_csv(P/'electrodes/bundle_pairs_112.csv')
pc = pd.read_csv(P/'electrodes/pair_classification.csv').set_index('pair_index')
z = np.load(P/'checkpoints/bundle_shifts_150x112x508.npz', allow_pickle=False)
sh, centers = z['shifts'], z['window_centers']
report = {'shape': list(sh.shape), 'total_windows': int(sh.size)}
oldroot = P.parent/'NeuroTD_review242_results_audit_20261031/extracted'
old = np.load(oldroot/'checkpoints/windows_all_trials.npz', allow_pickle=False)['shifts_samples']
reuse = []
for r in bp.itertuples():
    match = pc[((pc.label_i == r.label_i) & (pc.label_j == r.label_j)) |
               ((pc.label_i == r.label_j) & (pc.label_j == r.label_i))]
    if len(match):
        q = match.iloc[0]
        ref = old[:, int(match.index[0]), :]
        if q.label_i != r.label_i:
            ref = -ref
        reuse.append({'pair': f'{r.label_i}–{r.label_j}', 'mismatches': int(np.sum(sh[:,r.pair_index,:] != ref))})
report['reuse_checks'] = reuse

rows = []
for b in bp.bundle.unique():
    x = sh[:, bp.index[bp.bundle==b], :]
    a = np.abs(x)/32
    rows.append(dict(bundle=b, pooled_mean_nonzero_ms=float(a[x!=0].mean()),
                     pooled_mean_all_ms=float(a.mean()), zero_fraction=float((x==0).mean()),
                     boundary_fraction=float((np.abs(x)==500).mean())))
pd.DataFrame(rows).to_csv(O/'bundle_estimands_verified.csv', index=False)
report['bundle_estimates'] = rows
meta = pd.read_csv(oldroot/'tables/trial_metadata.csv')
maps = np.load(P/'mismatch/trial_maps.npz', allow_pickle=False)
mapscheck = []
for key in maps.files:
    for mi, m in enumerate(maps[key]):
        mapscheck.append(dict(set=key,map_id=mi, n_unique=int(len(np.unique(m))),
                              self_matches=int(np.sum(m==np.arange(150))),
                              max_reuse=int(np.bincount(m,minlength=150).max()),
                              cross_session=int(np.sum(m//50 != np.arange(150)//50)),
                              set_size_mismatches=int(np.sum(meta.set_size.to_numpy()[m] != meta.set_size.to_numpy()))))
report['mismatch_mapping'] = mapscheck

rows = []
for r in bp.itertuples():
    orig = sh[:,r.pair_index,:]
    for mi in (0,1):
        arr = np.stack([np.load(P/f'checkpoints/mismatch/b_m{mi}_p{r.pair_index:03d}_t{t:03d}.npz',allow_pickle=False)['shifts'] for t in range(150)])
        a = np.abs(arr)/32
        oa = np.abs(orig)/32
        rows.append(dict(pair=f'{r.label_i}–{r.label_j}',bundle=r.bundle,map_id=mi,
                         matched_mean_all_ms=float(oa.mean()),mismatch_mean_all_ms=float(a.mean()),
                         matched_mean_nonzero_ms=float(oa[orig!=0].mean()), mismatch_mean_nonzero_ms=float(a[arr!=0].mean()),
                         matched_zero=float((orig==0).mean()),mismatch_zero=float((arr==0).mean())))
    if r.pair_index % 28 == 0:
        print('checked mismatch pair',r.pair_index,flush=True)
df = pd.DataFrame(rows)
df['delta_all_ms'] = df.matched_mean_all_ms-df.mismatch_mean_all_ms
df['delta_nonzero_ms'] = df.matched_mean_nonzero_ms-df.mismatch_mean_nonzero_ms
df.to_csv(O/'mismatch_all_and_nonzero_verified.csv',index=False)
report['mismatch_summary'] = df.groupby('map_id').agg(n=('pair','size'),
    matched_all_median=('matched_mean_all_ms','median'),mismatch_all_median=('mismatch_mean_all_ms','median'),
    delta_all_median=('delta_all_ms','median'),delta_nz_median=('delta_nonzero_ms','median'),
    delta_all_negative=('delta_all_ms',lambda x:int((x<0).sum())),
    delta_nz_negative=('delta_nonzero_ms',lambda x:int((x<0).sum()))).reset_index().to_dict('records')
published = pd.read_csv(P/'mismatch/bundle_matched_vs_mismatch_maps.csv')
report['mismatch_max_diff_from_csv'] = max(float(np.max(np.abs(df[df.map_id==mi].mismatch_mean_all_ms.to_numpy()-published[f'm{mi}_mean_abs'].to_numpy()))) for mi in (0,1))

# Reconstruct the label mapping used by stage_s5. The summary incorrectly used
# bundle pair IDs for checkpoints that mostly use the original 60-pair IDs.
focus = [0,3,6,7,8,9,10,31,33,53,54,59]
planned = [(int(r.pair_index),r.label_i,r.label_j) for r in bp.itertuples()
           if r.contact_gap==1 and r.label_i.endswith('1')]
planned += [(pid,pc.loc[pid].label_i,pc.loc[pid].label_j) for pid in focus+[6,7,8]]
seen=set(); real={}
for pid,a,b in planned:
    if (a,b) in seen: continue
    seen.add((a,b))
    assert pid not in real, 'Unexpected checkpoint path collision'
    real[pid]=(a,b)
fixed=[]
for D in (32000,128000):
    for pid,(a,b) in real.items():
        vals=[np.abs(np.load(P/f'checkpoints/mismatch/sh{D}_p{pid:03d}_t{t:03d}.npz',allow_pickle=False)['shifts']).mean()/32 for t in range(150)]
        mislabeled=bp.loc[bp.pair_index==pid].iloc[0]
        fixed.append(dict(shift_seconds=D/32000,checkpoint_pair_id=pid,actual_pair=f'{a}–{b}',
                          published_pair=f'{mislabeled.label_i}–{mislabeled.label_j}',mean_abs_ms=float(np.mean(vals))))
pd.DataFrame(fixed).to_csv(O/'timeshift_labels_corrected.csv',index=False)
report['timeshift_mislabeled_rows'] = sum(r['actual_pair']!=r['published_pair'] for r in fixed)
report['timeshift_total_rows'] = len(fixed)
planned_qc=set(pd.read_csv(P/'qc/focus_pairs.csv').apply(lambda r:f'{r.label_i}–{r.label_j}',axis=1))
actual_qc=set(pd.read_csv(P/'qc/focus_pair_summary.csv').pair)
report['qc_unchecked_planned_pairs']=sorted(planned_qc-actual_qc)
report['qc_added_pairs']=sorted(actual_qc-planned_qc)

inj=pd.read_csv(P/'injection/recovery_all_conditions.csv')
report['injection_nan_rmse']=int(inj.rmse.isna().sum())
report['injection_fixed_rows']=int((inj.tau!='tv').sum())
summ=inj.groupby(['tau','mix','amp']).agg(n=('recover2','size'),
     median_recovery=('recover2','median'),mean_recovery=('recover2','mean'),
     min_recovery=('recover2','min'),median_rmse_samples=('rmse','median'),max_rmse_samples=('rmse','max')).reset_index()
summ['median_rmse_ms']=summ.median_rmse_samples/32
summ['max_rmse_ms']=summ.max_rmse_samples/32
summ.to_csv(O/'injection_distribution_verified.csv',index=False)
report['qc_nonfinite']=int(pd.read_csv(P/'qc/channel_trial_metrics.csv').n_nonfinite.sum())
report['qc_rows']=len(pd.read_csv(P/'qc/channel_trial_metrics.csv'))
report['epochs_descriptive_only']=pd.read_csv(P/'epochs/bundle_by_epoch.csv').groupby('epoch').mean_abs_nz.median().to_dict()
(O/'audit_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
