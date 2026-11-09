from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
O=P/'deeper_analysis_20261106'; O.mkdir(exist_ok=True)
bp=pd.read_csv(P/'electrodes/bundle_pairs_112.csv')
sh=np.load(P/'checkpoints/bundle_shifts_150x112x508.npz',allow_pickle=False)['shifts']
ab=np.abs(sh)/32
pairrows=[]; sesrows=[]; leave=[]; metricrows=[]
for r in bp.itertuples():
    x=sh[:,r.pair_index,:]; a=ab[:,r.pair_index,:]; nz=x!=0
    pairrows.append(dict(pair=f'{r.label_i}–{r.label_j}',bundle=r.bundle,i=r.label_i,j=r.label_j,
                         zero=(x==0).mean(),mean_all=a.mean(),mean_nz=a[nz].mean(),
                         boundary=(np.abs(x)==500).mean(),boundary_given_nonzero=(np.abs(x[nz])==500).mean(),
                         positive_given_nonzero=(x[nz]>0).mean(),signed_mean=x.mean()/32,
                         nearzero_nonzero=((np.abs(x[nz])<=2)).mean(),above_12ms=(a[nz]>12).mean(),
                         count_nz=nz.sum()))
    for s in range(3):
        xx=x[s*50:(s+1)*50]; aa=np.abs(xx)/32
        sesrows.append(dict(pair=f'{r.label_i}–{r.label_j}',bundle=r.bundle,session=s+1,
                            mean_nz=aa[xx!=0].mean(),mean_all=aa.mean(),zero=(xx==0).mean()))
pair=pd.DataFrame(pairrows); sesdf=pd.DataFrame(sesrows)
pair.to_csv(O/'pair_distributions.csv',index=False); sesdf.to_csv(O/'pair_by_session.csv',index=False)
for b in bp.bundle.unique():
    idx=bp.index[bp.bundle==b]; a=ab[:,idx,:]; x=sh[:,idx,:]
    p=pair[pair.bundle==b]
    metricrows.append(dict(bundle=b,pooled_nz=a[x!=0].mean(),pair_equal_nz=p.mean_nz.mean(),
                           median_pair_nz=p.mean_nz.median(),pooled_all=a.mean(),zero=(x==0).mean(),
                           boundary=(np.abs(x)==500).mean(),nonzero_boundary=(np.abs(x[x!=0])==500).mean(),
                           pooled_nz_excluding_boundary=a[(x!=0)&(np.abs(x)!=500)].mean(),
                           positive_given_nonzero=(x[x!=0]>0).mean(),
                           effective_pairs_weight_only=1/np.sum((p.count_nz/p.count_nz.sum())**2)))
    for lab in sorted(set(bp.loc[idx,'label_i'])|set(bp.loc[idx,'label_j'])):
        keep=bp.index[(bp.bundle==b)&(bp.label_i!=lab)&(bp.label_j!=lab)]
        aa=ab[:,keep,:]; xx=sh[:,keep,:]
        leave.append(dict(bundle=b,removed_channel=lab,mean_nz=aa[xx!=0].mean(),
                          delta_from_full=aa[xx!=0].mean()-a[x!=0].mean(),mean_all=aa.mean()))
pd.DataFrame(metricrows).to_csv(O/'bundle_estimand_sensitivity.csv',index=False)
pd.DataFrame(leave).to_csv(O/'leave_one_channel_out.csv',index=False)

# Re-read every full-bundle mismatch checkpoint for distributional checks.
rows=[]
for r in bp.itertuples():
    actual=pair.iloc[r.pair_index]
    for m in (0,1):
        xx=np.stack([np.load(P/f'checkpoints/mismatch/b_m{m}_p{r.pair_index:03d}_t{t:03d}.npz',allow_pickle=False)['shifts'] for t in range(150)])
        aa=np.abs(xx)/32; nz=xx!=0; z=(xx==0).mean(); mu=aa[nz].mean()
        # Exact decomposition, reference mismatch nonzero mean:
        # M_real-M_control=(z_control-z_real)*mu_control+(1-z_real)*(mu_real-mu_control)
        zero_term=(z-actual.zero)*mu
        conditional_term=(1-actual.zero)*(actual.mean_nz-mu)
        rows.append(dict(pair=actual.pair,bundle=r.bundle,map=m,control_zero=z,control_mean_nz=mu,
                         control_mean_all=aa.mean(),control_boundary=(np.abs(xx)==500).mean(),
                         control_positive_given_nonzero=(xx[nz]>0).mean(),zero_term_ms=zero_term,
                         conditional_term_ms=conditional_term,total_difference_ms=actual.mean_all-aa.mean(),
                         decomposition_residual=actual.mean_all-aa.mean()-zero_term-conditional_term))
    if r.pair_index%28==0: print('mismatch distribution',r.pair_index,flush=True)
mm=pd.DataFrame(rows); mm.to_csv(O/'mismatch_decomposition.csv',index=False)

# Descriptive association of pair fingerprints across sessions: no independence p values.
corr=[]
for metric in ['mean_all','mean_nz','zero']:
    for b in ['ALL']+list(bp.bundle.unique()):
        d=sesdf if b=='ALL' else sesdf[sesdf.bundle==b]
        piv=d.pivot(index='pair',columns='session',values=metric)
        for a,c in combinations([1,2,3],2):
            corr.append(dict(bundle=b,metric=metric,session_a=a,session_b=c,
                             spearman_rho=float(spearmanr(piv[a],piv[c]).statistic),
                             mean_abs_change=float(np.abs(piv[a]-piv[c]).mean())))
pd.DataFrame(corr).to_csv(O/'session_pattern_correlations_descriptive.csv',index=False)

# Nonzero weights for original four-region analysis, especially within vs cross bundle.
orig=pd.read_csv(P/'delays/versionA_trial_pair_summary_reused.csv')
pc=pd.read_csv(P/'electrodes/pair_classification.csv')
g=orig.groupby(['region','pair_index']).agg(S=('S_nonzero_ms','sum'),N=('N_nonzero','sum')).reset_index()
g=g.merge(pc[['pair_index','label_i','label_j','same_bundle']],on='pair_index')
g['mean_nz']=g.S/g.N
g['region_weight']=g.N/g.groupby('region').N.transform('sum')
g.to_csv(O/'original_regions_pair_weights.csv',index=False)
og=g.groupby(['region','same_bundle']).agg(S=('S','sum'),N=('N','sum'),n_pairs=('pair_index','size')).reset_index()
og['mean_nz']=og.S/og.N; og['region_weight']=og.N/og.groupby('region').N.transform('sum')
og.to_csv(O/'original_regions_within_vs_cross_bundle.csv',index=False)

inj=pd.read_csv(P/'injection/recovery_all_conditions.csv')
d=inj[(inj.tau!='tv')&(inj.tau!='600')]
d.groupby(['label','mix','amp']).agg(n=('rmse','size'),mean_recovery=('recover2','mean'),
    min_recovery=('recover2','min'),median_rmse_samples=('rmse','median'),max_rmse_samples=('rmse','max')).reset_index().to_csv(O/'injection_by_source.csv',index=False)

# Self-contained publication-style diagnostic figures, with explicit descriptive labels.
colors={'AHL':'#3569a3','AL':'#d47a28','ECL':'#488a5e','PHR':'#9668a6'}
fig,axs=plt.subplots(1,3,figsize=(14,4.4))
for b in colors:
    idx=bp.index[bp.bundle==b]; p=pair.iloc[idx]; q=mm[(mm.bundle==b)&(mm['map']==0)]
    axs[0].scatter(q.control_mean_all,p.mean_all,s=16,color=colors[b],label=b,alpha=.8)
    axs[1].scatter(q.control_mean_nz,p.mean_nz,s=16,color=colors[b],alpha=.8)
    axs[2].scatter(q.control_boundary*100,p.boundary*100,s=16,color=colors[b],alpha=.8)
for ax,lim in zip(axs,[(0,12),(0,16),(0,17)]):
    ax.plot(lim,lim,color='gray',ls='--',lw=1);ax.set_xlim(lim);ax.set_ylim(lim);ax.grid(alpha=.15)
axs[0].set(xlabel='Mismatched mean |delay| (ms)',ylabel='Matched mean |delay| (ms)',title='All windows, including zero')
axs[1].set(xlabel='Mismatched nonzero mean (ms)',ylabel='Matched nonzero mean (ms)',title='Nonzero windows only')
axs[2].set(xlabel='Mismatched boundary windows (%)',ylabel='Matched boundary windows (%)',title='Search boundary: +/-15.625 ms')
axs[0].legend(fontsize=8);fig.suptitle('112 same-bundle channel pairs: descriptive controls (map 0)',fontsize=12)
fig.tight_layout();fig.savefig(O/'matched_control_diagnostics.png',dpi=180);fig.savefig(O/'matched_control_diagnostics.pdf');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11,4.5))
mt=pd.DataFrame(metricrows)
for b in colors:
    ix=list(colors).index(b); d=sesdf[sesdf.bundle==b]
    vals=[]
    for s in range(3):
        x=sh[s*50:(s+1)*50,bp.index[bp.bundle==b],:]; vals.append(np.abs(x[x!=0]).mean()/32)
    axs[0].plot([1,2,3],vals,'o-',color=colors[b],label=b)
    lo=pd.DataFrame(leave); vals=lo[lo.bundle==b].mean_nz
    axs[1].scatter([ix]*len(vals),vals,color=colors[b],alpha=.7,s=28)
    axs[1].scatter(ix,mt.loc[mt.bundle==b,'pooled_nz'].iloc[0],color='black',marker='x',s=70)
axs[0].set(xlabel='Session',ylabel='Pooled nonzero mean |delay| (ms)',title='Changes across sessions',xticks=[1,2,3]);axs[0].legend()
axs[1].set(xticks=range(4),xticklabels=list(colors),ylabel='Pooled nonzero mean |delay| (ms)',title='Remove one channel (dots); full data (x)')
for ax in axs:ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(O/'session_and_channel_sensitivity.png',dpi=180);fig.savefig(O/'session_and_channel_sensitivity.pdf');plt.close(fig)

summary={'bundle_estimands':metricrows,
         'max_decomposition_error':float(mm.decomposition_residual.abs().max()),
         'mean_decomposition_by_map':mm.groupby('map')[['zero_term_ms','conditional_term_ms','total_difference_ms']].mean().to_dict('index'),
         'original_region_composition':og.to_dict('records'),
         'all_pair_session_correlations':[r for r in corr if r['bundle']=='ALL']}
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
