from revision_runtime import resolve_path
import os
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).resolve().parents[1]/'.mplconfig')
import itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_analysis import ROOT,input_data,REGIONS,COMP_NAMES,pairwise,measures,save_json

t,S,N,P=input_data()
base=Path(resolve_path('audit_v1', 'tables'))
pp=pd.read_csv(base/'trial_pair_summary.csv')
rows=[]
def calc(tag,ss,nn):
    # Direct, independent delete-session formula.
    mus=ss.sum(0)/nn.sum(0);delta=pairwise(mus)
    leave=[]
    for a,b in [(0,50),(50,100),(100,150)]:
        keep=np.r_[np.arange(a),np.arange(b,150)]
        leave.append(pairwise(ss[keep].sum(0)/nn[keep].sum(0)))
    leave=np.array(leave);se=np.sqrt(2/3*np.sum((leave-leave.mean(0))**2,axis=0))
    z=np.abs(delta/se)
    # Closed-form two-sided Student-t(2) tail, independent of scipy.sf.
    p=np.minimum(1,6*(1-z/np.sqrt(z*z+2)))
    np.testing.assert_allclose(p,np.minimum(1,12*stats.t.sf(z,2)),atol=1e-12)
    for k,name in enumerate(COMP_NAMES):
        rows.append(dict(variant=tag,comparison=name,estimate_ms=delta[k],se_ms=se[k],p_bonferroni=p[k],
                         status='sensitivity_only_not_separately_calibrated'))
calc('original',S,N)
nraw=t.pivot(index=['session','trial_pos'],columns='region',values='n_windows_raw')[REGIONS].to_numpy(float)
calc('include_zero',S,nraw)
mask=~t.drop_duplicates(['session','trial_pos']).sort_values(['session','trial_pos']).artifact.to_numpy(bool)
calc('exclude_artifact',S*mask[:,None],N*mask[:,None])
for electrode in sorted(set(pp.label_i)|set(pp.label_j)):
    sub=pp[(pp.label_i!=electrode)&(pp.label_j!=electrode)]
    a=sub.groupby(['session','trial_pos','region'])[['S_nonzero_ms','N_nonzero']].sum()
    s=a.S_nonzero_ms.unstack()[REGIONS].to_numpy(float)
    n=a.N_nonzero.unstack()[REGIONS].to_numpy(float)
    if s.shape==(150,4) and np.isfinite(s).all() and (n>0).all():calc('leave_electrode_'+electrode,s,n)
sen=pd.DataFrame(rows);sen.to_csv(ROOT/'outputs/sensitivity.csv',index=False)
r=pd.read_csv(ROOT/'outputs/real_results.csv')
np.testing.assert_allclose(sen[sen.variant=='original'].p_bonferroni,r[r.method=='session_jackknife_t2'].p_family,atol=1e-12)
save_json(ROOT/'outputs/sensitivity_verification.json',dict(independent_jackknife_check_passed=True,
    independent_t2_tail_check_passed=True,artifact_trials_excluded=int((~mask).sum()),variants=sen.variant.nunique()))

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,ax=plt.subplots(figsize=(9,5.2))
x=r[r.method=='session_jackknife_t2'].reset_index(drop=True)
y=np.arange(6)
ax.errorbar(x.estimate_ms,y,xerr=np.array([x.estimate_ms-x.lower_simultaneous95_ms,x.upper_simultaneous95_ms-x.estimate_ms]),
    fmt='o',capsize=4,color='#205b83',label='Pooled contrast, session jackknife + t(2)')
sd=pd.read_csv(ROOT/'outputs/session_differences.csv',index_col=0)
for i,color in enumerate(['#a47c38','#449981','#a95364']):
    ax.scatter(sd.iloc[i],y+(i-1)*.10,marker='x',color=color,s=34,label=sd.index[i],zorder=4)
ax.axvline(0,color='grey',ls='--');ax.set_yticks(y,COMP_NAMES);ax.invert_yaxis()
ax.set_xlabel('Difference in pooled nonzero absolute delay (ms)')
ax.set_title('Original label-defined groups: one participant, three sessions')
ax.legend(fontsize=8,loc='lower left');fig.text(.12,.015,'Intervals: simultaneous 95% Bonferroni t(2); conditional on session-level assumptions.',fontsize=9)
fig.tight_layout(rect=(0,.05,1,1));fig.savefig(ROOT/'outputs/figure1_contrasts.png');plt.close(fig)

v=pd.read_csv(ROOT/'outputs/validation.csv');a=v[(v.null_type=='G')&v.alpha.eq(.05)]
methods=['session_jackknife_t2','HAC_b1_t2','model_envelope']
fig,axes=plt.subplots(1,2,figsize=(12,4.7),gridspec_kw={'width_ratios':[3,1.1]})
names=list(dict.fromkeys(a.scenario)); colors=['#205b83','#a95364','#449981']
for ax,ns in zip(axes,[names[:10],names[10:]]):
    for i,(m,color) in enumerate(zip(methods,colors)):
        q=a[a.method==m].set_index('scenario').loc[ns];xx=np.arange(len(ns))+(i-1)*.19
        ax.errorbar(xx,q.fwer,yerr=[q.fwer-q.fwer_ci_low,q.fwer_ci_high-q.fwer],fmt='o',color=color,ms=4,capsize=2,label=m)
    ax.axhline(.05,color='black',ls='--',lw=1,label='Nominal 5%');ax.set_xticks(np.arange(len(ns)),ns);ax.set_ylabel('Family-wise false positive rate')
axes[0].set_ylim(0,.115);axes[1].set_ylim(0,1.04)
axes[0].set_title('Legacy reference models');axes[1].set_title('Uncalibrated challenges')
axes[0].legend(fontsize=8);fig.text(.08,.015,'E1: AR=.97. E2: shared random group offset treated as measurement error; different dependence assumption.',fontsize=9)
fig.tight_layout(rect=(0,.05,1,1));fig.savefig(ROOT/'outputs/figure2_calibration.png');plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5))
for i,tag in enumerate(['original','include_zero','exclude_artifact','leave_electrode_AHL1']):
    q=sen[sen.variant==tag].set_index('comparison').loc[COMP_NAMES]
    ax.plot(np.arange(6),q.estimate_ms,'o-',label=tag,lw=1)
ax.axhline(0,color='grey',ls='--');ax.set_xticks(np.arange(6),COMP_NAMES);ax.set_ylabel('Difference (ms)')
ax.set_title('Sensitivity changes the composition or estimand; no automatic recalibration')
ax.legend(fontsize=9);fig.tight_layout();fig.savefig(ROOT/'outputs/figure3_sensitivity.png');plt.close(fig)
print(sen[sen.variant.isin(['original','include_zero','exclude_artifact','leave_electrode_AHL1'])].to_string(index=False))
