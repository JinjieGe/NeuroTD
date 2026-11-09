from revision_runtime import resolve_path
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(resolve_path('evidence', ''))
out=Path(resolve_path('figures_28', 'deliverables'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
orange='#D55E00';pink='#CC79A7';green='#009E73';blue='#0072B2'
s=pd.read_csv(root/'search_range/selection_by_seed.csv')
s=s[(s.grid=='current')&(s.analysis=='A')&(s.noise_std>0)]
summary=s.groupby('search_range').agg(n=('selected_W','size'),mean_rmse_ms=('selected_rmse_ms','mean'))
assert summary.n.eq(300).all()
print(summary)
counts=pd.crosstab(s.search_range,s.selected_W).reindex(['quarter','half'])
print(counts)
summary.to_csv(out/'Fig_R2_8_1_summary.csv');counts.to_csv(out/'Fig_R2_8_1_selection_counts.csv')
fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),gridspec_kw={'wspace':.48})
x=np.arange(2)
bottom=np.zeros(2)
for w,c in [(513,orange),(1025,blue)]:
 vals=counts[w].to_numpy();axs[0].bar(x,vals,bottom=bottom,color=c,width=.56,label=f'W = {w}')
 for xx,v,b in zip(x,vals,bottom):
  if v:axs[0].text(xx,b+v/2,str(v),ha='center',va='center',color='white',fontsize=10,fontweight='bold')
 bottom+=vals
axs[0].set(xticks=x,xticklabels=['⌊W/4⌋','⌊W/2⌋'],ylabel='Conditions selecting each window',ylim=(0,345),xlabel='Search half-width (samples)')
axs[0].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.11),ncol=2)
vals=summary.loc[['quarter','half'],'mean_rmse_ms'].to_numpy()
axs[1].bar(x,vals,color=[orange,blue],width=.56)
for xx,v in zip(x,vals):axs[1].text(xx,v+.28,f'{v:.3f}',ha='center',fontsize=10)
axs[1].set(xticks=x,xticklabels=['⌊W/4⌋','⌊W/2⌋'],ylabel='Mean selected-delay RMSE (ms)',ylim=(0,12.6),xlabel='Search half-width (samples)')
for i,ax in enumerate(axs):ax.text(-.18,1.08,chr(97+i),transform=ax.transAxes,fontsize=15,fontweight='bold');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.subplots_adjust(left=.1,right=.985,bottom=.21,top=.86)
for ext in ['png','pdf','svg']:fig.savefig(out/f'Fig_R2_8_1.{ext}',dpi=260)
plt.close(fig)
a=pd.read_csv(root/'reviewer5/artifact_sensitivity_by_region_session.csv')
sessions=[f'sub-01_ses-0{i}' for i in [1,2,3]]
regions=['Amygdala','Hippocampus','Middle Temporal Gyrus','Superior Temporal Gyrus']
m=a[a.session.isin(sessions)&a.region.isin(regions)].pivot(index='session',columns='region',values='median_keep_artifact_ms').reindex(sessions)
m.to_csv(out/'Fig_R2_8_2_session_medians.csv')
p=pd.read_csv(root/'reviewer5/hippocampus_stg_paired_sessions.csv').sort_values(['subject','session'])
assert len(p)==19 and p.subject.nunique()==5 and (p.hippocampus_minus_stg_ms>0).sum()==15
p.to_csv(out/'Fig_R2_8_2_paired_sessions.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7.1,3.0),gridspec_kw={'wspace':.43})
for r,c,mk,short in zip(regions,[pink,orange,green,blue],['D','s','^','v'],['AMY','HPC','MTG','STG']):
 axs[0].plot([1,2,3],m[r],color=c,marker=mk,ls=':',lw=1.4,ms=5,label=short)
axs[0].set(xticks=[1,2,3],xticklabels=['ses-01','ses-02','ses-03'],xlabel='Named session of sub-01',ylabel='Median absolute delay (ms)',ylim=(8.4,11.25))
axs[0].legend(ncol=2,frameon=False,loc='upper center')
subjects=list(p.subject.unique());colors=[orange,pink,green,blue,'#666666']
for i,(sub,col) in enumerate(zip(subjects,colors)):
 q=p[p.subject==sub];xx=i+np.linspace(-.17,.17,len(q));axs[1].scatter(xx,q.hippocampus_minus_stg_ms,color=col,s=23,zorder=3)
axs[1].axhline(0,color='black',ls='--',lw=.8)
axs[1].set(xticks=range(5),xticklabels=[s.replace('sub-','') for s in subjects],xlabel='Subject (one point per session)',ylabel='HPC − STG median delay (ms)')
axs[1].text(.03,.98,'15 / 19 sessions > 0\n5 subjects',ha='left',va='top',transform=axs[1].transAxes,fontsize=8)
for i,ax in enumerate(axs):ax.text(-.2,1.06,chr(97+i),transform=ax.transAxes,fontsize=15,fontweight='bold');ax.grid(alpha=.2);ax.set_axisbelow(True)
fig.subplots_adjust(left=.1,right=.985,bottom=.2,top=.91)
for ext in ['png','pdf','svg']:fig.savefig(out/f'Fig_R2_8_2.{ext}',dpi=260)
