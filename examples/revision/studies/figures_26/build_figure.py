from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).parent/'deliverables'
regions=['Amygdala','Hippocampus','Middle temporal gyrus','Superior temporal gyrus']
total=np.array([457200,228600,3429000,457200])
retained=np.array([381221,227669,3116228,383694])
all_trials=np.array([9.28125,10.5,9.375,8.875])
exclude=np.array([9.25,10.5,9.375,8.875])
assert total.sum()==4572000 and retained.sum()==4108812
pd.DataFrame(dict(region=regions,window_estimates=total,retained_nonzero=retained,median_all_trials_ms=all_trials,median_artifact_excluded_ms=exclude)).to_csv(P/'Fig_R2_6_source_data.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'axes.edgecolor':'#666666','axes.linewidth':.8,'pdf.fonttype':42,'svg.fonttype':'none'})
orange='#D55E00';grey='#999999';black='#111111'
fig,axs=plt.subplots(1,2,figsize=(10.5,4.8))
fig.subplots_adjust(left=.19,right=.98,bottom=.24,top=.84,wspace=.43)
y=np.arange(4)
ax=axs[0]
ax.barh(y-.16,total/1e6,height=.30,color=grey,label='Before zero-delay filter')
ax.barh(y+.16,retained/1e6,height=.30,color=orange,label='After zero-delay filter')
ax.set_yticks(y,regions);ax.invert_yaxis();ax.set_xlim(0,3.8)
ax.set_xlabel('Window count (millions)');ax.set_title('iEEG observation counts',pad=14)
ax.legend(loc='lower right',fontsize=8,framealpha=.95)
ax=axs[1]
ax.plot(all_trials,y,'o',mfc='white',mec=black,ms=10,mew=1.3,label='All 150 trials',zorder=3)
ax.plot(exclude,y,'s',color=orange,ms=5,label='134 trials (artifacts excluded)',zorder=4)
ax.set_yticks(y,['AMY','HPC','MTG','STG']);ax.invert_yaxis();ax.set_xlim(0,16)
ax.set_xlabel('Median absolute delay (ms)');ax.set_title('Artifact exclusion sensitivity',pad=14)
ax.legend(loc='upper left',fontsize=7.7,framealpha=.95)
for yy,v in zip(y,all_trials):ax.text(15.8,yy,f'{v:g}',ha='right',va='center',fontsize=9,color=black)
ax.text(.04,.03,'AMY change: −0.03125 ms\nOther median changes: 0 ms',transform=ax.transAxes,fontsize=8.3,va='bottom')
for k,ax in enumerate(axs):
 ax.set_axisbelow(True);ax.grid(axis='x',ls='--',lw=.65,color='#bcbcbc',alpha=.65)
 ax.text(-.18,1.16,'ab'[k],transform=ax.transAxes,fontsize=22,fontweight='bold')
 ax.set_ylim(3.7,-1.2)
fig.text(.19,.105,'One participant, three sessions. Counts are observations, not independent biological replicates.',fontsize=9,color='#444444')
fig.text(.19,.055,'Panel b: pooled nonzero absolute delays; coincident markers overlap. No inferential test is shown.',fontsize=9,color='#444444')
for ext in ['png','pdf','svg']:fig.savefig(P/f'Fig_R2_6.{ext}',dpi=320,facecolor='white')
