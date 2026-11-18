from revision_runtime import resolve_path
from pathlib import Path
import json, hashlib, shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
A=Path(resolve_path('figure_tables', 'Table_R2_2_oracle_summary.csv'))
B=Path(resolve_path('audit_v1', 'tables/posthoc_leave_one_electrode_out_pooled_differences.csv'))
for p in (A,B):shutil.copy2(p,ROOT/p.name)
s=pd.read_csv(A);r=pd.read_csv(B).fillna('none')
assert len(s)==6 and (s.n==200).all()
assert list(s.loc[s.model=='discrete','loss_gt10_count'])==[0,35,42]
orange='#D55E00';blue='#0072B2';green='#009E73';gray='#737373'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'axes.linewidth':.8,'axes.edgecolor':'#555555','pdf.fonttype':42,'savefig.facecolor':'white'})
fig,axs=plt.subplots(1,3,figsize=(10.5,3.7))
fig.subplots_adjust(left=.07,right=.99,bottom=.23,top=.82,wspace=.38)
for k,m in enumerate(['discrete','continuous']):
 q=s[s.model==m].sort_values('noise_std');ax=axs[k];x=np.arange(3)
 ax.plot(x,q.selected_mean_rmse_ms,'s:',color=orange,label='E-selected')
 ax.plot(x,q.oracle_mean_rmse_ms,'*--',color='black',markersize=8,label='Oracle')
 ax.set_yscale('log');ax.set_xticks(x,['0.02','0.10','0.20']);ax.set_xlabel('Noise standard deviation')
 ax.set_ylabel('Mean delay RMSE (ms)')
 ax.set_title(m.capitalize()+' delays')
 ax.grid(True,which='major',alpha=.25);ax.legend(frameon=False,fontsize=8,loc='upper left')
 if m=='continuous':ax.set_ylim(.20,1.2)
 else:ax.set_ylim(.06,80)
ax=axs[2];q=s[s.model=='discrete'].sort_values('noise_std');v=q.loss_gt10_count.to_numpy()/2
ax.bar(np.arange(3)-.16,v,width=.30,color=orange,label='Discrete')
ax.bar(np.arange(3)+.16,[0,0,0],width=.30,edgecolor=blue,facecolor='white',label='Continuous')
ax.scatter(np.arange(3)+.16,[0,0,0],marker='D',color=blue,s=20,zorder=3)
for j,n in enumerate(q.loss_gt10_count):ax.text(j-.16,v[j]+1,f'{n}/200',ha='center',fontsize=8)
ax.set_ylim(-1.5,28);ax.set_xticks(range(3),['0.02','0.10','0.20']);ax.set_xlabel('Noise standard deviation')
ax.set_ylabel('Selection loss >10 ms (%)');ax.set_title('Large selection losses');ax.legend(frameon=False,fontsize=8,loc='upper left')
for k,ax in enumerate(axs):ax.text(-.20,1.13,'abc'[k],transform=ax.transAxes,fontsize=16,fontweight='bold');ax.spines[['top','right']].set_visible(False)
fig.savefig(OUT/'R1_3_en.png',dpi=240);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10.5,4.0));fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.84,wspace=.35)
z=r[r.dropped=='none'].iloc[0];names=['AMY','HIP','MTG','STG'];vals=[z['mu_'+x] for x in names]
ax=axs[0];ax.bar(names,vals,color=[gray,orange,gray,gray],width=.6)
for j,vv in enumerate(vals):ax.text(j,vv+.13,f'{vv:.3f}',ha='center',fontsize=9)
ax.set_ylim(0,11);ax.set_ylabel('Mean absolute delay (ms)');ax.set_title('Original contact set')
ax=axs[1];drops=['none','AHL1','AHL2','PHR1'];labels=['Original','Drop AHL1','Drop AHL2','Drop PHR1']
for col,color,mark,label in [('d_HIP-MTG',blue,'o','HIP − MTG'),('d_HIP-STG',orange,'s','HIP − STG')]:
 yy=[r[r.dropped==dd].iloc[0][col] for dd in drops];ax.plot(range(4),yy,mark+':',color=color,label=label)
ax.axhline(0,color='black',ls='--',lw=.8);ax.set_xticks(range(4),labels,rotation=15);ax.set_ylim(-.8,2.3);ax.set_ylabel('Difference in means (ms)');ax.set_title('Hippocampal contact sensitivity');ax.legend(frameon=False,fontsize=9,loc='upper left')
for k,ax in enumerate(axs):ax.text(-.15,1.10,'ab'[k],transform=ax.transAxes,fontsize=16,fontweight='bold');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
fig.savefig(OUT/'R3_1_en.png',dpi=240);plt.close(fig)

fig,ax=plt.subplots(figsize=(10.5,4.7));ax.set_xlim(0,10.5);ax.set_ylim(0,4.7);ax.axis('off')
ax.text(.1,4.4,'a',fontsize=16,fontweight='bold');ax.text(.45,4.4,'Implemented analysis workflow',fontsize=12,fontweight='bold')
texts=['Adjacent-spike\nalignment','Adjusted\nintervals $d_i$','Per-sweep\nslope β','Cell-level\naggregation','Cross-cell\ngene correlation']
for j,tx in enumerate(texts):
 x=.12+j*2.1;ax.add_patch(FancyBboxPatch((x,2.98),1.82,.95,boxstyle='round,pad=0.03',facecolor='#F3F6F8',edgecolor=blue,linewidth=1.1));ax.text(x+.91,3.455,tx,ha='center',va='center',fontsize=10)
 if j<4:ax.annotate('',xy=(x+2.05,3.455),xytext=(x+1.87,3.455),arrowprops={'arrowstyle':'->','color':'black'})
ax.text(.12,2.58,r'$d_i = \alpha + \beta i + \epsilon_i$',fontsize=13);ax.text(4.1,2.58,'β: interval trend, not a transmission delay',fontsize=10)
ax.text(.1,2.02,'b',fontsize=16,fontweight='bold');ax.text(.45,2.02,'Interpretation requiring additional evidence',fontsize=12,fontweight='bold')
txs=['Same-condition repeats\nReliability','Conventional ISI/adaptation controls\nIncremental value','Independent data and perturbation\nMechanistic validation']
for j,tx in enumerate(txs):
 x=.12+j*3.5;ax.add_patch(FancyBboxPatch((x,.57),3.22,1.02,boxstyle='round,pad=0.03',facecolor='white',edgecolor=gray,linestyle='--',linewidth=1));ax.text(x+1.61,1.08,tx,ha='center',va='center',fontsize=10)
ax.text(.12,.13,'Schematic: processing arrows do not imply biological causation.',fontsize=9,color=gray)
fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98);fig.savefig(OUT/'R3_2_en.png',dpi=240);plt.close(fig)
print('Generated three English figures from audited tables and a labeled workflow schematic.')
