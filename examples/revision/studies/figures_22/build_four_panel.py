from revision_runtime import resolve_path
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

ROOT=Path(__file__).parent
OUT=ROOT/'deliverables'; OUT.mkdir(parents=True,exist_ok=True)
A=Path(resolve_path('score_uncertainty', 'results/selection_by_repeat.csv'))
d=pd.read_csv(A); d=d[d.noise_std>0].copy()
assert len(d)==1200 and not d.duplicated(['model','noise_std','replicate_id']).any()
assert np.allclose(d.selected_rmse_ms-d.oracle_rmse_ms,d.selection_loss_ms)
summary=[]
for (m,n),g in d.groupby(['model','noise_std']):
    assert len(g)==200
    summary.append(dict(model=m,noise_std=n,n=200,selected_mean_rmse_ms=g.selected_rmse_ms.mean(),oracle_mean_rmse_ms=g.oracle_rmse_ms.mean(),median_loss_ms=g.selection_loss_ms.median(),mean_loss_ms=g.selection_loss_ms.mean(),max_loss_ms=g.selection_loss_ms.max(),loss_gt10_count=int((g.selection_loss_ms>10).sum())))
pd.DataFrame(summary).to_csv(OUT/'Table_R2_2_oracle_summary.csv',index=False)
d.to_csv(OUT/'source_data_batch_A.csv',index=False)


plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'axes.edgecolor':'#666666','axes.linewidth':.8,'xtick.color':'#444444','ytick.color':'#444444','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
orange='#D55E00'; black='#111111'; noises=[.02,.10,.20]
fig,axs=plt.subplots(2,2,figsize=(10.5,8.5))
fig.subplots_adjust(left=.075,right=.985,bottom=.135,top=.91,wspace=.30,hspace=.52)
rng=np.random.default_rng(2402)
for row,m in enumerate(['discrete','continuous']):
    g=d[d.model==m]
    for col in range(2):
        ax=axs[row,col]; ax.set_axisbelow(True);ax.grid(True,which='major',color='#bcbcbc',ls='--',lw=.65,alpha=.65)
        ax.text(-.18,1.08,'abcd'[row*2+col],transform=ax.transAxes,fontsize=23,fontweight='bold',va='top')
    ax=axs[row,0]
    for field,color,marker,ls,label in [('selected_rmse_ms',orange,'s','--','NeuroTD (E-selected window)'),('oracle_rmse_ms',black,'*',':','NeuroTD (oracle window)')]:
        vals=[g[g.noise_std==n][field].mean() for n in noises]
        ax.plot(noises,vals,color=color,marker=marker,ls=ls,lw=1.6,ms=7 if marker=='s' else 10,label=label,zorder=4)
    ax.set_yscale('log');ax.set_ylim(.065,100);ax.set_yticks([.1,1,10]);ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_xticks(noises,['0.02','0.10','0.20']);ax.set_xlim(.008,.212)
    ax.set_xlabel('Noise Std');ax.set_ylabel('Mean delay RMSE (ms)')
    ax.set_title(m.capitalize()+' delays',pad=13)
    ax.legend(loc='upper left',fontsize=8.4,framealpha=.95,handlelength=2.5)
    ax.text(.97,.035,'n = 200 per noise level',ha='right',transform=ax.transAxes,fontsize=8,color='#555555')
    ax=axs[row,1]
    means=[];meds=[]
    for j,n in enumerate(noises):
        v=g[g.noise_std==n].selection_loss_ms.to_numpy();means.append(v.mean());meds.append(np.median(v))
        ax.scatter(j+rng.uniform(-.16,.16,len(v)),v,s=10,color=orange,alpha=.35,lw=0,rasterized=True,zorder=3)
    ax.plot(range(3),means,ls='--',marker='s',color=orange,lw=1.3,ms=6,label='Mean',zorder=5)
    ax.plot(range(3),meds,ls=':',marker='*',color=black,lw=1,ms=9,label='Median',zorder=6)
    ax.set_xlim(-.45,2.45);ax.set_xticks(range(3),['0.02','0.10','0.20']);ax.set_xlabel('Noise Std');ax.set_ylabel('Selection loss (ms)')
    if row==0:
        ax.set_yscale('symlog',linthresh=.1,linscale=.65);ax.set_ylim(-.025,230);ax.set_yticks([0,.1,1,10,100]);ax.set_yticklabels(['0','0.1','1','10','100'])
        ax.axhline(10,color='#777777',ls='--',lw=.85)
        for j,n in enumerate(noises):ax.text(j,145,f'{sum(g[g.noise_std==n].selection_loss_ms>10)}/200 > 10 ms',ha='center',fontsize=8)
        ax.set_title('Selection loss: all repetitions',pad=13)
    else:
        ax.set_ylim(-.012,.42);ax.set_yticks([0,.1,.2,.3,.4]);ax.set_title('Selection loss: all repetitions',pad=13)
    ax.legend(loc='upper left' if row else 'center left',fontsize=8.5,framealpha=.95)

fig.text(.075,.055,'200 independent noisy signal pairs per condition; original estimator and score E retained.',fontsize=9,color='#444444')
fig.text(.075,.032,'All observations retained. Native window-center grids; b: symmetric-log scale; d: linear scale.',fontsize=9,color='#444444')
base=OUT/'Fig_R2_2'
fig.savefig(str(base)+'.png',dpi=320)
fig.savefig(str(base)+'.pdf')
fig.savefig(str(base)+'.svg')
plt.close(fig)
