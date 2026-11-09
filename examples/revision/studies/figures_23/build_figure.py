from revision_runtime import resolve_path
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT=Path(__file__).parent
OUT=ROOT/'deliverables'
SRC=Path(resolve_path('score_uncertainty', 'results'))
d=pd.read_csv(SRC/'repeat_results.csv')
s=pd.read_csv(SRC/'selection_by_repeat.csv')
z=d[(d.noise_std==0)&(d.replicate_id==0)].copy()
s=s[s.noise_std>0].copy()
assert len(s)==1200 and not s.duplicated(['model','noise_std','replicate_id']).any()
assert set(s.selected_W)=={513,1025}
W=[33,65,129,257,513,1025]
noise=[.02,.10,.20]
z[['model','window_size','E','rmse_ms']].to_csv(OUT/'Figure_R2_3_zero_noise.csv',index=False)
counts=s.groupby(['model','noise_std','selected_W']).size().rename('count').reset_index()
counts.to_csv(OUT/'Figure_R2_3_selection_counts.csv',index=False)
pd.DataFrame({'W_samples':W,'nominal_duration_ms':[w/5120*1000 for w in W],'endpoint_span_ms':[(w-1)/5120*1000 for w in W]}).to_csv(OUT/'Table_R2_3.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'axes.edgecolor':'#666666','axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
orange='#D55E00';blue='#0072B2';black='#111111'
fig,axs=plt.subplots(2,2,figsize=(10.5,8.2))
fig.subplots_adjust(left=.085,right=.985,bottom=.12,top=.93,wspace=.28,hspace=.50)
for j,m in enumerate(['discrete','continuous']):
    g=z[z.model==m].set_index('window_size').loc[W]
    ax=axs[0,j]
    ax.plot(range(6),g.E,ls='--',marker='s',color=orange,lw=1.6,ms=6,label='Residual score E')
    mi=int(np.argmin(g.E.to_numpy()))
    ax.plot(mi,g.E.iloc[mi],'*',color=black,ms=13,zorder=5,label='Minimum E')
    ax.set_xticks(range(6),W);ax.set_xlabel('Window size (samples)');ax.set_ylabel('Residual score E')
    ax.set_title(m.capitalize()+' delays: zero noise',pad=12)
    if j==0:
        ax.set_yscale('log');ax.set_ylim(.12,900);ax.set_yticks([.1,1,10,100]);ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.annotate('0.245',(4,g.E.loc[513]),xytext=(-18,14),textcoords='offset points',ha='center',fontsize=9)
        ax.annotate('0.348',(5,g.E.loc[1025]),xytext=(0,13),textcoords='offset points',ha='center',fontsize=9)
    else:
        ax.set_ylim(.29,.67)
        for x,w,off in [(3,257,-19),(4,513,-20),(5,1025,11)]:
            ax.annotate(f'{g.E.loc[w]:.3f}',(x,g.E.loc[w]),xytext=(0,off),textcoords='offset points',ha='center',fontsize=9)
    ax.legend(loc='upper left' if j else 'upper right',fontsize=8.5,framealpha=.95)
    ax.set_xlim(-.3,5.3)
    ax=axs[1,j]
    bottom=np.zeros(3)
    for w,c in [(513,orange),(1025,blue)]:
        ns=np.array([sum((s.model==m)&(s.noise_std==n)&(s.selected_W==w)) for n in noise])
        for n in noise: assert sum((s.model==m)&(s.noise_std==n))==200
        vals=ns/2
        ax.bar(range(3),vals,bottom=bottom,width=.58,color=c,label=f'W = {w}',zorder=3)
        for x,v,b,n in zip(range(3),vals,bottom,ns):
            if n:ax.text(x,b+v/2,f'{n}/200\n({v:g}%)',ha='center',va='center',color='white',fontsize=9,fontweight='bold')
        bottom+=vals
    assert np.allclose(bottom,100)
    ax.set_ylim(0,119);ax.set_yticks([0,25,50,75,100]);ax.set_xticks(range(3),['0.02','0.10','0.20'])
    ax.set_xlabel('Noise Std');ax.set_ylabel('Selection frequency (%)')
    ax.set_title(m.capitalize()+' delays: repeated noise',pad=12)
    ax.legend(loc='upper center',ncol=2,fontsize=8.5,frameon=False,handlelength=1.2)
for k,ax in enumerate(axs.ravel()):
    ax.set_axisbelow(True);ax.grid(axis='y',color='#bcbcbc',ls='--',lw=.65,alpha=.65)
    ax.text(-.17,1.08,'abcd'[k],transform=ax.transAxes,fontsize=22,fontweight='bold',va='top')
fig.text(.085,.045,'Original estimator, E, and six candidate windows retained. n = 200 noisy signal pairs per condition.',fontsize=9,color='#444444')
fig.text(.085,.022,'Panels a–b: deterministic zero-noise examples; c–d: selection frequencies, not accuracy measures.',fontsize=9,color='#444444')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'Fig_R2_3.{ext}',dpi=320)
plt.close(fig)
