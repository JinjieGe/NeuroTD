"""Presentation-only corrections to existing scientific figures; no estimation."""
from revision_runtime import resolve_path
from pathlib import Path
import csv
import json
import os
import shutil
ROOT=Path(resolve_path('dynamic_fixed', ''))
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
os.environ['XDG_CACHE_HOME']=str(ROOT/'cache')
os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
out=ROOT/'outputs'
backup=ROOT/'review_archive/automatic_report'
backup.mkdir(parents=True,exist_ok=True)
for p in out.iterdir():
    if p.suffix in ['.md','.png'] or p.name=='sha256.json':
        if not (backup/p.name).exists(): shutil.copy2(p,backup/p.name)
conf=json.loads((ROOT/'frozen_config.json').read_text())
summary=list(csv.DictReader((out/'test_summary.csv').open()))
desc=list(csv.DictReader((out/'delay_and_E_summary.csv').open()))
labels=[f"{c['model']}\nSD={c['noise_std']:.2f}\nrho={c['ar1_rho']:g}" for c in conf['cells']]
plt.rcParams.update({'font.size':10,'figure.dpi':160,'axes.spines.top':False,'axes.spines.right':False})

fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
for ax,endpoint in zip(axes,['false_positive_rate','power']):
    for offset,mode,color,marker in [(-.14,'Primary','#2166ac','o'),(.14,'Eight-configuration maximum','#b35806','s')]:
        rs=[next(r for r in summary if int(r['cell_id'])==cid and r['endpoint']==endpoint and
            (r['primary']=='True' if mode=='Primary' else r['stage']=='family_max')) for cid in range(7)]
        rates=np.array([float(r['rate']) for r in rs])
        err=np.array([[float(r['rate'])-float(r['ci95_low']) for r in rs],[float(r['ci95_high'])-float(r['rate']) for r in rs]])
        ax.errorbar(np.arange(7)+offset,rates,yerr=err,fmt=marker,capsize=3,color=color,label=mode)
    if endpoint=='false_positive_rate':
        ax.axhline(.05,color='grey',ls='--',lw=1,label='Nominal 5%')
        ax.set_ylim(-.005,.17)
    else: ax.set_ylim(-.025,1.055)
    ax.set(xticks=range(7),xticklabels=labels,ylabel='False-positive rate' if endpoint=='false_positive_rate' else 'Detection rate')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y',alpha=.2)
handles,names=axes[0].get_legend_handles_labels()
fig.legend(handles,names,loc='outside lower center',ncol=3,frameon=False)
fig.suptitle('Whole-record tests: 200 held-out records per condition and endpoint\n95% exact binomial intervals conditional on the realized calibration set',fontsize=12)
fig.savefig(out/'01_error_rates_and_power.png');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for i,model in enumerate(['discrete','continuous']):
    cid=0 if model=='discrete' else 3
    for j,split in enumerate(['test_null','test_dynamic']):
        ax=axes[i,j]
        with np.load(ROOT/f'checkpoints/cell{cid}/{split}/rep0000.npz') as z:
            ax.plot(z['centers_H1']/5120,z['truth_H1']*1000/5120,color='black',label='Generating delay',lw=1.6)
            ax.plot(z['centers_H1']/5120,z['raw_H1']*1000/5120,color='#8c8c8c',lw=.55,alpha=.65,label='Raw H=1')
            for h,color in [(1,'#2166ac'),(256,'#1b9e77'),(513,'#b35806')]:
                ax.plot(z[f'centers_H{h}']/5120,z[f'fused_H{h}']*1000/5120,lw=1,
                    marker='o' if h>1 else None,ms=3,label=f'Fused H={h}',color=color)
        ax.set(title=f'{model}; {"constant delay" if j==0 else "dynamic delay"}; SD=0.02',xlabel='Time (s)',ylabel='Delay (ms)')
        ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
axes[0,0].set_ylim(2.1,2.6)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=5,frameon=False)
fig.suptitle('Prespecified replicate 0; sparse markers are returned centers\nLines between sparse markers guide the eye; they are not extra estimates',fontsize=12)
fig.savefig(out/'03_example_curves.png');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(14,4.5),layout='constrained',sharey=True)
for ax,split in zip(axes,['test_null','test_dynamic']):
    vals=[]
    for cid,cell in enumerate(conf['cells']):
        p=conf['primary'][cell['model']]
        d=next(d for d in desc if int(d['cell_id'])==cid and d['split']==split and int(d['hop'])==p['hop'] and d['stage']==p['stage'])
        vals.append(float(d['rmse_ms_mean']))
    bars=ax.bar(range(7),vals,color='#2166ac')
    ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
    ax.set(xticks=range(7),xticklabels=labels,ylabel='Mean delay RMSE (ms)',title='Constant-delay tests' if split=='test_null' else 'Dynamic-delay tests',ylim=(0,1.15))
    ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.2)
fig.suptitle('Primary configurations at fixed W=513; mean of 200 record-level RMSEs',fontsize=12)
fig.savefig(out/'04_delay_accuracy.png');plt.close(fig)
print('Updated figures 1, 3 and 4; original images and draft prose retained in review_archive/automatic_report.')
