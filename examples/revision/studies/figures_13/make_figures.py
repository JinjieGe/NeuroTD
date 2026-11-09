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
for lang in ['en','cn']:
 cn=lang=='cn'
 plt.rcParams.update({'font.family':'Arial Unicode MS' if cn else 'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'axes.linewidth':.8,'axes.edgecolor':'#555555','pdf.fonttype':42,'savefig.facecolor':'white'})
 fig,axs=plt.subplots(1,3,figsize=(10.5,3.7))
 fig.subplots_adjust(left=.07,right=.99,bottom=.23,top=.82,wspace=.38)
 for k,m in enumerate(['discrete','continuous']):
  q=s[s.model==m].sort_values('noise_std');ax=axs[k];x=np.arange(3)
  ax.plot(x,q.selected_mean_rmse_ms,'s:',color=orange,label='最小 E 选窗' if cn else 'E-selected')
  ax.plot(x,q.oracle_mean_rmse_ms,'*--',color='black',markersize=8,label='真值参考' if cn else 'Oracle')
  ax.set_yscale('log');ax.set_xticks(x,['0.02','0.10','0.20']);ax.set_xlabel('噪声标准差' if cn else 'Noise standard deviation')
  ax.set_ylabel('平均延迟 RMSE（毫秒）' if cn else 'Mean delay RMSE (ms)')
  ax.set_title(('离散延迟' if m=='discrete' else '连续延迟') if cn else m.capitalize()+' delays')
  ax.grid(True,which='major',alpha=.25);ax.legend(frameon=False,fontsize=8,loc='upper left')
  if m=='continuous':ax.set_ylim(.20,1.2)
  else:ax.set_ylim(.06,80)
 ax=axs[2];q=s[s.model=='discrete'].sort_values('noise_std');v=q.loss_gt10_count.to_numpy()/2
 ax.bar(np.arange(3)-.16,v,width=.30,color=orange,label='离散' if cn else 'Discrete')
 ax.bar(np.arange(3)+.16,[0,0,0],width=.30,edgecolor=blue,facecolor='white',label='连续' if cn else 'Continuous')
 ax.scatter(np.arange(3)+.16,[0,0,0],marker='D',color=blue,s=20,zorder=3)
 for j,n in enumerate(q.loss_gt10_count):ax.text(j-.16,v[j]+1,f'{n}/200',ha='center',fontsize=8)
 ax.set_ylim(-1.5,28);ax.set_xticks(range(3),['0.02','0.10','0.20']);ax.set_xlabel('噪声标准差' if cn else 'Noise standard deviation')
 ax.set_ylabel('选窗损失 >10 毫秒（%）' if cn else 'Selection loss >10 ms (%)');ax.set_title('较大选窗损失' if cn else 'Large selection losses');ax.legend(frameon=False,fontsize=8,loc='upper left')
 for k,ax in enumerate(axs):ax.text(-.20,1.13,'abc'[k],transform=ax.transAxes,fontsize=16,fontweight='bold');ax.spines[['top','right']].set_visible(False)
 fig.savefig(OUT/f'R1_3_{lang}.png',dpi=240);plt.close(fig)

 fig,axs=plt.subplots(1,2,figsize=(10.5,4.0));fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.84,wspace=.35)
 z=r[r.dropped=='none'].iloc[0];names=['AMY','HIP','MTG','STG'];vals=[z['mu_'+x] for x in names]
 ax=axs[0];ax.bar(names,vals,color=[gray,orange,gray,gray],width=.6)
 for j,vv in enumerate(vals):ax.text(j,vv+.13,f'{vv:.3f}',ha='center',fontsize=9)
 ax.set_ylim(0,11);ax.set_ylabel('平均绝对延迟（毫秒）' if cn else 'Mean absolute delay (ms)');ax.set_title('原触点集合' if cn else 'Original contact set')
 ax=axs[1];drops=['none','AHL1','AHL2','PHR1'];labels=['原集合','去掉 AHL1','去掉 AHL2','去掉 PHR1'] if cn else ['Original','Drop AHL1','Drop AHL2','Drop PHR1']
 for col,color,mark,label in [('d_HIP-MTG',blue,'o','HIP − MTG'),('d_HIP-STG',orange,'s','HIP − STG')]:
  yy=[r[r.dropped==dd].iloc[0][col] for dd in drops];ax.plot(range(4),yy,mark+':',color=color,label=label)
 ax.axhline(0,color='black',ls='--',lw=.8);ax.set_xticks(range(4),labels,rotation=15);ax.set_ylim(-.8,2.3);ax.set_ylabel('均值差（毫秒）' if cn else 'Difference in means (ms)');ax.set_title('海马触点组成敏感性' if cn else 'Hippocampal contact sensitivity');ax.legend(frameon=False,fontsize=9,loc='upper left')
 for k,ax in enumerate(axs):ax.text(-.15,1.10,'ab'[k],transform=ax.transAxes,fontsize=16,fontweight='bold');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
 fig.savefig(OUT/f'R3_1_{lang}.png',dpi=240);plt.close(fig)

 fig,ax=plt.subplots(figsize=(10.5,4.7));ax.set_xlim(0,10.5);ax.set_ylim(0,4.7);ax.axis('off')
 ax.text(.1,4.4,'a',fontsize=16,fontweight='bold');ax.text(.45,4.4,'已实现的计算流程' if cn else 'Implemented analysis workflow',fontsize=12,fontweight='bold')
 texts=['相邻尖峰\n波形对齐','修正间隔\n序列 $d_i$','逐 sweep\n斜率 β','细胞级\n汇总','跨细胞\n基因相关'] if cn else ['Adjacent-spike\nalignment','Adjusted\nintervals $d_i$','Per-sweep\nslope β','Cell-level\naggregation','Cross-cell\ngene correlation']
 for j,tx in enumerate(texts):
  x=.12+j*2.1;ax.add_patch(FancyBboxPatch((x,2.98),1.82,.95,boxstyle='round,pad=0.03',facecolor='#F3F6F8',edgecolor=blue,linewidth=1.1));ax.text(x+.91,3.455,tx,ha='center',va='center',fontsize=10)
  if j<4:ax.annotate('',xy=(x+2.05,3.455),xytext=(x+1.87,3.455),arrowprops={'arrowstyle':'->','color':'black'})
 ax.text(.12,2.58,r'$d_i = \alpha + \beta i + \epsilon_i$',fontsize=13);ax.text(4.1,2.58,'β：间隔变化趋势，并非传导延迟' if cn else 'β: interval trend, not a transmission delay',fontsize=10)
 ax.text(.1,2.02,'b',fontsize=16,fontweight='bold');ax.text(.45,2.02,'仍需独立验证的解释' if cn else 'Interpretation requiring additional evidence',fontsize=12,fontweight='bold')
 txs=['同条件重复\n可靠性','对照传统 ISI／适应指标\n增量价值','独立数据与干预\n机制验证'] if cn else ['Same-condition repeats\nReliability','Conventional ISI/adaptation controls\nIncremental value','Independent data and perturbation\nMechanistic validation']
 for j,tx in enumerate(txs):
  x=.12+j*3.5;ax.add_patch(FancyBboxPatch((x,.57),3.22,1.02,boxstyle='round,pad=0.03',facecolor='white',edgecolor=gray,linestyle='--',linewidth=1));ax.text(x+1.61,1.08,tx,ha='center',va='center',fontsize=10)
 ax.text(.12,.13,'流程示意；箭头不表示生物学因果关系。' if cn else 'Schematic: processing arrows do not imply biological causation.',fontsize=9,color=gray)
 fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98);fig.savefig(OUT/f'R3_2_{lang}.png',dpi=240);plt.close(fig)
print('Generated six bilingual PNGs from audited tables and a labeled workflow schematic.')
