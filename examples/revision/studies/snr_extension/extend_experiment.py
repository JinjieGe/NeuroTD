"""Frozen-design dose-response extension; original NeuroTD source is read-only."""
from revision_runtime import resolve_path
import os
for v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[v]='1'
import sys, json, hashlib, time, platform, shutil, base64, html
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PREV=Path(resolve_path('snr_initial', ''))
sys.path.insert(0,str(PREV))
import run_analysis as old

PLAN=json.loads((ROOT/'plan.json').read_text())
DELAYS=PLAN['true_delays_samples'];SNRS=PLAN['snr_db'];REPS=PLAN['replicates']
KINDS=['white','ar1'];MODES=['y_only','both'];FS=old.FS
for folder in ['results','figures','checkpoints','provenance']:(ROOT/folder).mkdir(exist_ok=True)

def progress(stage,**values):
    v=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),stage=stage,**values)
    (ROOT/'progress.json').write_text(json.dumps(v,ensure_ascii=False,indent=2))
    print(json.dumps(v,ensure_ascii=False),flush=True)

def prepare():
    # Previous official hashes are rechecked; no raw files are copied or altered.
    official=pd.read_csv(PREV/'provenance/raw_files.csv');verified=[]
    for row in official.itertuples():
        got=old.sha(row.path);assert got==row.official_sha256
        verified.append(dict(path=row.path,sha256=got,verified=True,asset_id=row.asset_id))
    pd.DataFrame(verified).to_csv(ROOT/'provenance/raw_verified.csv',index=False)
    templates={};rows=[]
    for sess in old.SESSIONS:
        with h5py.File(old.RAW/f'sub-01_{sess}.nwb') as f:
            g=f['intervals/trials'];eligible=np.flatnonzero(~g['artifact'][:].astype(bool)&g['correct'][:].astype(bool))
            for ordinal,trial in zip(['first','middle','last'],eligible[[0,len(eligible)//2,-1]]):
                a=old.read_trial(f,int(trial))
                for channel,bundle in zip([0,8,16,24],['AHL','AL','ECL','PHR']):
                    name=f'{sess}_t{int(trial):02d}_{bundle}'
                    templates[name]=a[FS-1000:2*FS+1000,channel].copy()
                    rows.append(dict(template=name,session=sess,trial_pos=int(trial),trial_id=int(g['id'][trial]),selection=ordinal,channel=channel,bundle=bundle,set_size=int(g['set_size'][trial]),sha256=hashlib.sha256(templates[name].tobytes()).hexdigest()))
    assert len(templates)==36
    np.savez_compressed(ROOT/'checkpoints/templates.npz',**templates)
    pd.DataFrame(rows).to_csv(ROOT/'provenance/template_manifest.csv',index=False)
    files=[Path(__file__),ROOT/'plan.json',PREV/'run_analysis.py',old.SOURCE]
    (ROOT/'provenance/source_hashes.json').write_text(json.dumps({str(p):old.sha(p) for p in files},indent=2))
    (ROOT/'provenance/environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),indent=2))
    fn=old.original();checks=[]
    for ti,name in enumerate(list(templates)[::3]):
        d=DELAYS[ti%len(DELAYS)];x,y=old.make_pair(templates[name],d)
        for noisy in [False,True]:
            if noisy:
                rng=np.random.default_rng(PLAN['seed']+ti)
                kind=KINDS[ti%2]
                nx=old.noise_array(rng,kind,len(x));ny=old.noise_array(rng,kind,len(y))
                x=x+nx*np.sqrt(np.mean(x*x)*10);y=y+ny*np.sqrt(np.mean(y*y)*10)
            st=time.perf_counter();centers,s,_=old.estimate(x,y);secs=time.perf_counter()-st
            co,so=fn(x,y,old.W,old.H);m=int(np.count_nonzero(s!=so))
            checks.append(dict(template=name,true_delay_samples=d,noisy=noisy,windows=len(s),mismatches=m,fast_seconds=secs))
            assert np.array_equal(centers,co) and m==0
    pd.DataFrame(checks).to_csv(ROOT/'results/baseline_validation.csv',index=False)
    progress('prepared',templates=36,validation_records=len(checks),estimated_serial_seconds=round(np.median([a['fast_seconds'] for a in checks])*253728))

def seed_for(ti,di,ni,mi,rep):
    # Unique seed per condition; reuse only across SNR levels for paired comparisons.
    return PLAN['seed']+((((ti*len(DELAYS)+di)*2+ni)*2+mi)*REPS+rep)

def evaluate(s,d):
    s=s.astype(float);v=s/32;e=(s-d)/32
    return dict(mean_abs_ms=float(np.mean(np.abs(v))),mean_signed_ms=float(np.mean(v)),signed_bias_ms=float(np.mean(e)),rmse_ms=float(np.sqrt(np.mean(e*e))),median_abs_ms=float(np.median(np.abs(v))),within1ms_fraction=float(np.mean(np.abs(e)<=1)),zero_fraction=float(np.mean(s==0)),boundary_fraction=float(np.mean(np.abs(s)==500)),opposite_sign_fraction=float(np.mean(s*d<0)) if d!=0 else np.nan)

def run_template(args):
    ti,name,template=args;cp=ROOT/f'checkpoints/{name}.csv';sp=ROOT/f'checkpoints/{name}_shifts.npz'
    if cp.exists() and sp.exists():return name,len(pd.read_csv(cp)),0.
    rows=[];arrays=[];st=time.perf_counter()
    for di,d in enumerate(DELAYS):
        x,y=old.make_pair(template,d);px=np.mean(x*x);py=np.mean(y*y)
        assert px>0 and py>0
        _,s,_=old.estimate(x,y)
        rows.append(dict(template=name,delay_samples=d,noise='none',mode='none',snr_db=np.inf,replicate=-1,seed=-1,actual_snr_y_db=np.inf,**evaluate(s,d)));arrays.append(s)
        for ni,kind in enumerate(KINDS):
            for mi,mode in enumerate(MODES):
                for rep in range(REPS):
                    seed=seed_for(ti,di,ni,mi,rep);rng=np.random.default_rng(seed)
                    nx=old.noise_array(rng,kind,len(x));ny=old.noise_array(rng,kind,len(y))
                    for snr in SNRS:
                        scale=10**(-snr/20)
                        ex=nx*np.sqrt(px)*scale if mode=='both' else np.zeros_like(x)
                        ey=ny*np.sqrt(py)*scale
                        _,s,_=old.estimate(x+ex,y+ey)
                        actual=10*np.log10(py/np.mean(ey*ey));assert abs(actual-snr)<1e-10
                        if mode=='both':assert abs(10*np.log10(px/np.mean(ex*ex))-snr)<1e-10
                        rows.append(dict(template=name,delay_samples=d,noise=kind,mode=mode,snr_db=snr,replicate=rep,seed=seed,actual_snr_y_db=actual,**evaluate(s,d)));arrays.append(s)
    frame=pd.DataFrame(rows)
    assert len(frame)==7048 and len(arrays)==len(frame)
    np.savez_compressed(sp,shifts_samples=np.stack(arrays),window_centers=np.arange(1000,30501,500))
    temp=cp.with_suffix('.partial');frame.to_csv(temp,index=False);temp.replace(cp)
    return name,len(frame),time.perf_counter()-st

def null_control():
    rows=[];arrays=[]
    for ni,kind in enumerate(KINDS):
        for rep in range(200):
            seed=PLAN['seed']+1000000+ni*200+rep;rng=np.random.default_rng(seed)
            x=old.noise_array(rng,kind,FS);y=old.noise_array(rng,kind,FS)
            _,s,_=old.estimate(x,y)
            # No ground-truth lag exists; do not label deviation from zero as an accuracy error.
            rows.append(dict(noise=kind,replicate=rep,seed=seed,mean_abs_ms=np.mean(np.abs(s.astype(float)))/32,mean_signed_ms=np.mean(s)/32,boundary_fraction=np.mean(np.abs(s)==500)))
            arrays.append(s)
    pd.DataFrame(rows).to_csv(ROOT/'results/noise_only_control.csv',index=False)
    np.savez_compressed(ROOT/'checkpoints/noise_only_shifts.npz',shifts_samples=np.stack(arrays))
    return pd.DataFrame(rows)

def validate_production(templates):
    fn=old.original();rows=[]
    for ti,(name,template) in enumerate(templates.items()):
        df=pd.read_csv(ROOT/f'checkpoints/{name}.csv');saved=np.load(ROOT/f'checkpoints/{name}_shifts.npz')['shifts_samples']
        baseline=df.index[df.noise=='none'][ti%len(DELAYS)]
        chosen=[baseline]
        for j in [0,1]:
            di=(ti+j*3)%8;ni=(ti+j)%2;mi=(ti//2+j)%2;snr=SNRS[(ti+j*5)%len(SNRS)];rep=(ti+j)%REPS
            chosen.extend(df.index[(df.delay_samples==DELAYS[di])&(df.noise==KINDS[ni])&(df['mode']==MODES[mi])&(df.snr_db==snr)&(df.replicate==rep)].tolist())
        assert len(chosen)==3
        for idx in chosen:
            r=df.iloc[idx];x,y=old.make_pair(template,int(r.delay_samples))
            if r.noise!='none':
                rng=np.random.default_rng(int(r.seed));nx=old.noise_array(rng,r.noise,FS);ny=old.noise_array(rng,r.noise,FS)
                ex=nx*np.sqrt(np.mean(x*x))*10**(-r.snr_db/20) if r['mode']=='both' else np.zeros_like(x)
                ey=ny*np.sqrt(np.mean(y*y))*10**(-r.snr_db/20)
                x=x+ex;y=y+ey
            centers,expected=fn(x,y,old.W,old.H)
            mismatch=int(np.count_nonzero(expected!=saved[idx]));assert mismatch==0,(name,idx,mismatch)
            rows.append(dict(template=name,row=int(idx),noise=r.noise,mode=r['mode'],snr_db=r.snr_db,delay_samples=r.delay_samples,mismatches=mismatch,n_windows=len(expected)))
        if ti%6==5:progress('production_validation',templates_checked=ti+1,total_templates=36)
    pd.DataFrame(rows).to_csv(ROOT/'results/production_validation.csv',index=False)

def summarize(df):
    noisy=df[df.noise!='none'].copy();keys=['template','delay_samples','noise','mode']
    template=noisy.groupby(keys+['snr_db']).agg(n=('mean_abs_ms','size'),abs_mean=('mean_abs_ms','mean'),abs_var=('mean_abs_ms','var'),signed_mean=('mean_signed_ms','mean'),rmse_mean=('rmse_ms','mean'),boundary_mean=('boundary_fraction','mean')).reset_index()
    template.to_csv(ROOT/'results/template_summary.csv',index=False)
    rows=[]
    for k,g in template.groupby(['delay_samples','noise','mode','snr_db']):
        assert len(g)==36 and (g.n==20).all()
        rows.append(dict(zip(['delay_samples','noise','mode','snr_db'],k),n_templates=len(g),records=int(g.n.sum()),mean_abs_ms=g.abs_mean.mean(),q25_template_abs_ms=g.abs_mean.quantile(.25),q75_template_abs_ms=g.abs_mean.quantile(.75),conditional_mc_se_ms=np.sqrt(np.sum(g.abs_var/g.n))/len(g),mean_signed_ms=g.signed_mean.mean(),mean_rmse_ms=g.rmse_mean.mean(),mean_boundary_fraction=g.boundary_mean.mean()))
    summary=pd.DataFrame(rows);summary['truth_abs_ms']=summary.delay_samples.abs()/32
    summary['excess_abs_ms']=summary.mean_abs_ms-summary.truth_abs_ms
    summary.to_csv(ROOT/'results/summary.csv',index=False)
    # Paired differences: same template and same independent noise realization.
    b=noisy[noisy.snr_db==20][keys+['replicate','mean_abs_ms','rmse_ms']].rename(columns={'mean_abs_ms':'abs_20db','rmse_ms':'rmse_20db'})
    contrast=noisy.merge(b,on=keys+['replicate'],validate='many_to_one')
    contrast['change_abs_ms']=contrast.mean_abs_ms-contrast.abs_20db
    contrast['change_rmse_ms']=contrast.rmse_ms-contrast.rmse_20db
    contrast['increase_gt1ms']=contrast.change_abs_ms>1
    contrast['decrease_gt1ms']=contrast.change_abs_ms< -1
    contrast.to_csv(ROOT/'results/paired_contrasts.csv',index=False)
    contr=contrast.groupby(['delay_samples','noise','mode','snr_db']).agg(mean_change_abs_ms=('change_abs_ms','mean'),mean_change_rmse_ms=('change_rmse_ms','mean'),fraction_increase_gt1ms=('increase_gt1ms','mean'),fraction_decrease_gt1ms=('decrease_gt1ms','mean')).reset_index()
    contr['fraction_within1ms']=1-contr.fraction_increase_gt1ms-contr.fraction_decrease_gt1ms
    contr.to_csv(ROOT/'results/contrast_summary.csv',index=False)
    # Adjacent-SNR paired contrast checks monotonicity rather than just endpoints.
    step=[]
    for high,low in zip(SNRS[:-1],SNRS[1:]):
        a=noisy[noisy.snr_db==high];b=noisy[noisy.snr_db==low]
        m=a.merge(b,on=keys+['replicate'],suffixes=('_high','_low'),validate='one_to_one')
        m['step_abs_ms']=m.mean_abs_ms_low-m.mean_abs_ms_high
        g=m.groupby(['delay_samples','noise','mode']).agg(mean_change_abs_ms=('step_abs_ms','mean'),fraction_increase=('step_abs_ms',lambda v:np.mean(v>0)),fraction_increase_gt1ms=('step_abs_ms',lambda v:np.mean(v>1))).reset_index()
        g['from_snr_db']=high;g['to_snr_db']=low;step.append(g)
    pd.concat(step).to_csv(ROOT/'results/adjacent_snr_changes.csv',index=False)
    # Baseline deviations retained, never silently exclude a template if it fails.
    df[df.noise=='none'].to_csv(ROOT/'results/no_added_noise_baselines.csv',index=False)
    return summary,contr,template

def savefig(fig,name):
    fig.savefig(ROOT/f'figures/{name}.png',dpi=190);fig.savefig(ROOT/f'figures/{name}.pdf');plt.close(fig)

def plots(s,contr,t,null):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    ds=[0,80,160,240,320,400];colors=plt.get_cmap('viridis')(np.linspace(.05,.9,len(ds)))
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ni,kind in enumerate(KINDS):
        for mi,mode in enumerate(MODES):
            ax=axs[ni,mi]
            for d,color in zip(ds,colors):
                g=s[(s.noise==kind)&(s['mode']==mode)&(s.delay_samples==d)].sort_values('snr_db',ascending=False)
                ax.plot(g.snr_db,g.mean_abs_ms,'-o',ms=3,color=color,label=f'{d/32:g} ms')
                ax.fill_between(g.snr_db,g.q25_template_abs_ms,g.q75_template_abs_ms,color=color,alpha=.08)
            if mode=='both':ax.axhline(null[null.noise==kind].mean_abs_ms.mean(),color='black',ls=':',lw=1,label='Noise-only mean')
            ax.set_title(f'{kind} noise / {mode.replace("_"," ")}');ax.set_xlim(21,-21);ax.set_ylim(-.3,16);ax.set_xlabel('Added-noise SNR (dB; noise increases to the right)');ax.set_ylabel('Mean estimated |delay| (ms)');ax.grid(alpha=.2)
    axs[0,1].legend(title='True delay',ncol=2,fontsize=8)
    fig.suptitle('Does lower SNR imply a larger estimated delay?\n36 fixed recording templates; shading = template IQR, not a confidence interval')
    savefig(fig,'01_snr_delay_curves')
    fig,axs=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ni,kind in enumerate(KINDS):
        for mi,mode in enumerate(MODES):
            a=s[(s.noise==kind)&(s['mode']==mode)&(s.delay_samples>=0)].pivot(index='delay_samples',columns='snr_db',values='excess_abs_ms').reindex(index=ds,columns=SNRS)
            ax=axs[ni,mi];im=ax.imshow(a,aspect='auto',cmap='RdBu_r',vmin=-12.5,vmax=12.5)
            ax.set_xticks(range(len(SNRS)),labels=SNRS,rotation=45);ax.set_yticks(range(len(ds)),labels=[v/32 for v in ds]);ax.set_xlabel('Added-noise SNR (dB)');ax.set_ylabel('True delay (ms)');ax.set_title(f'{kind} / {mode.replace("_"," ")}')
    fig.colorbar(im,ax=axs,label='Mean |estimated delay| minus |true delay| (ms)',shrink=.8)
    fig.suptitle('Red: inflated magnitude; blue: reduced magnitude\nAn average near zero does not establish individual accuracy')
    savefig(fig,'02_inflation_deflation_map')
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,kind in zip(axs,KINDS):
        for d,color in zip(ds,colors):
            g=s[(s.noise==kind)&(s['mode']=='both')&(s.delay_samples==d)].sort_values('snr_db',ascending=False)
            ax.plot(g.snr_db,g.mean_rmse_ms,'-o',color=color,ms=3,label=f'{d/32:g} ms')
        ax.set_title(f'{kind}, both channels');ax.set_xlim(21,-21);ax.set_xlabel('Added-noise SNR (dB)');ax.set_ylabel('Mean record-wise delay RMSE (ms)');ax.grid(alpha=.2)
    axs[1].legend(title='True delay',fontsize=8,ncol=2);fig.suptitle('Accuracy can deteriorate even when the estimated magnitude decreases')
    savefig(fig,'03_delay_error')
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,kind in zip(axs,KINDS):
        for d,color in zip([-320,-80,0,80,320],plt.get_cmap('coolwarm')(np.linspace(.05,.95,5))):
            g=s[(s.noise==kind)&(s['mode']=='both')&(s.delay_samples==d)].sort_values('snr_db',ascending=False)
            ax.plot(g.snr_db,g.mean_signed_ms,'-o',ms=3,color=color,label=f'{d/32:g} ms')
        ax.axhline(0,color='gray',lw=.7);ax.set_title(f'{kind}, both channels');ax.set_xlim(21,-21);ax.set_xlabel('Added-noise SNR (dB)');ax.set_ylabel('Mean signed estimate (ms)');ax.grid(alpha=.2)
    axs[1].legend(title='True signed delay',fontsize=8);fig.suptitle('Signed delay and absolute delay are different outcomes')
    savefig(fig,'04_signed_estimates')
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,kind in zip(axs,KINDS):
        g=contr[(contr.noise==kind)&(contr['mode']=='both')&(contr.snr_db==-20)&(contr.delay_samples>=0)].sort_values('delay_samples')
        x=np.arange(len(g));up=g.fraction_increase_gt1ms;down=g.fraction_decrease_gt1ms;near=g.fraction_within1ms
        ax.bar(x,up,color='#d55e00',label='Increase >1 ms');ax.bar(x,near,bottom=up,color='#cccccc',label='Within +/-1 ms');ax.bar(x,down,bottom=up+near,color='#0072b2',label='Decrease >1 ms')
        ax.set_xticks(x,labels=g.delay_samples/32);ax.set_xlabel('True delay (ms)');ax.set_ylabel('Fraction of constructed signal pairs');ax.set_title(f'{kind}: 20 to -20 dB');ax.set_ylim(0,1)
    axs[1].legend(fontsize=8);fig.suptitle('Paired magnitude change for the same template and noise realization\nThreshold of 1 ms is descriptive, not a statistical significance threshold')
    savefig(fig,'05_increase_decrease_fractions')

def report(s,contr,df,null):
    bas=df[df.noise=='none'];bmax=bas.rmse_ms.max()
    table=s[(s['mode']=='both')&s.snr_db.isin([20,0,-10,-20])&(s.delay_samples.isin([0,80,160,240,320,400]))][['noise','delay_samples','snr_db','mean_abs_ms','mean_signed_ms','mean_rmse_ms','conditional_mc_se_ms']].copy()
    table.insert(1,'true_delay_ms',table.pop('delay_samples')/32)
    table=table.sort_values(['noise','true_delay_ms','snr_db'],ascending=[True,True,False]);table.to_csv(ROOT/'results/key_table.csv',index=False)
    nullsum=null.groupby('noise').agg(n=('mean_abs_ms','size'),mean_abs_ms=('mean_abs_ms','mean'),mean_signed_ms=('mean_signed_ms','mean'),record_sd_abs_ms=('mean_abs_ms','std')).reset_index();nullsum.to_csv(ROOT/'results/noise_only_summary.csv',index=False)
    resulttext='\n'.join(f"{row.noise}：平均绝对估计值 {row.mean_abs_ms:.3f} ms，有符号均值 {row.mean_signed_ms:.3f} ms。" for row in nullsum.itertuples())
    endpoint=table[table.snr_db==-20]
    meta=pd.read_csv(ROOT/'provenance/template_manifest.csv')
    outline=f'''NeuroTD：新增噪声SNR与估计延迟大小——扩展实验
完成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}

本轮完成253,440组加噪信号对、288组无新增噪声基准、400组独立噪声对照。使用36个固定模板，来自同一名受试者的3个会话、9个无伪迹且正确试次。原项目、E、原窗口参数均未修改。

研究问题
固定真实延迟和同一段波形，逐渐降低新增噪声SNR，估计的平均绝对延迟会变大还是变小？同时检查RMSE，避免将数值变小误认为更准确。

实验设计
每会话选择无伪迹且正确试次列表的首个、中间和最后一个试次，每个试次分别取四个电极束的第一通道，共36段1秒模板。片段统一为试次内1–2秒；不同会话的阶段标签并未据此认定一致。生成延迟为-10、-2.5、0、2.5、5、7.5、10、12.5 ms。
SNR为20、15、10、5、0、-2.5、-5、-7.5、-10、-15、-20 dB；白噪声或AR(1)噪声（rho=0.95）；仅y加噪或x、y分别加独立噪声；每条件20次噪声实现。相同标准化噪声跨SNR配对，减小不同随机噪声造成的比较波动。
SNR=10log10(固定模板均方功率/新增噪声均方功率)。固定模板本来来自真实记录，所以这不是原始脑区的生理SNR。未引入幅度增益差异，也未改变信号共同结构。
估计器W=1001、hop=500、矩形窗、搜索±500采样点即±15.625 ms，采样率32000 Hz。目标函数、符号和候选搜索与原函数相同；只使用先前已验证的线性FFT计算交叉项。基准和生产结果累计132次整段逐窗核对，必须与原函数完全一致后才输出完成状态。

结果如何阅读
“延迟大小”指先对每个窗口估计值取绝对值，再在记录内平均；不是先求有符号平均再取绝对值。每个汇总点平均36个模板、每模板20次噪声实现，共720个构造信号对。
无新增噪声基准最大RMSE：{bmax:.6g} ms。所有基准均保留，不因结果不好而删模板。

强噪声端点（-20 dB，双通道独立加噪）：
{endpoint.to_string(index=False,float_format=lambda x:f'{x:.3f}')}

没有共同模板、只有两路独立噪声的对照（不存在真实事件延迟，不能计算“真实延迟误差”）：
{resulttext}

解释原则
1. 应检查每条固定真实延迟曲线，不能混合不同真值做一个总体SNR—延迟相关系数。
2. 正的有符号估计和负的有符号估计可能相互抵消，但它们的绝对值不会抵消。噪声使估计变分散时，有符号均值接近零并不意味着平均绝对延迟接近零。
3. 低SNR下的估计受有限搜索范围、模板、噪声颜色和受噪通道数影响。噪声对照的均值是本设置的经验结果，不是理论上通用的噪声延迟值。不能外推到任意搜索范围。
4. 本轮不测量真实海马SNR，也不检验海马—其他脑区差异是否经SNR调整后仍存在。不得将加噪实验写成已经排除海马噪声混杂。

不确定性与比较
图中的阴影为36个固定模板均值的四分位范围，不是95%置信区间。conditional_mc_se_ms只描述在这36段固定模板下重复生成独立新增噪声的Monte Carlo标准误；不包含受试者间不确定性。窗口没有作为独立生物样本。
paired_contrasts.csv保存相同模板和噪声实现相对于20 dB的变化；adjacent_snr_changes.csv保存每两个相邻SNR水平的变化。增加/减少超过1 ms的比例采用运行前固定的描述性阈值，不是显著性检验。

图表
图01：不同真实延迟下，平均估计绝对延迟随SNR的曲线；黑点线为独立噪声对照的均值，仅在双通道面板展示。
图02：平均绝对估计值减真实绝对延迟。红色为幅度偏大，蓝色为偏小；白色不代表每个估计都准确。
图03：真实延迟RMSE。用于确认幅度下降是否同时伴随准确性恶化。
图04：有符号平均延迟，包含负延迟条件；与图01共同说明取绝对值的影响。
图05：从20 dB降到-20 dB时，每条配对记录增加、减少或变化不超过1 ms的比例。

范围与复现
本轮是原范围内一名受试者的固定模板实验，不是36名受试者，也不是独立验证的生物学总体。模板来自9个试次，同一试次的多个通道并非生物学独立重复。
plan.json保存运行前方案。provenance/保存原始文件校验、模板身份和代码哈希；checkpoints/保存所有逐窗口有符号估计；results/保存完整结果与验证表。
运行命令：neurotd312环境中依次运行extend_experiment.py prepare、extend_experiment.py run。已有模板批次可恢复；改变方案时必须使用新目录，避免混用检查点。
'''
    (ROOT/'结果说明.txt').write_text(outline)
    def pic(name):
        b=base64.b64encode((ROOT/f'figures/{name}.png').read_bytes()).decode();return f'<img src="data:image/png;base64,{b}">'
    pieces=['<h1>SNR降低，估计延迟一定变大吗？</h1><p class="lead">NeuroTD 加噪扩展实验 · 36个固定模板 · 253,440组加噪信号对</p>',
            '<p class="note">此处SNR指人为新增噪声的强度。它不能替代对海马原始记录SNR的测量。</p>']
    for heading,name in [('一、关系曲线','01_snr_delay_curves'),('二、何时高估，何时低估','02_inflation_deflation_map'),('三、数值变化与估计误差','03_delay_error'),('四、有符号延迟','04_signed_estimates'),('五、逐次配对变化','05_increase_decrease_fractions')]:
        pieces.extend([f'<h2>{heading}</h2>',pic(name)])
    pieces.extend(['<h2>关键数值表：双通道独立加噪</h2>',table.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0),'<h2>方法、解释与适用边界</h2><div class="full">'+html.escape(outline).replace('\n','<br>')+'</div>'])
    htmlpage='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>NeuroTD SNR扩展实验</title><style>body{max-width:1150px;margin:40px auto;padding:0 24px;color:#24364b;font-family:-apple-system,"PingFang SC",sans-serif;line-height:1.7}h1{font-size:32px}h2{padding-top:25px;border-bottom:2px solid #c9dbe7}.lead{color:#57728a}.note{background:#fff6e8;border-left:4px solid #dd9944;padding:16px}img{max-width:100%}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px;text-align:right;border-bottom:1px solid #ddd}th{background:#e6eef5}tr:nth-child(even){background:#f8fafb}.full{font-size:14px;overflow-wrap:anywhere}</style><body>'+''.join(pieces)+'</body></html>'
    (ROOT/'扩展实验报告.html').write_text(htmlpage)

def run():
    start=time.time();templates=dict(np.load(ROOT/'checkpoints/templates.npz'));done=0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_template,(i,name,a)) for i,(name,a) in enumerate(templates.items())]
        for f in as_completed(futures):
            name,n,secs=f.result();done+=1
            progress('simulation',templates_complete=done,total_templates=len(templates),template=name,records=n,template_seconds=round(secs,1),elapsed_seconds=round(time.time()-start))
    df=pd.concat([pd.read_csv(ROOT/f'checkpoints/{name}.csv') for name in templates],ignore_index=True)
    assert len(df)==253728 and not df.duplicated(['template','delay_samples','noise','mode','snr_db','replicate']).any()
    assert np.isfinite(df[['mean_abs_ms','mean_signed_ms','rmse_ms']]).all().all()
    assert (df.loc[df.noise!='none','actual_snr_y_db']-df.loc[df.noise!='none','snr_db']).abs().max()<1e-10
    df.to_csv(ROOT/'results/all_records.csv',index=False)
    null=null_control();validate_production(templates)
    s,contr,t=summarize(df);plots(s,contr,t,null);report(s,contr,df,null)
    hashes=json.loads((ROOT/'provenance/source_hashes.json').read_text())
    assert old.sha(old.SOURCE)==hashes[str(old.SOURCE)]
    assert old.sha(PREV/'run_analysis.py')==hashes[str(PREV/'run_analysis.py')]
    checks=dict(noisy_pairs=int((df.noise!='none').sum()),baseline_pairs=int((df.noise=='none').sum()),noise_only_pairs=len(null),templates=len(templates),baseline_max_rmse_ms=float(df[df.noise=='none'].rmse_ms.max()),validation_records=24+108,all_checked_shifts_match=True,original_code_unchanged=True,elapsed_run_seconds=round(time.time()-start,2))
    (ROOT/'results/completion_checks.json').write_text(json.dumps(checks,indent=2))
    progress('complete',**checks)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run()
    else:raise ValueError('Use prepare or run')
