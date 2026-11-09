"""Standalone, checkpointed Reviewer 2.7 experiment. Does not mutate NeuroTD."""
from revision_runtime import resolve_path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import ast, hashlib, json, time, urllib.request, itertools, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
import pandas as pd
from scipy import signal, stats, fft
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
RAW=Path(resolve_path('raw', 'sub-01'))
AUDIT=Path(resolve_path('audit_v3', ''))
DELAY=Path(resolve_path('audit_v1', 'checkpoints/windows_all_trials.npz'))
SOURCE=Path(resolve_path('repo', 'src/neurotd/time_varying_shift.py'))
FS=32000; W=1001; H=500; S=500; NTRIAL=256000
ASSETS=['6f367ffb-c6a9-468f-a974-394eabae6060','109195d4-4594-4c64-97bd-94d1dd51e575','4ff056eb-a9b7-465b-a9d2-7e10c7be17d3']
SESSIONS=['ses-01','ses-02','ses-03']
for d in ['provenance','results','figures','checkpoints']:(ROOT/d).mkdir(exist_ok=True)

def progress(stage,**kwargs):
    obj={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,**kwargs}
    (ROOT/'progress.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    print(json.dumps(obj,ensure_ascii=False),flush=True)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def provenance():
    rows=[]
    url='https://api.dandiarchive.org/api/dandisets/000574/versions/0.250610.0938/assets/?path=sub-01%2F&page_size=100'
    listing=json.load(urllib.request.urlopen(url,timeout=60))
    (ROOT/'provenance/archive_listing.json').write_text(json.dumps(listing,indent=2))
    for sess,asset in zip(SESSIONS,ASSETS):
        meta=json.load(urllib.request.urlopen(f'https://api.dandiarchive.org/api/assets/{asset}/',timeout=60))
        (ROOT/f'provenance/{sess}_official.json').write_text(json.dumps(meta,indent=2))
        path=RAW/f'sub-01_{sess}.nwb'
        expected=meta['digest']['dandi:sha2-256']
        # Download only if missing; never overwrite an unverified existing raw file.
        if not path.exists():
            RAW.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.part')
            urllib.request.urlretrieve(f'https://api.dandiarchive.org/api/assets/{asset}/download/',tmp)
            assert sha(tmp)==expected
            tmp.rename(path)
        actual=sha(path)
        assert actual==expected,(sess,'SHA mismatch')
        assert path.stat().st_size==meta['contentSize']
        rows.append(dict(session=sess,path=str(path),bytes=path.stat().st_size,sha256=actual,official_sha256=expected,verified=True,asset_id=asset))
        progress('raw_file_verified',session=sess)
    pd.DataFrame(rows).to_csv(ROOT/'provenance/raw_files.csv',index=False)
    pd.DataFrame([{'path':str(p),'sha256':sha(p)} for p in [SOURCE,DELAY,ROOT/'analysis_plan.json',Path(__file__)] ]).to_csv(ROOT/'provenance/analysis_inputs.csv',index=False)

@njit(cache=True,nogil=True)
def estimate_prefix(x,y):
    n=len(x); starts=np.arange(S,n-W-S+1,H)
    best=np.zeros(len(starts),np.int16); costs=np.full(len(starts),np.inf)
    prefix=np.zeros(n+1)
    for tau in range(-S,S+1):
        prefix[0]=0.
        for i in range(n):
            j=i-tau
            v=0.
            if 0<=j<n:
                d=x[i]-y[j];v=d*d
            prefix[i+1]=prefix[i]+v
        for k in range(len(starts)):
            a=starts[k];m=(prefix[a+W]-prefix[a])/W
            if m<costs[k]:
                costs[k]=m;best[k]=-tau
    return starts+S,best,costs

def estimate(x,y):
    """Same unweighted MSE search, using linear FFT convolution for cross-products.

    No circular correlation, no phase normalization, no baseline FFTShift method.
    Energy terms include every candidate's full comparison segment.
    """
    starts=np.arange(S,len(x)-W-S+1,H)
    xs=np.stack([x[a:a+W] for a in starts])
    ys=np.stack([y[a-S:a+W+S] for a in starts])
    nfft=fft.next_fast_len(ys.shape[1]+W-1)
    prod=fft.irfft(fft.rfft(ys,nfft,axis=1)*fft.rfft(xs[:,::-1],nfft,axis=1),nfft,axis=1)[:,W-1:W+2*S]
    ey=np.pad(np.cumsum(ys*ys,axis=1),((0,0),(1,0)))
    cost=(np.sum(xs*xs,axis=1)[:,None]+ey[:,W:]-ey[:,:-W]-2*prod)/W
    # Original tau ascending means delay descending; preserve tie direction.
    choice=2*S-np.argmin(cost[:,::-1],axis=1)
    return starts+S,(choice-S).astype(np.int16),cost[np.arange(len(starts)),choice]

def original():
    tree=ast.parse(SOURCE.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_windowed_mse_with_hop')
    scope={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),scope)
    return scope[node.name]

def read_trial(f,t):
    g=f['intervals/trials']; idx=g['timeseries_index'][:].astype(int)
    refs=g['timeseries'][0 if t==0 else idx[t-1]:idx[t]]
    row=next(r for r in refs if f[r['timeseries']].name=='/acquisition/ecephys.lfp')
    start,count=int(row['idx_start']),int(row['count'])
    assert start==t*NTRIAL and count==NTRIAL
    ts=f['acquisition/ecephys.lfp/timestamps']
    assert abs(float(ts[start])-float(g['start_time'][t]))<1e-5
    assert abs(float(ts[start+1]-ts[start])-1/FS)<1e-8
    return np.asarray(f['acquisition/ecephys.lfp/data'][start:start+count,:])

def metadata_templates():
    trials=[]; templates={}
    for sess in SESSIONS:
        with h5py.File(RAW/f'sub-01_{sess}.nwb','r') as f:
            g=f['intervals/trials']
            for t in range(50):
                trials.append(dict(session=sess,trial_pos=t,trial_id=int(g['id'][t]),artifact=bool(g['artifact'][t]),correct=bool(g['correct'][t]),set_size=int(g['set_size'][t]),duration_s=float(g['stop_time'][t]-g['start_time'][t])))
            t=int(np.flatnonzero(~g['artifact'][:].astype(bool)&g['correct'][:].astype(bool))[0])
            a=read_trial(f,t)
            for ch,bundle in zip([0,8,16,24],['AHL','AL','ECL','PHR']):
                name=f'{sess}_{bundle}'
                # Keep padding so known delayed template has no circular wrap or synthetic edge.
                templates[name]=a[FS-1000:2*FS+1000,ch].copy()
    pd.DataFrame(trials).to_csv(ROOT/'results/trial_metadata.csv',index=False)
    np.savez_compressed(ROOT/'checkpoints/templates.npz',**templates)
    return pd.DataFrame(trials),templates

def make_pair(template,delay):
    return template[1000:1000+FS].copy(),template[1000-delay:1000-delay+FS].copy()

def validate(templates):
    fn=original(); rows=[]; pairs=pd.read_csv(AUDIT/'electrodes/pair_classification.csv')
    z=np.load(DELAY)
    tests=[]
    for name in [list(templates)[0],list(templates)[5],list(templates)[-1]]:
        for d in [0,80,320]:
            x,y=make_pair(templates[name],d)
            tests.append((name+f'_delay{d}',x,y,None))
    rng=np.random.default_rng(270811)
    x,y=make_pair(next(iter(templates.values())),80)
    tests.append(('noisy_both',x+rng.normal(0,np.std(x),len(x)),y+rng.normal(0,np.std(y),len(y)),None))
    for si,sess in enumerate(SESSIONS):
        with h5py.File(RAW/f'sub-01_{sess}.nwb') as f:a=read_trial(f,0)
        for pi in [0,1]:
            p=pairs.iloc[pi]
            tests.append((f'{sess}_fulltrial_pair{pi}',a[:,int(p.channel_i)].copy(),a[:,int(p.channel_j)].copy(),z['shifts_samples'][si*50,pi]))
    estimate(np.zeros(3000),np.zeros(3000))
    for name,x,y,old in tests:
        t=time.time(); c,s,_=estimate(x,y); fast=time.time()-t
        t=time.time();co,so=fn(x,y,W,H);slow=time.time()-t
        mismatch=int(np.count_nonzero(s!=so))
        previous=None if old is None else int(np.count_nonzero(s!=old))
        rows.append(dict(test=name,n_windows=len(s),mismatch_original=mismatch,mismatch_checkpoint=previous,fast_seconds=fast,original_seconds=slow))
        pd.DataFrame(rows).to_csv(ROOT/'results/estimator_validation.csv',index=False)
        progress('estimator_validation',test=name,mismatches=mismatch,fast_s=round(fast,3),original_s=round(slow,3))
        assert np.array_equal(c,co) and mismatch==0
        assert previous in [None,0]
    return rows

def real_quality(trials):
    out=ROOT/'results/channel_trial_quality.csv'
    if out.exists(): return pd.read_csv(out)
    allrows=[]
    for sess in SESSIONS:
        cp=ROOT/f'checkpoints/quality_{sess}.csv'
        if cp.exists():allrows.extend(pd.read_csv(cp).to_dict('records'));continue
        rows=[]
        with h5py.File(RAW/f'sub-01_{sess}.nwb') as f:
            for t in range(50):
                a=read_trial(f,t)
                assert np.isfinite(a).all()
                freq,p=signal.welch(a,fs=FS,window='hann',nperseg=FS,noverlap=FS//2,axis=0,detrend='constant')
                rms=np.sqrt(np.mean(a*a,axis=0))
                hf=p[(freq>=1000)&(freq<=10000)].sum(0)/p[(freq>=1)&(freq<=10000)].sum(0)
                line={}
                for hz in [50,60]:
                    delta=np.abs(freq-hz)
                    line[hz]=10*np.log10(p[delta<=1].mean(0)/p[(delta>=2)&(delta<=5)].mean(0))
                for ch in range(32):
                    rows.append(dict(session=sess,trial_pos=t,channel=ch,bundle=['AHL','AL','ECL','PHR'][ch//8],rms=rms[ch],hf_fraction=hf[ch],line50_db=line[50][ch],line60_db=line[60][ch]))
                if (t+1)%10==0:progress('real_signal_quality',session=sess,trials_complete=t+1,trials_total=50)
        pd.DataFrame(rows).to_csv(cp,index=False);allrows.extend(rows)
    df=pd.DataFrame(allrows);df.to_csv(out,index=False);return df

def observational(trials,q):
    z=np.load(DELAY);pairs=pd.read_csv(AUDIT/'electrodes/pair_classification.csv')
    # Decompress each NPZ member once; do not reread the complete archive for every pair.
    all_shifts=z['shifts_samples'];all_sessions=z['session'];all_trials=z['trial_pos']
    q=q.set_index(['session','trial_pos','channel']); rows=[]
    for ri in range(len(all_sessions)):
        sess=str(all_sessions[ri]);t=int(all_trials[ri])
        for pi,p in pairs.iterrows():
            a=q.loc[(sess,t,int(p.channel_i))];b=q.loc[(sess,t,int(p.channel_j))]
            s=all_shifts[ri,pi].astype(float); ab=np.abs(s)/32
            rows.append(dict(session=sess,trial_pos=t,pair_index=pi,legacy_region=p.region,same_bundle=p.same_bundle,bundle_i=p.bundle_i,bundle_j=p.bundle_j,
                mean_abs_ms=ab.mean(),median_abs_ms=np.median(ab),mean_signed_ms=s.mean()/32,mean_nonzero_ms=ab[s!=0].mean() if np.any(s!=0) else np.nan,
                zero_fraction=np.mean(s==0),boundary_fraction=np.mean(np.abs(s)==S),
                line50_db=max(a.line50_db,b.line50_db),line60_db=max(a.line60_db,b.line60_db),hf_fraction=(a.hf_fraction+b.hf_fraction)/2,
                rms_ratio_log2=abs(np.log2(a.rms/b.rms))))
    df=pd.DataFrame(rows).merge(trials,on=['session','trial_pos'],validate='many_to_one')
    df.to_csv(ROOT/'results/pair_trial_quality_delay.csv',index=False)
    corr=[]
    for subset,sel in [('clean',~df.artifact),('all',np.ones(len(df),bool)),('clean_correct',~df.artifact&df.correct)]:
        for keys,g in df.loc[sel].groupby(['session','set_size','pair_index']):
            if len(g)<10:continue
            for metric in ['line50_db','line60_db','hf_fraction','rms_ratio_log2']:
                for outcome in ['mean_abs_ms','mean_nonzero_ms']:
                    v=g[[metric,outcome]].dropna()
                    rho=stats.spearmanr(v[metric],v[outcome]).statistic if len(v)>=10 and v[metric].nunique()>1 and v[outcome].nunique()>1 else np.nan
                    corr.append(dict(subset=subset,session=keys[0],set_size=keys[1],pair_index=keys[2],same_bundle=bool(g.same_bundle.iloc[0]),metric=metric,outcome=outcome,n_trials=len(v),spearman_rho=rho))
    c=pd.DataFrame(corr);c.to_csv(ROOT/'results/within_pair_correlations.csv',index=False)
    summaries=[]
    for keys,g in c.groupby(['subset','metric','outcome','same_bundle']):
        r=g.spearman_rho.dropna();summaries.append(dict(zip(['subset','metric','outcome','same_bundle'],keys),n_groups=len(g),n_nonconstant=len(r),median_rho=r.median(),q25=r.quantile(.25),q75=r.quantile(.75),fraction_positive=(r>0).mean()))
    pd.DataFrame(summaries).to_csv(ROOT/'results/correlation_summary.csv',index=False)
    df.loc[~df.artifact].groupby(['session','legacy_region']).agg(n_pair_trials=('mean_abs_ms','size'),mean_abs_ms=('mean_abs_ms','mean'),line60_db=('line60_db','median'),rms_ratio_log2=('rms_ratio_log2','median'),zero_fraction=('zero_fraction','mean')).reset_index().to_csv(ROOT/'results/legacy_region_descriptive_ONLY.csv',index=False)
    return df,c

def noise_array(rng,kind,n):
    a=rng.standard_normal(n+1000)
    if kind=='ar1':a=signal.lfilter([1],[1,-.95],a)
    a=a[1000:];a-=a.mean();a/=np.sqrt(np.mean(a*a))
    return a

def outcome(s,delay):
    e=(s.astype(float)-delay)/32
    return dict(n_windows=len(s),mean_abs_ms=np.mean(np.abs(s.astype(float)))/32,signed_bias_ms=e.mean(),rmse_ms=np.sqrt(np.mean(e*e)),within2_fraction=np.mean(np.abs(s-delay)<=2),zero_fraction=np.mean(s==0),boundary_fraction=np.mean(np.abs(s)==500))

def injection_template(args):
    ti,name,template=args;cp=ROOT/f'checkpoints/injection_{name}.csv'
    if cp.exists():return name,len(pd.read_csv(cp)),0.
    st=time.time();rows=[];shifts=[]
    for delay in [0,80,320]:
        x,y=make_pair(template,delay)
        px=np.mean(x*x);py=np.mean(y*y)
        _,s,_=estimate(x,y)
        rows.append(dict(template=name,delay_samples=delay,noise='none',mode='none',snr_db=np.inf,replicate=-1,seed=-1,actual_snr_x_db=np.inf,actual_snr_y_db=np.inf,**outcome(s,delay)));shifts.append(s)
        for ni,kind in enumerate(['white','ar1']):
            for mi,mode in enumerate(['y_only','both']):
                for rep in range(20):
                    seed=270811+ti*100000+delay*100+ni*1000+mi*100+rep
                    rng=np.random.default_rng(seed)
                    nx=noise_array(rng,kind,len(x));ny=noise_array(rng,kind,len(y))
                    for snr in [20,10,0,-10]:
                        ex=nx*np.sqrt(px/(10**(snr/10))) if mode=='both' else np.zeros_like(x)
                        ey=ny*np.sqrt(py/(10**(snr/10)))
                        _,s,_=estimate(x+ex,y+ey)
                        rows.append(dict(template=name,delay_samples=delay,noise=kind,mode=mode,snr_db=snr,replicate=rep,seed=seed,actual_snr_x_db=10*np.log10(px/np.mean(ex*ex)) if mode=='both' else np.inf,actual_snr_y_db=10*np.log10(py/np.mean(ey*ey)),**outcome(s,delay)));shifts.append(s)
    pd.DataFrame(rows).to_csv(cp,index=False)
    np.savez_compressed(ROOT/f'checkpoints/injection_shifts_{name}.npz',shifts_samples=np.stack(shifts))
    return name,len(rows),time.time()-st

def injections(templates):
    start=time.time();done=0
    args=[(i,n,a) for i,(n,a) in enumerate(templates.items())]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(injection_template,a) for a in args]
        # Poll completion without losing logs for long-running first tasks.
        while futures:
            for f in list(futures):
                if f.done():
                    name,n,secs=f.result();done+=1;futures.remove(f)
                    progress('noise_injection',template=name,templates_complete=done,templates_total=12,records=n,template_seconds=round(secs,2),elapsed_seconds=round(time.time()-start))
            if futures:time.sleep(1)
    df=pd.concat([pd.read_csv(ROOT/f'checkpoints/injection_{name}.csv') for name in templates],ignore_index=True)
    assert len(df)==11556 and df[['mean_abs_ms','rmse_ms','signed_bias_ms']].notna().all().all()
    assert (df.loc[df.noise!='none','actual_snr_y_db']-df.loc[df.noise!='none','snr_db']).abs().max()<1e-10
    df.to_csv(ROOT/'results/injection_records.csv',index=False)
    # Equal template weighting; SD describes independent added-noise realizations, not subjects.
    group=['delay_samples','noise','mode','snr_db']
    df.groupby(group,dropna=False).agg(records=('rmse_ms','size'),mean_rmse_ms=('rmse_ms','mean'),median_rmse_ms=('rmse_ms','median'),mean_abs_ms=('mean_abs_ms','mean'),mean_signed_bias_ms=('signed_bias_ms','mean'),boundary_fraction=('boundary_fraction','mean'),zero_fraction=('zero_fraction','mean')).reset_index().to_csv(ROOT/'results/injection_summary.csv',index=False)
    df.groupby(['template']+group).agg(rmse_mean_ms=('rmse_ms','mean'),rmse_sd_ms=('rmse_ms','std'),abs_delay_mean_ms=('mean_abs_ms','mean'),abs_delay_sd_ms=('mean_abs_ms','std')).reset_index().to_csv(ROOT/'results/injection_template_summary.csv',index=False)
    return df

def plots(real,c,inj):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    cols=['#0072B2','#D55E00','#009E73','#CC79A7']
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    metrics=['line50_db','line60_db','hf_fraction','rms_ratio_log2']
    labels=['50 Hz prominence','60 Hz prominence','High-frequency fraction','RMS imbalance']
    for ax,same in zip(axs,[True,False]):
        v=c[(c.subset=='clean')&(c.outcome=='mean_abs_ms')&(c.same_bundle==same)]
        data=[v.loc[v.metric==m,'spearman_rho'].dropna() for m in metrics]
        ax.boxplot(data,tick_labels=labels,showfliers=False);ax.axhline(0,color='gray',lw=.8)
        ax.set_ylim(-1,1);ax.tick_params(axis='x',rotation=23);ax.set_ylabel('Within-pair Spearman correlation')
        ax.set_title('Same bundle' if same else 'Different bundles (legacy pairs)')
    fig.suptitle('Signal-quality proxies vs mean absolute delay\nArtifact-free trials; zeros retained; descriptive correlations, one participant')
    fig.savefig(ROOT/'figures/01_real_quality_correlations.png',dpi=200);fig.savefig(ROOT/'figures/01_real_quality_correlations.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    noisy=inj[inj.noise!='none']
    for j,d in enumerate([0,80,320]):
        for k,(noise,mode) in enumerate(itertools.product(['white','ar1'],['y_only','both'])):
            v=noisy[(noisy.delay_samples==d)&(noisy.noise==noise)&(noisy['mode']==mode)]
            g=v.groupby('snr_db')[['mean_abs_ms','rmse_ms']].mean().sort_index(ascending=False)
            label=f'{noise}, {mode.replace("_"," ")}'
            for ax,metric in zip(axs[:,j],['mean_abs_ms','rmse_ms']):ax.plot(g.index,g[metric],'-o',color=cols[k],ms=4,label=label)
        axs[0,j].axhline(d/32,color='black',ls='--',lw=1,label='True |delay|')
        axs[0,j].set_title(f'True delay = {d/32:g} ms')
        for ax in axs[:,j]:ax.set_xlim(22,-12);ax.set_xticks([20,10,0,-10]);ax.set_xlabel('Added-noise SNR (dB)');ax.grid(alpha=.2)
    axs[0,0].set_ylabel('Mean estimated |delay| (ms)');axs[1,0].set_ylabel('Delay RMSE (ms)')
    axs[0,2].legend(fontsize=8);fig.suptitle('Controlled noise experiment: fixed real-data templates, known shifts\n12 templates × 20 noise realizations per setting; W=1001, H=500; mean curves')
    fig.savefig(ROOT/'figures/02_controlled_noise.png',dpi=200);fig.savefig(ROOT/'figures/02_controlled_noise.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for j,noise in enumerate(['white','ar1']):
        v=noisy[(noisy.delay_samples==0)&(noisy.noise==noise)&(noisy['mode']=='both')]
        for name,g in v.groupby('template'):
            a=g.groupby('snr_db').mean(numeric_only=True).sort_index(ascending=False)
            axs[j].plot(a.index,a.mean_abs_ms,alpha=.6,lw=1,label=name)
        axs[j].set_title(noise+' noise, true delay = 0');axs[j].set_xlim(22,-12);axs[j].set_xlabel('Added-noise SNR (dB)');axs[j].set_ylabel('Mean estimated |delay| (ms)');axs[j].grid(alpha=.2)
    axs[1].legend(fontsize=6,ncol=2)
    fig.suptitle('Template sensitivity: each line is one recorded template, not a subject')
    fig.savefig(ROOT/'figures/03_template_sensitivity.png',dpi=200);fig.savefig(ROOT/'figures/03_template_sensitivity.pdf');plt.close(fig)

def validate_completed_injections(templates,inj):
    """Check saved production output against the exact original function."""
    fn=original(); checks=[]
    for ti,(name,template) in enumerate(templates.items()):
        local=pd.read_csv(ROOT/f'checkpoints/injection_{name}.csv')
        saved=np.load(ROOT/f'checkpoints/injection_shifts_{name}.npz')['shifts_samples']
        selected=local.index[local.noise=='none'].tolist()
        # Fixed row choices span both noise types, both modes and low SNR.
        selected += [160,320,640,960]
        for row in selected:
            r=local.iloc[row];x,y=make_pair(template,int(r.delay_samples))
            if r.noise!='none':
                rng=np.random.default_rng(int(r.seed))
                nx=noise_array(rng,r.noise,len(x));ny=noise_array(rng,r.noise,len(y))
                ex=nx*np.sqrt(np.mean(x*x)/(10**(r.snr_db/10))) if r['mode']=='both' else np.zeros_like(x)
                ey=ny*np.sqrt(np.mean(y*y)/(10**(r.snr_db/10)))
                x=x+ex;y=y+ey
            _,expected=fn(x,y,W,H)
            mismatch=int(np.count_nonzero(expected!=saved[row]))
            checks.append(dict(template=name,row=row,noise=r.noise,mode=r['mode'],snr_db=r.snr_db,delay_samples=r.delay_samples,n_windows=len(expected),mismatch=mismatch))
            assert mismatch==0,(name,row,mismatch)
        progress('production_validation',template=name,checks_complete=len(checks))
    pd.DataFrame(checks).to_csv(ROOT/'results/production_original_validation.csv',index=False)

def main():
    stage=sys.argv[1] if len(sys.argv)>1 else 'all'
    if stage in ['prepare','all']:
        provenance();trials,templates=metadata_templates();v=validate(templates)
        short=[r['fast_seconds'] for r in v if r['n_windows']<100]
        progress('baseline_passed',median_one_second_seconds=float(np.median(short)),estimated_injection_serial_seconds=float(np.median(short)*11556))
    if stage in ['run','all']:
        trials=pd.read_csv(ROOT/'results/trial_metadata.csv');templates=dict(np.load(ROOT/'checkpoints/templates.npz'))
        q=real_quality(trials);real,c=observational(trials,q)
        inj=injections(templates);validate_completed_injections(templates,inj);plots(real,c,inj)
        assert sha(SOURCE)==pd.read_csv(ROOT/'provenance/analysis_inputs.csv').iloc[0].sha256
        progress('complete',real_pair_trials=len(real),noise_records=len(inj),original_code_unchanged=True)

if __name__=='__main__':main()
