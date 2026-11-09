"""Reviewer 2.4.3: unchanged NeuroTD, fixed-window whole-curve null experiment.

This independent harness never writes in the original project or the 2.4.1 run.
Dense raw estimates are computed by the original function. Sparse hops reuse
identical window estimates; the original overlap-add equations are reproduced
and checked against direct original calls before the experiment can start.
"""
from __future__ import annotations
from revision_runtime import resolve_path
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PRIOR = Path(resolve_path('score_uncertainty', ''))
PROJECT = Path(resolve_path('repo', ''))
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(ROOT/'cache')
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT/'src'))
import numpy as np
from neurotd.time_varying_shift import compute_windowed_mse_with_hop as original_estimator
from neurotd.time_varying_shift import evaluate_alignment_quality_sliding as original_E
from neurotd.time_varying_shift import compute_shifts_sliding_window_gt

FS, W, HOPS = 5120, 513, [1, 128, 256, 513]
MASTER_SEED = 202610180243
SPLITS = [('calibration_null',199), ('test_null',200), ('test_dynamic',200)]
CELLS = [(m, s, 0.0) for m in ['discrete', 'continuous'] for s in [.02,.10,.20]] + [('continuous',.02,.8)]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)

def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text())

def source_hashes():
    return {str(p.relative_to(PROJECT)):digest(p) for p in sorted((PROJECT/'src').rglob('*.py'))}

def config():
    return dict(version=1,master_seed=MASTER_SEED,fs=FS,window=W,hops=HOPS,
        cells=[dict(model=m,noise_std=s,ar1_rho=rho) for m,s,rho in CELLS],
        splits=dict(SPLITS),constant_delay_samples=dict(discrete=12,continuous=0),
        primary=dict(discrete=dict(hop=256,stage='raw'),continuous=dict(hop=1,stage='fused')),
        weights=dict(discrete='rectangle',continuous='Hann'),
        statistic='T=mean((delay_ms-mean(delay_ms))**2), on native returned window centers',
        sensitivity='native-center T for all hops/stages; max(T) across eight variants calibrated as one family statistic',
        p_value='(1 + count(T_calibration >= T_test))/(199+1); reject p<=0.05; ties conservative',
        rng='PCG64(SeedSequence([202610180243,cell_id,split_id,replicate_id])); x then y noise',
        E='original sqrt(mean(r_k)); r_k=sqrt(MSE)/(sqrt(mean(x^2)*mean(y_aligned^2))+1e-16)',
        scope='preselected W=513 mechanism experiment; no data-driven W selection; not a real-data test',
        all_source_hashes=source_hashes(),
        harness_hashes={name:digest(ROOT/name) for name in ['simulate.py','report.py']},
        clean_signal_files={m:digest(PRIOR/f'clean_signals/{m}.npz') for m in ['discrete','continuous']})

def signals(model):
    with np.load(PRIOR/f'clean_signals/{model}.npz') as z:
        s={k:z[k].copy() for k in z.files}
    d0=12 if model=='discrete' else 0
    s['null_y']=np.roll(s['x'],d0)
    s['null_truth']=np.full(len(s['x']),float(d0))
    return s

def weights(w, fun):
    a = np.ones(w) if fun=='rectangle' else .5-.5*np.cos(2*np.pi*np.arange(w)/(w-1))
    return a/a.sum()

def reconstruct(n, centers, shifts, w, fun):
    # Same order and arithmetic as original compute_windowed_mse_with_hop.
    wt=weights(w,fun)
    delay=np.zeros(n,dtype=float)
    accum=np.zeros(n,dtype=float)
    for center,shift in zip(centers.astype(int),shifts):
        start=center-w//2
        end=start+w
        if 0<=start and end<=n:
            delay[start:end] += shift*wt
            accum[start:end] += wt
    valid=accum>0
    delay[valid] /= accum[valid]
    return delay,valid

def independent_E(x,y,centers,shifts,w):
    r=[]
    for c,d in zip(centers,shifts):
        lo,hi=int(c)-w//2,int(c)+w//2+1
        xx,yy=x[lo:hi],y[lo+int(d):hi+int(d)]
        assert len(xx)==w and len(yy)==w
        r.append(np.sqrt(np.mean((xx-yy)**2))/(np.sqrt(np.mean(xx**2)*np.mean(yy**2))+1e-16))
    r=np.asarray(r)
    return float(np.sqrt(np.mean(r))),r

def score(x,y,centers,shifts,w):
    q=np.round(shifts).astype(int)
    E,per=original_E(x,y,q,centers,w)
    audit,r=independent_E(x,y,centers,q,w)
    assert np.isfinite(E) and len(per)==len(centers)
    assert np.isclose(E,audit,rtol=1e-12,atol=1e-12)
    assert np.allclose(per,np.sqrt(r),rtol=1e-12,atol=1e-12)
    return float(E),float(abs(E-audit))

def truth_at(model,s,centers,dynamic):
    if not dynamic: return s['null_truth'][centers]
    if model=='continuous': return s['delays'][centers]*FS
    return compute_shifts_sliding_window_gt(centers,s['segment_centers'],s['segment_shifts'])

def noise(rng,n,sigma,rho):
    if rho==0: return rng.normal(0,sigma,n)
    # Stationary marginal SD=sigma, not innovation SD; independent x/y processes.
    z=rng.normal(0,sigma*np.sqrt(1-rho*rho),n)
    z[0]=rng.normal(0,sigma)
    for i in range(1,n): z[i]+=rho*z[i-1]
    return z

def noisy_pair(s,cell_id,split_id,rep):
    _,sigma,rho=CELLS[cell_id]
    seed=[MASTER_SEED,cell_id,split_id,rep]
    rng=np.random.default_rng(np.random.SeedSequence(seed))
    dynamic=SPLITS[split_id][0]=='test_dynamic'
    x=s['x']+noise(rng,len(s['x']),sigma,rho)
    y=s['y' if dynamic else 'null_y']+noise(rng,len(s['x']),sigma,rho)
    return x,y,seed,dynamic

def metrics(x,y,s,model,dynamic,centers,dense_raw):
    fun='rectangle' if model=='discrete' else 'Hann'
    rows=[]
    arrays={'centers_H1':centers,'raw_H1':dense_raw}
    # Common evaluation support is for an additional reconstruction diagnostic,
    # never used to silently interpolate or redefine original returned estimates.
    full_curves=[]
    common=np.zeros(len(x),dtype=bool)
    common[centers[0]:min(centers[::h][-1] for h in HOPS)+1]=True
    for h in HOPS:
        c,raw=centers[::h],dense_raw[::h]
        full,valid=reconstruct(len(x),c,raw,W,fun)
        common &= valid
        full_curves.append(full)
        gt=truth_at(model,s,c,dynamic)
        arrays[f'centers_H{h}']=c
        arrays[f'raw_H{h}']=raw
        arrays[f'fused_H{h}']=full[c]
        arrays[f'truth_H{h}']=gt
        for stage,delay in [('raw',raw),('fused',full[c])]:
            ms=delay*1000/FS
            err=(delay-gt)*1000/FS
            E,audit=score(x,y,c,delay,W)
            rows.append(dict(hop=h,stage=stage,M=len(c),overlap_fraction=max(0,1-h/W),
                T_ms2=float(np.var(ms)),rmse_ms=float(np.sqrt(np.mean(err**2))),
                bias_ms=float(np.mean(err)),range_ms=float(np.ptp(ms)),
                E=E,E_audit_abs_diff=audit,gt_variance_ms2=float(np.var(gt*1000/FS))))
    assert common.sum()>0
    arrays['common_reconstruction_indices']=np.flatnonzero(common)
    for h,full in zip(HOPS,full_curves):
        arrays[f'common_fused_H{h}']=full[common]
        for row in rows:
            if row['hop']==h and row['stage']=='fused':
                row['common_fused_variance_ms2']=float(np.var(full[common]*1000/FS))
    return rows,arrays

def unit_tests():
    rng=np.random.default_rng(73)
    x=rng.normal(size=257)
    tests=[]
    for fun in ['rectangle','Hann']:
        for noisy in [False,True]:
            y=np.roll(x,5)+(rng.normal(0,.1,len(x)) if noisy else 0)
            c,raw=original_estimator(x,y,33,1,fun,False)
            if not noisy: assert np.all(raw==5)
            for h in [1,8,16,33]:
                cc,rr=original_estimator(x,y,33,h,fun,False)
                cf,ff=original_estimator(x,y,33,h,fun,True)
                full,_=reconstruct(len(x),c[::h],raw[::h],33,fun)
                assert np.array_equal(cc,c[::h]) and np.array_equal(rr,raw[::h])
                assert np.array_equal(cf,cc) and np.array_equal(ff,full[cc])
                E,diff=score(x,y,cf,ff,33)
                tests.append(dict(weights=fun,noisy=noisy,hop=h,raw_exact=True,fused_exact=True,E_abs_diff=diff))
    # Constant preservation with positive covered weights, and conservative ties.
    assert np.var(np.repeat(12.,100))==0
    from report import monte_carlo_p
    assert monte_carlo_p(np.zeros(199),np.array([0.,1.])).tolist()==[1.,.005]
    return tests

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    old=read_json(PRIOR/'frozen_config.json')
    assert source_hashes()==old['full_source_hashes'],'Source differs from 2.4.1 baseline'
    conf=config()
    path=ROOT/'frozen_config.json'
    if path.exists(): assert read_json(path)==conf,'Protocol/source/input changed'
    else: write_json(path,conf)
    rows=[]
    # Re-evaluate saved, freshly reproduced 2.4.1 baselines without rerunning
    # their expensive delay search while that experiment is using the machine.
    for model in ['discrete','continuous']:
        s=signals(model)
        with (PROJECT/f'simulation_results_{model}.csv').open() as f: ref=list(csv.DictReader(f))
        with (PRIOR/'baseline/comparison.csv').open() as f: previous=list(csv.DictReader(f))
        for sigma in [0.,.02]:
            rng=np.random.default_rng(0 if model=='discrete' else 2025)
            x=s['x']+rng.normal(0,sigma,len(s['x']))
            y=s['y']+rng.normal(0,sigma,len(s['y']))
            for w in [33,65,129,257,513,1025]:
                p=PRIOR/f'baseline/{model}_noise{sigma:.2f}_W{w}.npz'
                with np.load(p) as z:
                    c,d=z['centers'],z['shifts']
                    E,diff=score(x,y,c,d,w)
                    gt=truth_at(model,s,c,True)
                    err=float(np.sqrt(np.mean((d-gt)**2))*1000/FS)
                prev=next(r for r in previous if r['model']==model and float(r['noise_std'])==sigma and int(r['window_size'])==w)
                assert np.isclose(E,float(prev['reference_E']),rtol=1e-8,atol=1e-10)
                assert np.isclose(err,float(prev['reference_rmse_ms']),rtol=1e-8,atol=1e-10)
                # The original CSV is independently checked, not only the prior audit table.
                rr=next(r for r in ref if int(r['window_size'])==w and float(r['noise_std'])==sigma)
                assert np.isclose(E,float(rr['quality_rmse_neurotd']),rtol=1e-8,atol=1e-10)
                assert np.isclose(err,float(rr['rmse_neurotd'])*1000/FS,rtol=1e-8,atol=1e-10)
                rows.append(dict(model=model,noise_std=sigma,window=w,E=E,rmse_ms=err,E_audit_abs_diff=diff,
                    baseline_file=str(p),baseline_sha256=digest(p),E_matches=True,rmse_matches=True,
                    basis='saved original-estimator shifts, score recomputed here; full direct checks queued'))
    write_json(ROOT/'baseline/rechecked_24.json',rows)
    tests=unit_tests()
    write_json(ROOT/'baseline/unit_tests.json',dict(passed=True,n_cases=len(tests),cases=tests))
    with (PRIOR/'baseline/comparison.csv').open() as f: timing=list(csv.DictReader(f))
    seconds=[float(r['elapsed_seconds']) for r in timing if r['model']=='continuous' and int(r['window_size'])==W]
    count=len(CELLS)*sum(n for _,n in SPLITS)
    estimate=dict(whole_pair_runs=count,derived_variants_per_run=8,
        baseline_seconds_W513=seconds,core_hours_low=min(seconds)*count/3600,
        core_hours_high=max(seconds)*count/3600,
        note='H1 original raw search once per pair; reuse exact raw windows for other hops. E audit, IO, plotting, thermal slowdown add time. Queue wait excluded.')
    write_json(ROOT/'baseline/runtime_estimate.json',estimate)
    env=dict(python=sys.version,platform=platform.platform(),executable=sys.executable,
        packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','matplotlib']},
        git_commit=subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','-C',str(PROJECT),'status','--short'],text=True),
        source_unchanged=True)
    write_json(ROOT/'environment.json',env)
    print(json.dumps(dict(baselines=len(rows),unit_tests=len(tests),estimate=estimate)),flush=True)

def prior_state():
    try: p=read_json(PRIOR/'progress.json')
    except (OSError,ValueError): return False,{'status':'unreadable'}
    if p.get('status')!='complete': return False,p
    # Read-only advisory lock check: reporting also finishes before the lock is released.
    try:
        with (PRIOR/'run.lock').open('rb') as f:
            fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)
            fcntl.flock(f,fcntl.LOCK_UN)
    except (BlockingIOError,OSError): return False,p
    if not p.get('source_unchanged'): return False,p
    return True,p

def wait_for_prior(done=0):
    while True:
        ready,prior=prior_state()
        if ready: return
        write_json(ROOT/'progress.json',dict(status='waiting_for_2.4.1',completed=done,
            total=len(CELLS)*sum(n for _,n in SPLITS),prior_status=prior.get('status'),
            prior_completed=prior.get('completed'),prior_total=prior.get('total'),updated_epoch=time.time()))
        if (PRIOR/'fatal_error.json').exists():
            raise RuntimeError('2.4.1 reports failure; remain stopped and request review rather than compete or restart it')
        time.sleep(30)

def direct_baseline():
    rows=[]
    for model in ['discrete','continuous']:
        s=signals(model)
        fun='rectangle' if model=='discrete' else 'Hann'
        for sigma in [0.,.02]:
            wait_for_prior()
            rng=np.random.default_rng(0 if model=='discrete' else 2025)
            x=s['x']+rng.normal(0,sigma,len(s['x']))
            y=s['y']+rng.normal(0,sigma,len(s['y']))
            start=time.monotonic()
            c,raw=original_estimator(x,y,W,1,fun,False)
            dense_seconds=time.monotonic()-start
            results,arrays=metrics(x,y,s,model,True,c,raw)
            for h in HOPS:
                cc,rr=original_estimator(x,y,W,h,fun,False) if h!=1 else (c,raw)
                cf,ff=original_estimator(x,y,W,h,fun,True)
                assert np.array_equal(cc,c[::h]) and np.array_equal(rr,raw[::h])
                assert np.array_equal(cf,cc) and np.array_equal(ff,arrays[f'fused_H{h}'])
            primary_h,primary_stage=(256,'raw') if model=='discrete' else (1,'fused')
            with np.load(PRIOR/f'baseline/{model}_noise{sigma:.2f}_W513.npz') as z:
                assert np.array_equal(z['centers'],arrays[f'centers_H{primary_h}'])
                assert np.array_equal(z['shifts'],arrays[f'{primary_stage}_H{primary_h}'])
            rows.append(dict(model=model,noise_std=sigma,dense_raw_seconds=dense_seconds,
                all_hops_raw_and_fused_exact=True,original_baseline_shifts_exact=True))
    write_json(ROOT/'baseline/direct_checks.json',rows)
    write_json(ROOT/'baseline/timing_after_queue.json',dict(checks=rows,
        approximate_compute_hours=np.median([r['dense_raw_seconds'] for r in rows])*len(CELLS)*sum(n for _,n in SPLITS)/3600))

def run():
    lock=(ROOT/'run.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        os.nice(10)
        priority='nice +10 applied'
    except PermissionError:
        priority='OS denied priority adjustment; strict serial dependency gate retained'
        print(priority,flush=True)
    write_json(ROOT/'process.json',dict(pid=os.getpid(),started_epoch=time.time(),priority=priority,workers=1))
    conf=read_json(ROOT/'frozen_config.json')
    assert config()==conf
    wait_for_prior()
    assert config()==conf
    if not (ROOT/'baseline/direct_checks.json').exists(): direct_baseline()
    # Deterministic zero-noise controls, not counted as independent repetitions.
    for model in ['discrete','continuous']:
        s=signals(model)
        fun='rectangle' if model=='discrete' else 'Hann'
        for dynamic in [False,True]:
            path=ROOT/f'checks/{model}_{"dynamic" if dynamic else "constant"}.json'
            if path.exists(): continue
            wait_for_prior()
            y=s['y' if dynamic else 'null_y']
            c,raw=original_estimator(s['x'],y,W,1,fun,False)
            rows,arrays=metrics(s['x'],y,s,model,dynamic,c,raw)
            if not dynamic:
                d0=12 if model=='discrete' else 0
                assert np.all(raw==d0),'Unique noiseless shift assumption fails; do not hide this control failure'
                assert max(r['T_ms2'] for r in rows)<1e-20
            write_npz(path.with_suffix('.npz'),**arrays)
            write_json(path,dict(model=model,dynamic=dynamic,metrics=rows))
    start=time.monotonic()
    total=len(CELLS)*sum(n for _,n in SPLITS)
    done=0
    for cell_id,(model,sigma,rho) in enumerate(CELLS):
        s=signals(model)
        fun='rectangle' if model=='discrete' else 'Hann'
        for split_id,(split,nrep) in enumerate(SPLITS):
            for rep in range(nrep):
                path=ROOT/f'checkpoints/cell{cell_id}/{split}/rep{rep:04d}.json'
                x,y,seed,dynamic=noisy_pair(s,cell_id,split_id,rep)
                hashes=dict(x_hash=array_hash(x),y_hash=array_hash(y))
                if path.exists() and path.with_suffix('.npz').exists():
                    old=read_json(path)
                    assert old['seed_words']==seed and all(old[k]==v for k,v in hashes.items())
                    assert digest(path.with_suffix('.npz'))==old['arrays_sha256']
                    done+=1
                    continue
                wait_for_prior(done)
                assert source_hashes()==conf['all_source_hashes'],'Original source changed; stop'
                tick=time.monotonic()
                c,raw=original_estimator(x,y,W,1,fun,False)
                rows,arrays=metrics(x,y,s,model,dynamic,c,raw)
                write_npz(path.with_suffix('.npz'),**arrays)
                row=dict(cell_id=cell_id,model=model,noise_std=sigma,ar1_rho=rho,
                    split=split,replicate=rep,seed_words=seed,**hashes,metrics=rows,
                    arrays_sha256=digest(path.with_suffix('.npz')),elapsed_seconds=time.monotonic()-tick)
                write_json(path,row)
                done+=1
                p=dict(status='running',completed=done,total=total,cell_id=cell_id,model=model,
                    noise_std=sigma,ar1_rho=rho,split=split,replicate=rep,
                    elapsed_seconds=time.monotonic()-start,updated_epoch=time.time())
                write_json(ROOT/'progress.json',p)
                if done%10==0: print(json.dumps(p),flush=True)
    assert config()==conf
    write_json(ROOT/'progress.json',dict(status='simulation_complete',completed=done,total=total,updated_epoch=time.time()))
    from report import build_report
    build_report(ROOT)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','report','test'])
    a=parser.parse_args()
    try:
        if a.action=='prepare': prepare()
        elif a.action=='run': run()
        elif a.action=='test': print(json.dumps(unit_tests()))
        else:
            from report import build_report
            build_report(ROOT)
    except Exception:
        write_json(ROOT/'failure.json',dict(action=a.action,traceback=traceback.format_exc(),epoch=time.time()))
        raise
