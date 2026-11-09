"""Read-only NeuroTD replication and independent-noise uncertainty audit.

All scientific estimators are imported unchanged from the source repository.
This file and all outputs are outside that repository.
"""
from __future__ import annotations
from revision_runtime import resolve_path

import argparse
import ast
import csv
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parent
PROJECT = Path(resolve_path('repo', ''))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache' / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'cache'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / 'src'))

import numpy as np
import pandas as pd
from neurotd.neuro_signal import SegmentedSignal, my_ricker
from neurotd.time_varying_shift import (
    SignalParameters, compute_shifts_sliding_window_gt,
    compute_windowed_mse_with_hop, evaluate_alignment_quality_sliding,
)

WINDOWS = [33, 65, 129, 257, 513, 1025]
NOISES = [0.0, 0.02, 0.10, 0.20]
REPEATS = 200
MASTER_SEED = 20261018
FS = 5120


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode() + a.tobytes()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def continuous_generators():
    """Load source definitions, NOT the executable plotting/benchmark pipeline."""
    path = PROJECT / 'examples/simulations/fig3_simulation_continuous_shift.py'
    tree = ast.parse(path.read_text())
    chosen = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    module = types.ModuleType('audit_original_continuous_definitions')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


def clean_signals(model):
    if model == 'discrete':
        p = SignalParameters()
        s = SegmentedSignal(duration=1.0, sampling_rate=FS, n_segments=10, shape_func=my_ricker)
        shifted = s.generate_shifted(p.shifts)
        original_centers = s.n_samples_per_seg // 2 + s.n_samples_per_seg * np.arange(s.n_segments)
        return dict(x=s.data, y=shifted.data, t=s.time,
                    segment_centers=original_centers, segment_shifts=p.shifts)
    g = continuous_generators()
    rng = np.random.default_rng(2025)
    t, x, centers, widths = g.generate_reference_signal(duration=1.0, sampling_rate=FS, N_spike=10, rng=rng)
    _, delays = g.build_delay_curve(t=t, centers=centers, duration=1.0,
        Delta0=0.0, Delta1=0.01 * 0.2, fDelta=0.1, spike_jitter_std=0.001,
        boundary_mode='clamped', rng=rng)
    y = g.apply_continuous_delay(t, x, delays)
    return dict(x=x, y=y, t=t, delays=delays, spike_centers=centers, widths=widths)


def signal_manifest(signals):
    return {k: array_hash(v) for k, v in signals.items()}


def independent_score(x, y, shifts, centers, w):
    """Reproduce the exact code's normalization, not a redefined NRMSE."""
    r, valid_centers, normalizers = [], [], []
    half = w // 2
    for center, shift in zip(centers, shifts):
        lo, hi = int(center) - half, int(center) + half + 1
        a, b = lo + int(shift), hi + int(shift)
        if lo >= 0 and hi <= len(x) and a >= 0 and b <= len(y):
            xx, yy = x[lo:hi], y[a:b]
            denominator = np.sqrt(np.mean(xx ** 2) * np.mean(yy ** 2)) + 1e-16
            normalizers.append(denominator)
            r.append(np.sqrt(np.mean((xx - yy) ** 2)) / denominator)
            valid_centers.append(center)
    return np.asarray(r), np.asarray(valid_centers, dtype=int), np.asarray(normalizers)


def estimate(model, signals, x, y, w):
    start = time.perf_counter()
    continuous = model == 'continuous'
    hop = 1 if continuous else w // 2
    centers, shifts = compute_windowed_mse_with_hop(x, y, w, hop,
        window_fun='Hann' if continuous else 'rectangle', continuous=continuous)
    # Exactly as the original fig3 script: score rounded shifts; keep floats for RMSE.
    score_shifts = np.round(shifts).astype(int) if continuous else shifts
    e, returned_per_window = evaluate_alignment_quality_sliding(x, y, score_shifts, centers, w)
    r, valid_centers, denominators = independent_score(x, y, score_shifts, centers, w)
    independent_e = float(np.sqrt(np.mean(r)))
    if len(r) != len(centers) or not np.array_equal(valid_centers, centers):
        raise AssertionError('Unexpected invalid centers: do not silently drop windows')
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(shifts)) or not np.isfinite(e):
        raise AssertionError('Nonfinite estimator/score output')
    if not np.allclose(np.sqrt(r), returned_per_window, rtol=1e-12, atol=1e-12):
        raise AssertionError('Per-window return is not sqrt(r)')
    if not np.isclose(e, independent_e, rtol=1e-12, atol=1e-12):
        raise AssertionError('Independent E does not match')
    gt = (signals['delays'][centers] * FS if continuous else
          compute_shifts_sliding_window_gt(centers, signals['segment_centers'], signals['segment_shifts']))
    row = dict(model=model, window_size=w, hop=hop, M=len(r),
        E=float(e), E_independent=independent_e, E_abs_diff=abs(float(e)-independent_e),
        mean_r=float(np.mean(r)), rmse_ms=float(np.sqrt(np.mean((shifts-gt)**2)) * 1000 / FS),
        max_shift_samples=w//2, min_normalizer=float(denominators.min()),
        normalizer_at_floor_count=int(np.sum(denominators <= 1e-16)),
        elapsed_seconds=time.perf_counter()-start)
    arrays = dict(centers=centers, shifts=shifts, score_shifts=score_shifts, r=r,
                  ground_truth_samples=gt, normalizer=denominators,
                  returned_per_window=returned_per_window)
    return row, arrays


def score_tests():
    tests = []
    for name, x, y, expected in [
        ('identical_constant', np.ones(9), np.ones(9), 0.0),
        ('all_zero', np.zeros(9), np.zeros(9), 0.0),
        ('gain_two', np.ones(9), 2*np.ones(9), np.sqrt(0.5)),
    ]:
        e, per = evaluate_alignment_quality_sliding(x, y, np.array([0]), np.array([4]), 3)
        assert np.isclose(e, expected, atol=1e-14)
        r, _, _ = independent_score(x,y,np.array([0]),np.array([4]),3)
        assert np.allclose(per, np.sqrt(r))
        tests.append(dict(test=name, E=float(e), expected_E=expected, passed=True))
    # Covariance identity on a small matrix with both shared and independent variation.
    rng = np.random.default_rng(123)
    a = rng.normal(size=(100,5)) + rng.normal(size=(100,1))
    explicit = np.cov(a, rowvar=False, ddof=1).sum()/25
    direct = np.var(a.mean(axis=1), ddof=1)
    assert np.isclose(explicit, direct)
    tests.append(dict(test='covariance_identity', explicit=float(explicit), direct=float(direct), passed=True))
    save_json(ROOT/'baseline/unit_tests.json', tests)


def config():
    return dict(master_seed=MASTER_SEED, windows=WINDOWS, noise_stds=NOISES,
        nonzero_repeats=REPEATS, zero_noise_deterministic_checks=2, fs=FS,
        models=['discrete','continuous'], execution='one worker, unchanged source estimator',
        noise_rng='PCG64(SeedSequence([master_seed, model_id, noise_index, replicate_id]))',
        paired_windows=True, baseline_seeds={'discrete':0,'continuous':2025},
        E='sqrt(mean(r_k)); r_k = sqrt(MSE)/(sqrt(mean(x^2)*mean(y_aligned^2))+1e-16)',
        score_shifts='continuous: np.round(shifts).astype(int); discrete: unchanged integer shifts',
        full_source_hashes={str(p.relative_to(PROJECT)):digest(p) for p in sorted((PROJECT/'src').rglob('*.py'))})


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    conf = config()
    cp = ROOT/'frozen_config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == conf, 'Frozen configuration/source changed'
    else:
        save_json(cp, conf)
    versions = {}
    for name in ['numpy','scipy','pandas','matplotlib','tslearn','dtw-python','scikit-image']:
        versions[name] = importlib.metadata.version(name)
    info = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        versions=versions, estimator_path=inspect.getfile(compute_windowed_mse_with_hop),
        git_head=subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','-C',str(PROJECT),'status','--short'],text=True),
        baseline_csv_hashes={p.name:digest(p) for p in Path(resolve_path('baseline_tables')).glob('simulation_results_*.csv')})
    save_json(ROOT/'environment.json', info)
    for model in conf['models']:
        signals=clean_signals(model)
        save_npz(ROOT/f'clean_signals/{model}.npz', **signals)
        save_json(ROOT/f'clean_signals/{model}_hashes.json',signal_manifest(signals))
    (ROOT/'baseline').mkdir(exist_ok=True)
    (ROOT/'baseline/score_function_source.txt').write_text(inspect.getsource(evaluate_alignment_quality_sliding))
    score_tests()
    return conf


def baseline():
    prepare()
    rows=[]
    for model,seed in [('discrete',0),('continuous',2025)]:
        signals=clean_signals(model)
        old=pd.read_csv(Path(resolve_path('baseline_tables', f'simulation_results_{model}.csv')))
        for noise in [0.0,0.02]:
            rng=np.random.default_rng(seed)
            x=signals['x']+rng.normal(0,noise,len(signals['x']))
            y=signals['y']+rng.normal(0,noise,len(signals['y']))
            for w in WINDOWS:
                row,arrays=estimate(model,signals,x,y,w)
                ref=old[np.isclose(old.noise_std,noise)&(old.window_size==w)]
                assert len(ref)==1, 'Missing or duplicate local baseline'
                ref=ref.iloc[0]
                row.update(noise_std=noise, seed=seed,
                    reference_E=float(ref.quality_rmse_neurotd),
                    reference_rmse_ms=float(ref.rmse_neurotd)*1000/FS)
                row['E_matches_local_csv']=bool(np.isclose(row['E'],row['reference_E'],rtol=1e-8,atol=1e-10))
                row['rmse_matches_local_csv']=bool(np.isclose(row['rmse_ms'],row['reference_rmse_ms'],rtol=1e-8,atol=1e-10))
                rows.append(row)
                pd.DataFrame(rows).to_csv(ROOT/'baseline/comparison.csv',index=False)
                save_npz(ROOT/f'baseline/{model}_noise{noise:.2f}_W{w}.npz', **arrays)
                print(json.dumps(row),flush=True)
    timings=pd.DataFrame(rows).groupby(['model','window_size']).elapsed_seconds.median()
    totals={m:float(timings.loc[m].sum()*(3*REPEATS+2)) for m in ['discrete','continuous']}
    save_json(ROOT/'baseline/runtime_estimate.json',dict(single_worker_seconds=totals,
        total_hours=sum(totals.values())/3600, excludes='plotting, checkpoint IO, thermal and foreground load variation'))
    print('RUNTIME_ESTIMATE',totals,flush=True)


def run():
    lock=(ROOT/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save_json(ROOT/'process.json',dict(pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    conf=json.loads((ROOT/'frozen_config.json').read_text())
    assert conf==config(), 'Config/source mismatch before run'
    signals={m:clean_signals(m) for m in conf['models']}
    for m,s in signals.items():
        assert signal_manifest(s)==json.loads((ROOT/f'clean_signals/{m}_hashes.json').read_text())
    total=2*(2+3*REPEATS)*len(WINDOWS)
    done=0
    started=time.time()
    for model_id,model in enumerate(conf['models']):
        s=signals[model]
        for noise_index,noise in enumerate(NOISES):
            for rep in range(2 if noise==0 else REPEATS):
                seed_words=[MASTER_SEED,model_id,noise_index,rep]
                rng=np.random.default_rng(np.random.SeedSequence(seed_words))
                x=s['x']+rng.normal(0,noise,len(s['x']))
                y=s['y']+rng.normal(0,noise,len(s['y']))
                noise_hashes=dict(x_hash=array_hash(x),y_hash=array_hash(y))
                for w in WINDOWS:
                    path=ROOT/f'checkpoints/{model}/noise{noise:.2f}/rep{rep:04d}_W{w}.npz'
                    meta=path.with_suffix('.json')
                    if path.exists() and meta.exists():
                        previous=json.loads(meta.read_text())
                        assert previous['seed_words']==seed_words and previous['x_hash']==noise_hashes['x_hash']
                        done+=1
                        continue
                    try:
                        row,arrays=estimate(model,s,x,y,w)
                        row.update(noise_std=noise,replicate_id=rep,seed_words=seed_words,**noise_hashes)
                        save_npz(path,**arrays)
                        save_json(meta,row)
                        done+=1
                        progress=dict(status='running',completed=done,total=total,model=model,
                            noise_std=noise,replicate_id=rep,window_size=w,
                            elapsed_seconds=time.time()-started,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                        save_json(ROOT/'progress.json',progress)
                        if w==WINDOWS[-1]:
                            print(json.dumps(progress),flush=True)
                    except Exception:
                        save_json(ROOT/'failure.json',dict(model=model,noise_std=noise,replicate_id=rep,
                            window_size=w,traceback=traceback.format_exc()))
                        raise
    assert conf==config(), 'Source changed during run'
    save_json(ROOT/'progress.json',dict(status='simulation_complete',completed=done,total=total,
        elapsed_seconds=time.time()-started,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    summarize()


def summarize():
    # Defined in companion script, outside the source repository.
    from summarize import build_report
    build_report(ROOT)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['baseline','run','summarize'])
    args=parser.parse_args()
    try:
        {'baseline':baseline,'run':run,'summarize':summarize}[args.action]()
    except Exception:
        save_json(ROOT/'fatal_error.json',dict(action=args.action,traceback=traceback.format_exc(),
            time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        raise
