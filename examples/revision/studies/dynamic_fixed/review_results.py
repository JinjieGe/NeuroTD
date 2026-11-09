"""Independent post-run audit; does not call the simulation/report summaries."""
from revision_runtime import resolve_path
from pathlib import Path
from collections import defaultdict
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR']=resolve_path('dynamic_fixed', 'cache/matplotlib')
os.environ['XDG_CACHE_HOME']=resolve_path('dynamic_fixed', 'cache')
import numpy as np
from scipy.stats import binomtest

ROOT = Path(resolve_path('dynamic_fixed', ''))
PROJECT = Path(resolve_path('repo', ''))
PRIOR = Path(resolve_path('score_uncertainty', ''))
OUT = ROOT/'outputs'
conf = json.loads((ROOT/'frozen_config.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
readcsv = lambda p: list(csv.DictReader(p.open()))
def close(x,y):
    assert np.isclose(float(x),float(y),rtol=1e-11,atol=1e-12), (x,y)

with (ROOT/'run.lock').open('rb') as lock:
    fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
    fcntl.flock(lock,fcntl.LOCK_UN)
assert not (ROOT/'failure.json').exists()
actual = {str(p.relative_to(PROJECT)):sha(p) for p in sorted((PROJECT/'src').rglob('*.py'))}
assert actual == conf['all_source_hashes']
assert all(sha(ROOT/k)==v for k,v in conf['harness_hashes'].items())
assert all(sha(PRIOR/f'clean_signals/{k}.npz')==v for k,v in conf['clean_signal_files'].items())
env = json.loads((ROOT/'environment.json').read_text())
git_status = subprocess.check_output(['git','-C',str(PROJECT),'status','--short'],text=True)
assert git_status == env['git_status']
assert subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()==env['git_commit']

records = {}
groups = defaultdict(list)
paths = sorted((ROOT/'checkpoints').rglob('rep*.json'))
assert len(paths)==4193
for p in paths:
    r=json.loads(p.read_text())
    key=(r['cell_id'],r['split'],r['replicate'])
    assert key not in records
    assert r['seed_words']==[conf['master_seed'],key[0],list(conf['splits']).index(key[1]),key[2]]
    assert all(r[k]==v for k,v in conf['cells'][key[0]].items())
    assert sha(p.with_suffix('.npz'))==r['arrays_sha256']
    assert {(q['hop'],q['stage']) for q in r['metrics']}=={(h,s) for h in conf['hops'] for s in ['raw','fused']}
    assert len(r['metrics'])==8
    records[key]=r
    groups[key[:2]].append(r)
for (cid,split),rs in groups.items():
    rs.sort(key=lambda r:r['replicate'])
    assert [r['replicate'] for r in rs]==list(range(conf['splits'][split]))
assert len(groups)==21

flat=readcsv(OUT/'repeat_metrics.csv')
assert len(flat)==33544
for q in flat:
    r=records[(int(q['cell_id']),q['split'],int(q['replicate']))]
    a=next(m for m in r['metrics'] if (m['hop'],m['stage'])==(int(q['hop']),q['stage']))
    for field in a:
        if field!='stage': close(q[field],a[field])

def values(cid,split,h,stage):
    rs=groups[cid,split]
    return np.array([max(q['T_ms2'] for q in r['metrics']) if stage=='family_max' else
        next(q['T_ms2'] for q in r['metrics'] if (q['hop'],q['stage'])==(h,stage)) for r in rs])

tests=readcsv(OUT/'whole_curve_tests.csv')
summary=readcsv(OUT/'test_summary.csv')
assert len(tests)==25200 and len(summary)==126
testlookup={(int(t['cell_id']),t['split'],int(t['replicate']),int(t['hop']),t['stage']):t for t in tests}
assert len(testlookup)==25200
for row in summary:
    cid,h,stage=int(row['cell_id']),int(row['hop']),row['stage']
    split='test_null' if row['endpoint']=='false_positive_rate' else 'test_dynamic'
    cal=np.sort(values(cid,'calibration_null',h,stage))
    obs=values(cid,split,h,stage)
    pv=(200-np.searchsorted(cal,obs,side='left'))/200
    k=int(np.sum(pv<=.05))
    ci=binomtest(k,200).proportion_ci(confidence_level=.95,method='exact')
    assert int(row['rejections'])==k and int(row['n'])==200 and int(row['calibration_n'])==199
    close(row['rate'],k/200); close(row['ci95_low'],ci.low); close(row['ci95_high'],ci.high)
    close(row['test_T_mean'],obs.mean()); close(row['calibration_T_q95'],np.quantile(cal,.95))
    for rep,(v,p) in enumerate(zip(obs,pv)):
        t=testlookup[cid,split,rep,h,stage]
        close(t['T_ms2'],v); close(t['p_value'],p)
        assert (t['reject']=='True')==(p<=.05)

desc=readcsv(OUT/'delay_and_E_summary.csv')
assert len(desc)==168
for d in desc:
    rs=groups[int(d['cell_id']),d['split']]
    qs=[next(q for q in r['metrics'] if (q['hop'],q['stage'])==(int(d['hop']),d['stage'])) for r in rs]
    assert int(d['n'])==len(qs)
    for field in ['T_ms2','rmse_ms','bias_ms','range_ms','E']:
        a=np.array([q[field] for q in qs])
        for suffix,calc in [('mean',np.mean),('median',np.median),('q025',lambda a:np.quantile(a,.025)),('q975',lambda a:np.quantile(a,.975))]:
            close(d[f'{field}_{suffix}'],calc(a))
    if d['stage']=='fused': close(d['common_fused_variance_ms2_mean'],np.mean([q['common_fused_variance_ms2'] for q in qs]))

# Prespecified audit sample: first and last replicate in each of the 21 groups.
sample=[rs[i] for rs in groups.values() for i in [0,-1]]
sys.path.insert(0,str(PROJECT/'src'))
from neurotd.time_varying_shift import evaluate_alignment_quality_sliding
arrayhash=lambda a: hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()
max_E_difference=0.
for r in sample:
    cid,split,rep=r['cell_id'],r['split'],r['replicate']
    with np.load(PRIOR/f"clean_signals/{r['model']}.npz") as clean:
        x0=clean['x'].copy()
        y0=clean['y'].copy() if split=='test_dynamic' else np.roll(x0,conf['constant_delay_samples'][r['model']])
    rng=np.random.default_rng(np.random.SeedSequence(r['seed_words']))
    def draw_noise():
        sigma,rho=r['noise_std'],r['ar1_rho']
        if rho==0: return rng.normal(0,sigma,len(x0))
        z=rng.normal(0,sigma*np.sqrt(1-rho*rho),len(x0))
        z[0]=rng.normal(0,sigma)
        for i in range(1,len(z)): z[i]+=rho*z[i-1]
        return z
    x,y=x0+draw_noise(),y0+draw_noise()
    assert arrayhash(x)==r['x_hash'] and arrayhash(y)==r['y_hash']
    with np.load(ROOT/f'checkpoints/cell{cid}/{split}/rep{rep:04d}.npz') as z:
        for q in r['metrics']:
            h,stage=q['hop'],q['stage']
            c,d,truth=z[f'centers_H{h}'],z[f'{stage}_H{h}'],z[f'truth_H{h}']
            assert len(c)==q['M'] and np.isfinite(d).all()
            assert np.array_equal(c,z['centers_H1'][::h])
            assert np.array_equal(z[f'raw_H{h}'],z['raw_H1'][::h])
            ms=d*1000/conf['fs'];err=(d-truth)*1000/conf['fs']
            close(q['T_ms2'],np.var(ms));close(q['rmse_ms'],np.sqrt(np.mean(err**2)))
            close(q['bias_ms'],np.mean(err));close(q['range_ms'],np.ptp(ms))
            if stage=='fused': close(q['common_fused_variance_ms2'],np.var(z[f'common_fused_H{h}']*1000/conf['fs']))
            E,_=evaluate_alignment_quality_sliding(x,y,np.round(d).astype(int),c,conf['window'])
            close(q['E'],E);max_E_difference=max(max_E_difference,abs(E-q['E']))

audit={'passed':True,'reviewed_epoch':time.time(),'source_files_verified':len(actual),
    'original_git_state_preserved':True,'frozen_scripts_unchanged':True,'clean_inputs_unchanged':True,
    'checkpoints_hashed':len(records),'unique_seed_tuples':len({tuple(r['seed_words']) for r in records.values()}),
    'repeat_metric_rows_checked':len(flat),'p_values_recomputed':len(tests),'test_summary_rows_checked':len(summary),
    'descriptive_summary_rows_checked':len(desc),'array_and_noise_sample_records':len(sample),
    'array_metric_variants_checked':8*len(sample),'original_E_sample_max_abs_diff':max_E_difference,
    'sample_rule':'first and last replicate of every cell/split; no outcome-based selection',
    'scope':'No estimator runs, protocol changes or re-calibration. Full metadata/summary audit; sampled array audit.'}
(OUT/'independent_review.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
print('Primary descriptive results: cell, split, variance, RMSE, bias, E')
for d in desc:
    primary=conf['primary'][d['model']]
    if d['split']!='calibration_null' and (int(d['hop']),d['stage'])==(primary['hop'],primary['stage']):
        print(d['cell_id'],d['split'],*(round(float(d[k]),6) for k in ['T_ms2_mean','rmse_ms_mean','bias_ms_mean','E_mean']))
print('Family max: cell, endpoint, rejections, rate')
for d in summary:
    if d['stage']=='family_max':print(d['cell_id'],d['endpoint'],d['rejections'],d['rate'])
print('Null sensitivity: cell, hop, raw T, fused T, common fused T, raw RMSE, fused RMSE')
for cid in range(7):
    for h in conf['hops']:
        rows=[next(d for d in desc if int(d['cell_id'])==cid and d['split']=='test_null' and int(d['hop'])==h and d['stage']==s) for s in ['raw','fused']]
        a,b=rows
        print(cid,h,*(round(float(v),8) for v in [a['T_ms2_mean'],b['T_ms2_mean'],b['common_fused_variance_ms2_mean'],a['rmse_ms_mean'],b['rmse_ms_mean']]))
