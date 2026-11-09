"""Independent arithmetic and integrity checks; does not run the estimator."""
from revision_runtime import resolve_path
import sys,json,hashlib,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from run_analysis import ROOT,input_data,stage_scores,refs_load,envelope_p,criticals,provenance,save_json

t,S,N,P=input_data()
source=Path(resolve_path('audit_v1', 'tables/trial_pair_summary.csv'))
pp=pd.read_csv(source)
agg=pp.groupby(['session','trial_pos','region'])[['S_nonzero_ms','N_nonzero','n_windows_raw']].sum()
ref=t.set_index(['session','trial_pos','region'])[agg.columns].sort_index()
np.testing.assert_allclose(agg.sort_index(),ref,rtol=0,atol=1e-9)
assert len(pp)==9000
out=dict(trial_pair_rows=9000,trial_region_rows=600,source_pair_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         group_reaggregation_pass=True)
for path,h in json.loads((ROOT/'provenance_before.json').read_text()).items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
out['source_and_frozen_code_hashes_unchanged']=True
refs=refs_load(); rr=pd.read_csv(ROOT/'outputs/real_results.csv')
ses=pd.read_csv(ROOT/'outputs/session_differences.csv',index_col=0)
mu=[]
for _,g in t.groupby('session',sort=True):
    means=g.groupby('region').S_nonzero_ms.sum()/g.groupby('region').N_nonzero.sum()
    mu.append([means[a]-means[b] for a,b in itertools.combinations(['Amygdala','Hippocampus','Middle Temporal Gyrus','Superior Temporal Gyrus'],2)])
np.testing.assert_allclose(np.array(mu),ses.to_numpy(),atol=1e-12)
_,p=stats.ttest_1samp(np.array(mu),0,axis=0)
np.testing.assert_allclose(np.minimum(p*6,1),rr[rr.method=='session_equal_t2'].p_family,atol=1e-12)
out['session_t_independent_scipy_check']=True
pooled=t.groupby('region').S_nonzero_ms.sum()/t.groupby('region').N_nonzero.sum()
truth=[pooled[a]-pooled[b] for a,b in itertools.combinations(['Amygdala','Hippocampus','Middle Temporal Gyrus','Superior Temporal Gyrus'],2)]
np.testing.assert_allclose(truth,rr[rr.method=='model_envelope'].estimate_ms,atol=1e-12)
for scen in ('A','C2','D2'):
    sc,_,_,_=stage_scores(scen,'G',1,1000,P)
    saved=np.load(ROOT/f'checkpoints/cal_{scen}.npz')['max_score']
    np.testing.assert_array_equal(np.max(np.abs(sc[:,0,2]),axis=1),saved[:1000])
out['seed_replay_checks']=3
q=np.array([0,1,3,10,100,1000.])
pv=envelope_p(q,refs)
brute=np.max([(1+(a[:,None]>=q[None,:]/1.15).sum(0))/(len(a)+1) for a in refs.values()],axis=0)
np.testing.assert_array_equal(pv,brute)
assert np.all(np.diff(pv)<=0)
crit=criticals(refs)
for alpha,c in crit.items():
    assert float(envelope_p(np.array(c+1e-8),refs))<float(alpha)
out['empirical_tail_count_and_threshold_checks']=True
val=pd.read_csv(ROOT/'outputs/validation.csv')
assert len(val)==480 and val.n.eq(20000).all()
assert len(list((ROOT/'checkpoints').glob('cal_*.npz')))==10
assert len(list((ROOT/'checkpoints').glob('val_*.json')))==32
assert np.isfinite(rr[['estimate_ms','se_ms','p_family','lower_simultaneous95_ms','upper_simultaneous95_ms']]).all().all()
assert rr.p_family.between(0,1).all()
out.update(calibration_records=200000,validation_records=600000,challenge_records=40000,
           pilot_records=10000,all_checks_passed=True)
save_json(ROOT/'outputs/verification.json',out)
print(json.dumps(out,ensure_ascii=False,indent=2))
