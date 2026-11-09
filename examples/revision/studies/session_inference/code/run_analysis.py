"""Standalone inference audit. No writes to source data or original NeuroTD project."""
from revision_runtime import resolve_path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import sys, json, time, hashlib, argparse, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from infer import REGIONS, COMP_NAMES, pairwise, influence, _residualize, hac_variances
from genmodels import generate, estimate_params, SCENARIOS, CORE, SESSIONS, _center, _ar

ROOT=Path(__file__).resolve().parents[1]
SEED=202611080242
ALPHAS=(.05,.01,.001)
METHODS=('session_equal_t2','session_jackknife_t2','HAC_b1_normal','HAC_b1_t2','model_envelope')
MARGIN=1.15
BATCH=500

def save_json(path, obj):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)); tmp.replace(path)

def hfile(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def input_data():
    t=pd.read_csv(ROOT/'inputs/trial_region_summary.csv')
    assert not t.duplicated(['session','trial_pos','region']).any()
    assert len(t)==600 and t.session.nunique()==3
    arrays=[]
    for col in ('S_nonzero_ms','N_nonzero'):
        a=t.pivot(index=['session','trial_pos'],columns='region',values=col)[REGIONS].to_numpy(float)
        assert a.shape==(150,4) and np.isfinite(a).all()
        arrays.append(a)
    S,N=arrays
    assert N.sum()==4108812 and t.n_windows_raw.sum()==4572000
    assert np.all(N>0)
    return t,S,N,estimate_params(t)

def measures(S,N):
    """Preserve pooled estimand; compute alternative SEs without window independence."""
    u,d=influence(S,N,np.ones(150))
    r=_residualize(u,SESSIONS,np.ones(150),False)
    # Identity for Bartlett bandwidth n with demeaned length-n segments:
    # gamma0+2 sum(1-lag/n)gamma_lag = 2/n sum cumulative_residual^2.
    v=np.zeros(d.shape)
    sd=[]; leave=[]
    for a,b in SESSIONS:
        v+=2/(b-a)*(np.cumsum(r[...,a:b,:],axis=-2)**2).sum(-2)
        sd.append(pairwise(S[...,a:b,:].sum(-2)/N[...,a:b,:].sum(-2)))
        leave.append(pairwise((S.sum(-2)-S[...,a:b,:].sum(-2))/(N.sum(-2)-N[...,a:b,:].sum(-2))))
    sd=np.stack(sd,axis=-2); leave=np.stack(leave,axis=-2)
    se_h=np.sqrt(v)
    se_j=np.sqrt(2/3*((leave-leave.mean(-2,keepdims=True))**2).sum(-2))
    eq=sd.mean(-2); se_eq=sd.std(axis=-2,ddof=1)/np.sqrt(3)
    assert np.all(se_h>0) and np.all(se_j>0) and np.all(se_eq>0)
    return d,se_h,se_j,eq,se_eq,sd

def eq_truth(scen,truth,P):
    if scen in ('A','B1','B2','E1'): return truth
    w=np.stack([P['N_real'][a:b].sum(0) for a,b in SESSIONS])
    return truth+pairwise(_center(P['m']-P['m'].mean(0),w).mean(0))

def cp(k,n):
    return (float(stats.beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(stats.beta.ppf(.975,k+1,n-k)) if k<n else 1.)

def sample(scen,ntype,rng,n,P):
    if scen=='E1':
        N=np.broadcast_to(P['N_real'],(n,150,4)).copy()
        chol=np.linalg.cholesky(P['R']); y=np.zeros_like(N)
        for a,b in SESSIONS:
            y[:,a:b]=9+P['sd'].mean(0)*_ar(rng,n,b-a,np.ones(4)*.97,chol)
        return y*N,N,np.zeros(6)
    if scen=='E2':
        S,N,tr=generate('C1',ntype,rng,n,P)
        S=S+N*rng.normal(0,.2,(n,1,4))
        return S,N,tr
    return generate(scen,ntype,rng,n,P)

def stage_scores(scen,ntype,stage,n,P):
    idx=(list(SCENARIOS)+['E1','E2']).index(scen)
    ti=['G','P1','P2'].index(ntype)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,stage,idx,ti]))
    scores=[]; ests=[]; ses=[]
    for start in range(0,n,BATCH):
        S,N,tr=sample(scen,ntype,rng,min(BATCH,n-start),P)
        d,h,j,eq,eqse,sd=measures(S,N)
        et=eq_truth(scen,tr,P)
        err=np.stack([(eq-et)/eqse,(d-tr)/j,(d-tr)/h],axis=1)
        obs=np.stack([eq/eqse,d/j,d/h],axis=1)
        scores.append(np.stack([err,obs],axis=1))
        ests.append(np.stack([d,eq],axis=1))
    return np.concatenate(scores),np.concatenate(ests),tr,eq_truth(scen,tr,P)

def refs_load():
    return {s:np.sort(np.load(ROOT/f'checkpoints/cal_{s}.npz')['max_score']) for s in SCENARIOS}

def envelope_p(x,refs):
    p=np.zeros_like(x,dtype=float)
    for a in refs.values():
        p=np.maximum(p,(1+len(a)-np.searchsorted(a,x/MARGIN,side='left'))/(len(a)+1))
    return p

def criticals(refs):
    out={}
    for alpha in ALPHAS:
        # (1+# >= x)/(R+1)<alpha. Strict order-statistic cutoff.
        ks=[]
        for a in refs.values():
            rank=int(np.floor((1-alpha)*(len(a)+1)))
            ks.append(a[min(rank,len(a)-1)]*MARGIN)
        out[str(alpha)]=float(max(ks))
    return out

def validate_one(scen,ntype,stage,n,P,refs):
    sc,est,tr,et=stage_scores(scen,ntype,stage,n,P)
    err=np.abs(sc[:,0]); obs=np.abs(sc[:,1]); crit=criticals(refs)
    probs=[np.minimum(1,6*2*stats.t.sf(obs[:,0],2)),
           np.minimum(1,6*2*stats.t.sf(obs[:,1],2)),
           np.minimum(1,6*2*stats.norm.sf(obs[:,2])),
           np.minimum(1,6*2*stats.t.sf(obs[:,2],2)),envelope_p(obs[:,2],refs)]
    rows=[]
    for mi,name in enumerate(METHODS):
        own=et if mi==0 else tr
        mask=np.isclose(own,0,atol=1e-10)
        e=err[:,[0,1,2,2,2][mi]]
        for alpha in ALPHAS:
            c=(crit[str(alpha)] if mi==4 else
               stats.norm.isf(alpha/12) if mi==2 else stats.t.isf(alpha/12,2))
            cover=np.all(e<=c,axis=1)
            nk=int(cover.sum()); clo,chi=cp(nk,n)
            false=np.any(probs[mi][:,mask]<alpha,axis=1) if mask.any() else None
            k=int(false.sum()) if false is not None else None
            lo,hi=cp(k,n) if k is not None else (None,None)
            rows.append(dict(scenario=scen,null_type=ntype,method=name,alpha=alpha,n=n,
                n_true_nulls=int(mask.sum()),false_rejections=k,fwer=k/n if k is not None else None,
                fwer_ci_low=lo,fwer_ci_high=hi,gate=(hi<=1.25*alpha) if hi is not None else None,
                simultaneous_coverage=nk/n,coverage_ci_low=clo,coverage_ci_high=chi,
                centered_error_exceedance=1-nk/n))
    return rows

def audit(P,S,N):
    rng=np.random.default_rng(2345)
    A,B,tr=generate('B2','G',rng,20,P)
    d,h,j,eq,eqse,sd=measures(A,B)
    u,_=influence(A,B,np.ones(150)); r=_residualize(u,SESSIONS,np.ones(150),False)
    old=hac_variances(r,50,[1.],SESSIONS)[:,0,:]
    assert np.allclose(h*h,old,rtol=1e-10,atol=1e-15)
    shift=np.array([0,.7,.3,-.2])
    d2,h2,j2,eq2,eqse2,_=measures(A+B*shift,B)
    assert np.allclose(d2-d,pairwise(shift),atol=1e-12)
    assert np.allclose(h2,h,rtol=1e-10,atol=1e-12)
    # Ratio delete-cluster SE is not translation sensitive.
    assert np.allclose(j2,j,rtol=1e-10,atol=1e-12)
    assert np.allclose(eq2-eq,pairwise(shift),atol=1e-12)
    assert np.allclose(eqse2,eqse,rtol=1e-10,atol=1e-12)
    # Reconcile exact input aggregate, units and unique identities above.
    out=dict(n_trials=150,n_sessions=3,n_subjects=1,nonzero_windows=int(N.sum()),
        max_fast_hac_variance_error=float(np.max(np.abs(old-h*h))),
        pooled_group_means_ms=(S.sum(0)/N.sum(0)).tolist(),
        pooled_differences_ms=pairwise(S.sum(0)/N.sum(0)).tolist(),
        translation_invariance_pass=True)
    save_json(ROOT/'outputs/audit.json',out)

def provenance():
    paths=[ROOT/'inputs/trial_region_summary.csv',ROOT/'code/infer.py',ROOT/'code/genmodels.py',ROOT/'PLAN.txt',ROOT/'code/run_analysis.py',
      Path(resolve_path('audit_v2', 'scripts/infer.py')),
      Path(resolve_path('audit_v2', 'scripts/genmodels.py')),
      Path(resolve_path('repo', 'src/neurotd/time_varying_shift.py')),
      Path(resolve_path('audit_v1', 'tables/trial_region_summary.csv'))]
    return {str(p):hfile(p) for p in paths if p.exists()}

def real_results(S,N,refs):
    d,h,j,eq,eqse,sd=measures(S,N); c=criticals(refs)
    pp=[np.minimum(1,12*stats.t.sf(np.abs(eq/eqse),2)),np.minimum(1,12*stats.t.sf(np.abs(d/j),2)),
        np.minimum(1,12*stats.norm.sf(np.abs(d/h))),np.minimum(1,12*stats.t.sf(np.abs(d/h),2)),envelope_p(np.abs(d/h),refs)]
    rows=[]
    for mi,m in enumerate(METHODS):
        dd=eq if mi==0 else d; se=[eqse,j,h,h,h][mi]
        cv=c['0.05'] if mi==4 else stats.norm.isf(.05/12) if mi==2 else stats.t.isf(.05/12,2)
        for k,name in enumerate(COMP_NAMES):
            rows.append(dict(method=m,comparison=name,estimate_ms=dd[k],se_ms=se[k],p_family=pp[mi][k],
                lower_simultaneous95_ms=dd[k]-cv*se[k],upper_simultaneous95_ms=dd[k]+cv*se[k],
                reference_df=2 if mi in (0,1,3) else None,scope='model_conditional_or_exploratory'))
    pd.DataFrame(rows).to_csv(ROOT/'outputs/real_results.csv',index=False)
    pd.DataFrame(sd,index=['ses-01','ses-02','ses-03'],columns=COMP_NAMES).to_csv(ROOT/'outputs/session_differences.csv')
    # Exact session sign-flip diagnostic: requires independent symmetric session contrasts.
    flips=np.array(list(itertools.product([-1,1],repeat=3)))
    pv=(np.abs(flips@sd/3)>=np.abs(eq)[None,:]-1e-12).mean(0)
    pd.DataFrame(dict(comparison=COMP_NAMES,estimate_ms=eq,p_exact=pv,p_bonferroni=np.minimum(1,6*pv))).to_csv(ROOT/'outputs/session_signflip.csv',index=False)
    save_json(ROOT/'outputs/envelope_critical_values.json',c)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true'); args=ap.parse_args()
    t,S,N,P=input_data(); audit(P,S,N)
    pre=provenance(); prov=ROOT/'provenance_before.json'
    if not prov.exists():save_json(prov,pre)
    else: assert json.loads(prov.read_text())==pre,'Input/code changed: review required'
    start=time.monotonic()
    if args.pilot:
        for s in SCENARIOS: stage_scores(s,'G',99,1000,P)
        elapsed=time.monotonic()-start
        save_json(ROOT/'outputs/pilot.json',dict(n=10000,elapsed_seconds=elapsed,estimated_full_seconds=elapsed*84))
        print('Pilot',elapsed,'seconds; formal estimate',elapsed*84/60,'minutes',flush=True);return
    tasks=10+30+2; done=0
    def progress(label):
        nonlocal done
        done+=1; elapsed=time.monotonic()-start
        save_json(ROOT/'progress.json',dict(status='running',completed_tasks=done,total_tasks=tasks,latest=label,elapsed_seconds=elapsed))
        print(f'{done}/{tasks} {label} elapsed={elapsed:.1f}s',flush=True)
    for s in SCENARIOS:
        path=ROOT/f'checkpoints/cal_{s}.npz'
        if not path.exists():
            sc,_,_,_=stage_scores(s,'G',1,20000,P)
            np.savez_compressed(path,max_score=np.max(np.abs(sc[:,0,2]),axis=1))
        progress('calibration '+s)
    refs=refs_load(); allrows=[]
    for stage,scenarios,types in [(2,list(SCENARIOS),['G','P1','P2']),(3,['E1','E2'],['G'])]:
        for s in scenarios:
            for nt in types:
                path=ROOT/f'checkpoints/val_{s}_{nt}.json'
                if path.exists(): rows=json.loads(path.read_text())
                else:
                    rows=validate_one(s,nt,stage,20000,P,refs);save_json(path,rows)
                allrows.extend(rows);progress('validation '+s+' '+nt)
    pd.DataFrame(allrows).to_csv(ROOT/'outputs/validation.csv',index=False)
    real_results(S,N,refs)
    after=provenance(); assert after==pre
    save_json(ROOT/'provenance_after.json',after)
    save_json(ROOT/'progress.json',dict(status='computed_pending_review',completed_tasks=tasks,total_tasks=tasks,elapsed_seconds=time.monotonic()-start))
    print('COMPUTED',flush=True)

if __name__=='__main__': main()
