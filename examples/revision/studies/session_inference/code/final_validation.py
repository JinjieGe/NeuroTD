import time,json
import numpy as np
import pandas as pd
from scipy import stats
from run_analysis import ROOT,input_data,SCENARIOS,ALPHAS,stage_scores,cp,save_json,provenance

_,S,N,P=input_data(); before=provenance(); start=time.monotonic(); rows=[]
for si,s in enumerate(SCENARIOS):
    for ni,nt in enumerate(['G','P1','P2']):
        path=ROOT/f'checkpoints/final_{s}_{nt}.json'
        if path.exists(): r=json.loads(path.read_text())
        else:
            sc,_,tr,_=stage_scores(s,nt,4,20000,P)
            error=np.abs(sc[:,0,1]);obs=np.abs(sc[:,1,1]);mask=np.isclose(tr,0)
            p=np.minimum(1,12*stats.t.sf(obs,2));r=[]
            for alpha in ALPHAS:
                k=int(np.any(p[:,mask]<alpha,axis=1).sum());lo,hi=cp(k,20000)
                c=stats.t.isf(alpha/12,2); ncover=int(np.all(error<=c,axis=1).sum());cl,ch=cp(ncover,20000)
                r.append(dict(scenario=s,null_type=nt,n=20000,alpha=alpha,method='session_jackknife_t2',
                    false_rejections=k,fwer=k/20000,fwer_ci_low=lo,fwer_ci_high=hi,gate=hi<=1.25*alpha,
                    simultaneous_coverage=ncover/20000,coverage_ci_low=cl,coverage_ci_high=ch))
            save_json(path,r)
        rows.extend(r)
        done=si*3+ni+1
        save_json(ROOT/'progress.json',dict(status='final_validation',completed_tasks=done,total_tasks=30,latest=f'{s} {nt}',elapsed_seconds=time.monotonic()-start))
        print(done,'/30',s,nt,round(time.monotonic()-start,1),flush=True)
pd.DataFrame(rows).to_csv(ROOT/'outputs/final_validation.csv',index=False)
assert before==provenance()
save_json(ROOT/'progress.json',dict(status='computed_pending_review',additional_validation_records=600000,
    total_formal_simulations=1440000,elapsed_seconds=time.monotonic()-start))
print('FINAL VALIDATION COMPUTED',flush=True)
