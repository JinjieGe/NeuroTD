from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy.stats import beta

P=Path(__file__).resolve().parent
ip=P.parent/'NeuroTD_review242_round2_audit_20261031/scripts/infer.py'
# Previously audited pure inference module; no pipeline or uploaded task is run.
spec=importlib.util.spec_from_file_location('audited_inference',ip)
I=importlib.util.module_from_spec(spec); spec.loader.exec_module(I)
bp=pd.read_csv(P/'electrodes/bundle_pairs_112.csv')
sh=np.load(P/'checkpoints/bundle_shifts_150x112x508.npz',allow_pickle=False)['shifts']
S=np.zeros((150,4)); N=np.zeros((150,4))
for j,b in enumerate(bp.bundle.unique()):
    x=sh[:,bp.index[bp.bundle==b],:].reshape(150,-1)
    S[:,j]=np.abs(x).sum(1)/32
    N[:,j]=(x!=0).sum(1)
w=np.ones(150); sessions=[(0,50),(50,100),(100,150)]
bo=I.boot_inference(S,N,w,sessions,np.random.default_rng(np.random.SeedSequence([20261031,3,0])),20000,Ls=[10])
df=pd.read_csv(P/'realdata/final_decisions.csv')
r={'max_delta_difference':float(np.max(np.abs(bo['delta']-df.difference_ms))),
   'max_raw_p_difference':float(np.max(np.abs(bo['p'][1]-df.M2_L10_raw_p))),
   'max_ci_low_difference':float(np.max(np.abs(bo['delta']-bo['half95'][1]-df.M2_L10_ci95_low))),
   'valid_bootstrap_replicates':bo['n_valid'].tolist()}
print('real inference reproduced',r,flush=True)
sd=(S/N).std(0)
checks=[]
for name,phi,seed in [('A_iid',0,20261031*20),('B2_phi0.6',0.6,20261031*20+1)]:
    rng=np.random.default_rng(seed)
    Y=np.zeros((1000,150,4))
    for s in range(3):
        e=rng.normal(0,1,size=(1000,50,4))*sd
        Y[:,s*50,:]=e[:,0]/np.sqrt(1-phi**2)
        for t in range(1,50):Y[:,s*50+t,:]=phi*Y[:,s*50+t-1,:]+e[:,t]
    rej=0
    for i in range(1000):
        out=I.boot_inference(Y[i]*N,N,w,sessions,np.random.default_rng(seed+1000+i),1999,Ls=[10])
        rej+=bool((np.minimum(6*out['p'][1],1)<0.05).any())
        if (i+1)%250==0: print(name,i+1,'rejections',rej,flush=True)
    checks.append(dict(scenario=name,n=1000,rejections=rej,fwer=rej/1000,
                       cp_low=float(beta.ppf(.025,rej,1000-rej+1)),cp_high=float(beta.ppf(.975,rej+1,1000-rej))))
r['validation_reproduced']=checks
(P/'independent_checks/inference_reproduction.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
