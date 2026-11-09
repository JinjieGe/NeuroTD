"""Small, clearly synthetic inferential fixtures. No scientific result files."""
from pathlib import Path
import json
import numpy as np
from report import summarize_statistics,monte_carlo_p,binomial_interval

conf=dict(cells=[dict(model='discrete',noise_std=.02,ar1_rho=0.)],
    hops=[1,128,256,513],splits=dict(calibration_null=199,test_null=200,test_dynamic=200),
    primary=dict(discrete=dict(hop=256,stage='raw')))
records=[]
for split,n in conf['splits'].items():
    for rep in range(n):
        value=1. if split=='test_dynamic' else 0.
        metrics=[]
        for h in conf['hops']:
            for stage in ['raw','fused']:
                metrics.append(dict(hop=h,stage=stage,M=8,T_ms2=value,rmse_ms=value,bias_ms=0.,
                    range_ms=value,E=.5,common_fused_variance_ms2=value))
        records.append(dict(cell_id=0,split=split,replicate=rep,metrics=metrics))
summary,tests,desc=summarize_statistics(records,conf)
assert len(summary)==18 and len(tests)==3600 and len(desc)==24
assert all(r['rate']==(1. if r['endpoint']=='power' else 0.) for r in summary)
assert sum(r['primary'] for r in summary)==2
assert monte_carlo_p([1.,2.,3.],[0.,1.,2.,3.,4.]).tolist()==[1.,1.,.75,.5,.25]
assert binomial_interval(0,200)[0]==0. and binomial_interval(200,200)[1]==1.
assert np.isclose(binomial_interval(0,200)[1],1-.025**(1/200))
try:
    summarize_statistics(records[:-1],conf)
except AssertionError:
    incomplete_rejected=True
else:
    raise AssertionError('Incomplete results accepted')
result=dict(passed=True,synthetic_fixture_only=True,monte_carlo_ties=True,
    independent_split_checks=True,whole_family_statistic=True,incomplete_rejected=incomplete_rejected,
    exact_binomial_edges=True)
(Path(__file__).parent/'baseline/report_selftest.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
