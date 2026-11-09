"""Synthetic software integration test; NOT study observations or study evidence."""
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from audit import ROOT, WINDOWS, save_json, save_npz
from summarize import build_report


def main():
    test=Path(tempfile.mkdtemp(prefix='neurotd_SYNTHETIC_software_test_'))
    conf=json.loads((ROOT/'frozen_config.json').read_text())
    conf['nonzero_repeats']=3
    save_json(test/'frozen_config.json',conf)
    save_json(test/'progress.json',{'status':'synthetic_test_only'})
    (test/'baseline').mkdir()
    shutil.copyfile(ROOT/'baseline/comparison.csv',test/'baseline/comparison.csv')
    for model in ['discrete','continuous']:
        for noise in [0.,.02,.1,.2]:
            for rep in range(2 if noise==0 else 3):
                for w in WINDOWS:
                    r=np.arange(1,9,dtype=float)*(1+noise*(rep+1))
                    centers=np.arange(8)
                    p=test/f'checkpoints/{model}/noise{noise:.2f}/rep{rep:04d}_W{w}.npz'
                    save_npz(p,r=r,centers=centers)
                    save_json(p.with_suffix('.json'),dict(model=model,noise_std=noise,
                        replicate_id=rep,window_size=w,hop=1 if model=='continuous' else w//2,
                        E=float(np.sqrt(r.mean())),mean_r=float(r.mean()),rmse_ms=1.,
                        E_abs_diff=0.,min_normalizer=1.,normalizer_at_floor_count=0,
                        x_hash=f'{model}-{noise}-{rep}',y_hash=f'{model}-{noise}-{rep}'))
    build_report(test)
    assert json.loads((test/'progress.json').read_text())['status']=='complete'
    save_json(ROOT/'baseline/report_pipeline_test.json',dict(passed=True,
        type='synthetic software test only; not a scientific experiment',temporary_output=str(test)))
    print('Summary pipeline software test PASSED:',test)


if __name__=='__main__': main()
