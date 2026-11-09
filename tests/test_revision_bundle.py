"""Small scientific/portability regressions. These are not manuscript experiments."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'examples/revision'


class RevisionBundleTests(unittest.TestCase):
    def test_all_python_syntax_and_scope(self):
        for p in BUNDLE.rglob('*.py'):ast.parse(p.read_text())
        catalog=json.loads((BUNDLE/'catalog.json').read_text())
        self.assertFalse(any('2.5' in v['comments'] for v in catalog.values()))

    def test_staged_scientific_checks(self):
        spec=importlib.util.spec_from_file_location('revision_launcher',BUNDLE/'run.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory(prefix='neurotd_revision_test_') as tmp:
            work=launcher.stage(Path(tmp));env=launcher.environment(work)
            code='''
import sys, numpy as np
from pathlib import Path
from revision_runtime import resolve_path
sys.path.insert(0,resolve_path('score_uncertainty'))
import audit
audit.score_tests()
for model in ['discrete','continuous']:
    s=audit.clean_signals(model)
    row,arrays=audit.estimate(model,s,s['x'],s['y'],65)
    assert np.isfinite(row['E']) and row['E_abs_diff'] < 1e-10
sys.path.insert(0,resolve_path('dynamic_selection','code'))
import science
assert science.expected_task_count()==29379
sys.path.insert(0,resolve_path('snr_initial'))
import run_analysis as snr
rng=np.random.default_rng(772)
x=rng.normal(size=2400);y=np.roll(x,30)
c,d,_=snr.estimate(x,y)
co,do=snr.original()(x,y,snr.W,snr.H)
assert np.array_equal(c,co) and np.array_equal(d,do)
assert np.all(d==30)
sys.path.insert(0,resolve_path('dataset_reporting'))
from audit_inputs import patchseq_summary
p=patchseq_summary(Path(resolve_path('repo','data/processed/time_delays_all_cells.csv')))
assert p['sweep_rows']==10637 and p['cells']==1294
print('PASS: E identity, two clean models, task count, FFT/direct shift equivalence, processed-table counts')
'''
            (work/'score_uncertainty/baseline').mkdir(exist_ok=True)
            subprocess.run([sys.executable,'-c',code],env=env,cwd=work,check=True)
            # Changing an existing staged script must be caught before rerun.
            target=work/'score_uncertainty/audit.py';target.write_text(target.read_text()+'\n# changed\n')
            with self.assertRaises(ValueError):launcher.stage(work)


if __name__=='__main__':unittest.main()
