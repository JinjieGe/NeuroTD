"""Small scientific/portability regressions. These are not manuscript experiments."""
import ast
import csv
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

    def test_revised_tables_and_matched_controls(self):
        with tempfile.TemporaryDirectory(prefix='neurotd_supplement_test_') as tmp:
            work = Path(tmp) / 'stage'
            output = work / 'supplement_tables'
            subprocess.run([sys.executable, str(BUNDLE / 'run.py'), 'evidence',
                            '--workspace', str(work)], check=True)
            tables = {}
            for number, count in {1: 66, 2: 66, 3: 6, 4: 4, 5: 36, 6: 6, 7: 3, 8: 4, 9: 6}.items():
                with (output / f'Table_S{number}.csv').open(encoding='utf-8', newline='') as source:
                    header, *rows = csv.reader(source)
                self.assertEqual(len(rows), count)
                self.assertTrue(all(len(row) == len(header) for row in rows))
                tables[number] = rows

            baseline = {r[1].split()[0]: r[3] for r in tables[1] if float(r[0]) == 0}
            self.assertEqual((baseline['513'], baseline['1025']), ('0.000†', '70.849†'))
            self.assertEqual(sum(cell.endswith('†') for row in tables[1] for cell in row), 6)
            self.assertEqual(sum(int(r[-1].split('/')[1]) for r in tables[3]), 1200)
            self.assertIn(['Discrete', '0.20', '16.712', '0.448', '16.264', '153/200'], tables[3])
            self.assertEqual(tables[7], [['0.05', '4.635', '4.9355', '30/30'],
                                         ['0.01', '0.745', '0.8741', '30/30'],
                                         ['0.001', '0.025', '0.0583', '30/30']])
            self.assertEqual(tables[6][0], ['HIP–AMY', '0.778', '0.0269', '7.34',
                                            '[0.682, 0.874]', '<0.001'])
            self.assertIn(['12.500', '−20', '8.669', '15.661'], tables[8])
            self.assertIn(['Hippocampus–Middle Temporal Gyrus', '1 / 3', '9 / 37'], tables[9])

            summary = json.loads((work / 'dynamic_fixed/outputs/matched_control_summary.json').read_text())
            self.assertEqual(summary['holm_family_size'], 72)
            scenarios = {row['scenario']: row for row in summary['scenarios']}
            self.assertEqual(len(scenarios), 7)
            counts = [row['panels_with_any_false_rejection'] for row in scenarios.values()]
            self.assertEqual((min(counts), max(counts)), (5, 14))
            white = scenarios['stationary_white']
            self.assertEqual(white['constant_panels'], 200)
            self.assertAlmostEqual(white['mean_dynamic_record_rmse_ms'], 0.15219838438)
            self.assertAlmostEqual(white['mean_dynamic_record_detection_fraction'], 7188 / 7200)
            self.assertGreater(scenarios['repeated_waveform']['mean_dynamic_record_rmse_ms'], 11)

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
