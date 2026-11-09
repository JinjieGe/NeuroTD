"""Finalize the completed local review and hashes, without changing scientific code."""
from revision_runtime import resolve_path
from pathlib import Path
import hashlib
import json
import time
ROOT=Path(resolve_path('dynamic_fixed', ''))
OUT=ROOT/'outputs'
PROJECT=Path(resolve_path('repo', ''))
conf=json.loads((ROOT/'frozen_config.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert {str(p.relative_to(PROJECT)):sha(p) for p in sorted((PROJECT/'src').rglob('*.py'))}==conf['all_source_hashes']
assert all(sha(ROOT/k)==v for k,v in conf['harness_hashes'].items())
audit=json.loads((OUT/'independent_review.json').read_text())
assert audit['passed'] and audit['p_values_recomputed']==25200
required=['analysis_zh.md','response_zh.md','response_en.md','methods_en.md',
    '01_error_rates_and_power.png','02_null_variability_by_hop.png','03_example_curves.png','04_delay_accuracy.png']
assert all((OUT/n).is_file() and (OUT/n).stat().st_size>0 for n in required)
old=json.loads((ROOT/'review_archive/automatic_report/sha256.json').read_text())
unchanged=['repeat_metrics.csv','test_summary.csv','whole_curve_tests.csv','delay_and_E_summary.csv','verified_facts.json']
assert all(sha(OUT/n)==old[n] for n in unchanged)
review={'status':'complete','reviewed_epoch':time.time(),'scientific_design_unchanged':True,
    'original_project_code_unchanged':True,'frozen_simulation_and_report_scripts_unchanged':True,
    'manuscript_edited':False,'statistical_csvs_unchanged':unchanged,
    'numeric_audit':'independent_review.json','visual_review':{
        '01_error_rates_and_power.png':'Reviewed final image: FPR zoom, percent axes, legend below data; no occlusion.',
        '02_null_variability_by_hop.png':'Reviewed original image: seven conditions, native grids explicitly distinguished, panel scales differ.',
        '03_example_curves.png':'Reviewed final image: no misleading floating-point offset on constant curve; sparse centers marked; same replicate 0.',
        '04_delay_accuracy.png':'Reviewed final image: readable labels, shared RMSE scale, values agree with summaries.'},
    'drafts_reviewed':['analysis_zh.md','response_zh.md','response_en.md','methods_en.md'],
    'limitations_explicit':['fixed W=513','known waveform/lag/noise null','single calibration set',
        '9.5% conditional FPR','low high-noise power','low power for some unstandardized max tests',
        'smoothness does not guarantee accuracy','no real-data or selected-window calibration'],
    'review_support_hashes':{n:sha(ROOT/n) for n in ['review_results.py','review_figures.py','finalize_review.py']}}
(OUT/'review_notes.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
manifest={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='sha256.json'}
(OUT/'sha256.json').write_text(json.dumps(manifest,indent=2))
assert all(sha(OUT/k)==v for k,v in manifest.items())
p=json.loads((ROOT/'progress.json').read_text())
assert p['completed']==p['total']==4193 and p['status'] in ['complete_pending_visual_review','complete']
p.update(status='complete',reviewed_epoch=review['reviewed_epoch'],updated_epoch=time.time(),
    source_unchanged=True,visual_review_complete=True,independent_review_passed=True,
    manuscript_edited=False,deliverables=['outputs/'+n for n in required])
(ROOT/'progress.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))
print(json.dumps({'status':p['status'],'verified_output_files':len(manifest),
    'independent_review_passed':True,'source_unchanged':True,'manuscript_edited':False},indent=2))
