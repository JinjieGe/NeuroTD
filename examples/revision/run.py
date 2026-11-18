"""Stage and run revision supplements without writing into source/evidence folders."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CATALOG = json.loads((HERE/'catalog.json').read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stage(workspace):
    workspace = workspace.resolve()
    if workspace == REPO or HERE == workspace or HERE in workspace.parents:
        raise ValueError('Use a separate output workspace, not a source directory.')
    files = [(p, workspace/p.relative_to(HERE/'studies')) for p in (HERE/'studies').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [(p, workspace/p.relative_to(HERE/'inputs')) for p in (HERE/'inputs').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [(HERE/'revision_runtime.py', workspace/'revision_runtime.py')]
    # Preflight all files before writing anything. Never silently replace a
    # running experiment's code or input files when the checkout changes.
    for source, target in files:
        if target.exists() and sha(source) != sha(target):
            raise ValueError(f'Workspace file differs: {target}. Use a new workspace.')
    for source, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    for name in CATALOG:
        for d in ('results','outputs','figures','deliverables','inputs','checkpoints','benchmarks','logs','pilot','baseline'):
            (workspace/name/d).mkdir(parents=True, exist_ok=True)
    return workspace


def environment(workspace):
    env = os.environ.copy()
    env.update(NEUROTD_REVISION_WORKSPACE=str(workspace), NEUROTD_REPO=str(REPO),
               PYTHONPATH=os.pathsep.join([str(workspace), str(REPO/'src')]),
               MPLBACKEND='Agg', MPLCONFIGDIR=str(workspace/'.matplotlib'),
               PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    return env


def execute(workspace, study, script, args):
    target = (workspace/study/script).resolve()
    if workspace/study not in target.parents or not target.is_file() or target.suffix != '.py':
        raise ValueError('Choose a packaged Python script within the selected study.')
    command=[sys.executable, '-u', str(target), *args]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logdir=workspace/'run_manifests'; logdir.mkdir(exist_ok=True)
    source=REPO/'src/neurotd/time_varying_shift.py'
    record=dict(study=study,script=script,command=command,started_utc=stamp,
                python=sys.version,source_sha256=sha(source),script_sha256=sha(target),
                archived_input_tables=True,full_historical_reproduction_claim=False)
    record['dependencies']={}
    for package in ('numpy','scipy','pandas','matplotlib','tslearn','dtw-python','scikit-image','h5py','numba'):
        try:record['dependencies'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:record['dependencies'][package]='not installed'
    record['staged_python_sha256']={str(p.relative_to(workspace)):sha(p) for p in workspace.rglob('*.py')}
    for key, arguments in [('git_commit',['rev-parse','HEAD']),('git_status',['status','--short'])]:
        try:record[key]=subprocess.check_output(['git','-C',str(REPO),*arguments],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):record[key]='unavailable (e.g. ZIP download)'
    path=logdir/f'{stamp}.json';path.write_text(json.dumps(record,indent=2))
    result=subprocess.run(command,cwd=workspace/study,env=environment(workspace))
    record.update(exit_code=result.returncode,source_unchanged=sha(source)==record['source_sha256'])
    path.write_text(json.dumps(record,indent=2))
    if not record['source_unchanged']:raise RuntimeError('Core source changed during execution.')
    return result.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('list')
    descriptions = {
        'init': 'Prepare a run workspace.',
        'run': 'Run an individual study script.',
        'figures': 'Generate figures from archived outcomes.',
        'evidence': 'Rebuild Tables S1–S9, regional comparisons and matched-control results.',
    }
    for action, help_text in descriptions.items():
        p=sub.add_parser(action, help=help_text)
        p.add_argument('--workspace',type=Path,required=True)
        if action=='run':
            p.add_argument('--study',choices=sorted(CATALOG),required=True)
            p.add_argument('--script',required=True)
            p.add_argument('arguments',nargs=argparse.REMAINDER)
        if action=='figures':
            p.add_argument('--study',choices=['figures_22','figures_23','figures_26','figures_28','figures_13'],required=True)
    args=parser.parse_args()
    if args.action=='list':
        for key,v in CATALOG.items():print(f"{key:22} {','.join(v['comments']):23} {v['title']}")
        return
    if args.action=='figures' and args.workspace.exists():
        parser.error('Archived-figure regeneration requires a new workspace to avoid mixing runs.')
    workspace=stage(args.workspace)
    if args.action=='init':print(f'Ready: {workspace}');return
    if args.action=='figures':
        for source in (HERE/'evidence').rglob('*.csv'):
            study=source.parent.name
            dest=workspace/study/('outputs' if study in ('dynamic_fixed','session_inference') else 'results')/source.name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        (workspace/'ARCHIVED_FIGURES_ONLY.txt').write_text('Plots from archived aggregate tables; NOT a new experiment run.\n')
        script={'figures_22':'build_four_panel.py','figures_23':'build_figure.py','figures_26':'build_figure.py','figures_28':'make_figures.py','figures_13':'make_figures.py'}[args.study]
        raise SystemExit(execute(workspace,args.study,script,[]))
    if (workspace/'ARCHIVED_FIGURES_ONLY.txt').exists():
        parser.error('Use a separate fresh workspace for experiments, not archived-figure outputs.')
    if args.action == 'evidence':
        table_dir = workspace / 'supplement_tables'
        steps = [
            ('score_uncertainty', 'supplement_tables.py', ['--output-dir', str(table_dir)]),
            ('session_inference', 'code/calibrated_welch.py', ['--table-dir', str(table_dir)]),
            ('dynamic_fixed', 'summarize_matched_controls.py', []),
        ]
        for study, script, options in steps:
            result = execute(workspace, study, script, options)
            if result:
                raise SystemExit(result)
        print(f'Supplementary tables S1–S9: {table_dir}')
        print(f"Regional comparisons: {workspace / 'session_inference/outputs'}")
        print(f"Matched controls: {workspace / 'dynamic_fixed/outputs/matched_control_summary.json'}")
        return
    arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    raise SystemExit(execute(workspace,args.study,args.script,arguments))


if __name__=='__main__':main()
