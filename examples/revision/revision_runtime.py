"""Path configuration for staged revision experiments; no scientific operations."""
import os
from pathlib import Path


def resolve_path(key, relative=''):
    workspace = Path(os.environ['NEUROTD_REVISION_WORKSPACE']).resolve()
    if key == 'repo':
        base = Path(os.environ['NEUROTD_REPO']).resolve()
        # Historical baseline locations are now explicit archived inputs.
        if relative.endswith(('simulation_results_discrete.csv', 'simulation_results_continuous.csv')):
            return str(workspace/'baseline_tables'/Path(relative).name)
    elif key == 'raw':
        base = Path(os.environ.get('NEUROTD_RAW_DATA', str(workspace/'external/raw'))).resolve()
    elif key == 'external':
        base = Path(os.environ.get('NEUROTD_EXTERNAL_INPUTS', str(workspace/'external'))).resolve()
    else:
        base = workspace/key
    return str(base/relative)
