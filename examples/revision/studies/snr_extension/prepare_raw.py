"""Verify the three versioned public NWB inputs without requiring old delay checkpoints."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request
import pandas as pd
from revision_runtime import resolve_path
sys.path.insert(0,resolve_path('snr_initial'))
import run_analysis as old


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true',help='Explicitly allow large public NWB downloads if missing.')
    args=p.parse_args();rows=[]
    for session,asset in zip(old.SESSIONS,old.ASSETS):
        with urllib.request.urlopen(f'https://api.dandiarchive.org/api/assets/{asset}/',timeout=60) as response:
            meta=json.load(response)
        path=old.RAW/f'sub-01_{session}.nwb';expected=meta['digest']['dandi:sha2-256']
        if not path.exists():
            if not args.download:raise FileNotFoundError(f'{path}; supply NEUROTD_RAW_DATA or explicitly use --download.')
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.part')
            urllib.request.urlretrieve(f'https://api.dandiarchive.org/api/assets/{asset}/download/',tmp)
            if old.sha(tmp)!=expected:raise ValueError(f'Checksum mismatch: {tmp}')
            tmp.rename(path)
        actual=old.sha(path)
        if actual!=expected or path.stat().st_size!=meta['contentSize']:raise ValueError(f'Input mismatch: {path}')
        rows.append(dict(session=session,path=str(path),bytes=path.stat().st_size,sha256=actual,official_sha256=expected,asset_id=asset))
    out=Path(resolve_path('snr_initial','provenance'));out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'raw_files.csv',index=False)
    print('Verified three named inputs. Next run extend_experiment.py prepare.')


if __name__=='__main__':main()
