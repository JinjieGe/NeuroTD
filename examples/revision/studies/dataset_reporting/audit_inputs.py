"""Metadata-only dataset reporting for Comments 2.6/2.8; excludes reliability (2.5).

New integration utility, not a claim to recover the original submission cohort.
Reads trial metadata and the published processed interval table. Does not load
voltage arrays, infer stimulation units, or estimate biological reliability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_runtime import resolve_path


def patchseq_summary(path):
    df=pd.read_csv(path,index_col=0)
    if 'time_delays(ms)' not in df:raise ValueError('Expected published interval table.')
    cells=df.index.astype(str).str.rsplit('-',n=1).str[0]
    count=pd.Series(cells).value_counts()
    lengths=df['time_delays(ms)'].map(lambda s: len(np.fromstring(str(s).strip('[]'),sep=' ')))
    return dict(sweep_rows=len(df),cells=int(count.size),sweeps_per_cell_min=int(count.min()),
                sweeps_per_cell_median=float(count.median()),sweeps_per_cell_max=int(count.max()),
                fewer_than_two_intervals=int((lengths<2).sum()),
                scope='processed interval table; no stimulus-matched reliability claim')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nwb-dir',type=Path)
    ap.add_argument('--interval-table',type=Path,default=Path(resolve_path('repo','data/processed/time_delays_all_cells.csv')))
    a=ap.parse_args();root=Path(__file__).parent;rows=[]
    if a.nwb_dir:
        import h5py
        files=sorted(a.nwb_dir.rglob('*.nwb'))
        if not files:raise FileNotFoundError('No NWB files under supplied directory.')
        for p in files:
            with h5py.File(p,'r') as f:
                g=f.get('intervals/trials')
                if g is None:
                    rows.append(dict(file=p.name,status='no_trial_table'));continue
                n=len(g['id']);duration=g['stop_time'][:]-g['start_time'][:]
                rows.append(dict(file=p.name,subject=p.parent.name,trials=n,
                                 duration_min_s=float(duration.min()),duration_max_s=float(duration.max()),
                                 artifact_flagged=int(np.count_nonzero(g['artifact'][:])) if 'artifact' in g else None,
                                 fields=';'.join(g.keys()),status='metadata_only'))
        pd.DataFrame(rows).to_csv(root/'results/nwb_trial_inventory.csv',index=False)
    result=patchseq_summary(a.interval_table)
    result['input_sha256']=hashlib.sha256(a.interval_table.read_bytes()).hexdigest()
    (root/'results/patchseq_table_inventory.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
