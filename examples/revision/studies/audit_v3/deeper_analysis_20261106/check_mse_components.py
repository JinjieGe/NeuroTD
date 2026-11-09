from pathlib import Path
import numpy as np
import pandas as pd
import h5py

O=Path(__file__).resolve().parent; P=O.parent
bp=pd.read_csv(P/'electrodes/bundle_pairs_112.csv')
sh=np.load(P/'checkpoints/bundle_shifts_150x112x508.npz',allow_pickle=False)['shifts']
channels=pd.read_csv(P/'electrodes/channel_verification_table.csv')
labcol=channels[channels.session=='ses-01'].set_index('label').waveform_column.to_dict()
want=[('AHL1','AHL2'),('AL6','AL7'),('AL1','AL8'),('ECL1','ECL2'),('PHR6','PHR7')]
rows=[]; taus=np.arange(-500,501); offsets=np.arange(1001)
windows=np.linspace(0,507,20,dtype=int)
for ses in (1,2,3):
    src=P.parent/f'NeuroTD_section33_reproduction_20261030/data/sub-01/sub-01_ses-0{ses}.nwb'
    with h5py.File(src,'r') as f:
        wav=f['acquisition/ecephys.lfp/data'][:256000,:]
    for la,lb in want:
        a=wav[:,int(labcol[la])]; b=wav[:,int(labcol[lb])]
        pid=int(bp[(bp.label_i==la)&(bp.label_j==lb)].pair_index.iloc[0])
        for k in windows:
            start=500+int(k)*500
            x=a[start:start+1001]; ys=b[start-taus[:,None]+offsets[None,:]]
            ey=np.mean(ys**2,axis=1); cross=-2*np.mean(ys*x,axis=1)
            mse=np.mean(x*x)+ey+cross
            pred=-taus[np.argmin(mse)]; only_energy=-taus[np.argmin(ey)]
            rows.append(dict(pair=f'{la}–{lb}',session=ses,trial=0,window=int(k),
                signal_rms_x=float(np.sqrt(np.mean(x*x))),signal_rms_y=float(np.sqrt(ey[500])),
                mse_energy_corr=float(np.corrcoef(mse,ey)[0,1]),
                energy_sd=float(np.std(ey)),cross_sd=float(np.std(cross)),
                estimate_samples=int(pred),energy_only_estimate_samples=int(only_energy),
                matches_saved=bool(pred==sh[(ses-1)*50,pid,k]),
                energy_min_within2=bool(abs(pred-only_energy)<=2)))
    print('MSE component checks completed session',ses,flush=True)
d=pd.DataFrame(rows);d['energy_to_cross_sd']=d.energy_sd/d.cross_sd
d.to_csv(O/'mse_component_checks_300_windows.csv',index=False)
s=d.groupby('pair').agg(n=('window','size'),median_energy_to_cross_sd=('energy_to_cross_sd','median'),
    median_mse_energy_corr=('mse_energy_corr','median'),energy_min_within2_fraction=('energy_min_within2','mean'),
    matches_saved_fraction=('matches_saved','mean')).reset_index()
s.to_csv(O/'mse_component_summary.csv',index=False)
print(s.to_string(index=False))
