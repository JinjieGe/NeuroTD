"""Build plain-language interpretation from completed, validated outputs only."""
import json,html,hashlib
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
checks=json.loads((P/'results/completion_checks.json').read_text())
assert checks['all_checked_shifts_match'] and checks['noisy_pairs']==253440
s=pd.read_csv(P/'results/summary.csv');n=pd.read_csv(P/'results/noise_only_summary.csv')
g=s[(s['mode']=='both')&(s.delay_samples>=0)&s.snr_db.isin([20,0,-10,-20])]
table=g.pivot(index=['noise','delay_samples'],columns='snr_db',values='mean_abs_ms').reset_index()
table['True delay (ms)']=table.pop('delay_samples')/32
table=table.rename(columns={'noise':'Noise type',20.:'SNR 20 dB',0.:'SNR 0 dB',-10.:'SNR −10 dB',-20.:'SNR −20 dB'})
table=table[['Noise type','True delay (ms)','SNR 20 dB','SNR 0 dB','SNR −10 dB','SNR −20 dB']]
table['Noise type']=table['Noise type'].map({'white':'White noise','ar1':'Colored noise'})
table.to_csv(P/'results/snr_delay_summary.csv',index=False)
paragraphs = [
    "We tested the effect of added noise on delay magnitude while holding the waveform, true shift, window, search range, and E definition fixed.",
    "The table shows independent noise added to both channels. Each entry averages the absolute window delays within a record, then averages over 36 templates and 20 noise realizations per template."
]
for kind, label in [('white', 'White noise'), ('ar1', 'Colored noise')]:
    a = s[(s.noise == kind) & (s['mode'] == 'both') & (s.snr_db == -20) & (s.delay_samples >= 0)]
    z = a[a.delay_samples == 0].iloc[0]
    large = a[a.delay_samples == 400].iloc[0]
    no = n[n.noise == kind].iloc[0]
    paragraphs.append(f"{label} at -20 dB gave mean absolute estimates of {z.mean_abs_ms:.3f} ms for a true shift of 0 ms and {large.mean_abs_ms:.3f} ms for 12.5 ms. The corresponding RMSEs were {z.mean_rmse_ms:.3f} and {large.mean_rmse_ms:.3f} ms.")
    paragraphs.append(f"Across true shifts of 0–12.5 ms, mean absolute estimates ranged from {a.mean_abs_ms.min():.3f} to {a.mean_abs_ms.max():.3f} ms. The noise-only control averaged {no.mean_abs_ms:.3f} ms under the same ±15.625 ms search bounds.")
paragraphs.extend([
    "At lower SNR, estimates depend increasingly on noise and search bounds. Small true delays may be overestimated and large delays underestimated. Read the absolute and signed estimates alongside paired changes and RMSE.",
    "Opposite signed errors can cancel in the mean. Independent noise-only channels share no event with a defined transmission delay.",
    "The results show that lower SNR can change apparent delays in either direction. Regional SNR confounding remains possible in the original recordings.",
    "The shifted template carries its recorded background noise into both channels. Only the added noise is independent. Distinct real channels may have different event waveforms.",
    "Monte Carlo standard errors quantify repeated added noise conditional on these templates. All 36 templates came from nine trials in one participant."
])
answer='Added-noise SNR and estimated delay magnitude\n\n'+'\n\n'.join(paragraphs[:2])+'\n\n'+table.to_string(index=False,float_format=lambda x:f'{x:.3f}')+'\n\n'+'\n\n'.join(paragraphs[2:])
(P/'interpretation.txt').write_text(answer)
page=P/'extended_report.html';content=page.read_text()
section='<section id="interpretation"><h2>Effect of added noise on estimated delay magnitude</h2>'+''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs[:2])+table.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)+''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs[2:])+'</section>'
if '<section id="interpretation">' not in content:
    content=content.replace('<h2>Delay magnitude versus SNR</h2>',section+'<h2>Delay magnitude versus SNR</h2>')
page.write_text(content)
files={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(P.rglob('*')) if f.is_file() and '__pycache__' not in str(f) and f.name!='output_manifest.json'}
(P/'output_manifest.json').write_text(json.dumps(files,ensure_ascii=False,indent=2))
print(answer)
