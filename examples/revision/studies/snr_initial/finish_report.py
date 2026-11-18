"""Summarize completed results, add integrity checks and a portable report."""
import json, platform, sys, time, hashlib, html, base64
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, matplotlib, h5py
import run_analysis as r

P=r.ROOT
assert json.loads((P/'progress.json').read_text())['stage']=='complete'
real=pd.read_csv(P/'results/pair_trial_quality_delay.csv')
c=pd.read_csv(P/'results/correlation_summary.csv')
inj=pd.read_csv(P/'results/injection_records.csv')
summ=pd.read_csv(P/'results/injection_summary.csv')
trials=pd.read_csv(P/'results/trial_metadata.csv')
q=pd.read_csv(P/'results/channel_trial_quality.csv')
templates=dict(np.load(P/'checkpoints/templates.npz'))
old=pd.read_csv(r.AUDIT/'delays/versionA_trial_pair_summary_reused.csv')
chk=real.merge(old[['session','trial_pos','pair_index','mean_nonzero_ms']],on=['session','trial_pos','pair_index'],suffixes=('_new','_old'),validate='one_to_one')
maxdiff=float((chk.mean_nonzero_ms_new-chk.mean_nonzero_ms_old).abs().max())
assert maxdiff<1e-10
oldq=pd.read_csv(r.AUDIT/'qc/channel_trial_metrics.csv')
qc=q.merge(oldq[['session','trial_pos','channel','rms']],on=['session','trial_pos','channel'],suffixes=('_new','_old'),validate='one_to_one')
rmsdiff=float(np.max(np.abs(qc.rms_new-qc.rms_old)/(np.abs(qc.rms_old)+1e-30)))
assert rmsdiff<1e-10
assert len(real)==9000 and len(q)==4800 and not real.duplicated(['session','trial_pos','pair_index']).any()
assert len(inj)==11556 and not inj.duplicated(['template','delay_samples','noise','mode','snr_db','replicate']).any()

# Additional production checks specifically include one-channel added noise.
fn=r.original();extra=[]
for name,template in templates.items():
    df=pd.read_csv(P/f'checkpoints/injection_{name}.csv')
    s=np.load(P/f'checkpoints/injection_shifts_{name}.npz')['shifts_samples']
    for row in [80,560]:
        v=df.iloc[row];x,y=r.make_pair(template,int(v.delay_samples))
        rng=np.random.default_rng(int(v.seed))
        nx=r.noise_array(rng,v.noise,len(x));ny=r.noise_array(rng,v.noise,len(y))
        assert v['mode']=='y_only'
        y=y+ny*np.sqrt(np.mean(y*y)/10**(v.snr_db/10))
        _,expected=fn(x,y,r.W,r.H)
        mismatch=int(np.count_nonzero(s[row]!=expected));assert mismatch==0
        extra.append(dict(template=name,row=row,mode=v['mode'],noise=v.noise,snr_db=v.snr_db,n_windows=len(expected),mismatch=mismatch))
pd.DataFrame(extra).to_csv(P/'results/production_y_only_validation.csv',index=False)
env=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,h5py=h5py.__version__)
(P/'provenance/environment.json').write_text(json.dumps(env,indent=2))
checks=dict(pair_trial_rows=len(real),channel_trial_rows=len(q),constructed_pairs=len(inj),baseline_pairs=int((inj.noise=='none').sum()),noisy_pairs=int((inj.noise!='none').sum()),legacy_nonzero_mean_max_difference_ms=maxdiff,legacy_RMS_max_relative_difference=rmsdiff,prepare_records_checked=16,production_records_checked=84,extra_one_channel_records_checked=24,all_checked_shifts_equal_original=True,original_source_unchanged=r.sha(r.SOURCE)==pd.read_csv(P/'provenance/analysis_inputs.csv').iloc[0].sha256)
assert checks['original_source_unchanged']
(P/'results/completion_checks.json').write_text(json.dumps(checks,indent=2))

clean=c[(c.subset=='clean')&(c.outcome=='mean_abs_ms')].copy()
selection=summ[(summ.noise!='none')&(summ['mode']=='both')&(summ.snr_db.isin([20,-10]))].copy()
selection['true_delay_ms']=selection.delay_samples/32
selected=selection[['true_delay_ms','noise','snr_db','mean_abs_ms','mean_rmse_ms','mean_signed_bias_ms','records']].sort_values(['true_delay_ms','noise','snr_db'],ascending=[True,True,False])
zero=selection[(selection.delay_samples==0)&(selection.snr_db==-10)]
baseline=inj[inj.noise=='none']
base_max=float(baseline.rmse_ms.max())
quality_medians=clean[['metric','same_bundle','n_nonconstant','median_rho','q25','q75']]
quality_medians.to_csv(P/'results/report_correlation_table.csv',index=False)
selected.to_csv(P/'results/report_noise_table.csv',index=False)

scope=trials.groupby('session').agg(total_trials=('trial_pos','size'),artifact_trials=('artifact','sum'),correct_trials=('correct','sum')).reset_index()
scope['clean_trials']=scope.total_trials-scope.artifact_trials
scope.to_csv(P/'results/report_scope_table.csv',index=False)
template_rows=[]
for name,tpl in templates.items():
    sess,bundle=name.split('_');eligible=trials[(trials.session==sess)&~trials.artifact&trials.correct]
    template_rows.append(dict(template=name,trial_pos=int(eligible.trial_pos.iloc[0]),trial_id=int(eligible.trial_id.iloc[0]),channel={'AHL':0,'AL':8,'ECL':16,'PHR':24}[bundle],relative_trial_start_s=1,signal_duration_s=1,template_sha256=hashlib.sha256(tpl.tobytes()).hexdigest()))
pd.DataFrame(template_rows).to_csv(P/'provenance/template_manifest.csv',index=False)

zero_text = ". ".join(
    f"{row.noise} noise gave a mean absolute delay of {row.mean_abs_ms:.3f} ms "
    f"and a delay RMSE of {row.mean_rmse_ms:.3f} ms" for row in zero.itertuples()
)
text=f'''NeuroTD signal quality and delay estimates
Reviewer 2.7 | {time.strftime('%Y-%m-%d')}

Data and estimator checks
DANDI dataset version 0.250610.0938 identifies three NWB assets for sub-01, ses-01 through ses-03. Checks of official file sizes and SHA-256 hashes allowed reuse of approximately 4.1 GB of recordings. The study covers 150 eight-second trials and 32 channels. Removing 16 trials marked as artifacts leaves 134 trials. Sensitivity summaries cover all 150 trials and the 124 artifact-free trials with correct responses.
The analysis reused 150 x 60 x 508 signed window estimates, generated 9,000 pair-by-trial summaries, and retained valid zero delays in the primary analysis. The 11,520 added-noise pairs and 36 no-added-noise baselines give 11,556 constructed pairs. On 124 full records checked window by window, the shifts matched the original function. The largest difference from the archived nonzero mean-delay summaries was {maxdiff:.3g} ms.
Linear FFT convolution accelerates the cross-product in the original MSE objective. Each candidate retains its full energy term, shift convention, and search bounds. Equivalence was checked on the specified records.

Observational analysis
The four prespecified proxies were 50 Hz and 60 Hz spectral prominence, the high-frequency power fraction, and the RMS amplitude imbalance between channels. Line and high-frequency power may include neural activity. Direct physiological SNR is unknown.
Welch spectra used one-second Hann segments with 50% overlap within each eight-second recording. Line prominence was 10 log10(mean power within 1 Hz of the center / mean power 2–5 Hz from the center). The high-frequency fraction used 1,000–10,000 Hz power divided by 1–10,000 Hz power. Each pair used the larger line prominence, the average high-frequency fraction, and abs(log2(RMS_x/RMS_y)). The delay outcome was the mean absolute delay across 508 valid windows per trial.
Spearman correlations were computed across trials within each session, task load, and electrode pair. At least ten trials and variation in the predictor were required. The 12 within-bundle and 48 between-bundle pairs are shown separately. Regional labels are retained only for descriptive sensitivity checks because anatomical mapping remains uncertain.

Correlations in the primary analysis
{quality_medians.to_string(index=False,float_format=lambda x:f'{x:.3f}')}
The median 60 Hz correlations were about 0.012 within bundles and 0.032 between bundles. For RMS imbalance they were about 0.273 and 0.131, respectively. The high-frequency fraction had negative median correlations. The four proxies showed different patterns and leave regional noise confounding unresolved.

Known-delay experiment
Each of three sessions and four bundles contributed one template. The first artifact-free correct trial supplied a one-second segment from 1–2 s after trial onset. Original margins allowed shifts without wrapping or zero padding. The template retains background activity from the recording.
The generating model was y(t)=x(t-d) with d of 0, 80, or 320 samples (0, 2.5, or 10 ms). Settings were a 1,001-sample rectangular window, hop of 500, search range of ±500 samples (±15.625 ms), unit gain, and fixed window size. The original E was retained.
White or AR(1) noise (rho=0.95) was added to y alone or independently to both channels at 20, 10, 0, and -10 dB, with 20 realizations per condition. Noise power was scaled to template power and checked per record. Paired SNR comparisons reused the same standardized noise realization. The independent unit was a complete constructed pair.
Maximum RMSE across the 36 baselines was {base_max:.6g} ms. With true delay zero, both channels noisy, and SNR of -10 dB, {zero_text}.

Selected conditions (12 templates x 20 realizations per row)
{selected.to_string(index=False,float_format=lambda x:f'{x:.3f}')}

Interpretation
Report mean absolute delay, delay RMSE, and signed bias separately. Noise can impair recovery without increasing every estimated delay magnitude. One participant, uncertain regional labels, and the absence of direct SNR measurements limit inference about hippocampal timing, correspondence between real waveforms, and transmission direction.

Reading the outputs
Figure 1 shows trial-level quality correlations by session, load, and electrode pair. The box plots describe these coefficients. Figure 2 compares mean absolute estimates with the true magnitude (dashed line) and delay RMSE below. Noise increases from left to right. Each condition averages 12 templates and 20 noise realizations. Figure 3 shows individual templates with a true delay of zero. The between-template spread is descriptive.
The results directory contains numerical outputs, sensitivity analyses, and validation records. Checkpoints hold templates, estimated shifts, and batches. The provenance directory records sources, hashes, paths, environment, and the template list. The prespecified design is in analysis_plan.json. In the staged workspace run run_analysis.py prepare, run_analysis.py run, then finish_report.py. Completed batches are reused.
'''
(P/'results_summary.txt').write_text(text)

def table(df):return df.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)
def picture(name):
    data=base64.b64encode((P/f'figures/{name}.png').read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{name}">'
parts=[f'<h1>How does signal quality affect delay estimates?</h1><p class="sub">NeuroTD · Reviewer 2.7 · Signal-quality study · {time.strftime("%Y-%m-%d")}</p>',
       '<div class="cards"><div><b>3</b>verified NWB sessions</div><div><b>4,800</b>channel-by-trial records</div><div><b>11,556</b>constructed signal pairs</div><div><b>124</b>records checked against the original estimator</div></div>',
       '<p class="note">Results cover one participant. Observed recordings use quality proxies. Constructed signals have known shifts and added-noise levels.</p>',
       '<h2>Data and validation</h2>',table(scope),
       '<h2>Quality and delay in observed recordings</h2>',picture('01_real_quality_correlations'),table(quality_medians),
       '<h2>Delay estimates under controlled added noise</h2>',picture('02_controlled_noise'),table(selected),
       '<h2>Sensitivity across templates</h2>',picture('03_template_sensitivity'),
       '<h2>Methods and interpretation</h2><div class="full">'+html.escape(text).replace('\n','<br>')+'</div>']
page='<!doctype html><html lang="en"><meta charset="utf-8"><title>NeuroTD signal quality and delay estimates</title><style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:1150px;margin:40px auto;color:#203047;line-height:1.7;padding:0 24px}h1{font-size:32px}h2{border-bottom:2px solid #c9dbe7;padding-top:24px}.sub{color:#617080}.cards{display:flex;gap:16px}.cards div{background:#edf4f8;border-radius:12px;padding:18px;flex:1}.cards b{display:block;font-size:30px;color:#006a96}.note{border-left:4px solid #d89239;background:#fff7ea;padding:16px}img{max-width:100%}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#e8eff5}tr:nth-child(even){background:#f7f9fb}.full{font-size:14px;overflow-wrap:anywhere}@media print{body{margin:0}img{break-inside:avoid}h2{break-after:avoid}}</style><body>'+''.join(parts)+'</body></html>'
(P/'report.html').write_text(page)
print(json.dumps(checks,ensure_ascii=False,indent=2));print(text)
