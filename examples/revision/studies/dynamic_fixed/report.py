"""Whole-recording Monte Carlo inference and English reports."""
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
from scipy.stats import beta

def monte_carlo_p(calibration, observed):
    c=np.asarray(calibration,dtype=float)
    o=np.atleast_1d(observed).astype(float)
    assert len(c)>0 and np.isfinite(c).all() and np.isfinite(o).all()
    return (1+np.sum(c[:,None]>=o[None,:],axis=0))/(len(c)+1)

def binomial_interval(k,n):
    assert 0<=k<=n and n>0
    return (0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k)))

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader();w.writerows(rows)

def summarize_statistics(records,conf):
    summaries=[];tests=[];descriptive=[]
    for cid,cell in enumerate(conf['cells']):
        groups={split:sorted([r for r in records if r['cell_id']==cid and r['split']==split],key=lambda r:r['replicate'])
            for split in conf['splits']}
        for split,n in conf['splits'].items():
            assert len(groups[split])==n
            assert [r['replicate'] for r in groups[split]]==list(range(n))
        variants=[(h,s) for h in conf['hops'] for s in ['raw','fused']]+[(0,'family_max')]
        def values(split,h,stage):
            if stage=='family_max': return np.array([max(q['T_ms2'] for q in r['metrics']) for r in groups[split]])
            return np.array([next(q['T_ms2'] for q in r['metrics'] if q['hop']==h and q['stage']==stage) for r in groups[split]])
        for h,stage in variants:
            cal=values('calibration_null',h,stage)
            for split,label in [('test_null','false_positive_rate'),('test_dynamic','power')]:
                observed=values(split,h,stage)
                p=monte_carlo_p(cal,observed)
                k=int((p<=.05).sum());n=len(p)
                lo,hi=binomial_interval(k,n)
                primary=conf['primary'][cell['model']]
                summaries.append(dict(cell_id=cid,**cell,hop=h,stage=stage,endpoint=label,
                    primary=(h==primary['hop'] and stage==primary['stage']),n=n,rejections=k,rate=k/n,
                    ci95_low=lo,ci95_high=hi,calibration_n=len(cal),
                    calibration_T_q95=float(np.quantile(cal,.95)),test_T_mean=float(observed.mean())))
                for r,t,pv in zip(groups[split],observed,p):
                    tests.append(dict(cell_id=cid,**cell,split=split,replicate=r['replicate'],hop=h,stage=stage,
                        T_ms2=float(t),p_value=float(pv),reject=bool(pv<=.05)))
        for split,rs in groups.items():
            for h,stage in variants[:-1]:
                qs=[next(q for q in r['metrics'] if q['hop']==h and q['stage']==stage) for r in rs]
                d=dict(cell_id=cid,**cell,split=split,hop=h,stage=stage,n=len(qs),M=qs[0]['M'])
                for field in ['T_ms2','rmse_ms','bias_ms','range_ms','E']:
                    a=np.array([q[field] for q in qs])
                    d.update({field+'_mean':float(a.mean()),field+'_median':float(np.median(a)),
                        field+'_q025':float(np.quantile(a,.025)),field+'_q975':float(np.quantile(a,.975))})
                if stage=='fused':
                    d['common_fused_variance_ms2_mean']=float(np.mean([q['common_fused_variance_ms2'] for q in qs]))
                descriptive.append(d)
    return summaries,tests,descriptive

def build_report(root):
    root=Path(root)
    conf=json.loads((root/'frozen_config.json').read_text())
    expected=len(conf['cells'])*sum(conf['splits'].values())
    paths=sorted((root/'checkpoints').rglob('rep*.json'))
    assert len(paths)==expected,f'Incomplete run {len(paths)}/{expected}; no final report generated'
    records=[];flat=[]
    for path in paths:
        r=json.loads(path.read_text())
        assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()==r['arrays_sha256']
        assert len(r['metrics'])==8
        assert len({(q['hop'],q['stage']) for q in r['metrics']})==8
        for q in r['metrics']:
            assert all(np.isfinite(q[k]) for k in ['T_ms2','rmse_ms','bias_ms','E'])
            flat.append({k:r[k] for k in ['cell_id','model','noise_std','ar1_rho','split','replicate']}|q)
        records.append(r)
    # Verify identity, input hashes and frozen protocol at end, not only launch.
    from simulate import config,source_hashes,PROJECT
    assert config()==conf
    summaries,tests,desc=summarize_statistics(records,conf)
    out=root/'outputs';out.mkdir(exist_ok=True)
    write_csv(out/'whole_curve_tests.csv',tests)
    write_csv(out/'test_summary.csv',summaries)
    write_csv(out/'delay_and_E_summary.csv',desc)
    write_csv(out/'repeat_metrics.csv',flat)
    primary=[s for s in summaries if s['primary']]
    baseline=json.loads((root/'baseline/rechecked_24.json').read_text())
    direct=json.loads((root/'baseline/direct_checks.json').read_text())
    assert len(baseline)==24 and len(direct)==4
    assert all(r['all_hops_raw_and_fused_exact'] and r['original_baseline_shifts_exact'] for r in direct)
    zero=[json.loads(p.read_text()) for p in sorted((root/'checks').glob('*_constant.json'))]
    assert len(zero)==2
    facts=dict(independent_signal_pairs=expected,derived_variants=8*expected,
        source_unchanged=True,source_hashes=source_hashes(),baseline_conditions=24,
        direct_original_conditions=4,all_sparse_and_fusion_checks_exact=True,
        E_audit_max_abs_diff=max(r['E_audit_abs_diff'] for r in flat),
        noiseless_constant_max_T_ms2=max(q['T_ms2'] for r in zero for q in r['metrics']),
        primary_tests=primary,calibration_repeats=199,held_out_null_repeats=200,dynamic_repeats=200)
    (out/'verified_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False))
    make_plots(root,conf,records,summaries,desc)
    write_prose(out,conf,facts,summaries,desc)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='sha256.json'}
    (out/'sha256.json').write_text(json.dumps(manifest,indent=2))
    p=json.loads((root/'progress.json').read_text())
    p.update(status='complete_pending_visual_review',completed=expected,total=expected,
        source_unchanged=True,finished_epoch=time.time())
    (root/'progress.json').write_text(json.dumps(p,indent=2))
    print(json.dumps({k:v for k,v in facts.items() if k not in ['source_hashes','primary_tests']}),flush=True)

def make_plots(root,conf,records,summaries,desc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'outputs'
    plt.rcParams.update({'font.size':9,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
    labels=[f"{c['model']}\nSD={c['noise_std']:.2f}, rho={c['ar1_rho']:g}" for c in conf['cells']]
    for ax,endpoint in zip(axes,['false_positive_rate','power']):
        for offset,stage,color in [(-.16,'primary','#2166ac'),(.16,'family_max','#b35806')]:
            rs=[next(r for r in summaries if r['cell_id']==cid and r['endpoint']==endpoint and
                (r['primary'] if stage=='primary' else r['stage']=='family_max')) for cid in range(len(labels))]
            rate=np.array([r['rate'] for r in rs])
            err=np.array([[r['rate']-r['ci95_low'] for r in rs],[r['ci95_high']-r['rate'] for r in rs]])
            ax.errorbar(np.arange(len(labels))+offset,rate,yerr=err,fmt='o',capsize=3,color=color,label=stage)
        if endpoint=='false_positive_rate': ax.axhline(.05,color='grey',ls='--',lw=1,label='Nominal 0.05')
        ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel=endpoint.replace('_',' '),ylim=(-.02,1.02))
        ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right',fontsize=8)
    fig.suptitle('Held-out whole-curve inference; 200 records per endpoint\n95% exact binomial intervals conditional on the realized calibration set')
    fig.savefig(out/'01_error_rates_and_power.png');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
    for cid,cell in enumerate(conf['cells']):
        ax=axes.flat[cid]
        for stage,color in [('raw','#2166ac'),('fused','#b35806')]:
            rs=[next(d for d in desc if d['cell_id']==cid and d['split']=='test_null' and d['stage']==stage and d['hop']==h) for h in conf['hops']]
            ax.plot(range(4),[d['T_ms2_mean'] for d in rs],'-o',color=color,label=stage)
        ax.set(title=f"{cell['model']}; SD={cell['noise_std']:g}, rho={cell['ar1_rho']:g}",
            xticks=range(4),xticklabels=conf['hops'],xlabel='Hop (samples)',ylabel='Mean null variance (ms²)')
        ax.grid(alpha=.2)
    axes.flat[0].legend();axes.flat[7].axis('off')
    fig.suptitle('Constant true delay: apparent variation on native returned centers\nHop changes center sampling as well as dependence; these are not matched-grid raw variances')
    fig.savefig(out/'02_null_variability_by_hop.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for i,model in enumerate(['discrete','continuous']):
        cid=next(i for i,c in enumerate(conf['cells']) if c['model']==model and c['noise_std']==.02 and c['ar1_rho']==0)
        for j,split in enumerate(['test_null','test_dynamic']):
            ax=axes[i,j]
            with np.load(root/f'checkpoints/cell{cid}/{split}/rep0000.npz') as z:
                c=z['centers_H1'];t=c/conf['fs']
                ax.plot(t,z['truth_H1']*1000/conf['fs'],color='black',label='Generating delay')
                ax.plot(t,z['raw_H1']*1000/conf['fs'],color='#8c8c8c',lw=.55,alpha=.6,label='Raw H=1')
                for h,color in [(1,'#2166ac'),(256,'#1b9e77'),(513,'#b35806')]:
                    ax.plot(z[f'centers_H{h}']/conf['fs'],z[f'fused_H{h}']*1000/conf['fs'],lw=1,label=f'Fused H={h}',color=color)
            ax.set(title=f'{model}; {split}; SD=0.02',xlabel='Time (s)',ylabel='Delay (ms)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Preselected examples: replicate 0, not selected for appearance')
    fig.savefig(out/'03_example_curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,split in zip(axes,['test_null','test_dynamic']):
        vals=[]
        for cid,cell in enumerate(conf['cells']):
            p=conf['primary'][cell['model']]
            d=next(d for d in desc if d['cell_id']==cid and d['split']==split and d['hop']==p['hop'] and d['stage']==p['stage'])
            vals.append(d['rmse_ms_mean'])
        ax.bar(range(len(labels)),vals,color='#2166ac')
        ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel='Mean delay RMSE (ms)',title=split)
        ax.tick_params(axis='x',labelsize=7);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Original default mode at fixed W=513; stability is not accuracy')
    fig.savefig(out/'04_delay_accuracy.png');plt.close(fig)

def write_prose(out,conf,facts,summaries,desc):
    def rate_text(r): return f"{r['rejections']}/{r['n']} ({100*r['rate']:.1f}%; 95% CI {100*r['ci95_low']:.1f}–{100*r['ci95_high']:.1f}%)"
    lines=[];details=[]
    for cid,c in enumerate(conf['cells']):
        f=next(r for r in summaries if r['cell_id']==cid and r['primary'] and r['endpoint']=='false_positive_rate')
        p=next(r for r in summaries if r['cell_id']==cid and r['primary'] and r['endpoint']=='power')
        setting=conf['primary'][c['model']]
        d=next(r for r in desc if r['cell_id']==cid and r['split']=='test_null' and r['hop']==setting['hop'] and r['stage']==setting['stage'])
        lines.append(f"| {c['model']} | {c['noise_std']:.2f} | {c['ar1_rho']:g} | {rate_text(f)} | {rate_text(p)} | {d['rmse_ms_mean']:.4f} |")
        details.append(f"{c['model']}, noise SD {c['noise_std']:.2f}, AR(1) rho {c['ar1_rho']:g}: false positives {rate_text(f)}; power {rate_text(p)}.")
    table='| Model | Noise SD | AR rho | False positive rate | Power | Mean null RMSE (ms) |\n|---|---:|---:|---|---|---:|\n'+'\n'.join(lines)
    fprs=[r for r in summaries if r['primary'] and r['endpoint']=='false_positive_rate']
    inconsistent=[r for r in fprs if r['ci95_low']>.05 or r['ci95_high']<.05]
    interp = (f"The conditional binomial intervals excluded 5% in {len(inconsistent)} primary conditions."
              if inconsistent else "The conditional binomial intervals included 5% in every primary condition.")
    report=f'''# Fixed-window delay dynamics under a constant-delay null

The original estimator, alignment score E, search range, weights and reconstruction formula were used throughout. Each of {facts['independent_signal_pairs']} independently generated noisy signal pairs was analyzed under eight paired hop and reconstruction settings.

## Primary results

{table}

{interp} These intervals condition on one finite calibration library.

The maximum noiseless constant-delay curve variance was {facts['noiseless_constant_max_T_ms2']:.4g} ms². This checks recovery of a constant delay under the tested waveforms, delay grid and search bounds. Other signals may admit multiple equally good matches.

## Design and verification

For the 24 benchmarks from 2.4.1, recomputed E and delay RMSE were compared with the original CSV. Before the full experiment, both signal models and two noise levels were checked at W=513 against direct estimator calls. Raw shifts and reconstructions at all four hops matched point by point. Across full runs, the maximum absolute difference in the independent E check was {facts['E_audit_max_abs_diff']:.4g}.

All signals lasted 1 s at 5120 Hz. The window size was fixed at W=513. Hops of 1, 128, 256 and 513 samples gave nominal overlaps of 99.8%, 75.0%, 50.1% and 0%. The primary discrete configuration used rectangular weights, H=256 and raw shifts. The primary continuous configuration used Hann weights, H=1 and overlap-add reconstruction. Minimum-E window selection was outside this fixed-window calibration.

Constant delays were 12 samples for the discrete model and 0 samples for the continuous model. The null y was an exact circular shift of its clean reference x. The dynamic positive controls used the original clean y and original generating delays. Independent white noise was added to x and y with SD 0.02, 0.10 or 0.20. A further continuous-model condition used independent stationary AR(1) noise with rho=0.8 and marginal SD 0.02.

Each condition used 199 constant-delay calibration pairs, 200 independent constant-delay test pairs and 200 independent dynamic test pairs. T is the temporal variance of the estimated delay on its native output centers (ms²). The Monte Carlo p-value is (1 + number of calibration T values at least as large as the test T)/200. The null is rejected at p ≤ 0.05. Whole records retain within-record dependence and ties count in the numerator. The maximum T across all eight settings was calibrated separately.

## Hop and reconstruction

A fixed center with the same data, weights and W yields the same raw local estimate for every hop. Centers for larger hops were drawn from the full H=1 grid. Direct calls verified this reuse. Hop changes the number and placement of sampled centers, so differences in variance also reflect sampling. Normalized overlap-add retains the original boundary weights. Reconstruction variability on the common valid interval is reported separately from the native-grid statistic. The common grid excludes zero-weight Hann endpoints.

## Interpretation

The Clopper–Pearson intervals condition on the sampled library of 199 calibration records. They omit variation from sampling a new calibration threshold. The p-value resolution is 0.005. These tests characterize specified clean waveforms and known null parameters. Testing a real recording with unknown lag and noise would require a corresponding reference model. Inference following selection or filtering would also require reproducing those steps on the null data.

Delay RMSE and bias accompany the variance. A flat estimate can have a large delay error, and a failure to detect dynamics does not establish a constant delay. The continuous model uses the original forward generating delay, Δ(t). Window-level between-group Welch comparisons are a separate analysis.

Leonardi and Van De Ville showed that mismatched window and signal timescales can produce apparent fluctuations in sliding correlations. The direction of a hop effect in NeuroTD requires direct measurement. See Leonardi and Van De Ville, NeuroImage 104 (2015), 430–436, https://doi.org/10.1016/j.neuroimage.2014.09.007.

## Figures and output

Figure 1 gives held-out false-positive rates and power with conditional 95% binomial intervals. Figure 2 gives mean apparent variance under the null, separately for raw and reconstructed curves. Figure 3 shows prespecified replicate 0. Figure 4 gives mean delay RMSE at the original default mode and fixed W.

test_summary.csv lists primary and sensitivity tests. whole_curve_tests.csv stores record-level p-values. delay_and_E_summary.csv includes variance, RMSE, bias and E. repeat_metrics.csv retains all paired configurations. The run directory contains random seeds, unaggregated curves and source checksums.
'''
    (out/'analysis.md').write_text(report)
    methods='''We evaluated apparent delay dynamics using a prespecified fixed window of 513 samples at 5120 Hz. We retained the original weighted-MSE estimator, candidate lag range, alignment score E and normalized overlap-add reconstruction. We compared hops of 1, 128, 256 and 513 samples, both before and after reconstruction. The default-mode primary configurations were rectangular weights with hop 256 and raw estimates for the discrete model, and Hann weights with hop 1 and reconstructed estimates for the continuous model. No data-driven window selection was performed in this fixed-window experiment.

For each tested signal/noise condition, we generated 199 constant-delay calibration records, 200 independent constant-delay test records and 200 independent records from the original time-varying-delay model. Constant lags were 12 samples for the discrete waveform and zero for the continuous waveform. We tested independent Gaussian noise with SD 0.02, 0.10 and 0.20 in both models, plus stationary AR(1) noise with rho=0.8 and marginal SD=0.02 in the continuous model. The statistic was the temporal variance of the estimated delay on the estimator's native output centers. Monte Carlo p-values were (1 + number of calibration statistics at least as large as the test statistic)/200. We tested at alpha=0.05, treating an entire recording as the inferential unit and retaining all within-record dependence. We separately calibrated the maximum statistic across the eight hop/reconstruction configurations. Exact 95% binomial intervals for held-out rejection rates are conditional on the realized calibration set.
'''
    en=f'''# Proposed response to Reviewer 2.4.3

We thank the reviewer for raising the possibility of window-induced apparent dynamics. We agree that a fluctuating estimated delay curve alone is insufficient evidence for a changing neural delay. The sliding-correlation findings of Leonardi and Van De Ville motivate a method-specific assessment rather than an assumption that overlap necessarily increases variability for every estimator.

{methods}

At the original default mode within this fixed-window experiment, the held-out results were:

{table}

The noiseless constant-delay controls recovered constant curves, with maximum variance {facts['noiseless_constant_max_T_ms2']:.4g} ms². For a fixed center, signal pair, window and weighting, the original local estimate was identical across hop settings. Changing the hop changes which centers are sampled and their dependence, while overlap-add reconstruction additionally changes the returned curve. Accordingly, we report native-grid variability separately from reconstruction diagnostics on a common valid time support, and report delay RMSE and bias alongside variability. A stable but inaccurate estimate is not treated as successful delay recovery.

These results characterize false-positive rates and detection power under the tested, fully specified constant-delay null models. They do not establish that all variations in real recordings reflect neural state changes. The calibration uses known signal/noise models and one constant lag per waveform, so it is not yet a general test for an unknown constant lag in real data. Moreover, the experiment conditions on W=513; inference after data-driven window selection would require repeating that selection inside every null realization. We therefore interpret real-data delay profiles descriptively unless an appropriately matched recording-level null has been calibrated. This analysis does not correct window-level between-group Welch tests.

References to revised manuscript locations and final figure/table numbers should be added only after these additions have actually been adopted. The manuscript and external documents have not been edited by this local analysis.
'''
    (out/'response_en.md').write_text(en)
    (out/'methods_en.md').write_text('# Proposed methods addition\n\n'+methods)
