"""Summaries of independent-noise experiments (not independent windows)."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_report(root):
    root=Path(root)
    out=root/'results'
    figs=root/'figures'
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    conf=json.loads((root/'frozen_config.json').read_text())
    metas=sorted((root/'checkpoints').rglob('rep*_W*.json'))
    expected=2*(conf['zero_noise_deterministic_checks']+3*conf['nonzero_repeats'])*6
    if len(metas)!=expected:
        raise RuntimeError(f'Incomplete experiment: {len(metas)}/{expected}; not writing final conclusions')
    rows=[]
    for path in metas:
        row=json.loads(path.read_text())
        row['checkpoint']=str(path.with_suffix('.npz').relative_to(root))
        rows.append(row)
    df=pd.DataFrame(rows).sort_values(['model','noise_std','replicate_id','window_size'])
    keys=['model','noise_std','replicate_id','window_size']
    assert not df.duplicated(keys).any()
    assert np.isfinite(df[['E','mean_r','rmse_ms']].to_numpy()).all()
    df.to_csv(out/'repeat_results.csv',index=False)
    summaries=[]
    validation=[]
    for (model,noise,w),part in df.groupby(['model','noise_std','window_size'],sort=True):
        part=part.sort_values('replicate_id')
        samples=[]
        expected_centers=None
        for row in part.to_dict('records'):
            with np.load(root/row['checkpoint']) as z:
                centers=z['centers']
                if expected_centers is None:
                    expected_centers=centers.copy()
                assert np.array_equal(expected_centers,centers), 'Mismatched centers; cannot stack'
                r=z['r'].copy()
                assert np.all(np.isfinite(r))
                assert np.isclose(np.sqrt(np.mean(r)),row['E'],rtol=1e-12,atol=1e-12)
                samples.append(r)
        R=np.stack(samples)
        B,M=R.shape
        assert B==(2 if noise==0 else conf['nonzero_repeats'])
        E=part.E.to_numpy()
        mean_r=R.mean(axis=1)
        if noise==0:
            assert np.array_equal(R[0],R[1])
            v_full=v_ind=0.0
            e_sd=0.0
        else:
            v_full=float(np.var(mean_r,ddof=1))
            v_ind=float(np.var(R,axis=0,ddof=1).sum()/M**2)
            e_sd=float(np.std(E,ddof=1))
        # Explicit identity check on <=12 windows avoids an enormous MxM matrix.
        cols=np.unique(np.linspace(0,M-1,min(M,12)).astype(int))
        small=R[:,cols]
        if noise==0:
            explicit=direct=0.0
        else:
            explicit=float(np.cov(small,rowvar=False,ddof=1).sum()/len(cols)**2)
            direct=float(np.var(small.mean(axis=1),ddof=1))
        assert np.isclose(explicit,direct,rtol=1e-10,atol=1e-12)
        summaries.append(dict(model=model,noise_std=noise,window_size=w,hop=int(part.hop.iloc[0]),
            B=B,M=M,independent_noise_repeats=noise>0,
            E_mean=float(E.mean()),E_repeat_sd=e_sd,E_mean_mc_se=e_sd/np.sqrt(B) if noise>0 else 0.0,
            E_median=float(np.median(E)),E_q025=float(np.quantile(E,.025)),E_q975=float(np.quantile(E,.975)),
            mean_r_mean=float(mean_r.mean()),mean_r_repeat_sd=np.sqrt(v_full),
            V_full=v_full,V_ind=v_ind,V_cross=v_full-v_ind,
            mean_r_sd_full=np.sqrt(v_full),mean_r_sd_ignore_covariance=np.sqrt(v_ind),
            variance_ratio=v_full/v_ind if v_ind>0 else np.nan,
            rmse_ms_mean=float(part.rmse_ms.mean()), E_audit_max_abs_diff=float(part.E_abs_diff.max()),
            min_normalizer=float(part.min_normalizer.min()),
            at_normalizer_floor_total=int(part.normalizer_at_floor_count.sum())))
        validation.append(dict(model=model,noise_std=noise,window_size=w,centers_equal=True,
            n_invalid=0,explicit_cov_subset_variance=explicit,direct_subset_variance=direct,
            covariance_identity_abs_diff=abs(explicit-direct)))
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/'uncertainty_summary.csv',index=False)
    covcols=['model','noise_std','window_size','B','M','V_full','V_ind','V_cross',
             'mean_r_sd_full','mean_r_sd_ignore_covariance','variance_ratio']
    summary[covcols].to_csv(out/'covariance_decomposition.csv',index=False)
    pd.DataFrame(validation).to_csv(out/'validation_checks.csv',index=False)
    selections=[]
    for (model,noise,rep),part in df.groupby(['model','noise_std','replicate_id']):
        part=part.sort_values('window_size')
        assert len(part)==6 and part.x_hash.nunique()==1 and part.y_hash.nunique()==1
        best=part.iloc[int(np.argmin(part.E.to_numpy()))]
        oracle=part.iloc[int(np.argmin(part.rmse_ms.to_numpy()))]
        ties=part.loc[part.E==best.E,'window_size'].to_list()
        selections.append(dict(model=model,noise_std=noise,replicate_id=rep,
            selected_W=int(best.window_size),selected_E=float(best.E),selected_rmse_ms=float(best.rmse_ms),
            oracle_W=int(oracle.window_size),oracle_rmse_ms=float(oracle.rmse_ms),
            selection_loss_ms=float(best.rmse_ms-oracle.rmse_ms),
            exact_tie_count=len(ties),exact_tied_windows=json.dumps(ties)))
    selected=pd.DataFrame(selections)
    selected.to_csv(out/'selection_by_repeat.csv',index=False)
    frequencies=[]
    for (model,noise),part in selected.groupby(['model','noise_std']):
        for w in conf['windows']:
            frequencies.append(dict(model=model,noise_std=noise,window_size=w,
                n_selected=int((part.selected_W==w).sum()),B=len(part),
                selection_frequency=float((part.selected_W==w).mean())))
    freq=pd.DataFrame(frequencies)
    freq.to_csv(out/'selection_frequencies.csv',index=False)
    plt.rcParams.update({'font.size':10,'figure.dpi':130})
    noises=[.02,.1,.2]
    models=['discrete','continuous']
    windows=conf['windows']
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,m in enumerate(models):
        for j,n in enumerate(noises):
            ax=axes[i,j]
            d=df[(df.model==m)&(df.noise_std==n)]
            ax.boxplot([d[d.window_size==w].E.to_numpy() for w in windows],tick_labels=windows,
                showfliers=True,flierprops={'markersize':2})
            ax.set(title=f'{m}; noise SD={n:g}; B=200',xlabel='Window length (samples)',ylabel='E (original score)')
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('E across independent noise realizations; windows within each run are not independent replicates')
    fig.savefig(figs/'01_E_distributions.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,m in enumerate(models):
        for j,n in enumerate(noises):
            ax=axes[i,j]
            d=summary[(summary.model==m)&(summary.noise_std==n)].sort_values('window_size')
            ax.plot(range(6),d.mean_r_sd_full,'o-',label='Across-repeat SD (includes covariance)')
            ax.plot(range(6),d.mean_r_sd_ignore_covariance,'s--',label='Ignoring cross-window covariance')
            ax.set(xticks=range(6),xticklabels=windows,xlabel='Window length (samples)',
                ylabel='SD of mean(r)',title=f'{m}; noise SD={n:g}')
            ax.grid(alpha=.2)
            if i==0 and j==0: ax.legend(fontsize=7)
    fig.suptitle('Covariance decomposition using independent repeats at matched window centers')
    fig.savefig(figs/'02_covariance_effect.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,m in enumerate(models):
        for j,n in enumerate(noises):
            ax=axes[i,j]
            d=freq[(freq.model==m)&(freq.noise_std==n)].sort_values('window_size')
            ax.bar(range(6),d.selection_frequency,color='#247BA0')
            ax.set(xticks=range(6),xticklabels=windows,ylim=(0,1),xlabel='Selected W (samples)',
                ylabel='Selection frequency',title=f'{m}; noise SD={n:g}; B=200')
    fig.suptitle('Unchanged argmin(E) selection; frequent selection does not establish delay accuracy')
    fig.savefig(figs/'03_selection_frequencies.png');plt.close(fig)
    captions='''# Figure captions\n\n1. Fixed-window E distributions across 200 independent Gaussian-noise realizations for each model and nonzero noise level. Clean signals and true delays are fixed. Six candidate windows share the same noisy pair within each realization. Boxplots show median, quartiles, Tukey whiskers and outliers, not confidence intervals.\n\n2. Across-realization SD of mean(r), compared with the variance calculation omitting off-diagonal cross-window covariances. Covariances are estimated across independent repeats at matched time positions, not by assuming temporal stationarity within one run. Equality between the full covariance expression and the variance of the row means is an algebraic identity, not coverage validation. Discrete hop=W//2; continuous hop=1 with original Hann weighting and reconstruction.\n\n3. Frequency of window selection by the unchanged minimum-E rule. Ties follow the first candidate in ascending W. These frequencies describe selection stability, not correctness or independent validation of true delays.\n\nEmpirical 2.5% and 97.5% quantiles in the CSV describe the simulated E distribution; they are not calibrated confidence bounds for a single observed recording. Zero-noise duplicates are deterministic checks, not independent evidence.\n'''
    (out/'figure_captions.md').write_text(captions)
    positive=summary[summary.noise_std>0]
    greater=int((positive.variance_ratio>1+1e-8).sum())
    less=int((positive.variance_ratio<1-1e-8).sum())
    baseline=pd.read_csv(root/'baseline/comparison.csv')
    maxdiff=float(df.E_abs_diff.max())
    unchanged=__import__('audit').config()['full_source_hashes']==conf['full_source_hashes']
    if not unchanged: raise AssertionError('Source changed')
    facts=dict(total_runs=len(df),nonzero_runs=int((df.noise_std>0).sum()),
        zero_deterministic_runs=int((df.noise_std==0).sum()),n_failed=0,
        E_max_abs_diff=maxdiff,source_unchanged=unchanged,
        n_variance_ratio_above_one=greater,n_variance_ratio_below_one=less,
        min_variance_ratio=float(positive.variance_ratio.min()),max_variance_ratio=float(positive.variance_ratio.max()),
        baseline_E_matches=int(baseline.E_matches_local_csv.sum()),
        baseline_rmse_matches=int(baseline.rmse_matches_local_csv.sum()),baseline_rows=len(baseline))
    (out/'verified_facts.json').write_text(json.dumps(facts,indent=2))
    sample=positive[['model','noise_std','window_size','E_mean','E_repeat_sd','variance_ratio']]
    table='| '+' | '.join(sample.columns)+' |\n| '+' | '.join(['---']*len(sample.columns))+' |\n'
    table+='\n'.join('| '+' | '.join(format(v,'.5g') if isinstance(v,(float,int)) else str(v) for v in row)+' |'
                     for row in sample.itertuples(index=False,name=None))
    report=f'''# Independent-repeat uncertainty in the alignment score

## Experiment

For each fixed discrete and continuous signal, six window sizes were tested at noise SDs of 0.02, 0.10 and 0.20 with 200 independent noise realizations per condition. Each window size had two deterministic zero-noise checks. All {len(df)} window evaluations completed. The original estimator was used throughout. Only NeuroTD was evaluated in the independent-repeat experiment.

E = sqrt(mean(r_k)), with residual r_k as defined in the implementation. As in the Figure 3 script, continuous delays were rounded only for scoring. Delay RMSE used the unrounded estimates. The largest absolute difference in the independent E check was {maxdiff:.3g}. Residual normalization used sqrt(mean(x²)*mean(y²)) + 1e-16.

## Benchmark

At noise SDs of 0 and 0.02, {facts['baseline_E_matches']}/24 E values and {facts['baseline_rmse_matches']}/24 delay RMSEs agreed with the archived CSV within the prespecified tolerance (baseline/comparison.csv). These checks cover the local benchmark. The dependency versions differ from the manuscript environment.

## Results

Across 36 fixed-window conditions with positive noise, the variance of the mean including covariance was larger than the diagonal-only variance in {greater} conditions and smaller in {less}. The ratios ranged from {facts['min_variance_ratio']:.4g} to {facts['max_variance_ratio']:.4g}.

{table}

## Interpretation

- E is a descriptive residual summary. Dependence affects sampling variance and leaves the denominator M in E unchanged.
- SD(E_b) measures variation in a full analysis across noise realizations. SD(E_b)/sqrt(B) is the Monte Carlo standard error of the replicate mean.
- Covariance is estimated at matched time positions across independent repetitions. Window positions may have different means. No stationarity assumption or effective sample size of M/2 is imposed.
- The covariance identity checks the calculation. Coverage of 95% intervals would require separate validation.
- The clean signals and generating delays are fixed across repeats. The analysis captures conditional noise uncertainty and retains poor estimates for large noiseless windows and repeated waveforms. It does not quantify between-participant variation or validate the selected lags.
- Manuscript q = 1/(1+sqrt(mean(r_k²))) and implemented E = sqrt(mean(r_k)) aggregate residuals differently. A monotone transformation of E alone preserves its ordering. q and E may select different windows.
- Real-data Welch tests, cross-trial reliability and dynamic-curve artifacts are outside this experiment.

## Outputs

- results/uncertainty_summary.csv contains the E distribution and standard errors at fixed W.
- results/covariance_decomposition.csv decomposes the variance of the mean.
- results/selection_by_repeat.csv reports window selection and errors against the generating delay.
- checkpoints/ contains the per-window results and random seeds.
'''
    (root/'report.md').write_text(report)
    en=f'''# Proposed author response — Reviewer 2.4, first point\n\nWe thank the reviewer for highlighting dependence between overlapping windows. We distinguish the descriptive aggregation of alignment residuals from uncertainty in that aggregation. An arithmetic mean of window-level residuals does not require the windows to be independent. Dependence enters the sampling variance through cross-window covariance terms; replacing the number of windows in the score denominator with an effective sample size would change the residual summary rather than correct its uncertainty.\n\nThe implementation audit identifies the score used in these analyses as E = sqrt(mean(r_k)), where r_k is the per-window normalized residual defined in the implementation. This differs from the q formula in the submitted manuscript and must be reconciled explicitly. We retained the implemented estimator, residual score and minimum-E selection rule. For each of the discrete and continuous simulation models, we held the clean signals and generating delays fixed and performed 200 independent Gaussian-noise realizations at each of three nonzero noise levels (0.02, 0.10 and 0.20), evaluating all six candidate windows. Each realization reran delay estimation and residual evaluation; all candidate windows within a realization used the same noisy signal pair.\n\nWe report the across-realization standard deviation of E separately from the Monte Carlo standard error of its mean. We also estimated the covariance contribution to the variance of the mean residual across independent realizations at matched window centers. In the 36 fixed-window, nonzero-noise configurations, the full variance was greater than the diagonal-only variance in {greater} configurations and smaller in {less}. Thus the effect is measured from the dependence structure rather than assigned from the overlap percentage. The covariance identity used to check implementation is an algebraic check, not an independent validation or a confidence-interval coverage assessment.\n\nThese simulations quantify uncertainty conditional on the tested signal models and noise processes. They do not calibrate confidence in delays from real recordings, establish equivalence between residual optimality and delay accuracy, or address the window-level statistical tests and possible dynamic artifacts raised in the remaining parts of this comment. References to completed manuscript revisions and final table/figure numbers should be inserted only after these changes are adopted.\n'''
    (root/'response_en.md').write_text(en)
    methods='''# Proposed manuscript additions (not applied)\n\n## Methods: residual-score uncertainty\n\nThe implemented alignment-error summary was E=[M^{-1} sum_k r_k]^{1/2}, with r_k defined by the implementation-specific normalization reported in the accompanying source audit. M denotes the number of valid windows, not the number of statistically independent observations. To assess noise-related uncertainty without assuming independent windows, we fixed the noiseless signals and generating delays separately for the discrete and continuous models and repeated the complete estimation procedure over 200 independent Gaussian-noise realizations at each of sigma=0.02,0.10,0.20. Each realization was shared across candidate windows W={33,65,129,257,513,1025}. Discrete analyses used H=W//2 and rectangular weights; continuous analyses used H=1, Hann weights and the original overlap-weighted reconstruction. Continuous delays were rounded only for the alignment-error scoring, following the original pipeline. We reported the across-realization standard deviation of E and, separately, its mean's Monte Carlo standard error. Empirical quantiles describe the simulated score distribution, not calibrated confidence intervals for a single recording.\n\nFor matched window positions, Var(mean(r))=M^{-2} sum_{i,j} Cov(r_i,r_j). We estimated this quantity across independent realizations and compared it with the diagonal-only sum M^{-2} sum_i Var(r_i). This comparison does not assume that window scores have a constant mean over time. Zero-noise repeats were deterministic implementation checks only.\n\n## Discussion\n\nOverlapping windows need not represent independent observations. The residual summary can still be computed descriptively, while its uncertainty depends on covariance between window scores. Our repeated-noise simulations assess this dependence conditional on the tested clean signals; they do not resolve uncertainty from participant variability, model misspecification or ambiguous event correspondence in real data. These results should not be interpreted as corrected window-level inferential tests or as evidence that residual-minimizing windows necessarily minimize delay error.\n\n## Required reconciliation\n\nThe submitted q=1/(1+sqrt(mean(r_k^2))) is not the implemented E. Update the score definition, table labels, preference direction and analysis descriptions consistently; do not silently relabel E as q. Source normalization uses sqrt(mean(x^2)*mean(y^2))+1e-16, so avoid assigning unverified scale-invariance or calibrated-confidence properties to it.\n'''
    (root/'manuscript_additions.md').write_text(methods)
    progress=json.loads((root/'progress.json').read_text())
    progress.update(status='complete',source_unchanged=True,
        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (root/'progress.json').write_text(json.dumps(progress,indent=2))
    print(json.dumps(facts),flush=True)
