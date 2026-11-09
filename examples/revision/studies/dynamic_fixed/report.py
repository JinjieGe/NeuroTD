"""Whole-recording Monte Carlo inference and local bilingual draft outputs."""
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
    interp=(f'{len(inconsistent)} 个主要条件的条件性二项区间不包含 5%，需要结合有限校准集的不确定性解释，不能宣称所有条件都已精确控制在 5%。'
        if inconsistent else '所有主要条件的条件性二项区间均包含 5%。这与名义水平相容，但不是精确等于 5% 的证明。')
    report=f'''# NeuroTD 2.4.3 补充模拟结果

本实验直接检验恒定真实延迟下的表观动态，以及整条曲线的零模型校准。原估计器、E、搜索范围、权重计算和重建公式均保留，原项目未修改。模拟完成 {facts['independent_signal_pairs']} 对独立噪声信号，每对派生八种步长与重建配置。这些派生配置是配对分析，不是额外独立重复。

## 主要结果

{table}

{interp}

零噪声恒定延迟对照的最大曲线方差为 {facts['noiseless_constant_max_T_ms2']:.4g} ms²。该对照验证所测试波形、延迟网格和搜索范围下的常量恢复，不证明所有信号都具有唯一可辨识延迟。

## 基准和方法

重新核算 2.4.1 已保存的 24 个基准配置的 E 与延迟 RMSE，并与原项目 CSV 比较。正式实验前又直接调用原估计器核对两种模型、两个噪声水平下的 W=513 结果，以及四种步长的原始估计和重建；全部数值逐项一致。每次完整模拟独立核算 E，最大绝对差为 {facts['E_audit_max_abs_diff']:.4g}。

固定 W=513，fs=5120 Hz，记录长度 1 秒。步长 H=1、128、256、513，名义重叠约为 99.8%、75.0%、50.1%、0%。离散波形使用原矩形权重，连续波形使用原 Hann 权重。主要配置为离散 H=256 的原始逐窗估计和连续 H=1 的重建估计，分别对应原脚本模式。本实验没有更改最小 E 选窗规则，也没有运行该规则；所有统计结论明确以预先固定 W=513 为条件，不能自动推广到数据驱动选窗后的输出。

恒定延迟为离散 12 个采样点、连续 0 个采样点。使用原模型的干净参考波形，零模型的 y 是 x 的精确周期平移。动态阳性对照使用原模拟的干净 y 和原真值定义。白噪声标准差为 0.02、0.10、0.20；另在连续模型 SD=0.02 下加入平稳 AR(1) 噪声 rho=0.8，边际标准差仍为 0.02。x 与 y 的噪声过程独立。

每个条件有 199 次恒定延迟校准、200 次独立恒定延迟测试和 200 次独立动态测试。主统计量 T 为一次分析返回的延迟曲线在其原始中心网格上的方差，单位 ms²。p=(1+校准 T 大于等于观测 T 的次数)/200，在 p≤0.05 时拒绝恒定延迟零模型。完整记录是统计单位，保留记录内部的窗口重叠和噪声相关性。并列统计量保守处理。八个配置的 max(T) 另行作为一个整体统计量校准，不能把八个未校正 p 值择优报告为显著。

## 如何解释步长与重建

同一 x、y、W 和权重下，相同中心的原始延迟估计不随步长改变。本实验调用原函数计算 H=1 的全部原始窗口，其他步长复用完全相同的中心子集，避免重复搜索。与直接原函数调用的核对证明这种缓存不改变结果。原始曲线在不同步长下的方差变化还包含中心采样数量和位置变化，不能全部归因于重叠。

重建使用原归一化重叠相加公式，保留其边界权重。除原函数返回中心的方差外，汇总表另提供各步长共同有效时间区间上的重建方差。后者只是诊断，不替换原估计器输出。无重叠 Hann 窗端点没有正权重，公共网格明确剔除这些点，没有把它们当作零延迟。

## 统计边界与论文回复

表中区间是给定本次校准集合后的 Clopper–Pearson 二项区间，不包含重新抽取 199 条校准记录时的阈值不确定性。有限校准分辨率为 0.005。零模型参数已知，干净波形固定，因此这是模型条件下的方法校准，不是从真实记录估计未知固定延迟和噪声结构后的通用检验。

RMSE 与偏差和曲线方差一并保留。曲线很平不代表延迟估计正确，低检出率也不能被描述为真实延迟恒定。源代码连续真值约定沿用原生成延迟 Δ(t)，没有私自改成逆时间映射。

这些结果不能证明真实神经记录中的变化全部是生理变化，不能代替 2.4.2 的真实数据组间检验，也不能把文中窗口级 Welch 检验称为已校正。若最终展示曲线经过选窗、筛选或其他选择，这些步骤必须在对应零模拟中完整重跑后才能作正式推断。

Leonardi 与 Van De Ville 的滑动相关研究提示窗长与信号时间尺度失配可产生伪动态；它不直接证明任何重叠比例都会增加 NeuroTD 的曲线方差。回复应承认窗口化和噪声可能造成表观变化，再呈现本方法的直接对照，避免声称已普遍排除伪影。

参考：Leonardi N, Van De Ville D. On spurious and real fluctuations of dynamic functional connectivity during rest. NeuroImage 104 (2015), 430–436. https://doi.org/10.1016/j.neuroimage.2014.09.007

## 图和数据

图 1：独立测试集的误报率和检出率。误差线是条件性 95% 二项区间。图 2：恒定延迟下的平均表观方差，原始与重建曲线分别显示。图 3：预先规定的第 0 次重复，不按视觉效果选择。图 4：默认模式在固定窗长下的平均延迟 RMSE。

test_summary.csv 为主要及敏感性检验汇总；whole_curve_tests.csv 保留每条测试记录的 p 值；delay_and_E_summary.csv 同时报告方差、RMSE、偏差及原 E；repeat_metrics.csv 保留全部配对配置。随机种子、未汇总曲线和源文件校验值在独立实验目录中留存。
'''
    (out/'analysis_zh.md').write_text(report)
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
    zh=f'''# Reviewer 2.4.3 中文回复草稿

感谢审稿人指出窗口化可能造成表观动态。我们同意，仅凭估计延迟曲线随时间波动，不能认定真实神经延迟发生了变化。Leonardi 与 Van De Ville 关于滑动相关的发现提示有必要对 NeuroTD 作直接检验，而不能预先假定所有估计器都会因重叠而增加波动。

我们保留原加权 MSE 延迟估计器、候选延迟范围、E 和归一化重叠重建方式，在预先固定 W=513、采样率5120 Hz 下比较 H=1、128、256、513，并分别检查重建前后的曲线。离散模型的主要配置为矩形权重、H=256、不重建；连续模型为 Hann 权重、H=1、原重建，均对应原脚本的运行模式。本项固定窗实验不进行数据驱动选窗。

每个信号与噪声条件使用199次恒定延迟模拟校准、200次独立恒定延迟模拟评估误报率，以及200次原动态延迟模型模拟评估检出率。两种模型均测试标准差0.02、0.10、0.20的独立高斯噪声，另在连续模型中测试 rho=0.8、边际标准差0.02的平稳 AR(1) 噪声。统计单位是整条记录，统计量为原输出中心上的延迟方差。Monte Carlo p=(1+校准统计量不小于观测值的次数)/200，在0.05水平检验，保留整条记录中的窗口依赖。另对八个配置的最大统计量整体校准，避免从多个未校正结果中选择显著者。

原默认模式在固定窗长下的独立测试结果如下：

{table}

零噪声恒定延迟对照恢复了常量曲线。固定信号、窗口、权重及中心位置时，改变步长不改变该中心的原始局部估计；步长会改变采样位置及依赖结构，重叠重建还会进一步改变输出曲线。因此，我们将原输出网格上的波动与共同有效时间区间上的重建诊断分开报告，并同时给出 RMSE 和偏差，不把平稳但错误的估计作为成功恢复。误报率与检出率区间是给定本次校准集合后的95%二项区间，不包含校准阈值自身的重复抽样不确定性。

本实验支持的是已知信号与噪声模型、固定窗长及每种波形一个恒定延迟值下的条件性校准，不能证明真实记录中的所有变化都是神经状态变化，也不是针对真实数据未知恒定延迟的一般性检验。若曲线经过数据驱动选窗，必须在每条零模拟中重跑选窗流程后再作推断。真实数据未完成匹配的记录级零模型校准前，我们将其延迟曲线解释为描述性结果。这项分析不替代窗口级组间 Welch 检验的校正。

实际采用以上补充后再填写修订稿章节、页码和图表编号。本地模拟没有修改论文正文或外部文档。
'''
    (out/'response_en.md').write_text(en)
    (out/'response_zh.md').write_text(zh)
    (out/'methods_en.md').write_text('# Proposed methods addition\n\n'+methods)
