"""Summarize completed results, add integrity checks and a portable Chinese report."""
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

zero_text='；'.join(f"{row.noise}噪声下平均绝对延迟为{row.mean_abs_ms:.3f} ms，延迟RMSE为{row.mean_rmse_ms:.3f} ms" for row in zero.itertuples())
text=f'''NeuroTD：Reviewer 2.7 信号质量—延迟实验，本机完成报告
完成日期：{time.strftime('%Y-%m-%d')}

1. 已完成什么
从DANDI官网下载版本0.250610.0938的资产目录及三个原始文件的官方元数据。本机已有的三个NWB文件逐一通过官方SHA-256与字节数校验，因此复用约4.1 GB原始数据，没有重复下载。数据范围为sub-01的ses-01、ses-02、ses-03，与此前原稿分析范围一致。本轮未下载或分析其他受试者。
完成150个8秒试次×32通道的信号质量计算。主分析排除16个带artifact标记试次，共134个；另保留全部150试次和124个无artifact且回答正确试次的敏感性结果。
原有150×60×508的有符号窗口延迟全部复用，不重跑数百万个已有估计。重新生成9,000条配对×试次汇总；主分析保留有效零延迟。原项目与原始文件均未修改。
完成11,520组新增噪声模拟及36组无新增噪声基准，共11,556组。累计124次整段记录与原函数进行逐窗核对，已检查的延迟均完全一致。真实数据汇总与旧表的非零延迟均值最大差为{maxdiff:.3g} ms。
本轮独立脚本以线性FFT卷积计算同一个MSE目标中的交叉乘积，并保留每个候选片段的完整能量项、搜索范围与符号。这是原MSE搜索的等价计算加速，不是把NeuroTD换成论文中的FFTShift或相位相关基线。浮点计算的验证仅覆盖已列明的测试记录，不能推定所有可能信号上的数值行为均已穷尽。

2. 真实数据：比较的是信号质量代理指标，不是已知真实SNR
没有纯净信号和噪声真值，不能把RMS、某个频段功率或对齐残差直接命名为真实SNR。预先固定四项指标：50 Hz与60 Hz邻域谱突出程度、高频功率占比、配对信号RMS幅度比。50/60 Hz突出可能包含真实神经振荡；高频也可能包含神经活动；RMS不平衡也不等于噪声。因此它们只用于探索潜在混杂。
每个8秒通道以1秒Hann分段、50%重叠计算Welch谱。线频突出程度=10log10[中心±1 Hz平均功率/距离中心2–5 Hz平均功率]；高频比例=1000–10000 Hz功率/1–10000 Hz功率。配对使用两通道中较大的线频突出程度、两通道高频比例均值、abs(log2(RMS_x/RMS_y))。延迟结果是每个试次所有508个有效窗口的平均绝对延迟。
在同一会话、同一任务负荷和同一电极对内，跨试次计算Spearman相关；至少10试次且指标有变化才报告。相关分布不构成独立样本检验，没有把重叠窗口、重复电极对或3个会话当作独立受试者。
原来的60对中，12对同束、48对跨束，分别展示。脑区映射此前存在未解决问题，因此原脑区标签仅在明确标注的描述性敏感性表中保留。不能据此宣称海马SNR更低，或其延迟差已被SNR解释。

主分析相关汇总：
{quality_medians.to_string(index=False,float_format=lambda x:f'{x:.3f}')}
本轮没有观察到四项代理指标与延迟之间一致的正相关模式。60 Hz谱突出程度的相关中位数接近0（同束约0.012，跨束约0.032）；RMS幅度不平衡的相关中位数为同束约0.273、跨束约0.131。高频比例则呈负的相关中位数。这些方向差异说明不能把所有代理指标统一解释成“噪声强度”，也不能从弱相关推出噪声混杂已被排除。

3. 已知延迟加噪实验
三个会话×四个电极束各取一个预先规定模板，共12段。使用每会话首个无artifact且正确试次，固定取相对试次开始1–2秒的一秒记录，并保留原记录边界余量；没有循环移位或零填充。此片段只是固定模板，不宣称其生理上没有噪声。
构造y(t)=x(t-d)，d=0、80、320采样点，即0、2.5、10 ms。使用原矩形窗W=1001、hop=500、搜索±500采样点（±15.625 ms），幅度增益1，不更改E，不做自动选窗。
新增噪声包括白噪声及AR(1)有色噪声（rho=0.95），分别只加在y或独立加在x、y。新增噪声SNR=20、10、0、-10 dB；每条件20个随机实现。同一实现跨SNR复用标准化噪声，实现配对比较。噪声按模板均方功率缩放，逐条检查实际新增噪声SNR。统计单位是完整构造信号对。
36组无新增噪声基准的最大RMSE为{base_max:.6g} ms。
真实延迟为0、两通道都加噪声且SNR=-10 dB时：{zero_text}。

部分关键数值（每行12模板×20噪声实现，240个构造信号对）：
{selected.to_string(index=False,float_format=lambda x:f'{x:.3f}')}

4. 如何用于回复审稿人
这是一项对“噪声是否会造成表观延迟”的直接敏感性检验，并以原范围真实数据补充信号质量与延迟的关联分析。应同时查看估计绝对延迟、真实延迟RMSE和有符号偏差，不能把它们混为一谈。低SNR使估计失准不等于所有真延迟下绝对延迟都会增大。
这一结果不能单独排除原脑区比较中的噪声混杂：本轮只有一名受试者；真实数据的SNR未被直接识别；脑区标注与配对结构需要进一步核对；受试者层面的统计推断、波形对应假设及生理机制解释仍是后续工作。也没有据此处理跨区域传输方向。

5. 如何阅读输出
图01：各质量指标与平均绝对延迟的试次间相关。每个箱图汇总同一会话/任务负荷/电极对内的相关系数，无p值或校准置信区间。
图02：上排是估计的平均绝对延迟，黑虚线是真实绝对延迟；下排是真实延迟RMSE。横轴向右新增噪声增加。曲线为12模板×20噪声实现的平均，不是不同受试者的平均。
图03：真实延迟为0时，每条线单独展示一个模板，便于发现模板差异。线间离散不是生物学总体置信区间。
results/包含完整数值、全部敏感性结果、验证记录。checkpoints/保存模板、每个构造信号对的估计数组和分批结果。provenance/保存官方来源、SHA-256、输入路径、环境和模板清单。analysis_plan.json为查看结果前固定的方案。
运行：使用neurotd312环境执行run_analysis.py prepare，再执行run_analysis.py run，最后执行finish_report.py。已有分批结果会复用。此脚本包含本机绝对路径；换电脑时需调整顶部路径。
'''
(P/'结果说明.txt').write_text(text)

def table(df):return df.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)
def picture(name):
    data=base64.b64encode((P/f'figures/{name}.png').read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{name}">'
parts=[f'<h1>信号质量会影响延迟估计吗？</h1><p class="sub">NeuroTD · Reviewer 2.7 · 本机实验报告 · {time.strftime("%Y-%m-%d")}</p>',
       '<div class="cards"><div><b>3</b>已验证NWB会话</div><div><b>4,800</b>通道×试次质量记录</div><div><b>11,556</b>构造信号对</div><div><b>124</b>原函数核对记录</div></div>',
       '<p class="note">本轮范围为一名受试者。原项目与E未修改；真实数据采用质量代理指标，加噪实验具有已知的构造延迟与新增噪声强度。</p>',
       '<h2>数据范围与验证</h2>',table(scope),
       '<h2>真实信号的质量—延迟关联</h2>',picture('01_real_quality_correlations'),table(quality_medians),
       '<h2>控制新增噪声，观察估计如何变化</h2>',picture('02_controlled_noise'),table(selected),
       '<h2>不同模板的敏感性</h2>',picture('03_template_sensitivity'),
       '<h2>完整说明与解释边界</h2><div class="full">'+html.escape(text).replace('\n','<br>')+'</div>']
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>NeuroTD 信号质量—延迟实验</title><style>body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;max-width:1150px;margin:40px auto;color:#203047;line-height:1.7;padding:0 24px}h1{font-size:32px}h2{border-bottom:2px solid #c9dbe7;padding-top:24px}.sub{color:#617080}.cards{display:flex;gap:16px}.cards div{background:#edf4f8;border-radius:12px;padding:18px;flex:1}.cards b{display:block;font-size:30px;color:#006a96}.note{border-left:4px solid #d89239;background:#fff7ea;padding:16px}img{max-width:100%}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#e8eff5}tr:nth-child(even){background:#f7f9fb}.full{font-size:14px;overflow-wrap:anywhere}@media print{body{margin:0}img{break-inside:avoid}h2{break-after:avoid}}</style><body>'+''.join(parts)+'</body></html>'
(P/'实验报告.html').write_text(page)
print(json.dumps(checks,ensure_ascii=False,indent=2));print(text)
