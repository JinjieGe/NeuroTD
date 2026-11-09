from pathlib import Path
import json,hashlib,html
import pandas as pd
import numpy as np
from run_analysis import ROOT,save_json,provenance

r=pd.read_csv(ROOT/'outputs/real_results.csv');v=pd.read_csv(ROOT/'outputs/final_validation.csv')
s=pd.read_csv(ROOT/'outputs/sensitivity.csv')
assert len(v)==90 and v.n.eq(20000).all()
assert len(list((ROOT/'checkpoints').glob('final_*.json')))==30
assert v[v.alpha.eq(.05)].gate.all() and v[v.alpha.eq(.01)].gate.all()
assert (v[v.alpha.eq(.05)].coverage_ci_low>=.94).all()
for p,h in json.loads((ROOT/'provenance_before.json').read_text()).items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
q=r[r.method=='session_jackknife_t2'].set_index('comparison')
order=[('HIP-AMY','AMY-HIP',-1),('HIP-MTG','HIP-MTG',1),('HIP-STG','HIP-STG',1),
       ('MTG-AMY','AMY-MTG',-1),('AMY-STG','AMY-STG',1),('MTG-STG','MTG-STG',1)]
rows=[]
for name,k,sgn in order:
    a=q.loc[k];lo=a.lower_simultaneous95_ms;hi=a.upper_simultaneous95_ms
    if sgn<0:lo,hi=-hi,-lo
    rows.append(dict(comparison=name,estimate_ms=sgn*a.estimate_ms,p_bonferroni=a.p_family,
        lower_simultaneous95_ms=lo,upper_simultaneous95_ms=hi,df=2,
        conditional_significant_05=bool(a.p_family<.05)))
pd.DataFrame(rows).to_csv(ROOT/'outputs/recommended_results.csv',index=False)
table='\n'.join(f"{x['comparison']:9s} {x['estimate_ms']:+.6f} ms  P_adj={x['p_bonferroni']:.6f}  同时95%区间 [{x['lower_simultaneous95_ms']:.6f}, {x['upper_simultaneous95_ms']:.6f}]" for x in rows)
summary=v.groupby('alpha').agg(cases=('gate','size'),passed=('gate','sum'),max_fwer=('fwer','max'),
    max_upper=('fwer_ci_high','max'),min_coverage=('simultaneous_coverage','min'),min_coverage_lower=('coverage_ci_low','min')).reset_index()
summary.to_csv(ROOT/'outputs/final_validation_summary.csv',index=False)
ver=json.loads((ROOT/'outputs/verification.json').read_text())
ver.update(final_validation_records=600000,total_formal_simulation_datasets=1440000,
    final_validation_rows_checked=90,final_primary_05_gate_passed=True,final_primary_01_gate_passed=True,
    final_primary_001_all_cases_passed=False,final_simultaneous95_coverage_gate_passed=True,
    all_original_source_hashes_still_match=True)
save_json(ROOT/'outputs/final_verification.json',ver)

report=f'''NeuroTD Reviewer 2.4.2：本机重新审核与补充统计分析
日期：2026-11-08

结论
本轮已在本机跑通、复核并形成一个明确的、附带假设条件的修订方案：
保留原非零绝对延迟池化均值差，采用会话删除法估计标准误，以t(2)作为小样本参考，六项比较Bonferroni校正。
在本轮独立终检的全部30个既往模型情景中，该方法通过5%及1%家族水平的预定操作性校准标准。
真实数据中，原HIP标签组相对另外三组的比较在5%家族水平下有条件支持；其他三组未达到该水平。
这不是恢复原来的“六组均p<0.001”，也不是已证明真实数据满足会话独立假设。
推荐作为透明披露假设的补充统计结果。正文原来的无条件脑区显著性、群体推广和神经传导解释仍需收窄。

1. 这次实际做了什么
数据仍为sub-01、3个会话、150个试次，原60个通道对，4,572,000个窗口，其中4,108,812个非零。
输入为既有逐试次逐标签组S（非零绝对延迟之和）和N（非零数），未重新运行延迟估计器。
从9,000条试次×通道对记录独立聚合得到600条试次×组记录，与输入一致。
原组池化均值（ms）：AMY 8.898783，HIP 9.676558，MTG 8.981442，STG 8.674186。
这些简称均为原文件标签，不表示已完成微丝解剖定位核验。
原估计器、E及旧实验源码没有修改；输入和核心源文件运行前后SHA-256一致。

正式统计模拟合计1,440,000个数据集：
- 200,000个用于有限模型包络参考（10情景各20,000）；
- 600,000个用于五种方法比较（10情景×3种零假设×20,000）；
- 40,000个额外假设挑战（两情景各20,000）；
- 方法比较完成后，另用600,000个未参与比较的数据集独立终检。
另有10,000个耗时/程序试运行数据集，不进入正式统计。
每个数据集为150个合成试次的4组汇总值，不是新生成并重估1,440,000条原始神经波形。
第一批正式运行约87.7秒；终检约一分钟量级。省时来自复用原延迟、向量化统计运算与小批量单进程计算。
原模型和旧种子未变；本轮新增模拟采用另行冻结的新种子202611080242，不冒充旧模拟的逐位复现。

2. 为什么这次与以前不同
旧分块方法主要估计固定会话内部的重复试次不确定性，较长相关可能被短块漏掉。
本轮直接将整场会话作为删除单元，同场所有试次、通道对和窗口一起保留或删除，不按窗口数增加自由度。
令d为原池化差，d_(-s)为删除第s会话后的池化差：
SE_jack=sqrt[(2/3) sum_s(d_(-s)-mean_s d_(-s))^2]。
T=d/SE_jack；P_adj=min(1,6×2×Pr[t_2>=abs(T)])。
同时95%区间为d±t_(2,1-.05/12)×SE_jack。
参考自由度2来自三个会话减一，不是通过自相关估计出来的“有效窗口自由度”，也不是把原Welch的自由度机械改成2。
原点估计的池化定义保持不变，但不确定性评估层级改变；不能声称与原固定会话试次推断完全同义。

关键假设是会话间误差具有足够独立性、三个会话可支持所讨论记录条件的近似采样解释、会话层汇总近似适合小样本t参考。
仅有三个会话时，这些条件无法靠数据充分验证。删除会话jackknife的通用小样本精确性没有在此证明。
Ibragimov和Muller的分组t理论可说明会话等权t敏感性分析的动机，但不能直接当作对本轮不等权比值jackknife的定理保证。

3. 真实数据结果
全部P为双侧、六重Bonferroni校正；区间为相同参考模型下同时95%区间。
{table}

HIP-MTG在1%家族水平亦有条件支持；HIP-STG约0.010195，不能四舍五入成“P<0.01”。
没有一项推荐分析达到P<0.001。非显著不等于证明等效或没有差异。

4. 校准与独立终检
冻结评价门槛沿用此前规则：家族误报率CP95%区间上界<=1.25×名义alpha。
这是一项允许有限偏差的操作性门槛，不是证明严格alpha控制；每个区间是逐情景区间，不是30情景同时置信保证。
最终600,000个数据集结果：
- alpha=.05：30/30通过。最大观察家族误报率4.875%，最大区间上界5.183%。
- alpha=.01：30/30通过。最大观察家族误报率1.045%，最大区间上界1.196%。
- alpha=.001：27/30通过；A-G、B1-G、B2-G未达到区间上界门槛，最大观察误报率0.145%，最大区间上界0.208%。
  未通过上界标准不等于已证明超过容许上限，可能同时受到有限模拟精度影响；本轮不据此宣称.001水平已验证。
- 同时95%区间覆盖率最低95.125%，其最低CP下界94.817%，通过预先保留的>=94%标准。
校准包括独立、AR=.3/.6、真实结构近似、随机且与信号相关的分母、AR=.85、慢变成分、趋势、重尾异方差和异常试次。
每种情景均有全局零假设和两种部分零假设，以避免只检查“所有组都相同”。
以上是固定生成参数下的模拟证据；真实参数由同一真实表估计，其不确定性没有被完全覆盖。

5. 其他方法也完整保留，未挑最小P
会话等权t(2)：同样三项HIP比较在.05下有条件支持，P分别约.0290、.00716、.01033。
但它比较会话等权的均值差，不完全等于原池化对象。C/D模型的会话等权总体对比不一定为零，验证时使用其自己的真值。
因此不能把这项方法在C/D中的空FWER字段读成0；同时覆盖率是另行按正确真值评价。
HAC b=1+正态：仅作诊断反例。固定大带宽配正态参考严重失准；这不是对旧fixed-b参考方法的复现。
其结果表中的数值0是极端尾概率的浮点下溢，不是数学上P恰为0，不能用于显著性声明。
HAC b=1+t(2)：核心情景保守，但旧慢变压力情景的全局零假设FWER约8.275%，不能仅靠换成t(2)就宣布有效。
有限模型包络：对10个旧模型的最大六项标准化误差校准，并固定1.15保守系数，旧情景FWER最大约2.505%。
它能产生很小的HIP比较P值，但超出参照模型后不可靠，且参数来自真实数据。它不是覆盖全部未知干扰参数的精确最大化Monte Carlo检验。
没有因为它给出更小P就把它设为正文依据。

6. 主动测试的失败边界
E1：会话内AR=.97、会话彼此独立。会话jackknife FWER=4.845%；HAC-t(2)=21.305%；模型包络=9.735%。
E2：每组加入跨三个会话持续共享的随机偏移（SD=.2 ms），把该偏移视为测量误差而非目标效应。
会话jackknife FWER=52.065%，模型包络=98.335%。会话删除无法估计所有会话共同携带的随机偏差。
E2不是发现真实数据误报率为52%，也不是证明真实会话存在该偏移。
它改变了会话依赖及采样解释：如果持续偏移被定义为该受试者的真实条件效应，零假设也需相应改变，不能仍把它计作误报。
该挑战揭示真实效应与共享测量偏差的不可辨识边界，支持明确限定会话独立假设，不支持任何无条件或跨受试者推断。

7. 敏感性分析与能否推广
排除16个artifact试次后，三项HIP比较的描述方向和.05结果保持，校正P约.0197、.00766、.00744。
将零值纳入后差值及P发生改变；这是改变估计对象，不能挑选其中的P<.001替代主结果。
去掉AHL1后，HIP-MTG由+0.695变为-0.544 ms，HIP-STG由+1.002变为-0.237 ms。
这同时删除原HIP组3对中的2对，是电极组成敏感性，不是证明AHL1错误。
这些变体未重新逐一建立校准库，P只作敏感性说明；标签及接触点敏感性仍阻止一般脑区生物学结论。
仅三会话的精确符号翻转分析给出未校正P=.25或.5，六重校正均为1。
这不反驳t检验，而是说明t结果利用了更强的分布假设；不能将其宣传为无分布假设显著性。
新增方法推荐是在旧结果及本轮真实P均已知后的事后选择；已公开记录，并增加未参与选择的终检模拟。
模拟终检独立，不会使真实数据分析变成前瞻性或独立生物学验证。

8. 如何完成审稿修订
建议用以下三层结论回答：
一，接受原窗口级Welch不适当，原p<.001及统一星号撤回。
二，报告会话级参考自由度2的条件性补充分析，以及独立模拟终检；三项HIP标签比较有.05支持，其他没有。
三，将解剖未确认、单受试者、三会话、共享误差及电极组成限制明确写入解释。
若无法为会话间独立性提供可信依据，正文使用描述性结果，将所有新P保留为有条件的补充敏感性分析。
因此本轮完成了可执行且有数据支持的统计回应方案，但没有证明“所有依赖都已解决”或“脑区生理差异已确立”。
正文及在线回复文档未改动。附带中文和英文回复草稿、Methods/Results/Discussion替换建议。

9. 复核与可复现性
输入600条与9,000条更细表重聚合一致；HAC快速恒等式与旧显式协方差公式误差<8e-19。
会话t由SciPy独立复算；jackknife及t(2)尾概率由独立公式复核。
三个参考情景种子重放前1,000条逐位一致；经验尾计数与直接计数一致。
最终90条验证汇总、30个终检检查点和全部关键文件哈希核对通过。
3张PNG已逐张目视检查。脚本、冻结计划、种子、结果、检查点和原文件哈希均保存在本目录。

方法参考
Ibragimov & Muller (2010), t-Statistic Based Correlation and Heterogeneity Robust Inference.
https://www.princeton.edu/~umueller/tstat.pdf
Dufour (2006), Monte Carlo tests with nuisance parameters. 用于区分有限参照网格与完整干扰参数空间校准；本轮不声称满足该文全部精确性条件。
https://jeanmariedufour.research.mcgill.ca/Dufour_2006_JE_MCT.pdf
'''
(ROOT/'analysis_CN.txt').write_text(report)

zh='''Reviewer 2.4.2 回复草稿（正文尚未修改；完成修改后再更新时态与页码）

感谢审稿人指出窗口级统计推断中的依赖问题。我们同意，原分析中的窗口延迟不能被视为独立重复，Welch检验和Bonferroni校正本身并不能解决这一问题。因此，我们不再将原窗口级p<0.001作为正式统计依据。

我们保留原NeuroTD估计器和延迟输出，对相同受试者、三个会话和150个试次进行了会话层面的补充分析。我们保留原非零绝对延迟的池化均值差，以整场会话为删除单元估计jackknife标准误，并采用自由度为2的小样本t参考分布，对六个双侧比较进行Bonferroni校正。该自由度由三个会话确定，不由窗口数量确定；会话之间的独立性及会话层采样解释是这一分析的必要假设。

我们比较了多种推断方法，并在明确记录方法推荐过程后，用另一套未参与方法比较的600,000个合成数据集进行终检。终检覆盖10类既往生成情景及三种零假设结构。会话jackknife分析在5%和1%家族水平下均通过全部30个情景的预定操作性标准，即家族误报率95%区间上界不超过名义水平的1.25倍。在5%水平下，最大观察误报率为4.875%，对应各情景区间上界最大值为5.183%；同时95%区间的最低覆盖率为95.125%。这些结果支持在所评估模型范围内使用该分析，但不构成对未知真实依赖结构的普遍保证。0.1%水平没有通过全部情景的标准。

在这一条件性分析下，原HIP标签组相对AMY、MTG及STG标签组的池化均值差分别为0.778、0.695和1.002 ms，校正后P值分别为0.0290、0.00726和0.0102。其余三项比较未达到5%家族水平。会话等权t分析得到相同的三项拒绝模式，但对应的是略有不同的汇总对象。我们将据此撤回原“全部比较p<0.001”的表述和相应统一显著性标记。

我们同时检验了方法的失败边界：当随机测量偏移跨会话共享时，会话层方法也不能可靠区分真实效应与共享偏差。因此，新结果将明确标为依赖会话层假设的补充证据，不作为无分布假设检验或跨受试者推断。原微电极解剖标签仍待确认，且部分差值对电极组成敏感，结果解释将限定于本受试者的记录触点及这些会话。上述修订以实际数据支持的证据强度替代原窗口级显著性声明。
'''
(ROOT/'response_ZH.txt').write_text(zh)
en='''Draft response to Reviewer 2.4.2
Status: supplementary analyses completed; manuscript edits have not yet been applied. Add page/line references and change future tense only after those edits are made.

We agree that the original window-level analysis did not adequately account for dependence. Overlapping delay estimates are not independent replicates, and neither Welch's test nor Bonferroni adjustment resolves this issue. We therefore do not retain the original window-level p<0.001 values as formal inferential evidence.

Keeping the NeuroTD estimator and estimated delays unchanged, we reanalysed the same participant's three sessions and 150 trials. We retained the original differences in pooled nonzero absolute-delay means, estimated standard errors by deleting one entire session at a time, and used a Student-t reference with two degrees of freedom, with Bonferroni adjustment for six two-sided contrasts. The reference degrees of freedom arise from the three sessions, rather than the number of windows. This analysis requires a defensible session-level sampling interpretation and sufficiently independent errors between sessions; it is not an estimate of a universal effective window count.

After comparing candidate procedures and documenting the post-hoc recommendation, we evaluated this procedure using a further 600,000 synthetic datasets that had not been used in the method comparison. The final evaluation included ten legacy generating scenarios and three global/partial-null configurations per scenario. All 30 configurations met the prespecified operational criterion at family levels of 0.05 and 0.01: the upper 95% binomial confidence bound for family-wise error was no greater than 1.25 times the nominal level. At the 0.05 level, the maximum observed family-wise error was 4.875%, the maximum upper confidence bound was 5.183%, and the minimum simultaneous 95% interval coverage was 95.125%. These are model-conditional validation results, not a proof of exact error control under arbitrary real-data dependence. The 0.001 level did not meet the criterion in all configurations.

Under this conditional analysis, the original HIP-labelled group exceeded the AMY-, MTG-, and STG-labelled groups by 0.778, 0.695, and 1.002 ms, with adjusted p-values of 0.0290, 0.00726, and 0.0102, respectively. The remaining three contrasts did not reach the 0.05 family level. A separately labelled equal-session t analysis yielded the same three rejections, although its aggregation target differs slightly. We will replace the original blanket p<0.001 statement and associated significance annotations with this qualified account.

We also explicitly tested a failure boundary: shared random measurement offsets across sessions can invalidate session-level inference. Accordingly, the new p-values will be presented as assumption-dependent supplementary evidence, not as distribution-free evidence or population-level inference. Microelectrode anatomical labels remain unverified, and some contrasts are sensitive to contact composition. Interpretation will therefore be restricted to this participant's recorded contacts and sessions. If session independence cannot be adequately justified, the main-text presentation will remain descriptive and the conditional inference will be confined to the supplement.
'''
(ROOT/'response_EN.txt').write_text(en)
methods='''Proposed manuscript replacement text — not applied to manuscript

Methods
Window estimates were not treated as independent replicates. For each original label-defined group, the descriptive estimator was the sum of nonzero absolute delays divided by the number of nonzero estimates across the three sessions. For each pairwise contrast d, we recomputed this ratio difference after deleting each complete session, giving d_(-s), s=1,2,3. The session-jackknife standard error was [ (2/3) sum_s {d_(-s)-mean_s d_(-s)}^2 ]^(1/2). We report conditional two-sided Student-t reference p-values with two degrees of freedom and Bonferroni adjustment for six comparisons, together with corresponding simultaneous 95% intervals. These are supplementary, assumption-dependent analyses: independence of session-level errors and an appropriate session-level sampling interpretation are required. They do not establish a corrected number of independent windows or generalize to multiple participants. The method recommendation was retrospective; a separate simulation set evaluated it after the comparison of candidate methods.

Results
In this single participant, the pooled nonzero absolute-delay differences for HIP-labelled versus AMY-, MTG-, and STG-labelled contacts were 0.778, 0.695, and 1.002 ms. Conditional session-jackknife adjusted p-values were 0.0290, 0.00726, and 0.0102, respectively. Corresponding simultaneous 95% intervals were [0.187,1.369], [0.432,0.958], and [0.552,1.453] ms. The other three contrasts did not reach the 0.05 family level. The original window-level Welch p-values are not used for inference. The labels are retained for provenance and do not indicate independently verified microelectrode anatomical locations.

Discussion
The supplementary analysis reduces the inferential replication level from individual windows to complete sessions and substantially weakens the original significance claims. Simulation evaluation supports the reported procedure within the tested generating scenarios, but does not establish validity under arbitrary dependence. In particular, shared random measurement offsets across sessions are not captured by deleting sessions. With only three sessions from one participant, the assumed independence and distributional approximation cannot be thoroughly verified. Contact-composition sensitivity and unresolved anatomical mapping further limit interpretation. The estimates should be understood as alignment-delay summaries of the recorded signals, not as established regional transmission times or population-level physiological differences.

Figure caption addition
Points show pooled nonzero absolute-delay contrasts; crosses show individual-session contrasts. Intervals are conditional simultaneous 95% session-jackknife t(2) intervals adjusted for six contrasts. Three sessions belong to one participant. Original anatomical labels are provisional. No individual window is counted as an independent replicate in this supplementary inference.

Implementation status
These paragraphs are proposed replacements only. The original manuscript and online response document have not been edited.
'''
(ROOT/'manuscript_replacement_EN.txt').write_text(methods)

# Standalone human-readable report; no external assets needed beyond relative local figures.
body=html.escape(report).replace('\n','<br>')
doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>NeuroTD 2.4.2 本机分析</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.75 system-ui;color:#203040}}img{{width:100%;margin:20px 0}}pre{{white-space:pre-wrap}}h1{{font-size:28px}}</style>
<h1>NeuroTD 2.4.2 本机分析</h1><p>2026-11-08 · 单受试者、三会话 · 条件性统计推断</p>
<img src="outputs/figure1_contrasts.png" alt="差值及会话级区间">
<div>{body}</div><img src="outputs/figure2_calibration.png" alt="第一套验证与假设挑战"><img src="outputs/figure3_sensitivity.png" alt="敏感性比较"></html>'''
(ROOT/'analysis_CN.html').write_text(doc)
save_json(ROOT/'outputs/delivery_facts.json',dict(recommended_results=rows,validation=summary.to_dict('records'),
    total_formal_simulation_datasets=1440000,window_estimator_rerun=False,original_code_modified=False,manuscript_modified=False,
    inference_scope='conditional_on_session_independence_and_other_stated_assumptions',
    unconditional_effective_df_correction_proven=False))
print(table)
