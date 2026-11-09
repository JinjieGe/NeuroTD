# NeuroTD 修订实验与证据代码 / Revision evidence

本目录把评分、选窗、统计依赖、数据报告和 SNR 分析整理为独立模块。**本次不包含 Comment 2.5 的可靠性工作。** 原估计器 `src/neurotd/time_varying_shift.py` 未改动。

先阅读 [逐条意见与代码对应表](../../docs/revision-evidence-map.md)、[运行说明](../../docs/revision-running.md) 和 [证据边界与缺项](../../docs/revision-limitations.md)。路径从仓库根目录运行。

```bash
python -m pip install -e .
python -m pip install -r examples/revision/requirements.txt
python examples/revision/run.py list
python examples/revision/run.py init --workspace revision_runs/my_run
```

## 目录职责

| 路径 | 内容 |
|---|---|
| `run.py`, `revision_runtime.py` | 统一启动、路径配置、独立工作目录与运行记录 |
| `studies/score_uncertainty/` | 2.2、2.3、2.4.1：E、已知延迟、独立重复、不确定性 |
| `studies/dynamic_fixed/` | 2.4.3：固定窗口、步长、重叠重建 |
| `studies/dynamic_selection/` | 2.4.3：将完整选窗过程纳入恒定延迟零分布 |
| `studies/session_inference/` | 2.4.2：会话层条件性统计及额外校准 |
| `studies/dataset_reporting/` | 2.6：NWB 元数据、公开处理后 sweep 表计数 |
| `studies/search_range/` | 2.8.1：W/4 与 W/2 搜索范围消融 |
| `studies/snr_initial/`, `studies/snr_extension/` | 2.7.1：信号质量代理指标与加噪声敏感性 |
| `studies/figures_*/` | 回复图表生成；使用保存的表，不表示重新执行实验 |
| `studies/audit_v1/`, `audit_v2/`, `audit_v3/` | 前三轮统计与电极审计的历史代码；不能替代最新结论 |
| `inputs/` | 小型处理后输入和明确已知真值的合成信号；不含原始 NWB |
| `evidence/` | 历史运行的汇总表，和新运行结果分开 |
| `source_manifest.csv` | 原脚本与整理后脚本的 SHA-256、来源包及对应意见 |

`run.py` 会将脚本和必要输入复制到指定工作目录后再执行。工作目录内代码/输入已存在而内容不同会停止，避免用新代码继续旧检查点。不得在 `studies/` 内直接运行长实验。

## 一个可快速执行的例子

```bash
python examples/revision/run.py run --workspace revision_runs/inventory \
  --study dataset_reporting --script audit_inputs.py
```

它统计仓库中已公开的处理后间隔表，不会下载数据、重算所有时延或执行 2.5。

重新生成 2.2 的英文四联图（来自已保存的1200次模拟汇总）：

```bash
python examples/revision/run.py figures --workspace revision_runs/figure22 \
  --study figures_22
```

图会出现在 `revision_runs/figure22/figures_22/deliverables/`。图表工作目录标记为 `ARCHIVED_FIGURES_ONLY`，不能随后混用为新实验目录。

## 解释原则

- E 是代码的内部拟合误差准则；真实延迟 RMSE 是不同的评价量。
- Monte Carlo 重复单位是完整信号对，不是重叠窗口。
- 2.4.2 的新分析只支持所列模型假设下、一个参与者三个会话的条件性推断。
- 2.7 的合成加噪声 SNR 不是实际脑区的生理 SNR；不排除真实脑区的信号质量或形态混杂。
- 图表成功生成、代码成功运行、历史数值一致及生物学结论成立是不同的验证层级。
- 不上传审稿邮件、Google 文档副本、聊天记录、账号配置、原始电生理数据或大检查点。
