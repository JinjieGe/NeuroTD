# 修订实验运行手册

所有命令在仓库根目录执行，Python >=3.12。先安装项目及补充依赖：

```bash
python -m pip install -e .
python -m pip install -r examples/revision/requirements.txt
```

下文 `revision_runs/full` 可替换为实验室电脑的独立工作路径。统一入口会写运行命令、代码哈希、退出状态；实验原有随机种子、预先设定方案、检查点和自检保留。脚本搬移涉及路径，历史原始哈希和本轮哈希在 `source_manifest.csv` 中分别记录。

## 2.2、2.3、2.4.1：E、选窗和独立重复

```bash
python examples/revision/run.py run --workspace revision_runs/full --study score_uncertainty --script audit.py -- baseline
python examples/revision/run.py run --workspace revision_runs/full --study score_uncertainty --script audit.py -- run
python examples/revision/run.py run --workspace revision_runs/full --study score_uncertainty --script audit.py -- summarize
```

第一步创建新配置、核对基准并估计耗时。第二步为长任务；1200组有噪声信号之外另有确定性零噪声检查。不要把这些重复计为不同受试者。检查 `baseline/comparison.csv`，不能仅以退出码代替数值核对。`summarize` 需要完整检查点。

## 2.4.3：先固定窗，再将选窗纳入零分布

固定窗使用随包提供的干净合成信号：

```bash
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_fixed --script simulate.py -- prepare
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_fixed --script simulate.py -- run
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_fixed --script simulate.py -- report
```

完整选窗：

```bash
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_selection --script code/checks.py
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_selection --script code/pipeline.py
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_selection --script code/analyze.py
```

`checks.py`先生成/核对输入及逐点一致性，失败时不得跳过。原调度器含 macOS 的资源监测；其他平台应先检查资源检测回退及 worker 数，不要直接把实验室电脑的并行设置搬到低内存机器。完整流程的结果不能解释为真实数据动态曲线已经校准。

## 2.4.2：最新会话层统计

输入是随包保存的600行试次×原标签组汇总，不重新计算原始时延。

```bash
python examples/revision/run.py run --workspace revision_runs/stats --study session_inference --script code/run_analysis.py -- --pilot
python examples/revision/run.py run --workspace revision_runs/stats --study session_inference --script code/run_analysis.py
python examples/revision/run.py run --workspace revision_runs/stats --study session_inference --script code/verify_results.py
python examples/revision/run.py run --workspace revision_runs/stats --study session_inference --script code/final_validation.py
python examples/revision/run.py run --workspace revision_runs/stats --study session_inference --script code/sensitivity_and_figures.py
```

先阅读该模块的 `PLAN.txt` 和 `FOLLOWUP_VALIDATION_PLAN.txt`。后者增加600000个汇总层模拟数据集，不是600000次原始信号重估。旧 `audit_v1–v3` 不是推荐的一键工作流；重跑它们需要原始NWB及旧检查点、历史源版本，见缺项清单。

## 2.6：数据清点

```bash
python examples/revision/run.py run --workspace revision_runs/inventory --study dataset_reporting --script audit_inputs.py
python examples/revision/run.py run --workspace revision_runs/inventory --study dataset_reporting --script audit_inputs.py -- --nwb-dir /path/to/dandiset_000574
```

第一条检查公开处理后间隔表；第二条另读NWB试次元数据，不载入全部电压。原始Patch-seq处理与基因分析仍在 `examples/patchseq/`，原始数据与刺激单位须单独核实。本模块不做2.5。

## 2.7.1：扩展加噪声实验

`NEUROTD_RAW_DATA` 应指向包含 `sub-01/` 的目录。三个文件必须匹配脚本固定的DANDI资产及校验值。

```bash
export NEUROTD_RAW_DATA=/path/to/dandiset_000574
python examples/revision/run.py run --workspace revision_runs/snr --study snr_extension --script prepare_raw.py
python examples/revision/run.py run --workspace revision_runs/snr --study snr_extension --script extend_experiment.py -- prepare
python examples/revision/run.py run --workspace revision_runs/snr --study snr_extension --script extend_experiment.py -- run
python examples/revision/run.py run --workspace revision_runs/snr --study snr_extension --script interpret_results.py
```

缺少数据时第一条准备命令会停止；确需下载可在末尾显式加 `-- --download`，会下载较大的公开NWB文件。原初始真实数据代理指标分析还依赖 `audit_v1/checkpoints/windows_all_trials.npz` 和 `audit_v3/electrodes/pair_classification.csv`；不能用空文件代替。扩展加噪声准备已与该旧检查点解耦。

## 2.8.1：搜索范围

```bash
python examples/revision/run.py run --workspace revision_runs/range --study search_range --script run_search_range_ablation.py -- --workers 2
python examples/revision/run.py run --workspace revision_runs/range --study search_range --script analyze_search_range_ablation.py
```

保留运行前定义的两种有效中心规则、全部种子和失败结果。基准表属于历史本机复现，不是逐字抄录的论文表S1。

## 从保存表重画回复图

```bash
python examples/revision/run.py figures --workspace revision_runs/fig22 --study figures_22
python examples/revision/run.py figures --workspace revision_runs/fig23 --study figures_23
python examples/revision/run.py figures --workspace revision_runs/fig26 --study figures_26
python examples/revision/run.py figures --workspace revision_runs/fig28 --study figures_28
python examples/revision/run.py figures --workspace revision_runs/fig13 --study figures_13
```

每次使用新目录。`figures_26`是从已核对并嵌入脚本的汇总数绘图，不是从NWB重推这些数字。`figures_13`包含历史接触点敏感性图；不得称其为生物学机制验证。部分中文图需要系统中文字体。

## 轻量检查

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_simulation_validation.py' -v
python -m unittest discover -s tests -p 'test_revision_bundle.py' -v
```

第二组检查语法、2.5排除、E恒等式、两类干净信号、完整任务数、快速MSE与原函数逐点一致性、处理后数据计数以及不同代码不能复用旧工作目录。它不等于全量复现实验。
