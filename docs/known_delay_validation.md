# 用已知真实延迟验证窗口选择

实际运行范围和已发现的失败情形见 [本次验证记录](known_delay_validation_checks.md)。

## 目的：检验 E 选窗，而不是用真值替代 E

原模拟已经计算四种方法的延迟 RMSE。这次新增的验证明确区分：

- **E**：当前代码的 `sqrt(mean(r_k))`，其中 `r_k` 为代码计算的逐窗归一化对齐误差。越小越好；不是论文公式 (6) 的 q。
- **延迟 RMSE**：估计延迟与模拟真值的均方根差，按 `1000/fs` 换成 ms。
- **E 选窗**：只读取 E，选择最小 E 的候选窗口，不读取真实延迟。
- **oracle 选窗**：事后利用真值，选择延迟 RMSE 最小的候选窗口，仅作模拟参照。
- **额外误差**：E 选窗的延迟 RMSE 减去 oracle 的延迟 RMSE。越小，说明这次选窗损失越小。

这不是把所有输出改成 oracle 结果。原算法、原噪声生成流程、原图表计算和默认种子不变；原有脚本在跑完网格后新增验证目录。核心估计器及 E 函数没有修改，FFT/CTW 基线也没有校正。

## 运行

Python >=3.12；先在项目根目录、正确的 Python 环境中执行 `python -m pip install -e .`。

先做小规模测试（2 噪声 × 2 窗口，连续模拟仍包含原脚本的单次轨迹计算）：

```bash
python examples/validate_known_delays.py --simulation both --smoke
```

完整模拟（每种 11 噪声 × 6 窗口；默认离散 seed=0，连续 seed=2025）：

```bash
python examples/validate_known_delays.py --simulation both
```

多随机种子，建议先小规模确认耗时，再预先确定正式实验种子：

```bash
python examples/validate_known_delays.py --simulation both --smoke --seeds 0 1 2
python examples/validate_known_delays.py --simulation both --seeds 0 1 2 3 4
```

每次默认生成新的 `validation_runs/<时间戳>/`，无需改路径或覆盖旧结果。也可使用 `--output-dir 新目录`；目录已存在时拒绝运行。程序采用无界面的绘图方式，图片直接保存，每个模拟的实时进度在其 `run.log` 中。日志中出现历史名称 quality/q 不代表它是论文 q，以新验证文件的 E 定义为准。

不要在源码顶端手工改种子：入口会使用 `NEUROTD_SIM_SEED`。离散模拟改变测量噪声；连续模拟也会改变随机峰和延迟曲线，因此多种子连续实验不是固定波形下纯噪声重复。候选窗口之间复用同一 seed 的噪声实现，避免用不同噪声比较窗口。

## 读取结果

每个 `discrete_seed_*` / `continuous_seed_*` 下的 `known_delay_validation_*` 包含：

| 文件 | 内容 |
|---|---|
| `candidate_metrics.csv` | 每种方法、每个噪声及窗口的 RMSE、MAE、有符号偏差、有效点数；仅 NeuroTD 有 E 及真值超出搜索范围的比例 |
| `window_selection.csv` | E 所选 W、oracle W、各自延迟 RMSE、额外误差、误差比、是否达到 oracle 的误差 |
| `delay_traces.csv.gz` | 四方法的中心点、估计延迟、真值、逐点误差，单位 samples；可直接由 pandas 读取 |
| `selection_vs_truth.png` | NeuroTD 的 E 选窗 vs oracle 延迟 RMSE，以及额外误差 |
| `four_methods_known_delay_rmse.png` | 每个固定 W 下四种方法的延迟 RMSE，不偷偷为各方法挑不同窗口 |
| `summary.md` | 当前 seed 的描述性结果、指标定义及限制 |

入口顶层另存 `all_window_selections.csv`（各次运行合并表）和 `run_manifest.json`（解释器、依赖版本、源码哈希、Git 信息和运行状态）。不自动生成显著性结论。

### 两种评价时间点

- `native_centers`：各候选窗口及各方法原本输出的有效中心点，与原模拟表的 RMSE 定义一致。
- `shared_centers`：所有候选窗口和四种方法**共同拥有的准确中心索引**；不插值，不外推。用于检验评价时刻变化的影响。

共同点可能很少，甚至没有。必须同时报告 `n_shared_centers`，不能把它当作更可靠的大样本评估。共享中心也不能消除四种方法搜索范围和取信号方式的差别。

最小值使用完整精度，严格相等时选较小 W；E 的平局绝不用真实 RMSE 打破。RMSE 最优标记容差为 `rtol=1e-10, atol=1e-12 ms`。oracle RMSE 为 0 时误差比留空，仍报告绝对额外误差，避免除零。

## 科学解释边界

1. 本次保留原脚本真值约定：离散模型按最近分段中心取生成延迟；连续模型按输出中心取生成延迟。窗口内可能包含不同真实延迟。
2. 连续生成式 `y(t)=x(t-Delta(t))` 的 Delta 是 y 时间轴上的生成延迟；严格的 x 参考轴对应位移应通过反映射求出，通常不完全等于 `Delta(t_x)`。这次为保持论文复现约定，不偷偷改变真值定义；它是后续精确时间映射审计的限制。
3. 已知模拟真值提供独立于拟合残差的评价目标，但不是独立留出数据；如果根据这些结果继续调整方法，应在新的预先确定条件/种子上验证。
4. 单 seed、多个噪声水平、重叠窗口不能当成独立重复样本；本次不对逐窗误差计算显著性或置信区间。多 seed 验证也不自动证明真实神经数据中的生理有效性。
5. 若 E 选窗不理想，应如实报告额外误差；不能改用 oracle 曲线冒充无真值条件下的表现。也不能仅凭成功执行就宣称解决审稿人的全部质疑。

## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖真值误差计算、单位转换、E 与 oracle 分离、E 平局无真值泄漏、零误差除法、共同中心及缺失中心、非有限输入、超出搜索范围标记和输出保护。
