# 修订意见—代码—证据对应表

整理日期：2026-11-08。这是代码索引，不是审稿意见已全部解决的声明。审稿编号依照本项目回复文档；2.6 对应原邮件 Major Comment 5。2.5 明确排除。

| 意见 | 代码入口（`examples/revision/studies/` 下） | 验证内容 | 限定 |
|---|---|---|---|
| 1.1 / 1.3 | 原始 `time_varying_shift.py`；`score_uncertainty/`；`figures_13/` | MSE 与相关的条件性联系；特定条件下的性能；已知真值评价 | 文献/创新定位本身不是新实验；不声称所有基线已重新校准 |
| 1.4 | `examples/`, `data/`, `docs/`, `tests/` 的既有重组 | 核心包、示例、数据、图片职责分离 | 此整合继承 `reviewer1-4-restructure` 分支 |
| 2.2 | `score_uncertainty/audit.py`, `summarize.py` | 原 E、oracle、选窗损失，1200个有噪声信号对 | 干净模板固定；重复波形高噪声失效必须保留 |
| 2.3 | 同上；`figures_23/build_figure.py`；`search_range/` | 候选窗口评分、选择频率、搜索范围敏感性 | 频率不等于准确性；不宣称消除了所有大窗偏向 |
| 2.4.1 | `score_uncertainty/` | 跨独立重复的均值方差、窗口协方差、E 的 SD 和 MCSE | 不把 M 替换为有效样本量来改 E |
| 2.4.2 | `session_inference/code/` | 保留合并非零绝对延迟均值目标，会话删除 jackknife，t(df=2)，六项 Bonferroni | 校准条件有限；不是人群推断；不是恢复原窗口级 p<0.001 |
| 2.4.3 | `dynamic_fixed/` → `dynamic_selection/` | 4193对固定窗信号；29379对完整选窗信号，误报率和检出率 | 动态拒绝不等于轨迹正确；不自动校准真实记录 |
| 2.6 | 原始 `examples/ieeg/`、`examples/patchseq/`；新增 `dataset_reporting/`；`figures_26/` | 原始流程、人数/试次、处理后 sweep 数、图示 | 元数据清点不等于原始信号全复现；不包含同条件可靠性分析 |
| 2.7.1 | `snr_initial/` → `snr_extension/` | 36模板、8个真值、11个SNR、2噪声、2加噪方式、20重复；253440组加噪对 | 一人三会话；SNR按新增噪声定义；共同模板由构造保证 |
| 2.7.2 | 原始数据配对代码；`audit_v3/` 中目标函数/电极诊断 | 配对方式和解释边界 | 生物学文献不是波形对应关系的实验证明；未发布新的“已排除形态混杂”结果 |
| 2.7.3 | 原始 iEEG 脚本的有符号输出和绝对值汇总 | 解释符号、配对顺序与分析范围 | 本目录没有新增因果或方向性传输估计 |
| 2.8.1 | `search_range/run_search_range_ablation.py` | 搜索半宽 W/4 与 W/2、共同/各自有效中心比较 | 实验性消融不修改正式估计器 |
| 2.8.2 | `dynamic_fixed/`；原 iEEG/Patch-seq 调用链 | 区分窗内加权、延迟重建、显示处理 | 图4/5未使用重叠加权延迟平滑；不要把图3机制转移到所有应用 |
| 2.8.3 | `score_uncertainty/` | W=1025、重复波形与较低残差下的错误对应 | 失败条件保留，不按最终排名删去 |
| 2.8.4 / 3.1 | `session_inference/`；`figures_28/`；`audit_v1–v3/` | 会话差异、计数、接触点组成及敏感性 | 三个会话不是三个参与者；解剖标签仍需核实 |
| 3.2 | 原 `examples/patchseq/`；`dataset_reporting/`；`figures_13/` | sweep→ISTD→细胞→基因关联的原始流程与计数 | 关联不是基因机制；没有新提交 Metascape；2.5另行处理 |

## 当前文档与较新聊天的版本差异

两份指定文档的内容并非完全同步。详细回复中仍有把2.4.2标为未完成的段落；后续聊天已报告会话层条件性校准，因此该代码单独保留为最新入口，并保留失败边界。某些文档尚未包含最新2.7加噪声扩展；此处仍纳入其可核对代码。

历史审计 `audit_v1–v3` 保留是为了追溯方法如何被否定、修改和核查；其旧显著性结果不应覆盖 `session_inference` 的条件性结论。整合操作没有改写在线回复，也没有把计划中的分析标为已完成。

## 已有分支整合

- 文件组织来源：`reviewer1-4-restructure`，提交 `9c998f2`。
- 已知延迟验证来源：`validation/known-delay-window-selection`，提交 `01637a1`（含 `b214938`）。
- 本轮整合修复了两个分支合并后示例路径的衔接；估计器与 E 未修改。

已知延迟验证的独立入口仍为 `python examples/validate_known_delays.py --simulation discrete --smoke`，详见 [原验证说明](known_delay_validation.md)。
