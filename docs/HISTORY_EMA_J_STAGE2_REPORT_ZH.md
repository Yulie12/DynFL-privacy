# DynFL History-EMA J Stage 2 修改与验证报告

## 交付状态

已对上传的 `DynFL-privacy-feature-sample-dp-multistage-audit (8).zip` 进行源码修改，新增 History-EMA J 的影子评分、运行时记录、历史回测和 DP 边界开关。既有正式 `online_upper_bound` 证书评估器及其失败关闭规则仍保留，新的 J 不参与正式 Mode 选择。

论文原文件 `main(4).tex` 没有包含在这次 ZIP 内；此前单独上传的 TeX 在当前执行环境中无法提取正文。因此没有冒充已修改主论文，而是交付了可合并的 LaTeX 替换片段 `paper/J_history_ema_replacement.tex`。取得可读取的主论文后，还需要将片段并入原文并统一公式编号、引用和相关段落。

## 已确定的实现约束

- \(J\) 比较候选 Mode 的历史更新 EMA 预测与历史全局更新 EMA 的归一化平方偏离：
  \[
  J(h)=\frac{\|\widehat{\Delta}_C(h)-\bar{\Delta}_G\|_2^2}
  {\|\bar{\Delta}_G\|_2^2+\varepsilon}.
  \]
- Cloud 聚合使用真实样本量权重 \(I_i\)。混合 Mode 轮次不会把某个 Mode 的子集重新归一化为权重和 1；Mode EMA 仅在可识别的单一 Mode 完整 Cloud 更新轮次更新。
- LIIE 在当前源码中是 Edge-only，没有 Cloud 提交。影子逻辑将其记为明确的 no-op Cloud 贡献，不构造虚假提交记录。
- 同一 Edge 下的 LIIEIIIC 客户端作为一个联合 Edge→Cloud 单元；\(E_B\) 不作为贡献乘数。
- 在 `DYNFL_HISTORY_EMA_J_SHADOW=1` 的新方法运行中，LIIEIIIC 的 `E_C_upd` 不保护；LIIC 使用 `L_C_upd` 上传前 packet-DP；LIIEIIIC 使用 `L_E_upd` 上传前 DP；HE 候选和 aggregate-DP 路径被关闭。
- 影子运行只对可确认符合上传前 DP 保护边界的更新记录历史；不符合条件的轮次明确跳过。

## 文件变更

### 修改的已有文件

1. `dynfed/fmnist_lenet5_dynamic.py`
   - 新增可选影子运行开关，默认关闭。
   - 决策后、本轮训练前记录三种目标 Mode 的 J 预测。
   - Cloud 聚合后记录实际 Mode 更新、实际全局更新、排除原因。
   - 使用真实 Cloud 样本量权重构造模式贡献；LIIEIIIC 每个 Edge 只形成一个联合输出。
   - EMA 状态写入 `history_ema_j_state.pt`，影子事件写入 `history_ema_j_shadow.jsonl`。
   - 新逻辑不会把 J 传入正式选择器。

2. `dynfed/selection.py`
   - 只在新方法影子开关开启时，将候选范围限制为 LIIE、LIIEIIIC、LIIC。
   - 对这三种 Mode 关闭 HE 候选。
   - 将 LIIEIIIC 的 `E_C_upd` 标记为不保护的边界，不将其当作 DP 机制选择点。

3. `experiments/run_paper_config.py`
   - 支持专用 History-EMA J shadow 配置。
   - 从配置读取开关及 EMA 衰减参数。
   - 仅对新 shadow 配置放宽 DP-only 的配置验证；其他正式配置仍沿用原验证要求。

### 新增文件

- `dynfed/history_ema_j.py`：归一化 J、历史 EMA、状态持久化和 JSONL 记录。
- `experiments/backtest_history_ema_j.py`：按 round/Mode 配对预测和实际观测，计算 MAE、RMSE、Pearson、Spearman。
- `tests/test_history_ema_j_shadow.py`：J 公式、EMA 更新、状态恢复、回测配对、三 Mode 范围及 DP 边界测试。
- `configs/history_ema_j_fmnist_lenet5_shadow.json`：FMNIST/LeNet-5 新方法影子配置模板。
- `docs/HISTORY_EMA_J_STAGE2_SHADOW_ZH.md`：使用说明、输出字段、回测规则及限制。
- `paper/J_history_ema_replacement.tex`：论文 J 定义替换片段。
- `paper/README_J_history_ema_replacement.md`：主论文合并说明。

## 验证结果

| 验证 | 结果 |
|---|---|
| 新增/修改 Python 文件 `py_compile` | 通过 |
| 新增 History-EMA J 测试 | 6 passed |
| `test_selection.py`、`test_online_upper_bound_integration.py`、`test_online_j_runtime_integration.py`、`test_stage11_locked_batch_and_no_cap.py` 与新增测试合并运行 | 80 passed，5 subtests passed |
| 专用配置 `run_paper_config.py --dry-run` | 通过 |
| 全量 `python -m pytest -q tests` | 执行至约 82% 后超过 5 分钟超时；超时前未见失败摘要，不能据此宣称全量通过 |
| 真实训练与历史回测 | 尚未执行：仓库不包含 `certificates/REAL_history_ema_j_fmnist_lenet5_predecision_required.json` |

Python 启动时出现与本项目无关的 spreadsheet warmup 日志，但上述 `py_compile`、定向 pytest 和配置 dry-run 的返回码均为成功。

## 未解决事项

1. **主论文尚未直接合并。** 当前 ZIP 不含 `main(4).tex`，之前的单独上传在本执行环境不可读。LaTeX 替换片段已准备好，但不能声称全文公式和交叉引用已经完成修改。
2. **真实证书缺失。** 专用配置保留 `learning.objective=online_upper_bound`，并使用新的候选身份。必须提供对应的有效真实证书，正式训练才会继续；没有伪造合成证书。
3. **未完成真实历史回测。** 目前验证了合成向量和日志配对，不代表真实 Fashion-MNIST 模型上 J 具有预测能力。
4. **其他训练入口未接入。** 本次运行时影子接入针对 `dynfed/fmnist_lenet5_dynamic.py`；CIFAR-10/ResNet-18 及其他训练入口尚未接入。
5. **旧代码未整体删除。** 原有 HE、aggregate-DP 和在线证书代码仍供旧实验路径使用；新方法的专用运行开关不使用 HE/aggregate-DP，且正式 `online_upper_bound` 证书机制未被替换。这样可避免在没有新证书和完整回归验证时破坏旧主线。

## 推荐的下一步

1. 将 `main(4).tex` 重新上传为可读取的文本，或放入源码 ZIP 后重传，以便直接合并论文修改。
2. 提供新的真实 `online_upper_bound` 证书或其生成依据，确认三 Mode 的候选身份。
3. 使用同一数据划分与模型设置运行影子采集，生成 `history_ema_j_shadow.jsonl`，再运行历史回测脚本。
