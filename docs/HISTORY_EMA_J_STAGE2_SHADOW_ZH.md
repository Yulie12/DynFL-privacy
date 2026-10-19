# History-EMA J：Stage 2 影子评估与历史回测

## 本次方法边界

本实现只增加新方法的影子评估，不接管正式 Mode 选择。既有 `learning_objective="online_upper_bound"`、证书校验、失败关闭规则和 Pareto/选择流程未被替换。

- \(J\) 以历史全局更新 EMA 为参考，计算候选 Mode 的历史更新 EMA 预测与该参考之间的归一化平方偏离：
  \[
  J(h)=\frac{\|\widehat{\Delta}_{C}(h)-\bar{\Delta}_{G}\|_2^2}
  {\|\bar{\Delta}_{G}\|_2^2+\varepsilon}.
  \]
- \(\bar{\Delta}_{G}\) 仅从合格的 Cloud 全局更新更新 EMA；候选 Mode EMA 仅从可确认其输入更新在上传前经过 DP 保护的实际 Cloud 提交贡献更新。
- Cloud 聚合使用的真实样本量权重 \(I_i\) 被保留。混合 Mode 轮次中，不会把某一 Mode 的子集权重重新归一化为 1。
- LIIE 在当前源码里是 Edge-only，不向 Cloud 提交。因此它的 Cloud 贡献按明确的 no-op 处理，日志状态为 `no_cloud_submission_zero_contribution`；不会伪造一个 Cloud 数据包。
- LIIEIIIC 的联合单元是同一 Edge 对应的一份 `cloud_updates` 记录，使用组的样本数权重。`E_edge_loops` 只作为执行路径元数据，不乘到贡献上。
- 本方法不对 `E_C_upd` 加保护，不使用 HE，也不使用 aggregate-DP 作为 J 的历史样本。若旧执行配置实际走 aggregate-DP，或上传前 DP 保护不能从运行路径确认，则该轮会被排除，而不是当作有效历史。
- 新 J 目前是影子诊断，**不会替换**正式 `online_upper_bound` 选择器，也不代表 J 与模型精度存在已证明的单调关系。

## 开启影子采集

在项目根目录设置环境变量后运行现有训练命令：

### 建议先做配置检查

```bash
python experiments/run_paper_config.py --config configs/history_ema_j_fmnist_lenet5_shadow.json --dry-run
```

该专用配置会自动设置 `DYNFL_HISTORY_EMA_J_SHADOW=1` 和 EMA 衰减参数。若从其他入口运行，可显式设置：

Windows PowerShell:
```powershell
$env:DYNFL_HISTORY_EMA_J_SHADOW = "1"
$env:DYNFL_HISTORY_EMA_J_DECAY = "0.9"
```

Linux / macOS:
```bash
export DYNFL_HISTORY_EMA_J_SHADOW=1
export DYNFL_HISTORY_EMA_J_DECAY=0.9
```

`--dry-run` 仅检查配置；实际采集仍必须指向有效、与新候选身份匹配的真实 `online_upper_bound` 证书文件。当前仓库没有该真实证书，正式训练会按原机制失败关闭；影子功能不会替正式选择器生成或替代证书。

## 输出

- `history_ema_j_shadow.jsonl`：每轮候选预测、实际观测、跳过原因和全局 EMA 更新事件。
- `history_ema_j_state.pt`：用于恢复的 EMA 向量状态；与模型 checkpoint 一样属于本地实验状态，不应当作公开输出。

JSONL 的 `event` 主要取值：

- `prediction`：决策前可得的 J 预测。首次无历史时 `predicted_j=null`。
- `realized_j`：本轮实际观察的 Mode 贡献与**本轮之前**的全局 EMA 的偏离。
- `observation`：可用于更新 Mode 历史 EMA 的实际上传前 DP 保护贡献。
- `global_ema_update`：本轮全局 EMA 更新成功或被跳过的原因。

LIIE 的 `realized_j` 只有在存在历史全局 EMA 且该轮确实选中 LIIE 时才记录；因为 LIIE 没有 Cloud 提交，不能从 LIIE-only 训练中推导 Cloud 全局更新历史。

## 历史回测

```bash
python experiments/backtest_history_ema_j.py path/to/history_ema_j_shadow.jsonl \
  --output path/to/history_ema_j_backtest.json
```

默认只统计没有同时选择多个 Mode 的轮次，避免把混合选择轮次的某个 Mode 边际贡献误当作“该 Mode 单独运行”的完整 Cloud 更新。需要诊断混合轮次时可显式传入 `--include-mixed`。

报告给出 MAE、RMSE、Pearson 与 Spearman 相关。预测和观测必须同轮、同 Mode 配对；缺失历史、未确认 DP 保护、aggregate-DP 或其他不合格路径不会填零参与误差。

## 修改文件

- `dynfed/history_ema_j.py`：EMA 状态、J 公式、状态保存/恢复。
- `dynfed/selection.py`：仅在影子环境变量开启时，限制候选为三种目标 Mode，关闭 HE 候选，并移除 LIIEIIIC 的 E_C_upd 保护候选。
- `dynfed/fmnist_lenet5_dynamic.py`：决策前预测与 Cloud 聚合后影子记录；默认关闭，且不替换正式选择算法。开启影子协议时，运行候选限定为三种目标 Mode，关闭 HE 与 aggregate-DP，并将 LIIEIIIC 的 E_C_upd 设为无保护。
- `experiments/backtest_history_ema_j.py`：JSONL 历史回测。
- `tests/test_history_ema_j_shadow.py`：公式、历史更新、状态恢复和回测配对测试。

## 当前限制

1. 当前仓库中 LIIE 是 Edge-only，不能把它描述为真实 Cloud 提交模式。LIIE 的 no-op J 需要一个可用的历史全局 EMA 作为参考。
2. 当前旧运行配置允许 aggregate-DP/HE 候选。启用 `DYNFL_HISTORY_EMA_J_SHADOW=1` 后，目标三种 Mode 的候选配置会切换为上传前 DP 保护、无 E_C_upd 保护、关闭 HE 和 aggregate-DP；正式选择器算法与 `online_upper_bound` 证书失败关闭规则保留。由于候选身份改变，真实证书必须按新候选身份重新准备；本仓库未伪造证书。
3. `online_upper_bound` 证书机制保留，但旧证书针对理想全梯度误差目标，不会自动变成历史 EMA-J 的证明。
4. 本次只接入 FMNIST/LeNet-5 主训练路径；其他训练入口尚未接入。
5. 本次无法修改论文主文件 `main(4).tex`，因为当前可读取的源码 ZIP 不含该文件，而先前上传的 TeX 文件在当前运行环境无法提取正文。请将 TeX 文件作为纯文本重新上传，或把它放入项目包后重新压缩；代码侧修改已单独交付。
