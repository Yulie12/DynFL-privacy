# DynFL Stage 13：真实预决策 LIIC 输入自动接入（不伪造证明）

## 这次一口气做完的范围

在 `dynfed/fmnist_lenet5_dynamic.py` 每轮 **候选枚举之后、Pareto 计算之前、Client 本地训练之前**，直接从正在运行的 LeNet-5 和候选枚举中读取：

- 当前完整模型的**可训练参数**范数与维数（GN-v2 61706）；
- 每个 Client 当前被分配用于训练的样本数；
- 真实 `train_batch_size`（包括不足一批的尾批）、`local_epochs` 与是否限步；
- 训练 LR、weight decay、Sample-DP-SGD 的逐样本裁剪阈值与**当前候选** optimizer sigma；
- `privacy_unit=sample` 的加噪逐样本梯度 SGD，与非样本隐私的普通裁剪 SGD 的分支区别。

这些值不再写死在假设性 LIIC proof 里。只在源证明的 `liic_sgd_proof` 显式包含 `runtime_fields_from_trainer: true` 时进行自动填充。若已有字段与真实运行不一致，则失败；**不**自动补入 `ideal_step`、理想梯度参考、`ideal_rms`、更新/聚合等价性布尔值、DP 声明或额外强偏差上界。

### 运行时完整接入

沿用现有 Stage 11 配置和 `online_upper_bound` 开关；该配置必须指向有效且已经独立证明的 `dynfl.online_j.v2` manifest。源文件逐轮、逐 Client、逐候选的模式证书仍然是强制要求，不能由训练器代造。训练器生成 `full_dynfl/online_j_predecision_round_000.json` 等文件，并让 **同一轮的 Pareto 选择器直接读取自动修订后的 manifest**。

外部 manifest 中为某一个 `LIIC` 候选提供的结构示意（其余字段遵守现有 v2 schema）：

```json
{
  "runtime_mode_equivalence_verified": true,
  "local_dp_included_in_mode_rms": true,
  "aggregate_share_std": 0.0,
  "liic_sgd_proof": {
    "runtime_fields_from_trainer": true,
    "ideal_step": 0.15,
    "evidence": "这里必须标注真实、可独立核查的数学证明出处",
    "training_state_matches_runtime": true,
    "ideal_reference_bound_proved": true,
    "noise_distribution_matches_runtime": true,
    "extra_transformations_absent_or_modeled": true
  }
}
```

**上面的 `true` 和数字只是字段格式，不是证明。** 不可以直接复制此片段声称现实系统已认证。所有 v2 顶层的条件、真实 Client 理想更新参考及其他模式证书，依然都要独立核查。未证明时选模器保持**失败即停止**，绝不替换成旧的 Fusion/经验半径；v1 静态 manifest 入口保持兼容。

### 目前就能执行的真实五轮运行时检查

即使还没有真实数学证书，也能对已跑通的 **旧目标训练** 开启无侵入 shadow 采集。此功能只读当前模型和候选元数据，不执行额外 SGD，不查看 Client 训练梯度或这轮训练完成后的更新，不改变优化器、DP 账本、预算或 mode 的选择。

在 Windows PowerShell，先执行 `$env:DYNFL_ONLINE_J_SHADOW_AUDIT='1'`，随后按原来的 Stage11 `legacy` 五轮命令运行。每一轮会生成：

- `full_dynfl/online_j_liic_shadow_round_000.json` 至 `...004.json`

这些文件的 `formal_online_j_certified` 明确为 `false`。它们只是下一步真实证书校验的运行时证据，并不是严格学习误差上界。结束后可以 `Remove-Item Env:DYNFL_ONLINE_J_SHADOW_AUDIT`。

### 安全界限

- 条件于理想梯度的证明无法由优化器参数、历史方差或当前私有梯度估计代替。
- 未证明 Split/Edge 多阶段贡献线性等价时，不能套用 LIIC 半径；系统不会替它们合成半径。
- LIIC 本地 Sample-DP-SGD 高斯扰动已包含在 LIIC 模式半径里，不可在 Cloud 的 `Q` 再加一次。
- 多阶段 Client 状态、实际通信更新的进一步裁剪与数值误差需单独验证。
- Shadow 采集文件及其参数范数属于私有实验诊断日志，不应未经审查当作公开 DP 输出。
- 验证范围是代码单元及真实 LeNet-5 模型结构；不能把目前没有证书的现实5轮说成新 J 完整对照。
