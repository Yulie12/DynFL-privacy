# Stage 16 — 100轮批量执行 + 结构化在线 J 区间证明

## 安装范围

Stage 15 已安装后，用一个增量 Patch 更新。仅修改
`dynfed/online_j_bound.py`、`dynfed/selection.py`，新增批量实验入口、测试和本说明。
**不改** DP 预算、RDP、实际噪声、模型训练器、Split/Edge 运行和准入。

## 100轮批量实验

运行 `run_dynfl_100round_and_J.ps1` 即自动处理：

1. 用100轮的**最终固定RDP目标**，为 LeNet-5/FashionMNIST、ResNet-18/CIFAR-10 各跑 seed2026/full_dynfl 的前10轮。
2. 检验每一新轮的 `round_metrics.csv:actual_optimizer_steps > 0`；任一新轮无训练，马上停止整批，不通过改预算或禁用DP掩盖。
3. 通过后，对每个模型、3个种子、4个原配置基线单策略串行推进至100轮。每次用 `--max-new-rounds 10`，下一批使用 `--resume-from-run` 读取**同一策略**的前一批 checkpoint。
4. 每批完成后原子保存 `batch_state.json`，最终写 `batch_results.csv`。中途如有崩溃/异常，自动禁止回退到早期账本重跑已经尝试的发布；只能人工核验失败批次的最新预先记账检查点后处理。

入口会保留原配置的 train batch=256、epochs=3、无 local batch cap、模型/样本数、模型特异的原学习率、四策略、HE profiled；只改运行目标为100轮，不加任何额外隐私预算。**100轮训练不会被当作原隐私声明或正式端到端DP证明**，应单独审核发布实现及账本。

预检命令：

```powershell
& "D:\soft\Python310\python.exe" experiments/run_100round_batch.py --dry-run
& ".\run_dynfl_100round_and_J.ps1" -PrecheckOnly
```

如果10轮均有有效训练、需要完成剩余批次：

```powershell
& ".\run_dynfl_100round_and_J.ps1"
```

也可以直接执行最后一条，自动预检并在通过时接着跑满100轮。中断后保持 `out/stage16_100round_batch/` 不变，重复该命令可继续**已成功提交的**批次。

## 新的 J 证明性下界

原 `online_upper_bound` 结构化上界保持不变：

\[
U = \left(\sqrt{(A+R)^2+Q_{\mathrm{upper}}}+N\right)^2,
\]

Stage 15 的逐块上界取两条严格上界中的较小值。

旧的下界保守地忽略噪声：

\[
L_0 = [\max\{0,A-R-N\}]^2.
\]

但是，仅在**额外证书**证明本轮所有已选 Cloud 贡献中各 participant 的累计聚合噪声标准差是**精确值**，并且噪声对干净学习更新条件正交时，才能利用噪声方差作为下界：

\[
Q_{\mathrm{exact}}=d\sum_i (\gamma a_i\sigma_i)^2,\qquad
L=\left[\max\left\{0,\sqrt{[\max(0,A-R)]^2+Q_{\mathrm{exact}}}-N\right\}\right]^2.
\]

若有严格分块半径 \(R_b\)，则 \(c_b=\max(0,\|A_b\|-R_b)\) 形成独立坐标块的干净误差下界：

\[
L_{\mathrm{block}}=\left[\max\left\{0,\sqrt{\sum_b c_b^2+Q_{\mathrm{exact}}}-N\right\}\right]^2.
\]

最终取兼容的两个下界较大者。**方差上界不是方差下界**。只有 `aggregate_noise_variance_exactly_proved=true`、`assumptions.aggregate_noise_orthogonal_to_clean_update_given_predecision=true`，且每个已选证书 `aggregate_share_std_exactly_proved=true`，才会启用正值方差下界。否则新的 `q_aggregate_proved_floor` 为0，原保守下界不变。

新辅助函数 `certified_j_interval_order(left,right)` 只有当两个合法的条件区间不重叠，即 `left.upper < right.lower` 时，才认证左边的真实条件学习误差更小；重叠时返回 `undetermined`。两项区间必须由调用者确保对应同一轮、同一理想学习目标。选择器审核新增 `online_j_proved_noise_floor` 列。

Stage 16 **没有**从先前的450条 LIIC Shadow 记录推造理想梯度或给 Split/Momentum 凭空补证书；在真实七模式证书缺失时仍不能运行正式 `online_upper_bound` 对照。以上证明增强已使用独立公开合成证书做单测。
