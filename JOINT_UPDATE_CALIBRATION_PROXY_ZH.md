# DynFL Sample-DP：Joint Update-Space Calibration Proxy（v3.5）

## 1. 目标

正式 Sample-DP 论文路径不再使用

`J_fusion + J_emb + J_grad + J_opt`

这样的独立标量 penalty 拼接。理论目标保持为同一个 round-update error 的条件二阶矩：

\[
J_{\mathrm{learn}}^t
=\mathbb E_t\|R^t\|_2^2
=\|e_{\mathrm{alg}}^t+e_{\mathrm{agg}}^t+b_s^t\|_2^2
+\operatorname{Tr}(V_s^t).
\]

客户端 Sample-DP trajectory error 定义为

\[
D_i^t=u_{i,s}^t-u_{i,0}^t,
\qquad
b_i^t=\mathbb E_t[D_i^t],
\qquad
V_i^t=\operatorname{Cov}_t(D_i^t).
\]

这里不要求训练映射为线性。`u_private` 和 `u_clean` 都由真实训练代码执行得到；校准只观察它们的输出差。

## 2. 三层必须分开

### 2.1 Exact theory

`D_i, b_i, V_i, J_learn` 是理论随机变量/条件矩，不是由 RDP 数字反推出来的。

### 2.2 Paired Monte-Carlo estimator

在独立 public/calibration workload 上，对相同模型状态、相同 mode、相同 minibatch schedule 做 matched clean/private trials：

\[
D_i^{(m)}=u_{i,s}^{(m)}-u_{i,0}^{(m)}.
\]

代码用流式 Welford 统计量估计

\[
\widehat b_i=\frac1M\sum_mD_i^{(m)},
\qquad
\widehat v_i
=\frac1{M-1}\sum_m\|D_i^{(m)}-\widehat b_i\|_2^2
\approx\operatorname{Tr}(V_i).
\]

只保存一个 update-sized 均值向量和一个 variance-trace scalar，因此额外统计内存是 `O(d)`，不是 `O(d^2)`。

### 2.3 Online selector proxy

在线每轮决策**不做 paired training**。Selector 只加载离线/周期刷新得到的 calibration table，按当前 state key、client、mode 查询：

- `clean_update_mean`：\(\widehat u_{i,0}\)
- `bias_mean`：\(\widehat b_i\)
- `variance_trace`：\(\widehat{\operatorname{Tr}(V_i)}\)

若 client privacy RNG 条件独立：

\[
\widehat b_s=\sum_i a_i\widehat b_i,
\qquad
\widehat v_s=\sum_i a_i^2\widehat v_i.
\]

在线 second objective 为

\[
\widehat J_{\mathrm{learn}}^{\mathrm{sel}}
=
\|\widehat e_{\mathrm{alg}}+\widehat e_{\mathrm{agg}}+\widehat b_s\|_2^2
+\widehat v_s.
\]

它保持与 exact objective 相同的结构，而不是重新发明另一套 penalty。

## 3. e_alg 两种策略

`joint_calibration_e_alg_policy=table`（正式推荐）：calibration table 还必须提供当前 state 的 ideal/reference update `Delta_*`。代码计算

\[
\widehat e_{\mathrm{alg}}=\widehat\Delta_{F,0}-\widehat\Delta_*.
\]

此时均值项可化简为

\[
\widehat e_{\mathrm{alg}}+\widehat e_{\mathrm{agg}}+\widehat b_s
=
\widehat\Delta_{P,0}-\widehat\Delta_*+\widehat b_s.
\]

`joint_calibration_e_alg_policy=zero`：显式采用 `e_alg≈0` 的经验近似，只用于消融/诊断，必须与 paired-MC ground truth 验证，不能写成恒等式。

## 4. 在线复杂度

代码实现了固定 admission 下的增量 joint-proxy lookahead。对一个单-client mode replacement，只更新：

- Cloud represented sample mass；
- \(\sum m_i(\widehat u_{i,0}+\widehat b_i)\) 向量 numerator；
- \(\sum m_i^2\widehat v_i\) scalar numerator；
- `zero-e_alg` 情况额外更新 full-clean numerator。

因此每个邻居的 learning lookahead 是 `O(d)`，不再每次扫描全部 clients，也不会触发训练 replay。

## 5. Calibration table 文件

实现：`dynfed/joint_calibration.py`

格式标识：`dynfl_joint_update_calibration_v1`

每个 cell 由 `(state_key, mode, client_id?)` 索引。Lookup 顺序：

1. exact client + requested state；
2. mode-level fallback；
3. 若 `state_key=round:k`，使用最近的已校准 round；
4. `default`。

当前正式配置用 `state_key=auto_round`，runtime 会在 round `t` 自动查询 `round:t`。

> 注意：校准 cell 必须来自与该 state/mode 下实际 Sample-DP 参数相匹配的 workload。若自动 RDP calibration 使噪声 multiplier 随状态明显变化，应在相应 state 上刷新 calibration，而不是把一张 mode-only 表无限期复用。

## 6. 自动采集 paired clean/private trajectories（v3.5）

v3.5 在现有训练执行器中加入了一个**显式开启、单独运行**的 calibration capture 路径。它不是在线 selector 的一部分。推荐流程是：

1. 先运行一个单-policy calibration pass；
2. 在指定 round 周期抽取少量 client / candidate mode；
3. 对每个 cell 跑一次 clean trajectory 和 `M` 次 private trajectory；
4. 直接把 `\widehat u_0, \widehat b, \widehat v` 与 `Delta_*` 写入 `.pt` table；
5. 正式训练只加载该 table，不再重跑 paired trajectories。

专用入口：

```bash
python experiments/run_joint_calibration_capture.py \
  --config configs/paper_v34_cifar10_resnet18_joint_proxy.json \
  --output calibration/paper_v34_cifar10_resnet18_joint.pt \
  --rounds 1 --trials 2 --period 1 --max-clients 1 \
  --sample-limit 16 --scope candidate_modes --dry-run
```

去掉 `--dry-run` 才会真正执行。`--trials` 至少为 2；正式校准应在估计稳定性实验后选择更大的 `M`。

### 6.1 clean/private 配对语义

clean path **不是**把 Sample-DP 的 Gaussian `sigma` 设为 0。代码会切换到同一训练程序的 non-private execution path，从而使 Sample-DP clipping distortion 也保留在

\[
D_i=u_{i,s}-u_{i,0}
\]

中。clean 与 private 共享相同初始模型、数据、local epoch、minibatch shuffle / training seed 和 optimizer schedule；private trials 只改变 `dp_seed`。因此这里直接估计 nonlinear training trajectory 的输出矩，不要求 Jacobian 或线性传播。

### 6.2 workload 与实际 step count

当前自动 capture 从每个 client 的 held-out validation pool 取 calibration 样本，并可用 `--sample-limit` 限制**唯一 held-out 样本数**。随后代码从该 pool 有放回重采样到该 client 的实际 training sample count，使 clean/private trajectory 的 minibatch / optimizer-step 长度与正常训练尽量一致。`Delta_*` 也在这些同一批重采样 calibration workloads 上计算，避免 reference objective 与 paired trajectory 使用不同样本质量。

### 6.3 重要隐私边界

仓库自动 split 出来的 held-out validation subset 只是工程上的 calibration workload，**不能自动等同于公开数据**。如果论文要声称 calibration 本身不产生额外隐私泄露，应把该路径替换/接入真正独立的 public calibration dataset。自动 capture 的统计方法不变，但数据来源必须在论文与实验配置中明确。

### 6.4 mode fallback 的含义

代码优先保存 `(state, client, mode)` 精确 cell。mode-level fallback 从已经估计好的 client conditional moments 构造：均值取 client moment 加权平均，variance 只平均各 client 的 **within-client DP-RNG variance trace**，不会把 client 间 bias heterogeneity 错算进 `Tr(V_i)`。fallback 仍然只是 population proxy；有 exact client cell 时优先使用 exact cell。

### 6.5 HE profiled calibration

专用 capture runner 默认 `he_execution=profiled`。在这个**辅助 calibration pass** 中，HE 只影响通信/密码执行，不改变本地 clean/private learning trajectory，因此即便本机没有 CKKS backend，也允许枚举 HE-protected collaboration modes 并用 modeled HE cost 完成 calibration。这个例外只对启用了 calibration capture 且显式使用 profiled HE 的运行生效；正式训练仍遵守原有 real-backend 规则。

## 7. 从 paired update 文件构建 calibration table

如果使用外部 public workload 或已有 paired update 文件，原 builder 仍然保留。脚本：

```bash
PYTHONPATH=. python experiments/build_joint_calibration.py \
  --manifest calibration/example_joint_calibration_manifest.json \
  --output calibration/joint_update_calibration.pt \
  --min-trials 5
```

Manifest 中每个 `pairs[]` 是一次 matched trial。`clean`/`private` 文件可保存：

- flat tensor / list / NumPy array；
- nested `{"end": {...}, "edge": {...}}` state difference；
- 或包含 `state_diff` 字段的 mapping。

`ideal_updates[]` 为 `table` e_alg policy 提供 \(\Delta_*\) reference。

## 8. 正式配置

示例：`configs/paper_v34_cifar10_resnet18_joint_proxy.json`

关键字段：

```json
"privacy": {
  "unit": "sample"
},
"learning": {
  "objective": "joint_calibration",
  "calibration_path": "calibration/paper_v34_cifar10_resnet18_joint.pt",
  "state_key": "auto_round",
  "e_alg_policy": "table",
  "missing_policy": "error"
}
```

`missing_policy=error` 是正式实验推荐值：缺 calibration 数据时直接失败，不允许悄悄退回旧的 fusion+DP objective。

## 9. Sample-RDP event count 对齐

Sample-RDP accountant 现在按 runtime 的实际 minibatch count，并乘 Flow 中真正重复执行 training block 的 `E_edge_loops`。通信 forwarding/HE 本身不额外生成 Sample-RDP event。

因此 privacy 路径与 learning 路径分工明确：

- Flow / actual mechanism executions -> unified Sample-RDP feasibility；
- 同一批真实 Sample-DP executions -> nonlinear private trajectory -> paired update-space moments -> joint learning proxy。

RDP epsilon 不能代数推出 `b_i,V_i`。

## 10. 验证要求

正式实验至少应抽样部分 rounds/candidates 做昂贵 paired-MC ground truth，并报告：

- Spearman rank correlation（主要）；
- Pearson correlation；
- top-k overlap / selected-candidate regret（可选）。

这证明 online calibration proxy 的排序确实代理理论 `J_learn`，而不是仅仅数值尺度相似。

## 11. 当前边界

本版本已经实现：calibration 统计器、table 格式、外部-file builder、自动 paired trajectory capture、在线 lookup、joint objective、增量候选 lookahead、round-conditioned lookup、Sample-only 正式 runner 约束和审计字段。

本版本**不会伪造 calibration 数据**，也不会在每次在线决策前重新测量。自动 capture 必须单独运行；正式配置只读取已生成的 `.pt` table，缺文件时 `missing_policy=error` 会按设计失败。

当前 `state_key=round:t` 仍是低维 state proxy：如果 calibration pass 与正式训练的模型轨迹差异很大，round number 并不能保证两者处于相同参数状态。因此论文实验仍应抽样计算 paired-MC ground truth 并报告 ranking correlation；后续可把 table key 扩展到模型/梯度/损失等可观测 state features。
