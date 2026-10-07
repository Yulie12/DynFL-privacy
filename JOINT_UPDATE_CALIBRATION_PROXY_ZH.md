# DynFL Sample-DP：Joint Update-Space Calibration Proxy（v3.4）

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

## 6. 从 paired update 文件构建 calibration table

脚本：

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

## 7. 正式配置

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

## 8. Sample-RDP event count 对齐

Sample-RDP accountant 现在按 runtime 的实际 minibatch count，并乘 Flow 中真正重复执行 training block 的 `E_edge_loops`。通信 forwarding/HE 本身不额外生成 Sample-RDP event。

因此 privacy 路径与 learning 路径分工明确：

- Flow / actual mechanism executions -> unified Sample-RDP feasibility；
- 同一批真实 Sample-DP executions -> nonlinear private trajectory -> paired update-space moments -> joint learning proxy。

RDP epsilon 不能代数推出 `b_i,V_i`。

## 9. 验证要求

正式实验至少应抽样部分 rounds/candidates 做昂贵 paired-MC ground truth，并报告：

- Spearman rank correlation（主要）；
- Pearson correlation；
- top-k overlap / selected-candidate regret（可选）。

这证明 online calibration proxy 的排序确实代理理论 `J_learn`，而不是仅仅数值尺度相似。

## 10. 当前边界

本版本已经实现：calibration 统计器、table 格式、builder、在线 lookup、joint objective、增量候选 lookahead、round-conditioned lookup、Sample-only 正式 runner 约束和审计字段。

本版本**不会伪造 calibration 数据**。真正的 `paper_v34...joint.pt` 仍需通过独立 public/calibration workload 产生 paired clean/private trajectories 后构建。正式配置缺该文件会按设计报错。
