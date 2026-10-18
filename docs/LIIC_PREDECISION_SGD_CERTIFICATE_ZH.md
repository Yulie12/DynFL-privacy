# DynFL 第十二阶段：真实 LeNet-5 LIIC 优化器结构的在线误差半径

## 0. 实际完成的范围

此补丁**只添加** `dynfed/liic_online_certificate.py`、新版在线 `J` v2 的 LIIC 半径计算入口、测试、公开示例。不更改隐私预算、RDP、DP 机制、训练器、可信 Split、SecAgg、其他六种 Mode 或准入规则。它不是能够直接完成正式七模式五轮实验的完整证书生成器。

- 核对的真实入口：`dynfed/fmnist_lenet5_dynamic.py::_client_train_worker()` 调用 `dynfed/split_learning.py::split_local_train_lenet5()`。
- 只对 `mode=LIIC` 的全模型、**新建零动量 SGD**（真实 LeNet-5 `_make_optimizer(momentum=0)`）建模，允许非零 `weight_decay`。ResNet-18 使用 Momentum 0.9，本公式不适用。
- `privacy_unit=sample` 时，`sample_dp.py::clip_and_aggregate_per_sample_grads()` 逐样本裁剪范数到 `C`，对**梯度求和**加入 `N(0,(2Cσ)^2 I)`，再除以**真实 minibatch 样本数**；不是在整个训练结束后一次加入。这部分随机性属于模式误差半径，不是 Cloud/SecAgg 汇总份额的 `Q`。
- 非 Sample-DP 的完整本地路径执行 `clip_grad_norm_(max_norm=5)`，然后 `optimizer.step()`；不调用其内部 per-minibatch Gaussian。
- 如果 Cloud 发布之前有额外**整更新范数裁剪**，需要给出 `release_update_clip_norm`；若有尚未建模的其他变换则直接拒绝认证。

## 1. 严格代数

给定第 `t` 轮选模前的当前模型参数 `θ₀`，执行 `K` 个小批量 SGD 步，学习率 `η`、权重衰减 `λ`。把经过裁剪的训练梯度记为 `q_k`，满足 `||q_k|| ≤ C`（对于 Sample-DP 成立，因为各逐样本梯度先裁剪后平均）。把高斯梯度噪声记为 `z_k`。

\[
\theta_{k+1}=(1-\eta\lambda)\theta_k-\eta(q_k+z_k),\qquad
\alpha=1-\eta\lambda \in[0,1],\quad
b_k=\eta\alpha^{K-1-k},\quad S=\sum_{k=0}^{K-1}b_k.
\]

于是实际待上传的**训练前剪裁、未发生 Cloud 汇总噪声的**完整更新精确满足：

\[
u=\theta_K-\theta_0=-\lambda S\theta_0-\sum_kb_kq_k-\sum_kb_kz_k.
\]

用第 `t` 轮决策前已经取得的某个 Client 理想一步更新参考 `\hat v_i=-\eta_* g_{i,\mathrm{ref}}`，并假定**另有独立有效的证明**：

\[\|v_i-\hat v_i\|_{L_t^2}\le s_i,\qquad v_i=-\eta_*\nabla F_i(w_t).\]

将上面的 `u` 加减 `-S g_ref` 可得：

\[
\boxed{
\|u-v_i\|_{L_t^2}
\le \underbrace{|S/\eta_*-1|\,\|\hat v_i\|+\lambda S\|\theta_0\|}_{\text{中心和 weight decay}}
+\underbrace{\sum_k b_k E_k}_{\text{裁剪梯度偏差}}
+\underbrace{\sqrt{d\sum_k b_k^2(2C\sigma/B_k)^2}}_{\text{本地 DP-SGD 的高斯 RMS}}
+s_i .}
\]

其中 `B_k` 是第 `k` 个真实 minibatch 的样本数。若无法事前严格控制 `||q_k-g_ref||`，**始终可安全取**：

\[E_k=C+\|g_{i,\mathrm{ref}}\|=C+\|\hat v_i\|/\eta_* .\]

如果通过有效的逐步、路径一致证明给出更小 `E_k`，需要设置 `gradient_reference_residual_caps` 和 `gradient_residual_caps_proved=true`。过去几轮的梯度均值/梯度差、有限次蒙特卡罗以及 Secant 估计**不能自行充当有证明的路径一致上界**。

高斯分量之间的加权平方和来自**独立、条件零均值的各 minibatch 噪声**；但 `q_k` 可以依赖早期噪声，**不能**把整个 SGD 漂移项和高斯项误当作正交。因此上面用的是 Minkowski 的 RMS 相加。

如果训练之后还执行范数投影 `P_{C_u}(u)`（整更新裁剪），利用投影的 1-Lipschitz 性：

\[
\|P_{C_u}(u)-v_i\|_{L_t^2}
\le \|u-\hat v_i\|_{L_t^2}
+\max(0,\|\hat v_i\|-C_u)+s_i.
\]

它不是把 DP 噪声当成 Cloud 聚合后再加一次，而是把整个发布前确定性裁剪直接算进模式误差。

## 2. 接入现有 v2 manifest 的方式

在 `clients[client_id].candidates[certificate_candidate_key(candidate)]` 的**真实 LIIC** 条目中，使用 `liic_sgd_proof` 代替直接填写 `relative_ideal_rms`。例如（只是字段形式示意，不是正式证书）：

```json
{
  "runtime_mode_equivalence_verified": true,
  "local_dp_included_in_mode_rms": true,
  "aggregate_share_std": 0,
  "liic_sgd_proof": {
    "schema": "dynfl.liic.sgd.v1",
    "mode": "LIIC", "model": "lenet5", "optimizer": "torch.optim.SGD",
    "momentum": 0, "source": "sample_dp_sgd",
    "evidence": "需要真实独立证明/审计记录",
    "training_state_matches_runtime": true,
    "ideal_reference_bound_proved": true,
    "noise_distribution_matches_runtime": true,
    "extra_transformations_absent_or_modeled": true,
    "initial_parameter_norm": 7.0,
    "learning_rate": 0.15, "ideal_step": 0.15,
    "weight_decay": 0.0001, "gradient_clip_norm": 3.0,
    "sample_noise_multiplier": 0.1,
    "planned_optimizer_steps": 3, "batch_sizes": [120,120,120]
  }
}
```

`batch_sizes` 可用 `planned_batch_sizes(sample_count=n, epochs=3, batch_size=256)` 在第 t 轮训练前得到；有 `local_steps` 上限而又涉及提前抽样时，当前计算器会拒绝，防止把实际 batch 规模估错。当前样例中的 `initial_parameter_norm`、`ideal_reference`、`ideal_rms`、`sigma` **均不得凭空复制到真实实验**。它们分别需要来自真正的模型状态、合规的参考/梯度证书、真实机制参数。

*v2 理想参考半径 `ideal_rms` 必须单独有证据。仅凭模型当前参数、裁剪阈值或上一轮训练结果，无法得到足够紧且普适的 `||∇F_i(w_t)-g_ref||` 上界。* 若没有这一项，代码仍拒绝真正的模式证书——不能用 `assumptions=true` 伪装它已经证明。

本地 DP-SGD 扰动已在 `relative_ideal_rms` 中计算；`aggregate_share_std` **只用于确实在聚合前生成的分布式更新发布噪声份额**。例如实测 `he3` 且没有额外聚合 DP 时，该值应为零；若真实机制同时启动聚合 DP，必须独立核对其权重与份额噪声标定。

## 3. 如何验证

解压 patch 到 `E:\YTT\GROUP` 后：

```powershell
cd E:\YTT\GROUP\DynFL-privacy
git apply --check --ignore-space-change "E:\YTT\GROUP\DynFL_LIIC_SGD_certificate_stage12.patch"
git apply --ignore-space-change "E:\YTT\GROUP\DynFL_LIIC_SGD_certificate_stage12.patch"
& "D:\soft\Python310\python.exe" -m pytest -q tests/test_liic_online_certificate.py tests/test_online_upper_bound_integration.py
& "D:\soft\Python310\python.exe" examples/liic_predecision_radius_demo.py
```

测试不仅验证代数，还会调用**实际** `split_local_train_lenet5(mode='LIIC')` 完成公开小数据训练，并用独立理想梯度计算事后真实更新误差。实际梯度只在测试中使用，**真实在线选模中不可偷看当轮私有梯度**。

## 4. 下一步必须补齐的证据

1. 从第 `t` 轮真实模型状态**预决策**提取 `initial_parameter_norm`、真实参数维度、客户端样本量和训练步数，进行状态一致性检查。
2. 为 `ideal_reference` 及其 `ideal_rms` 建立合法、可验证的历史参考与异质性证书。没有可靠半径之前，不能自动标为严格在线 J。
3. 与 `LIIC` 真实聚合路径逐一对应：任何更新级裁剪、来源归一化及分布式 SecAgg 噪声份额都要在候选证书中反映。
4. 其余六种 Mode 需要独立推导，不能把 LIIC 的本地 SGD 证书复制过去。对于可信 Client–Edge Split，内部不添加 DP；Edge-only 仅有快响应且合规者才允许选择。
5. 只有 `J<=U` 的真实性、上界紧致度和两两排序诊断经过审查后，才适合与旧 Fusion 做 5 轮正式对照。
