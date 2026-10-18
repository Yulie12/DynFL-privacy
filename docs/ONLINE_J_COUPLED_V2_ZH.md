# DynFL：仅修改在线学习目标 J 的结构化证书（v2）

## 0. 本补丁的边界

- **只修改 `dynfed/online_j_bound.py`、`dynfed/selection.py` 的在线学习误差评分和诊断字段**，增加对应测试与独立数学示例。
- 不修改训练器、资源/信任准入、DP 采样/噪声、RDP 账本、预算上限、SecAgg、HE、网络时延或现有对照配置。
- 当前正式路径只接受 `learning_objective=online_upper_bound`；`legacy_fusion_dp`、`joint_calibration` 与 oracle 诊断均已移除，配置会被直接拒绝。
- **不会自动生成真实误差证书**。数据不足会报错，不允许回退到旧 Fusion 或离线表。JSON 中的 `assumptions: true` 是外部证明的声明，程序无法仅凭布尔值证明这些假设真实成立。
- 当前所有“v2”测试只构造可核验的公开合成情形，**不证明真实七模式/DP 机制具备相应证书**。

## 1. 为什么重构

v1 采用单个历史方向和每个 Mode 的大参考球，尽管可上界，却常对实际与理想更新重复付出不确定半径。v2 将二者共同包含的**同一个 Client 理想梯度贡献**在求差时先抵消，再界定误差。

固定本轮目标（不以真实模式平均更新替代）：

\[
F(w_t)=\sum_{i\in\mathcal P}p_iF_i(w_t),\quad p_i=\frac{n_i}{\sum_jn_j},\quad
v_i^t=-\eta_t^*\nabla F_i(w_t),\quad \Delta_t^*=\sum_i p_i v_i^t.
\]

对候选模式配置 \(h\)，假设经**对真实执行链路另行证明**：其 Cloud 预噪更新确实具有同维 per-client 分解

\[
\Delta_{\mathrm{clean}}^t(h)=\gamma\sum_i a_i(h)u_i^t(h),
\qquad a_i=\begin{cases}n_i/\sum_{j\in C(h)}n_j&i\in C(h)\\0& i\notin C(h)\end{cases}.
\]

\(\hat v_i\) 必须由第 t 轮**决策前**的数据产生，且已得到有效证书：

\[
\|v_i-\hat v_i\|_{L^2_t}\le s_i,\qquad
\|u_i(h)-v_i\|_{L^2_t}\le r_{i,h}.
\]

精确分解（先做代数抵消）：

\[
\Delta_{\mathrm{clean}}-\Delta^*
=\sum_i(\gamma a_i-p_i)\hat v_i
+\sum_i(\gamma a_i-p_i)(v_i-\hat v_i)
+\gamma\sum_i a_i(u_i-v_i).
\]

因此定义（不要求训练组之间独立）：

\[
A=\left\|\sum_i(\gamma a_i-p_i)\hat v_i\right\|,\quad
R_{\rm ref}=\sum_i|\gamma a_i-p_i|s_i,\quad
R_{\rm mode}=|\gamma|\sum_i|a_i|r_{i,h},\quad R=R_{\rm ref}+R_{\rm mode}.
\]

若参与者在 SecAgg 聚合**之前**产生 DP 份额 \(z_i\)，Cloud 聚合可**等价表示** \(\xi_{\rm agg}=\gamma\sum_i a_i z_i\)。只有当份额相对于全部预噪状态条件零均值，参与权重预噪固定，且不同份额条件不相关，已证明第 i 个**未加权**份额是坐标独立的 \(N(0,\sigma_i^2)\)，才能计算

\[
Q=d\gamma^2\sum_i a_i^2\sigma_i^2.
\]

（**实际机制不是 Cloud 聚合后才加噪。**）本地 DP-SGD 对优化轨迹的影响已包含在模式残差证书，不得重复进入 \(Q\)。数值误差 RMS 为 \(r_{\rm num}\) 时：

\[
\boxed{U_J(h)=\left[\sqrt{(A+R)^2+Q}+r_{\rm num}\right]^2.}
\]

反向 \(L^2\) 三角不等式还给出一个保守下界。默认 \(Q\) 只是噪声能量上界，不能用于下界；只有另行证明聚合噪声方差**精确**且与干净更新正交时，才取 \(Q_{\rm floor}=Q\)，否则 \(Q_{\rm floor}=0\)：

\[
\boxed{L_J(h)=\Big[\max\big(0,\sqrt{\max(0,A-R)^2+Q_{\rm floor}}-r_{\rm num}\big)\Big]^2.}
\]

只有当 \(U_J(h_1)<L_J(h_2)\) 或 \(U_J(h_2)<L_J(h_1)\) 时，才可以**严格认证**真实 \(J\) 的排序；否则排序为 undetermined。该区间比较仅作诊断，正式选模仍以 \(U_J\) 为目标，不能把“按上界排序”解释为“真实 \(J\) 的认证排序”。

## 2. 可选“偏差—方差”更紧公式

将 \(u_i-v_i=b_i+\varepsilon_i\)，其中 \(b_i=\mathbb E[u_i-v_i\mid\mathcal G]\)，\(\varepsilon_i\) 在包含全部理想梯度的 \(\mathcal G\) 下条件均值零。若可另行证明

\[
\|b_i\|\le\beta_i,\quad\mathbb E\|\varepsilon_i\|^2\le v_i,\quad
\mathbb E\langle\varepsilon_i,\varepsilon_j\rangle=0\; (i\ne j),
\]

则启用 `mode_uncertainty=bias_variance`：

\[
\boxed{U_J^{\rm BV}=\left[\sqrt{(A+R_{\rm ref}+|\gamma|\sum_i|a_i|\beta_i)^2
+\gamma^2\sum_i a_i^2 v_i+Q}+r_{\rm num}\right]^2.}
\]

注意 **不同组训练过程条件独立 ≠ 不同组历史预测残差零均值**。必须对中心化创新单独认证；否则使用 `rms`，不得启用 BV。

## 3. v2 manifest 最小字段

证书必须与本轮 `online_bound_round` 一致、`available_before_round` 必须是本轮索引，且有明确的 `evidence` 来源。`target` 固定为 `ideal_full_gradient_one_step`，`aggregation_rule` 为 `cloud_participant_sample_weighted`。

```json
{
  "schema": "dynfl.online_j.v2",
  "target": "ideal_full_gradient_one_step",
  "rounds": {
    "round:0": {
      "round_index": 0,
      "available_before_round": 0,
      "evidence": "具体数学证明或审核记录，不可用 synthetic 冒充真实证书",
      "aggregation_rule": "cloud_participant_sample_weighted",
      "cloud_step": 1.0,
      "numeric_rms": 0.0,
      "mode_uncertainty": "rms",
      "reference_direction": [0.1, -0.2],
      "assumptions": {
        "global_objective_is_population_sample_weighted": true,
        "linear_client_contributions_match_runtime": true,
        "ideal_reference_rms_bounds_proved": true,
        "mode_relative_ideal_bounds_proved": true,
        "noise_shares_drawn_before_secagg_and_weight_fixed": true,
        "noise_conditionally_zero_mean_and_uncorrelated": true,
        "aggregate_share_second_moment_bounds_proved": true,
        "numeric_rms_bound_proved": true
      },
      "population": {
        "0": {"samples": 6, "ideal_reference_scale": 1.0, "ideal_rms": 0.01},
        "1": {"samples": 4, "ideal_reference_scale": 1.0, "ideal_rms": 0.02}
      },
      "clients": {
        "0": {"candidates": {"<exact candidate key>": {
          "relative_ideal_rms": 0.03,
          "aggregate_share_std": 0.0,
          "runtime_mode_equivalence_verified": true,
          "local_dp_included_in_mode_rms": true
        }}},
        "1": {"candidates": {"<exact candidate key>": {
          "relative_ideal_rms": 0.02,
          "aggregate_share_std": 0.0,
          "runtime_mode_equivalence_verified": true,
          "local_dp_included_in_mode_rms": true
        }}}
      },
      "no_cloud_update_is_zero_verified": true
    }
  }
}
```

此 JSON **仅是字段示例**，所有半径及机制条件必须由可核验的证明/部署记录支撑；不得复制数字直接运行正式训练。`<exact candidate key>` 需使用 `certificate_candidate_key(candidate)` 生成，包含 Mode、链路机制和具体噪声计划。

每个客户端也可改用 `"ideal_reference": [..d个数字..]`，与压缩 `"ideal_reference_scale"` 二选一。压缩形式中 `ideal_reference_scale * reference_direction` 是确定的参考中心，`ideal_rms` 必须覆盖客户端异质性与历史不确定性。共用一个大方向时，参考向量占用 \(O(d+N)\) 存储，而不是 \(O(Nd)\)。

`bias_variance` 模式必须把各候选的 `relative_ideal_rms` 改为 `relative_ideal_bias` 和 `relative_ideal_variance`，并额外把 `mode_fluctuations_centered_given_ideal` 和 `mode_fluctuations_cross_client_orthogonal_given_ideal` 显式证为 `true`。

## 4. 为什么现在不能直接用 v2 跑正式5轮

**关键缺口不是选模调用，而是每个合法候选的 `relative_ideal_rms` 或 `bias/variance` 的事前真实证书。** 当前公式推导的 `relative_ideal_rms`（`dynfl.liic.sgd.v1`）只支持 LeNet-5 / LIIC / 零动量 `torch.optim.SGD` 的 `sample_dp_sgd` 或 `clipped_sgd` 路径。ResNet-18 正式配置使用 SGD `momentum=0.9, weight_decay=5e-4`，不能复用该证书；必须另行证明与实际优化器、步数、batch 和噪声机制一致的事前半径。可信 Client–Edge Split 不应添加内部 DP，而且必须证明其执行轨迹对应上述 Client 贡献；多阶段 Edge 需要递推传播误差；跨 Cloud `LIC` 还需独立信任边界审计。已存在的示例和测试均不能自动完成这些证明。

目前 v2 在候选预筛选和最终 Pareto 中均使用同一个 `U_J`；新增 `L_J` 只作为诊断，不会篡改目标，也不能保证真实排序。

## 5. 本地验证（不触碰隐私预算）

```powershell
cd E:\YTT\GROUP\DynFL-privacy

& "D:\soft\Python310\python.exe" -m pytest -q `
  tests/test_online_upper_bound_integration.py `
  tests/test_jlearn_v2_aggregate_dp.py `
  tests/test_online_j_blockwise.py `
  tests/test_online_j_exact_noise_and_ranking.py `
  tests/test_online_j_liic_envelope.py `
  tests/test_online_j_runtime_integration.py `
  tests/test_liic_online_certificate.py
```

这些测试使用公开合成证书验证公式与失败关闭行为，**不会**生成可用于正式训练的真实证书。
