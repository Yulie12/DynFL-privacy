# Stage 18：新版 J 的直接训练对接（中央仿真器诊断）

## 使用方式与限制

`online_j_oracle_diagnostic` 是**可运行的、显式不认证的仿真诊断**。它不再使用 `legacy_fusion_dp`，也不需要不存在的 `REAL_...json`。每轮在选择模式、训练 Client 之前，对模拟器中所有 Client 的训练样本执行当前模型的全数据梯度计算，获得理想一步更新的参考范数；从真实候选的 SGD 步数、clip、momentum、weight_decay、sample optimizer sigma 推导 RMS 上界；再将这个预决策结果交给全局 Pareto。

**不适用于不可信联邦服务端：** 直接使用全体 Client 原始训练数据构造参考梯度属于中央 oracle，违背隐私隔离；它也没有为 PyTorch 有限精度、真实 CKKS、混合 Split、独立 DP/HE 组合构建全部证明。所有 JSON/CSV 中都明确注明 `formal_online_j_certified=false`、`learning_proxy_source=SIMULATOR_ORACLE_NEW_J_NOT_DEPLOYMENT_CERTIFIED`。请勿把它称为严格已证实的正式 `U_t`，也不要用它替代论文中的隐私保证。

严格的 `online_upper_bound` **没有退化**，仍要求完整 v2 预决策证明材料。

## 实际计算的学习目标

对于当前模拟器的经验目标 `F = Σ_i p_i F_i`，在决策前用**中央 oracle** 计算各 Client 的 `v_i=-η_*∇F_i(w_t)` 和 `v=Σ_i p_i v_i`。然后采用同世界分解（不需要把训练偏差假装成零）：

\[
 A=|\gamma-1|\|v\|,\quad
 R_{\rm ref}=\sum_i|\gamma a_i-p_i|s_i,\quad s_i=\|v_i\|+\|v\|,
\]
\[
 R_{\rm mode}=\gamma\sum_{i\in S_t}a_i(B_{i,h}+\|v_i\|),\quad
 U_{\rm coupled}=(A+R_{\rm ref}+R_{\rm mode})^2.
\]

同时使用无需参考方向的直接 Minkowski 上界：

\[
 U_{\rm direct}=\left(\|v\|+\gamma\sum_{i\in S_t}a_i B_{i,h}\right)^2,\quad
 U_t^{\rm oracle}=\min(U_{\rm coupled},U_{\rm direct}).
\]

其中 `a_i=n_i/Σ_{j∈S_t}n_j`，不参与 Cloud 的 `a_i=0`。这两种界在**被支持的真实线性聚合/SGD 路径**下都保守合法；只是可能很宽松。`Q=0` 是因为这个诊断**仅支持没有额外 Gaussian Cloud release 的 HE3 路径**，而非任意保护机制。公式之外的实际 FP/CKKS 数值差异未经形式化认证。

`B_{i,h}` 为本地真实 SGD 路径的二阶范数上界：令 `Θ_0=||θ_0||`、`M_0=0`、`D_0=0`，每一步递推

\[
 Z_k=\sqrt d\,2C\sigma/B_k\quad(\text{Sample-DP 路径，否则 }0),
\]
\[
 M_{k+1}=\mu M_k+C+Z_k+\lambda\Theta_k,\quad
 D_{k+1}=D_k+\eta M_{k+1},\quad
 \Theta_{k+1}=\Theta_k+\eta M_{k+1}.
\]

最终 `B_{i,h}=D_K`。这个界不假设 stochastic gradient 无偏；本地 DP-SGD 只计入 `Z_k`，不在 `Q` 中重复计数。在 **当前 HE-only、无独立 Cloud 高斯份额** 的范围内 `Q=0`。真实计算中的有限精度等残差**未经严格认证**，因此此功能只能诊断，不等同于正式证明。

支持仅 `LIIC`/`LIIEIIIC` 的完整模型 SGD + HE3、无 cap 本地 minibatches。其它 Cloud 模式或者同时启用 Packet DP 的候选不能悄悄用这个公式代替，程序会拒绝。

**重要：** 当前两种模式本地更新几乎相同，且均可送到 Cloud，因而真实推导的 `B_i` 可能相同；即使换新 J，LIIC 仍可能因时延更低而支配 LIIEIIIC。这不是程序错误，也不能为了迫使切换而虚构偏差。

## 实际启动

```powershell
cd E:\YTT\GROUP\DynFL-privacy
& D:\soft\Python310\python.exe -m pytest -q tests/test_online_j_oracle_diagnostic.py
& D:\soft\Python310\python.exe experiments/run_paper_config.py `
  --config configs/stage18_fmnist_lenet5_oracle_new_J_5r.json `
  --policies full_dynfl `
  --max-new-rounds 3
```

运行成功后 `out/stage18_simulator_new_J/fmnist_lenet5` 下会有独立批次目录，每轮会自动产生 `online_j_ORACLE_NONCERTIFIED_round_000.json`，在 `pareto_profile_audit.csv` 中检查 `learning_proxy_source`、`learning_error_objective` 和 `online_j_radius`。输出运行后再执行 `analyze_dynfl_mode_switches.ps1`。

同样支持 **诊断性质的** `configs/stage18_cifar10_resnet18_oracle_new_J_5r.json`，但 ResNet-18 具有 BatchNorm 等状态依赖和 CKKS 真实数值误差，因此计算出的是按模拟器梯度定义的诊断量，不构成严格经验目标的数学证明。原始 Raw 数据不可离开可信的实验仿真器。

## 当前仍缺少什么才能正式运行新的 U_t

要切换回严格 `online_upper_bound`，需要真实的隐私安全参考梯度范围或充分证明的代用中心、七模式计算图对应的条件 RMS 证书、运行时 Edge 聚合权重与证书一致性、DP release 噪声二阶矩以及 HE/FP 数值误差上界。数据驱动的模拟器 oracle **不能当作私有数据不泄露的证明**。
