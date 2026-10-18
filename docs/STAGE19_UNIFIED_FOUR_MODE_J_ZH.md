# Stage 19：统一 Cloud 有效系数的新 J（中央仿真诊断版）

**安装依赖：** 已应用 Stage 16、`risk_constant_removal_stage16.patch`、Stage 18 新 J 补丁。

## 数学目标

真实目标依然是 `J_learn^t(h)=E_t ||Delta_t(h)-Delta_t*||^2`，选模使用其预决策保守上界。令 `v_i=-eta_* ∇F_i(w_t)`，`p_i=n_i/Σ n_j`，`gamma=server_step`，Cloud 有效系数：

- `LIIC`：`c_i=gamma*n_i/Σ_{j进入Cloud}n_j`。
- `LIIEIIIC`：Edge 根据当前参与 Client 样本数加权，Cloud 根据该 Edge 实际代表的参与样本数加权，两层系数相消，因此仍为 `c_i=gamma*n_i/Σ_{j进入Cloud}n_j`。
- `LIEIIC`：可信 Client–Edge Split 完成后向 Cloud 提交更新，同样用上述有效系数；不凭空添加可信链路的隐私噪声。
- `LIIE`（以及现有 `LIE` Edge-only）：当前轮不向 Cloud 提交更新，`c_i=0`；理想更新缺失项 `-p_i*v_i` 保留。

每轮选模前，在**集中仿真器**上计算 `v_i` 和 Gram 矩阵 `G_ij=<v_i,v_j>`（不把整段梯度写入 JSON）。对于任意候选 Profile：

```
w_i = c_i - p_i
A_t = sqrt(max(0, Σ_i Σ_j w_i G_ij w_j))
R_t = Σ_{i进入Cloud} c_i * (B_{i,h_i} + ||v_i||)
J_select = U_t^sim = (A_t + R_t)^2
```

`B_i` 是 SGD 更新的 RMS 范数保守包络。`LIIC` 单次本地训练；`LIIEIIIC` 按真实 `E_edge_loops=3` 次完整 SGD 循环递推上界；`LIEIIC` 只在 **client-unit、可信内部特征/梯度链路、无额外 Packet-DP** 的 Split 路径下复用每步两个优化器各裁剪5的联合粗界 `C=10`。现有 Sample-DP Split 仍要求独立机制，**不会假冒同一条可信 SGD 路径**。涉及其他机制时失败即拒绝。

本诊断仅支持 HE3 Cloud 输出且**没有独立上传高斯噪声**的路径，故额外 `Q=0`；真实本地 Sample-DP-SGD 噪声包含在 `B_i` 中，不能重复计入 `Q`。CKKS 和有限精度残差**未经形式化证明**，所以仍显式标注 `formal_online_j_certified=false`，不允许宣称这是部署级严格认证上界。正式 `online_upper_bound` 的证书检查没有放宽。

**注意：** 该 `U` 可非常宽松，LIIC 仍可能支配 LIIEIIIC；不能为制造 Mode 切换而任意修改误差项。本版选用包含真实缺失贡献的 `A+R` 上界而不是旧版 `min(direct,coupled)` 诊断，保证缺失的理想 Cloud 贡献明确进入选模目标。它是一个合法的保守界结构，但无法凭此证明每次选择都是精确 `J_learn` 最优。

## Windows 运行

```powershell
cd E:\YTT\GROUP\DynFL-privacy
& D:\soft\Python310\python.exe -m pytest -q tests/test_online_j_oracle_diagnostic.py
& D:\soft\Python310\python.exe experiments/run_paper_config.py `
  --config configs/stage19_fmnist_lenet5_four_mode_new_J_5r.json `
  --policies full_dynfl `
  --max-new-rounds 3
```

ResNet18 对应 `configs/stage19_cifar10_resnet18_four_mode_new_J_5r.json`。建议先对 LeNet5 进行三轮测试，再启动 ResNet18。`out/stage19_four_mode_new_J` 下的每轮 `online_j_ORACLE_NONCERTIFIED_round_*.json` 将包含 `ideal_client_update_gram`、`objective_formula`、`edge_training_stages`。`pareto_profile_audit.csv` 的 `learning_error_objective` 就是本版 `U_t^sim`。注意不能跨阶段续跑旧检查点。

### 剩余正式上线条件

在不可信服务端不能直接读取 Client 原始数据来计算 Gram；可信 Split Sample-DP、多模式额外上传 DP/噪声的协方差、真实 CKKS 数值误差、候选实际准入/中途丢包权重与 Profile 一致性，都需要独立的决策前证书。这些未被本补丁宣称完成。
