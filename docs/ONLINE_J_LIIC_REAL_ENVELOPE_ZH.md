# DynFL：直接分析 Stage13 的真实五轮 LIIC 上界组成（无须重训）

## 为什么还不能宣称完整 `J` 已经被认证

Stage13 已从实际五轮运行中获得 450 条 LIIC 选模前训练元数据，`formal_online_j_certified=false`。**仅凭优化器元数据，不可能推出无条件有限的 `J` 上界**：对单参数线性损失 `F_M(θ)=Mθ`，无论 `M` 多大，裁剪 SGD 的梯度范数均不超过 `C`，而理想更新 `-η* M` 的幅值可以无限增大。

**本阶段额外利用了真实 FashionMNIST/LeNet-5 的网络结构与输入取值范围，给出了独立的架构级梯度范数上界**，因此已经不需要外部手填理想梯度参考；这个界会很保守。若换成别的网络、数据归一化或损失函数，该界立即失效。

## 真实 SGD 可证明的局部更新上界

在目前真实的 `LIIC` 零动量、uncapped LeNet-5 SGD 中，

`θ_(k+1)=α θ_k - η(q_k+z_k)`，`α=1-ηλ`，`||q_k||<=C`。

`b_k=η α^(K-1-k)`，每 minibatch 高斯噪声协方差为 `(2Cσ/B_k)^2 I`，使用真实尾批 `B_k`。由递推式、Minkowski 不等式及独立高斯协方差求和得到

`||θ_K-θ_0||_(L2) <= (1-α^K)||θ_0|| + C Σ_k b_k + [d Σ_k b_k^2(2Cσ/B_k)^2]^(1/2) =: B_i`。

此公式不需要样本梯度无偏，不需要 Lipschitz 常数，允许 `q_k` 依赖之前的噪声，且**不会把本地 Sample-DP-SGD 噪声再次计入终端 Q**。它以正确的训练步数、各批大小、样本裁剪阈值和当前模型范数为输入。有限精度 SGD 实现额外误差、后续不受控变换另须计入；这个计算器不替它们提供认证。

如果真实 Cloud 更新满足 `Δ=γ Σ_i a_i u_i + ξ + e_num` 且 `a_i=n_i/Σ_j n_j`，则

`||γ Σ_i a_i u_i||_(L2) <= γ Σ_i a_i B_i =: B_cloud`。

从该网络的架构/归一化约束可以解析得到 `||∇F(w_t)||<=G_t`，但**只有在另外证明**实际聚合、后续独立噪声和数值误差对应模型时，才能得到保守的

`J_t <= [B_cloud + η* G_t + r_aggregate + r_num]^2`。

这里使用 Minkowski，不要求不同来源无条件正交。这个上界可能很松，而且它的数值大小与真实 `J` 的模式排序不必一致。缺少聚合路径/误差证据时，软件不会把 `B_cloud` 或 `G_t` 自动认定为完整 `J` 的认证结果。

### LeNet-5 结构性梯度范数上界：无需读取任何私有梯度

真实预处理是 `x=(pixel/255-0.286)/0.3205`，原始像素在 `[0,255]`，所以归一化后 `|x_j|<2.25`，`||x||_2<=28*2.25`。当前 GN-v2 61706 参数模型结构为 `conv5 -> ReLU -> avgpool2 -> conv5 -> ReLU -> avgpool2 -> fc400-120 -> ReLU -> fc120-84 -> ReLU -> fc84-10`，损失为单标签 softmax 交叉熵。

设完整模型参数范数的有效上界 `R>=||θ_t||_2`，则每层的权重 Frobenius 范数和偏置范数均不超过 `R`。卷积核 5×5 的线性算子范数不超过 `5||W||_F`（输入 patch 重叠最多25次），不重叠 2×2 平均池化的算子范数不超过 `1/2`，ReLU 导数绝对值不超过1，交叉熵 logits 梯度范数不超过 `sqrt(2)`。

定义前向界 `X=28*2.25`、

`H1=(5RX+28R)/2`，`H2=(5RH1+10R)/2`，`H3=RH2+R`，`H4=RH3+R`。

定义反向界 `D5=sqrt(2)`、`D4=R D5`、`D3=R D4`、`D2=(R D3)/2`、`D1=(5 R D2)/2`。

那么每个样本的 61706 维梯度均满足

`||∇ell(θ;x,y)||_2 <= G_arch(R) = hypot(5D1 X,28D1,5D2 H1,10D2,D3 H2,D3,D4 H3,D4,D5 H4,D5)`。

平均梯度使用 Jensen 不等式也满足该界，**无需从数据推断梯度分布**。该函数已在真实 LeNet-5 的 PyTorch CE 反向传播上通过测试。但浮点计算得到的 `R`/数值残差仍需额外数值误差审查才能称端到端机器级严格证书。

## 无须再跑五轮，直接读取现有 Shadow

Stage13 的 Shadow JSON 尚未单独存放 epoch 参数，因此必须显式传入你锁定的 `--local-epochs 3`。若元数据与该步数不一致，脚本失败，不靠重复列表猜测。

```powershell
& 'D:\soft\Python310\python.exe' examples/report_real_liic_envelope.py `
  --run 'out\stage13_real_liic_predecision\2026-10-17_03-03-07_lenet5_dynamic_newtex202608\full_dynfl' `
  --local-epochs 3
```

输出五行真实裁剪更新范数上界和 LeNet-5 架构级 `G_arch(R)`。脚本同时验证 `client_decisions.csv` 中 LIIC 已选 Client 与 Shadow 中的 LIIC 输入是否完全吻合；**事后的 Cloud admission 仍需单独验证**，因此输出称为 *selected-cohort proxy*，不谎称是正式认证的 Cloud `J`。不会查看原始训练样本、梯度、更新，不会影响预算或选模。

## 剩余确切缺口

1. 对 FashionMNIST/LeNet-5，已得到**架构级保守** `G_t`；今后需要更紧的逐层/方向性梯度证书才能有实用排序能力，也要独立确认完整数值误差界。
2. 真实 Cloud admission、Edge 多阶段反馈下的更新分解、Distributed DP 聚合噪声、数值误差必须逐路径证明。LIIC 局部 SGD 上界不能冒充可信 Split 和 LIIEIIIC 多阶段证书。
3. 才能在具有严格证书的候选集运行 `online_upper_bound` 七模式对照；未认证时仍然 fail closed。

该阶段已经打通真实 LIIC 局部更新和理想总体梯度的**两侧可计算上界**。但是，暂未证明的 Cloud 更新分解/后续噪声/数值误差仍阻止声称完成完整七模式 `J` 认证。
