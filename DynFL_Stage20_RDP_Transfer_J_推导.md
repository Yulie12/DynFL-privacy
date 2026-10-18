# DynFL Stage 20：RDP 可知噪声下的分层 SGD 传递上界推导

> **定位**：这是给 Stage 19 `online_j_oracle_diagnostic` 寻找更紧上界的**条件数学推导**及 Shadow Audit 设计，不是已完成的隐私认证，也未自动替换 Pareto 目标。适用路径：ResNet18 full-model 的 LIIC、LIIEIIIC；LIIE/SKIP 的 Cloud 系数为零。LIEIIC 的 Split 路径另证。

## 0. 要证明什么，以及时间条件

在第 \(t\) 轮决策时刻，记决策前信息为 \(\mathcal H_t\)，当前可训练参数向量为 \(\theta_t\in\mathbb R^d\)。为一套拟议的选模方案 \(h\) 定义 Cloud 更新 \(\Delta_t(h)\)，理想全体梯度更新为

\[
\Delta_t^*=-\eta_t^*\sum_{i=1}^N p_i\nabla F_i(\theta_t),\qquad p_i=n_i/\sum_j n_j.
\]

目标是构造**在当前随机训练运行之前**可算出的 \(U_t(h)\)，满足

\[
\mathbb E\big[\|\Delta_t(h)-\Delta_t^*\|_2^2\mid\mathcal H_t\big]\le U_t(h).
\tag{0}
\]

\(\Delta_t^*\) 现在来自模拟器读取客户原始数据的中心 Oracle，因此证明 (0) **还不能推出真实不可信 FL 服务端可计算**。

## 1. RDP 会计如何给出实际加噪程度

`dynfed/sample_dp.py::clip_and_aggregate_per_sample_grads` 对每个样本梯度进行**跨所有可训练参数的全局 L2 裁剪**：

\[
\bar g_{i,s,k}^{(j)}=g_{i,s,k}^{(j)}\min\{1,C_{i,s,k}/\|g_{i,s,k}^{(j)}\|\},\quad
G_{i,s,k}=\frac1{B_{i,s,k}}\sum_{j=1}^{B_{i,s,k}}\bar g_{i,s,k}^{(j)}.
\]

于是 \(\|G_{i,s,k}\|\le C_{i,s,k}\)。代码对**裁剪后的梯度总和**加噪，再除以实际批量大小 \(B_{i,s,k}\)。如果使用替换邻接、总和敏感度 \(2C\)，以其为单位定义的 noise multiplier 为 \(\sigma_{i,s,k}\)，则每个可训练参数坐标进入 SGD 的噪声为

\[
\xi_{i,s,k}\sim\mathcal N(0,\tau_{i,s,k}^2 I_d),\qquad
\boxed{\tau_{i,s,k}=\frac{2 C_{i,s,k}\sigma_{i,s,k}}{B_{i,s,k}}.}\tag{1}
\]

最后一个不足批量的 minibatch **必须使用它自己的实际** \(B_{i,s,k}\)，不能统一替换成配置的最大 batch size。

**RDP 不等于直接读取 \(\varepsilon\) 就能反推出 \(\sigma\)。** 对未经 subsampling 的单次高斯机制，以 \(2C\) 为敏感度有

\[
\rho_{i,s,k}(\alpha)=\frac{\alpha}{2\sigma_{i,s,k}^2},\quad \alpha>1.
\]

若有 subsampling，必须使用实际采样模型与 sampling rate \(q_{i,s,k}\) 对应的 \(\rho_\alpha(q,\sigma)\)；不能将 Poisson 和无放回采样的放大界混用。组合之后，常见的 RDP 转 \((\varepsilon,\delta)\) 关系为

\[
\varepsilon_t(\delta)\le \inf_{\alpha>1}\left(\rho_{\le t}(\alpha)+\frac{\log(1/\delta)}{\alpha-1}\right).
\tag{2}
\]

不同步数和不同 \(\sigma\) 可对应相同的总 \(\varepsilon\)；所以 Shadow Audit 的正确数据源是**每个候选选出的 \(\sigma\)**，而不是一个总 \(\varepsilon\)。DynFL 的决策记录包含 `selected_sample_optimizer_noise_multiplier`；`SampleStageEventAudit` 还会核对 Worker 报告的 `executed_sample_optimizer_sigma` 与裁剪阈值。RDP 负责预算可行性、累积组合和一致性检查，式 (1) 负责本轮随机更新的二阶矩。

例如 \(C=3\), \(\sigma=4.6\)：批量 \(B=60\) 时 \(\tau=0.46\)，批量 \(B=256\) 时 \(\tau\approx0.1078125\)。这是说明公式的数值示例，不是从你当前实验中读出的实际 \(\sigma\)。

## 2. 单个 SGD 阶段的精确展开（动量 + weight decay）

设学习率 \(\eta\)、momentum \(\mu\)、weight decay \(\lambda\)，每一阶段重新初始化 \(m_0=0\)。PyTorch SGD（无 Nesterov / 无 dampening）满足

\[
\begin{aligned}
m_k&=\mu m_{k-1}+G_k+\xi_k+\lambda\theta_{k-1},\\
\theta_k&=\theta_{k-1}-\eta m_k.
\end{aligned}\tag{3}
\]

定义二维块状态 \(x_k=(\theta_k,m_k)^\top\)、矩阵及注入向量

\[
M=\begin{pmatrix}1-\eta\lambda&-\eta\mu\\\lambda&\mu\end{pmatrix},
\quad b=\begin{pmatrix}-\eta\\1\end{pmatrix},\quad e_1=\begin{pmatrix}1\\0\end{pmatrix}.
\]

因为 \(M\) 只含标量系数（作用于每个参数坐标相同），有**精确代数恒等式**

\[
x_K=M^Kx_0+\sum_{k=1}^KM^{K-k}b(G_k+\xi_k),
\]

\[
\boxed{\theta_K=f_K\theta_0+\sum_{k=1}^K\beta_{K,k}(G_k+\xi_k)},
\quad f_K=e_1^\top M^Ke_1,\ \beta_{K,k}=e_1^\top M^{K-k}b.
\tag{4}
\]

这是对随机**非线性神经网络训练轨迹**仍然成立的展开，因为 \(G_k\) 只作为输入；无须假设梯度独立、无偏或 Hessian 为常数。当 \(\lambda=0\) 时，\(f_K=1\)，且

\[
\beta_{K,k}=-\eta\frac{1-\mu^{K-k+1}}{1-\mu}.
\]

由此即使 \(G_k\) 受到之前噪声影响，也可利用 \(\|G_k\|\le C_k\) 控制梯度部分。

## 3. 三阶段 Edge 聚合与重置动量

对一个选中的 LIIEIIIC Edge \(e\)，设参加所有三个阶段的固定 Client 集合为 \(\mathcal C_e\)，权重为

\[
\alpha_{i|e}=\frac{n_i}{N_e},\quad N_e=\sum_{j\in\mathcal C_e}n_j.
\]

记阶段索引 \(s=1,2,3\)，令 \(f_{i,s}\)、\(\beta_{i,s,k}\) 由式 (4) 按阶段的学习率、动量、衰减与步数计算。假设每阶段对所有参与 Client 共享同一 Edge 回传模型 \(\Theta_{e,s-1}\) 且其优化器动量重置，则

\[
\begin{aligned}
\Theta_{e,s}&=\sum_{i\in\mathcal C_e}\alpha_{i|e}\theta_{i,s,K_{i,s}}\\
&=F_{e,s}\Theta_{e,s-1}+\sum_{i\in\mathcal C_e}\alpha_{i|e}\sum_k\beta_{i,s,k}(G_{i,s,k}+\xi_{i,s,k}),\\
F_{e,s}&=\sum_{i\in\mathcal C_e}\alpha_{i|e}f_{i,s}.
\end{aligned}\tag{5}
\]

反复代入，有

\[
\boxed{\Theta_{e,3}=\left(\prod_{s=1}^3F_{e,s}\right)\theta_t+\sum_{s=1}^3\sum_{i\in\mathcal C_e}\sum_k
\alpha_{i|e}\beta_{i,s,k}\left(\prod_{r=s+1}^3F_{e,r}\right)(G_{i,s,k}+\xi_{i,s,k})}.\tag{6}
\]

**重点**：每阶段独立重新创建动量，因此不能把三个阶段的所有 minibatch 当成“连续动量不重置”的单阶段更新；过去噪声对后续阶段的影响通过 \(\prod F_{e,r}\) 传播。若实际阶段采用不同 Client 集合/不同归一化权重或执行重试，必须使用真实被认证的传播图，式 (6) 不能原样套用。

## 4. Cloud 传递系数

设当轮**事先固定、实际会被 Cloud 接纳**的客户端总样本量为

\[
N_C=\sum_{i\in\mathcal C_{\rm direct}}n_i+\sum_eN_e.
\]

Cloud 更新采用服务器步长 \(\gamma\) 与按样本数加权的上传模型：

\[
\Delta_t(h)=\gamma\left(\frac1{N_C}\left[\sum_{i\in\mathcal C_{\rm direct}}n_i\theta_{i,K_i}
+\sum_eN_e\Theta_{e,3}\right]-\theta_t\right).
\tag{7}
\]

将式 (4)、(6) 代入，定义

\[
\begin{aligned}
\bar f(h)&=\sum_{i\in\mathcal C_{\rm direct}}\frac{n_i}{N_C}f_i+\sum_e\frac{N_e}{N_C}\prod_{s=1}^3 F_{e,s},\\
a(h)&=\gamma[\bar f(h)-1].
\end{aligned}
\]

对 LIIC 中第 \(k\) 步、LIIEIIIC 中第 \(s,k\) 步，传递系数分别是

\[
\boxed{\kappa_{i,k}^{\rm LIIC}=\gamma\frac{n_i}{N_C}\beta_{i,k}},\tag{8a}
\]

\[
\boxed{\kappa_{i,s,k}^{\rm LIIEIIIC}=\gamma\frac{n_i}{N_C}\beta_{i,s,k}\prod_{r=s+1}^3 F_{e(i),r}}.\tag{8b}
\]

所以真实训练更新具有结构

\[
\boxed{\Delta_t(h)=a(h)\theta_t+\sum_{j\in\mathcal E(h)}\kappa_j G_j+\sum_{j\in\mathcal E(h)}\kappa_j\xi_j},\tag{9}
\]

其中 \(\mathcal E(h)\) 是该方案所有 Sample-DP SGD 步的集合。对于 **LIIE/SKIP**，没有 Cloud 上传，因此其 \(\kappa=0\)，但其理想人口梯度 \(p_i\nabla F_i\) **仍保留在** \(\Delta_t^*\) 内；绝不在理想参考中丢弃。

若整个 Cloud 都没有接纳贡献，可单独定义 \(\Delta=0\)，并令 \(U=\|\Delta_t^*\|^2\)。

## 5. 二阶矩上界的严格证明

定义确定性项（条件于决策前信息 \(\mathcal H_t\)）：

\[
D_t(h)=a(h)\theta_t-\Delta_t^*.
\]

令

\[
G(h)=\sum_j|\kappa_j| C_j,\qquad
V(h)=d\sum_j\kappa_j^2\tau_j^2.
\]

因为每步的**裁剪平均梯度**都满足 \(\|G_j\|\le C_j\)，不管它如何依赖历史噪声，都有

\[
\left\|\sum_j\kappa_jG_j\right\|\le G(h) \quad\text{逐样本路径成立}.
\]

如果所有 \(\xi_j\) 条件于 \(\mathcal H_t\) 相互独立、均值为零，且 \(\mathbb E_t\xi_j\xi_j^\top=\tau_j^2I_d\)，并且 \(\kappa_j\) 不受**本轮随后产生的噪声**影响，则

\[
\mathbb E_t\left\|\sum_j\kappa_j\xi_j\right\|^2 =V(h).
\]

Minkowski 的 \(L^2\) 三角不等式给出

\[
\begin{aligned}
\sqrt{\mathbb E_t\|\Delta_t(h)-\Delta_t^*\|^2}
&\le\|D_t(h)\|+\left\|\sum_j\kappa_jG_j\right\|_{L^2}
+\left\|\sum_j\kappa_j\xi_j\right\|_{L^2}\\
&\le\|a(h)\theta_t-\Delta_t^*\|+G(h)+\sqrt{V(h)}.
\end{aligned}
\]

因此

\[
\boxed{\displaystyle
U_t^{\mathrm{transfer}}(h)=\left[\|a(h)\theta_t-\Delta_t^*\|+\sum_j|\kappa_j|C_j+\sqrt{d\sum_j\kappa_j^2\left(\frac{2C_j\sigma_j}{B_j}\right)^2}\right]^2
}\tag{10}
\]

满足所需 \(U_t^{\mathrm{transfer}}(h)\ge\mathbb E_t\|\Delta_t(h)-\Delta_t^*\|^2\)。

如果只有决策前的范数证书，则

\[
\|a\theta_t-\Delta^*\|\le |a|\|\theta_t\|+\|\Delta^*\|,
\]

可给出不需要点积的保守版。若可信模拟器能提供 \(\langle\theta_t,\Delta_t^*\rangle\)，确定性项可精确写为

\[
\|a\theta_t-\Delta_t^*\|=\sqrt{a^2\|\theta_t\|^2+\|\Delta_t^*\|^2-2a\langle\theta_t,\Delta_t^*\rangle}.
\tag{11}
\]

**不能轻易写成** \([\|D\|+G]^2+V\)。过去的 DP 噪声影响后续 SGD 梯度，交叉项未必零；(10) 保留了这种依赖下的有效性。

如果还有额外的独立、中心化 Cloud 释放噪声，其通过线性聚合后的已证二阶矩可合并到 \(V\)；如果出现真实 HE 数值误差、裁剪后的非线性上传、浮点操作、随机客户端丢包、隐私机制额外噪声或未覆盖的模型状态，需要相应的**非零已证误差界**，不能默认为零。一般可用再加上 \(r_{\rm num}\) 的 RMS 三角界。

## 6. 与 Stage 19 当前界的不同

Stage 19 的诊断式大致为

\[
U_t^{\mathrm{old}}=(A_t+\sum_{i\in\mathrm{Cloud}}c_i[B_{i,h_i}+\|v_i\|])^2,
\]

其中 \(B_{i,h_i}\) 对多阶段 SGD 使用多次逐步 Minkowski，且未充分利用 Edge 阶段间动量重置与独立噪声方差结构。

新式 (10)：

- 通过 \(M^K\) 精确保留 weight decay 与动量；
- 用 \(\sqrt{\sum\kappa_j^2\tau_j^2}\) 替代噪声范数的逐步绝对值和；
- Edge 平均及 Cloud 归一化直接进入 \(\kappa\)，减少重复松弛；
- 理想更新是**完整向量** \(\Delta_t^*\)，不必逐个加 \(c_i\|v_i\|\)。

**注意：未证明新式在所有参数与方案上都严格小于旧式。** 两式的误差项分解不同；只有确认它们都覆盖同一实际随机更新后，才能严谨比较上界紧致性。

## 7. RDP 参数核对与 Shadow Audit（必须 fail closed）

在真正按轮计算新式 (10) 前，至少记录并校验：

1. **选模前**：当前 \(\theta_t\) 的可训练参数维度与范数、\(\Delta_t^*\) 的范数和（Oracle 场景中的）点积；每个候选的 \(n_i\)、Edge ID、模式、每阶段实际计划的 \(B_{i,s,k}\)、\(C_{i,s,k}\)、\(\sigma_{i,s,k}\)、学习率、动量、weight decay、Cloud 步长 \(\gamma\)；计划采用的 Cloud cohort 和 Edge 权重。
2. **RDP 会计**：已选 \(\sigma\)、每步实际采样率/批大小、RDP orders 与累计账本、预计事件数；Worker 回执的 `executed_sample_optimizer_sigma`、`executed_sample_optimizer_clip_norm`、`sample_dp_optimizer_steps` 必须与计划逐阶段一致。\(\varepsilon\) 不能直接替代 \(\sigma\)。
3. **训练路径**：确认优化器在每阶段重置；准确核对 Edge 群组、阶段次数、样本数及回传模型，实际 Cloud 接纳集合和输入样本权重。
4. **额外保护**：只接受能解释的真实上传机制；额外 DP、非线性 clipping、真实 HE 误差没有证明就拒绝输出“有效上界”。`HE profiled` 只能做性能/算法模拟，绝不能据此声称真实密文保护。
5. **敏感性**：Oracle 的向量范数、Gram、点积及 shadow 日志可能泄漏训练信息，仅能在受控模拟器本地保存；未做 DP 化的这些统计不能直接对外发布。

**Shadow 阶段**：只在日志中同时记录 `U_old`、`U_transfer`、`D_norm`、`G_radius`、`sqrt(V)`、`sigma`、`batch_schedule`、`cohort_consistency` 和拒绝原因；Pareto 继续用 Stage 19 老的诊断 \(J\)。通过端到端训练路径审计、数值误差审计之后，再考虑将新式投入选择。

## 8. 目前三轮实验能与不能证明的事

已有 CIFAR-10 / ResNet18 / 20 Client + 2 Edge / 3 轮结果中，Stage 19 的混合候选新 \(J\) 为 **19636.5379、19634.0653、19635.0531**，相应半径约 140.1。可确认它们随着轮次更新、但变化很小，且 LIIEIIIC 的保守半径远大于 LIIC。

旧 JSON 快照目前未保存完成计算式 (11) 所需的当前参数范数与点积，也没有将每一候选的 \(\sigma\)、批量计划作为一致性信息保存。**不能仅凭那三轮 CSV 正式重算 (10)，不能伪造新的 ResNet18 J 数值。** 下一次 Shadow Run 应从决策前快照增加这些审计字段，再与每轮 Worker 实际执行记录匹配。

## 9. 参考文件与验证范围

`dynfl_transfer_j_reference.py` 实现了式 (4)、(8)、(10) 的独立参考模型；`test_dynfl_transfer_j_reference.py` 包括 PyTorch SGD 的逐步核对及 Edge 加权聚合测试。参考测试通过只证明参考代码在所测有限例子符合代数与方差结构，**不能替代运行路径、独立性及整个 DynFL 系统的形式化验证**。

**推荐下一步**：为决策前 Oracle 快照增加 `theta_norm`、`theta_dot_ideal_delta`、`trainable_dimension`、计划 `C/B/sigma` 等字段；另写不参与 Pareto 的 Shadow Audit 读取快照与选模 CSV。若实际 Cloud cohort 与快照不一致，或 RDP sigma/阶段事件不一致，立即报告 `UNSUPPORTED / NOT_CERTIFIED`，不填写一个无根据的 \(U\)。
