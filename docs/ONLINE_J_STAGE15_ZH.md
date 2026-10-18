# Stage 15：真实 LeNet-5 逐层梯度范数界 + 可选分块在线 J

**仅修改 J 数学与诊断；不改预算、DP、SecAgg、训练器、可信 Split 或资源准入。**

## 逐层结构性梯度范数界

Stage 14 的全局参数范数界，把五层的每一个 `||W_l||_F` 和 `||b_l||` 都替换为 `R=||θ||`；这产生重复的、过度保守的乘积。Stage 15 用训练前模型参数分别计算五层的真实权重和偏置范数上界。

在固定归一化 FashionMNIST、28×28 输入、ReLU、2×2 AvgPool、三层 FC、十分类 CE 下，令 `X=28*2.25`：

```
H1=(5 W1 X + 28 b1)/2
H2=(5 W2 H1 + 10 b2)/2
H3=W3 H2+b3
H4=W4 H3+b4
D5=sqrt(2); D4=W5 D5; D3=W4 D4; D2=W3 D3/2; D1=5 W2 D2/2
G_layer=hypot(5 D1 X,28 D1,5 D2 H1,10 D2,D3 H2,D3,D4 H3,D4,D5 H4,D5)
```

其中五层顺序是 `conv1/conv2/fc1/fc2/fc3`。这是一种条件成立的解析梯度**范数**上界，不是模式方向性误差证书，浮点舍入仍需独立认证。

不用重新训练，检查自己已保存的最终 `checkpoint.pt`：

```powershell
& 'D:\soft\Python310\python.exe' examples/check_lenet5_checkpoint_gradient_cap.py `
  --checkpoint 'out\stage13_real_liic_predecision\2026-10-17_03-03-07_lenet5_dynamic_newtex202608\full_dynfl\checkpoint.pt'
```

程序打印原 Stage 14 的 `G_whole`、新的 `G_layer` 和缩紧倍数。**最终 checkpoint 是第5轮更新后的模型：结果只能用于之后的第6轮事前模型，不能冒充第1～5轮的事前证书。** 只加载可信的本地 checkpoint（`torch.load` 使用 pickle）。未来开启 Shadow 时，每轮 JSON 会附带训练前 `layer_norm_uppers`；旧 JSON 不会自动回填。

## 分块 v2 的 J 改进（独立可证明半径才启用）

仍然使用相同的理想更新 `Δ*=-η*∇F(w_t)`、参与权重抵消式，并保持 `Q` 是 pre-SecAgg 独立零均值噪声第二矩，局部 DP 噪声不重复计数。将模型参数划分为互不相交、完整覆盖的坐标块 `b`：

```
A_b = ||[Σ_i(γa_i-p_i) vhat_i]_b||
R_b = Σ_i |γa_i-p_i| s_(i,b) + γΣ_i a_i r_(i,mode,b)
U_block = (sqrt(Σ_b (A_b+R_b)^2 + Q)+r_num)^2
L_block = (max(0,sqrt(Σ_b max(0,A_b-R_b)^2)-r_num))^2
```

由于不同坐标块正交（与随机误差独立性无关），若每块 `s_(i,b)`、`r_(i,mode,b)` 有**独立的事前 RMS 证明**，上式有效。软件与 Stage 14 标量式并行计算，取 `U=min(U_scalar,U_block)`、`L=max(L_scalar,L_block)`，不允许因新增数据变差。

可选 v2 state：

```json
"parameter_blocks":[{"name":"block0","start":0,"end":1},{"name":"block1","start":1,"end":2}]
```

对应 `population[i].ideal_rms_by_block`、`clients[i].candidates[key].relative_ideal_rms_by_block` 必须逐块有有效证明。维度及参数平铺顺序必须与真实模型一致。不能把一个整体半径随意平分给各层。原 v2 标量证书不受影响；Bias-variance 模式不支持该块接口，会严格拒绝。

**仍不等于七模式 J 认证：** 聚合参与关系、真实执行模式误差、数值误差与独立噪声机制缺少严格证书时继续 fail closed；不触发新的五轮真实训练。
