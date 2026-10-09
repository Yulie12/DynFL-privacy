# Split-DP-v2：独立表示机制验证（不修改原项目）

## 目的与边界

当前 `resnet18_pretrained_head` Split 路径的 End 输出 `128 × 4 × 4 = 2048` 维特征图，原 Edge 接续 `layer3 / layer4 / avgpool / classifier`。**不能**不改 Edge 网络就把 2048 维向量换为 64 维。

本原型用于回答一个更小的问题：冻结原 End 编码器后，`空间均值池化 -> 公共固定正交投影(32/64/128 维) -> 逐样本 L2 裁剪 -> Gaussian DP -> 每样本只发布一次、之后缓存重放` 是否保留了分类信号。

为隔离影响，这里采用**全新、简单的线性分类头**，并分别测试 `clean / clip_only / clip_plus_dp`。不接入 DynFL 七模式，不接入其调度/HE/隐私账本，因此**不是正式联邦训练，也不构成端到端 DP 证明**。标签和评估指标不受 DP 保护，结果只能作为内部诊断。

## 在 AutoDL 上运行

将这个目录单独放在 `/root/autodl-tmp/split_dp_v2_probe/`（不要覆盖 DynFL 文件），并确认 DynFL 原项目位于 `/root/autodl-tmp/DynFL-privacy/`。

```bash
cd /root/autodl-tmp/DynFL-privacy
python /root/autodl-tmp/split_dp_v2_probe/test_mechanism.py
python /root/autodl-tmp/split_dp_v2_probe/probe.py \
    --project-root /root/autodl-tmp/DynFL-privacy \
    --dims 64 \
    --train-limit 1200 --test-limit 400 \
    --epsilon-feature-total 4 --delta-feature 1e-6 \
    --clip-norm 0.25 --epochs 12 --lr 0.02 \
    --output out/split_dp_v2_probe_64.csv
```

输出 `out/split_dp_v2_probe_64.csv` 及对应 JSON。默认特征保护预算仅是研究选用的 **ε_feature ≤ 4, δ_feature = 1e-6**，与正式 ε_total=8 还需要进行严格组合；这不是沿用原始正式账本的证明。如果一次指定 `--dims 32 64 128`，程序会把三次**针对同一数据记录的发布**一起校准到 `epsilon-feature-total`；单独多次运行不会自动跨进程记账。

## 验证门槛

1. `clean` 明显高于 CIFAR-10 的 10% 随机基线：否则先排查冻结编码器和降维损失。
2. `clip_only` 与 `clean` 差距小：否则先修正裁剪前的表示尺度/归一化。
3. `clip_plus_dp` 明显高于 10%，且缓存重放计数正常：再考虑正式训练集成；若仍约 10%，本设计不能满足效用要求，不能绕过隐私噪声。
4. 在拟集成时，应先修改正式 End/Edge 接口、校验离散缓存生命周期，按真实发布次数扩展 per-record / per-mechanism 账本并验证 `label-gradient / update / restart / dynamic switching`，最后才能恢复七模式。**不能只把此脚本的结果当作正式机制。**

## 会导致 DP 声称无效的因素

- 这个原型仅提供“固定公共投影 + L2 裁剪 + 高斯机制”的**特征发布理论界限**：替换邻接，特征敏感度 ≤ `2C`，无放大；`ρ = k/(2σ_multiplier²)`，`ε ≤ ρ+2√(ρ ln(1/δ))`。`k` 是同一样本的**新发布**次数。
- 缓存**只在当前进程内生效**，重新启动、重新训练、切换模式/编码器、不同增强视图、不同投影或不同终端会重新产生噪声，必须组合核算。重放已有受保护数值不产生新的特征发布成本，但**其他依赖私有数据的消息仍可能产生额外成本**。
- 固定编码器必须来自公共预训练权重，且固定数据预处理；不得根据私有数据拟合投影/归一化后不计费用。
- 标签、模型梯度、聚合更新、指标、checkpoint、缓存泄漏、客户端关联记录的威胁模型均未证明；运行日志包含私有派生诊断，不能当 DP 公告发布。
- 原型使用可复现的 `torch.Generator`，并非密码学安全随机源，不可作为生产私有发布实现。
- 多次独立运行同一批私人样本要累计预算；`epsilon_feature_total=4` 不是免费额外预算，不等于正式端到端 `(8,1e-5)`。

## 原始文件

本目录全部为新增文件，不涉及原项目源码、原始 JSON、校准表、PDF 或 `main.tex`。
