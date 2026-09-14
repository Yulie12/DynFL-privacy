# 完整本地 epoch 修复与 5 轮诊断

日期：2026-09-14。

## 修改

正式 mainline fusion 不再将 `L_block_cycles=5` 传作实际优化步数上限。
普通模式执行 3 个完整 epoch；LIEIIIC/LIIEIIIC 执行 9 个完整 epoch，
保持客户端私有模型和优化器连续性。逻辑成本参数仍保留，legacy 路径仍保持旧上限。
候选评估中的训练调用也采用相同的 epoch/step 约定。

新增实际批次数与实际优化器调用次数，写入客户端决策和逐轮指标；更新 execution
revision 为 `paper_flow_v30_mainline_fusion_method2_epochs`，避免续跑旧步数语义。
没有修改梯度裁剪 5.0、SGD 动量 0.9、DP 会计、HE、正式 clip=0.1 或学习率。

## 验证

- 全量回归 328 项通过；候选训练路径调整后相关回归 21 项通过。
- 新测试通过实际 worker 和真实 SGD.step 调用验证完整数据批次，覆盖普通、
  本地多级、拆分多级模式，避免只检查配置字段写着“9 epochs”。
- `git diff --check` 通过。

## 诊断设置与结果

配置：`configs/method2_epochs_5round_diagnostic.json`。
命令：`D:\soft\Python310\python.exe experiments\run_paper_config.py --config configs\method2_epochs_5round_diagnostic.json --max-new-rounds 5`。

沿用之前 C=0.2 诊断的 CIFAR10 分类头、100 客户端/10 边缘、extreme_edge_label_skew、
seed=42、lr=0.01、selection_period=5。诊断资源门控与之前运行保持一致，
不等同于正式模板中 require_feasible=true 的比较。隐私 horizon 仍为 100，ε=8、δ=1e-5。
Random 用于覆盖多级模式，不将它与 Ours 的精度差解读为修复的因果效果。

有效结果目录：
`out/method2_epochs_5round_diagnostic/2026-09-14_10-54-44_lenet5_dynamic_newtex202608`。
同根目录下 10-52-58 的启动已中止，资源门控不匹配，不用于对照。

| 第 5 轮指标 | Ours | Random |
| --- | ---: | ---: |
| 测试准确率 | 13.30% | 13.30% |
| clipping 比例 | 100% | 100% |
| 未裁剪更新范数均值 | 0.27845 | 0.50985 |
| 未裁剪更新范数 P90 | 0.28040 | 1.04729 |
| 未裁剪更新范数最大值 | 0.28044 | 1.22028 |
| 噪声/信号范数比 | 51.55 | 52.43 |
| 实际优化器步数（全客户端总计） | 300 | 492 |
| 全局 DP 发布累计次数 | 5 | 5 |
| 累计 epsilon | 1.604967 | 1.604967 |

客户端日志确认普通模式为 3 epochs / 3 batches / 3 optimizer steps，
两种多级模式为 9 / 9 / 9。这一数据集每客户端不足 128 样本，每 epoch 只有一个 GPU batch；
一般情况下优化步数应为 epochs × batches_per_epoch，不能把 epoch 和 step 普遍等同。
两组各 5 轮均为 real HE，第 5 轮最大数值误差分别约 2.29e-8、2.61e-8。

## 当前结论

本地步数/多级训练一致性已修复并有实际运行证据。Ours 没选择多级模式，
其 C=0.2 轨迹基本复现旧诊断；Random 多级更新的未裁剪范数明显增大。
因此不能再把 0.2804 当作所有模式共用的自然更新尺度。

本次只验证 C=0.2；没有确定最优/合理 clip 区间，也没有证明训练收敛或自适应裁剪有效。
不能从这 5 轮直接进入“最优 C 已确定”的结论。固定/自适应选择和 100 轮实验尚未执行。
按当前工作顺序，本次不改写 TeX 的实验结论。
