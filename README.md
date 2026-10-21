# DynFL privacy: paper implementation

This repository implements `paper/main.tex` with three modes: `FL_E`, `FL_EC`,
and `FL_C`. The sole training entry point is `experiments/run_paper.py`.

```powershell
D:\soft\Python310\python.exe -m pytest -q
D:\soft\Python310\python.exe experiments/run_paper.py --pilot --rounds 7 --seeds 42 --policies proposed
```

正式运行：

```powershell
python experiments/run_paper.py --config configs/paper_fmnist.json
python experiments/run_paper.py --config configs/paper_cifar10.json
python experiments/run_suite.py --config configs/paper_fmnist.json --dry-run
python experiments/summarize_paper.py out/paper_fmnist/YOUR_RUN
python experiments/audit_paper.py out/paper_fmnist/YOUR_RUN
```

仅使用本地 Fashion-MNIST IDX / CIFAR-10 数据与本地预训练 ResNet-18 权重。
`--pilot` 固定为 6 clients / 2 Edges，减少样本、batch、搜索预算；轮数和随机种子
由命令显式指定。`--rounds` 改变会计总轮数，不能作为正式100轮运行的中途暂停。
新配置拒绝旧字段与旧策略。程序不会覆盖已有输出目录。

## 论文公式与执行

- `dynfed/paper_history.py`：先冻结上一轮全局参考，计算接受单元的样本权重与
  模式平均偏差，更新偏差 EMA，最后更新全局 EMA。第一轮只建立全局参考。
  缺失估计与无 Cloud 更新的代价为 `None`。
- `dynfed/paper_timeline.py`：逐阶段递推 `FL_EC` 训练、上传、聚合、反馈。
  `FL_E` 只等待本 Edge 返回，异步 Cloud 上传进入跨轮队列；只有截止时刻前到达的
  单元被接受。所有权重与历史使用同一接受集合。
- `dynfed/paper_search.py`：缺历史时优先覆盖可行的缺失模式，再比较时延；历史
  可用时启动双目标搜索与归一化 Chebyshev 选择。单次搜索有明确评价次数上限。
- `dynfed/paper_privacy.py` / `paper_data.py`：本地逐样本联合梯度裁剪和 Gaussian
  DP-SGD。替换邻接的梯度和敏感度为 `2C`，噪声标准差为 `2Cσ`；采用论文保守
  Gaussian RDP，不声称采样放大。校准包含剩余轮数与 `FL_EC` 全部阶段；执行后
  记录实际机制次数。没有额外上传包噪声或 HE 协议。

论文未明确的全 `FL_E` 情况采用如下约定：由最早异步到达事件触发 Cloud 聚合；
这属于异步事件等待，不是等待所有 `FL_E` 上传的同步屏障。有同步模式时，
Cloud 截止时间和返回最大值均不依赖 `FL_E`。

`required_compute/client_compute_capacity` 使用一致的归一化计算资源单位；内存以 MB
配置；每个 Edge 的可同时激活客户端数由 `edge_group_capacity` 配置。时延是模型
时间，选择耗时和实际主机执行时间分开记录。模拟时间线运行在单机真实训练之上。

## 实验与可复现输出

每个 seed 共享固定数据划分、系统 trace 和相同初始模型。测试集只用于评估。
固定模式不可行时记录 SKIP。`FL_EC` 每组执行 `edge_stages` 次真实反馈训练，
向 Cloud 仅提交一次；`FL_C` 每客户端提交一次。

输出包括原始配置、划分/trace 与哈希、初始模型、逐轮指标、决策、DP回执、
Cloud接受单元与样本权重、逐轮绝对时间线、历史参考/偏差/预测向量和最终待接收队列。
总轮数结束时仍未接受的上传保持 pending，不能计入历史或最终全局模型。
汇总提供各 seed 记录、均值/标准差、配对差异；未达到准确率目标时保持空值。
审计脚本从逐轮保存的接受单元重新计算偏差 EMA、全局 EMA、预测向量和 Cloud 权重，
并检查初始模型配对、无重复接受、DP执行次数和预算上限。

论文的候选检查点独立重放排名、按 Edge 历史消融尚未实现。可选 NSGA-II 未保留；
小规模搜索对照提供严格受组合数上限控制的穷举。正式100轮实验及CIFAR-10训练仍需
在适合的计算环境执行。逐样本训练优先实现可核查的隐私机制，当前性能未优化。

## 清理

旧七模式、Split、Online-J证书、包级DP、HE/SecAgg、模拟准确率目标、旧四策略、
桥接训练、旧配置、旧实验与旧测试已从活动源码移除。删除清单见
`docs/CLEANUP_MANIFEST.md`；删除前工作副本保存在
`out/before_paper_cleanup_20261021/source_before_cleanup.zip`。
论文、数据和已有实验结果保留。新实现不兼容旧检查点，当前不提供断点续训入口。
