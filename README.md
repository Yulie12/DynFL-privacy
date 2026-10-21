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
  本轮预先确定更新单元集合，100%到达后Cloud聚合。`FL_E`局部链路可以提前完成，
  但不得跨全局屏障启动下一轮；无跨轮队列。所有模式从同一已确认Cloud版本训练。
- `dynfed/paper_barrier.py`：Cloud广播至全部客户端（含SKIP），收齐同版本ACK后才
  放行下一轮。拒绝缺失、重复、过时ACK或提前启动。
- `dynfed/paper_search.py`：缺历史时优先覆盖可行的缺失模式，再比较时延；历史
  可用时启动双目标搜索与归一化 Chebyshev 选择。单次搜索有明确评价次数上限。
- `dynfed/paper_privacy.py` / `paper_data.py`：本地逐样本联合梯度裁剪和 Gaussian
  DP-SGD。替换邻接的梯度和敏感度为 `2C`，噪声标准差为 `2Cσ`；采用论文保守
  Gaussian RDP，不声称采样放大。校准包含剩余轮数与 `FL_EC` 全部阶段；执行后
  记录实际机制次数。没有额外上传包噪声或 HE 协议。

当前唯一协议是 `protocol=global_barrier_100, cloud_quorum=1.0`，不再使用周期时钟。
90%阈值留作后续对照，当前配置明确拒绝它，防止伪实现。广播与ACK是单机逻辑网络事件，
并非物理分布式通信；假设所有客户端最终可达，不实现失联恢复。
默认ACK为64字节、模型安装为1ms，均为可配置假设，非实测参数。

`required_compute/client_compute_capacity` 使用一致的归一化计算资源单位；内存以 MB
配置；每个 Edge 的可同时激活客户端数由 `edge_group_capacity` 配置。时延是模型
时间，选择耗时和实际主机执行时间分开记录。模拟时间线运行在单机真实训练之上。

## 实验与可复现输出

每个 seed 共享固定数据划分、系统 trace 和相同初始模型。测试集只用于评估。
固定模式不可行时记录 SKIP。`FL_EC` 每组执行 `edge_stages` 次真实反馈训练，
向 Cloud 仅提交一次；`FL_C` 每客户端提交一次。

输出包括原始配置、划分/trace 与哈希、初始模型、逐轮指标、决策、DP回执、
Cloud接受单元与样本权重、逐轮绝对时间线、历史参考/偏差/预测向量、逐客户端ACK和版本。
每轮100%接受本轮预定单元；最终不存在跨轮pending。审计拒绝旧协议输出。
汇总提供各 seed 记录、均值/标准差、配对差异；未达到准确率目标时保持空值。
审计脚本从逐轮保存的接受单元重新计算偏差 EMA、全局 EMA、预测向量和 Cloud 权重，
并检查初始模型配对、无重复接受、DP执行次数和预算上限。

论文的候选检查点独立重放排名、按 Edge 历史消融尚未实现。可选 NSGA-II 未保留；
小规模搜索对照提供严格受组合数上限控制的穷举。正式矩阵使用
`python experiments/research_campaign.py --phase all --output OUTPUT`：
先执行多种子试运行和消融，再执行两任务正式矩阵。`campaign_status.json`记录阶段、
PID和完成情况。每个完成场景会审计并统计均值/标准差，全部正式运行完成后才导出论文
正式表格；pilot使用独立标记的表格。向量化/分类头解析梯度保持同一DP-SGD机制。

## 清理

旧七模式、Split、Online-J证书、包级DP、HE/SecAgg、模拟准确率目标、旧四策略、
桥接训练、旧配置、旧实验与旧测试已从活动源码移除。删除清单见
`docs/CLEANUP_MANIFEST.md`；删除前工作副本保存在
`out/before_paper_cleanup_20261021/source_before_cleanup.zip`。
论文、数据和已有实验结果保留。旧协议检查点不兼容；shared_init_v1支持轮级与本地DP步级
事务恢复，使用run_paper.py --resume，禁止将旧账本重置后拼接新结果。

## shared_init_v1正式准备

核心方法为latency_only、joint_no_ema、proposed；共享两轮真实计费初始化，无EMA联合臂
只将gamma_d设为0，gamma_C保持.9。独立候选日志、故障/恢复边界、最终策略审核见
`docs/FORMAL_PREPARATION_ZH.md`。未预登记的100客户端100轮运行会被拒绝。
`python experiments/prepare_formal.py --output NEW_OUTPUT`只预登记，不启动训练。
当前不要启动正式矩阵；待验证及耗时估算完成后另行申请。
