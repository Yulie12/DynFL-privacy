# 准备工作交付与并发修改阻塞

2026-10-21，Asia/Singapore。本报告是实际检查快照，不是持续监控状态。

14:08补充只读核验：上述并发改动已由其他写入者提交为e0edd3a
（experiment: add staged eight-candidate protocol），当前HEAD不再是1cad04d。
当时没有匹配的Python实验进程。另一个pilot实际配置为CIFAR-10分类头、10客户端2Edge、
11轮、seed42核心三臂、10轮初始化、每轮固定8候选、600/100样本、batch8、本地1步。
本任务没有创建该提交、启动该pilot或验证该新协议；不将49项旧基线测试结果套用到它。

## 本任务已完成（基于可恢复提交1cad04d）

- 两轮真实计费共享初始化；核心三臂latency_only、joint_no_ema(0/.9)、proposed(.9/.9)。
- 移除主名单中的独立J-only方法；其他五项依据系统放置/固定混合/随机基线问题保留。
- 原有自适应邻域/Pareto/Chebyshev搜索不变，保存全部已评价候选、T/J、归档和执行分布。
- 轮级检查点、本地DP步事务、模型/历史/账本/随机状态/版本/ACK恢复、环境指纹、并发锁。
- 缺失/重复/迟到/旧版本更新及ACK检查；失败不放行下一轮、不回滚未知隐私消耗。
- 直接终止进程导致未知主机耗时时标记null和缺失标志，不以零或最后一次尝试冒充总成本。
- 逐运行预登记、配置/源码/数据指纹检查、独立重建Cloud更新和RDP、候选选择与实际执行关联、
  完整性及实时审计统计门禁；统计同时报告各指标有效样本数。
- 本任务最后一次完整测试：49 passed in 19.58s，使用D:\soft\Python310\python.exe；
  包含真实CUDA LeNet逐位初始化/恢复测试。此结果不覆盖随后出现的外部改动。

## 实际验证数据

全部位于out中，不进入论文正式结果表。

| 目录 | 配置 | 完成及保存审计 |
|---|---|---|
| preparation_final_fmnist | 6客户端2Edge，6轮，3种子8策略 | 24/24 |
| preparation_final_cifar10 | 同上，冻结ResNet-18骨干/分类头训练 | 24/24 |
| recovery_cost_smoke_fmnist | 6客户端2Edge，3轮，seed42核心三臂 | 3/3 |
| recovery_cost_smoke_cifar10 | 同上，分类头训练 | 3/3 |
| shared_init_benchmark_fmnist_v1 | 完整数据，100客户端10Edge，4轮，seed42/proposed | 1/1 |
| shared_init_benchmark_cifar10_v1 | 同上，分类头训练 | 1/1 |

两个6轮验证在三种子下的核心初始化模型、模式历史、参考和账本均逐位一致。
每核心运行的两轮初始化实际20个DP步骤、累计epsilon约4.346863（6轮会计窗口）；
模型系统时间Fashion-MNIST为1.9404–2.1122s、CIFAR-10为1.1372–1.4230s。
本地训练调用实测时间分别约0.5727–1.8405s、0.3528–0.5138s，含步事务I/O，
不含全部外围日志/数据预处理。epsilon不是可线性加减的阶段成本。

100客户端4轮测时各执行1064个DP步骤，其中两轮初始化664步；主机运行实测约70.39s
及54.65s，预处理另计。粗估两个100轮正式矩阵训练/事务写盘22.3–33.8小时，
含审计和I/O波动预留24–48小时；不是保证，不能据4轮结果推断100轮学习效果。
粗估需预留约300GiB原始事务/检查点空间；检查时可用约614GiB。
测时源版本099b617，之后增加的是审计/环境及成本记录，不是训练超参数或数学公式修改。

相关机器可读记录：out/preparation_review_20261021/final_validation.json、timing_projection.json。
完整策略说明及恢复边界见FORMAL_PREPARATION_ZH.md。

## 预登记已生成，但当前不能核发启动许可

out/formal_shared_init_ready_final_20261021保存以1cad04d为基线的48个正式运行单元配置、
代码提交/指纹、数据文件/权重指纹和输出位置。正式运行0已启动。
此前多次清单是工程冻结过程的历史记录，不能混用。

最后一致性校验实际拒绝：`RuntimeError: preregistered source mismatch`。
这不是训练失败，也不是为了得到好结果而重跑；防护正确识别当前工作区不同于已登记源代码。

13:57只读检查发现本任务没有创建的改动：

- 8个已跟踪文件改变：两个正式配置、paper_config、paper_runner、paper_search、
  audit_paper、run_paper、test_paper_protocol。
- 新增stage1_20_fmnist.json、stage1_20_cifar10.json和plot_staged_selection.py。
- 正式配置改为10轮初始化、selection_period=1、evaluation_budget=8、staged_eight_v1，
  并增加固定八比例候选和分配代理优化；与本轮已验证的2轮/周期5/预算256方案不同。
- 捕获非本任务启动的PID39896，运行stage1_20_cifar10的11轮pilot，输出目录
  out/staged_eight_validation_20261021/cifar10。未停止或干预该进程；PID只代表当时快照。

这些可能来自用户或另一个任务，均保留，不覆盖、不回退、不纳入本任务提交。
本轮没有启动任何100轮正式运行，没有push或合并main，没有修改paper/main.tex。

## 需要用户决定

先确认以哪个实验版本继续：1cad04d的两轮共享初始化/原搜索方案，还是新增的
10轮初始化/固定八候选方案。涉及协议、搜索与预算，不能由代理自行合并或猜测。
确认后应先协调同一工作区的写入者，再重新冻结、测试和预登记。当前阶段状态是
“原基线准备验证通过，但工作区并发改动导致冻结与启动条件失效”，不是“正式实验已完成”。
