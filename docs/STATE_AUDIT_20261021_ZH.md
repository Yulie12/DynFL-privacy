# 完整研究任务：第一阶段状态报告

检查日期：2026-10-21，Asia/Singapore。报告是本轮实际读取进程、文件、配置、源码及
执行测试所得的快照，不是实时监控页面。未启动正式实验，未修改科学参数或论文正文。

## 1. 版本及恢复点

- 分支：`feature/history-ema-j-shadow`。
- 起点HEAD：`1b4b4ffe7b1bc0889306fcd7735714011bce69ee`。
- 本地远程跟踪关系：`origin/feature/history-ema-j-shadow`，ahead 6；没有fetch/push。
- 起点已跟踪工作区无修改。4个未跟踪用户文件保留原位：
  `paper/ELSEVIER_TEMPLATE_README`、`paper/main.pdf`、`paper/main_test.pdf`、
  `paper/realtime-tex-0.0.1.vsix`。
- 本地注释标签：`audit-start-20261021-093956`，指向上述起点HEAD。
- 恢复目录：`out/audit_start_20261021_093956/`。
  - `tracked_head.zip`：109个条目，CRC通过；SHA256
    `06f0fda01374f14156c399d3e5e63d55ed35aed14d969f7dd3c79c1a9516fc33`。
  - `user_untracked.zip`：4个条目，CRC通过；SHA256
    `543c463a52ed3fa701744b807ce321fb57c8db63da90f7d368b9c742070a5180`。
- 原始实验数据、历史备份、用户论文PDF均未删除或覆盖。

## 2. 实时进程及资源快照

- 检查时没有匹配本项目实验脚本的Python进程。
- 指定解释器实测为 `D:\soft\Python310\python.exe`。
- PyTorch `2.5.1+cu121`；CUDA可用；GTX1660Ti，6144MiB显存。
- 检查瞬间GPU占用1107MiB、利用率1%，不是零占用承诺，也不据此认定有训练进程。
- E盘可用663,974,912,000字节，约618.3GiB。
- 新协议全规模运行时长尚未基准测定；不得把旧协议耗时当成实测估计。

## 3. 场景完成及审计状态

| 结果目录（均在out下） | 协议 | 完成运行单元 | 本轮核验 |
|---|---|---:|---|
| global_barrier_100_smoke_fmnist | 新100%屏障 | 8/8，5轮，seed42 | 重新独立审计8项通过，跨轮单元0 |
| global_barrier_100_smoke_cifar10 | 新100%屏障 | 4/4，3轮，seed42 | 重新独立审计4项通过，跨轮单元0 |
| 新版正式矩阵 | 新100%屏障 | 0/48 | 没有新版正式配置目录，未启动 |
| research_campaign_v5 | 旧周期/跨轮 | 95个完整运行 | 90个旧审计通过；另5个旧正式运行无整场景审计 |
| research_campaign_v4 | 更早旧协议 | 47个完整运行 | 42个旧审计通过，版本已废弃 |

v5被主动停止于旧正式CIFAR-10；seed42/random_feasible保留90/100轮，无完整summary。
v4保留periodic_tick_0.25/seed44/proposed的19/20轮中断。
旧结果只能作为历史过程证据，不进入新版正式统计。

新版冒烟原始配置文件SHA256：

- Fashion-MNIST：`5cd95364a0e04e1dfacd1c40c4d3f324d48f8855db85d73f6e81f96e62796e21`。
- CIFAR-10：`6a3c05b8f6a3ab4879e1e5cd6bc43e62919fee0741a3287c55a4743e755811fd`。

## 4. 配置与执行入口

实际调用：`experiments/research_campaign.py` → `dynfed.paper_runner.run` →
`run_policy` → `VersionBarrier`（轮前/轮后require_ready、模型版本、全部客户端ACK）。
不存在独立调用旧轮次循环的正式分支。模型仍是LeNet-5全参数和ResNet-18冻结骨干/分类头。

两任务均为100客户端、10Edge、100轮、seed42/43/44、8策略、batch64、local_steps=2、
FL_EC的edge_stages=3、clip=1、epsilon=8、delta=1e-5、gamma_d=gamma_C=.9、
selection_period=5、evaluation_budget=256、100%更新单元准入。
本地参数是固定优化步数，不应改写成local epochs。
学习率Fashion-MNIST=.05，CIFAR-10=.01；训练/测试样本分别60000/10000、50000/10000。

原始配置文件SHA256：

- configs/paper_fmnist.json：`ef18861943bf873fc65a438ad9afb7bf3c8ca2f153332412dae2da9c2f06259b`。
- configs/paper_cifar10.json：`5bcd0b04d74ba89ea7555b163660422150ece67def58b784aa9ce0b3377b9c32`。

补齐默认值后按JSON(sort_keys=True,separators=(',',':'),ensure_ascii=False)计算的SHA256：

- Fashion-MNIST：`0c4ad89c892e30e7ad96299690988232b0af07f89ef1e686bb20aa0a1f7d610c`。
- CIFAR-10：`637b740d1c40eb11f299b7a6f549a7abd77cb68757fe971c25d2c18e931f8145`。

这些是配置快照，不冒充已经冻结并启动的48项正式预登记清单。

## 5. 论文及测试

- paper/main.tex、refs.bib、elsarticle-num.bst及实际引用的eecblockflow.png存在。
- 原稿SHA256：`9472b3b155c888cc8856e5e64564e6870062cd12c9fc5e080cebd7302ec65871`。
- 当前没有paper/results/formal_main.tex，正式结果未写入论文。
- 已存在历史编译产物paper/build/barrier100.pdf及日志；不将其冒充本轮新编译成果。
- 本机MiKTeX的pdflatex和bibtex可执行文件存在；沿用离线、禁shell-escape、禁自动安装
  的独立jobname编译方式，避免覆盖用户main.pdf和旧辅助文件。
- 本轮指定解释器运行 `-m pytest -q`：28 passed in 4.93s。
- 两个新版冒烟目录均在本轮通过audit()重新核验；没有修改其原始输出。

## 6. 第二阶段初步发现：未过正式启动门槛

已确认：EMA平滑的是模式平均增量相对于冻结历史全局增量参考的偏差；不是平滑原始模式
更新，也不是直接计算相对于参数模型的距离。预测/观测先使用旧参考，之后更新模式EMA，
最后更新全局参考。缺历史为None，第一次Cloud更新只初始化全局参考。

仍需处理的工程项：

- campaign的--resume仅支持整个场景全部运行已经有summary的情形；部分完成的策略—种子
  会直接报错，并非逐轮断点恢复。现有final_state也不含重建中途状态所需的全部账本/控制量。
- 当前没有完整的训练失败、掉线、超时状态及恢复测试；28项通过不代表这些边界已覆盖。
- 正式预登记清单缺逐运行单元配置/源码哈希锁定；独立汇总入口也应加审计门禁。
- 重复更新、错误版本更新等需要明确的接收校验与故障注入测试，不能仅依赖正常planner
  恰好不生成这些输入。

需要用户确认的科学定义：

- 目前“无EMA”和“单轮评价”不是两个独立定义。gamma_d=0就是保留最近一次有效偏差；
  若两者都据此实现，它们是相同方法，不能包装成两个不同基线。
- 当前轮候选的真实单轮J需要先训练候选才能得到，不能在普通在线选模前使用；否则引入
  额外训练/隐私成本或事后信息。若指“前一次有效观测”，必须明确采用这一名称。
- 同时改gamma_C会改变J_obs的参考目标；识别模式偏差平滑本身的作用，应固定gamma_C，
  另将两系数均为零作为参考变化敏感性对照。

建议待确认的核心比较：无J（latency_only）、最近有效偏差（gamma_d=0，gamma_C=.9）、
完整History-EMA-J（.9/.9），保持其余预算一致。两系数为0另列敏感性。
这是待确认方案，不是已修改配置或已开始实验。按用户要求在此暂停科学设计扩展。

## 下一步

确认以上基线定义后，补齐失败/恢复、接收校验、逐单元预登记及审计门禁；重新运行单元、
集成和小规模训练，再冻结配置。只有必要检查通过才报告完整配置并启动正式矩阵。
当前不声称完整研究流程、正式实验或论文定稿已经完成。
