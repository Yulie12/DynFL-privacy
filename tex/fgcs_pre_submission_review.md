# FGCS 投稿前模拟审稿报告

稿件题目：Dynamic Cloud-Edge-End Mode Selection for Federated Learning under Resource and Privacy Constraints

审查范围：完整 18 页 PDF，包括正文、公式、算法、图 1 至图 9、表 1 至表 4、附录和参考文献。

评估基准：Future Generation Computer Systems 的主题范围与一般完整研究论文要求，包括 distributed systems、cloud/edge/IoT、dynamic resource management and scheduling、security、algorithm design、scaling and performance evaluation。三位审稿人采用相同稿件和标准，分别独立评阅，报告冻结后才进行综合。

## 总体结论

主题契合度高，但当前投稿准备度不足。两位审稿人建议 Reject and resubmit，一位建议 Major revision。综合建议为“先完成实质性预投稿大修，再作为新稿投稿 FGCS”。

稿件的主要优势是问题对口、统一建模较完整，并覆盖云边端执行位置、聚合流、DP、HE、资源约束和多目标选择。当前风险不在 scope，而在算法执行语义、隐私保证、端到端性能口径、基线强度和实验完整度。

## Reviewer 1

侧重算法一致性、成本模型、Pareto 搜索、异步聚合、隐私定义和端到端性能。

推荐：Reject and resubmit。

### Major Concerns

1. F1-M1，Blocking Yes。admitted set 同时被作为候选决策的结果和 Algorithm 1 的固定输入。不同候选会改变到达顺序、缓冲触发和集合成员，当前目标评价可能与实际执行事件不自洽。
2. F1-M2，Blocking Yes。收敛误差推导采用均匀平均、独立噪声和无陈旧度的一步更新，实际算法却采用异步 admitted sets、非均匀 edge-normalized 权重、多级聚合和不同模式。理论代理与真实更新算子没有闭合。
3. F1-M3，Blocking Yes。embedding、backward gradient 和 update 的邻接关系、敏感度与保护主体没有完整定义。两个通道分别报告 epsilon，不能自动得到完整训练管线的联合 DP 保证。
4. F1-M4，Blocking Yes。Ours 的 logical time 为 7847.0 秒，但另有 3834.97 秒决策时间。若未计入，顺序端到端时间约 11682 秒，可能高于 Individual Optimal 的 10446.3 秒。
5. F1-M5，Blocking No。近似 Pareto 档案没有与精确前沿比较，缺少 Pareto recall、hypervolume gap、Tchebycheff regret 和参数敏感性。
6. F1-M6，Blocking No。成本模型的大量 profile 系数、排队、并发、切换和 CKKS 参数未公开，无法独立复算。
7. F1-M7，Blocking No。单数据集、单模型、少量种子和内部基线不足以支撑实用性与可扩展性主张。

### Minor Comments

修正 Appendix Appendix A；统一缓冲比例符号；不要只突出 best accuracy；同时展示未平滑曲线；明确 HE 只保护什么攻击面。

## Reviewer 2

侧重领域新颖性、相关工作定位、强基线和完整期刊实验包。

推荐：Reject and resubmit。

### Major Concerns

1. F2-M1，Blocking Yes。链路表示、RDP、CKKS 开销、PL 代理、Pareto 档案和 Tchebycheff 选择主要由成熟组件组成。稿件没有明确证明新增了何种不可由现有框架表达或求解的能力。
2. F2-M2，Blocking Yes。比较对象主要为作者自定义的 Individual Optimal、Random 和改变隐私约束的 No Protection，无法建立相对现有 HFL、split-FL、隐私感知卸载和多目标调度方法的竞争力。
3. F2-M3，Blocking Yes。实验集中于 CIFAR-10、预训练 ResNet-18 和单一模拟拓扑，没有充分覆盖数据、模型、异质性、规模和动态资源变化。
4. F2-M4，Blocking Yes。收敛误差代理的假设、参数取得方式和对真实训练效用的排序能力没有形成理论或经验闭环。
5. F2-M5，Blocking Yes。DP 和 HE 的隐私语义、威胁模型与可比较性不完整。
6. F2-M6，Blocking No。代码、配置、设备 profile、数据划分和原始日志缺失，复现性不足。
7. F2-M7，Blocking No。三种子主实验和单种子控制实验不足以支撑稳定优势。

### Minor Comments

增加模式对照表；提高图表独立可读性；收缩 practical、global 和 privacy-preserving 等措辞；明确符号和表格统计口径。

## Reviewer 3

侧重统计、复现性、模拟真实性、图表与实际主张，重点判断能否在大修中修复。

推荐：Major revision。

### Major Concerns

1. F3-M1，Blocking Yes。随机重复过少，缺少置信区间、配对效应量和未平滑轨迹，多项目差异可能落在当前随机波动范围内。
2. F3-M2，Blocking Yes。设备、链路、拓扑、切分、训练、CKKS、DP 和收敛代理参数披露不足，且没有代码与数据可用性声明。
3. F3-M3，Blocking Yes。核心系统指标是 modeled logical time，缺少实测或网络仿真校准，也缺少带宽、掉线、拥塞、慢设备和缓冲阈值变化下的稳健性实验。
4. F3-M4，Blocking Yes。embedding、gradient 和 update 的敏感度及组合保证未闭合，可能影响 epsilon 的有效解释。
5. F3-M5，Blocking No。缺少现有系统方法、标准多目标优化器和精确小规模基线。
6. F3-M6，Blocking No。PL 型代理未被校准，单种子消融不足以证明该目标是 necessary。
7. F3-M7，Blocking No。CKKS 的攻击者、密钥持有方、串谋假设和数值误差未定义，不能与 DP 作为相同隐私强度直接比较。

### Minor Comments

明确标准差阴影含义；解释表 3 与表 4 的种子口径；修正交叉引用；改善七种模式的命名和总览图；增加 Code and Data Availability。

## Cross-review synthesis

### 共识优点

- 主题高度契合 FGCS 的云、边、IoT、分布式资源管理和安全方向。
- 联合考虑执行放置、聚合流、资源、DP、HE 与效用，问题设定具有系统研究价值。
- 链路块和链路流为复杂协作模式提供了统一表示。
- 已包含主比较、规模、更新周期、隐私预算和消融，具备完整论文的雏形。

### 共识阻断问题

1. 隐私保证没有闭合。F1-M3、F2-M5、F3-M4。
2. 收敛误差代理与实际更新过程及经验效用没有充分对应。F1-M2、F2-M4、F3-M6。
3. 系统性能证据不足，logical time 可能遗漏搜索成本，且缺少校准。F1-M4、F2-M3、F3-M3。
4. 缺少强且公平的外部基线。F1-M7、F2-M2、F3-M5。
5. 实验和复现材料不足。F1-M6/F1-M7、F2-M3/F2-M6/F2-M7、F3-M1/F3-M2。

### 单独但关键的问题

- F1-M1 指出的 admitted set 循环依赖可能改变候选目标值和最终选择，必须由算法定义或离散事件验证直接解决。
- F2-M1 指出的创新定位不足可能导致编辑阶段或首轮审稿质疑。需要从“组件组合”提升为具有明确新增表达能力、算法性质或可测系统收益的贡献。

## 投稿前必须完成的修改

### P0

1. 重写异步事件和 admitted set 定义，使每个候选配置的到达、缓冲、接纳和聚合过程自洽。
2. 将搜索、切换、DP、CKKS、通信、训练、等待和聚合纳入统一端到端时间，重新计算主结果。
3. 明确 DP 邻接关系、保护单位和每种消息的敏感度，给出跨轮、跨链路和跨通道的最终保证。
4. 明确 HE 威胁模型、密钥边界、安全参数和数值误差，不再把 HE 与 DP epsilon 直接等价比较。
5. 验证收敛代理与真实后续损失或准确率的排序关系；否则将其改称 empirical surrogate 并收缩理论主张。
6. 增加同约束的现有 HFL、split-FL、资源调度和标准多目标优化基线。

### P1

1. 在小规模实例上与精确 Pareto 前沿比较。
2. 至少增加一个数据集和一个不同模型，并系统改变 non-IID、带宽、设备能力、缓冲阈值和客户端规模。
3. 主实验和关键消融增加随机种子，报告配对差异、置信区间和效应量。
4. 公开模拟器、训练代码、配置、数据划分、设备 profile 和逐轮日志。
5. 最好增加小规模真实或硬件在环验证。若做不到，至少使用可复核的网络模拟器或实测 profile 校准 logical time。

### P2

1. 增加最接近工作的结构化比较表。
2. 为七种模式增加直观名称和对照表。
3. 展示未平滑轨迹并清楚区分模拟指标、理论代理和实测指标。
4. 修正附录交叉引用、符号和表格统计说明。
5. 增加 Code and Data Availability。

## 最终投稿判断

FGCS 是合适的目标期刊，但不建议以当前版本直接投稿。当前稿件不是简单语言润色即可达到可投状态。若 P0 全部完成并补充主要 P1 项，稿件可进入较合理的 FGCS 投稿区间。若无法闭合隐私保证、端到端时间和 admitted set 语义，应删除或显著收缩 privacy feasibility、practical compromise 和 global coordination 等核心措辞，否则首轮拒稿风险较高。

本报告不能代替 FGCS 编辑的正式决定，也未通过外部复现实验验证代码或数值。
