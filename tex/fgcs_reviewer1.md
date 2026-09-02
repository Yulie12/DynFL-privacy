# FGCS 独立审稿报告 F1

## 审查范围与结论

本文题为 *Dynamic Cloud-Edge-End Mode Selection for Federated Learning under Resource and Privacy Constraints*。本报告基于稿件全部 18 页，包括正文、算法、实验、附录和参考文献。审查按 FGCS 对分布式系统、云边端协同、动态资源管理、调度、系统安全、算法与可扩展性研究的要求进行，重点检查算法一致性、成本模型、Pareto 搜索、异步聚合、隐私形式化及端到端性能主张。

**FGCS 适配性**

主题与 FGCS 明显匹配。云边端分层联邦学习、异构资源约束、动态模式选择、异步缓冲聚合和隐私开销均属于期刊核心读者关心的范围。不过，当前稿件更接近一个由模拟器支撑的系统建模与启发式选择框架。算法闭环、隐私保证和端到端性能证据尚不足以支撑完整系统论文的中心主张。

**总体评价**

稿件把七种云边端训练路径编码为协作链路图，并将资源可行性、通信和计算时延、DP 预算、HE 开销与收敛代理纳入统一选择问题。这一组织方式清楚，也具有工程动机。主要问题并非题目不适合 FGCS，而是多个核心模块在形式化和实验实现之间没有闭合。特别是 admitted set 的循环依赖、收敛代理与实际异步加权训练不一致、隐私通道的保证对象不完整，以及遗漏搜索耗时后的端到端加速结论，均直接影响中心结论。

**建议**

Reject and resubmit，拒稿后重投。原因是需要重新定义并实现若干核心机制，并重做端到端实验。若作者能够完成下述 Blocking 项，稿件可作为一篇 FGCS 范围内的完整系统算法论文重新评估。

## 主要优点

1. 问题设置符合云边端分布式学习的真实矛盾。稿件同时考虑弱设备资源限制、链路变化、模式切换、跨边缘融合和隐私机制，而不是只优化单一通信指标。
2. 协作链路块、链路流和传输指示矩阵提供了较统一的模式描述语言。图 2 至图 4 对七种候选路径及其时延组成的解释较直观。
3. 稿件尝试区分特征裁剪偏差、特征噪声、更新裁剪偏差与更新噪声，并在附录给出推导。这比仅以经验权重构造效用函数更透明。
4. 实验报告了三随机种子的主结果，并包含策略更新周期、隐私预算、规模和组件消融。表 3 与表 4 给出了准确率、逻辑时间、通信量和隐私损失等多维结果。
5. 作者主动承认系统参数来自 profiling，且物理部署和搜索开销仍是限制。这为后续修订提供了明确方向。

## 潜在读者

本文会吸引研究分层联邦学习、边缘智能、IoT 资源调度、多目标优化、隐私保护分布式学习和异步聚合的 FGCS 读者。系统研究者会关注模式切换是否真正降低 wall-clock completion time，算法研究者会关注近似 Pareto 前沿质量，隐私研究者会关注不同释放对象和邻接关系下的组合保证。

## Major Concerns

### F1-M1 [algorithmic-consistency]

**Severity** Major  
**Blocking** Yes

**Claim pointer** 算法在每个决策时刻对全局模式剖面进行评价，并在异步缓冲聚合下最小化事件级系统时延和收敛误差。

**Evidence pointer** 第 6 页第 3.2.3 节，式 (11)；第 9 页式 (27) 至式 (28)；第 10 至 11 页 Algorithm 1 的 Require、步骤 9 至 13。

**Concern** admitted set `A^(t)` 同时被当作算法输入和候选剖面的结果。式 (28) 的系统时延取决于 `A^(t)`，式 (27) 的 DP 方差也取决于该集合大小。然而第 6 页又规定 admitted clients 由候选模式对应的完成时间、上传时间、缓冲阈值和排序共同决定。改变任一客户端模式会改变到达顺序、触发时刻和集合成员。Algorithm 1 却在搜索前要求一个固定 `A^(t)`，并用它评价所有候选。由此，候选剖面与其目标值可能不是同一个异步事件下的自洽结果。混合 edge-only、direct-cloud 和 two-level 模式时，不同聚合端点及不同触发事件如何共同形成单一 `A^(t)` 和单一 `max` 也未定义。

**Why it matters** 这不是局部表述问题。它会改变每个候选的时延、DP 方差、聚合权重和收敛目标，从而可能改变支配关系、Pareto archive 和最终选择。

**Resolution test** 对每个候选剖面由模式、链路到达过程和各层缓冲规则联合求出 edge 与 cloud 事件、admitted sets、staleness 和返回路径。给出无循环的事件算法或固定点定义，并证明目标评价与执行语义一致。用小规模离散事件模拟逐候选核对式 (11)、式 (27)、式 (28) 和实际执行日志。

### F1-M2 [technical-soundness]

**Severity** Major  
**Blocking** Yes

**Claim pointer** 式 (27) 是可用于选择的 DP 诱导收敛误差度量，删除该目标会显著降低准确率，因此该项对联合权衡是必要的。

**Evidence pointer** 第 8 页式 (24) 至式 (26)；第 9 页式 (27) 和式 (31)；第 13 页第 5.6 节；第 15 页表 4；第 16 页式 (A.20) 至式 (A.31)。

**Concern** 推导采用独立同分布噪声、`K` 个客户端均匀平均、无陈旧度项的一步更新，并假设原始全局目标满足 smoothness 和 PL 条件。实际算法则使用异步缓冲 admitted sets、edge-normalized 非均匀权重式 (39)、多级聚合、不同模式和可能不同完成时刻。附录没有把实际更新算子还原到式 (A.20)，也没有处理非均匀权重、客户端异质性、异步 staleness 或 mode-dependent split training。ResNet-18/CIFAR-10 上也未验证 PL、Jacobian 和 Lipschitz 常数，或说明这些不可观测量如何被可靠估计后输入选择器。

**Why it matters** 收敛代理是 Pareto 搜索的两个核心目标之一。若它不是实际训练过程的有效上界或经验证的排序代理，Pareto 解和“convergence error term is necessary”的因果解释均不成立。表 4 的一次种子消融只能说明该实现配置下删除某评分项相关于准确率下降，不能验证理论模型。

**Resolution test** 重新推导与式 (39) 和异步事件语义一致的加权、分层、含 staleness 误差界，明确所有假设及 mode-dependent 更新。若无法给出上界，应将其降格为经验 surrogate，并在大量候选和多轮状态上报告 surrogate 与真实后续损失或准确率变化的 rank correlation、校准曲线和失效案例。至少跨多个 non-IID 强度、模型和数据集验证排序稳定性。

### F1-M3 [privacy-formulation]

**Severity** Major  
**Blocking** Yes

**Claim pointer** RDP accountant 对 embedding 和 update 两个 DP 通道进行累计，候选满足固定总隐私目标，因此方法维持隐私可行性。

**Evidence pointer** 第 4 页式 (6)；第 7 页式 (15) 至式 (18)；第 8 页式 (22)；第 10 页式 (37) 与 Algorithm 1 步骤 2 至 4；第 12 页表 2 和表 3；第 14 页图 8。

**Concern** 隐私保证的邻接关系、保护主体和发布机制没有形成一致定义。特征机制用任意两个输入间 `2C` 的 replacement sensitivity，但稿件同时把 embedding 与 backward gradient 放入同一通道，而没有给出 backward gradient 的敏感度或说明它只是已保护特征的后处理。更新机制看起来是 client-level DP，但最终只分别报告 `epsilon_emb` 和 `epsilon_upd`，没有给出两通道对同一训练数据联合释放后的总 `(epsilon, delta)` 保证。客户端采样、缓冲 admission、模式选择本身是否依赖私有数据，以及自适应选择下的 accountant 条件也未处理。式 (17) 以最大可能释放次数校准很保守，但不能替代对机制、邻接和组合的正式说明。

**Why it matters** “privacy feasibility”是标题、摘要和算法约束中的中心主张。当前数值只证明内部 accountant 没超过两个分立阈值，不能让读者判断究竟保护单样本、单客户端还是中间表征，也不能推出完整训练管线的联合 DP 保证。

**Resolution test** 明确每个通道的相邻数据集、随机机制、采样和释放单位。对 gradient link 给出敏感度证明或证明其为已 DP 输出的后处理。说明模式选择和 admission 的数据依赖性。给出自适应跨轮、跨链路和跨通道组合后的单一最终 `(epsilon_total, delta_total)`，并用标准 accountant 独立复算表 3 与图 8。若两个通道保护不同秘密，应明确禁止将它们作为同一个总体隐私预算比较。

### F1-M4 [performance-evaluation]

**Severity** Major  
**Blocking** Yes

**Claim pointer** Ours 相比 Individual Optimal 将 logical time 降低 24.9%，并提供训练效率方面的端到端优势。

**Evidence pointer** 第 11 页第 5.2 节；第 12 页表 3 和第 5.3 节；第 13 页图 6；第 14 页结论。第 12 页另报告 `S_P=1` 时总决策时间为 3834.97 秒。

**Concern** 主要效率比较使用 modeled logical time，未显示包含全局搜索本身的决策时间。表 3 中 Ours 的 logical time 为 7847.0 秒，Individual Optimal 为 10446.3 秒。正文另称 `S_P=1` 的总决策时间为 3834.97 秒。若这部分未计入 logical time，则 Ours 的顺序端到端时间约为 11682 秒，反而高于 Individual Optimal。即使搜索可以并行或与训练重叠，稿件也未给出调度时间线、硬件并行度或 wall-clock 测量来证明该开销被隐藏。通信量和 HE 的性能同样主要来自参数化模型，除“SEAL CKKS during training”的陈述外没有加密参数、密文布局、序列化字节、加解密吞吐或硬件信息。

**Why it matters** FGCS 系统论文的性能结论必须把调度器本身、重配置、隐私处理和训练路径放在同一 wall-clock 边界内。遗漏搜索开销可能逆转论文最醒目的加速结论。

**Resolution test** 报告从策略输入准备、搜索、模式切换、DP/CKKS 处理、通信、等待、聚合到模型返回的完整 wall-clock。明确哪些阶段重叠及其资源占用。主表同时给出 sequential critical path 和可实现并行 critical path。至少在真实或受控仿真的多机云边端环境中验证模型误差，并重新计算全部加速比例和置信区间。

### F1-M5 [pareto-search]

**Severity** Major  
**Blocking** No

**Claim pointer** 有界 archive 搜索近似精确 Pareto 集，并用理想点 Tchebycheff 规则选出实际折中方案。

**Evidence pointer** 第 9 页式 (34) 至式 (36)；第 10 页第 4 节；第 11 页 Algorithm 1 步骤 14 至 40；第 12 页表 2。

**Concern** 稿件描述了小规模精确枚举，却没有报告任何近似前沿相对精确前沿的验证。`K_P=16`、`I_max=50` 只有一个默认值。archive 超限时按 latency 顺序近似均匀保留并强制保留两个端点，这并不保证 Tchebycheff 最优附近或非凸前沿得到保存。随后 edge coverage repair 在已选单点附近逐设备替换，可能产生依赖修复顺序的结果，也可能不再是原可行域或覆盖约束可行域中的全局非支配解。相同距离时“returns one minimizer”没有确定性 tie-break，影响复现。

**Why it matters** 算法创新主要落在全局 Pareto 搜索。没有 approximation quality、稳定性和 constrained search 的证据，无法区分有效多目标算法与对当前模拟配置有效的启发式。

**Resolution test** 在可精确枚举的小规模实例上报告 Pareto recall、hypervolume gap、Tchebycheff regret 和最终决策一致率。对 `K_P`、`I_max`、初始化、客户端遍历顺序和 repair 顺序做敏感性测试。将式 (38) 直接纳入可行域再搜索，或证明修复步骤保持可行性、终止性和所声称的非支配性质。规定可复现 tie-break。

### F1-M6 [cost-model]

**Severity** Major  
**Blocking** No

**Claim pointer** 联合成本模型完整捕捉计算、通信、等待、聚合、切换和 HE 开销，并可用于跨模式比较。

**Evidence pointer** 第 4 至 7 页式 (8) 至式 (14)；第 12 页表 2；第 17 页 Appendix B。

**Concern** 多数关键项以 profile-based coefficient 表示，但稿件没有提供 profile 表、硬件、单位和拟合误差。`T_exec` 涵盖 end、edge、cloud 子模型计算，但资源可行性只明确约束 end device。edge/cloud contention、队列、并发客户端竞争和内存容量没有进入可行域。通信总量与逻辑时延缺少可重算的逐对象明细。HE 使用固定 `alpha_HE=6`，Appendix B 却说应从真实密文序列化校准，正文没有给出该校准。切换成本也只给公式，表 2 未列 `eta_h`、`eta_p`。因此“joint cost model”目前不可独立复算，也难以判断是否漏算或重复计算。

**Why it matters** 成本模型决定 Pareto dominance 和所有系统效率结论。未报告的参数可能让排序由人为设定主导。

**Resolution test** 发布每个模式和设备类型的计算、激活与模型状态、每条链路对象大小、带宽和基础时延、聚合系数、切换系数、CKKS 参数与实测开销。给出预测对实测的误差分布和消融，并加入 edge/cloud 排队、并发资源和内存约束，或相应收窄完整成本模型的表述。

### F1-M7 [completeness-reproducibility]

**Severity** Major  
**Blocking** No

**Claim pointer** 200 轮 CIFAR-10/ResNet-18 实验说明方法在异构云边端环境中提供实用的隐私、效用与效率折中。

**Evidence pointer** 第 11 至 15 页第 5 节、表 2 至表 4、图 5 至图 9；第 14 页结论。

**Concern** 证据仅来自一个数据集、一个预训练模型、12000 个训练样本、一个 extreme edge label skew 配置和主要三个随机种子。对比方法是作者构造的 Individual Optimal、No Protection 和 Random，没有与现有 HFL scheduling、split FL、resource-aware FL 或多目标启发式进行可比实现。主表没有显著性检验或配对置信区间。受控研究仅用 seed 42。稿件也没有代码、配置、candidate policy 枚举、设备 profile、数据划分生成法和 simulator 实现细节的可用性声明。

**Why it matters** 目前无法判断收益来自新搜索机制、候选模式设计、特定惩罚项、边缘归一化聚合，还是单一模拟参数组合。实用性与可扩展性主张超出证据范围。

**Resolution test** 增加至少一个不同任务或数据集、一个非预训练或不同规模模型、多个 non-IID 和资源异质性等级，并加入相关系统和调度基线。主比较采用足够随机种子，报告配对差值、置信区间和效应量。公开代码、数据划分、全部参数、profile 原始测量和运行脚本，或提供可匿名访问的复现包。

## Minor Comments

### F1-m1 [writing-clarity]

**Severity** Minor  
**Affected element** 附录引用。

**Evidence pointer** 第 7 至 9 页多处出现 “Appendix Appendix A”。

**Issue** 交叉引用重复，显示排版或 LaTeX 引用命令未清理。

**Required correction** 全文检查附录、图表和公式交叉引用，统一为 “Appendix A”。

### F1-m2 [notation-consistency]

**Severity** Minor  
**Affected element** 缓冲聚合比例符号。

**Evidence pointer** 第 6 页使用 `rho_buf`；第 12 页表 2 写 “Aggregation fraction rho = 1.0”。

**Issue** 表中符号和正文不一致，且读者无法确认主实验是否确实使用 `rho_buf=1.0`。若为 1.0，edge 层缓冲机制实际上等待全部关联客户端，与缓解 straggler blocking 的动机也需要解释。

**Required correction** 统一符号，明确每一层的阈值及主实验实际 admission 行为。

### F1-m3 [claim-moderation]

**Severity** Minor  
**Affected element** 表 3 后对 No Protection 的比较。

**Evidence pointer** 第 11 页称 Ours 的 mean best accuracy 与 No Protection 相差 0.22 个百分点；表 3 同时显示 final 和 last-10 accuracy 分别低 2.39 和 2.09 个百分点。

**Issue** 只突出 best accuracy 会弱化持续训练表现的差距。best-of-run 还受 200 次观察中的选择偏差影响。

**Required correction** 对 final、last-10 和 best 三项同等陈述，并将主要效用判断放在预先指定的终点或曲线下面积，而不是只用最大值。

### F1-m4 [figures-and-tables]

**Severity** Minor  
**Affected element** 图 5 和图 9 的平滑曲线与不确定性。

**Evidence pointer** 第 13 页图 5；第 15 页图 9。

**Issue** 图中使用 EMA 后再画样本标准差，可能掩盖逐轮波动。图 9 又只有单随机种子，没有不确定性，但视觉上与三种子主图相似。

**Required correction** 同时提供未平滑曲线或明确平滑顺序，主文或补充材料展示原始每种子轨迹。图 9 明示单种子，不据此作稳定性结论。

### F1-m5 [security-reporting]

**Severity** Minor  
**Affected element** HE 的安全性表述。

**Evidence pointer** 第 10 页式 (39) 后；第 11 页第 5.1 节；第 17 页 Appendix B。

**Issue** 稿件将 HE 与 DP 并列为 privacy mechanism，但没有威胁模型、密钥持有方、协同或恶意服务器假设、CKKS 多项式模数和安全等级。HE 本身也不防止最终聚合输出泄露。

**Required correction** 增加威胁模型与密码参数表，准确限定 HE 只保护传输和聚合过程中的更新机密性，不将其等同于 DP 的统计隐私保证。

## 技术失败项汇总

在论文中心论证成立前必须解决 F1-M1、F1-M2、F1-M3 和 F1-M4。它们分别涉及算法执行语义闭环、收敛目标与实际聚合的一致性、隐私保证的完整定义，以及是否遗漏搜索开销而逆转端到端性能结论。

## 原创性与主张范围

稿件的潜在新意主要是把云边端模式、链路对象和隐私策略统一编码，再做全局多目标选择。各组成部分，包括分层 FL、split training、buffered asynchronous aggregation、RDP accounting、CKKS、局部邻域 Pareto archive 和理想点 Tchebycheff，本身并非新方法。当前 Related Work 主要逐类介绍邻近方向，没有对最相近的联合 mode-selection 或多目标 cloud-edge FL 方法做功能矩阵和算法差异实验。因此，原创性可以被评价为有一定系统集成价值，但尚未证明为分布式云边端系统中的显著算法推进。作者应把主张限制在可验证的模式建模与选择框架，除非补充近似质量、基线和真实端到端证据。

## 最终建议

**Reject and resubmit**

题目适合 FGCS，稿件也有清晰的系统问题和较完整的初始框架。然而，四个 Blocking 问题会影响候选目标值、隐私可行性以及最主要的效率结论。它们需要重新定义算法、更新理论或调整主张，并进行新的端到端实验，不适合通过常规 major revision 在现有证据上局部修补。
