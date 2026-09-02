# 代码—论文一致性检查清单

基准文档：`tex/paper/main.tex`。主真实训练路径：
`experiments/run_fmnist_lenet5.py` ->
`dynfed/fmnist_lenet5_dynamic.py`。当前执行版本为 `paper_flow_v10_baseline_stability`；旧版本
checkpoint 不允许自动续训。

状态定义：`PASS` 为公式、selection、actual execution、日志证据闭环；
`APPROX` 为论文已允许或需明确披露的近似；`FAIL` 为当前不一致；
`EVIDENCE` 为实现存在但日志不足。

## 当前审计结果

| ID | 检查点 | TeX 对应逻辑 | Selection / Actual / 日志 | 当前状态 | 通过标准或后续动作 |
| --- | --- | --- | --- | --- | --- |
| 1 | 七种模式集合 | `H={LIE,...,LIIEIIIC}` | `MODE_SPECS`、flow mode sets | PASS | 三处集合完全相同 |
| 2 | `LIEIIC` 拓扑 | `B_E*L_B -> A_c`，云聚合前无 `A_e` | 已改为 `CLOUD_DIRECT_MODES` | PASS | flow 中只有 `cloud_aggregate_direct` |
| 3 | 多级模式拓扑 | 仅 `LIEIIIC/LIIEIIIC` 先边聚合再上云 | 两种模式执行 `E_B` 次训练—边聚合—返回 | PASS | `multi_edge_loop_client_cycles` 与事件日志可证明 |
| 4 | 真实计算单元 | 端、边、云是逻辑执行层 | 子模型在同一宿主进程/GPU 内顺序执行 | APPROX | 论文需明确“逻辑仿真，不是物理分布式部署” |
| 5 | `N_lambda` 统一 | 通信、隐私预算、时延共用链路执行次数 | `_mode_link_events` 同时驱动通信和 DP 计数 | PASS | 每个 mode 的事件表必须有回归测试 |
| 6 | `L_B/E_B` 真实执行 | link block 与 edge loop 重复执行 | 每周期严格 `L_B` optimizer steps，多级模式执行 `E_B` 周期 | PASS | 不能只在时延公式中乘次数 |
| 7 | 传输大小默认值 | `S_emb=1.6MB`、`S_upd=4.0MB`、`alpha_HE=6` | `OBJECT_SIZES`、`PRIVACY_ALPHA` 已统一 | PASS | config 与 TeX 表一致 |
| 8 | 返回链路通信量 | return link 计通信、不计 DP/HE | event 表显式加入返回 update，机制固定 `none` | PASS | return volume 必须出现在 candidate communication |
| 9 | 链路速率矩阵 | 每条有向链路使用 `R_uv^t` | 每个 client-round-link 确定性生成一次，所有候选复用并写入 `link_state.csv` | PASS | 正式实验需声明速率 profile 的来源或校准方法 |
| 10 | 聚合时延 | `beta*sum(S_tilde)+T_fix` | selection/actual 共用 flow executor，按实际 admitted 有效载荷逐事件计算 | PASS | 日志必须保留 payload、beta、fixed 与 aggregation_time |
| 11 | Buffered admission | 按 pre-aggregation arrival 取前 `ceil(rho*N)`，触发时并列全收 | selection 与 executor 均含 tie 规则 | PASS | objective IDs 必须等于 actual IDs |
| 12 | 宿主机调度隔离 | 逻辑时延不应受单 CPU 串行排队影响 | wall-clock 仅记录，arrival 使用逻辑路径时间 | PASS | serial/process_pool 不得改变 admitted set |
| 13 | `S_P` 更新周期 | 非更新轮复用旧 profile | 可行时复用；预算失效时触发 feasibility repair | PASS | TeX 伪代码已加入 `x^(t-1) in X^t` 条件 |
| 14 | Feature-DP | clip 后按 replacement sensitivity `Delta_z=2C` 加高斯噪声 | `_protect_tensor_dp` 使用自动校准的 `sigma_feat*2C` | PASS | record-level RDP 单独记账 |
| 15 | Update-DP | 全局向量 L2 clip，再加 `N(0,sigma_u^2*C_u^2 I)` | `apply_unified_dp` | PASS | 禁止按 tensor 独立裁剪作为默认实现 |
| 16 | DP 事件预算 | 每个 protected link 按 `N_lambda*alpha/(2*sigma^2)` 在 RDP 空间组合 | feature 与 update 的实际扰动阶段按 event 表执行 | PASS | decision log 必须记录两类事件和转换后的 epsilon |
| 17 | Per-link policy | `pi_i(lambda)` 可对同类型不同链路选不同机制 | candidate 按 directed link 保存机制，`LIIEIIIC` 两条 update 链路可独立选择 | PASS | 决策、DP 账本、HE 执行和日志必须共用 link key |
| 18 | 真实 HE | HE update 用 CKKS，服务端只解密聚合结果 | SEAL/TenSEAL 真实密文聚合并有数值测试 | PASS | `require_real_he=true` 时后端缺失必须失败 |
| 19 | 混合 HE/DP 聚合 | 每客户端按其 link policy 执行 | SEAL/TenSEAL 分别累加 ciphertext 与 plaintext/DP 加权和，再在密文域合并 | PASS | 保留 mixed encrypted/plaintext 数值等价测试 |
| 20 | Privacy accounting | feature/update 不同邻接单位独立记账，HE 的 RDP 增量为 0 | 两个 per-client RDP ledger、候选投影硬约束 | PASS | 跨客户端报告 max，不得求和 |
| 21 | 内存约束 | `M_mem=s_state*m_i+a_i <= Q_mem,i` | mode requirement 与异构 client capacity 均进入候选可行性并写入决策日志 | PASS | profile 值来源需在实验设置中披露 |
| 22 | 资源/risk 可行集 | TeX 的 `X^t` 当前仅含资源与 DP budget | 代码额外把 risk、edge/cloud CPU 作为硬约束；已修正 `label:none` 将所有 split mode 误判为 risk=1 的问题 | FAIL | 选择：补入 TeX 公式，或从 proposed feasible set 移除 |
| 23 | 近似 Pareto search | bounded archive、邻域扩展、理想点 Tchebycheff | `choose_global_pareto_profile` | PASS | archive 与增量 Omega 回归测试均需保留 |
| 23A | 小规模 exact solver | TeX 描述枚举 `prod_i abs(S_i)` 个全局 profile | 当前没有全局穷举 solver | FAIL | 增加仅用于小 N 验证的 exact implementation |
| 23B | 可选 reference point | TeX 允许指定 reference 或使用 ideal point | 当前只实现 archive ideal point | FAIL | 增加显式 reference 参数或从 TeX 删除该选项 |
| 23C | selection-only 仿真的 `S_P` | 所有 proposed-method 仿真应按周期搜索 | 真实训练遵守；`run_selection_experiment` 仍每轮搜索 | FAIL | 在 `SelectionConfig` 和纯 selection runner 中实现复用/修复 |
| 24 | `Omega_hat` | clipping/noise/local variance + cloud fusion penalty | 每个客户端按当前模型和本地数据实测 feature clipping excess；其余平滑度、方差等统计量仍为常数 profile | APPROX | 保留 clipping profile 回归测试；论文披露其余 profile 来源与校准方法 |
| 25 | DP aggregation size | update-noise variance依赖实际 aggregation set | 直接云/边组按 admitted count；多级 final update 仍受 object-shared policy 限制 | APPROX | per-link policy 完成后按 endpoint 分组重算 K |
| 26 | Cloud fusion | selector 使用全 profile 的样本占比 | 同时记录 selector ratio、actual population ratio、actual admitted share | PASS | 不得把三者混成一个字段 |
| 27 | Switching cost | mode change + placement-state distance | 已加入 path time，但 placement 由工作量 proxy 估算 | APPROX | 用真实分层 state bytes 替换 proxy |
| 28 | 训练拓扑与聚合 | edge-only 不进入 cloud FedAvg；cloud mode 同步全局模型 | returned client states、rebasing、edge/cloud aggregation | PASS | 每种模式需单独 topology test |
| 29 | GPU/执行器 | GPU用于真实模型训练，不应改变仿真语义 | CUDA+serial 可用；process_pool 仅 CPU | PASS | 日志分别记录 device/executor/wall time |
| 30 | Baseline 公平性 | Appendix baseline semantics | 主路径均复用相同数据、模型、划分 | APPROX | 还需逐 baseline 固定候选域与隐私机制测试 |
| 31 | 实验表格可复现性 | TeX 表格必须来自当前执行版本和 `T=100` 主设置 | 现有表格是旧版 50 轮输出且早于 `paper_flow_v10_baseline_stability` | FAIL | 用当前严格可行、逐链路隐私、阶段化时延、共享稳定候选集和 admitted-payload 聚合版本重跑 100 轮多 seed，再重建图表与表格 |
| 32 | TeX 唯一规范源 | 实现只能对应一个明确版本 | `main.tex` 为规范源；`main_with_changes.tex` 是旧审稿快照 | PASS | 禁止用旧快照反向校验当前代码 |
| 33 | 固定 seed 可复现性 | 同一 seed 的模型初始化和客户端 batch 顺序一致 | PyTorch/CUDA 全局 seed、确定性 cuDNN、按 round/client/stage 派生 DataLoader seed | PASS | 保留固定 seed 的本地训练回归测试，并对正式实验执行多 seed |

## 第 80 轮下降的证据

被检查运行使用 `T=100`、`S_P=10`、`epsilon_0=4.0`、
`epsilon_upd=0.05`。CSV 第 79 个零基 round 后所有客户端剩余预算约为
`5.84e-15`；第 80 个零基 round 的机制由 `upd:dp` 全部切换为
`upd:he3`。因此临界点恰好满足 `4.0/0.05=80`。

DP 路径包含 `C_u=1.0` 的 update clipping；HE 路径保留原始更新而不做
DP clipping。在 `lr=0.15` 与 extreme edge label skew 下，去掉 clipping
会放大 client drift。该现象不应通过“给 HE 偷加 DP clipping”掩盖。上述证据
只解释旧标量账本为什么在第 80 轮产生强制切换，不能作为正式 DP 保证。
当前实现固定总目标，并根据 `T*N_lambda` 自动校准噪声；50/100/200 轮比较时
不再按轮数放大 epsilon。

## 每次修改后的固定核对顺序

1. 从 TeX 找到变量、公式、mode/link 定义和默认参数。
2. 检查 candidate 枚举、可行性筛选、`N_lambda`、预算和目标值。
3. 检查 actual training 是否执行同一 mode、机制、次数和 admitted set。
4. 检查 round/client/flow 日志能否独立证明前三步。
5. 运行 conformance、SEAL、resume 测试和最小真实训练 smoke run。
6. 只有当前 execution revision 的多 seed 完整运行可写回论文图表。

## 修改优先级

- P0：当前无未完成项；继续保留内存、per-link、混合 HE/DP 与 staged-latency 回归测试。
- P1：为 link-specific `R_uv^t` 和 aggregation beta/fixed 提供可复现校准来源；按 endpoint 完善 `Omega_hat` 的 K；使用真实 state bytes 校准 switching profile。
- P2：baseline 语义测试、完整 100 轮多 seed 重跑、论文图表重建。
