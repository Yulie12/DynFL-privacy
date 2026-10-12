# V370：L/E/C 三层资源约束修订（兼容旧实验）

## 1. 严格区分三层需求

- **L 客户端**：训练模型部分所需的计算、参数、激活和优化器内存，分别与该客户端容量比较。这里仍沿用原有归一化模型和设备异构容量；**不是实测显存**。
- **E 边缘**：将连接到同一个 Edge 的 *选中客户端* CPU 服务需求求和，再与该 Edge 的独立 CPU 容量比较。内存使用一次共享更新聚合工作区（两个 update buffer）加各客户端需要缓存的输入包。
- **C 云端**：把所有到达 Cloud 的 CPU 服务需求合并后与 C 容量比较。Cloud 内存为一次共享工作区、直传客户端的包和每个 `(Edge, 模式)` 聚合组的一个输入包；**不是每一个层级客户端都给 Cloud 再存一份 update**。

`LIIEIIIC` 是 **Full-local training → Edge aggregation → Cloud aggregation**；因此它与 `LIIE`、`LIIC` 采用相同的 L 端计算系数 `1.15` 以及内存系数 `1.20`，E/C 的额外聚合工作量只在各自服务器计费。注意 `1.20` 仍是旧的归一化经验系数，非 ResNet18 head-only 的实测显存，后续需硬件标定。

## 2. 已修改文件

- `dynfed/training.py`：统一三个 Full-local 模式的本地内存系数。
- `dynfed/tier_resources.py`（新增）：各 Edge 与 Cloud 独立 CPU/内存容量检查；单位分别为归一化工作量和 MB；返回具体超限层、资源种类。
- `dynfed/selection.py`：增加 `tier_capacity_policy=legacy|enforce` 和 E/C 容量参数；新增 **可行性修复**，按训练样本量、候选时延和客户端 ID 确定性保留候选，超限则改为 SKIP，不改变模式、不放松 L 端/DP 检查。
- `dynfed/fmnist_lenet5_dynamic.py`：在训练、固定模式冻结和 DP 事件发生**之前**应用三层容量准入；日志新增 `tier_capacity_runtime_skip`、`tier_capacity_skip_reason`，轮级增加 `tier_capacity_audit_json`；同时修正 `has_feasible_candidate` 使用最后一个客户端候选集的变量作用域问题。
- `experiments/run_fmnist_lenet5.py` 与 `experiments/run_paper_config.py`：贯通 CLI 与论文 JSON 配置。
- `tex/paper/main.tex`：添加三层容量数学表达，并明确它是 opt-in 事后准入修复，原论文的实验并未启用。
- `tests/test_v370_three_tier_resources.py`（新增）：覆盖 Full-local 相同 L 内存、独立 E/C 容量、聚合缓存不重复计数、SKIP 修复和旧模式兼容。

## 3. 怎么使用

**默认是 `legacy`，以确保旧 100 轮实验的资源约束口径可复现**（但修正了所有 Full-local 的 L 端内存系数，因此要严格复现旧实验，请使用原始 ZIP）。

PowerShell 单行命令示例：

```powershell
D:\soft\Python310\python.exe experiments\run_paper_config.py --config configs\v370_paper_cifar.json --rounds 1 --seeds 42 --he-execution profiled --dry-run
```

要启用新模型，在一个**复制的**实验 JSON 的 `system` 字段内加入：

```json
"tier_capacity_policy": "enforce",
"edge_cpu_limit": 100.0,
"cloud_cpu_limit": 200.0,
"edge_memory_limit_mb": 128.0,
"cloud_memory_limit_mb": 512.0
```

**上面四项容量都是演示占位值，不是实测容量、不是推荐论文参数。** CPU 需求在 `ModeSpec` 中为归一化系数 ×（客户端样本数 / 150），而 E/C 内存按 `update_payload_mb` 和 `embedding_payload_mb` 表示。没有对 Edge/Cloud 的实测标定前，不能将此门槛写成“真实设备资源约束”。CPU/内存需求来自同一模式的不同阶段，不跨层相加。

## 4. 当前方案的边界（论文必须说明）

1. 这是**已选模式的容量准入修复**，不是容量约束与 Pareto 搜索的完全联合优化；超限客户端仅变成 SKIP，没有探索另一种模式是否能让更多客户端参与。这可能高估动态策略的 SKIP，适合作为下一步实现全局约束优化前的安全校验/诊断。
2. 设备内存仍是归一化 Profile 而非实际 peak VRAM，Edge/Cloud CPU 仍是归一化服务需求；E/C 缓冲内存是保守的同步聚合代理，不是物理峰值。应采用同一模型和相同标定条件公平重跑 Fixed 与 Ours。
3. 对 `require_edge_cloud_coverage` / `require_cloud_participation`，若新的资源门槛导致无法满足覆盖要求，会显式报错而非偷偷降低覆盖率。
4. `profiled` HE 不等于真实加密；当前 RDP 数值不构成端到端 DP 证明。这些隐私边界保持不变。
5. 第 99 轮预算边界精度问题**没有在这次资源修订中更改**，避免把两个独立问题混在一起。

## 5. 推荐验证顺序

先用 `--dry-run` 检查配置映射，再做 1–2 轮小样本、低服务器负载的 `enforce` smoke；确认没有越过 E/C 容量、`tier_capacity_runtime_skip` 可解释，再进行同配置 Ours、Fixed 100 轮。**不能把历史实验输出当作新约束实验结果。**
