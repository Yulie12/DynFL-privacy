# DynFL 第一阶段：固定四边缘的云端 Aggregate-DP 等价性实验

**作用范围：** 仅用于内部可信聚合器（trusted curator）的**分层聚合算术验证**。这是在已有 `validate_trusted_aggregate_trajectory.py` 上增加的最小选项，并没有实现四个独立的边缘进程、网络通信、HE、SecAgg 或端到端 DynFL 隐私协议。

## 本次改动

- `--edges N`：固定、不重叠的 N 个客户端组，客户端编号 `i` 固定分配到 `i % N` 边缘。
- `--aggregation-topology hierarchical`：客户端**先独立 L2 裁剪**，每个边缘仅在内部聚合所属客户端更新，云端按每个边缘的实际客户端数量加权合并。
- `--check-direct-parity`：每轮同时在内部计算直接聚合结果，对照分层聚合，容差 `rtol=2e-5, atol=2e-7`；差距超标时中止。
- **只在最终云端聚合结果上加一次高斯噪声，再应用 `server_step`**；`--privacy-horizon 20` 对应 20 次云端 DP 事件，不是每个边缘一次。
- 默认仍为 `--edges 1 --aggregation-topology direct`，保持已有诊断的训练计算路径与随机数种子不变。

## 安装（优先用 patch）

把 `dynfl_fixed_hierarchical_stage1.patch` 放在 `E:\\YTT\\GROUP\\DynFL-privacy` 根目录，然后执行：

```powershell
# 先备份本机已有的轨迹实验脚本
Copy-Item .\experiments\validate_trusted_aggregate_trajectory.py `
    .\experiments\validate_trusted_aggregate_trajectory.py.stage1.bak

git apply --check .\dynfl_fixed_hierarchical_stage1.patch
git apply .\dynfl_fixed_hierarchical_stage1.patch
```

若 `git apply --check` 失败，不要强行覆盖本机脚本；请核对本机与此前上传 ZIP 的版本差异。也可以从配套 ZIP 中提取两个源码文件，但要先备份已有脚本。

## 单元测试

```powershell
D:\soft\Python310\python.exe -m unittest discover -s tests -p test_fixed_hierarchical_trajectory.py -v
```

## 与现有 40 客户端实验匹配的验证命令

```powershell
D:\soft\Python310\python.exe `
    .\experiments\validate_trusted_aggregate_trajectory.py `
    --model resnet18_pretrained_head `
    --device cuda `
    --clients 40 `
    --edges 4 `
    --aggregation-topology hierarchical `
    --check-direct-parity `
    --train-limit 4000 `
    --test-limit 500 `
    --rounds 20 `
    --privacy-horizon 20 `
    --local-epochs 3 `
    --lr 0.01 `
    --clip-norms 0.10 `
    --server-steps 0.5 `
    --epsilon 8 `
    --delta 1e-5 `
    --methods trusted_aggregate_dp `
    --seed 42 `
    --output-root out\head_eps8_fixed_4edges_parity_20r
```

查看每轮输出的 `hierarchy_direct_max_abs_error`，确认没有异常中止。原始直聚合实验同条件 Seed=42 在第 20 轮得到 Accuracy=0.370、Loss≈1.7655；分层实验应与之非常接近。两条路径的浮点加法顺序不同，不要求每一位都完全一致；如出现明显差距，先检查聚合权重、客户端映射、每轮模型状态、噪声种子、裁剪顺序。

## 隐私与发布边界（不得省略）

这是固定分组的单机/单进程研究诊断：每个边缘的明文中间和仅在受信环境中计算，**未被单独 DP 保护，也不能向不可信边缘或观察者公开**。 `report.json` 中的 raw norm、clipping bias、clipped fraction 等属于内部诊断指标，并不自动获得 DP 保证；使用私人数据对外发布时需另行审计。多次独立运行公开其模型和指标也不能只记一次 ε=8。

实际部署还需证明：每个客户端的更新在传输阶段受保护；真实边缘与云的可见信息符合威胁模型；HE/SecAgg 的重构/解密边界、失联与串谋处理、噪声生成可信性、跨轮与跨边缘发布等都被隐私记账覆盖。已有 `validate_edge_dp_trajectory.py` 采用的 `edge_local` 或 `distributed_he` 有不同的发布/信任条件，**不得直接用其 ε 或精度与这一步等同**。
