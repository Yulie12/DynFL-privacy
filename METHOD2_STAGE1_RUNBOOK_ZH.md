# 方法2第一阶段运行说明

## 已连通范围

统一配置入口、公开资源选择器、独立完整/受信边缘拆分训练、固定参与客户端、真实SEAL、受信edge0解密和单次全局DP发布。每轮公开资源剖面重估并选择，执行日志记录实际模式及决策时间。没有accuracy目标、私有验证集选择、探索奖励或DP/HE替换。

新策略名为resource_placement，不是原论文完整Pareto方法。旧v29配置不变。网页和断点续跑尚未接入，CLI第一阶段可运行，但不能认定全部论文实验准备完毕。

## 命令

短程验证保留100轮会计周期。

```powershell
D:\soft\Python310\python.exe experiments\run_paper_config.py --config configs\method2_resource_placement_cifar10.json --seeds 42 --max-new-rounds 5
```

100轮运行不加短程限制。每次从初始模型开始，不能续接之前的5轮。

```powershell
D:\soft\Python310\python.exe experiments\run_paper_config.py --config configs\method2_resource_placement_cifar10.json --seeds 42
```

## 选择目标与边界

目标为归一化可变阶段时延，加0.1倍归一化客户端到边缘通信量，再加0.1倍归一化计算时间总量。基准为同轮全部客户端本地计算，避免不同量纲直接相加。无精度项。边缘计算服务量受配置中的容量上限限制；客户端算力异质性反映在计算时间中，尚未建立内存、能耗等完整约束。

使用两初始方案和有界单客户端替换搜索，不是精确最优解或原Pareto算法。隐私检查先于选择，固定全部参与者，按公共样本配额校准；私有数据、loss、原始更新范数均不进入选择器。

`configs/method2_resource_profile.json` 明确标注分析假设，不是硬件实测。计算系数按样本和本地轮数缩放，网络带宽按公开随机资源轨迹变化。拆分传输体积使用模型接口维度，通信是分析值，不是真实网络抓包量。边缘串行服务与客户端并行到达用保守阶段时延模型表示，不是实际单机墙钟。

两条位置共享的HE、发布模型返回等开销不用于排序。记录的modeled_variable_phase_sec及modeled_client_edge_mb不是完整系统时延/总通信量，不能直接写进论文总开销表。真实HE字节及时间、训练、决策、评估墙钟分项在日志中另列。

## 首次验证

结果位于 `out/method2_resource_system_smoke/2026-09-13_14-21-21_edge_dp_trajectory/report.json`。100客户端5轮均通过资源和隐私检查，每轮真实加密全部51300个参数值。精度轨迹8.5%、11.5%、10.15%、10.05%、11.3%，累计epsilon约1.605，未放宽100轮预算。

当前剖面均选择LIIC，本地路径胜出不视为错误，不强制多样性。决策约16到22毫秒，不是训练总耗时。两种路径实际混合训练的独立回归已有记录。要评价动态调度优势，还需真实剖面校准、有代表性的资源场景和同条件固定位置对照，不能为获得优势任意改分析系数。
