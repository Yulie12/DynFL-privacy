# v30 正式实验准备度

当前主线已经从“独立 Method 2 诊断路径”推进到
`paper_flow_v30_mainline_fusion_method2`：动态协作 mode/Pareto 选择和固定的 Method 2
全局发布契约在同一个正式训练路径中对齐。后续不应再把旧 per-link DP/HE
机制选择当作论文主方法。

## 已闭合的主线语义

- 7 个 basic modes 保留；`LIC` 因 threat model 排除，实际动态候选为 6 个。
- selector 只动态选择协作/资源路径，不动态关闭结果 DP。
- 所有 fused mode 的 client contribution 从上一轮已发布 global model 独立产生。
- per-client clipping、真实权重敏感度 `2*C*max(w_i)`、distributed noise、real HE
  cloud aggregation、trusted custodian decrypt 形成统一 release gate。
- 一轮一个 formal release/accounting event；local epochs 不增加 release count。
- 正式融合要求 full participation、`rdp_auto`、real HE、full-update encryption。
- 旧 DP-vs-HE stability switching 在 formal fusion 中禁用。
- `no_protection`、pure HE、旧 packet-DP 作为单独 control/compatibility 路径，不能
  混入 v30 formal policy set。
- 报告继续保留 `end_to_end_dp_status=not_established`，不把 aggregate-release
  accounting 夸大成完整自适应协议的端到端 DP。

## 仍需完成的投稿前门槛

1. **全量回归**：必须 fresh `pytest -q` 全绿，不能只依赖 focused tests。
2. **真实 HE smoke**：使用正式 v30 配置完成短程 real-SEAL 运行，核对
   `global_release_count`、epsilon、noise std、HE numerical error 和 mode logs。
3. **实验参数冻结**：CIFAR-10/ResNet-18 的模型 scope、clip norm、local epochs、
   partition、epsilon/delta、round horizon 必须在正式批量实验前一次性冻结。
4. **多 seed 主实验**：protected dynamic/static policies 使用同一 v30 release
   contract；No Protection 等 control 使用单独匹配配置。
5. **图表重建**：所有主表、收敛图、reconfiguration 图、系统成本图都只使用
   v30 输出，旧 revision 结果不能混用。
6. **论文边界**：明确资源/链路 profile 是逻辑/分析场景、训练仍在单宿主机执行，
   不宣称物理 cloud-edge-end 部署隔离已经验证。

## 推荐的下一批实验顺序

先做 5-round real-HE smoke，然后做 20-round semantic validation，最后才开启
100-round multi-seed。正式主实验至少比较 Ours、Individual Optimal、Random、
Fixed FedAvg、Fixed SplitFed、Fixed HFL、NSGA-II；No Protection 和 privacy
ablations 单独运行并在论文中明确它们不属于 mandatory-release candidate set。
