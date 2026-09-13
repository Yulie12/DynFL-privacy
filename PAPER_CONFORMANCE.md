# Paper-to-Code Conformance — v30 Mainline Fusion

The normative specification is `tex/paper/main.tex`. The formal execution revision is
`paper_flow_v30_mainline_fusion_method2`. Earlier packet-level privacy-selection revisions remain
available only for controls/compatibility experiments and are not the v30 paper method.

## Formal mainline contract

1. The paper keeps seven basic collaboration modes, while `LIC` is structurally excluded
   under the honest-but-curious cloud threat model. The dynamic selector therefore searches
   six executable modes.
2. The selector chooses collaboration/resource placement. It does not choose whether the
   released model is protected by DP.
3. Every client contribution in a formal round starts from the previously released global
   model and that client's private data plus the public execution schedule. Same-round
   private aggregate feedback is forbidden in the fused path.
4. Each client contribution is L2 clipped. The formal weighted aggregate uses
   client-replacement sensitivity `2 * C * max_i(w_i)` with the actual public normalized
   release weights.
5. One formal global Gaussian release is accounted per completed FL round. The total noise
   multiplier is calibrated for the fixed round horizon with RDP.
6. Distributed noise shares are introduced before HE cloud aggregation/decryption. Real
   CKKS protects cloud-side confidentiality; it cannot replace DP.
7. Formal runs require full participation, `rdp_auto`, real HE, and full-update encryption
   (`he_aggregation_size=0`). Legacy DP-vs-HE stability substitution is disabled.
8. The supported privacy statement is the aggregate-release mechanism under the stated
   fixed public roster/weight assumptions. The implementation intentionally reports
   `end_to_end_dp_status=not_established` for the complete adaptive protocol.

## Direct implementation map

| Paper element | Implementation |
| --- | --- |
| Seven-mode taxonomy / six executable fused modes | `dynfed/training.py`, `dynfed/selection.py` |
| Dynamic mode/Pareto search | `dynfed/selection.py::choose_global_pareto_profile` |
| Mode-specific logical flow and timing | `dynfed/flow_executor.py` |
| Fixed global release accountant | `dynfed/independent_release.py::IndependentReleaseAccount` |
| Independent fused training base | `dynfed/fmnist_lenet5_dynamic.py::_training_base_state` |
| Client clipping and distributed aggregate noise | fused release branch in `dynfed/fmnist_lenet5_dynamic.py` |
| Real CKKS aggregate path | `dynfed/he_backend.py` and `fedavg_split_seal` path |
| Fused privacy/result-DP validation | `_validate_mainline_fusion` |
| Fixed formal release reporting | `_reported_release_mechanism`, `_privacy_reporting_scope`, `global_release_*` |
| Common aggregate privacy-noise objective | `dynfed/selection.py::_fusion_aggregate_noise_cost` |
| v30 formal configurations | `configs/paper_v30_cifar10_resnet18.json`, `configs/paper_v30_fmnist_lenet5.json` |
| Versioned runner | `experiments/run_paper_config.py`, `experiments/run_fmnist_lenet5.py` |

## Required evidence for a paper run

Each accepted run must contain `summary.json`, `round_metrics.csv`,
`client_decisions.csv`, `flow_events.csv`, and the reproducibility artifacts emitted by the
training runner. For v30, the summary/round logs must expose at least:

- `mainline_fusion=true`;
- `global_release_count` and `global_release_epsilon`;
- fixed global release contract metadata;
- aggregate sensitivity/noise metrics;
- real HE execution metrics and numerical aggregation error;
- mode distribution, switching, communication, modeled system time, selection time, and
  host wall time;
- `end_to_end_dp_status=not_established`.

## Baseline boundary

Protected methods such as Ours, Individual Optimal, Random, Fixed FedAvg, Fixed SplitFed,
Fixed HFL, and NSGA-II can share the same fused release contract. `no_protection`, pure HE,
and legacy packet-DP policies are controls and must be run under a separate configuration;
they are intentionally rejected inside a formal fused run.

## Work still required before submission

1. Run the complete regression suite and a fresh real-SEAL smoke on the merged v30 path.
2. Freeze the final dataset/model/DP hyperparameters used for the paper tables.
3. Run the main comparison and key ablations with the declared multi-seed protocol.
4. Rebuild every table/figure from v30 outputs; do not reuse pre-v30 numerical results.
5. Report system-profile provenance and keep the single-host/logical-placement limitation
   explicit.
6. Keep the aggregate-release privacy claim narrower than a full adaptive-protocol DP claim
   unless a separate end-to-end proof is added.

## 本次 Method 2 正式接入（2026-09-13）

- CIFAR 正式入口已固定为 `resnet18_pretrained_head`、5130 个可训练参数、
  `clip_norm=0.1`；100-round horizon 和每阶段 3 local epochs 保持不变。
- LIEIIIC/LIIEIIIC 连续执行 3 个客户端私有阶段（共 9 epochs），仅延续该
  客户端自己的模型和优化器状态。最后才裁剪并加入分布式噪声；一轮一次发布。
- 复用 `experiments/trusted_edge_custodian.py`：按固定公开权重形成加噪边缘包，
  独立 cloud worker 只处理密文，custodian 校验完整和后解密。每轮新会话/密钥；
  FL 总周期及续跑由持久化 global release account 约束。
- 正式加密载荷为完整可训练更新，而非填充冻结骨干的零更新。
- 七种基础模式定义保留；LIC 沿用当前受信域的执行限制。本次没有新增云端
  中间表示保护，也没有放宽信任模型。
- 选择器与训练共用 fused release flow：所有参与者都进入一次边缘加噪聚合和
  一次全局发布；重复阶段不再虚构重复跨客户端聚合反馈。通信统计对共享 E-C
  上传和 C-E 返回按边缘域去重。
- `release_calibration=global_release`；短跑 summary 也记录真实累计 epsilon，
  不再把 legacy packet ledger 的零消耗当成全局发布消耗。
- 新 revision 为 `paper_flow_v30_mainline_fusion_method2`，禁止接续旧融合语义
  的 checkpoint。旧补丁文件仅对应之前版本，本次没有重新生成那个补丁。

边界：这是机制与执行语义接入，不是收敛优势证明。原完整模型 HE profile
不能充当 5130 参数载荷的实测成本；正式系统性能结论仍需对应载荷与计算范围
的 profile。Fashion-MNIST/LeNet 仍是独立模型配置，不宣称已有同样的预训练
分类头证据。论文编译目前被本机 MiKTeX 初始化目录写入权限阻塞。

### 本次验证证据

- 全量回归 324 passed；日志字段调整后的 focused regression 27 passed。
- 最终 revision 的 Ours：100 客户端、10 边缘、真实 SEAL 完成 1 round，
  100-round 隐私 horizon 保留，release_count=1，epsilon=0.70478798，
  HE 最大绝对误差约 2.30e-8。结果：
  `out/method2_formal_final_check/2026-09-13_21-41-29_lenet5_dynamic_newtex202608/ours/partial_summary.json`。
- 同次 fixed_hfl 未完成训练：17 个客户端没有 device-feasible 固定候选。
  完整 roster 门控拒绝该基线；这不是允许通过删除客户端解决的错误。
  保留场景并记为不可行，或另设全部方法共同可行的场景，等待用户选择。
- Ours 的单轮测试精度为 7.1%，仅用于接入验证，不构成长程效用结论。
- TeX begin/end 环境配对检查通过；PDF 编译尚未通过环境初始化门槛。
