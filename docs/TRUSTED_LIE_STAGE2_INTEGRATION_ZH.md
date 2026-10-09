# Trusted LIE 联合 Sample-DP-SGD：第二阶段增量接入（仅原型）

## 修改范围

在第一阶段 `trusted_lie_joint_dp_v1.patch` **已应用**的前提下，本补丁仅更新：
- `dynfed/selection.py`：新增默认关闭的 `SelectionConfig.trusted_lie_joint_sample_dp`，在 LIE 模式按 `(embedding=0, label-gradient=0, joint-optimizer=K)` 预测隐私事件；其余模式仍走旧路径。
- `dynfed/fmnist_lenet5_dynamic.py`：把显式可信标志、保护机制传入真实客户端 worker，并在派遣前核对候选事件、机制和 DataLoader batch 大小。既有分发计费和实际事件一致性检查不变。
- `dynfed/trusted_split_sample_dp.py`：不再把未加保护的裁剪样本数和最大梯度范数输出到 worker 诊断，使用 `trusted_split_sensitive_gradient_stats_redacted=1` 明示这些数字已屏蔽、非实测零值。
- `tests/test_trusted_lie_stage2_integration.py`：新增 opt-in、安全拒绝、账本、worker、事件一致性测试。

## 显式实验前提

```python
from dataclasses import replace
selection = replace(selection,
    privacy_unit="sample",
    trusted_edge_split_execution=True,
    trusted_lie_joint_sample_dp=True,
    # 配合真实训练 batch；CPU/Lenet5 目前使用 64，CUDA/ResNet18 特定模型使用 128。
    split_batch_size=128,
    learning_objective="legacy_fusion_dp",
)
```

以上仅演示**程序化配置**，不修改仓库实验配置；这一步尚未新增 CLI 参数，不能直接拿旧正式运行命令做可信 LIE 实验。

`trusted_edge_split_execution=True` 是部署者对适用 Edge 计算域作出的**外部信任承诺**，代码本身没有身份认证、隔离部署、TLS/TEE 或远程证明。它原有逻辑会排除 `LIC`，本补丁不改变该行为。对于某客户端缺乏受信 Edge 的情况，不应启用此原型。

## 隐私模型与边界

可信域内的 `embedding`、标签相关反向梯度可以明文处理；**它们不是 DP 输出**。LIE 每个执行 batch 使用一次 End+Edge 联合逐样本裁剪、聚合、加噪。`SamplePrivacyLedger` 仅对实际 optimizer 步数计费；已存在的 dispatched worker 审计会对比预收账本与执行计数。

此版本仍不构成端到端 Sample-DP 证明：动态模式选择对私人信息的依赖、初始化/缓存与跨轮组合、模型及元数据的所有发布、资源侧信道、真实异机可信域、以及具体 batch/多阶段覆盖仍需单独审计。

旧版 Joint Proxy 校准文件假设带噪 embedding，不适合直接为当前 LIE 作出学习效用评估；因此配置 `learning_objective="joint_calibration"` 与本开关组合会**立即拒绝运行**，而不是默默使用过期代理。未来重新校准前请勿开展论文正式实验。资源不足优先 Split 的 admission 策略及其他六模式的可信路径也尚未在本补丁中实现。

## 安全回归命令

```bash
python -m pytest -q \
  tests/test_trusted_lie_stage2_integration.py \
  tests/test_trusted_lie_joint_dp.py \
  tests/test_sample_dp.py \
  tests/test_independent_execution.py \
  tests/test_selection.py \
  tests/test_resource_selection.py \
  tests/test_privacy_accounting.py \
  tests/test_dynfl_sample_hierarchical_preflight.py \
  tests/test_dynfl_v364_stage_event_audit.py \
  tests/test_dynfl_v365_predispatch_sample_accounting.py \
  tests/test_sample_dp_privacy_path_audit_contract.py
```

不要将单进程 Trusted LIE 原型误当成已交付的分布式可信训练或完整的隐私证明。
