# Trusted LIE 联合 Sample-DP-SGD v1（仅独立验证，未接入正式调度）

## 目标与约束

- 面向资源受限客户端，把 End 与 Edge 分置计算视为同一个**事先认可的可信域**。受信域内部的激活、logits 与标签相关梯度不做逐样本 embedding/label-gradient DP 加噪。
- 对一个 minibatch 的 **End + Edge 全部可训练参数**，为每个样本计算联合梯度，按**同一个全局 L2 上限**裁剪，向裁剪梯度和加一份高斯噪声，再除以 minibatch size，分别应用给两个优化器。
- 使用替换邻接时，求和梯度敏感度为 `2*C`，噪声坐标标准差为 `2*C*sample_optimizer_noise_multiplier`。这只是**单步机制**；本补丁不提供训练全过程 (ε,δ) 会计或端到端 DP 证明。
- 只有显式开启 `trusted_split_joint_sample_dp=True`、`trusted_edge=True`，并使用 `LIE`、`privacy_unit='sample'`、`mechanisms={'emb':'trusted','grad':'trusted'}` 且有限正值的 `sample_optimizer_noise_multiplier` 才能进入新路径，否则拒绝。
- 可信条件是调用方**声明并负责验证**的外部部署前提。设置 `trusted_edge=True` 并不实现远程认证、可信执行环境或安全网络传输。

## 范围与不支持项

这是**单进程参考训练分支**。虽保持 End/Edge 模型和参数独立，内部前向 `edge(end(x))` 在同一进程完成计算图，不实现真正远端 GPU 上的拆分 RPC、加密链路或 End/Edge 异机验证。因此不能据此宣称已实现真实云边端安全通信。

- **尚未接入**七模式动态资源选择器、Cloud 信任模型、正式隐私账本、生产级发布门禁和跨客户端模型聚合；其他 6 个模式不变。
- 现有 `SamplePrivacyLedger` 按 embedding/label-gradient 链路发布收费，其事件不能直接用于这条新路径。新的联合 DP 优化器事件需要另外核算组合，且不能将内部诊断指标无保护地发布到不可信节点。
- DP-SGD 的训练迭代/客户端采样、跨轮重用及公开选择策略都需要与实际会计对齐。**不能把此补丁的单步噪声机制称为全系统 Sample-DP**。
- 只支持 BatchNorm 处于 eval 模式的网络；训练态 BatchNorm 被明确拒绝，以防批间耦合破坏微批逐样本梯度的假设。
- 当前使用 per-sample 微批循环，是正确性参考，不是面向大规模 ResNet 的吞吐优化实现。

## 非破坏性验证

```bash
cd /root/autodl-tmp/DynFL-privacy
python -m pytest -q tests/test_trusted_lie_joint_dp.py tests/test_sample_dp.py tests/test_independent_execution.py
```

正常结果为 38 tests passed（依赖版本不同可能略有变化）。源代码和配置之外不产生需要保留的实验结论。

## 训练入口（尚非正式运行入口）

原 `split_local_train_lenet5(...)` 新增两个尾部关键字参数，默认均为 `False`，现有调用不变：

```python
split_local_train_lenet5(
    "LIE", global_end_state, global_edge_state, x, y,
    epochs, lr, device, model_name, input_shape, num_classes,
    mechanisms={"emb": "trusted", "grad": "trusted"},
    privacy_unit="sample",
    trusted_split_joint_sample_dp=True,
    trusted_edge=True,
    sample_optimizer_clip_norm=1.0,
    sample_optimizer_noise_multiplier=0.75,  # 示例测试值，非正式预算校准
    training_diagnostics=diagnostics,
)
```

会在 `training_diagnostics` 中记录 `trusted_split_joint_dp_steps`、`sample_dp_optimizer_steps` 和样本裁剪诊断；embedding/label-gradient 的 DP 发布事件为零。返回 End/Edge 差分与旧入口形式兼容，**但在正式会计和对外发布策略建立以前，不应送入现有未经改造的云端聚合流程。**
