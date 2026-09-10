# 分类头更新 DP 诊断

## 目的与边界

新增模型标识 `resnet18_pretrained_head`，冻结公开预训练骨干及 BatchNorm 状态，仅训练分类头。CIFAR10 可训练参数为 5130，原模型训练方式和历史结果保持不变。

该诊断沿用现有逐包客户端级 DP、整包裁剪、100 轮会计期限、epsilon 8、delta 1e-5、裁剪范数 1 和学习率 0.01。不是旧的一次性样本级特征发布诊断，也没有改为给 embedding 加噪。

配置 `configs/diagnostic_v29_cifar10_head.json` 关闭稳定性过滤，目的是直接观测 DP 效用。它不是按当前 TeX 稳定性规则运行的正式配置。不同策略仍可选择不同模式，不能将策略间的精度差全部归因于密码算法。模型传输仍保留原有状态格式，不能把降维诊断中的 HE 时间当作仅加密分类头的性能。

## 复现短验证

```powershell
D:\soft\Python310\python.exe experiments\validate_privacy_paths.py --config configs\diagnostic_v29_cifar10_head.json --clients 4 --edges 2 --train-limit 400 --test-limit 100 --max-new-rounds 5 --policies no_protection fixed_dp fixed_he ours --output-root out\v29_head_privacy_smoke
```

只新增运行 5 轮，隐私校准仍使用 100 轮。小规模、单种子结果仅用于定位故障，不是论文性能结论，也不证明所有 100 客户端实验都会得到相同表现。

## 已核对项目

分类头模式沿用原预训练模型的优化器与批大小。测试覆盖完整训练和 Split 训练路径，确认只有分类头参数进入更新及扰动。实际输出配置同时记录 `trainable_parameter_count=5130` 与 `omega_update_dimension=5130`，选择器的扰动维度与训练一致。

全套测试 153 项通过。真实训练结果位于 `out/v29_head_privacy_smoke/2026-09-10_13-24-26_privacy_paths`。

## 完成结果

无保护方法第 5 轮精度 32%，损失 1.818。固定 DP 第 5 轮精度 10%，损失 508.159，噪声仍远大于有效更新。降低维度缓解了噪声能量，但当前设置尚未恢复 DP 可训练性。

真实 HE 第 5 轮精度 38%，损失 1.674。关闭过滤的 Ours 第 5 轮精度 14%，损失 417.859。Ours 前三轮同时选择了 DP 与 HE，后两轮选择 DP，说明动态路径确实执行，但不代表选择质量或训练效用达标。四组进程均正常结束，DP 与 Ours 被报告为训练警告。

| 方法 | 第 5 轮精度 | 第 5 轮损失 | 累计实际耗时 |
| --- | --- | --- | --- |
| No Protection | 32% | 1.818 | 7.10 秒 |
| Fixed DP | 10% | 508.159 | 15.33 秒 |
| Fixed HE | 38% | 1.674 | 153.44 秒 |
| Ours，关闭过滤 | 14% | 417.859 | 49.56 秒 |

后续应核对只发布安全聚合结果所需的敏感度及状态依赖条件，而不是自行缩小噪声、增加预算或者改成样本级保护。HE 运行成功不代表最终发布模型已建立 DP 保证。
