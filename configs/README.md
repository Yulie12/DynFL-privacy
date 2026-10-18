# DynFL 配置分类（当前有效版本）

顶层 `configs/*.json` 只保留当前正式 Online-J 配置与固定信任清单。
legacy fusion、查表式 joint calibration、oracle 诊断及其历史归档配置
已从当前方法中移除，不提供任何入口。

## 正式论文实验（`learning.objective = online_upper_bound`）

| 配置 | 用途 | 所需事前证书 |
|---|---|---|
| `paper_v30_cifar10_resnet18.json` | CIFAR-10 / ResNet-18 主实验（100 clients, 10 edges, 3 seeds） | `certificates/REAL_cifar10_resnet18_predecision_required.json` |
| `paper_v31_cifar10_resnet18_fast_response.json` | 同一主实验的 fast-response 研究（clients 0–19 硬截止 20 s） | `certificates/REAL_cifar10_resnet18_predecision_required.json` |
| `paper_v30_fmnist_lenet5.json` | Fashion-MNIST / LeNet-5 辅助实验 | `certificates/REAL_fmnist_lenet5_predecision_required.json` |
| `stage11_cifar10_resnet18_layer4_online_5r_STAGING.json` | CIFAR-10 layer4 online-J 5 轮模板 | `certificates/REAL_cifar10_resnet18_layer4_5r_predecision_required.json` |
| `stage11_fmnist_lenet5_online_5r_STAGING.json` | Fashion-MNIST online-J 5 轮模板 | `certificates/REAL_fmnist_lenet5_5r_predecision_required.json` |

所有正式配置只接受 `online_upper_bound`，并要求显式
`learning.online_bound_manifest_path`。`certificates/` 下的真实事前证书
当前**不存在**。缺证书、证书轮次不匹配、候选身份不匹配或任一必要假设
未证明时，运行必须在任何客户端训练前失败关闭；不允许回退到
`legacy_fusion_dp`、`joint_calibration` 或 oracle 诊断，也不允许用
合成证书冒充真实证书。

`system.mainline_fusion` 字段仅为拒绝旧 Method2 覆盖层而保留；
将其设为 `true` 会被配置校验拒绝。

## 固定信任清单（不是可独立运行的实验配置）

以下文件由 `system.static_pair_admission_manifest` 引用，只固定
Client→Edge 配对：

- `v370_paper_trust20.json`
- `v370_paper_trust20_2edges.json`
- `v370_paper_trust40.json`
- `v370_layer4_trust40of40_smoke.json`
- `v370_smoke_manifest.json`

## 校验方式

```bash
python experiments/run_paper_config.py --config configs/paper_v30_cifar10_resnet18.json --dry-run
```

`--dry-run` 只做配置展开与校验；即使通过，正式训练仍需真实事前证书。
若证书缺失，实际运行会在训练前报错退出，这是预期行为。