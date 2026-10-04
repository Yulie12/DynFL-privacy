# DynFL 原主线 head256 诊断补丁

该补丁以用户之前已应用的 `--server-step` 主线补丁为基础，只增加独立模型名 `resnet18_pretrained_head256`。原来的 `resnet18_pretrained_head` 保持不变。

- 预训练 ResNet-18 冻结部分、云-边-端分割点、七种模式、Pareto 动态选择、原始模式相关 DP/HE/SecAgg 逻辑保持不变。
- 在 ResNet-18 平均池化后的 **512 维** 特征上，使用无参数、与数据无关的固定投影 `x[..., ::2]`（选取偶数位置的 256 个通道）。端—边的中间激活维度仍按原模型传输，**仅分类头的可训练 DP 更新维度**从 5,130 降至 2,570。
- 新模型只用于独立诊断；不与旧 5,130 维检查点互相 resume。
- 训练前请在原项目根目录备份三个修改文件，并将 ZIP 中相同路径的文件复制覆盖；新增测试文件和 config 可直接复制。不要用 ZIP 覆盖其他项目文件。

## PowerShell：诊断命令（先 dry-run）

```powershell
D:\soft\Python310\python.exe `
    .\experiments\run_paper_config.py `
    --config .\configs\paper_v30_cifar10_resnet18_head256.json `
    --seeds 42 `
    --policies full_dynfl `
    --rounds 20 `
    --he-execution profiled `
    --server-step 0.5 `
    --max-new-rounds 10 `
    --output-root out\dynfl_h20_head256_probe `
    --dry-run
```

展开参数应包括 `--model resnet18_pretrained_head256` 和 `--rounds 20`。删除 `--dry-run` 再进行独立的新实验。

## 必须一并检查

1. `trainable_parameters=2570`（仅分类头）、DP 更新维度=2570；全模型参数约 1,118 万 **不代表 DP 更新维度**。
2. 裁剪率、信号范数、噪声范数、有效噪声/信号；相同 ε 并不意味着降维后的精度必然更好。
3. 对比已有 5,130 维基线 R10 Accuracy=24.75%、Loss=2.0855；如改善，再补多种子，以及相同新模型的禁用 Update DP 诊断以区分表达能力和 DP 噪声效应。
4. `profiled` HE 实际使用明文聚合、估计 HE 成本，不是实加密，且原项目中的 `end_to_end_dp=not_established` 尚未解决。正式结果需恢复真实 HE 和协议审计。
5. 固定投影并非通用的降维最优投影；它是一组可复现的、无需使用私人数据的维度消融。
