# 隐私路径短程验证

本流程检查程序运行、真实 HE 运算和 DP 扰动的实际影响，不把两轮运行成功当作训练收敛或隐私证明。验证日志含未保护的诊断量，仅供内部排查。

## 已修复

- 客户端更新裁剪使用 float64 累加范数，避免 float32 平方溢出后把有效更新错误清零。
- NaN、Inf 更新不能进入裁剪流程。逐包 DP 的比例、裁剪阈值和噪声倍率必须有效，非法值显式报错。
- 逐轮打印真实墙钟时间与模型时延，分别标为 wall_time 和 logical_time。
- 打印损失、有效更新范数、实际噪声范数、噪声与更新之比、裁剪比例、DP 发布数和真实 HE 运算耗时。
- 记录稳定性过滤前后的候选总数以及更新保护机制分布，用来解释 Ours 为什么没有选择某种机制。
- 检测到非有限损失或更新时保存失败日志并终止该方法，不覆盖上一轮有效检查点。
- 验证脚本隔离运行各方法，单个方法失败后继续检查其他方法，最后返回非零退出码并保留错误输出。

## 小规模检查命令

在仓库目录使用下面的一行命令，不需要 PowerShell 反引号续行。

```powershell
D:\soft\Python310\python.exe experiments\validate_privacy_paths.py --config configs\paper_v28_cifar10_resnet18.json --clients 10 --edges 2 --train-limit 1200 --test-limit 200 --max-new-rounds 2 --output-root out\privacy_paths_real_smoke
```

这是 CIFAR-10 与 ResNet-18 的缩小规模诊断，不是论文结果。正式配置文件不会被修改。预算校准仍使用配置中的 100 轮，不会由于仅执行两轮而降低噪声。

默认分别检查 no_protection、fixed_dp、fixed_he、fixed_dp_he 和 ours，使用真实 SEAL、完整更新加密和一个 HE 工作进程。可以用 `--policies` 指定其中几种方法。

在原实验规模下检查时，去掉 clients、edges、train-limit、test-limit 四个覆盖参数即可，执行时间会明显增加。不要在短程验证仍显示失效时启动全部随机种子实验。

## 查看结果

控制台会打印本次独立输出目录。目录内保存以下文件。

| 文件 | 内容 |
| --- | --- |
| effective_config.json | 实际实验参数，包含明确指定的规模覆盖 |
| validation_report.json | 各方法的运行状态、精度、损失、噪声比、DP 事件、真实 HE 轮数与耗时 |
| validation_summary.csv | 与 JSON 对应的单表汇总，可直接用表格软件检查 |
| 方法目录中的 command.json | 该方法实际执行的完整命令 |
| 方法目录中的 console.log | 标准输出和错误输出，包括原始异常 |
| 子运行目录中的 round_metrics.csv | 全部逐轮诊断量 |
| 子运行目录中的 protection_releases.csv | 每个实际云端更新包的来源、机制、加噪位置、HE 执行与保护规则审计 |
| 子运行目录中的 checkpoint.pt | 由训练器按原有目录结构保存的检查点，非有限轮次不覆盖有效检查点 |

`short_run_finished` 表示执行完设定轮次且末轮数值有限、噪声没有超过更新。`completed_with_training_warning` 表示程序完成但末轮噪声已经主导更新，不能据此称训练正常。

`finite` 表示已检查数值有限，不保证精度提高。`noise_dominates_update` 表示实测噪声范数大于聚合前更新范数，不是严格的不可训练阈值，也不会因此自动删除候选模式。`non_finite` 表示数值失效。

`he_exec=real` 才表示实际执行 HE。`profiled` 仍是明文聚合加开销模型。真实 HE 数值精度由已有 `he_max_abs_error` 字段记录。

`update_eps=0` 表示当前记录中没有 DP 消耗，不表示零隐私泄漏。`end_to_end_dp=not_established` 明确表示完整可见记录的 DP 保证尚未建立。

## 尚未完成

本次没有降低噪声、改变预算或强制 Ours 选择 DP，也没有修改受信域和训练拓扑。共享边缘状态影响多个更新时的敏感度、选择器与实际噪声代价的一致性，以及统一保护目标下的候选集合，仍须继续处理。

关于按位置配置保护，以及预算不足后是否能切换为 HE，见 [保护位置与规则](PROTECTION_LAYOUT_DESIGN_ZH.md)。

不同方法可能选择不同协作拓扑，因此它们的精度差不能直接全部归因于 DP 或 HE。现有加密聚合单元测试使用相同输入更新比较明文与 CKKS，适合验证加密的数值误差。
