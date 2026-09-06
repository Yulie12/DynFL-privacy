# V25 实验运行手册

所有命令均在 `E:\YTT\GROUP\DynFL-privacy` 下运行。V25 的执行标识为
`paper_flow_v25_trusted_edge_domain`。V24 及更早版本的结果和检查点不能用于
V25 论文图表。

V25 采用以下保护边界。

- 每个客户端与其关联边缘组成可信执行域
- Split 模式的 embedding 和反向梯度只在该可信域内传输
- 跨域进入云端的模型更新使用 Update DP 或 CKKS
- Feature DP 在正式配置中停用

同一时间只运行一个使用 GPU 的训练进程。

## 1 运行前检查

```powershell
D:\soft\Python310\python.exe -m pytest -q
D:\soft\Python310\python.exe experiments\run_paper_config.py --config configs\paper_v25_cifar10_resnet18.json --seeds 42 --policies ours --max-new-rounds 1 --dry-run
D:\soft\Python310\python.exe experiments\run_controlled_sweeps.py --config configs\paper_v25_cifar10_resnet18.json --studies privacy --seeds 42 --rounds 2 --dry-run
```

检查打印命令中存在以下参数。

- `paper_flow_v25_trusted_edge_domain`
- `--trusted-edge-split-execution`
- `--dp-update-epsilon-budget 8.0`
- `--he-backend seal`
- `--he-execution profiled`

正式 v25 命令中不应再出现 `--dp-feature-epsilon-budget`。

## 2 八种方法冒烟测试

先让八种方法各运行两轮。该结果只用于检查，不写入论文。

```powershell
D:\soft\Python310\python.exe experiments\run_paper_config.py `
  --config configs\paper_v25_cifar10_resnet18.json `
  --seeds 42 `
  --policies ours individual_optimal no_protection random fixed_fedavg fixed_splitfed fixed_hfl nsga2 `
  --max-new-rounds 2 `
  --output-root out\paper_v25_all_methods_smoke
```

每种方法应产生 `partial_summary.json` 和两行 `round_metrics.csv`。确认精度为
有限数值，`dp_feature_enabled` 为 `false`，`dp_feature_horizon_events` 为 `0`，
`trusted_edge_split_execution` 为 `true`。

## 3 真实 CKKS 验证

主实验采用实测 CKKS 剖面以避免每轮重复完整加密。下面的独立实验执行真实
SEAL 加密、密文加法和最终解密，用于证明 HE 路径真实可运行。

```powershell
D:\soft\Python310\python.exe experiments\run_paper_config.py `
  --config configs\paper_v25_cifar10_resnet18.json `
  --seeds 42 `
  --policies ours `
  --max-new-rounds 1 `
  --he-execution real `
  --output-root out\paper_v25_cifar10_resnet18_real_he_validation
```

结果中 `he_process_fallbacks` 必须为 `0`，并记录 `he_wall_time_sec`、
`he_ciphertext_bytes` 和 `he_max_abs_error`。

## 4 CIFAR 10 与 ResNet 18 主实验

五个随机种子分别运行。先完成一个种子的八种方法并检查结果，再开始下一个。

```powershell
D:\soft\Python310\python.exe experiments\run_paper_config.py `
  --config configs\paper_v25_cifar10_resnet18.json `
  --seeds 40
```

将 `40` 依次替换为 `41`、`42`、`43` 和 `44`。每个种子完成后确认八种方法
均存在 `summary.json`，每个文件中的 `rounds` 为 `100`，执行标识为 v25。

五个种子全部完成后生成统计表和论文图片。

```powershell
D:\soft\Python310\python.exe experiments\aggregate_multiseed_results.py `
  --root out\paper_v25_cifar10_resnet18 `
  --seeds 40 41 42 43 44 `
  --output-dir out\paper_v25_cifar10_resnet18_aggregate `
  --paper-figure tex\paper\figures\cifar10_resnet18_100r_wall_time_accuracy_v25.png `
  --paper-reconfiguration-figure tex\paper\figures\cifar10_reconfiguration_trace_v25.png
```

重点检查 `summary_statistics.csv`、`paired_comparisons.csv`、
`time_statistics.csv`、`reconfiguration_statistics.csv` 和
`reconfiguration_summary.csv`。

## 5 受控实验

各组实验分开运行，避免一次启动过多长任务。

### 5.1 策略更新周期

```powershell
D:\soft\Python310\python.exe experiments\run_controlled_sweeps.py `
  --config configs\paper_v25_cifar10_resnet18.json `
  --studies period
```

比较周期 `1`、`5`、`10` 和 `20`。

### 5.2 Update DP 隐私预算

```powershell
D:\soft\Python310\python.exe experiments\run_controlled_sweeps.py `
  --config configs\paper_v25_cifar10_resnet18.json `
  --studies privacy
```

比较总 Update DP 目标 `1`、`2`、`4` 和 `8`。该实验不会改变 Feature DP
参数，因为 V25 不执行 Feature DP。

### 5.3 客户端规模

```powershell
D:\soft\Python310\python.exe experiments\run_controlled_sweeps.py `
  --config configs\paper_v25_cifar10_resnet18.json `
  --studies scale
```

比较 `20`、`50` 和 `100` 个客户端下 Ours 与 Individual Optimal 的决策时间
和系统时间。

### 5.4 消融实验

```powershell
D:\soft\Python310\python.exe experiments\run_controlled_sweeps.py `
  --config configs\paper_v25_cifar10_resnet18.json `
  --studies ablation
```

比较完整方法、无全局协调、无收敛误差代价估计和固定 LIIEIIIC。

全部受控实验完成后运行聚合。

```powershell
D:\soft\Python310\python.exe experiments\aggregate_controlled_results.py `
  --root out\paper_v25_controlled `
  --output-dir out\paper_v25_controlled_aggregate
```

## 6 精确 Pareto 对照

```powershell
D:\soft\Python310\python.exe experiments\validate_pareto_search.py `
  --output-dir out\pareto_validation_v25
```

该实验只在小规模实例上比较精确枚举与有界搜索，不训练 ResNet 18。

## 7 收敛误差代价校准

主实验完成后直接使用逐轮日志，无需重新训练。

```powershell
D:\soft\Python310\python.exe experiments\calibrate_convergence_cost.py `
  --root out\paper_v25_cifar10_resnet18 `
  --seeds 40 41 42 43 44 `
  --output-dir out\paper_v25_convergence_calibration
```

## 8 Fashion MNIST 与 LeNet 5

依次对种子 `40`、`41`、`42`、`43` 和 `44` 运行下面命令。

```powershell
D:\soft\Python310\python.exe experiments\run_paper_config.py `
  --config configs\paper_v25_fmnist_lenet5.json `
  --seeds 40
```

完成后聚合。

```powershell
D:\soft\Python310\python.exe experiments\aggregate_multiseed_results.py `
  --root out\paper_v25_fmnist_lenet5 `
  --seeds 40 41 42 43 44 `
  --dataset fmnist `
  --model lenet5 `
  --output-dir out\paper_v25_fmnist_lenet5_aggregate `
  --paper-figure tex\paper\figures\fmnist_lenet5_100r_wall_time_accuracy_v25.png
```

## 9 中断恢复

恢复时只指定一个种子和需要恢复的方法。

```powershell
D:\soft\Python310\python.exe experiments\run_paper_config.py `
  --config configs\paper_v25_cifar10_resnet18.json `
  --seeds 42 `
  --policies ours `
  --resume-from-run out\paper_v25_cifar10_resnet18\EXISTING_RUN_DIRECTORY
```

恢复目录中必须存在该方法的 `checkpoint.pt`、`round_metrics.csv` 和顶层
`config.json`。不得用 v24 或更早版本的检查点恢复 v25。

## 10 最终核对

1. 所有论文数值均来自 v25 完整运行，不使用 `partial_summary_table.csv`
2. 主实验与消融表使用五个种子的均值和 95% 置信区间
3. 曲线使用原始逐轮精度，不使用 EMA 平滑
4. 效率主指标使用系统模型时间加实测决策时间
5. 主实验只声明 CKKS 剖面执行，真实 CKKS 证据来自独立验证
6. `max_feature_epsilon` 恒为零，论文只报告 `epsilon_upd`
7. 图名、表格和正文均标识 v25 结果
8. TeXStudio 按 PdfLaTeX、BibTeX、PdfLaTeX、PdfLaTeX 顺序编译
9. 编译日志中不存在未定义引用、缺失图片或 `Overfull hbox`
