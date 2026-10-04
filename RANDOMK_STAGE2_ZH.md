# DynFL Random-k Stage 2 独立对照（2026-10-04）

## 包内文件 / 如何应用

把这个 ZIP **覆盖解压到 DynFL-privacy 项目根目录**（即与 `dynfed/`、`experiments/`、`tests/` 同级）。
已包含 Stage-1 的完整文件和本轮 Stage-2 的更新，因此不要求先安装旧 ZIP。
想核查相对于上一版 Stage-1 的代码变动，可看包内 `randomk_stage1_to_stage2.patch`；该 diff 不必再重复应用。
正式 `experiments/run_fmnist_lenet5.py` / SecAgg / selector / 原有 paper 配置完全没有修改。

## 新增内容

- `--mask-strategy randomk`：所有可训练坐标中全局均匀选取 `ceil(q*d)` 个。
- `--mask-strategy layerwise_randomk`：每个可训练张量中分别选择 `ceil(q*d_l)` 个。
- `--mask-strategy classifier_only`：仅发布分类头坐标；**依然进行完整本地训练，再投影更新**。真正的 head-only 训练请用 `--model resnet18_pretrained_head --mask-strategy randomk --fractions 1` 单独运行。
- `--diagnostic-disable-dp`：保留更新裁剪，关闭高斯噪声，只用于噪声消融；不具有 DP 保证，不记录 ε。
- 逐客户端 streaming 聚合，避免一次保存所有客户端完整更新；每轮 flush CSV，降低意外中断造成的数据损失。
- 私有诊断：`selected_fraction_actual`、`mean_client_signal_retention`（平均投影前后的客户端范数比）、`signal_norm`、`noise_norm`、`noise_to_signal`、`clipping_fraction`、`expected_noise_norm` 等。

## 隐私定义和不支持事项

固定公开参与名单、固定公开非负权重、客户端替换邻接，且受信聚合器每轮只发布**一个聚合更新**：选中子向量逐客户端 L2 裁剪至 `C`，整体灵敏度不超过 `2 C max(weight_i)`；加 `N(0, (sigma*S)^2 I_k)`，`sigma` 使用仓库的非子采样 RDP 校准并按实际 `--rounds` 次组合。公开随机掩码独立于训练数据。随机稀疏化不意味着 epsilon 按维度下降，也不自动使 noise/signal 下降。

- **这不是局部 DP，也不是已有 packet_protection 的替代方案。** 单客户端完整原始更新在逻辑上到达受信聚合器；此独立脚本没有传输层/HE/真实 SecAgg/Edge 路由。`end_to_end_dynfl_dp=not_established`。
- `--diagnostic-disable-dp` 不可称为 ε=144 等任何形式的 DP run。依赖私有训练数据的范数、裁剪统计、内部 CSV 等均属于**私有诊断**，不能作为 DP 输出公开。
- 公开 `--seed` 只决定数据划分/掩码/初始化；噪声种子由 `secrets.randbits(128)` 独立生成（使用 NumPy 伪随机发生器，生产部署须另做密码学随机源审计）。
- 这里的 public weights 假设样本计数等权重可公开、对受保护客户端替换不变；若要保护样本计数，不能沿用本隐私界。
- 仍然本地训练全部可训练参数；阶段 2 的稀疏化仅减少发布维度和预期噪声能量，并未证明减少本地梯度计算或真实通信字节。

## PowerShell 命令

以项目根目录 `E:\YTT\GROUP\DynFL-privacy` 为工作目录。

**1. 测试新模块：**
```powershell
D:\soft\Python310\python.exe -m pytest -q `
  .\tests\test_randomk_update.py .\tests\test_randomk_stage2.py
```

**2. 完整的相关回归（测试之前确保你的 configs/ 完整）：**
```powershell
D:\soft\Python310\python.exe -m pytest -q `
  .\tests\test_randomk_update.py .\tests\test_randomk_stage2.py `
  .\tests\test_jlearn_v2_aggregate_dp.py .\tests\test_streaming_secagg.py `
  .\tests\test_privacy_diagnostics.py .\tests\test_privacy_path_validation.py `
  .\tests\test_result_dp_selection.py .\tests\test_selection.py `
  .\tests\test_protection_rules.py .\tests\test_paper_conformance.py `
  .\tests\test_liic_streaming_secagg_integration.py `
  .\tests\test_liie_streaming_secagg_integration.py `
  .\tests\test_liieiiic_streaming_secagg_integration.py
```

**3. 一轮 CIFAR-10 / ResNet-18 随机更新（与之前历史 ε=144 对照，注意 ε=144 隐私较弱）：**
```powershell
D:\soft\Python310\python.exe .\experiments\run_randomk_fedavg.py `
  --dataset cifar10 --model resnet18_pretrained --device cuda `
  --train-limit 2000 --test-limit 500 --clients 20 `
  --rounds 1 --local-epochs 1 --local-steps 1 --lr 0.01 `
  --mask-strategy randomk --fractions 1 0.1 0.01 `
  --epsilon 144 --delta 1e-5 --clip-norm 0.25 --seed 42 `
  --output out\randomk_stage2\smoke_randomk_private.csv
```

**4. 逐层 Random-k（先测 q=0.1）：**
```powershell
D:\soft\Python310\python.exe .\experiments\run_randomk_fedavg.py `
  --dataset cifar10 --model resnet18_pretrained --device cuda `
  --train-limit 2000 --test-limit 500 --clients 20 --rounds 1 `
  --local-epochs 1 --local-steps 1 --lr 0.01 `
  --mask-strategy layerwise_randomk --fractions 0.1 `
  --epsilon 144 --delta 1e-5 --clip-norm 0.25 --seed 42 `
  --output out\randomk_stage2\smoke_layerwise_private.csv
```

**5. 真正的 head-only 参数训练（不是 classifier_only 投影）：**
```powershell
D:\soft\Python310\python.exe .\experiments\run_randomk_fedavg.py `
  --dataset cifar10 --model resnet18_pretrained_head --device cuda `
  --train-limit 2000 --test-limit 500 --clients 20 --rounds 1 `
  --local-epochs 1 --local-steps 1 --lr 0.01 `
  --mask-strategy randomk --fractions 1 `
  --epsilon 144 --delta 1e-5 --clip-norm 0.25 --seed 42 `
  --output out\randomk_stage2\smoke_head_only_private.csv
```

**6. 无噪声消融（用于判断 q=0.1 的信号损失；剪裁仍保留；不是 DP）：**
```powershell
D:\soft\Python310\python.exe .\experiments\run_randomk_fedavg.py `
  --dataset cifar10 --model resnet18_pretrained --device cuda `
  --train-limit 2000 --test-limit 500 --clients 20 --rounds 1 `
  --local-epochs 1 --local-steps 1 --lr 0.01 `
  --mask-strategy randomk --fractions 0.1 --diagnostic-disable-dp `
  --epsilon 144 --delta 1e-5 --clip-norm 0.25 --seed 42 `
  --output out\randomk_stage2\smoke_nodp_private.csv
```

完整配置跑通之后，再将 `--train-limit 12000 --test-limit 2000 --clients 100 --rounds 3 --local-epochs 3 --local-steps -1`。务必重新启动实验目录，不要复用旧 checkpoint。将 `--epsilon 144` 换为 `32` 等更小值可以验证更有意义的隐私-效用权衡；但比较时必须保证相同轮次/发布次数且单独记录 accountant 值。

## 结论判据

不要只看 `noise_norm`。同时关注：
1. `mean_client_signal_retention`：稀疏后有用更新是否大幅缩水？
2. `noise_to_signal`：减少噪声的同时信号是否也同步变小？
3. `test_accuracy`、`clipping_fraction`：是否存在收敛趋势及系统性裁剪偏差？
4. 同一随机种子下 DP 与无噪声实验的差距，以及多 seed 的均值/方差。

Random-k 实验不通过不会破坏现有正式 DynFL 主线；除非有重复的真实 CIFAR-10 实验证据，否则不建议把这个消融写成“已经提升精度”。
