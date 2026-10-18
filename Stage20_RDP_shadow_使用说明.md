# Stage 20 RDP-aware Transfer J — 影子审计安装与运行

**前提**：你已经安装 Stage 19 (`online_j_oracle_diagnostic`)；目前使用 CIFAR-10 + ResNet18 + `privacy.unit=sample` + `he.execution=profiled`，以及实验性受保护三阶段 Sample-DP 路径。只用于中心模拟器诊断，不得将输出声称为真实 HE/DP 认证结果。

## 文件

- `DynFL_Stage20_RDP_Transfer_J_推导.md`：完整数学推导，包含 RDP 与噪声转换、两级样本权重、证明及适用条件。
- `DynFL_stage20_shadow_metadata.patch`：**只给决策前的 Oracle JSON 增加审计字段**，不修改旧 J、Pareto 目标、DP/训练实现。
- `dynfl_transfer_j_reference.py`：独立传递系数与数学上界实现。
- `audit_stage20_rdp_shadow.py`：读取带新审计字段的运行目录，计算**选中方案**的 Shadow J；验证所选 σ、裁剪 C、批次、RDP 事件计数和记录中的候选模式。
- `test_dynfl_transfer_j_reference.py` 与 `test_audit_stage20_rdp_shadow.py`：11项单元测试（参考/模拟数据；不是完整 FL 隐私证明）。

## 1. 解压到仓库根目录，安装补丁

在 PowerShell 执行（不要复制终端提示符）：

```powershell
cd E:\YTT\GROUP\DynFL-privacy

git apply --check .\DynFL_stage20_shadow_metadata.patch
if ($LASTEXITCODE -ne 0) { throw "当前代码版本无法直接应用 Stage20 补丁，请勿强制覆盖" }
git apply .\DynFL_stage20_shadow_metadata.patch
if ($LASTEXITCODE -ne 0) { throw "补丁应用失败" }

& "D:\soft\Python310\python.exe" -m unittest -v test_dynfl_transfer_j_reference test_audit_stage20_rdp_shadow
if ($LASTEXITCODE -ne 0) { throw "参考单元测试未通过" }
```

不要重复 `git apply` 同一个补丁。若此前已经应用，`git apply --check` 可能提示已应用而不能重复应用。

## 2. 创建全新输出目录，运行一轮

下面假定配置 `configs/stage19_resnet18_20c_2e_1cloud.json` 已存在且信任清单已改为 `configs/v370_paper_trust20_2edges.json`：

```powershell
$src="configs/stage19_resnet18_20c_2e_1cloud.json"
$dst="configs/stage20_resnet18_20c_2e_shadow_1r.json"
$config=Get-Content $src -Raw | ConvertFrom-Json
$config.training.rounds=1
$config.output_root="out/stage20_resnet18_20c_2e_shadow"
$utf8=New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path (Get-Location).Path $dst),($config | ConvertTo-Json -Depth 100),$utf8)

& "D:\soft\Python310\python.exe" experiments/run_paper_config.py --config $dst --policies full_dynfl --max-new-rounds 1 --dry-run
if ($LASTEXITCODE -ne 0) { throw "dry-run failed" }

$env:DYNFL_EXPERIMENTAL_SAMPLE_MULTISTAGE="1"
try {
    & "D:\soft\Python310\python.exe" experiments/run_paper_config.py --config $dst --policies full_dynfl --max-new-rounds 1
    if ($LASTEXITCODE -ne 0) { throw "运行失败，请保留原始日志" }
}
finally {
    Remove-Item Env:\DYNFL_EXPERIMENTAL_SAMPLE_MULTISTAGE -ErrorAction SilentlyContinue
}
```

**注意**：不要为凑够轮数关闭 DP 预算或隐私事件审计；如果实验性多阶段路径失败，应分析实际错误。

## 3. 计算 Shadow J（不改原选模结果）

```powershell
$root="out/stage20_resnet18_20c_2e_shadow"
$run=Get-ChildItem $root -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$dir=Join-Path $run.FullName "full_dynfl"
& "D:\soft\Python310\python.exe" .\audit_stage20_rdp_shadow.py $dir
```

输出位于 `full_dynfl/stage20_shadow_transfer_selected_NONCERTIFIED.csv`，其中包含 Stage19 已选方案的旧 J、新传递上界及三个分量 \(\|D\|,G,\sqrt V\)。算法仍使用 Stage19 J 选模。Shadow 只核对日志记录的候选 cohort/σ/事件**计划**，尚未验证实际 Worker/Edge/Cloud 运行图与完整数值误差，因此明确标记 `NOT_DEPLOYMENT_CERTIFIED`。

之前已经完成的3轮 Stage19 运行不能直接计算新界：旧快照缺少 `stage20_shadow_inputs`，脚本会拒绝运行，不会伪造数值。

## 4. 进一步证明责任

- 把 Worker 的逐阶段事件回执、实际 Edge 加权参与集合、Cloud 更新贡献清单与 Shadow 决策前计划逐项一致性核对。
- 检查对 BatchNorm running buffers 等非梯度状态的聚合是否影响当前定义的完整模型更新。
- 为真实 CKKS 与浮点数误差添加可证非零上界；当前 `HE profiled` 不做加密。
- 通过可信的客户端证书/安全聚合证明替换中心 Oracle 原始数据访问，才具备真实 FL 部署意义。
