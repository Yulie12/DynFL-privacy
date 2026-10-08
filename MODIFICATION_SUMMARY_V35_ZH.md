# DynFL v3.5：自动 Joint Calibration Capture 修改说明

## 目标

v3.4 已经让在线 selector 使用 joint update-space calibration table，但 table 仍需手工准备 paired update 文件。v3.5 把 paired clean/private trajectory 的**采集**接进现有训练执行器，同时保持以下边界：

- paired measurement 是离线/周期 calibration 工作，不是每轮在线决策前测量；
- clean/private 都执行真实 nonlinear training，不拟合线性 `Phi`；
- clean 与 private 共享 common randomness，private trials 只改变 DP RNG；
- 正式 selector 仍只读 `.pt` table。

## 主要新增

### `dynfed/joint_calibration.py`

新增 `JointCalibrationCaptureSession`：按 `(state_key, mode, client_id)` 流式积累 paired trajectories，最终生成 `JointUpdateCalibrationTable`。mode fallback 从 client conditional moments 构造，不把 client 间 bias heterogeneity 混入 DP-RNG variance。

### `dynfed/fmnist_lenet5_dynamic.py`

新增自动 capture 路径：

1. `_clean_calibration_payload`：生成真正 non-private baseline，而不是 `sigma=0` 的 clipped path；
2. `_run_paired_calibration_payload`：clean 跑 1 次，private 跑 M 次，固定 training seed、改变 DP seed；
3. `_capture_joint_calibration_round`：周期性抽取 client/mode cell，生成 paired moments；
4. `_calibration_reference_update`：在相同 calibration workload 上计算 `Delta_*=-eta grad F`；
5. round metrics / summary 记录 capture pair 数与 wall time。

held-out pool 会先限制唯一 calibration 样本数，再重采样到该 client 的实际 training sample count，从而尽量保持 runtime minibatch/optimizer-step 长度。

### `experiments/run_joint_calibration_capture.py`

新增独立 calibration runner。它从正式 paper config 读取模型、系统和 Sample-DP 设置，但使用单一 bootstrap policy 与 legacy selector 来产生候选，不要求正在构建的 calibration table 已存在。

默认使用 profiled HE，因为 HE 加密执行不改变本地 paired learning trajectory；该例外只用于 calibration capture。

## 新增 CLI

`experiments/run_fmnist_lenet5.py` 支持：

- `--joint-calibration-capture-path`
- `--joint-calibration-capture-trials`
- `--joint-calibration-capture-period`
- `--joint-calibration-capture-max-clients`
- `--joint-calibration-capture-sample-limit`
- `--joint-calibration-capture-scope {selected,candidate_modes}`

## 推荐第一次运行

先只检查命令：

```powershell
D:\soft\Python310\python.exe experiments\run_joint_calibration_capture.py `
  --config configs\paper_v34_cifar10_resnet18_joint_proxy.json `
  --output calibration\paper_v34_cifar10_resnet18_joint.pt `
  --rounds 1 `
  --trials 2 `
  --period 1 `
  --max-clients 1 `
  --sample-limit 16 `
  --scope candidate_modes `
  --dry-run
```

确认后去掉 `--dry-run` 做一个最小真实 capture。ResNet + per-sample DP 本身较重，因此先用 1 round / 1 client / 2 trials 验证链路，再扩大 M 和 client 数。

## 隐私边界

当前自动 capture 使用每个 client 的 held-out validation pool，它与该 client 的 training subset 不重叠，但**不应被自动称作 public data**。如果论文需要声称 calibration 不增加隐私泄露，正式 calibration 应改用独立 public dataset；外部 public paired files 仍可通过 `experiments/build_joint_calibration.py` 构表。

## 验证

v3.5 新增测试覆盖：

- exact client cell 与 mode fallback；
- within-client variance 语义；
- clean path 移除 Sample-DP；
- paired trials 固定 training seed、改变 DP seed；
- reference update 不修改真实模型；
- held-out pool 重采样到 runtime client sample mass；
- dedicated capture runner command 构造。
