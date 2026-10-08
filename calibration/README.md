# Joint update-space calibration files

此目录保存 Sample-DP joint learning proxy 的离线/周期校准产物。

在线 selector 只读取 `.pt` calibration table，不在每次决策前执行 clean/private paired training。

构建方式见仓库根目录 `JOINT_UPDATE_CALIBRATION_PROXY_ZH.md` 和 `experiments/build_joint_calibration.py`。


## v3.5 自动 capture

推荐先做一次很小的 dry-run：

```powershell
D:\soft\Python310\python.exe experiments\run_joint_calibration_capture.py `
  --config configs\paper_v34_cifar10_resnet18_joint_proxy.json `
  --output calibration\paper_v34_cifar10_resnet18_joint.pt `
  --rounds 1 --trials 2 --period 1 --max-clients 1 `
  --sample-limit 16 --scope candidate_modes --dry-run
```

确认命令无误后去掉 `--dry-run`。自动 capture 是独立 calibration pass；正式 selector 只加载输出 table。

`--sample-limit` 限制每个 client calibration pool 中唯一 held-out 样本数；执行 trajectory 时会从该 pool 重采样到 client 的实际 training sample count，以保持实际训练 step 数。

**隐私注意**：仓库的 held-out validation split 不自动等价于 public data。论文若需要“校准不产生额外隐私泄露”的正式表述，应使用独立 public calibration dataset。
