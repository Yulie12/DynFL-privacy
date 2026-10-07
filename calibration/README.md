# Joint update-space calibration files

此目录保存 Sample-DP joint learning proxy 的离线/周期校准产物。

在线 selector 只读取 `.pt` calibration table，不在每次决策前执行 clean/private paired training。

构建方式见仓库根目录 `JOINT_UPDATE_CALIBRATION_PROXY_ZH.md` 和 `experiments/build_joint_calibration.py`。
