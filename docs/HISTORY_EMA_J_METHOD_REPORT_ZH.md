# History-EMA J 清理与验证记录

本分支已将 `history_ema_j` 作为正式候选 profile 学习目标接入；新方法路径不再要求 `online_upper_bound` 证书。旧证书模块仍在仓库中供历史配置兼容，但本配置、命令和新方法选择路径不依赖它们。

## 方法边界

- HE 关闭；无 HE 成本模拟。
- 不使用 aggregate-DP。
- LIIC：`L_C_upd` 上传前 DP。
- LIIEIIIC：`L_E_upd` 上传前 DP，`E_C_upd` 不保护。
- LIIE：Client→Edge 单次聚合→Cloud，记录真实 Edge 聚合提交。
- Cloud 权重使用候选 profile 中 Cloud-reaching 客户端样本量；同一 Edge 下 LIIEIIIC 客户端作为联合组，`E_B` 不作为贡献放大系数。

## 验证

运行 `python -m pytest tests/test_history_ema_j.py -q` 与 `python -m py_compile dynfed/history_ema_j.py dynfed/selection.py dynfed/fmnist_lenet5_dynamic.py experiments/run_paper_config.py experiments/run_fmnist_lenet5.py experiments/backtest_history_ema_j.py`。

真实训练和历史回测尚需在实际数据与 GPU 环境中运行；不把 dry-run 或合成测试作为精度结论。
