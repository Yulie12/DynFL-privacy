# History-EMA J 新方法

## 目标

History-EMA J 是当前分支的主动学习目标，而非影子指标。候选 Cloud 方案的预测更新由历史 LIIC 与 LIIEIIIC 更新 EMA 按候选方案中的 Cloud 样本量权重组合；随后与历史全局更新 EMA 计算归一化平方偏离。LIIE 不提交 Cloud 更新，因此其 Cloud 贡献为零。

## 运行原则

- 本方法不读取、不生成、不要求任何 `online_upper_bound` 证书。
- `learning.objective` 必须为 `history_ema_j`，`learning.history_ema_j_enabled` 必须为 `true`。
- HE 必须关闭（`backend=none`、`execution=real`）；HE 成本不参与本方法。
- 不使用 aggregate-DP；LIIC 在 `L_C_upd` 使用上传前 DP，LIIEIIIC 在 `L_E_upd` 使用上传前 DP，`E_C_upd` 不做保护。
- 同一 Edge 的 LIIEIIIC 客户端按联合提交组处理；`E_B` 不作为贡献乘数。
- 初始历史不足时，所有候选 profile 使用相同的中性 J 分数，由既有延迟/可行性目标完成 warm-up；当全局历史以及 LIIC、LIIEIIIC 两种 Cloud 模式历史齐备后，J 参与全局候选 profile 的排序。

## 输出

- `history_ema_j.jsonl`：每轮预测、观测、EMA 更新与跳过原因。
- `history_ema_j_state.pt`：可恢复的 EMA 状态。
- `experiments/backtest_history_ema_j.py`：对真实历史日志做预测误差回测。

## 当前限制

真实训练与历史回测需在用户环境执行。历史回测指标用于评价 J 预测校准，不自动证明 J 与最终测试精度单调相关。
