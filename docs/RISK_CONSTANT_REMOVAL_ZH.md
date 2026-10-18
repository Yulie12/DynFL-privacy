# 删除未校准的模式风险常数（Stage 16 之后增量）

原 `OBJECT_RISK` 与 `MECHANISM_RISK` 表使用人为数值（例如 `upd=0.58`、`none=1.0`），并以 `risk_limit=0.5` 将 `LIIEIIIC` 全部剔除。该分数不是经测量、校准或证明的泄露风险，因此不能作为硬约束或优化排序依据。

本补丁删除这两张常数表，自动生成候选的兼容字段设为 `risk=0.0`、`feasible_risk=True`、`risk_hard_gate=False`。**0.0 只是旧日志字段的中性占位，不代表无安全风险或隐私保护证明。** `risk_limit` 参数暂保留供旧配置解析，但对新的自动候选无效；LeNet-5/ResNet-18 共用的训练汇总额外输出 `legacy_risk_metric_status`，解释旧风险列已禁用。

依然保留：明确的 Client–Edge/Client–Cloud 信任准入、Edge-only 仅限提前声明的快响应任务、FL-first Split 门控、可用 DP/HE 机制、RDP 账本、资源/内存/Edge/Cloud 容量，以及显式 fast-response deadline。普通 `time_limit` 仍依照项目已有语义作为报告参考值，不在此补丁中改造成硬截止时间。

这会令原风险值0.58、原唯一被风险门槛拒绝的 `LIIEIIIC` 进入其他约束的可行候选集；**并不保证一定被 Pareto 选中或一定发生 Mode 切换**，也未完成新在线 J 的七模式严格证书。

对已在运行的100轮程序，Python 源码变更不会回溯改变已经启动的进程；若继续使用旧检查点与旧参数，不能把跨版本结果混为同一组正式实验。建议对照时从相同初始种子重新启动新输出目录。
