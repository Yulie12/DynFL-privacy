# Untrusted Edge 下的 Joint Noise：下一阶段设计边界

## 当前结论

现有 `client_secure_aggregate_common.py` 的独立 Gaussian share 路线有一个不可回避的方差边界。若 K 个 iid share 在完整 cohort 下恰好产生目标方差，则只剩 H 个 share 对 Edge/串谋方未知时，条件剩余方差只有 H/K 倍目标值。反之，若要求任意 H 个未知 share 单独仍具有完整目标方差，则完整 K 客户端释放的噪声能量膨胀为 K/H。

因此，“完整 cohort 恰好 1x 目标噪声”与“最多 K-H 个客户端与 Edge 串谋后仍保留同一目标方差”不能由普通 iid client shares 同时得到（H<K）。

## 本 patch 做什么

新增一个 `external_joint_noise_share` 接口，使 SecAgg packet 路径不再绑定于当前独立噪声生成方式。诊断中的 ideal functionality 会先生成一个目标 Gaussian 向量，再生成 K 个 additive shares，使它们严格求和为该目标向量。这样可以验证：未来真实 MPC/threshold backend 只要提供同样的 client-local share 接口，现有 X25519 pairwise masking 与 fixed-cohort Edge sum 无需重新设计。

## 本 patch 不做什么

`ideal_joint_gaussian_shares` 是集中式模拟器：同一 Python 进程见到完整目标噪声和全部 shares，因此它不提供任何针对 untrusted Edge 或 Client–Edge collusion 的密码学保证。测试通过只能证明协议 plumbing 和目标噪声代数正确，不能在论文中写成“threshold/MPC 已实现”。

真正闭环所需 backend 至少要满足：

1. 目标噪声在 secret-shared 状态下共同采样；
2. Edge 和允许范围内的 colluding clients 不能恢复完整噪声；
3. 每个 client 只能取得自己的 additive share；
4. 所有有效 shares 的和具有经过证明的目标分布；
5. cohort/round/model version 被密码学绑定；
6. fixed-cohort 阶段任何 dropout 均 abort，无 release；
7. 有明确的 malicious-client / biasing-noise 防护或明确限定 honest-but-curious client 模型。

## 文献路线

Kairouz, Liu, Steinke (ICML 2021) 的 Distributed Discrete Gaussian 证明了将每个客户端的离散 Gaussian 与 SecAgg 组合是可分析路线，并明确讨论恶意/掉线客户端比例导致保证退化。它不是“任意大串谋下固定 1x 连续 Gaussian”的免费实现。

Sabater, Bellet, Ramon (Machine Learning 2022) 使用图上的相关 Gaussian noise 加独立噪声，在串谋/恶意参与者下取得接近 trusted-curator utility，并指出这是 privacy/utility/communication 的协议设计问题，而非简单修改一个 sigma 即可解决。

若要求由秘密共享参与方直接共同生成 DP noise，则需要真正的 MPC random sampling / distributed coin-flipping 类协议；当前仓库的 `cryptography`、TenSEAL/SEAL 并不自动提供该功能。
