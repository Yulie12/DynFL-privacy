# MPC joint-noise 下一协议边界

当前代码已经验证三件事：

1. X25519 SecAgg 可以隐藏单个客户端 update；
2. external-noise 接口可以把一个理想 joint noise share backend 接入现有 packet 路径；
3. iid exact-target shares 外加 pairwise zero-sum masks，不能把 coalition 条件方差从 `H/K` 提升到 `1`。

本阶段进一步固定一个更强的边界：如果每个客户端本地直接知道一个最终实值 Gaussian additive share `X_i`，并且最终噪声是 `Z=sum_i X_i`，那么仅仅设计 `X` 的相关协方差不能同时满足：

- `Var(Z)=sigma_target^2`（exact 1x aggregate variance）；
- 对任意允许的 Client–Edge coalition，`Var(Z | coalition view)=sigma_target^2`。

原因是

`sum_i Cov(Z, X_i) = Cov(Z, sum_i X_i) = Var(Z)`。

只要 `Var(Z)>0`，至少有一个本地 share 与 `Z` 相关。对 jointly Gaussian 构造，观察这样的 share 会严格降低条件方差。因此，下一协议不能再把“最终 Gaussian additive contribution”明文交给每个客户端后期待靠相关性解决串谋问题。

这不是一般 MPC 不可能性结论。下一层正确抽象应当是：

- 客户端只持有 secret-sharing / MPC state，而不是可直接交给串谋 Edge 的最终 Gaussian contribution；
- global random seed 或 global noise 在 MPC/threshold secret-sharing 域中联合产生；
- Gaussian/PRG 采样及其到模型向量的展开在秘密状态上完成；
- 只有“noisy aggregate”被打开；clean aggregate 与 global noise 都不单独打开；
- fixed cohort 继续保持，dropout 仍先采用 abort；
- 在没有正式实现 VSS/MPC backend 前，`cryptographic_realization` 与 `collusion_resistance_established` 必须保持 false。

因此下一工程阶段应先选择/实现一个真实的 secret-sharing backend，再把它接到现有 `external_noise` SecAgg 接口；不应再增加本地 Gaussian share 的 covariance heuristic。
