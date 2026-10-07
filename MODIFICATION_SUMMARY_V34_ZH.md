# DynFL v3.4 Sample-DP / Joint Calibration 代码修改摘要

## 本次落地目标

把正式 Sample-DP 代码路径与最新理论主线对齐：

1. Privacy：Flow/真实 minibatch mechanism execution -> unified Sample-RDP feasibility。
2. Learning：真实 clean/private trajectory difference -> empirical `b_i, Tr(V_i)` -> 单一 joint update-space learning objective。
3. Online selector：只查离线/周期 calibration table，不在每次决策前重跑 paired clean/private training。
4. 正式 paper runner：只允许 `privacy.unit=sample`。

## 核心代码

### `dynfed/joint_calibration.py`

新增 joint calibration 模块：

- `PairedUpdateMomentAccumulator`：流式 O(d) 估计 `clean mean`、`bias mean`、`Tr(Cov(D))`；
- `JointUpdateCalibrationTable`：按 `state_key + mode + client(optional)` 保存经验矩；
- 最近 `round:k` fallback；
- `ideal_update` 支持 `e_alg` 的 table policy；
- 明确不拟合、不线性化训练映射。

### `dynfed/selection.py`

新增 `learning_objective=joint_calibration`：

- formal score：`||e_alg + e_agg + b_s||^2 + Tr(V_s)`；
- independent client RNG 时 `Tr(V_s)=sum a_i^2 v_i`；
- `table` / `zero` 两种 `e_alg` policy；
- `missing_policy=error` 防止正式实验静默退回旧 objective；
- 在线单-client neighbor lookahead 用 O(d) 增量 numerator，不扫描全部 clients、不做训练 replay；
- 新增 joint diagnostic 字段写入 audit/result。

旧 `fusion + update-DP` objective 保留为兼容/诊断路径，但正式 v3.4 配置不使用。

### Sample-RDP event count

`_sample_dp_event_counts()` 改为使用 runtime 实际 minibatch event 数，并乘真实 hierarchical training-block repetition (`E_edge_loops`)；避免 runtime 重训但 accountant 少记 events。

### `dynfed/fmnist_lenet5_dynamic.py`

- `state_key=auto_round` 在 round t 自动转为 `round:t`；
- runtime 输出 joint proxy diagnostics。

### Offline calibration builder

`experiments/build_joint_calibration.py`：

从 public/calibration workload 生成的 paired clean/private update files 建 table。每个 pair 直接观测

`D = u_private - u_clean`

而不是从 RDP epsilon 或 Gaussian energy 反推。

### CLI / formal runner

`experiments/run_fmnist_lenet5.py` 新增 joint calibration 参数，privacy unit 默认改为 sample。

`experiments/run_paper_config.py`：

- 正式论文配置缺省/要求 Sample-DP；
- 显式拒绝 client-level formal runs；
- 传入 joint calibration 配置。

### 新正式配置

`configs/paper_v34_cifar10_resnet18_joint_proxy.json`

- `privacy.unit=sample`
- `learning.objective=joint_calibration`
- `state_key=auto_round`
- `e_alg_policy=table`
- `missing_policy=error`

该配置**故意不附带伪造 calibration table**。需要先在独立 public/calibration workload 上生成真实 paired outputs，再构建 `calibration/paper_v34_cifar10_resnet18_joint.pt`。

## 使用顺序

1. 离线/周期 calibration workload 生成 matched clean/private update files。
2. 运行：

```bash
PYTHONPATH=. python experiments/build_joint_calibration.py \
  --manifest calibration/my_manifest.json \
  --output calibration/paper_v34_cifar10_resnet18_joint.pt \
  --min-trials 5
```

3. 确认 table 覆盖所需 `state_key/client/mode`，并提供 `ideal_update`（table e_alg policy）。
4. 运行 formal config。
5. 另抽样少量 round/candidates 做 paired-MC validation，报告 proxy 与 MC ground truth 的 ranking correlation。

## 验证

本修改树运行：

```text
PYTHONPATH=. pytest -q tests
501 passed, 56 skipped
```

并通过 v3.4 formal config dry-run。

## 当前明确边界

- 不假设训练映射 `Phi` 线性；code path 只处理真实 output differences。
- 不在每个 online decision 前测 `b_i,V_i`。
- 不构造 full covariance matrix，存储是 O(d)，不是 O(d^2)。
- 当前 builder 消费 paired update files；真实 public/calibration trajectory 的采集方式由实验 workload 负责。
- 若训练状态/DP multiplier 漂移显著，应增加 state-conditioned calibration points 或周期刷新，不能把单一 mode-only table 当作严格常数无限期复用。
