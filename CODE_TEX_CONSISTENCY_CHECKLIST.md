# 代码—论文一致性检查清单（v30 主线融合）

规范源为 `tex/paper/main.tex`，正式执行版本为
`paper_flow_v30_mainline_fusion_method2`。旧的 per-link DP/HE 动态替换语义仅保留在
legacy/ablation 路径中，不属于 v30 正式主线。

## v30 固定契约

- 七种基础协作模式仍保留在 taxonomy：`LIE`、`LIC`、`LIIE`、`LIIC`、
  `LIEIIC`、`LIEIIIC`、`LIIEIIIC`。
- 当前 honest-but-curious cloud threat model 下，`LIC` 因直接暴露中间表示而
  不进入 executable set；正式 selector 在其余六种 mode 中动态选择。
- selector 决定协作/执行/资源路径，不决定是否启用结果 DP。
- 每个正式 round 的 client contribution 都从上一轮已发布 global model 出发，
  不能接收同轮其他客户端的未发布 private aggregate。
- 每个 client contribution 先做 client-level L2 clipping；正式 release 使用
  replacement sensitivity `2 * C * max_i(w_i)`。
- 每轮只产生一个 formal global DP release event。Local epochs/内部 trusted-edge
  执行阶段不会增加 release count。
- distributed Gaussian noise 在 cloud 可获得可解 aggregate 之前加入；cloud 只
  聚合 ciphertext；trusted custodian 只解密 already-noisy aggregate。
- HE 是 cloud-path confidentiality layer，不可替代结果 DP。
- 正式主线要求：`aggregation_fraction=1.0`、`dp_accounting_mode=rdp_auto`、
  real HE、`he_aggregation_size=0`，并禁用旧的 DP-vs-HE stability switching。
- 当前只声明上述 aggregate-release mechanism 在固定公共 roster/weights 假设下的
  client-level accounting；`end_to_end_dp_status=not_established` 保持不变。

## 一致性矩阵

| ID | 检查点 | TeX | 代码 | 状态 |
| --- | --- | --- | --- | --- |
| F1 | 7 basic / 6 executable modes | `H_base` 与 `H=H_base\{LIC}` | `MODE_SPECS` + fused candidate enumeration | PASS |
| F2 | 动态 selector 只选 mode | `x=h_N`，`pi_i` 为 singleton induced map | `SelectionConfig.mainline_fusion` | PASS |
| F3 | 固定 DP + HE release contract | Privacy Budget Accounting | fused branch in `fmnist_lenet5_dynamic.py` | PASS |
| F4 | client-replacement sensitivity | `S_t=2 C_u max_i q_i^t` | `IndependentReleaseAccount` + aggregate calibration | PASS |
| F5 | 一轮一次 accounting | one formal release per round | `IndependentReleaseAccount.reserve` | PASS |
| F6 | private same-round feedback 禁止 | independent contribution text | `_training_base_state(..., mainline_fusion=True)` and disabled shared-state loop | PASS |
| F7 | HE 不替代 DP | fixed confidentiality layer | `_reported_release_mechanism(...)=dp_he3` in fusion | PASS |
| F8 | selector utility 不因固定 HE 人为变化 | common privacy-noise term | fusion candidate accuracy removes mechanism penalty/jitter | PASS |
| F9 | aggregate DP-noise cost 使用同一 horizon | one release/round | `resolved_privacy_parameters` + `_fusion_aggregate_noise_cost` | PASS |
| F10 | formal policy set 不含 privacy-bypass controls | No Protection 仅独立 control | v30 configs / runner policy validation | PASS |
| F11 | full-update HE | full protected update | `he_aggregation_size=0` required | PASS |
| F12 | 审计字段描述 global release | release-level accounting language | `global_release_*`, fused reporting scope | PASS |
| F13 | 完整协议端到端 DP | 明确不声明 | `end_to_end_dp_status=not_established` | PASS |

## Legacy 路径边界

`fixed_dp`、`fixed_he`、`privacy_only`、`no_protection` 以及旧 packet-level
mechanism selection 仍可用于历史兼容或单独消融，但不得和
`mainline_fusion=True` 混在同一 formal run 中。论文主结果的动态 mode 选择不能
通过 pure HE 绕过 global DP release gate。

## 修改后固定核对顺序

1. `tests/test_mainline_fusion.py`：融合契约、CLI、配置、accounting、reporting。
2. `tests/test_result_dp_selection.py`：结果 DP gate 与 legacy 兼容性。
3. `tests/test_independent_release.py`：固定 roster、horizon、resume/accounting。
4. `tests/test_paper_conformance.py`：流、聚合、Pareto/TeX 数学一致性。
5. 全量 `pytest -q`。
6. `git diff --check` 与 patch replay `git apply --check`。
7. 真实 SEAL 短程验证与正式多 seed 结果分开记录。

本次 Method 2 增量范围及未闭合的性能证据见 `PAPER_CONFORMANCE.md` 的
“本次 Method 2 正式接入”段落。上表的 PASS 指契约检查，不表示收敛效果、
物理部署或系统 profile 已获验证。
