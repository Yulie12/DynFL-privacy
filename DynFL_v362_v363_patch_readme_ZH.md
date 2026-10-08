# DynFL v3.6.2 / v3.6.3 本地最小补丁使用说明

基线：用户 2026-10-08 上传的 `dynfl_fix_inputs.zip` 内 7 个源码文件。与旧版完整项目 ZIP 不同，`fmnist_lenet5_dynamic.py` 和 `validate_joint_proxy_ranking.py` 已有用户本地修改。本补丁以用户本次上传的源码为准，不要重新套用 v3.5.x / v3.6.1 旧补丁。

## 先应用 v3.6.2（推荐）

`dynfl_v362_mc_finite_sample_correction.patch` 修改：
- `experiments/independent_joint_proxy_capture.py`
- `experiments/validate_joint_proxy_ranking.py`
- `tests/test_independent_joint_proxy_capture.py`
- 新增 `tests/test_dynfl_joint_mc_correction.py`

仅对新的独立摘要输出采用 `dynfl_independent_joint_summary_v2` 和有限样本修正：

\[\widehat J_{\rm MC}=\|\bar R\|^2+\sum_i a_i^2(1-1/M_i)\widehat v_i.\]

其中 `mc_bias_squared` 仍是未经去偏的均值平方、`mc_variance_trace` 仍是无偏的单次更新方差迹贡献；新增 `mc_finite_sample_correction=\sum_i a_i^2\widehat v_i/M_i`，从二者之和减去，避免字段含义混乱。条件：固定基准/参考更新、给定初态及其他可固定条件后，不同客户端训练 DP 随机性独立；结果仍然只适用于局部 worker 更新边界。

旧 v1 summary 和旧 raw-vector 路径保留原有统计定义，以免悄悄重新解释历史 JSON。这些路径明确标为 legacy；不要把历史 v1 Spearman 和新 v2 数据混为一个同质的统计系列。

## 再考虑应用 v3.6.3（安全临时措施）

`dynfl_v363_sample_hierarchical_failclosed_preflight.patch` 修改：
- `dynfed/fmnist_lenet5_dynamic.py`
- 新增 `tests/test_dynfl_sample_hierarchical_preflight.py`

现有代码在首次 worker 完成后就核对并提交 Sample-RDP 账本，但 LIEIIIC / LIIEIIIC 的后续 Edge 重训练位于该提交之后。若强行使用 predicted event count 来替代真实的多 stage 观测，隐私审计证据不充分。此补丁在 calibration capture 和 worker dispatch 之前拒绝 `privacy_unit=sample` 且 `E_edge_loops>1` 的训练任务，保留单阶段 Sample 训练及原有 Client-level 执行。

**重要**：这不是恢复分层 Sample-DP 的功能性修复。应用后，如果正式训练选择 LIEIIIC 或 LIIEIIIC 的 Sample-DP 候选，运行将明确中止。要完整实现七种 Sample-DP 模式，需要将所有 stage 的真实事件先行收集/核对，并在泄露或发布边界之前统一执行正确的 ledger commit；不能仅移动一个 assert 或关闭 fail-closed。

## 本地执行（PowerShell，项目根目录）

把两个 patch 放在 `E:\YTT\GROUP\DynFL-privacy` 根目录：

```powershell
git apply --check .\dynfl_v362_mc_finite_sample_correction.patch
git apply .\dynfl_v362_mc_finite_sample_correction.patch

git apply --check .\dynfl_v363_sample_hierarchical_failclosed_preflight.patch
git apply .\dynfl_v363_sample_hierarchical_failclosed_preflight.patch

D:\soft\Python310\python.exe -m pytest -q `
    tests\test_dynfl_joint_mc_correction.py `
    tests\test_dynfl_sample_hierarchical_preflight.py `
    tests\test_independent_joint_proxy_capture.py `
    tests\test_sample_dp_privacy_path_audit_contract.py
```

若 `git apply --check` 失败，立即停止，不要用 `--reject`、`git reset --hard` 或重放旧补丁；提供报错即可。

## 已完成的验证

- 在独立复原的项目源树中，把上传的 7 个文件替换到对应目录，再重放两个补丁：两次 `git apply --check` 和 `git apply` 均成功。
- 指定的 13 组测试文件，共 **75 passed**（约 5.89 秒）。这不是完整 GPU 训练、全量 7 模式验收或协议级 Sample-DP 证明。
- 已检查 `git diff --check`（使用 `core.whitespace=cr-at-eol` 适配上传文件的 CRLF 行尾）。

## 现在不做的事

- 不修改 `dynfed/selection.py` 的 Pareto 排序算法。
- 不覆盖校准表与现有 `out/` 结果。
- 不把 ρ=1、0.5、−0.5 旧结果改写为修正后的结果。
- 不声称当前已有端到端 Sample-level DP 证明。
