# 第十一阶段：真实 DynFL 代码接入在线 J 上界（可审计中间版本）

> **结论先行**：新的 `online_upper_bound` 选模目标已经接入完整 Profile/Pareto/Tchebycheff 链条，在线上界代数和候选身份检查有自动测试；**但当前还不能称为七模式正式实验**，不能拿合成证书充当真实模型证书。缺少每轮、每种实际执行 Mode 的预决策误差证书时，程序应中止。

## 1. 新目标与数学假设

本阶段保持目标：

\[
\Delta_t^*=-\eta_t^*\nabla F(w_t),\qquad
J_t(h)=\mathbb E_t\|\Delta_t(h)-\Delta_t^*\|_2^2.
\]

对于能够从**真实训练/聚合实现**证明可拆成有效客户端等效贡献的模式，条件于本轮选择前的信息，令：

\[
\Delta_{\rm clean}=\gamma\sum_{i\in C(h)} a_i u_i(h),\quad
c_i(h)=-S_i(h)g_{\rm ref},\quad
\|u_i(h)-c_i(h)\|_{L_t^2}\le r_i(h),
\]

\[
\|\nabla F(w_t)-g_{\rm ref}\|_{L_t^2}\le\epsilon_t,\quad
a_i=\frac{n_i}{\sum_{j\in C(h)}n_j},\quad
Q_t=\gamma^2d\sum_{i\in C(h)}a_i^2\sigma_{i,h}^2.
\]

其中 \(Q_t\) 是**聚合前**参与方生成、在 SecAgg 下合成的条件中心化且相互正交的高斯噪声份额二阶矩，绝不声称 Cloud 聚合后再加 DP。非 Split 本地 DP-SGD 已造成的非线性更新偏差必须包含在相应的 \(r_i\)，不能重复算入 \(Q_t\)。各模式的 \(S_i,r_i,\sigma_i\) 必须覆盖当前模式、机制和噪声配置。

\[
B_t(h)=\left|\eta_t^*-\gamma\sum_i a_iS_i(h)\right|\|g_{\rm ref}\|
+\eta_t^*\epsilon_t+\gamma\sum_i a_ir_i(h).
\]

若数值误差均方根不超过 \(r_{\rm num}\)，则

\[
\boxed{J_t(h)\le U_t(h)=\left(\sqrt{B_t(h)^2+Q_t(h)}+r_{\rm num}\right)^2.}
\]

**注意**：这个版本选用共享历史参考梯度 \(g_{\rm ref}\) 的特殊情形；不能用它自动证明多阶段 Edge 模式共享相同参考方向。配置中所提供的证书有外部数学前提，代码只检查格式、身份、轮次与显式声明，并不在运行时证明梯度光滑性、偏差球、加权聚合、DP 顺序或噪声独立性。

## 2. 改动范围

- `dynfed/online_j_bound.py`：新增严格条件性上界计算器，所有缺失状态报错。一个模式签名包含 Mode、每条 Link 保护机制、DP 执行计划、Sample-DP 事件数/噪声参数。
- `dynfed/selection.py`：支持 `learning_objective="online_upper_bound"`；预筛选、完整 Profile、最终 Pareto 均读取同一证书。记录 `online_j_center_norm_sq/online_j_radius/online_j_noise_second_moment/...`，与旧 Joint Calibration 诊断分开。
- `dynfed/fmnist_lenet5_dynamic.py`：每轮切换 `online_bound_round`；可选训练 batch 256；新增取消本地五步上限；同步 selected Profile 诊断。
- `dynfed/split_learning.py`：新增可选的实际 `train_batch_size` 参数；不传时保留原始训练规则。
- `experiments/run_fmnist_lenet5.py`、`experiments/run_paper_config.py`：新增命令行与 JSON 映射。
- `tests/test_online_upper_bound_integration.py`：公式、聚合前方差、缺失轮次、缺失候选、证书缺项、选模接口验证。
- `tests/test_stage11_locked_batch_and_no_cap.py` 与准入测试：批次、无步数截断和纠正的 Mode 门槛。

## 3. 新的实验参数（来自已锁定约束）

四份 `configs/stage11_*_5r_STAGING.json`：FashionMNIST / LeNet-5 全模型；CIFAR-10 / ResNet-18 Layer4+分类器，均设 5 轮、batch=256、local epochs=3、不截断 batch 数、逐样本梯度裁剪 C=3、最终更新裁剪 C=3、RDP 自动会计、HE 仅 profile 耗时。旧目标组和新目标组使用**相同的纠正准入开关**，避免混淆 Mode 准入与学习目标的影响。

⚠️ **本阶段只锁了配置和 batch/step 数执行接口，尚不能宣称 DP 路径符合最终正式要求。** 原训练器的 Sample-DP-SGD 仍在本地 minibatch 事件处执行噪声和会计，而你指定的是“全部本地 batch 完成后、上传之前一次生成高斯更新噪声，按真实上传次数计 RDP”。两者不等价。还需修改实际噪声放置与隐私会计，并单独审计。

⚠️ 原代码的可信 Split Sample-DP 路径仍有内部 Sample-DP/联合 DP，尚不符合可信 Client–Edge Split **内部不加 DP**。纠正准入开关暂时**阻断 Sample-DP 下的 Split Mode**，并阻断未独立认证 Client–Cloud 可信边界的 LIC；因此当前没有能力宣称七模式正式生效。

⚠️ `he.execution="profiled"` 代表耗时建模而非真实 SEAL/HE 执行；在尚无实际 HE 后端时，现有代码可能会剪枝 HE 候选。配置的 `candidate_mechanisms` 列表不代表这几种机制一定实际启用。

## 4. 运行与检查

在新代码仓库根目录：

```powershell
python -m pytest -q tests/test_online_upper_bound_integration.py tests/test_stage11_locked_batch_and_no_cap.py tests/test_v370_admission.py
python examples/online_j_synthetic_demo.py
python experiments/run_paper_config.py --config configs/stage11_fmnist_lenet5_online_5r_STAGING.json --dry-run
python experiments/run_paper_config.py --config configs/stage11_cifar10_resnet18_layer4_online_5r_STAGING.json --dry-run
```

正式运行前，`learning.online_bound_manifest_path` 对应的真实事前证书文件必须存在，例如：

`certificates/REAL_fmnist_lenet5_5r_predecision_required.json`

现在**没有替你编造这些文件**。缺失时在线目标会报错，不会退回旧的 Fusion 代理或 `joint_calibration`。

对照实验配置：

```powershell
# 先做原方法基线（运行要具备真实数据集/依赖/设备）
python experiments/run_paper_config.py --config configs/stage11_fmnist_lenet5_legacy_5r_STAGING.json

# 在线方法：必须先提供当前真实状态的有效预决策证书，否则终止
python experiments/run_paper_config.py --config configs/stage11_fmnist_lenet5_online_5r_STAGING.json
```

**不要在补齐下面的依赖之前把此对照写成“正式实验成功”**：训练器与证书的 Mode 等效分解、参考梯度误差半径、各 Mode Momentum/多阶段/DP-SGD 预测残差、实际 SecAgg 份额协方差、以及与真实发布次数一致的 RDP 隐私会计。

## 5. 证书 JSON 格式（教学演示，不是生产证书）

```json
{
  "schema": "dynfl.online_j.v1",
  "target": "ideal_full_gradient_one_step",
  "rounds": {
    "round:0": {
      "round_index": 0,
      "available_before_round": 0,
      "evidence": "数学推导、部署实现及审计证据的标识（不要只写已验证）",
      "reference_gradient": [0.1, -0.2],
      "gradient_rms_error": 0.01,
      "ideal_step": 0.15,
      "cloud_step": 1.0,
      "numeric_rms": 0.001,
      "aggregation_rule": "cloud_participant_sample_weighted",
      "no_cloud_update_is_zero_verified": true,
      "assumptions": {
        "weighted_client_contributions_match_runtime": true,
        "gradient_reference_rms_bound_proved": true,
        "mode_rms_bounds_proved": true,
        "noise_shares_drawn_before_secagg_and_weight_fixed": true,
        "noise_conditionally_zero_mean_and_uncorrelated": true,
        "numeric_rms_bound_proved": true
      },
      "clients": {
        "0": {
          "candidates": {
            "<由 certificate_candidate_key(candidate) 精确生成的 key>": {
              "step_scale": 0.1,
              "mode_rms": 0.02,
              "aggregate_share_std": 0.01,
              "runtime_mode_equivalence_verified": true,
              "local_dp_included_in_mode_rms": true
            }
          }
        }
      }
    }
  }
}
```

应当从真实 Runtime 预决策快照**验证这些条件并自动生成证书**；手写 JSON 中的布尔值只是声明，不构成论文意义上的证明。`reference_gradient` 的长度必须与经过证明的更新空间维度一致；RDP、SEAL/数值误差及批次/模型更新的签名还需由真实代码侧审计。

## 6. 下一阶段最重要的改动

1. 修改 Sample-DP / 更新级噪声放置和 RDP 计数，严格符合**本地所有批次完成后、上传前一次高斯发布**（并按真实实际发送事件计）；确保不是 Cloud 完成聚合再加噪。
2. 逐 Mode 建立真实更新与历史参考梯度间的在线半径，并严格区分完整本地 SGD/Momentum、可信 Client–Edge Split（内部无 DP）、多阶段 Edge 反馈。
3. 最后再跑 LeNet-5 与 ResNet-18 的五轮成对实验：真实训练损失/准确率、Mode/时延、上界覆盖、紧致性、排序、事后决策损失，且相同 seed、数据划分、预算与可行域。

本阶段测试的“有效上界”始终是**有条件的形式化保证**，并非运行时自动认证真实 DynFL 的全部假设。
