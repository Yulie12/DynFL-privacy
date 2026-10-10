# V370 最终实验入口接线（仅代码入口，不包含真实信任数据）

## 适用范围与事实

用户已在 AutoDL 完成 V370 `dynfed/selection.py` 补丁应用、14 个隔离准入测试通过。此包仅修改实验入口：

- `experiments/run_paper_config.py`：读取 `system.static_pair_admission_manifest`，转发命令行标志；显式拒绝未打开 FL-first、旧 fast-only 总限制或重复的 fast deadline 来源。
- `experiments/run_fmnist_lenet5.py`：新增 `--static-pair-admission-manifest`，在构造 `SelectionConfig` 后调用 `enable_static_pair_admission()`。

本工具读取 AutoDL 上的实际源码进行 AST/锚点定位；**未知代码形态则报错停止**，不默默修改。修改前创建 `.before_v370_wiring` 同目录备份。没有改变 Pareto/校准/DP/HE 内容。

**限制：** 本包只在合成入口代码上测试通过，尚未在用户完整 AutoDL 项目验证。原 V370 补丁的通用 `time_limit` 是报告字段，不是每位普通 Client 的硬截止时间；普通 Client 的独立训练 deadline 还没有被完整实现。不要声称该项已实现或端到端 Sample-DP 已建立。

## 安装与校验

Windows PowerShell（ZIP 下载到 Downloads 后）：

```powershell
scp -P 15940 "C:\Users\19720\Downloads\v370_entry_wiring.zip" root@connect.bjb1.seetacloud.com:/root/autodl-tmp/DynFL-privacy/
```

AutoDL 终端：

```bash
cd /root/autodl-tmp/DynFL-privacy
unzip -o v370_entry_wiring.zip
python v370_entry_wiring/v370_wire_entry.py --check
python v370_entry_wiring/v370_wire_entry.py --apply
python -m compileall -q experiments/run_paper_config.py experiments/run_fmnist_lenet5.py
python experiments/run_fmnist_lenet5.py --help | grep static-pair-admission
python -m unittest discover -s tests -p 'test_v370_admission.py' -v
git diff --check
```

**如果 `--check` 报错，不运行 `--apply`，贴错误输出。** 若 `--help` 在依赖导入阶段出错，先不要启动训练。

## 建立正式信任清单

**不能**直接使用原 V370 包的 `admission_example.json` 作为研究真值。新 JSON 的数组必须由实验前固定的 Client→Edge 拓扑和各 Client 的可信选择填入，Client/Edge ID 应当是源码实际使用的 0-based ID。`fast_client_deadlines` 只能填写实验前预指定的快速响应客户端。

```json
{
  "trusted_client_edge_pairs": [],
  "fast_client_deadlines": []
}
```

空信任列表仅适于安全的接线/禁止 Split 冒烟测试，**不能用来评价七模式最终分布**。请填好真实受信任的连接对。V370 的旧版 manifest helper 当前只支持这两个字段，不支持 `client_training_deadlines`。

## 创建 V370 独立配置（请先提供真实清单）

将真实信任 JSON 保存为 `configs/v370_trust_manifest.json`，然后：

```bash
cd /root/autodl-tmp/DynFL-privacy
python - <<'PY'
import json
from pathlib import Path
src=Path('configs/dynfl_joint_calibration_safe.json')
c=json.loads(src.read_text(encoding='utf-8'))
s=c['system']
s['fl_first_split_on_demand']=True
s['edge_only_requires_fast_deadline']=False
s.pop('fast_client_deadlines', None)  # V370 清单是 fast deadlines 唯一真值
s['static_pair_admission_manifest']='configs/v370_trust_manifest.json'
c['output_root']='out/v370_joint_admission_smoke'
out=Path('configs/dynfl_joint_calibration_v370.json')
out.write_text(json.dumps(c, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print('WROTE',out)
PY
```

### 1 轮冒烟

```bash
DYNFL_EXPERIMENTAL_SAMPLE_MULTISTAGE=1 \
python experiments/run_paper_config.py \
  --config configs/dynfl_joint_calibration_v370.json \
  --seeds 42 --policies full_dynfl \
  --he-execution profiled \
  --max-new-rounds 1 \
  --output-root out/v370_joint_admission_smoke
```

确认日志出现 `V370 static pair admission: ENABLED`，并核对结果的可行候选、跳过原因及模式分布。仅有该日志不证明全部业务规则或 DP 安全完成验证。

## 回滚

```bash
cp experiments/run_paper_config.py.before_v370_wiring experiments/run_paper_config.py
cp experiments/run_fmnist_lenet5.py.before_v370_wiring experiments/run_fmnist_lenet5.py
```

回滚只撤销该接线包对入口的修改；原 V370 `dynfed/selection.py` 补丁不受影响。
