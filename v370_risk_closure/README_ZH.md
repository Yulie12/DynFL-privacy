# V370 风险硬约束收尾补丁

之前 `Candidate.feasible` 只使用 `feasible_resource/feasible_memory/feasible_privacy`，并未纳入 `feasible_risk`，导致 `risk=0.58`、风险上限为 `0.5` 时 `feasible=True`。这份补丁让 **V370 `strict_pair_admission=True`** 时风险成为硬约束，同时尽量避免改变原 V369/未开启 V370 的实验基线。它不调整风险数值计算、Pareto 评分、DP 或 HE。

在服务器项目根目录执行：

```bash
unzip -o v370_risk_closure.zip
python v370_risk_closure/fix_v370_risk_gate.py --check
python v370_risk_closure/fix_v370_risk_gate.py --apply
cp v370_risk_closure/test_v370_risk_gate.py tests/
python -m compileall -q dynfed/selection.py
python -m unittest discover -s tests -p 'test_v370_*.py' -v
git diff --check
```

预期 14 项原测试 + 6 项风险回归共 20 项通过。严格准入下风险违规的候选 `feasible=False`，而旧模式保持原 `feasible` 语义。

重要：本补丁还修复 strict V370 下 `choose_candidate` 和 Pareto 空池回退，防止把风险违规候选重新放进候选池；旧模式行为保持不变。本补丁不代表其他约束和所有训练路径已完成独立证明。

回滚：`cp dynfed/selection.py.before_v370_risk_gate dynfed/selection.py`。
