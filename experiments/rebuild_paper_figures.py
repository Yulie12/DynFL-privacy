from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "Privacy_Utility_Tradeoff__20260525" / "figures"

POLICY_LABELS = {
    "ours": "Ours",
    "individual_optimal": "Individual Optimal",
    "fixed_dp": "Fixed DP",
    "privacy_only": "Privacy Only",
    "no_protection": "No Protection",
    "random": "Random",
}
POLICY_ORDER_MAIN = ["ours", "fixed_dp", "privacy_only", "no_protection", "random"]
POLICY_ORDER_CORE = ["ours", "privacy_only", "random"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild paper figures from latest LeNet5 experiment outputs.")
    parser.add_argument("--root-main", default="out/fmnist_lenet5_paper_100r")
    parser.add_argument("--root-core", default="out/fmnist_lenet5_paper_100r_ep3")
    parser.add_argument("--fallback", action="store_true", help="Use older known outputs if unified roots are absent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    root_main = _latest_complete_run(ROOT / args.root_main, POLICY_ORDER_MAIN)
    root_core = _latest_complete_run(ROOT / args.root_core, POLICY_ORDER_CORE)

    if args.fallback:
        root_core = root_core or _fallback_100_root()

    outputs: list[Path] = []
    if root_main:
        outputs.extend(_plot_main(root_main))
        _write_paper_summary(root_main, POLICY_ORDER_MAIN, ROOT / "out" / "paper_100r_summary.csv")
    else:
        print("[WARN] no complete 100-round main run found")

    if root_core:
        outputs.extend(_plot_100_core(root_core))
        _write_paper_summary(root_core, POLICY_ORDER_CORE, ROOT / "out" / "paper_100r_ep3_summary.csv")
    else:
        print("[WARN] no complete 100-round three-policy diagnostic found")

    print("[OK] rebuilt figures:")
    for path in outputs:
        print(path)


def _latest_complete_run(root: Path, policies: list[str]) -> Path | None:
    if root.exists() and all((root / policy / "summary.json").exists() for policy in policies):
        return root
    if not root.exists():
        return None
    runs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        if all((run / policy / "summary.json").exists() for policy in policies):
            return run
    return None


def _fallback_100_root() -> Path | None:
    current = _latest_complete_run(ROOT / "out" / "fmnist_lenet5_routeA_100r_ep3", ["ours", "privacy_only"])
    random_root = _latest_complete_run(ROOT / "out" / "fmnist_lenet5_tex_100r_ep3_random", ["random"])
    if not current or not random_root:
        return None
    merged = ROOT / "out" / "paper_100r_ep3_merged_latest"
    merged.mkdir(parents=True, exist_ok=True)
    for policy in ["ours", "privacy_only"]:
        _write_pointer_files(current / policy, merged / policy)
    _write_pointer_files(random_root / "random", merged / "random")
    return merged


def _write_pointer_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["round_metrics.csv", "client_decisions.csv", "summary.json"]:
        target = dst / name
        data = (src / name).read_bytes()
        target.write_bytes(data)


def _plot_main(root: Path) -> list[Path]:
    outputs = [
        FIG_DIR / "fmnist_lenet5_accuracy_convergence_with_random.png",
        FIG_DIR / "fmnist_lenet5_time_accuracy_with_random.png",
        FIG_DIR / "fmnist_lenet5_privacy_timeline_with_random.png",
        FIG_DIR / "fmnist_lenet5_mode_distribution_with_random.png",
    ]
    _plot_accuracy(root, POLICY_ORDER_MAIN, outputs[0], title=None)
    _plot_time_accuracy(root, POLICY_ORDER_MAIN, outputs[1])
    _plot_privacy(root, POLICY_ORDER_MAIN, outputs[2])
    _plot_mode_distribution(root / "ours" / "client_decisions.csv", outputs[3])
    return outputs


def _plot_100_core(root: Path) -> list[Path]:
    output = FIG_DIR / "fmnist_lenet5_100r_ep3_accuracy.png"
    _plot_accuracy(root, POLICY_ORDER_CORE, output, title=None)
    return [output]


def _plot_accuracy(root: Path, policies: list[str], output: Path, title: str | None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for policy in policies:
        rows = _read_csv(root / policy / "round_metrics.csv")
        xs = [int(row["round"]) + 1 for row in rows]
        ys = [float(row["test_accuracy"]) for row in rows]
        ax.plot(xs, ys, linewidth=1.6, label=POLICY_LABELS.get(policy, policy))
    ax.set_xlabel("Round")
    ax.set_ylabel("Raw test accuracy")
    ax.set_ylim(0.0, 0.9)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_time_accuracy(root: Path, policies: list[str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for policy in policies:
        rows = _read_csv(root / policy / "round_metrics.csv")
        if any("cumulative_wall_time_sec" not in row for row in rows):
            raise RuntimeError("Paper curves require cumulative wall time metrics")
        xs = [float(row["cumulative_wall_time_sec"]) for row in rows]
        ys = [float(row["test_accuracy"]) for row in rows]
        ax.plot(xs, ys, linewidth=1.6, label=POLICY_LABELS.get(policy, policy))
    ax.set_xlabel("Cumulative measured wall time")
    ax.set_ylabel("Raw test accuracy")
    ax.set_ylim(0.0, 0.9)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_privacy(root: Path, policies: list[str], output: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for policy in policies:
        rows = _read_csv(root / policy / "client_decisions.csv")
        by_round: dict[int, float] = {}
        for row in rows:
            r = int(row["round"])
            by_round[r] = by_round.get(r, 0.0) + float(row["epsilon_used"])
        xs = sorted(by_round)
        ys = [by_round[x] for x in xs]
        cumulative = []
        total = 0.0
        for value in ys:
            total += value
            cumulative.append(total)
        label = POLICY_LABELS.get(policy, policy)
        ax1.plot([x + 1 for x in xs], ys, linewidth=1.8, label=label)
        ax2.plot([x + 1 for x in xs], cumulative, linewidth=1.8, label=label)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Round epsilon")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Cumulative epsilon")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_mode_distribution(decision_path: Path, output: Path) -> None:
    rows = _read_csv(decision_path)
    counts: dict[str, int] = {}
    for row in rows:
        mode = row["mode"]
        counts[mode] = counts.get(mode, 0) + 1
    modes = sorted(counts)
    values = [counts[mode] for mode in modes]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(modes, values)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Client-round decisions")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _write_paper_summary(root: Path, policies: list[str], output: Path) -> None:
    rows = []
    for policy in policies:
        summary = _read_json(root / policy / "summary.json")
        rows.append(
            {
                "policy": POLICY_LABELS.get(policy, policy),
                "final_test_accuracy": summary["final_test_accuracy"],
                "best_test_accuracy": summary["best_test_accuracy"],
                "end_to_end_wall_time_sec": summary["end_to_end_wall_time_sec"],
                "total_communication_volume": summary["total_communication_volume"],
                "feature_epsilon": summary["max_feature_epsilon"],
                "update_epsilon": summary["max_update_epsilon"],
                "mode_distribution": summary.get("mode_distribution", ""),
                "output_dir": summary.get("output_dir", str(root / policy)),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
