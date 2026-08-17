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
    "ours": "DynFedPrivacy",
    "individual_optimal": "Individual-Optimal",
    "fixed_dp": "Fixed-DP",
    "privacy_only": "Privacy-Only",
    "no_protection": "No-Protection",
    "random": "Random",
}
POLICY_ORDER_50 = ["ours", "fixed_dp", "privacy_only", "no_protection", "random"]
POLICY_ORDER_100 = ["ours", "privacy_only", "random"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild paper figures from latest LeNet5 experiment outputs.")
    parser.add_argument("--root-50", default="out/fmnist_lenet5_paper_50r")
    parser.add_argument("--root-100", default="out/fmnist_lenet5_paper_100r_ep3")
    parser.add_argument("--fallback", action="store_true", help="Use older known outputs if unified roots are absent.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Centered rolling window for accuracy smoothing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    root_50 = _latest_complete_run(ROOT / args.root_50, POLICY_ORDER_50)
    root_100 = _latest_complete_run(ROOT / args.root_100, POLICY_ORDER_100)

    if args.fallback:
        root_50 = root_50 or _fallback_50_root()
        root_100 = root_100 or _fallback_100_root()

    outputs: list[Path] = []
    if root_50:
        outputs.extend(_plot_50(root_50, args.smooth_window))
        _write_paper_summary(root_50, POLICY_ORDER_50, ROOT / "out" / "paper_50r_summary.csv")
    else:
        print("[WARN] no complete 50-round run found")

    if root_100:
        outputs.extend(_plot_100(root_100, args.smooth_window))
        _write_paper_summary(root_100, POLICY_ORDER_100, ROOT / "out" / "paper_100r_ep3_summary.csv")
    else:
        print("[WARN] no complete 100-round run found")

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


def _fallback_50_root() -> Path | None:
    current = _latest_complete_run(ROOT / "out" / "fmnist_lenet5_routeA_current", ["ours", "fixed_dp", "privacy_only", "no_protection"])
    random_root = _latest_complete_run(ROOT / "out" / "fmnist_lenet5_routeA_random50", ["random"])
    if not current or not random_root:
        return None
    merged = ROOT / "out" / "paper_50r_merged_latest"
    merged.mkdir(parents=True, exist_ok=True)
    for policy in ["ours", "fixed_dp", "privacy_only", "no_protection"]:
        _write_pointer_files(current / policy, merged / policy)
    _write_pointer_files(random_root / "random", merged / "random")
    return merged


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


def _plot_50(root: Path, smooth_window: int) -> list[Path]:
    outputs = [
        FIG_DIR / "fmnist_lenet5_accuracy_convergence_with_random.png",
        FIG_DIR / "fmnist_lenet5_time_accuracy_with_random.png",
        FIG_DIR / "fmnist_lenet5_privacy_timeline_with_random.png",
        FIG_DIR / "fmnist_lenet5_mode_distribution_with_random.png",
    ]
    _plot_accuracy(root, POLICY_ORDER_50, outputs[0], title=None, smooth_window=smooth_window)
    _plot_time_accuracy(root, POLICY_ORDER_50, outputs[1], smooth_window=smooth_window)
    _plot_privacy(root, POLICY_ORDER_50, outputs[2])
    _plot_mode_distribution(root / "ours" / "client_decisions.csv", outputs[3])
    return outputs


def _plot_100(root: Path, smooth_window: int) -> list[Path]:
    output = FIG_DIR / "fmnist_lenet5_100r_ep3_accuracy.png"
    _plot_accuracy(root, POLICY_ORDER_100, output, title=None, smooth_window=smooth_window)
    return [output]


def _plot_accuracy(root: Path, policies: list[str], output: Path, title: str | None, smooth_window: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for policy in policies:
        rows = _read_csv(root / policy / "round_metrics.csv")
        xs = [int(row["round"]) + 1 for row in rows]
        ys = [float(row["test_accuracy"]) for row in rows]
        smooth, spread = _rolling_mean_std(ys, smooth_window)
        (line,) = ax.plot(xs, smooth, linewidth=2.2, label=POLICY_LABELS.get(policy, policy))
        color = line.get_color()
        lower = [max(0.0, mean - std) for mean, std in zip(smooth, spread)]
        upper = [min(1.0, mean + std) for mean, std in zip(smooth, spread)]
        ax.fill_between(xs, lower, upper, color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Test accuracy (rolling mean, w={smooth_window})")
    ax.set_ylim(0.0, 0.9)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.02,
        "Shaded band: rolling std within one seed",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_time_accuracy(root: Path, policies: list[str], output: Path, smooth_window: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for policy in policies:
        rows = _read_csv(root / policy / "round_metrics.csv")
        xs = [float(row["logical_time"]) for row in rows]
        ys = [float(row["test_accuracy"]) for row in rows]
        smooth, spread = _rolling_mean_std(ys, smooth_window)
        (line,) = ax.plot(xs, smooth, linewidth=2.2, label=POLICY_LABELS.get(policy, policy))
        color = line.get_color()
        lower = [max(0.0, mean - std) for mean, std in zip(smooth, spread)]
        upper = [min(1.0, mean + std) for mean, std in zip(smooth, spread)]
        ax.fill_between(xs, lower, upper, color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("Logical time")
    ax.set_ylabel(f"Test accuracy (rolling mean, w={smooth_window})")
    ax.set_ylim(0.0, 0.9)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.02,
        "Shaded band: rolling std within one seed",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
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
                "total_logical_time": summary["total_logical_time"],
                "total_communication_volume": summary["total_communication_volume"],
                "total_epsilon_used": summary["total_epsilon_used"],
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


def _rolling_mean_std(values: list[float], window: int) -> tuple[list[float], list[float]]:
    window = max(1, int(window))
    radius = window // 2
    means: list[float] = []
    spreads: list[float] = []
    for idx in range(len(values)):
        left = max(0, idx - radius)
        right = min(len(values), idx + radius + 1)
        segment = values[left:right]
        mean = sum(segment) / max(len(segment), 1)
        var = sum((value - mean) ** 2 for value in segment) / max(len(segment), 1)
        means.append(mean)
        spreads.append(var ** 0.5)
    return means, spreads


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
