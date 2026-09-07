from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dynfed.version import CURRENT_EXECUTION_REVISION, CURRENT_UPDATE_PARAMETER_SCOPE

POLICY_LABELS = {
    "ours": "Ours",
    "individual_optimal": "Individual Optimal",
    "no_protection": "No Protection",
    "random": "Random",
    "fixed_fedavg": "Fixed FedAvg",
    "fixed_splitfed": "Fixed SplitFed",
    "fixed_hfl": "Fixed HFL",
    "ours_no_omega": "No Error Cost Estimate",
    "ours_fixed_liieiiic": "Fixed LIIEIIIC",
    "nsga2": "NSGA II",
}

POLICY_COLORS = {
    "ours": "#2ca02c",
    "individual_optimal": "#1f77b4",
    "no_protection": "#ff7f0e",
    "random": "#d62728",
    "fixed_fedavg": "#9467bd",
    "fixed_splitfed": "#8c564b",
    "fixed_hfl": "#17becf",
    "ours_no_omega": "#e39d26",
    "ours_fixed_liieiiic": "#7f7f7f",
    "nsga2": "#bcbd22",
}

MODE_ORDER = ("LIE", "LIC", "LIIE", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC")
MODE_COLORS = {
    "LIE": "#4e79a7",
    "LIC": "#f28e2b",
    "LIIE": "#59a14f",
    "LIIC": "#e15759",
    "LIEIIC": "#b07aa1",
    "LIEIIIC": "#76b7b2",
    "LIIEIIIC": "#edc948",
}

SUMMARY_METRICS = (
    "final_test_accuracy",
    "best_test_accuracy",
    "avg_last_10_accuracy",
    "total_logical_time",
    "accounted_system_time_sec",
    "total_communication_volume",
    "mean_effective_clients",
    "max_feature_epsilon",
    "max_update_epsilon",
    "he_wall_time_sec",
    "he_worker_cpu_time_sec",
    "he_ciphertext_bytes",
    "he_max_abs_error",
    "total_selection_wall_time_sec",
    "total_training_wall_time_sec",
    "end_to_end_wall_time_sec",
    "wall_time_sec",
)


@dataclass(frozen=True)
class SourceRun:
    seed: int
    policy: str
    run_dir: Path
    policy_dir: Path
    config: dict[str, Any]
    summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate compatible federated learning runs across random seeds."
    )
    parser.add_argument("--root", type=Path, default=ROOT / "out")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "ours",
            "individual_optimal",
            "no_protection",
            "random",
            "fixed_fedavg",
            "fixed_splitfed",
            "fixed_hfl",
            "nsga2",
        ],
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="resnet18_pretrained")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--edges", type=int, default=10)
    parser.add_argument("--selection-period", type=int, default=1)
    parser.add_argument("--privacy-budget", type=float, default=8.0)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-figure", type=Path)
    parser.add_argument("--paper-reconfiguration-figure", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    policies = list(dict.fromkeys(str(policy) for policy in args.policies))
    sources = discover_sources(
        root=args.root,
        seeds=seeds,
        policies=policies,
        dataset=args.dataset,
        model=args.model,
        rounds=args.rounds,
        clients=args.clients,
        edges=args.edges,
        selection_period=args.selection_period,
        privacy_budget=args.privacy_budget,
    )
    validate_sources(sources, seeds, policies)

    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = aggregate_summaries(sources, seeds, policies)
    write_csv(output_dir / "summary_statistics.csv", summary_rows)
    if "ours" in policies and len(policies) > 1:
        write_csv(
            output_dir / "paired_comparisons.csv",
            paired_comparisons(sources, seeds, policies),
        )

    round_rows, _ = aggregate_rounds(
        sources,
        seeds,
        policies,
        rounds=args.rounds,
    )
    write_csv(output_dir / "round_statistics.csv", round_rows)

    time_rows, plotted, common_time_horizon = aggregate_time(
        sources,
        seeds,
        policies,
        rounds=args.rounds,
    )
    write_csv(output_dir / "time_statistics.csv", time_rows)

    figure_path = figure_dir / "accuracy_across_seeds.png"
    plot_accuracy_over_time(
        plotted,
        policies,
        figure_path,
        tail_fraction=args.tail_fraction,
    )
    if args.paper_figure:
        paper_figure = args.paper_figure.resolve()
        paper_figure.parent.mkdir(parents=True, exist_ok=True)
        paper_figure.write_bytes(figure_path.read_bytes())

    if "ours" in policies:
        reconfiguration_rows, reconfiguration_summary = aggregate_reconfiguration(
            sources,
            seeds,
            rounds=args.rounds,
            clients=args.clients,
        )
        write_csv(output_dir / "reconfiguration_statistics.csv", reconfiguration_rows)
        write_csv(output_dir / "reconfiguration_summary.csv", reconfiguration_summary)
        reconfiguration_figure = figure_dir / "reconfiguration_trace.png"
        plot_reconfiguration(reconfiguration_rows, reconfiguration_figure)
        if args.paper_reconfiguration_figure:
            paper_reconfiguration_figure = args.paper_reconfiguration_figure.resolve()
            paper_reconfiguration_figure.parent.mkdir(parents=True, exist_ok=True)
            paper_reconfiguration_figure.write_bytes(reconfiguration_figure.read_bytes())

    manifest = {
        "seeds": seeds,
        "policies": policies,
        "dataset": args.dataset,
        "model": args.model,
        "rounds": args.rounds,
        "clients": args.clients,
        "edges": args.edges,
        "selection_period": args.selection_period,
        "privacy_budget": args.privacy_budget,
        "curve_processing": "raw unsmoothed test accuracy with linear time interpolation",
        "tail_fraction": args.tail_fraction,
        "common_system_time_horizon_sec": common_time_horizon,
        "sources": [
            {
                "seed": source.seed,
                "policy": source.policy,
                "run_dir": str(source.run_dir),
                "policy_dir": str(source.policy_dir),
                "execution_revision": source.config["training"].get("execution_revision"),
            }
            for source in sorted(sources.values(), key=lambda item: (item.seed, item.policy))
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Aggregated {len(seeds)} seeds and {len(policies)} policies")
    print(output_dir / "summary_statistics.csv")
    print(output_dir / "round_statistics.csv")
    print(output_dir / "time_statistics.csv")
    print(figure_path)


def discover_sources(
    *,
    root: Path,
    seeds: list[int],
    policies: list[str],
    dataset: str,
    model: str,
    rounds: int,
    clients: int,
    edges: int,
    selection_period: int,
    privacy_budget: float,
) -> dict[tuple[int, str], SourceRun]:
    selected: dict[tuple[int, str], SourceRun] = {}
    for config_path in root.rglob("config.json"):
        if any(part in {"merged_runs", "multiseed_runs"} for part in config_path.parts):
            continue
        try:
            config = read_json(config_path)
            selection = config["selection"]
            training = config["training"]
            resolved_privacy = config.get("resolved_privacy", {})
        except (OSError, ValueError, KeyError, TypeError):
            continue
        seed = int(selection.get("seed", -1))
        if seed not in seeds:
            continue
        if str(training.get("dataset_name")) != dataset:
            continue
        if str(training.get("model_name")) != model:
            continue
        if str(training.get("execution_revision")) != CURRENT_EXECUTION_REVISION:
            continue
        if training.get("update_parameter_scope") != CURRENT_UPDATE_PARAMETER_SCOPE:
            continue
        if int(selection.get("rounds", -1)) != rounds:
            continue
        if int(selection.get("num_clients", -1)) != clients:
            continue
        if int(selection.get("num_edges", -1)) != edges:
            continue
        if int(training.get("selection_period", -1)) != selection_period:
            continue
        feature_budget = selection.get("dp_feature_epsilon_budget")
        if feature_budget is None:
            feature_budget = resolved_privacy.get(
                "feature_budget", selection.get("initial_epsilon", -1.0)
            )
        update_budget = selection.get("dp_update_epsilon_budget")
        if update_budget is None:
            update_budget = resolved_privacy.get(
                "update_budget", selection.get("initial_epsilon", -1.0)
            )
        feature_enabled = bool(resolved_privacy.get("feature_dp_enabled", True))
        if feature_enabled and float(feature_budget) != float(privacy_budget):
            continue
        if float(update_budget) != float(privacy_budget):
            continue

        run_dir = config_path.parent
        for policy in policies:
            policy_dir = run_dir / policy
            summary_path = policy_dir / "summary.json"
            metrics_path = policy_dir / "round_metrics.csv"
            if not summary_path.exists() or not metrics_path.exists():
                continue
            summary = read_json(summary_path)
            if int(summary.get("rounds", -1)) != rounds:
                continue
            key = (seed, policy)
            source = SourceRun(seed, policy, run_dir, policy_dir, config, summary)
            previous = selected.get(key)
            if previous is None or summary_path.stat().st_mtime > (
                previous.policy_dir / "summary.json"
            ).stat().st_mtime:
                selected[key] = source
    return selected


def validate_sources(
    sources: dict[tuple[int, str], SourceRun],
    seeds: list[int],
    policies: list[str],
) -> None:
    missing = [
        f"seed={seed}, policy={policy}"
        for seed in seeds
        for policy in policies
        if (seed, policy) not in sources
    ]
    if missing:
        raise RuntimeError("Missing completed runs for " + "; ".join(missing))

    reference: dict[str, Any] | None = None
    reference_name = ""
    for key in sorted(sources):
        source = sources[key]
        normalized = normalized_config(source.config)
        if reference is None:
            reference = normalized
            reference_name = f"seed={source.seed}, policy={source.policy}"
            continue
        if normalized != reference:
            raise RuntimeError(
                "Configuration mismatch between "
                f"{reference_name} and seed={source.seed}, policy={source.policy}"
            )


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    selection = normalized.get("selection", {})
    selection.pop("seed", None)
    selection.pop("output_dir", None)
    normalized.pop("policies", None)
    normalized.pop("resume_from_run", None)
    return normalized


def aggregate_summaries(
    sources: dict[tuple[int, str], SourceRun],
    seeds: list[int],
    policies: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        row: dict[str, Any] = {
            "policy": policy,
            "label": POLICY_LABELS.get(policy, policy),
            "seeds": ";".join(str(seed) for seed in seeds),
            "n_seeds": len(seeds),
        }
        for metric in SUMMARY_METRICS:
            values = [float(sources[(seed, policy)].summary[metric]) for seed in seeds]
            mean = statistics.fmean(values)
            half_width = confidence_half_width(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = sample_std(values)
            row[f"{metric}_ci95_low"] = mean - half_width
            row[f"{metric}_ci95_high"] = mean + half_width
        rows.append(row)
    return rows


def paired_comparisons(
    sources: dict[tuple[int, str], SourceRun],
    seeds: list[int],
    policies: list[str],
) -> list[dict[str, Any]]:
    comparison_metrics = (
        "final_test_accuracy",
        "avg_last_10_accuracy",
        "total_communication_volume",
        "accounted_system_time_sec",
    )
    rows: list[dict[str, Any]] = []
    for baseline in policies:
        if baseline == "ours":
            continue
        for metric in comparison_metrics:
            ours = [float(sources[(seed, "ours")].summary[metric]) for seed in seeds]
            reference = [
                float(sources[(seed, baseline)].summary[metric]) for seed in seeds
            ]
            differences = [left - right for left, right in zip(ours, reference)]
            mean_difference = statistics.fmean(differences)
            std_difference = sample_std(differences)
            half_width = confidence_half_width(differences)
            t_result = stats.ttest_rel(ours, reference) if len(seeds) > 1 else None
            if len(seeds) > 1 and any(abs(value) > 1e-15 for value in differences):
                wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
            else:
                wilcoxon_p = 1.0
            rows.append(
                {
                    "baseline": baseline,
                    "baseline_label": POLICY_LABELS.get(baseline, baseline),
                    "metric": metric,
                    "n_pairs": len(seeds),
                    "ours_minus_baseline_mean": mean_difference,
                    "paired_difference_std": std_difference,
                    "paired_ci95_low": mean_difference - half_width,
                    "paired_ci95_high": mean_difference + half_width,
                    "cohen_dz": (
                        mean_difference / std_difference
                        if std_difference > 1e-15
                        else 0.0
                    ),
                    "paired_t_pvalue": (
                        float(t_result.pvalue) if t_result is not None else 1.0
                    ),
                    "wilcoxon_pvalue": wilcoxon_p,
                }
            )
    return rows


def aggregate_rounds(
    sources: dict[tuple[int, str], SourceRun],
    seeds: list[int],
    policies: list[str],
    *,
    rounds: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float]]]]:
    rows: list[dict[str, Any]] = []
    plotted: dict[str, dict[str, list[float]]] = {}
    for policy in policies:
        raw_by_seed: list[list[float]] = []
        for seed in seeds:
            metrics = read_csv(sources[(seed, policy)].policy_dir / "round_metrics.csv")
            if len(metrics) != rounds:
                raise RuntimeError(
                    f"Expected {rounds} rows for seed={seed}, policy={policy}, got {len(metrics)}"
                )
            values = [float(item["test_accuracy"]) for item in metrics]
            raw_by_seed.append(values)

        policy_means: list[float] = []
        policy_stds: list[float] = []
        policy_ci95: list[float] = []
        for index in range(rounds):
            raw_values = [values[index] for values in raw_by_seed]
            raw_mean = statistics.fmean(raw_values)
            raw_std = sample_std(raw_values)
            policy_means.append(raw_mean)
            policy_stds.append(raw_std)
            raw_ci95 = confidence_half_width(raw_values)
            policy_ci95.append(raw_ci95)
            rows.append(
                {
                    "policy": policy,
                    "round": index + 1,
                    "test_accuracy_mean": raw_mean,
                    "test_accuracy_std": raw_std,
                    "test_accuracy_ci95_low": raw_mean - raw_ci95,
                    "test_accuracy_ci95_high": raw_mean + raw_ci95,
                }
            )
        plotted[policy] = {
            "mean": policy_means,
            "std": policy_stds,
            "ci95": policy_ci95,
        }
    return rows, plotted


def aggregate_time(
    sources: dict[tuple[int, str], SourceRun],
    seeds: list[int],
    policies: list[str],
    *,
    rounds: int,
    grid_points: int = 200,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, list[float]]],
    float,
]:
    run_series: dict[tuple[int, str], tuple[list[float], list[float]]] = {}
    first_times: list[float] = []
    final_times: list[float] = []
    for policy in policies:
        for seed in seeds:
            metrics = read_csv(sources[(seed, policy)].policy_dir / "round_metrics.csv")
            if len(metrics) != rounds:
                raise RuntimeError(
                    f"Expected {rounds} rows for seed={seed}, policy={policy}, got {len(metrics)}"
                )
            if any("accounted_system_time_sec" not in item for item in metrics):
                raise RuntimeError(
                    "The selected run predates accounted system timing for "
                    f"seed={seed}, policy={policy}"
                )
            times = [float(item["accounted_system_time_sec"]) for item in metrics]
            if any(right <= left for left, right in zip(times, times[1:])):
                raise RuntimeError(
                    f"System time is not strictly increasing for seed={seed}, policy={policy}"
                )
            values = [float(item["test_accuracy"]) for item in metrics]
            run_series[(seed, policy)] = (times, values)
            first_times.append(times[0])
            final_times.append(times[-1])

    common_start = max(first_times)
    common_horizon = min(final_times)
    if common_horizon < common_start:
        raise RuntimeError("Completed runs do not share a positive system time interval")
    if abs(common_horizon - common_start) <= 1e-12:
        grid = [common_horizon]
    else:
        grid_points = max(2, int(grid_points))
        step = (common_horizon - common_start) / float(grid_points - 1)
        grid = [common_start + step * index for index in range(grid_points)]

    rows: list[dict[str, Any]] = []
    plotted: dict[str, dict[str, list[float]]] = {}
    for policy in policies:
        interpolated_by_seed = [
            interpolate_series(*run_series[(seed, policy)], grid) for seed in seeds
        ]
        means: list[float] = []
        stds: list[float] = []
        ci95_values: list[float] = []
        for index, wall_time in enumerate(grid):
            values = [series[index] for series in interpolated_by_seed]
            mean = statistics.fmean(values)
            std = sample_std(values)
            means.append(mean)
            stds.append(std)
            ci95 = confidence_half_width(values)
            ci95_values.append(ci95)
            rows.append(
                {
                    "policy": policy,
                    "accounted_system_time_sec": wall_time,
                    "test_accuracy_mean": mean,
                    "test_accuracy_std": std,
                    "test_accuracy_ci95_low": mean - ci95,
                    "test_accuracy_ci95_high": mean + ci95,
                }
            )
        plotted[policy] = {
            "time": grid,
            "mean": means,
            "std": stds,
            "ci95": ci95_values,
        }
    return rows, plotted, common_horizon


def interpolate_series(
    times: list[float], values: list[float], grid: list[float]
) -> list[float]:
    result: list[float] = []
    right = 1
    for target in grid:
        while right < len(times) and times[right] < target:
            right += 1
        if right >= len(times):
            result.append(values[-1])
            continue
        left = right - 1
        width = times[right] - times[left]
        ratio = 0.0 if width <= 0.0 else (target - times[left]) / width
        result.append(values[left] + ratio * (values[right] - values[left]))
    return result


def aggregate_reconfiguration(
    sources: dict[tuple[int, str], SourceRun],
    seeds: list[int],
    *,
    rounds: int,
    clients: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mode_fraction_by_seed: dict[int, dict[int, dict[str, float]]] = {}
    mechanism_fraction_by_seed: dict[int, dict[int, dict[str, float]]] = {}
    switch_fraction_by_seed: dict[int, dict[int, float]] = {}
    per_seed_summary: list[dict[str, float]] = []

    for seed in seeds:
        source = sources[(seed, "ours")]
        decisions = read_csv(source.policy_dir / "client_decisions.csv")
        metrics = read_csv(source.policy_dir / "round_metrics.csv")
        by_round: dict[int, list[dict[str, str]]] = {index: [] for index in range(rounds)}
        for decision in decisions:
            round_index = int(decision["round"])
            if 0 <= round_index < rounds:
                by_round[round_index].append(decision)

        mode_fraction_by_seed[seed] = {}
        switch_fraction_by_seed[seed] = {}
        previous_modes: dict[int, str] = {}
        switched = 0
        compared = 0
        distinct_modes: set[str] = set()
        changed_rounds = 0
        for round_index in range(rounds):
            rows = by_round[round_index]
            denominator = max(1, len(rows))
            counts = {mode: 0 for mode in MODE_ORDER}
            current_modes: dict[int, str] = {}
            for row in rows:
                mode = row["mode"]
                client_id = int(row["client_id"])
                current_modes[client_id] = mode
                if mode in counts:
                    counts[mode] += 1
                    distinct_modes.add(mode)
            mode_fraction_by_seed[seed][round_index] = {
                mode: counts[mode] / denominator for mode in MODE_ORDER
            }
            round_compared = 0
            round_switched = 0
            for client_id, mode in current_modes.items():
                if client_id not in previous_modes:
                    continue
                round_compared += 1
                round_switched += int(previous_modes[client_id] != mode)
            if round_switched > 0:
                changed_rounds += 1
            compared += round_compared
            switched += round_switched
            switch_fraction_by_seed[seed][round_index] = (
                round_switched / round_compared if round_compared else 0.0
            )
            previous_modes = current_modes

        mechanism_fraction_by_seed[seed] = {}
        for round_index, metric in enumerate(metrics):
            denominator = max(1, clients)
            mechanism_fraction_by_seed[seed][round_index] = {
                "feature_dp": float(metric["num_feature_dp_clients"]) / denominator,
                "update_dp": float(metric["num_update_dp_clients"]) / denominator,
                "he": float(metric["num_he_clients"]) / denominator,
            }
        per_seed_summary.append(
            {
                "distinct_modes": float(len(distinct_modes)),
                "client_mode_switch_rate": switched / compared if compared else 0.0,
                "round_profile_change_rate": changed_rounds / max(1, rounds - 1),
            }
        )

    rows: list[dict[str, Any]] = []
    for round_index in range(rounds):
        row: dict[str, Any] = {"round": round_index + 1}
        for mode in MODE_ORDER:
            values = [mode_fraction_by_seed[seed][round_index][mode] for seed in seeds]
            _add_mean_ci(row, f"mode_{mode}", values)
        for mechanism in ("feature_dp", "update_dp", "he"):
            values = [
                mechanism_fraction_by_seed[seed][round_index][mechanism]
                for seed in seeds
            ]
            _add_mean_ci(row, mechanism, values)
        switch_values = [switch_fraction_by_seed[seed][round_index] for seed in seeds]
        _add_mean_ci(row, "mode_switch", switch_values)
        rows.append(row)

    summary: list[dict[str, Any]] = []
    for metric in ("distinct_modes", "client_mode_switch_rate", "round_profile_change_rate"):
        values = [row[metric] for row in per_seed_summary]
        half_width = confidence_half_width(values)
        mean = statistics.fmean(values)
        summary.append(
            {
                "metric": metric,
                "n_seeds": len(seeds),
                "mean": mean,
                "std": sample_std(values),
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            }
        )
    return rows, summary


def _add_mean_ci(row: dict[str, Any], name: str, values: list[float]) -> None:
    mean = statistics.fmean(values)
    half_width = confidence_half_width(values)
    row[f"{name}_mean"] = mean
    row[f"{name}_ci95_low"] = mean - half_width
    row[f"{name}_ci95_high"] = mean + half_width


def plot_reconfiguration(rows: list[dict[str, Any]], output: Path) -> None:
    xs = [int(row["round"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.45), sharex=True)
    mode_values = [
        [float(row[f"mode_{mode}_mean"]) for row in rows]
        for mode in MODE_ORDER
    ]
    axes[0].stackplot(
        xs,
        mode_values,
        labels=MODE_ORDER,
        colors=[MODE_COLORS[mode] for mode in MODE_ORDER],
        alpha=0.9,
        linewidth=0,
    )
    for name, label, color in (
        ("update_dp", "Update DP", "#e15759"),
        ("he", "HE", "#59a14f"),
        ("mode_switch", "Mode switch", "#7f7f7f"),
    ):
        axes[1].plot(
            xs,
            [float(row[f"{name}_mean"]) for row in rows],
            label=label,
            color=color,
            linewidth=1.2,
        )
    axes[0].set_ylabel("Mode share (%)", fontsize=8.2)
    axes[1].set_ylabel("Client share (%)", fontsize=8.2)
    axes[1].set_xlabel("Communication round", fontsize=8.2)
    for axis in axes:
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.tick_params(labelsize=7.2, pad=2)
        axis.grid(True, alpha=0.22, linewidth=0.5)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=4,
        frameon=False,
        fontsize=6.2,
        columnspacing=0.65,
        handlelength=1.2,
    )
    axes[1].legend(frameon=False, fontsize=6.6, ncol=3)
    fig.tight_layout(pad=0.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_accuracy_over_time(
    plotted: dict[str, dict[str, list[float]]],
    policies: list[str],
    output: Path,
    *,
    tail_fraction: float,
    labels: dict[str, str] | None = None,
) -> None:
    # Size and typography are chosen for direct use in one journal column.
    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    for policy in policies:
        xs = plotted[policy]["time"]
        means = plotted[policy]["mean"]
        ci95_values = plotted[policy]["ci95"]
        color = POLICY_COLORS.get(policy)
        ax.plot(
            xs,
            means,
            linewidth=1.7,
            color=color,
            label=(labels or POLICY_LABELS).get(policy, policy),
        )
        ax.fill_between(
            xs,
            [max(0.0, mean - ci95) for mean, ci95 in zip(means, ci95_values)],
            [min(1.0, mean + ci95) for mean, ci95 in zip(means, ci95_values)],
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    xs = next(iter(plotted.values()))["time"]
    tail_fraction = min(0.5, max(0.1, float(tail_fraction)))
    tail_start_time = xs[-1] - tail_fraction * (xs[-1] - xs[0])
    tail_indices = [index for index, value in enumerate(xs) if value >= tail_start_time]
    if len(tail_indices) > 1 and xs[-1] > xs[0]:
        axins = ax.inset_axes([0.54, 0.09, 0.43, 0.32])
        tail_values: list[float] = []
        for policy in policies:
            means = plotted[policy]["mean"]
            policy_xs = plotted[policy]["time"]
            tail_xs = [policy_xs[index] for index in tail_indices]
            tail_means = [means[index] for index in tail_indices]
            tail_values.extend(tail_means)
            axins.plot(
                tail_xs,
                tail_means,
                linewidth=1.25,
                color=POLICY_COLORS.get(policy),
            )
        margin = max(0.01, (max(tail_values) - min(tail_values)) * 0.12)
        axins.set_xlim(tail_start_time, xs[-1])
        axins.set_ylim(max(0.0, min(tail_values) - margin), min(1.0, max(tail_values) + margin))
        axins.grid(True, alpha=0.24, linewidth=0.55)
        axins.xaxis.set_major_locator(MaxNLocator(nbins=3))
        axins.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axins.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axins.tick_params(labelsize=6.3, pad=1.0, length=2.0)
        for spine in axins.spines.values():
            spine.set_edgecolor("#555555")
            spine.set_linewidth(0.9)

    ax.set_xlabel("Accounted system time (s)", fontsize=8.5)
    ax.set_ylabel("Test accuracy (%)", fontsize=8.5)
    if xs[-1] > xs[0]:
        ax.set_xlim(xs[0], xs[-1])
    else:
        margin = max(0.1, abs(xs[0]) * 0.05)
        ax.set_xlim(xs[0] - margin, xs[0] + margin)
    ax.set_ylim(0.0, 0.9)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=7.5, pad=2.0)
    ax.grid(True, alpha=0.25, linewidth=0.55)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=7.0,
        ncol=2,
        handlelength=1.7,
        columnspacing=0.8,
        labelspacing=0.35,
        borderaxespad=0.35,
    )
    fig.tight_layout(pad=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def confidence_half_width(values: list[float], confidence: float = 0.95) -> float:
    if len(values) <= 1:
        return 0.0
    standard_error = sample_std(values) / len(values) ** 0.5
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    return critical * standard_error


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV file {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
