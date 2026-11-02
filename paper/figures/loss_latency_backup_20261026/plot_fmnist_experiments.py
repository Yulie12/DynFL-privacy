#!/usr/bin/env python3
"""Generate the paper figures from the completed three-seed FMNIST campaign.

The experiment outputs are read only. Seed 42, 43, and 44 values are aligned
by strategy and round. Arithmetic means and sample standard deviations
(``ddof=1``) are written to ``paper/figures/three_seed_aggregate`` before the
figures are rendered. Categorical candidate selections are never averaged as
candidate indices: the three observed sequences are displayed separately.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_ROOT = ROOT / "out" / "fmnist_joint_jt_100r_3seed_20261022"
AGGREGATE_DIR = HERE / "three_seed_aggregate"
SEEDS = (42, 43, 44)
ACCURACY_ROUNDS = tuple(range(91, 101))
ACCURACY_METRIC = "last10_accuracy"

FIXED = ["fixed_P0", "fixed_P1", "fixed_P2", "fixed_P3", "fixed_P4"]
GREEDY = ["greedy_lambda025", "greedy_lambda050", "greedy_lambda075"]
STABILITY = ["hysteresis_lambda050", "periodic_lambda050"]
OBJECTIVE_BASELINES = ["max_J", "min_T"]
PURE = ["pure_FL_E", "pure_FL_EC", "pure_FL_C"]
ADAPTIVE = GREEDY + STABILITY + OBJECTIVE_BASELINES
STATIC = FIXED + PURE
STRATEGIES = FIXED + GREEDY + STABILITY + OBJECTIVE_BASELINES + PURE
CANDIDATES = ["P0", "P1", "P2", "P3", "P4"]

EXPECTED_SOURCE_HASH = "e447a45e156ce87c851892355ef535dc684f7b56b9f03412b832af4554b96988"
ACCEPTED_SOURCE_EXCEPTIONS = {(43, "fixed_P2"), (43, "fixed_P3"), (43, "fixed_P4")}

LABELS = {
    "fixed_P0": r"Fixed $P_0$",
    "fixed_P1": r"Fixed $P_1$",
    "fixed_P2": r"Fixed $P_2$",
    "fixed_P3": r"Fixed $P_3$",
    "fixed_P4": r"Fixed $P_4$",
    "greedy_lambda025": r"Greedy $\lambda=0.25$",
    "greedy_lambda050": r"Greedy $\lambda=0.50$",
    "greedy_lambda075": r"Greedy $\lambda=0.75$",
    "hysteresis_lambda050": r"Hysteresis $\lambda=0.50$",
    "periodic_lambda050": r"Periodic $\lambda=0.50$",
    "max_J": r"Maximum $J$",
    "min_T": r"Minimum $T$",
    "pure_FL_E": r"Pure $FL_E$",
    "pure_FL_EC": r"Pure $FL_{EC}$",
    "pure_FL_C": r"Pure $FL_C$",
}

SHORT_LABELS = {
    "fixed_P0": r"F $P_0$",
    "fixed_P1": r"F $P_1$",
    "fixed_P2": r"F $P_2$",
    "fixed_P3": r"F $P_3$",
    "fixed_P4": r"F $P_4$",
    "greedy_lambda025": r"G .25",
    "greedy_lambda050": r"G .50",
    "greedy_lambda075": r"G .75",
    "hysteresis_lambda050": "Hyst.",
    "periodic_lambda050": "Period.",
    "max_J": r"Max $J$",
    "min_T": r"Min $T$",
    "pure_FL_E": r"Pure $E$",
    "pure_FL_EC": r"Pure $EC$",
    "pure_FL_C": r"Pure $C$",
}

COLORS = {
    "fixed_P0": "#4D4D4D",
    "fixed_P1": "#D55E00",
    "fixed_P2": "#0072B2",
    "fixed_P3": "#009E73",
    "fixed_P4": "#CC79A7",
    "pure_FL_E": "#332288",
    "pure_FL_EC": "#117733",
    "pure_FL_C": "#AA4499",
    "greedy_lambda025": "#009E73",
    "greedy_lambda050": "#56B4E9",
    "greedy_lambda075": "#E69F00",
    "hysteresis_lambda050": "#AA4499",
    "periodic_lambda050": "#44AA99",
    "max_J": "#882255",
    "min_T": "#332288",
}

LINESTYLES = {
    "fixed_P0": "-",
    "fixed_P1": "--",
    "fixed_P2": "-.",
    "fixed_P3": ":",
    "fixed_P4": (0, (5, 1, 1, 1)),
    "greedy_lambda025": "-",
    "greedy_lambda050": "--",
    "greedy_lambda075": "-.",
    "hysteresis_lambda050": ":",
    "periodic_lambda050": (0, (5, 1, 1, 1)),
    "max_J": (0, (7, 2)),
    "min_T": (0, (1, 1)),
    "pure_FL_E": (0, (7, 2)),
    "pure_FL_EC": (0, (3, 1, 1, 1)),
    "pure_FL_C": (0, (1, 1)),
}

CANDIDATE_COLORS = {
    "P0": "#999999",
    "P1": "#D55E00",
    "P2": "#0072B2",
    "P3": "#009E73",
    "P4": "#CC79A7",
}

ROUND_NUMERIC_COLUMNS = [
    "test_loss",
    "test_accuracy",
    "system_time",
    "round_duration",
    "selection_sec",
    "predicted_j",
    "observed_j",
    "prediction_error",
    "communication_mb",
    "executed_local_steps",
    "mode_changes",
    "max_epsilon",
    "admitted_clients",
    "cloud_units",
    "pending_units",
    "acknowledged_clients",
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.8,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.0,
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color="#D6D6D6", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.pdf", facecolor="white")
    fig.savefig(HERE / f"{stem}.png", dpi=400, facecolor="white")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_columns(frame: pd.DataFrame, groups: list[str], columns: list[str]) -> pd.DataFrame:
    pieces = []
    for column in columns:
        working = frame[groups].copy()
        working[column] = pd.to_numeric(frame[column], errors="coerce")
        stats = working.groupby(groups, sort=False, dropna=False)[column].agg(["mean", "std", "count"])
        stats = stats.rename(
            columns={"mean": f"{column}_mean", "std": f"{column}_std", "count": f"{column}_n"}
        )
        pieces.append(stats)
    result = pd.concat(pieces, axis=1).reset_index()
    expected = frame.groupby(groups, sort=False, dropna=False).size().reset_index(name="group_size")
    return expected.merge(result, on=groups, how="left")


def load_runs() -> tuple[dict[tuple[int, str], pd.DataFrame], dict]:
    required = {"round", *ROUND_NUMERIC_COLUMNS}
    frames: dict[tuple[int, str], pd.DataFrame] = {}
    identities: list[dict] = []
    source_files: list[dict] = []

    for seed in SEEDS:
        seed_identities = []
        for strategy in STRATEGIES:
            directory = RUN_ROOT / f"seed_{seed}" / strategy
            metrics_path = directory / "round_metrics.csv"
            summary_path = directory / "summary.json"
            identity_path = directory / "run_identity.json"
            for path in (metrics_path, summary_path, identity_path):
                if not path.exists():
                    raise RuntimeError(f"missing completed output: {path}")

            frame = pd.read_csv(metrics_path).sort_values("round").reset_index(drop=True)
            missing = required - set(frame.columns)
            if missing:
                raise RuntimeError(f"seed {seed} {strategy}: missing columns {sorted(missing)}")
            if len(frame) != 100 or frame["round"].astype(int).tolist() != list(range(1, 101)):
                raise RuntimeError(f"seed {seed} {strategy}: expected rounds 1 to 100")
            for column in ("test_loss", "test_accuracy", "system_time", "round_duration"):
                if not np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy(float)).all():
                    raise RuntimeError(f"seed {seed} {strategy}: nonfinite {column}")
            if not np.all(np.diff(frame["system_time"].to_numpy(float)) > 0):
                raise RuntimeError(f"seed {seed} {strategy}: system_time is not strictly increasing")
            if not math.isclose(
                float(frame["round_duration"].sum()),
                float(frame.iloc[-1]["system_time"]),
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise RuntimeError(f"seed {seed} {strategy}: cumulative time does not reconcile")
            if not (frame["acknowledged_clients"].astype(int) == 100).all():
                raise RuntimeError(f"seed {seed} {strategy}: incomplete acknowledgement barrier")
            if not (frame["pending_units"].astype(int) == 0).all():
                raise RuntimeError(f"seed {seed} {strategy}: pending units recorded")
            if float(frame["max_epsilon"].max()) > 8.0 + 1e-8:
                raise RuntimeError(f"seed {seed} {strategy}: privacy budget exceeded")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not math.isclose(
                float(summary["system_time"]),
                float(frame.iloc[-1]["system_time"]),
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise RuntimeError(f"seed {seed} {strategy}: summary time mismatch")
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            source_hash = str(identity["source_hash"])
            if source_hash != EXPECTED_SOURCE_HASH and (seed, strategy) not in ACCEPTED_SOURCE_EXCEPTIONS:
                raise RuntimeError(f"unexpected source identity for seed {seed} {strategy}: {source_hash}")

            frames[(seed, strategy)] = frame
            entry = {"seed": seed, "strategy": strategy, **identity}
            identities.append(entry)
            seed_identities.append(entry)
            for kind, path in (
                ("round_metrics", metrics_path),
                ("summary", summary_path),
                ("run_identity", identity_path),
            ):
                source_files.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "kind": kind,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256(path),
                    }
                )

        for key in ("config_hash", "partition_hash", "traces_hash", "data_hashes"):
            values = {json.dumps(item[key], sort_keys=True) for item in seed_identities}
            if len(values) != 1:
                raise RuntimeError(f"seed {seed}: within-seed identity mismatch for {key}")

        prefix_columns = ["test_loss", "test_accuracy", "system_time", "round_duration"]
        common = frames[(seed, "fixed_P0")].iloc[:5][prefix_columns].to_numpy(float)
        for strategy in ADAPTIVE:
            observed = frames[(seed, strategy)].iloc[:5][prefix_columns].to_numpy(float)
            if not np.allclose(observed, common, atol=1e-12, rtol=0.0):
                raise RuntimeError(f"seed {seed} {strategy}: common P0 prefix mismatch")

    return frames, {"identities": identities, "source_files": source_files}


def build_round_aggregates(
    frames: dict[tuple[int, str], pd.DataFrame], *, accuracy_only: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    per_seed_final = []
    for seed in SEEDS:
        for strategy in STRATEGIES:
            frame = frames[(seed, strategy)].copy()
            frame.insert(0, "strategy", strategy)
            frame.insert(0, "seed", seed)
            records.append(frame)
            accuracy_window = frame.loc[frame["round"].isin(ACCURACY_ROUNDS)]
            if accuracy_window["round"].astype(int).tolist() != list(ACCURACY_ROUNDS):
                raise RuntimeError(f"seed {seed} {strategy}: expected accuracy rounds 91 to 100")
            per_seed_final.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "final_loss": float(frame.iloc[-1]["test_loss"]),
                    "final_accuracy": float(frame.iloc[-1]["test_accuracy"]),
                    # Average within each seed first; SD is then across three seeds.
                    "last10_accuracy": float(accuracy_window["test_accuracy"].mean()),
                    "cumulative_system_time_s": float(frame.iloc[-1]["system_time"]),
                    "communication_mb": float(frame["communication_mb"].sum()),
                    "executed_local_steps": float(frame["executed_local_steps"].sum()),
                    "client_mode_reassignments": float(frame["mode_changes"].sum()),
                    "max_epsilon": float(frame["max_epsilon"].max()),
                    "mean_selection_ms": float(frame["selection_sec"].mean()) * 1000.0,
                }
            )
    all_rounds = pd.concat(records, ignore_index=True)
    round_stats = aggregate_columns(all_rounds, ["strategy", "round"], ROUND_NUMERIC_COLUMNS)
    if not (round_stats["group_size"] == len(SEEDS)).all():
        raise RuntimeError("round aggregate does not contain exactly three seeds")

    per_seed_final_frame = pd.DataFrame.from_records(per_seed_final)
    final_columns = [column for column in per_seed_final_frame.columns if column not in {"seed", "strategy"}]
    final_stats = aggregate_columns(per_seed_final_frame, ["strategy"], final_columns)
    if not (final_stats["group_size"] == len(SEEDS)).all():
        raise RuntimeError("final aggregate does not contain exactly three seeds")

    if accuracy_only:
        # Refuse any change to the existing endpoint metrics, including loss/T.
        for filename, fresh, keys in (
            ("final_metrics_per_seed.csv", per_seed_final_frame, ["seed", "strategy"]),
            ("final_metrics_mean_sd.csv", final_stats, ["strategy"]),
        ):
            previous = pd.read_csv(AGGREGATE_DIR / filename).sort_values(keys).reset_index(drop=True)
            fresh = fresh.sort_values(keys).reset_index(drop=True)
            if len(previous) != len(fresh):
                raise RuntimeError(f"{filename}: endpoint count changed")
            for column in previous:
                if column == "strategy":
                    unchanged = previous[column].equals(fresh[column])
                else:
                    unchanged = np.allclose(previous[column], fresh[column], rtol=1e-11, atol=1e-10)
                if not unchanged:
                    raise RuntimeError(f"{filename}: unexpected change to {column}")
    else:
        all_rounds.to_csv(AGGREGATE_DIR / "round_metrics_per_seed.csv", index=False, float_format="%.12g")
        round_stats.to_csv(AGGREGATE_DIR / "round_metrics_mean_sd.csv", index=False, float_format="%.12g")
    per_seed_final_frame.to_csv(AGGREGATE_DIR / "final_metrics_per_seed.csv", index=False, float_format="%.12g")
    final_stats.to_csv(AGGREGATE_DIR / "final_metrics_mean_sd.csv", index=False, float_format="%.12g")
    return round_stats, final_stats


def load_round6_candidates() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_records = []
    utility_records = []
    for seed in SEEDS:
        reference = None
        history_ids = set()
        for strategy in GREEDY:
            path = RUN_ROOT / f"seed_{seed}" / strategy / "candidates_round_005.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload["applied_round"]) != 6:
                raise RuntimeError(f"seed {seed} {strategy}: expected round 6 candidate record")
            history_ids.add(str(payload["history_id"]))
            current = pd.DataFrame(payload["candidates"])[
                ["name", "T", "J", "J_norm", "T_norm", "utility", "feasible"]
            ].sort_values("name").reset_index(drop=True)
            if not current["feasible"].astype(bool).all():
                raise RuntimeError(f"seed {seed} {strategy}: infeasible round 6 candidate")
            raw = current[["name", "T", "J", "J_norm", "T_norm"]].copy()
            if reference is None:
                reference = raw
                for _, row in raw.iterrows():
                    raw_records.append(
                        {
                            "seed": seed,
                            "candidate": str(row["name"]),
                            "T": float(row["T"]),
                            "J": float(row["J"]),
                            "J_norm": float(row["J_norm"]),
                            "T_norm": float(row["T_norm"]),
                        }
                    )
            elif not np.allclose(
                raw[["T", "J", "J_norm", "T_norm"]].to_numpy(float),
                reference[["T", "J", "J_norm", "T_norm"]].to_numpy(float),
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError(f"seed {seed}: Greedy strategies do not share round 6 scores")
            selected = str(payload["selected_candidate"])
            for _, row in current.iterrows():
                utility_records.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "lambda": float(payload["lambda"]),
                        "candidate": str(row["name"]),
                        "utility": float(row["utility"]),
                        "selected": str(row["name"]) == selected,
                    }
                )
        if len(history_ids) != 1:
            raise RuntimeError(f"seed {seed}: round 6 candidates use different histories")

    raw = pd.DataFrame.from_records(raw_records)
    utility = pd.DataFrame.from_records(utility_records)
    candidate_stats = aggregate_columns(raw, ["candidate"], ["T", "J", "J_norm", "T_norm"])
    utility_stats = aggregate_columns(utility, ["strategy", "lambda", "candidate"], ["utility"])
    selected_counts = (
        utility.groupby(["strategy", "lambda", "candidate"], sort=False)["selected"]
        .sum()
        .rename("selected_seed_count")
        .reset_index()
    )
    utility_stats = utility_stats.merge(selected_counts, on=["strategy", "lambda", "candidate"], how="left")

    raw.to_csv(AGGREGATE_DIR / "round6_candidate_per_seed.csv", index=False, float_format="%.12g")
    candidate_stats.to_csv(AGGREGATE_DIR / "round6_candidate_mean_sd.csv", index=False, float_format="%.12g")
    utility.to_csv(AGGREGATE_DIR / "round6_utility_per_seed.csv", index=False, float_format="%.12g")
    utility_stats.to_csv(AGGREGATE_DIR / "round6_utility_mean_sd.csv", index=False, float_format="%.12g")
    return raw, candidate_stats, utility_stats


def load_selection_sequences(
    frames: dict[tuple[int, str], pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sequence_records = []
    switch_records = []
    for seed in SEEDS:
        for strategy in ADAPTIVE:
            directory = RUN_ROOT / f"seed_{seed}" / strategy
            payloads = []
            for index in range(100):
                path = directory / f"candidates_round_{index:03d}.json"
                if not path.exists():
                    raise RuntimeError(f"missing selection record: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                payloads.append((int(payload["applied_round"]), str(payload["selected_candidate"])))
            if [item[0] for item in payloads] != list(range(1, 101)):
                raise RuntimeError(f"seed {seed} {strategy}: incomplete applied rounds")
            names = [item[1] for item in payloads]
            if any(name not in CANDIDATES for name in names):
                raise RuntimeError(f"seed {seed} {strategy}: unexpected candidate")
            candidate_switches = sum(left != right for left, right in zip(names, names[1:]))
            client_changes = int(frames[(seed, strategy)]["mode_changes"].sum())
            for round_number, name in payloads:
                sequence_records.append(
                    {"seed": seed, "strategy": strategy, "round": round_number, "candidate": name}
                )
            switch_records.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "candidate_switches": candidate_switches,
                    "client_mode_reassignments": client_changes,
                }
            )
    sequence = pd.DataFrame.from_records(sequence_records)
    switches = pd.DataFrame.from_records(switch_records)
    switch_stats = aggregate_columns(
        switches, ["strategy"], ["candidate_switches", "client_mode_reassignments"]
    )
    sequence.to_csv(AGGREGATE_DIR / "selection_sequences_by_seed.csv", index=False)
    switches.to_csv(AGGREGATE_DIR / "switching_per_seed.csv", index=False)
    switch_stats.to_csv(AGGREGATE_DIR / "switching_mean_sd.csv", index=False, float_format="%.12g")
    return sequence, switches, switch_stats


def threshold_crossings(frames: dict[tuple[int, str], pd.DataFrame], threshold: float = 1.30) -> pd.DataFrame:
    records = []
    for seed in SEEDS:
        for strategy in STRATEGIES:
            reached = frames[(seed, strategy)][frames[(seed, strategy)]["test_loss"] <= threshold]
            records.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "threshold": threshold,
                    "reached": not reached.empty,
                    "first_round": float(reached.iloc[0]["round"]) if not reached.empty else np.nan,
                    "first_system_time_s": float(reached.iloc[0]["system_time"]) if not reached.empty else np.nan,
                }
            )
    per_seed = pd.DataFrame.from_records(records)
    stats = aggregate_columns(per_seed, ["strategy", "threshold"], ["first_round", "first_system_time_s"])
    reached_counts = (
        per_seed.groupby(["strategy", "threshold"], sort=False)["reached"]
        .sum()
        .rename("reached_seed_count")
        .reset_index()
    )
    stats = stats.merge(reached_counts, on=["strategy", "threshold"], how="left")
    per_seed.to_csv(AGGREGATE_DIR / "loss_threshold_crossing_per_seed.csv", index=False)
    stats.to_csv(AGGREGATE_DIR / "loss_threshold_crossing_mean_sd.csv", index=False, float_format="%.12g")
    return stats


def round_series(round_stats: pd.DataFrame, strategy: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    table = round_stats[round_stats["strategy"] == strategy].sort_values("round")
    return table[f"{column}_mean"].to_numpy(float), table[f"{column}_std"].to_numpy(float)


def plot_loss_curves(
    round_stats: pd.DataFrame,
    strategies: list[str],
    stem: str,
    wide_panels: bool = False,
) -> None:
    if not wide_panels:
        fig, axis = plt.subplots(figsize=(3.55, 3.05))
        axes_and_groups = [(axis, strategies, None)]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.95), sharey=True)
        axes_and_groups = [
            (axes[0], GREEDY, "(a) Greedy weight sensitivity"),
            (
                axes[1],
                ["greedy_lambda050", "max_J", "min_T", "hysteresis_lambda050", "periodic_lambda050"],
                "(b) Objective and switching controls",
            ),
        ]

    for axis, current_strategies, title in axes_and_groups:
        for strategy in current_strategies:
            mean_t, _ = round_series(round_stats, strategy, "system_time")
            mean_loss, std_loss = round_series(round_stats, strategy, "test_loss")
            axis.fill_between(
                mean_t,
                mean_loss - std_loss,
                mean_loss + std_loss,
                color=COLORS[strategy],
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                mean_t,
                mean_loss,
                color=COLORS[strategy],
                linestyle=LINESTYLES[strategy],
                linewidth=1.25,
                label=LABELS[strategy],
                zorder=3,
            )
        axis.set_xlabel(r"Mean cumulative system latency, $T$ (s)")
        if title:
            axis.set_title(title, loc="left", pad=5)
        clean_axes(axis)
        axis.legend(loc="upper right", frameon=False, handlelength=2.4, fontsize=6.5)

    if wide_panels:
        axes_and_groups[0][0].set_ylabel("Mean test loss")
        fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.19, wspace=0.10)
    else:
        axis.set_ylabel("Mean test loss")
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=3,
            frameon=False,
            handlelength=2.1,
            columnspacing=0.8,
            fontsize=6.3,
        )
        fig.subplots_adjust(left=0.16, right=0.99, top=0.91, bottom=0.16)
    save(fig, stem)


def nondominated_candidates(table: pd.DataFrame) -> list[str]:
    result = []
    for _, row in table.iterrows():
        dominated = False
        for _, other in table.iterrows():
            if row["candidate"] == other["candidate"]:
                continue
            no_slower = float(other["T_mean"]) <= float(row["T_mean"]) + 1e-12
            no_smaller_j = float(other["J_mean"]) >= float(row["J_mean"]) - 1e-12
            strict = (
                float(other["T_mean"]) < float(row["T_mean"]) - 1e-12
                or float(other["J_mean"]) > float(row["J_mean"]) + 1e-12
            )
            if no_slower and no_smaller_j and strict:
                dominated = True
                break
        if not dominated:
            result.append(str(row["candidate"]))
    return result


def plot_round6_candidates(candidate_stats: pd.DataFrame) -> list[str]:
    table = candidate_stats.set_index("candidate").loc[CANDIDATES].reset_index()
    pareto = nondominated_candidates(table)
    fig, axis = plt.subplots(figsize=(3.55, 3.10))
    offsets = {"P0": (-32, 14), "P1": (7, 10), "P2": (5, 10), "P3": (-26, 10), "P4": (5, -17)}
    for _, row in table.iterrows():
        name = str(row["candidate"])
        axis.errorbar(
            float(row["T_mean"]),
            float(row["J_mean"]),
            xerr=float(row["T_std"]),
            yerr=float(row["J_std"]),
            fmt="o",
            markersize=5.8,
            color=CANDIDATE_COLORS[name],
            ecolor=CANDIDATE_COLORS[name],
            elinewidth=0.85,
            capsize=2.2,
            markeredgecolor="#222222" if name in pareto else "white",
            markeredgewidth=0.9 if name in pareto else 0.5,
            zorder=4,
        )
        axis.annotate(
            name,
            (float(row["T_mean"]), float(row["J_mean"])),
            xytext=offsets[name],
            textcoords="offset points",
            fontsize=7.5,
        )
    frontier = table[table["candidate"].isin(pareto)].sort_values("T_mean")
    axis.plot(
        frontier["T_mean"],
        frontier["J_mean"],
        color="#555555",
        linestyle=(0, (3, 2)),
        linewidth=0.85,
        label="Boundary of means",
        zorder=2,
    )
    axis.set_xlabel(r"Predicted round latency, $T$ (s)")
    axis.set_ylabel(r"Historical deviation contribution, $J$")
    clean_axes(axis, "both")
    axis.legend(loc="center left", frameon=False, fontsize=6.4)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.97, bottom=0.16)
    save(fig, "experiment_round6_J_T_candidates")
    return pareto


def plot_round6_utility(utility_stats: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.58), sharey=True)
    for axis, strategy in zip(axes, GREEDY):
        table = utility_stats[utility_stats["strategy"] == strategy].set_index("candidate").loc[CANDIDATES]
        x = np.arange(len(CANDIDATES))
        values = table["utility_mean"].to_numpy(float)
        errors = table["utility_std"].to_numpy(float)
        bars = axis.bar(
            x,
            values,
            yerr=errors,
            capsize=2.2,
            color=[CANDIDATE_COLORS[name] for name in CANDIDATES],
            alpha=0.95,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
        )
        for bar, count, error in zip(bars, table["selected_seed_count"], errors):
            if int(count) > 0:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + float(error) + 0.035,
                    f"{int(count)}/3",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    fontweight="bold" if int(count) == 3 else "normal",
                )
        lam = float(table["lambda"].iloc[0])
        axis.set_title(rf"$\lambda={lam:.2f}$", pad=5)
        axis.set_xticks(x, CANDIDATES)
        axis.set_xlabel("Candidate")
        axis.set_ylim(0, 1.05)
        clean_axes(axis)
    axes[0].set_ylabel(r"Mean joint utility, $U$")
    fig.text(
        0.50,
        0.015,
        "Labels above bars give the number of seeds that selected the candidate.",
        ha="center",
        va="bottom",
        fontsize=6.7,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.25, wspace=0.08)
    save(fig, "experiment_round6_joint_utility")


def plot_selection_switching(sequence: pd.DataFrame, switch_stats: pd.DataFrame) -> None:
    code = {name: index for index, name in enumerate(CANDIDATES)}
    rows = []
    for strategy in ADAPTIVE:
        for seed in SEEDS:
            current = sequence[(sequence["strategy"] == strategy) & (sequence["seed"] == seed)].sort_values("round")
            rows.append([code[value] for value in current["candidate"]])
    matrix = np.asarray(rows, dtype=float)
    cmap = ListedColormap([CANDIDATE_COLORS[name] for name in CANDIDATES])
    # Draw at the intended one-column size instead of shrinking a wide figure.
    fig, axes = plt.subplots(
        1, 2, figsize=(3.50, 4.15), gridspec_kw={"width_ratios": [1.7, 1.0]}
    )
    centers = 3 * np.arange(len(ADAPTIVE)) + 1
    compact_labels = [
        "Greedy\n" + r"$\lambda=0.25$",
        "Greedy\n" + r"$\lambda=0.50$",
        "Greedy\n" + r"$\lambda=0.75$",
        "Hysteresis\n" + r"$\lambda=0.50$",
        "Periodic\n" + r"$\lambda=0.50$",
        r"Maximum $J$",
        r"Minimum $T$",
    ]

    axis = axes[0]
    axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=4.5,
        extent=[0.5, 100.5, len(rows) - 0.5, -0.5],
    )
    axis.axvline(5.5, color="white", linewidth=0.8, linestyle=(0, (3, 2)))
    for boundary in range(3, len(rows), 3):
        axis.axhline(boundary - 0.5, color="white", linewidth=0.8)
    axis.set_xlim(0.5, 100.5)
    axis.set_xticks([1, 25, 50, 75, 100])
    axis.set_yticks(centers, compact_labels, fontsize=8.3)
    axis.tick_params(axis="x", labelsize=7.8, length=3, pad=2)
    axis.tick_params(axis="y", length=0, pad=4)
    axis.set_xlabel("Round", fontsize=8.5, labelpad=3)
    axis.set_title("(a) Candidate selection", loc="left", pad=27, fontsize=8.6)
    axis.text(
        0.0, 1.015, "Rows: 42, 43, 44", transform=axis.transAxes,
        fontsize=7.6, va="bottom", color="#333333",
    )
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=CANDIDATE_COLORS[name], markersize=5.0, label=name)
        for name in CANDIDATES
    ]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
        frameon=False,
        fontsize=7.6,
        handlelength=0.7,
        handletextpad=0.2,
        columnspacing=0.65,
        borderaxespad=0,
    )

    axis = axes[1]
    indexed = switch_stats.set_index("strategy").loc[ADAPTIVE]
    means = indexed["client_mode_reassignments_mean"].to_numpy(float)
    stds = indexed["client_mode_reassignments_std"].to_numpy(float)
    bars = axis.barh(
        centers + 0.45,
        means,
        height=0.70,
        xerr=stds,
        capsize=1.8,
        color="#777777",  # Candidate colors are reserved for panel (a).
        alpha=0.9,
        error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
    )
    axis.set_yticks([])
    axis.set_ylim(len(rows) - 0.5, -0.5)
    axis.set_xlabel("Client mode\nreassignments", fontsize=8.3, labelpad=3)
    axis.set_title("(b) Switching", loc="left", pad=27, fontsize=8.6)
    clean_axes(axis, "x")
    for boundary in range(3, len(rows), 3):
        axis.axhline(boundary - 0.5, color="#E5E5E5", linewidth=0.4, zorder=0)
    limit = float(np.max(means + stds))
    for center, strategy in zip(centers, ADAPTIVE):
        candidate_mean = float(indexed.loc[strategy, "candidate_switches_mean"])
        candidate_std = float(indexed.loc[strategy, "candidate_switches_std"])
        axis.text(
            0.02,
            center - 0.55,
            rf"{candidate_mean:.1f}$\pm${candidate_std:.1f}",
            transform=axis.get_yaxis_transform(),
            va="center",
            fontsize=7.8,
        )
    axis.set_xlim(0, math.ceil(limit * 1.05 / 500) * 500)
    axis.set_xticks([0, 1000, 2000])
    axis.tick_params(axis="x", labelsize=7.8, length=3, pad=2)
    fig.subplots_adjust(left=0.205, right=0.99, top=0.84, bottom=0.20, wspace=0.22)
    save(fig, "experiment_selection_switching")


def nondominated_endpoints(table: pd.DataFrame, accuracy_metric: str = ACCURACY_METRIC) -> list[str]:
    result = []
    for _, row in table.iterrows():
        dominated = False
        for _, other in table.iterrows():
            if row["strategy"] == other["strategy"]:
                continue
            no_slower = float(other["cumulative_system_time_s_mean"]) <= float(row["cumulative_system_time_s_mean"]) + 1e-12
            no_less_accurate = float(other[f"{accuracy_metric}_mean"]) >= float(row[f"{accuracy_metric}_mean"]) - 1e-12
            strict = (
                float(other["cumulative_system_time_s_mean"]) < float(row["cumulative_system_time_s_mean"]) - 1e-12
                or float(other[f"{accuracy_metric}_mean"]) > float(row[f"{accuracy_metric}_mean"]) + 1e-12
            )
            if no_slower and no_less_accurate and strict:
                dominated = True
                break
        if not dominated:
            result.append(str(row["strategy"]))
    return result


def plot_accuracy_latency_six(final_stats: pd.DataFrame) -> list[str]:
    """Compare five fixed references with Greedy; retain the legacy filename."""
    strategies = [*FIXED, *GREEDY]
    table = final_stats.set_index("strategy").loc[strategies].reset_index()
    fixed_table = table[table["strategy"].isin(FIXED)]
    fixed_nondominated = nondominated_endpoints(fixed_table)
    frontier = fixed_table[fixed_table["strategy"].isin(fixed_nondominated)].sort_values(
        "cumulative_system_time_s_mean"
    )
    fig, axis = plt.subplots(figsize=(3.50, 4.05))
    xmin, xmax = 182.2, 195.0
    ymin, ymax = 51.95, 56.60
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)

    # This staircase uses only observed fixed means, not interpolated strategies.
    step_x = np.r_[frontier["cumulative_system_time_s_mean"].to_numpy(float), xmax]
    step_y = np.r_[100 * frontier[f"{ACCURACY_METRIC}_mean"].to_numpy(float),
                   100 * float(frontier.iloc[-1][f"{ACCURACY_METRIC}_mean"])]
    region = axis.fill_between(
        step_x, ymin, step_y, step="post", color="#777777", alpha=0.055, zorder=0
    )
    region.set_gid("fixed_mean_dominated_region")
    boundary, = axis.step(
        step_x, step_y, where="post", color="#626262",
        linestyle=(0, (3, 2)), linewidth=0.8, zorder=2,
    )
    boundary.set_gid("fixed_reference_boundary")
    axis.vlines(
        step_x[0], ymin, step_y[0], color="#626262",
        linestyle=(0, (3, 2)), linewidth=0.8, zorder=2,
    )

    label_positions = {
        "fixed_P0": (193.8, 52.65, "right"),
        "fixed_P1": (187.6, 55.45, "center"),
        "fixed_P2": (193.3, 55.78, "left"),
        "fixed_P3": (185.9, 53.10, "center"),
        "fixed_P4": (194.6, 54.05, "right"),
        "greedy_lambda025": (184.3, 54.20, "left"),
        "greedy_lambda050": (185.8, 54.87, "left"),
        "greedy_lambda075": (194.6, 54.85, "right"),
    }
    labels = {
        **{strategy: rf"$P_{index}$" for index, strategy in enumerate(FIXED)},
        "greedy_lambda025": r"$\lambda=0.25$",
        "greedy_lambda050": r"$\lambda=0.50$",
        "greedy_lambda075": r"$\lambda=0.75$",
    }
    for _, row in table.iterrows():
        strategy = str(row["strategy"])
        x = float(row["cumulative_system_time_s_mean"])
        y = 100.0 * float(row[f"{ACCURACY_METRIC}_mean"])
        # Full sample SDs stay visible but do not obscure the observed means.
        axis.errorbar(
            x,
            y,
            xerr=float(row["cumulative_system_time_s_std"]),
            yerr=100.0 * float(row[f"{ACCURACY_METRIC}_std"]),
            fmt="none",
            ecolor="#8E8E8E" if strategy in FIXED else COLORS[strategy],
            alpha=0.30,
            elinewidth=0.5,
            capsize=1.3,
            capthick=0.5,
            zorder=1,
        )
        point, = axis.plot(
            x, y, marker="s" if strategy in FIXED else "o", linestyle="none",
            markersize=4.8,
            markerfacecolor="white" if strategy in FIXED else COLORS[strategy],
            markeredgecolor="#454545" if strategy in FIXED else "#202020",
            markeredgewidth=0.8,
            zorder=6,
        )
        point.set_gid(f"mean:{strategy}")
        label_x, label_y, alignment = label_positions[strategy]
        axis.annotate(
            labels[strategy], xy=(x, y), xytext=(label_x, label_y),
            ha=alignment, va="center", fontsize=8.3, color="#202020",
            arrowprops={"arrowstyle": "-", "color": "#7A7A7A", "linewidth": 0.55,
                        "shrinkA": 2.5, "shrinkB": 3.5},
            zorder=7,
        )

    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="white",
               markeredgecolor="#454545", markersize=4.8, label="Fixed"),
        Line2D([0], [0], linestyle=(0, (3, 2)), color="#626262", linewidth=0.8,
               label="Fixed boundary"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#555555",
               markeredgecolor="#202020", markersize=4.8, label="Greedy"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#777777", alpha=0.055,
                      edgecolor="none", label="Dominated region"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.995),
        ncol=2, frameon=False, fontsize=8.0, handlelength=1.6,
        handletextpad=0.45, columnspacing=1.4, labelspacing=0.5,
    )
    axis.set_xlabel("Cumulative system latency (s)", fontsize=8.6, labelpad=4)
    axis.set_ylabel("Last-10-round test accuracy (%)", fontsize=8.6, labelpad=4)
    axis.set_xticks([184, 188, 192])
    axis.set_yticks([52, 53, 54, 55, 56])
    axis.tick_params(labelsize=8.0, length=3, pad=2)
    clean_axes(axis, "both")
    axis.grid(color="#DCDCDC", linewidth=0.4, alpha=0.45)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.86, bottom=0.14)
    save(fig, "experiment_final_accuracy_latency_six")
    return frontier["strategy"].tolist()


def plot_performance_steps(final_stats: pd.DataFrame) -> tuple[float, float]:
    table = final_stats.set_index("strategy").loc[STRATEGIES].reset_index()
    marker = {
        **{key: "s" for key in FIXED},
        **{key: "o" for key in GREEDY},
        **{key: "o" for key in STABILITY},
        "max_J": "o",
        "min_T": "o",
        **{key: "^" for key in PURE},
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.30))
    for axis, y_column, y_scale, y_label, title in [
        (axes[0], "final_loss", 1.0, "Mean final test loss", "(a) Loss and training work"),
        (axes[1], ACCURACY_METRIC, 100.0, "Mean last-10-round test accuracy (%)", "(b) Accuracy and training work"),
    ]:
        for _, row in table.iterrows():
            strategy = str(row["strategy"])
            x = float(row["executed_local_steps_mean"])
            y = y_scale * float(row[f"{y_column}_mean"])
            axis.errorbar(
                x,
                y,
                xerr=float(row["executed_local_steps_std"]),
                yerr=y_scale * float(row[f"{y_column}_std"]),
                fmt=marker[strategy],
                markersize=5.2,
                color=COLORS[strategy],
                ecolor=COLORS[strategy],
                elinewidth=0.7,
                capsize=1.8,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=4,
            )
        axis.set_xlabel("Mean total executed local steps")
        axis.set_ylabel(y_label)
        axis.set_title(title, loc="left", pad=5)
        clean_axes(axis, "both")
        axis.xaxis.set_major_formatter(lambda value, position: f"{value / 1000:.0f}k")

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker[strategy],
            linestyle="none",
            markerfacecolor=COLORS[strategy],
            markeredgecolor=COLORS[strategy],
            markersize=4.9,
            label=SHORT_LABELS[strategy],
        )
        for strategy in STRATEGIES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=8,
        frameon=False,
        columnspacing=0.75,
        handletextpad=0.25,
        fontsize=6.1,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.29, wspace=0.25)
    save(fig, "experiment_final_performance_vs_local_steps")

    mean_steps = table["executed_local_steps_mean"].to_numpy(float)
    mean_loss = table["final_loss_mean"].to_numpy(float)
    mean_accuracy = table[f"{ACCURACY_METRIC}_mean"].to_numpy(float)
    return (
        float(np.corrcoef(mean_steps, mean_loss)[0, 1]),
        float(np.corrcoef(mean_steps, mean_accuracy)[0, 1]),
    )


def write_manifest(
    metadata: dict,
    pareto_candidates: list[str],
    fixed_reference_front: list[str],
    correlations: tuple[float, float],
) -> None:
    audit_path = RUN_ROOT / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise RuntimeError("campaign audit is not passed under its declared qualification")
    payload = {
        "schema": "paper.figure.three_seed_aggregate.v1",
        "seeds": list(SEEDS),
        "strategies": STRATEGIES,
        "rounds_per_run": 100,
        "run_count": len(SEEDS) * len(STRATEGIES),
        "statistics": {
            "mean": "arithmetic mean across the three seeds",
            "standard_deviation": "sample standard deviation with denominator n-1 (ddof=1)",
            "curve_alignment": "same strategy and training round; x is mean cumulative system time and the band is mean test loss plus or minus one sample SD",
            "categorical_selection": "candidate sequences are shown separately for each seed; candidate identifiers are not averaged",
            "accuracy_summary": accuracy_summary_definition(),
        },
        "campaign_audit": {
            "path": str(audit_path.relative_to(ROOT)),
            "sha256": sha256(audit_path),
            "status": audit["status"],
            "status_qualification": audit.get("status_qualification"),
            "strict_compliance": audit.get("strict_compliance"),
        },
        "accepted_source_identity_exceptions": [
            {"seed": seed, "strategy": strategy} for seed, strategy in sorted(ACCEPTED_SOURCE_EXCEPTIONS)
        ],
        "round6_descriptive_nondominated_candidates_using_means": pareto_candidates,
        "accuracy_latency_fixed_references": FIXED,
        "accuracy_latency_greedy_comparisons": GREEDY,
        "fixed_reference_nondominated_set_using_means": fixed_reference_front,
        "fixed_reference_boundary": "stepwise best observed fixed mean accuracy under a cumulative latency budget; Greedy points do not define the boundary",
        "correlation_across_15_strategy_means": {
            "local_steps_vs_final_loss": correlations[0],
            "local_steps_vs_last10_accuracy": correlations[1],
        },
        "raw_inputs": metadata["source_files"],
        "outputs": [
            "round_metrics_per_seed.csv",
            "round_metrics_mean_sd.csv",
            "final_metrics_per_seed.csv",
            "final_metrics_mean_sd.csv",
            "round6_candidate_per_seed.csv",
            "round6_candidate_mean_sd.csv",
            "round6_utility_per_seed.csv",
            "round6_utility_mean_sd.csv",
            "selection_sequences_by_seed.csv",
            "switching_per_seed.csv",
            "switching_mean_sd.csv",
            "loss_threshold_crossing_per_seed.csv",
            "loss_threshold_crossing_mean_sd.csv",
        ],
    }
    (AGGREGATE_DIR / "aggregate_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (HERE / "experiment_figure_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def accuracy_summary_definition() -> dict:
    return {
        "figure_and_table_metric": ACCURACY_METRIC,
        "rounds": list(ACCURACY_ROUNDS),
        "per_seed": "arithmetic mean of test_accuracy over rounds 91 through 100 inclusive",
        "across_seeds": "mean and sample SD (ddof=1) of the three per-seed means; n=3, not 30",
        "stored_unit": "fraction; multiply by 100 for percentage",
        "legacy_final_accuracy": "round 100 accuracy, retained unchanged for provenance; not used by the updated summary figures",
        "other_metrics": "final_loss remains round 100; cumulative latency, communication and local steps remain 100-round totals",
    }


def update_accuracy_manifests(fixed_reference_front: list[str], correlations: tuple[float, float]) -> None:
    """Update only metadata dependent on the revised accuracy summary."""
    for path in (AGGREGATE_DIR / "aggregate_manifest.json", HERE / "experiment_figure_audit.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["statistics"]["accuracy_summary"] = accuracy_summary_definition()
        payload.pop("six_strategy_descriptive_nondominated_set_using_means", None)
        payload["accuracy_latency_fixed_references"] = FIXED
        payload["accuracy_latency_greedy_comparisons"] = GREEDY
        payload["fixed_reference_nondominated_set_using_means"] = fixed_reference_front
        payload["fixed_reference_boundary"] = (
            "stepwise best observed fixed mean last-10-round accuracy under a cumulative latency budget; "
            "Greedy points do not define the boundary"
        )
        associations = payload["correlation_across_15_strategy_means"]
        if not math.isclose(associations["local_steps_vs_final_loss"], correlations[0], abs_tol=1e-10):
            raise RuntimeError("loss correlation unexpectedly changed")
        associations.pop("local_steps_vs_final_accuracy", None)
        associations["local_steps_vs_last10_accuracy"] = correlations[1]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accuracy-only", action="store_true",
        help="update last-10-round accuracy summaries and two dependent figures only",
    )
    args = parser.parse_args()
    configure_style()
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    frames, metadata = load_runs()
    if args.accuracy_only:
        audit = json.loads((RUN_ROOT / "audit.json").read_text(encoding="utf-8"))
        if audit.get("status") != "passed":
            raise RuntimeError("campaign audit is not passed under its declared qualification")
        _, final_stats = build_round_aggregates(frames, accuracy_only=True)
        fixed_reference_front = plot_accuracy_latency_six(final_stats)
        correlations = plot_performance_steps(final_stats)
        update_accuracy_manifests(fixed_reference_front, correlations)
        print(json.dumps({
            "status": "passed", "scope": "accuracy_only", "seeds": list(SEEDS),
            "runs": len(frames), "accuracy_rounds": list(ACCURACY_ROUNDS),
            "figures": 2, "fixed_reference_boundary": fixed_reference_front,
            "local_steps_vs_last10_accuracy": correlations[1],
        }, indent=2))
        return
    round_stats, final_stats = build_round_aggregates(frames)
    _, candidate_stats, utility_stats = load_round6_candidates()
    sequence, _, switch_stats = load_selection_sequences(frames)
    threshold_crossings(frames)

    plot_loss_curves(round_stats, FIXED, "experiment_fixed_loss_vs_cumulative_T")
    plot_loss_curves(round_stats, ADAPTIVE, "experiment_dynamic_loss_vs_cumulative_T", wide_panels=True)
    pareto_candidates = plot_round6_candidates(candidate_stats)
    plot_round6_utility(utility_stats)
    plot_selection_switching(sequence, switch_stats)
    fixed_reference_front = plot_accuracy_latency_six(final_stats)
    correlations = plot_performance_steps(final_stats)
    write_manifest(metadata, pareto_candidates, fixed_reference_front, correlations)

    print(
        json.dumps(
            {
                "status": "passed",
                "seeds": list(SEEDS),
                "runs": len(SEEDS) * len(STRATEGIES),
                "figures": 7,
                "aggregate_dir": str(AGGREGATE_DIR),
                "round6_boundary": pareto_candidates,
                "fixed_reference_boundary": fixed_reference_front,
                "correlations": correlations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
