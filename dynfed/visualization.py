"""Visualization for both sync (round-based) and async (event-based) experiments."""

from __future__ import annotations

import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_round_metrics(run_dirs: list[Path], output_path: Path) -> None:
    """Legacy: sync round-based accuracy/duration/comm plot."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    for run_dir in run_dirs:
        rows = _read_csv(run_dir / "round_metrics.csv")
        if not rows:
            continue
        label = f"{rows[0]['mode']}-{rows[0]['privacy']}"
        rounds = [int(row["round"]) for row in rows]
        axes[0].plot(rounds, [float(row["accuracy"]) for row in rows], label=label)
        axes[1].plot(rounds, [float(row["round_duration"]) for row in rows], label=label)
        axes[2].plot(rounds, [float(row["communication_volume"]) for row in rows], label=label)

    axes[0].set_title("Utility")
    axes[0].set_ylabel("accuracy")
    axes[1].set_title("Efficiency")
    axes[1].set_ylabel("round duration")
    axes[2].set_title("Communication")
    axes[2].set_ylabel("protected volume")
    for axis in axes:
        axis.set_xlabel("round")
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_async_comparison(
    result_root: Path,
    policies: list[str],
    output_path: Path,
) -> None:
    """Full comparison across async policies: accuracy + modes + resources + budget."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    from collections import Counter

    n_pol = len(policies)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    colors = {"ours": "#2171b5", "fixed_he": "#d6604d", "no_protection": "#4daf4a",
              "fixed_dp": "#ff7f00", "random": "#984ea3"}
    markers = {"ours": "o", "fixed_he": "s", "no_protection": "^",
               "fixed_dp": "D", "random": "v"}

    # ── 1. Accuracy over events ──
    ax = axes[0, 0]
    for policy in policies:
        rows = _read_csv(result_root / policy / "round_metrics.csv")
        if not rows:
            continue
        events = [int(r["event"]) for r in rows if r["tag"] == "cloud_agg"]
        accs = [float(r["test_accuracy"]) for r in rows if r["tag"] == "cloud_agg"]
        color = colors.get(policy, "#333333")
        ax.plot(events, accs, label=policy.upper(), color=color,
                marker=markers.get(policy, "."), markevery=max(1, len(events)//8), ms=5)
    ax.set_xlabel("Event")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── 2. Mode selection per end (ours only) ──
    ax = axes[0, 1]
    if "ours" in policies:
        rows = _read_csv(result_root / "ours" / "client_decisions.csv")
        if rows:
            mode_colors = {"LIEIIC": "#3182bd", "LIEIIIC": "#6baed6",
                           "LIC": "#31a354", "LIIC": "#74c476",
                           "LIIE": "#e6550d", "LIE": "#fdae6b",
                           "SKIP": "#cccccc"}
            ends = sorted(set(int(r["end_id"]) for r in rows))
            events_all = [int(r["event"]) for r in rows]
            t_min, t_max = min(events_all), max(events_all)
            for i, eid in enumerate(ends):
                erows = [r for r in rows if r["end_id"] == eid]
                xs = [int(r["event"]) for r in erows]
                modes = [r["mode"] for r in erows]
                # Accumulate fraction
                mode_seq: dict[str, list[float]] = {}
                total = 0
                for m in modes:
                    total += 1
                    for mm in set(modes):
                        mode_seq.setdefault(mm, []).append(0.0)
                    mode_seq[m][-1] = 1.0
                # Stack
                bottom = np.zeros(len(xs))
                for mm in sorted(mode_seq.keys()):
                    vals = np.array(mode_seq[mm])
                    color = mode_colors.get(mm, "#cccccc")
                    ax.fill_between(xs, bottom, bottom + vals,
                                    color=color, alpha=0.7 if len(ends) <= 4 else 0.5,
                                    label=mm if i == 0 else "")
                    bottom += vals
            ax.set_xlabel("Event")
            ax.set_ylabel("Mode selection per end")
            ax.set_title(f"Ours: Per-End Mode (stacked, {len(ends)} ends)")
            ax.legend(fontsize=6, loc="upper left", ncol=2)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)

    # ── 3. Total simulation time per policy ──
    ax = axes[1, 0]
    names, times, aggs = [], [], []
    for policy in policies:
        rows = _read_csv(result_root / policy / "round_metrics.csv")
        if rows:
            names.append(policy.upper())
            times.append(float(rows[-1]["time"]))
            aggs.append(sum(1 for r in rows if r["tag"] == "cloud_agg"))
    x = np.arange(len(names))
    w = 0.3
    bars1 = ax.bar(x - w/2, times, w, label="Sim Time (s)", color="#4292c6")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w/2, aggs, w, label="Cloud Aggs", color="#fdae6b")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Simulation Time (s)", color="#4292c6")
    ax2.set_ylabel("Cloud Aggregations", color="#fdae6b")
    ax.set_title("Efficiency Comparison")
    for tick in ax.get_yticklabels():
        tick.set_color("#4292c6")
    for tick in ax2.get_yticklabels():
        tick.set_color("#fdae6b")
    lines = [bars1, bars2]
    ax.legend(lines, ["Sim Time (s)", "Cloud Aggs"], fontsize=8, loc="upper left")

    # ── 4. Resource utilization over time (ours) ──
    ax = axes[1, 1]
    if "ours" in policies:
        rows = _read_csv(result_root / "ours" / "round_metrics.csv")
        if rows:
            events = [int(r["event"]) for r in rows]
            cpu_loads = [r["edge_cpu_loads"] for r in rows]
            cloud_load = [float(r["cloud_cpu_load"]) for r in rows]
            # Parse per-edge loads
            edge0, edge1, edge2 = [], [], []
            for s in cpu_loads:
                parts = s.split(";")
                vals = {}
                for p in parts:
                    try:
                        eid, v = p.split(":")
                        vals[int(eid)] = float(v)
                    except (ValueError, IndexError):
                        pass
                edge0.append(vals.get(0, 0))
                edge1.append(vals.get(1, 0))
                edge2.append(vals.get(2, 0))
            ax.plot(events, edge0, label="Edge0 CPU", color="#e41a1c", drawstyle="steps-post")
            ax.plot(events, edge1, label="Edge1 CPU", color="#377eb8", drawstyle="steps-post")
            ax.plot(events, edge2, label="Edge2 CPU", color="#4daf4a", drawstyle="steps-post")
            ax.plot(events, cloud_load, label="Cloud CPU", color="#984ea3",
                    drawstyle="steps-post", linestyle="--")
            ax.axhline(y=4.0, color="#e41a1c", linestyle=":", alpha=0.4, label="Edge limit")
            ax.set_xlabel("Event")
            ax.set_ylabel("CPU Load")
            ax.set_title("Resource Utilization (Ours)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25)

    # ── 5. Privacy budget remaining per end (ours) ──
    ax = axes[2, 0]
    if "ours" in policies:
        rows = _read_csv(result_root / "ours" / "client_decisions.csv")
        if rows:
            ends = sorted(set(int(r["end_id"]) for r in rows))
            for eid in ends:
                erows = [r for r in rows if r["end_id"] == eid]
                xs = [int(r["event"]) for r in erows]
                eps = [float(r["remaining_epsilon"]) for r in erows]
                ax.plot(xs, eps, label=f"End{eid}", marker=".", ms=3)
            ax.set_xlabel("Event")
            ax.set_ylabel("Remaining ε")
            ax.set_title("Privacy Budget Consumption (Ours)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25)

    # ── 6. Best accuracy summary bar ──
    ax = axes[2, 1]
    for i, policy in enumerate(policies):
        import json
        sp = result_root / policy / "summary.json"
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            color = colors.get(policy, "#333333")
            ax.bar(i, s["best_test_accuracy"], color=color, alpha=0.8, width=0.5)
            ax.text(i, s["best_test_accuracy"] + 0.01,
                    f'{s["best_test_accuracy"]:.3f}', ha="center", fontsize=9)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([p.upper() for p in policies], fontsize=9)
    ax.set_ylabel("Best Test Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Best Accuracy Comparison")
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(f"Async Training Comparison — {result_root.name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[OK] saved comparison to {output_path}")


def plot_per_end_mode_heatmap(result_root: Path, policy: str, output_path: Path) -> None:
    """Per-end mode selection timeline as a heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_csv(result_root / policy / "client_decisions.csv")
    if not rows:
        return

    ends = sorted(set(int(r["end_id"]) for r in rows))
    all_modes = sorted(set(r["mode"] for r in rows))
    mode_to_idx = {m: i for i, m in enumerate(all_modes)}
    events = [int(r["event"]) for r in rows]

    # Build event × end matrix
    e_max = max(events)
    matrix = np.full((len(ends), e_max + 1), -1, dtype=int)
    for r in rows:
        eid = int(r["end_id"])
        ev = int(r["event"])
        matrix[ends.index(eid), ev] = mode_to_idx[r["mode"]]

    fig, ax = plt.subplots(figsize=(12, 3 + 0.5 * len(ends)))
    im = ax.imshow(matrix, aspect="auto", cmap="tab10", interpolation="nearest",
                   vmin=-0.5, vmax=len(all_modes) - 0.5)
    cbar = fig.colorbar(im, ax=ax, ticks=list(range(len(all_modes))), shrink=0.6)
    cbar.set_ticklabels(all_modes, fontsize=8)

    ax.set_yticks(range(len(ends)))
    ax.set_yticklabels([f"End{e}" for e in ends], fontsize=9)
    ax.set_xlabel("Event", fontsize=10)
    ax.set_title(f"{policy}: Per-End Mode Selection Timeline", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[OK] saved heatmap to {output_path}")
