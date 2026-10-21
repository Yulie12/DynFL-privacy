"""Render staged-eight diagnostics from saved artifacts without re-estimation."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynfed.paper_runner import write_json
from dynfed.paper_storage import file_digest


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save_figure(figure, stem):
    figure.tight_layout()
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def render(run_dir, output_dir=None):
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "search_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    search_rounds = audit["rounds"]
    if not search_rounds:
        raise ValueError("no completed staged-eight search round is available")
    latest = search_rounds[-1]
    candidates = latest["candidates"]

    figure, axis = plt.subplots(figsize=(6.4, 4.5))
    for item in candidates:
        color = "tab:red" if item["pareto"] else "0.65"
        marker = "*" if item["name"] == latest["selected_candidate"] else "o"
        size = 150 if marker == "*" else 55
        axis.scatter(item["T"], item["J"], c=color, marker=marker, s=size, zorder=3)
        axis.annotate(item["name"], (item["T"], item["J"]), xytext=(4, 4),
                      textcoords="offset points", fontsize=8)
    front = sorted((item for item in candidates if item["pareto"]), key=lambda item: item["T"])
    axis.plot([item["T"] for item in front], [item["J"] for item in front],
              color="tab:red", linewidth=1, alpha=.7)
    axis.set(xlabel="Predicted round latency T (s)", ylabel="History score J",
             title=f"Round {latest['round']} eight-candidate Pareto front")
    axis.grid(alpha=.2)
    save_figure(figure, output_dir / f"pareto_round_{latest['round']:03d}")
    plt.close(figure)

    modes = read_csv(run_dir / "mode_ratios.csv")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    x = [int(row["round"]) for row in modes]
    bottom = [0.] * len(x)
    colors = {"FL_E": "#4C78A8", "FL_EC": "#F58518", "FL_C": "#54A24B"}
    for mode in ("FL_E", "FL_EC", "FL_C"):
        values = [float(row[mode + "_ratio"]) for row in modes]
        axis.bar(x, values, bottom=bottom, label=mode, color=colors[mode], width=.85)
        bottom = [a + b for a, b in zip(bottom, values)]
    axis.axvline(10.5, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Round", ylabel="Executed client fraction", ylim=(0, 1),
             title="Executed mode allocation by round")
    axis.legend(ncol=3, loc="upper center")
    save_figure(figure, output_dir / "mode_ratios")
    plt.close(figure)

    selected = []
    for round_report in search_rounds:
        item = next(value for value in round_report["candidates"]
                    if value["name"] == round_report["selected_candidate"])
        selected.append((round_report["round"], item["T"], item["J"],
                         round_report["pareto_front_size"]))
    figure, left = plt.subplots(figsize=(7.2, 4.2))
    rounds = [value[0] for value in selected]
    left.plot(rounds, [value[1] for value in selected], marker="o", label="selected T")
    left.set(xlabel="Round", ylabel="T (s)", title="Post-initialization selection dynamics")
    right = left.twinx()
    right.plot(rounds, [value[2] for value in selected], marker="s", color="tab:red", label="selected J")
    right.set_ylabel("J", color="tab:red")
    left.grid(alpha=.2)
    save_figure(figure, output_dir / "search_dynamics")
    plt.close(figure)

    sources = [audit_path, run_dir / "mode_ratios.csv", run_dir / "round_metrics.csv"]
    products = sorted(output_dir.glob("*.png")) + sorted(output_dir.glob("*.pdf"))
    metadata = {
        "schema": "paper.staged_eight_visualization.v1",
        "run_dir": str(run_dir.resolve()),
        "latest_search_round": latest["round"],
        "recomputed_scores": False,
        "sources": {path.name: file_digest(path) for path in sources},
        "outputs": {path.name: file_digest(path) for path in products},
    }
    write_json(output_dir / "visualization_metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.run_dir, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
