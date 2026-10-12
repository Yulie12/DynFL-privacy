import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def plot_lines(series, xlabel, ylabel, title, output):
    if not series:
        print(f"[SKIP] 没有可用数据：{title}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, x, y in series:
        ax.plot(x, y, marker="o", markersize=3,
                linewidth=1.8, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"[OK] {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)

    files = list(root.rglob("round_metrics.csv"))
    if not files:
        raise FileNotFoundError(
            f"未找到 round_metrics.csv：{root}"
        )

    # 若目录中包含重复运行，按策略只取最新的 CSV。
    latest = {}
    for path in files:
        policy = path.parent.name
        if policy not in latest or (
            path.stat().st_mtime > latest[policy].stat().st_mtime
        ):
            latest[policy] = path

    round_series = []
    logical_series = []
    policy_rows = {}

    for policy, path in sorted(latest.items()):
        rows = read_csv(path)
        parsed = []

        for index, row in enumerate(rows):
            acc = number(row.get("test_accuracy"))
            rnd = number(row.get("round"))

            if acc is None:
                continue

            logical = None
            for key in (
                "logical_time", "logical_time_sec",
                "total_logical_time"
            ):
                logical = number(row.get(key))
                if logical is not None:
                    break

            parsed.append({
                "round": rnd if rnd is not None else index,
                "accuracy": acc,
                "logical": logical,
            })

        if not parsed:
            print(f"[WARN] 没有有效精度数据：{path}")
            continue

        parsed.sort(key=lambda x: x["round"])
        # 将比例形式的精度统一转换为百分比。
        scale = 100.0 if max(
            item["accuracy"] for item in parsed
        ) <= 1.0 else 1.0

        round_offset = 1 if parsed and min(item["round"] for item in parsed) == 0 else 0
        xs = [item["round"] + round_offset for item in parsed]
        ys = [item["accuracy"] * scale for item in parsed]

        round_series.append((policy, xs, ys))
        policy_rows[policy] = parsed

        if all(item["logical"] is not None for item in parsed):
            logical = [item["logical"] for item in parsed]
            logical_series.append((
                policy,
                [v / 3600.0 for v in logical],
                ys,
            ))

        print(
            f"[DATA] {policy}: {len(parsed)} points; "
            f"last accuracy={ys[-1]:.2f}%"
        )

    plot_lines(
        round_series,
        "Training round",
        "Test accuracy (%)",
        "Test Accuracy vs. Training Round",
        out / "accuracy_vs_round.png",
    )

    plot_lines(
        logical_series,
        "Cumulative logical time (hours)",
        "Test accuracy (%)",
        "Test Accuracy vs. Logical Time",
        out / "accuracy_vs_logical_time.png",
    )

    # 实际墙钟时间从终端日志读取，避免误用逻辑时间。
    wall_series = []
    if args.log:
        log_path = Path(args.log)
        pattern = re.compile(
            r"\[(?P<policy>[^\]]+)\]\s+round\s+"
            r"(?P<round>\d+)/\d+\s+acc="
            r"(?P<acc>[-+]?\d*\.?\d+).*?"
            r"wall_time=(?P<wall>[-+]?\d*\.?\d+)s"
        )
        log_rows = {}

        with log_path.open("r", encoding="utf-8-sig",
                           errors="replace") as f:
            for line in f:
                match = pattern.search(line)
                if not match:
                    continue

                policy = match.group("policy")
                log_rows.setdefault(policy, []).append((
                    int(match.group("round")),
                    float(match.group("wall")),
                    float(match.group("acc")),
                ))

        for policy, values in sorted(log_rows.items()):
            values.sort(key=lambda item: item[0])
            rounds = [v[0] for v in values]
            wall = [v[1] / 60.0 for v in values]
            acc = [v[2] for v in values]
            if max(acc) <= 1.0:
                acc = [v * 100.0 for v in acc]
            wall_series.append((policy, wall, acc))

    if wall_series:
        plot_lines(
            wall_series,
            "Elapsed wall-clock time (minutes)",
            "Test accuracy (%)",
            "Test Accuracy vs. Wall-clock Time",
            out / "accuracy_vs_wall_time.png",
        )
    else:
        print(
            "[WARN] 日志中没有解析到逐轮 wall_time；"
            "请确认日志包含各轮的 acc=... wall_time=...s。"
        )

    print(f"\n图片输出目录：{out}")


if __name__ == "__main__":
    main()
