"""Export only completed, audited measurements into clearly labelled TeX tables."""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def checked(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("protocol") != "global_barrier_100":
        raise ValueError("old protocol results must not be inserted into the global-barrier paper")
    analysis = json.loads((path / "analysis.json").read_text(encoding="utf-8"))
    audits = json.loads((path / "audit.json").read_text(encoding="utf-8"))
    expected = len(config["seeds"])*len(config["policies"])
    if len(analysis["runs"]) != expected or len(audits) != expected or any(row["status"] != "passed" for row in audits):
        raise ValueError(f"incomplete or unaudited results: {path}")
    if any(row["rounds"] != config["rounds"] or row["max_epsilon"] > config["epsilon"]+1e-8
           for row in analysis["runs"]):
        raise ValueError(f"inconsistent rounds or privacy budget: {path}")
    return config, analysis


def pm(rows, key, scale=1., precision=2):
    values = [float(row[key])*scale for row in rows if row.get(key) is not None]
    if not values:
        return "--"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.
    return f"${mean:.{precision}f}\\pm{std:.{precision}f}$"


def table(caption, label, headers, rows):
    body = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
            "\\caption{"+caption+"}", "\\label{"+label+"}",
            "\\begin{tabular}{l"+"r"*(len(headers)-1)+"}", r"\toprule",
            " & ".join(headers)+r" \\", r"\midrule"]
    body.extend(" & ".join(row)+r" \\" for row in rows)
    body.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(body)


def export_studies(campaign):
    campaign = Path(campaign)
    destination = ROOT / "paper/results"
    destination.mkdir(exist_ok=True)
    rows, provenance = [], {}
    for name in ("ema_0", "ema_0.5", "ema_0.9", "ema_0.99", "ema_0_0.9", "ema_0.9_0"):
        config, analysis = checked(campaign/name)
        if config["clients"] != 6 or config["rounds"] != 20 or len(config["seeds"]) != 3:
            raise ValueError("pilot table settings differ from caption")
        records = analysis["runs"]
        rows.append([f"{config['gamma_d']:g} / {config['gamma_c']:g}",
                     pm(records,"final_test_accuracy",100), pm(records,"prediction_mae"),
                     pm(records,"system_time"), pm(records,"mode_changes")])
        provenance[name] = str((campaign/name).resolve())
    text = table("EMA ablation on Fashion-MNIST (pilot only): 6 clients, 2 Edges, 20 rounds, "
                 "1200 training and 500 test samples, three paired seeds. Zero smoothing retains "
                 "only the latest observation. Entries are mean and sample standard deviation.",
                 "tab:verified_pilot_ema", [r"$\gamma_d/\gamma_C$", r"Test acc. (\%)", "Prediction MAE", "System time (s)", "Mode changes"], rows)
    (destination/"pilot_ema.tex").write_text(text, encoding="utf-8")
    (destination/"pilot_provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")


def export_formal(campaign):
    campaign = Path(campaign)
    output = []
    provenance = {}
    labels = {"proposed":"Proposed", "fixed_fl_e":r"Fixed $FL_E$", "fixed_fl_ec":r"Fixed $FL_{EC}$",
              "fixed_fl_c":r"Fixed $FL_C$", "fixed_mixed":"Fixed mixed", "random_feasible":"Random feasible",
              "latency_only":"Time only", "deviation_only":"Deviation only"}
    for name, title in (("formal_fmnist","Fashion-MNIST / LeNet-5"), ("formal_cifar10","CIFAR-10 / frozen ResNet-18 head")):
        config, analysis = checked(campaign/name)
        if (config["rounds"],config["clients"],config["edges"],config["seeds"]) != (100,100,10,[42,43,44]):
            raise ValueError("formal settings do not match the paper")
        rows = []
        for policy, label in labels.items():
            records = [row for row in analysis["runs"] if row["policy"] == policy]
            if len(records) != 3:
                raise ValueError("formal method lacks all paired seeds")
            rows.append([label, pm(records,"final_test_accuracy",100), pm(records,"final_test_loss"),
                         pm(records,"system_time"),pm(records,"selection_sec"),
                         pm(records,"admission_fraction",100),pm(records,"mode_changes"),pm(records,"max_epsilon",precision=3)])
        output.append(table(title+": completed 100-round runs with 100 clients, 10 Edges and "
                            "seeds 42/43/44. Entries are mean and sample standard deviation; "
                            "modeled system time and measured selection time are separate.",
                            "tab:verified_"+name, ["Method", r"Acc. (\%)", "Loss", "System (s)",
                            "Search (s)",r"Admit (\%)","Changes",r"$\epsilon_{\max}$"], rows))
        provenance[name] = str((campaign/name).resolve())
    destination = ROOT/"paper/results"
    destination.mkdir(exist_ok=True)
    (destination/"formal_main.tex").write_text("\n".join(output),encoding="utf-8")
    (destination/"formal_provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    parser.add_argument("--kind",choices=("studies","formal"),required=True)
    args = parser.parse_args()
    (export_studies if args.kind=="studies" else export_formal)(args.campaign)
