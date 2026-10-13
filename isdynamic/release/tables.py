from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def write_table(root, name, rows):
    if not rows:
        raise FileNotFoundError(f"No completed results found for {name} under {root}")
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (out / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[key]:.4f}" if isinstance(row.get(key), float)
                                        else str(row.get(key, "--")) for key in keys) + " |")
    (out / f"{name}.md").write_text("\n".join(lines) + "\n")
    print(f"[table] {out / (name + '.csv')}")


def mask_table(root):
    rows = []
    for path in sorted((root / "eval").glob("*/*/*/*.json")):
        data = json.loads(path.read_text())
        model, group, variant = path.parts[-4:-1]
        rows.append({"Backbone": model, "Token group": group, "Variant": variant,
                     "Benchmark": path.stem, "J-M": 100 * data["metrics"]["J"]["M"],
                     "J-R": 100 * data["metrics"]["J"]["R"], "Source": str(path)})
    write_table(root, "motion_masks", rows)


def controls_table(root):
    rows = []
    for model_root in sorted((root / "controls").glob("*")):
        for experiment in sorted(model_root.glob("*")):
            metrics = [json.loads(path.read_text())["best_val"]
                       for path in sorted(experiment.glob("pair_lowrank_quadratic/*/*/metrics.json"))]
            if not metrics:
                continue
            accuracies, baselines, balanced = [], [], []
            for metric in metrics:
                tp, fn, fp, tn = (float(metric[key]) for key in ("tp", "fn", "fp", "tn"))
                total = tp + fn + fp + tn
                accuracies.append(float(metric["accuracy"]))
                baselines.append((tp + fn) / total)
                balanced.append((tp / (tp + fn) + tn / (tn + fp)) / 2)
            acc, baseline = statistics.fmean(accuracies), statistics.fmean(baselines)
            rows.append({"Backbone": model_root.name, "Experiment": experiment.name,
                         "Split": "YouTube-VOS val", "Blocks": len(metrics),
                         "Pair accuracy (%)": 100 * acc,
                         "Always-same baseline (%)": 100 * baseline,
                         "Gain (pp)": 100 * (acc - baseline),
                         "Balanced accuracy (%)": 100 * statistics.fmean(balanced),
                         "Source": str(experiment)})
    write_table(root, "probe_controls", rows)


def runtime_table(root):
    rows = []
    runtime = root / "runtime"
    pmd = runtime / "pi3x_sam2_feature_cache/summary.csv"
    if not pmd.exists():
        pmd = runtime / "pi3x/summary.csv"
    for path in (pmd, runtime / "flowsam/summary.csv", runtime / "seganymo/summary.csv"):
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                method = row.get("Method") or row.get("method")
                if path.parent.name == "flowsam" and not method.startswith("Flow-SAM"):
                    continue
                rows.append({"Method": method,
                             "Mean latency (ms/frame)": float(row.get("mean_ms") or row.get("amortized_ms_per_frame")),
                             "Peak VRAM (GiB)": float(row["peak_vram_gib"]),
                             "Frames": int(row.get("Measured frames") or row.get("num_frames")),
                             "Source": str(path)})
    geomotion = root / "runtime/geomotion/summary.json"
    if geomotion.exists():
        payload = json.loads(geomotion.read_text())
        for key, method in (("geomotion_coarse", "GeoMotion coarse"), ("geomotion_sam2", "GeoMotion + SAM2")):
            stats = payload[key]
            rows.append({"Method": method,
                         "Mean latency (ms/frame)": stats["mean_ms_per_frame"],
                         "Peak VRAM (GiB)": stats["peak_vram_gib"],
                         "Frames": payload["protocol"]["measured_frames"], "Source": str(geomotion)})
    write_table(root, "runtime", rows)


def ranks_table(root):
    rows = []
    for model in sorted((root / "probes").glob("*")):
        for experiment in sorted(model.glob("rank_*")):
            files = sorted(experiment.glob("pair_lowrank_quadratic/*/*/metrics.json"))
            if not files:
                continue
            payloads = [json.loads(path.read_text()) for path in files]
            parameters = sum(2 * int(item['config']['model']['rank']) * int(item['config']['model']['input_dim']) + 1 for item in payloads)
            full = sum(int(item['config']['model']['input_dim']) ** 2 + 1 for item in payloads)
            row = {"Backbone": model.name, "Rank": payloads[0]['config']['model']['rank'],
                   "Blocks": len(files), "Params": parameters, "Params/full (%)": 100 * parameters/full}
            for key, split in (("Val", "best_val"), ("DAVIS-all", "test_davis")):
                metrics = [item[split] if split == "best_val" else item['best_tests'][split] for item in payloads]
                gains = []
                for metric in metrics:
                    tp, fn, fp, tn = (float(metric[field]) for field in ('tp','fn','fp','tn'))
                    baseline = max(tp+fn,tn+fp)/(tp+fn+tn+fp)
                    gains.append((metric['accuracy']-baseline)/(1-baseline))
                row[f"{key} best pair accuracy (%)"] = 100*max(item['accuracy'] for item in metrics)
                row[f"{key} best NG (%)"] = 100*max(gains)
            rows.append(row)
    write_table(root, "rank_capacity", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("main", "controls", "runtime", "ranks"), default="main")
    args = parser.parse_args()
    {"main": mask_table, "controls": controls_table, "runtime": runtime_table, "ranks": ranks_table}[args.kind](args.root)


if __name__ == "__main__":
    main()
