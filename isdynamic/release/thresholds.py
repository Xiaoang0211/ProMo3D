from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from .tables import write_table

BENCHMARKS = ("davis_all", "segtrackv2", "fbms59")
RULES = [("fixed", value) for value in (0.3, 0.4, 0.5, 0.6, 0.7)] + [("otsu", value) for value in (0.0, 0.1, 0.2, 0.3)]


def main():
    parser = argparse.ArgumentParser(description="Apply every global rule to every benchmark; no rule is selected on target data.")
    parser.add_argument("--source_mask_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    for name in ("davis", "segtrack", "fbms"):
        parser.add_argument(f"--{name}_root", required=True)
    args = parser.parse_args()
    benchmarks = args.benchmarks.split(",")
    rows = []
    for mode, value in RULES:
        variant = f"bilinear_upsample_{mode}_{value:.1f}"
        row = {"Rule": f"Fixed {value:.2f}" if mode == "fixed" else f"Otsu + {value:.1f}",
               "Default": mode == "otsu" and value == 0.2}
        for benchmark in benchmarks:
            subprocess.run([sys.executable, "-m", "isdynamic.viz.rebinarize_motion_masks",
                            "--mask_root", str(args.source_mask_root), "--output_mask_root", str(args.output_root / "masks"),
                            "--model", "pi3x", "--group", "inter_frame_block", "--benchmark", benchmark,
                            "--variant", "bilinear_upsample", "--output_variant", variant,
                            "--threshold_mode", mode, "--threshold", str(value if mode == "fixed" else 0.5),
                            "--threshold_offset", str(value if mode == "otsu" else 0.0),
                            "--no-save_numpy", "--no-save_visualizations", "--skip_existing"], check=True)
            output = args.output_root / "eval/pi3x/inter_frame_block" / variant / f"{benchmark}.json"
            subprocess.run([sys.executable, "-m", "isdynamic.viz.evaluate_motion_masks",
                            "--mask_root", str(args.output_root / "masks"), "--model", "pi3x", "--group", "inter_frame_block",
                            "--benchmark", benchmark, "--variant", variant, "--davis_root", args.davis_root,
                            "--segtrack_root", args.segtrack_root, "--fbms_root", args.fbms_root,
                            "--output_json", str(output)], check=True)
            metrics = json.loads(output.read_text())["metrics"]["J"]
            row[f"{benchmark} J-M"] = 100 * metrics["M"]
            row[f"{benchmark} J-R"] = 100 * metrics["R"]
        values = [row[f"{benchmark} J-M"] for benchmark in benchmarks]
        row.update(Mean=statistics.fmean(values), Std=statistics.pstdev(values))
        rows.append(row)
    for mode, values in (("fixed", (0.4, 0.5, 0.6)), ("otsu", (0.0, 0.1, 0.2))):
        selected = [rows[RULES.index((mode, value))] for value in values]
        row = {"Rule": f"Across-rule std.: {mode} {min(values):.1f}-{max(values):.1f}"}
        for key in [f"{benchmark} J-M" for benchmark in benchmarks] + ["Mean"]:
            row[key] = statistics.pstdev(item[key] for item in selected)
        rows.append(row)
    write_table(args.output_root, "threshold_robustness", rows)


if __name__ == "__main__":
    main()
