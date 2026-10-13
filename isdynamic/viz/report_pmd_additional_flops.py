from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


MODEL_CHOICES = [
    "pi3",
    "pi3x",
    "vggt",
    "vggt_omega",
    "depth_anything_3",
    "map_anything",
    "dust3r",
]

MAPANYTHING_RESOLUTION_MAPPINGS: Dict[int, Dict[float, Tuple[int, int]]] = {
    518: {
        1.000: (518, 518),
        1.321: (518, 392),
        1.542: (518, 336),
        1.762: (518, 294),
        2.056: (518, 252),
        3.083: (518, 168),
        0.757: (392, 518),
        0.649: (336, 518),
        0.567: (294, 518),
        0.486: (252, 518),
    },
    512: {
        1.000: (512, 512),
        1.333: (512, 384),
        1.524: (512, 336),
        1.778: (512, 288),
        2.000: (512, 256),
        3.200: (512, 160),
        0.750: (384, 512),
        0.656: (336, 512),
        0.562: (288, 512),
        0.500: (256, 512),
    },
    504: {
        1.000: (504, 504),
        1.333: (504, 378),
        1.565: (504, 322),
        1.800: (504, 280),
        2.118: (504, 238),
        3.273: (504, 154),
        0.750: (378, 504),
        0.639: (322, 504),
        0.556: (280, 504),
        0.472: (238, 504),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report PMD additional parameters and FLOPs for the current motion-mask inference path. "
            "The FLOPs account for probe scoring plus block/pair averaging, and exclude the frozen backbone."
        )
    )
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--probe_root", type=str, required=True)
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--groups", type=str, default="")
    parser.add_argument("--context_size", type=int, default=8)
    parser.add_argument("--rank_override", type=int, default=0)
    parser.add_argument("--nominal_aspect_ratio", type=float, default=1.0)
    parser.add_argument("--process_res", type=int, default=518)
    parser.add_argument("--resolution_set", type=int, default=518, choices=[504, 512, 518])
    parser.add_argument("--image_size", type=int, default=512, choices=[224, 512])
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def _normalize_model_name(name: str) -> str:
    aliases = {
        "pi3": "pi3",
        "pi3x": "pi3x",
        "vggt": "vggt",
        "vggt_omega": "vggt_omega",
        "depth_anything_3": "depth_anything_3",
        "da3": "depth_anything_3",
        "map_anything": "map_anything",
        "mapanything": "map_anything",
        "dust3r": "dust3r",
    }
    lowered = name.strip().lower()
    if lowered not in aliases:
        raise ValueError(f"Unsupported model: {name}")
    return aliases[lowered]


def _group_list_for_args(args: argparse.Namespace) -> List[str]:
    if args.groups:
        groups = [item.strip() for item in args.groups.split(",") if item.strip()]
        if groups:
            return groups
    if args.group:
        return [args.group.strip()]

    defaults = {
        "pi3": ["inter_frame_block"],
        "pi3x": ["inter_frame_block"],
        "vggt": ["inter_frame_block"],
        "vggt_omega": ["inter_frame_block"],
        "depth_anything_3": ["inter_frame_block"],
        "map_anything": ["inter_frame_block"],
        "dust3r": ["dec_view1", "head_in_view1"],
    }
    return defaults[_normalize_model_name(args.model)]


def _nearest_multiple(x: int, patch: int) -> int:
    down = (x // patch) * patch
    up = down + patch
    return up if abs(up - x) <= abs(x - down) else down


def _count_patch_tokens(width: int, height: int, patch_size: int) -> Tuple[int, int, int]:
    if width % patch_size != 0 or height % patch_size != 0:
        raise ValueError(f"Spatial size {(width, height)} is not divisible by patch size {patch_size}")
    grid_w = width // patch_size
    grid_h = height // patch_size
    return grid_h, grid_w, grid_h * grid_w


def _square_target_tokens(target_size: int, patch_size: int) -> Tuple[int, int, int]:
    return _count_patch_tokens(target_size, target_size, patch_size)


def _depth_anything_tokens(process_res: int, patch_size: int, aspect_ratio: float) -> Tuple[int, int, int]:
    if aspect_ratio <= 0:
        raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")
    if aspect_ratio >= 1.0:
        width = process_res
        height = int(round(width / aspect_ratio))
    else:
        height = process_res
        width = int(round(height * aspect_ratio))
    width = max(1, _nearest_multiple(width, patch_size))
    height = max(1, _nearest_multiple(height, patch_size))
    return _count_patch_tokens(width, height, patch_size)


def _map_anything_tokens(resolution_set: int, aspect_ratio: float, patch_size: int) -> Tuple[int, int, int]:
    mapping = MAPANYTHING_RESOLUTION_MAPPINGS[resolution_set]
    target_key = min(mapping.keys(), key=lambda key: abs(key - aspect_ratio))
    width, height = mapping[target_key]
    return _count_patch_tokens(width, height, patch_size)


def _stereo_tokens(image_size: int, patch_size: int, aspect_ratio: float) -> Tuple[int, int, int]:
    if image_size == 224:
        return _count_patch_tokens(224, 224, patch_size)
    if aspect_ratio <= 0:
        raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")
    if aspect_ratio >= 1.0:
        width = image_size
        height = int(round(width / aspect_ratio))
    else:
        height = image_size
        width = int(round(height * aspect_ratio))
    width = max(patch_size, (width // patch_size) * patch_size)
    height = max(patch_size, (height // patch_size) * patch_size)
    return _count_patch_tokens(width, height, patch_size)


def infer_nominal_token_grid(args: argparse.Namespace) -> Tuple[int, int, int]:
    model = _normalize_model_name(args.model)
    aspect_ratio = float(args.nominal_aspect_ratio)
    if model in {"pi3", "pi3x", "vggt"}:
        return _square_target_tokens(target_size=518, patch_size=14)
    if model == "vggt_omega":
        return _square_target_tokens(target_size=512, patch_size=16)
    if model == "depth_anything_3":
        return _depth_anything_tokens(process_res=int(args.process_res), patch_size=14, aspect_ratio=aspect_ratio)
    if model == "map_anything":
        return _map_anything_tokens(
            resolution_set=int(args.resolution_set),
            aspect_ratio=aspect_ratio,
            patch_size=14,
        )
    if model == "dust3r":
        return _stereo_tokens(image_size=int(args.image_size), patch_size=16, aspect_ratio=aspect_ratio)
    raise ValueError(f"Unsupported model: {model}")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_probe_group_stats(
    probe_root: Path,
    groups: Sequence[str],
    *,
    rank_override: int = 0,
) -> Tuple[List[dict], int]:
    entries: List[dict] = []

    for group in groups:
        group_root = probe_root / group
        if not group_root.exists():
            raise FileNotFoundError(f"Probe group directory not found: {group_root}")
        cfg_paths = sorted(group_root.glob("*/config.json"))
        if not cfg_paths:
            raise FileNotFoundError(f"No config.json files found under probe group: {group_root}")

        sample_cfg = _load_json(cfg_paths[0])
        model_cfg = sample_cfg["model"]
        input_dim = int(model_cfg["input_dim"])
        rank = int(rank_override or model_cfg.get("rank", 0))
        model_type = str(model_cfg["model_type"])
        if "lowrank" not in model_type:
            raise ValueError(
                f"PMD FLOPs reporter currently expects low-rank quadratic probes, got {model_type} under {group_root}"
            )

        entries.append(
            {
                "group": group,
                "num_blocks": len(cfg_paths),
                "input_dim": input_dim,
                "rank": rank,
                "model_type": model_type,
                "sample_config": str(cfg_paths[0]),
            }
        )

    total_blocks = sum(int(item["num_blocks"]) for item in entries)
    return entries, total_blocks


def _per_token_pair_flops_lowrank(input_dim: int, rank: int) -> int:
    # Exact scalar-op count for the current implementation in common_models.py:
    # 4 matvecs with k outputs and d inputs -> 4 * k * (2d - 1)
    # 2 elementwise mults, 1 vector add, reduction sum, final scale+bias -> +2
    return int(8 * input_dim * rank + 2)


def _single_probe_params_lowrank(input_dim: int, rank: int) -> int:
    return int(2 * input_dim * rank + 1)


def _nan_safe_sum(values: Iterable[int]) -> int:
    total = 0
    for value in values:
        total += int(value)
    return total


def build_report(args: argparse.Namespace) -> dict:
    model = _normalize_model_name(args.model)
    probe_root = Path(args.probe_root)
    if not probe_root.exists():
        raise FileNotFoundError(f"probe_root not found: {probe_root}")

    groups = _group_list_for_args(args)
    group_entries, total_blocks = collect_probe_group_stats(
        probe_root=probe_root,
        groups=groups,
        rank_override=int(args.rank_override),
    )
    grid_h, grid_w, token_count = infer_nominal_token_grid(args)

    pairwise_references = max(1, int(args.context_size) - 1) if model == "dust3r" else 1
    total_score_maps = 0
    total_loaded_probe_params = 0
    probe_scoring_flops = 0
    per_group_reports: List[dict] = []
    input_dims = set()
    ranks = set()
    model_types = set()
    single_probe_params_set = set()

    for entry in group_entries:
        input_dim = int(entry["input_dim"])
        rank = int(entry["rank"])
        num_blocks = int(entry["num_blocks"])
        model_type = str(entry["model_type"])
        input_dims.add(input_dim)
        ranks.add(rank)
        model_types.add(model_type)

        single_probe_params = _single_probe_params_lowrank(input_dim=input_dim, rank=rank)
        per_token_pair_flops = _per_token_pair_flops_lowrank(input_dim=input_dim, rank=rank)
        group_score_maps = num_blocks * pairwise_references
        group_probe_scoring_flops = group_score_maps * token_count * per_token_pair_flops

        total_score_maps += group_score_maps
        total_loaded_probe_params += num_blocks * single_probe_params
        probe_scoring_flops += group_probe_scoring_flops
        single_probe_params_set.add(single_probe_params)
        per_group_reports.append(
            {
                **entry,
                "single_probe_extra_params": int(single_probe_params),
                "group_loaded_probe_params": int(num_blocks * single_probe_params),
                "per_token_pair_flops_current_impl": int(per_token_pair_flops),
                "group_total_score_maps": int(group_score_maps),
                "group_probe_scoring_flops": int(group_probe_scoring_flops),
                "group_probe_scoring_gflops": float(group_probe_scoring_flops / 1e9),
            }
        )

    pair_averaging_flops = total_blocks * pairwise_references * token_count
    block_averaging_flops = total_blocks * token_count
    total_additional_flops = probe_scoring_flops + pair_averaging_flops + block_averaging_flops

    input_dim_summary: int | List[int]
    rank_summary: int | List[int]
    model_type_summary: str | List[str]
    single_probe_param_summary: int | List[int]
    if len(input_dims) == 1:
        input_dim_summary = int(next(iter(input_dims)))
    else:
        input_dim_summary = sorted(int(value) for value in input_dims)
    if len(ranks) == 1:
        rank_summary = int(next(iter(ranks)))
    else:
        rank_summary = sorted(int(value) for value in ranks)
    if len(model_types) == 1:
        model_type_summary = str(next(iter(model_types)))
    else:
        model_type_summary = sorted(model_types)
    if len(single_probe_params_set) == 1:
        single_probe_param_summary = int(next(iter(single_probe_params_set)))
    else:
        single_probe_param_summary = sorted(int(value) for value in single_probe_params_set)

    report = {
        "model": model,
        "probe_root": str(probe_root),
        "groups": groups,
        "group_stats": per_group_reports,
        "rank_source": "override" if int(args.rank_override) > 0 else "config",
        "input_dim": input_dim_summary,
        "rank": rank_summary,
        "model_type": model_type_summary,
        "single_probe_extra_params": single_probe_param_summary,
        "total_loaded_probe_params": int(total_loaded_probe_params),
        "nominal_aspect_ratio": float(args.nominal_aspect_ratio),
        "nominal_token_grid": {
            "height": int(grid_h),
            "width": int(grid_w),
            "num_tokens": int(token_count),
        },
        "context_size": int(args.context_size),
        "pairwise_references": int(pairwise_references),
        "num_blocks": int(total_blocks),
        "total_score_maps": int(total_score_maps),
        "per_token_pair_flops_current_impl": int(per_token_pair_flops),
        "probe_scoring_flops": int(probe_scoring_flops),
        "pair_averaging_flops": int(pair_averaging_flops),
        "block_averaging_flops": int(block_averaging_flops),
        "total_additional_flops": int(total_additional_flops),
        "probe_scoring_gflops": float(probe_scoring_flops / 1e9),
        "total_additional_gflops": float(total_additional_flops / 1e9),
        "assumptions": {
            "flops_scope": "PMD-only additional compute beyond the frozen backbone.",
            "counting_convention": "Scalar multiply/add operations are counted explicitly from the current low-rank probe forward path.",
            "nominal_input": "Reported on a nominal single target frame using the model's default preprocessing settings.",
            "stereo_note": (
                "DUSt3R multiplies PMD scoring by context_size-1 because the target is paired with every reference frame."
            ),
            "rank_override_note": (
                "When rank_source=override, the FLOPs/params are a normalized what-if estimate and not proof that a trained checkpoint at that rank exists."
            ),
        },
    }
    if model == "depth_anything_3":
        report["process_res"] = int(args.process_res)
    if model == "map_anything":
        report["resolution_set"] = int(args.resolution_set)
    if model == "dust3r":
        report["image_size"] = int(args.image_size)
    return report


def main() -> None:
    args = parse_args()
    report = build_report(args)

    output_json = Path(args.output_json) if args.output_json else None
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
