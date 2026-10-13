from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _existing_summary(path: Path | None) -> Dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return _read_json(path)


def _ordered_union(preferred: Iterable[str], discovered: Iterable[str]) -> List[str]:
    items: List[str] = []
    seen = set()
    for value in preferred:
        if value and value not in seen:
            seen.add(value)
            items.append(value)
    for value in discovered:
        if value and value not in seen:
            seen.add(value)
            items.append(value)
    return items


def _feature_sort_key(feature_key: str) -> Tuple[str, int, str]:
    try:
        prefix, idx_str = feature_key.rsplit("_", 1)
        return prefix, int(idx_str), feature_key
    except ValueError:
        return feature_key, -1, feature_key


def _scan_metrics(results_dir: Path) -> List[Tuple[str, str, str, Path]]:
    records: List[Tuple[str, str, str, Path]] = []
    for metrics_path in sorted(results_dir.glob("pair_*/*/*/metrics.json")):
        feature_dir = metrics_path.parent
        group_dir = feature_dir.parent
        model_type_dir = group_dir.parent
        records.append((model_type_dir.name, group_dir.name, feature_dir.name, metrics_path))
    return records


def _infer_block_idx(feature_key: str) -> int:
    try:
        return int(feature_key.rsplit("_", 1)[1])
    except ValueError:
        return -1


def _infer_metric_name(metrics_blob: Dict[str, Any], existing_summary: Dict[str, Any] | None) -> str:
    if existing_summary is not None:
        metric_primary = existing_summary.get("metric_primary")
        if isinstance(metric_primary, str) and metric_primary:
            return metric_primary
        metric = existing_summary.get("metric")
        if isinstance(metric, str) and metric:
            return metric

    selection = metrics_blob.get("selection")
    if isinstance(selection, dict):
        metric = selection.get("metric")
        if isinstance(metric, str) and metric:
            return metric
    return "accuracy"


def _detect_summary_style(existing_summary: Dict[str, Any] | None) -> str:
    if existing_summary is None:
        return "results"
    if isinstance(existing_summary.get("results"), list):
        return "results"
    if isinstance(existing_summary.get("rows"), list):
        return "rows"
    return "results"


def _build_results_payload(
    *,
    existing_summary: Dict[str, Any] | None,
    scanned: List[Tuple[str, str, str, Path]],
) -> Dict[str, Any]:
    if not scanned:
        raise SystemExit("No metrics.json files found under results_dir.")

    preferred_groups = list(existing_summary.get("groups", [])) if existing_summary else []
    preferred_model_types = list(existing_summary.get("model_types", [])) if existing_summary else []
    discovered_groups = [group for _, group, _, _ in scanned]
    discovered_model_types = [model_type for model_type, _, _, _ in scanned]
    groups = _ordered_union(preferred_groups, discovered_groups)
    model_types = _ordered_union(preferred_model_types, discovered_model_types)

    metric_name = _infer_metric_name(_read_json(scanned[0][3]), existing_summary)
    results: List[Dict[str, Any]] = []
    for model_type, group, feature_key, metrics_path in scanned:
        metrics_blob = _read_json(metrics_path)
        run_dir = metrics_path.parent
        results.append(
            {
                "group": group,
                "block_idx": _infer_block_idx(feature_key),
                "feature_key": feature_key,
                "model_type": model_type,
                "metric_primary": metric_name,
                "best": metrics_blob.get("best", {}),
                "best_tests": metrics_blob.get("best_tests", {}),
                "run_dir": str(run_dir.resolve()),
            }
        )

    group_rank = {group: idx for idx, group in enumerate(groups)}
    model_rank = {model_type: idx for idx, model_type in enumerate(model_types)}
    results.sort(
        key=lambda item: (
            group_rank.get(str(item["group"]), 10**6),
            model_rank.get(str(item["model_type"]), 10**6),
            int(item["block_idx"]),
            str(item["feature_key"]),
        )
    )

    payload = dict(existing_summary or {})
    payload["metric_primary"] = metric_name
    payload["groups"] = groups
    payload["model_types"] = model_types
    payload["results"] = results
    payload.pop("rows", None)
    return payload


def _build_rows_payload(
    *,
    existing_summary: Dict[str, Any] | None,
    scanned: List[Tuple[str, str, str, Path]],
) -> Dict[str, Any]:
    if not scanned:
        raise SystemExit("No metrics.json files found under results_dir.")

    preferred_groups = list(existing_summary.get("groups", [])) if existing_summary else []
    preferred_model_types = list(existing_summary.get("model_types", [])) if existing_summary else []
    discovered_groups = [group for _, group, _, _ in scanned]
    discovered_model_types = [model_type for model_type, _, _, _ in scanned]
    groups = _ordered_union(preferred_groups, discovered_groups)
    model_types = _ordered_union(preferred_model_types, discovered_model_types)

    metric_name = _infer_metric_name(_read_json(scanned[0][3]), existing_summary)
    rows: List[Dict[str, Any]] = []
    for model_type, group, feature_key, metrics_path in scanned:
        metrics_blob = _read_json(metrics_path)
        best_val = metrics_blob.get("best_val")
        if isinstance(best_val, dict):
            best_val = best_val.get(metric_name)
        rows.append(
            {
                "group": group,
                "feature_key": feature_key,
                "model_type": model_type,
                "metric": metric_name,
                "best_val": best_val,
                "best_tests": metrics_blob.get("best_tests", {}),
                "metrics_path": str(metrics_path.resolve()),
            }
        )

    group_rank = {group: idx for idx, group in enumerate(groups)}
    model_rank = {model_type: idx for idx, model_type in enumerate(model_types)}
    rows.sort(
        key=lambda item: (
            group_rank.get(str(item["group"]), 10**6),
            model_rank.get(str(item["model_type"]), 10**6),
            _feature_sort_key(str(item["feature_key"])),
        )
    )

    payload = dict(existing_summary or {})
    payload["metric"] = metric_name
    payload["groups"] = groups
    payload["model_types"] = model_types
    payload["rows"] = rows
    payload.pop("results", None)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild blockwise summary.json from existing metrics.json files.")
    parser.add_argument("--results_dir", type=Path, required=True, help="Blockwise result root containing pair_*/*/*/metrics.json.")
    parser.add_argument("--summary_json", type=Path, default=None, help="Summary path to read/write. Defaults to <results_dir>/summary.json.")
    parser.add_argument("--style", choices=["auto", "results", "rows"], default="auto", help="Summary schema to emit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise SystemExit(f"results_dir not found: {results_dir}")

    summary_path = args.summary_json.resolve() if args.summary_json else (results_dir / "summary.json")
    existing_summary = _existing_summary(summary_path)
    scanned = _scan_metrics(results_dir)
    style = _detect_summary_style(existing_summary) if args.style == "auto" else args.style

    if style == "rows":
        payload = _build_rows_payload(existing_summary=existing_summary, scanned=scanned)
    else:
        payload = _build_results_payload(existing_summary=existing_summary, scanned=scanned)

    _write_json(summary_path, payload)
    print(f"Rebuilt summary: {summary_path}")
    print(f"groups: {payload.get('groups', [])}")
    print(f"model_types: {payload.get('model_types', [])}")
    if "results" in payload:
        print(f"rows_written: {len(payload['results'])}")
    else:
        print(f"rows_written: {len(payload['rows'])}")


if __name__ == "__main__":
    main()
