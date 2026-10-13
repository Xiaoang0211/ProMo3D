#!/usr/bin/env python3
import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isdynamic.core.feature_io import load_token_npz
from isdynamic.core.common_config import load_probe_config
from isdynamic.core.common_models import build_probe, load_attention_matrix
from isdynamic.core.common_trainer import (
    PairTokenProbeDataset,
    TokenProbeDataset,
    evaluate,
    evaluate_pairwise_batchwise,
    train_probe,
)


def apply_value_projection_to_data(
    data,
    value_matrix: torch.Tensor,
):
    features, labels, valid, groups = data
    if value_matrix.ndim != 2 or value_matrix.shape[0] != value_matrix.shape[1]:
        raise ValueError(f"value feature matrix must be square [C,C], got {tuple(value_matrix.shape)}")
    if features.shape[1] != value_matrix.shape[0]:
        raise ValueError(
            "Feature dimension mismatch for value projection: "
            f"features={tuple(features.shape)}, matrix={tuple(value_matrix.shape)}"
        )

    projected = features.float() @ value_matrix.float().transpose(0, 1)
    return projected, labels, valid, groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train token-level IsDynamic probe.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "configs" / "vggt" / "isdynamic_probe.json"),
        help="Path to JSON/YAML config.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and update only the requested test split(s) from an existing checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path used with --eval_only. Defaults to output/save_name.",
    )
    return parser.parse_args()


def evaluate_tests_only(
    model,
    cfg,
    test_data=None,
    named_test_data=None,
    checkpoint_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pairwise = bool(getattr(model, "is_pairwise", False))
    eval_batch_size = (
        int(cfg.train.eval_batch_size)
        if cfg.train.eval_batch_size is not None and int(cfg.train.eval_batch_size) > 0
        else int(cfg.train.batch_size)
    )

    best_path = Path(checkpoint_path) if checkpoint_path is not None else Path(cfg.output.out_dir) / cfg.output.save_name
    if not best_path.exists():
        raise FileNotFoundError(f"Eval-only checkpoint not found: {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))

    metrics_path = Path(cfg.output.out_dir) / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    else:
        report = {
            "best": None,
            "best_train": None,
            "best_val": None,
            "best_tests": {},
            "last": None,
            "selection": None,
        }

    ref_metrics = report.get("best_val") or report.get("best") or {}
    best_epoch = float(ref_metrics.get("epoch", cfg.train.epochs))
    best_lr = float(ref_metrics.get("learning_rate", cfg.train.learning_rate))
    best_train_loss = float(ref_metrics.get("train_loss", 0.0))

    named_test_datasets = {}
    best_test_metrics = None
    best_named_test_metrics = {}
    test_samples = report.get("test_samples", 0)
    test_samples_by_name = dict(report.get("test_samples_by_name") or {})

    if pairwise:
        eval_all_pairs = cfg.train.pair_eval_all_pairs
        test_dataset = None
        if test_data is not None:
            test_dataset = PairTokenProbeDataset(
                *test_data,
                num_pairs=cfg.train.pair_test_samples,
                same_group=cfg.train.pair_same_group,
                balance=False,
                label_mode=cfg.train.pair_label_mode,
                max_tokens_per_group=cfg.train.pair_tokens_per_group,
                seed=cfg.train.seed + 2,
                all_pairs=eval_all_pairs,
            )
        if named_test_data is not None:
            for offset, (test_name, test_tuple) in enumerate(named_test_data.items(), start=3):
                named_test_datasets[test_name] = PairTokenProbeDataset(
                    *test_tuple,
                    num_pairs=cfg.train.pair_test_samples,
                    same_group=cfg.train.pair_same_group,
                    balance=False,
                    label_mode=cfg.train.pair_label_mode,
                    max_tokens_per_group=cfg.train.pair_tokens_per_group,
                    seed=cfg.train.seed + offset,
                    all_pairs=eval_all_pairs,
                )
        if cfg.train.pair_eval_batchwise:
            if test_dataset is not None:
                best_test_metrics = evaluate_pairwise_batchwise(model, test_dataset, device=device, batch_size=eval_batch_size)
            best_named_test_metrics = {
                test_name: evaluate_pairwise_batchwise(model, dataset, device=device, batch_size=eval_batch_size)
                for test_name, dataset in named_test_datasets.items()
            }
        else:
            if test_dataset is not None:
                best_test_metrics = evaluate(
                    model,
                    DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=torch.cuda.is_available()),
                    device=device,
                    pairwise=True,
                )
            best_named_test_metrics = {
                test_name: evaluate(
                    model,
                    DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=torch.cuda.is_available()),
                    device=device,
                    pairwise=True,
                )
                for test_name, dataset in named_test_datasets.items()
            }
        if test_dataset is not None:
            test_samples = len(test_dataset)
        for test_name, dataset in named_test_datasets.items():
            test_samples_by_name[test_name] = len(dataset)
    else:
        test_dataset = TokenProbeDataset(test_data[0], test_data[1], test_data[2]) if test_data is not None else None
        if named_test_data is not None:
            for test_name, test_tuple in named_test_data.items():
                named_test_datasets[test_name] = TokenProbeDataset(test_tuple[0], test_tuple[1], test_tuple[2])
        if test_dataset is not None:
            best_test_metrics = evaluate(
                model,
                DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=torch.cuda.is_available()),
                device=device,
                pairwise=False,
            )
            test_samples = len(test_dataset)
        best_named_test_metrics = {
            test_name: evaluate(
                model,
                DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=torch.cuda.is_available()),
                device=device,
                pairwise=False,
            )
            for test_name, dataset in named_test_datasets.items()
        }
        for test_name, dataset in named_test_datasets.items():
            test_samples_by_name[test_name] = len(dataset)

    for metrics in ([best_test_metrics] if best_test_metrics is not None else []) + list(best_named_test_metrics.values()):
        metrics["epoch"] = best_epoch
        metrics["learning_rate"] = best_lr
        metrics["train_loss"] = best_train_loss

    existing_best_tests = dict(report.get("best_tests") or {})
    if best_test_metrics is not None:
        existing_best_tests["test"] = best_test_metrics
        report["test_samples"] = test_samples
    existing_best_tests.update(best_named_test_metrics)
    report["best_tests"] = existing_best_tests
    report["test_samples_by_name"] = test_samples_by_name
    report["checkpoint"] = str(best_path)
    report["pairwise"] = pairwise
    report["eval_only"] = True

    if report.get("config") is None:
        report["config"] = asdict(cfg)
    else:
        report.setdefault("config", {})
        dataset_cfg = report["config"].setdefault("dataset", {})
        if getattr(cfg.dataset, "test_npz", None):
            dataset_cfg["test_npz"] = cfg.dataset.test_npz
        merged_test_npzs = dict(dataset_cfg.get("test_npzs") or {})
        merged_test_npzs.update(getattr(cfg.dataset, "test_npzs", {}) or {})
        dataset_cfg["test_npzs"] = merged_test_npzs

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    args = parse_args()
    cfg = load_probe_config(args.config)

    if args.eval_only:
        test_data = None
        named_test_data = {}
        if cfg.dataset.test_npz:
            test_data = load_token_npz(
                cfg.dataset.test_npz,
                feature_key=cfg.dataset.feature_key,
                label_key=cfg.dataset.label_key,
                valid_key=cfg.dataset.valid_key,
            )
        if cfg.dataset.test_npzs:
            for test_name, test_path in (sorted(cfg.dataset.test_npzs.items()) if isinstance(cfg.dataset.test_npzs, dict) else cfg.dataset.test_npzs.items()):
                named_test_data[test_name] = load_token_npz(
                    test_path,
                    feature_key=cfg.dataset.feature_key,
                    label_key=cfg.dataset.label_key,
                    valid_key=cfg.dataset.valid_key,
                )

        fixed_matrix_model_types = {
            "pair_attention",
            "pair_value",
            "pair_attention_biased_quadratic",
            "pair_value_biased_quadratic",
            "pair_attention_biased_lowrank_quadratic",
            "pair_value_biased_lowrank_quadratic",
            "pair_attention_biased_lowrank_quadratic_strict",
            "pair_value_biased_lowrank_quadratic_strict",
        }

        if cfg.model.use_value_features:
            if cfg.model.model_type in fixed_matrix_model_types:
                raise ValueError(
                    "use_value_features is only supported for trainable probes, not fixed-matrix probe types."
                )
            if cfg.model.value_feature_matrix_path is None:
                raise ValueError("use_value_features=True requires model.value_feature_matrix_path.")

            value_feature_matrix = load_attention_matrix(
                cfg.model.value_feature_matrix_path,
                key=cfg.model.value_feature_matrix_key,
            )
            if test_data is not None:
                test_data = apply_value_projection_to_data(test_data, value_feature_matrix)
            if named_test_data:
                named_test_data = {
                    test_name: apply_value_projection_to_data(test_dataset, value_feature_matrix)
                    for test_name, test_dataset in named_test_data.items()
                }

        fixed_matrix = None
        if cfg.model.model_type in {
            "pair_attention",
            "pair_attention_biased_quadratic",
            "pair_attention_biased_lowrank_quadratic",
            "pair_attention_biased_lowrank_quadratic_strict",
        }:
            if cfg.model.attention_matrix_path is None:
                raise ValueError(
                    f"{cfg.model.model_type} requires model.attention_matrix_path in config."
                )
            fixed_matrix = load_attention_matrix(
                cfg.model.attention_matrix_path,
                key=cfg.model.attention_matrix_key,
            )
        if cfg.model.model_type in {
            "pair_value",
            "pair_value_biased_quadratic",
            "pair_value_biased_lowrank_quadratic",
            "pair_value_biased_lowrank_quadratic_strict",
        }:
            if cfg.model.value_matrix_path is None:
                raise ValueError(
                    f"{cfg.model.model_type} requires model.value_matrix_path in config."
                )
            fixed_matrix = load_attention_matrix(
                cfg.model.value_matrix_path,
                key=cfg.model.value_matrix_key,
            )

        model = build_probe(
            model_type=cfg.model.model_type,
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
            rank=cfg.model.rank,
            attention_matrix=fixed_matrix,
        )
        report = evaluate_tests_only(
            model,
            cfg=cfg,
            test_data=test_data,
            named_test_data=named_test_data or None,
            checkpoint_path=args.checkpoint,
        )
        print(json.dumps(report.get("best_tests") or {}, indent=2))
        print(f"Saved checkpoint to: {report['checkpoint']}")
        print(f"Saved metrics to: {Path(cfg.output.out_dir) / 'metrics.json'}")
        return

    train_data = load_token_npz(
        cfg.dataset.train_npz,
        feature_key=cfg.dataset.feature_key,
        label_key=cfg.dataset.label_key,
        valid_key=cfg.dataset.valid_key,
    )
    val_data = load_token_npz(
        cfg.dataset.val_npz,
        feature_key=cfg.dataset.feature_key,
        label_key=cfg.dataset.label_key,
        valid_key=cfg.dataset.valid_key,
    )
    test_data = None
    named_test_data = {}
    if cfg.dataset.test_npz:
        test_data = load_token_npz(
            cfg.dataset.test_npz,
            feature_key=cfg.dataset.feature_key,
            label_key=cfg.dataset.label_key,
            valid_key=cfg.dataset.valid_key,
        )
    if cfg.dataset.test_npzs:
        for test_name, test_path in cfg.dataset.test_npzs.items():
            named_test_data[test_name] = load_token_npz(
                test_path,
                feature_key=cfg.dataset.feature_key,
                label_key=cfg.dataset.label_key,
                valid_key=cfg.dataset.valid_key,
            )

    fixed_matrix_model_types = {
        "pair_attention",
        "pair_value",
        "pair_attention_biased_quadratic",
        "pair_value_biased_quadratic",
        "pair_attention_biased_lowrank_quadratic",
        "pair_value_biased_lowrank_quadratic",
        "pair_attention_biased_lowrank_quadratic_strict",
        "pair_value_biased_lowrank_quadratic_strict",
    }

    if cfg.model.use_value_features:
        if cfg.model.model_type in fixed_matrix_model_types:
            raise ValueError(
                "use_value_features is only supported for trainable probes, not fixed-matrix probe types."
            )
        if cfg.model.value_feature_matrix_path is None:
            raise ValueError("use_value_features=True requires model.value_feature_matrix_path.")

        value_feature_matrix = load_attention_matrix(
            cfg.model.value_feature_matrix_path,
            key=cfg.model.value_feature_matrix_key,
        )
        train_data = apply_value_projection_to_data(train_data, value_feature_matrix)
        val_data = apply_value_projection_to_data(val_data, value_feature_matrix)
        if test_data is not None:
            test_data = apply_value_projection_to_data(test_data, value_feature_matrix)
        if named_test_data:
            named_test_data = {
                test_name: apply_value_projection_to_data(test_dataset, value_feature_matrix)
                for test_name, test_dataset in named_test_data.items()
            }

    fixed_matrix = None
    if cfg.model.model_type in {
        "pair_attention",
        "pair_attention_biased_quadratic",
        "pair_attention_biased_lowrank_quadratic",
        "pair_attention_biased_lowrank_quadratic_strict",
    }:
        if cfg.model.attention_matrix_path is None:
            raise ValueError(
                f"{cfg.model.model_type} requires model.attention_matrix_path in config."
            )
        fixed_matrix = load_attention_matrix(
            cfg.model.attention_matrix_path,
            key=cfg.model.attention_matrix_key,
        )
    if cfg.model.model_type in {
        "pair_value",
        "pair_value_biased_quadratic",
        "pair_value_biased_lowrank_quadratic",
        "pair_value_biased_lowrank_quadratic_strict",
    }:
        if cfg.model.value_matrix_path is None:
            raise ValueError(
                f"{cfg.model.model_type} requires model.value_matrix_path in config."
            )
        fixed_matrix = load_attention_matrix(
            cfg.model.value_matrix_path,
            key=cfg.model.value_matrix_key,
        )

    model = build_probe(
        model_type=cfg.model.model_type,
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        rank=cfg.model.rank,
        attention_matrix=fixed_matrix,
    )
    report = train_probe(
        model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        named_test_data=named_test_data or None,
        cfg=cfg,
        eval_only=args.eval_only,
        checkpoint_path=args.checkpoint,
    )

    print(json.dumps(report["best"], indent=2))
    print(f"Saved checkpoint to: {report['checkpoint']}")
    print(f"Saved metrics to: {Path(cfg.output.out_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()
