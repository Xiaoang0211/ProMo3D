import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class DatasetConfig:
    train_npz: str
    val_npz: str
    test_npz: Optional[str] = None
    test_npzs: Optional[Dict[str, str]] = None
    feature_key: str = "tokens"
    label_key: str = "is_dynamic"
    valid_key: Optional[str] = "valid_mask"


@dataclass
class ProbeModelConfig:
    input_dim: int
    hidden_dim: int = 256
    model_type: str = "linear"
    dropout: float = 0.0
    rank: int = 256
    use_value_features: bool = False
    value_feature_matrix_path: Optional[str] = None
    value_feature_matrix_key: Optional[str] = None
    attention_matrix_path: Optional[str] = None
    attention_matrix_key: Optional[str] = None
    value_matrix_path: Optional[str] = None
    value_matrix_key: Optional[str] = None


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 256
    eval_batch_size: Optional[int] = None
    epochs: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    pos_weight: float = 1.0
    num_workers: int = 0
    use_step_lr: bool = True
    lr_step_size: int = 8
    lr_gamma: float = 0.2
    pair_train_samples: int = 200000
    pair_val_samples: int = 50000
    pair_test_samples: int = 50000
    pair_same_group: bool = True
    pair_balance: bool = True
    pair_label_mode: str = "same_label"
    pair_tokens_per_group: int = 64
    pair_anchor_local_index: int = 0
    pair_force_anchor_static_label: bool = True
    pair_min_anchor_static_pairs_per_batch: int = 4
    pair_min_anchor_dynamic_pairs_per_batch: int = 4
    pair_resample_each_epoch: bool = True
    pair_eval_all_pairs: bool = True
    pair_eval_batchwise: bool = True
    train_label_control: str = "none"
    selection_metric: str = "f1"
    position_residualization: str = "none"
    position_residualization_fourier_freqs: int = 4
    position_residualization_ridge: float = 1e-3


@dataclass
class OutputConfig:
    out_dir: str = "outputs/isdynamic_probe"
    save_name: str = "best_probe.pt"


@dataclass
class ProbeConfig:
    dataset: DatasetConfig
    model: ProbeModelConfig
    train: TrainConfig
    output: OutputConfig


def _load_yaml_if_available(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "YAML config requested but PyYAML is not installed. "
            "Install pyyaml or use JSON configs."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_probe_config(path: str) -> ProbeConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    if suffix == ".json":
        with cfg_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    elif suffix in {".yml", ".yaml"}:
        raw = _load_yaml_if_available(cfg_path)
    else:
        raise ValueError(f"Unsupported config extension: {suffix}")

    return ProbeConfig(
        dataset=DatasetConfig(**raw["dataset"]),
        model=ProbeModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        output=OutputConfig(**raw["output"]),
    )
