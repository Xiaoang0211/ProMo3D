import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from isdynamic.core.label_controls import apply_train_label_control
from isdynamic.core.output_paths import resolve_probe_out_dir
from isdynamic.core.position_residualization import apply_position_residualization_to_splits, save_position_residualizer
from isdynamic.core.stereo_config import ProbeConfig


class TokenProbeDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor):
        keep = valid_mask.bool()
        self.features = features[keep].float()
        self.labels = labels[keep].float()

        if self.features.ndim != 2:
            raise ValueError(f"Expected features [N,C], got {self.features.shape}")
        if self.labels.ndim != 1:
            raise ValueError(f"Expected labels [N], got {self.labels.shape}")
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError("Feature and label counts do not match after filtering.")

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class PairTokenProbeDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
        group_ids: torch.Tensor,
        num_pairs: int,
        same_group: bool = True,
        balance: bool = True,
        label_mode: str = "same_label",
        max_tokens_per_group: int = 64,
        seed: int = 42,
        all_pairs: bool = False,
        anchor_local_index: int = 0,
        force_anchor_static_label: bool = False,
        min_anchor_static_pairs_per_batch: int = 0,
        min_anchor_dynamic_pairs_per_batch: int = 0,
        batch_size_hint: int = 0,
    ):
        keep = valid_mask.bool()
        self.features = features[keep].float()
        self.labels = labels[keep].long()
        self.group_ids = group_ids[keep].long()
        self.num_pairs = num_pairs
        self.same_group = same_group
        self.balance = balance
        self.label_mode = label_mode
        self.max_tokens_per_group = max_tokens_per_group
        self.seed = seed
        self.all_pairs = all_pairs
        self.anchor_local_index = max(0, int(anchor_local_index))
        self.force_anchor_static_label = bool(force_anchor_static_label)
        self.min_anchor_static_pairs_per_batch = max(0, int(min_anchor_static_pairs_per_batch))
        self.min_anchor_dynamic_pairs_per_batch = max(0, int(min_anchor_dynamic_pairs_per_batch))
        self.batch_size_hint = max(0, int(batch_size_hint))
        self.enforce_anchor_batch_constraints = (
            self.batch_size_hint > 0
            and (self.min_anchor_static_pairs_per_batch > 0 or self.min_anchor_dynamic_pairs_per_batch > 0)
        )

        if self.features.ndim != 2:
            raise ValueError(f"Expected features [N,C], got {self.features.shape}")
        if self.labels.ndim != 1 or self.group_ids.ndim != 1:
            raise ValueError("Expected labels/group_ids to be 1D.")
        if not (self.features.shape[0] == self.labels.shape[0] == self.group_ids.shape[0]):
            raise ValueError("Mismatched token counts after filtering.")
        if self.enforce_anchor_batch_constraints and not self.same_group:
            raise ValueError("Anchor-pair batch constraints require pair_same_group=True.")
        if (
            self.enforce_anchor_batch_constraints or self.force_anchor_static_label
        ) and self.label_mode != "same_label":
            raise ValueError(
                "Anchor token constraints currently require pair_label_mode='same_label'."
            )
        if (
            self.enforce_anchor_batch_constraints
            and self.min_anchor_static_pairs_per_batch + self.min_anchor_dynamic_pairs_per_batch
            > self.batch_size_hint
        ):
            raise ValueError(
                "Anchor-pair quotas exceed batch size: "
                f"{self.min_anchor_static_pairs_per_batch}+{self.min_anchor_dynamic_pairs_per_batch}"
                f">{self.batch_size_hint}"
            )
        self._apply_anchor_label_override()
        if self.all_pairs:
            self.pair_i, self.pair_j, self.pair_labels = self._enumerate_all_pairs(seed=self.seed)
        else:
            self.pair_i, self.pair_j, self.pair_labels = self._sample_pairs(seed=self.seed)

    def _group_indices_by_order(self) -> Dict[int, np.ndarray]:
        groups_np = self.group_ids.cpu().numpy()
        unique_groups = np.unique(groups_np)
        return {int(g): np.flatnonzero(groups_np == g) for g in unique_groups}

    def _apply_anchor_label_override(self) -> None:
        if not self.force_anchor_static_label:
            return
        for idxs in self._group_indices_by_order().values():
            if idxs.size == 0 or self.anchor_local_index >= idxs.size:
                continue
            self.labels[idxs[self.anchor_local_index]] = 0

    def _group_indices_for_sampling(self, rng: np.random.Generator) -> Dict[int, np.ndarray]:
        group_to_indices = self._group_indices_by_order()
        if self.max_tokens_per_group > 0:
            for group_id, idxs in list(group_to_indices.items()):
                if idxs.size > self.max_tokens_per_group:
                    if self.force_anchor_static_label or self.enforce_anchor_batch_constraints:
                        if self.anchor_local_index >= idxs.size:
                            continue
                        anchor_idx = idxs[self.anchor_local_index]
                        remaining = np.delete(idxs, self.anchor_local_index)
                        keep_count = self.max_tokens_per_group - 1
                        if keep_count <= 0:
                            sampled = np.array([anchor_idx], dtype=np.int64)
                        else:
                            sampled_rest = rng.choice(remaining, size=keep_count, replace=False)
                            sampled = np.concatenate(
                                [np.array([anchor_idx], dtype=np.int64), sampled_rest.astype(np.int64)]
                            )
                        group_to_indices[group_id] = np.sort(sampled)
                    else:
                        group_to_indices[group_id] = rng.choice(
                            idxs, size=self.max_tokens_per_group, replace=False
                        )
        return group_to_indices

    def _anchor_partner_pools(
        self, group_to_indices: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, Tuple[int, np.ndarray]], Dict[int, Tuple[int, np.ndarray]]]:
        labels_np = self.labels.cpu().numpy()
        static_groups: Dict[int, Tuple[int, np.ndarray]] = {}
        dynamic_groups: Dict[int, Tuple[int, np.ndarray]] = {}
        for group_id, idxs in group_to_indices.items():
            if idxs.size <= self.anchor_local_index:
                continue
            anchor_idx = int(idxs[self.anchor_local_index])
            partners = np.delete(idxs, self.anchor_local_index)
            if partners.size == 0:
                continue
            static_partners = partners[labels_np[partners] == 0]
            dynamic_partners = partners[labels_np[partners] == 1]
            if static_partners.size > 0:
                static_groups[group_id] = (anchor_idx, static_partners.astype(np.int64, copy=False))
            if dynamic_partners.size > 0:
                dynamic_groups[group_id] = (anchor_idx, dynamic_partners.astype(np.int64, copy=False))
        return static_groups, dynamic_groups

    def _batch_anchor_requirements(self) -> List[Tuple[int, int, int]]:
        if not self.enforce_anchor_batch_constraints:
            return []
        schedule: List[Tuple[int, int, int]] = []
        remaining = int(self.num_pairs)
        while remaining > 0:
            batch_cap = min(self.batch_size_hint, remaining)
            need_static = min(self.min_anchor_static_pairs_per_batch, batch_cap)
            need_dynamic = min(self.min_anchor_dynamic_pairs_per_batch, max(0, batch_cap - need_static))
            schedule.append((batch_cap, need_static, need_dynamic))
            remaining -= batch_cap
        return schedule

    def _pair_label(self, y_i: int, y_j: int) -> int:
        if self.label_mode == "same_label":
            return int(y_i == y_j)
        if self.label_mode == "both_dynamic":
            return int((y_i == 1) and (y_j == 1))
        raise ValueError(f"Unsupported pair_label_mode: {self.label_mode}")

    def _sample_pairs(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(seed)
        num_tokens = self.features.shape[0]
        if num_tokens < 2:
            raise ValueError("Need at least two valid tokens for pairwise probing.")

        labels_np = self.labels.cpu().numpy()

        if self.same_group:
            group_to_indices = self._group_indices_for_sampling(rng)
            eligible_groups = [g for g, idxs in group_to_indices.items() if idxs.size >= 2]
            if not eligible_groups:
                raise ValueError("No group has at least 2 valid tokens for same-group pair sampling.")
            anchor_static_groups, anchor_dynamic_groups = self._anchor_partner_pools(group_to_indices)
        else:
            all_indices = np.arange(num_tokens, dtype=np.int64)

        pair_i = []
        pair_j = []
        pair_labels = []
        pos_target = self.num_pairs // 2 if self.balance else None
        neg_target = self.num_pairs - pos_target if self.balance else None
        pos_count = 0
        neg_count = 0
        max_attempts = max(self.num_pairs * 50, 1000)
        attempts = 0

        anchor_schedule = self._batch_anchor_requirements()
        if anchor_schedule:
            if self.min_anchor_static_pairs_per_batch > 0 and not anchor_static_groups:
                raise ValueError("No eligible groups provide anchor-to-static pairs for the requested constraint.")
            if self.min_anchor_dynamic_pairs_per_batch > 0 and not anchor_dynamic_groups:
                raise ValueError("No eligible groups provide anchor-to-dynamic pairs for the requested constraint.")
            total_required_pos = sum(static_need for _, static_need, _ in anchor_schedule)
            total_required_neg = sum(dynamic_need for _, _, dynamic_need in anchor_schedule)
            if self.balance:
                if total_required_pos > pos_target:
                    raise ValueError(
                        f"Anchor-to-static quota requires {total_required_pos} positive pairs, "
                        f"but pair_balance only allows {pos_target}."
                    )
                if total_required_neg > neg_target:
                    raise ValueError(
                        f"Anchor-to-dynamic quota requires {total_required_neg} negative pairs, "
                        f"but pair_balance only allows {neg_target}."
                    )

        def _record_pair(i: int, j: int, label: int) -> bool:
            nonlocal pos_count, neg_count
            if self.balance:
                if label == 1 and pos_count >= pos_target:
                    return False
                if label == 0 and neg_count >= neg_target:
                    return False
            pair_i.append(int(i))
            pair_j.append(int(j))
            pair_labels.append(float(label))
            if label == 1:
                pos_count += 1
            else:
                neg_count += 1
            return True

        def _sample_anchor_pairs(group_pool: Dict[int, Tuple[int, np.ndarray]], quota: int) -> None:
            if quota <= 0:
                return
            group_keys = list(group_pool.keys())
            if not group_keys:
                raise ValueError("Requested anchor quota cannot be satisfied: empty eligible group pool.")
            for _ in range(quota):
                group = int(group_keys[rng.integers(0, len(group_keys))])
                anchor_idx, partners = group_pool[group]
                partner_idx = int(partners[rng.integers(0, partners.size)])
                label = self._pair_label(int(labels_np[anchor_idx]), int(labels_np[partner_idx]))
                if not _record_pair(anchor_idx, partner_idx, label):
                    raise ValueError("Anchor quota conflicts with pair balance targets.")

        for batch_idx, (batch_cap, static_need, dynamic_need) in enumerate(anchor_schedule):
            _sample_anchor_pairs(anchor_static_groups, static_need)
            _sample_anchor_pairs(anchor_dynamic_groups, dynamic_need)
            filler_target = batch_cap - static_need - dynamic_need
            filler_count = 0
            filler_attempts = 0
            future_pos_reserved = sum(s for _, s, _ in anchor_schedule[batch_idx + 1 :])
            future_neg_reserved = sum(d for _, _, d in anchor_schedule[batch_idx + 1 :])
            while filler_count < filler_target and attempts < max_attempts:
                attempts += 1
                filler_attempts += 1
                if self.same_group:
                    group = eligible_groups[rng.integers(0, len(eligible_groups))]
                    idxs = group_to_indices[group]
                    idx_i, idx_j = rng.choice(idxs, size=2, replace=False)
                else:
                    idx_i, idx_j = rng.choice(all_indices, size=2, replace=False)
                label = self._pair_label(int(labels_np[idx_i]), int(labels_np[idx_j]))
                if self.balance:
                    if label == 1 and pos_count >= (pos_target - future_pos_reserved):
                        continue
                    if label == 0 and neg_count >= (neg_target - future_neg_reserved):
                        continue
                if _record_pair(int(idx_i), int(idx_j), label):
                    filler_count += 1
                if filler_attempts >= max_attempts:
                    break

        while len(pair_labels) < self.num_pairs and attempts < max_attempts:
            attempts += 1
            if self.same_group:
                group = eligible_groups[rng.integers(0, len(eligible_groups))]
                idxs = group_to_indices[group]
                idx_i, idx_j = rng.choice(idxs, size=2, replace=False)
            else:
                idx_i, idx_j = rng.choice(all_indices, size=2, replace=False)

            label = self._pair_label(int(labels_np[idx_i]), int(labels_np[idx_j]))
            _record_pair(int(idx_i), int(idx_j), label)

        if len(pair_labels) < self.num_pairs:
            raise RuntimeError(
                f"Could not sample enough pairs. Requested {self.num_pairs}, got {len(pair_labels)}."
            )

        return (
            torch.tensor(pair_i, dtype=torch.long),
            torch.tensor(pair_j, dtype=torch.long),
            torch.tensor(pair_labels, dtype=torch.float32),
        )

    def _enumerate_all_pairs(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(seed)
        labels_np = self.labels.cpu().numpy()
        num_tokens = self.features.shape[0]
        if num_tokens < 2:
            raise ValueError("Need at least two valid tokens for pairwise probing.")

        if self.same_group:
            group_to_indices = self._group_indices_for_sampling(rng)
            blocks = [idxs for idxs in group_to_indices.values() if idxs.size >= 2]
            if not blocks:
                raise ValueError("No group has at least 2 valid tokens for same-group pair enumeration.")
            part_i = []
            part_j = []
            for idxs in blocks:
                left, right = np.triu_indices(idxs.size, k=1)
                part_i.append(idxs[left])
                part_j.append(idxs[right])
            pair_i = np.concatenate(part_i, axis=0)
            pair_j = np.concatenate(part_j, axis=0)
        else:
            pair_i, pair_j = np.triu_indices(num_tokens, k=1)

        if self.label_mode == "same_label":
            pair_labels = (labels_np[pair_i] == labels_np[pair_j]).astype(np.float32)
        elif self.label_mode == "both_dynamic":
            pair_labels = ((labels_np[pair_i] == 1) & (labels_np[pair_j] == 1)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported pair_label_mode: {self.label_mode}")

        return (
            torch.from_numpy(pair_i.astype(np.int64)),
            torch.from_numpy(pair_j.astype(np.int64)),
            torch.from_numpy(pair_labels.astype(np.float32)),
        )

    def __len__(self) -> int:
        return self.pair_labels.shape[0]

    def __getitem__(self, idx: int):
        idx_i = self.pair_i[idx]
        idx_j = self.pair_j[idx]
        return self.features[idx_i], self.features[idx_j], self.pair_labels[idx]

    def resample(self, seed: int) -> None:
        if self.all_pairs:
            return
        self.pair_i, self.pair_j, self.pair_labels = self._sample_pairs(seed=seed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _finalize_binary_metrics(
    *,
    loss_sum: float,
    total: int,
    correct: int,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = correct / max(total, 1)
    loss = loss_sum / max(total, 1)
    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, pairwise: bool = False) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    tp = fp = fn = tn = 0
    loss_sum = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for batch in loader:
        if pairwise:
            features_i, features_j, labels = batch
            features_i = features_i.to(device)
            features_j = features_j.to(device)
            labels = labels.to(device)
            logits = model(features_i, features_j)
        else:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)

        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total += labels.numel()
        correct += (preds == labels).sum().item()
        loss_sum += loss.item() * labels.numel()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

    return _finalize_binary_metrics(
        loss_sum=loss_sum,
        total=total,
        correct=correct,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


@torch.no_grad()
def evaluate_pairwise_batchwise(
    model: nn.Module,
    dataset: PairTokenProbeDataset,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    features_cpu = dataset.features
    pair_i_cpu = dataset.pair_i
    pair_j_cpu = dataset.pair_j
    labels_cpu = dataset.pair_labels

    total_pairs = int(labels_cpu.shape[0])
    total = 0
    correct = 0
    tp = fp = fn = tn = 0
    loss_sum = 0.0

    for start in range(0, total_pairs, batch_size):
        end = min(start + batch_size, total_pairs)
        idx_i = pair_i_cpu[start:end]
        idx_j = pair_j_cpu[start:end]

        features_i = features_cpu[idx_i].to(device, non_blocking=True)
        features_j = features_cpu[idx_j].to(device, non_blocking=True)
        labels = labels_cpu[start:end].to(device, non_blocking=True)

        logits = model(features_i, features_j)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        batch_n = labels.numel()
        total += batch_n
        correct += (preds == labels).sum().item()
        loss_sum += loss.item() * batch_n
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

    return _finalize_binary_metrics(
        loss_sum=loss_sum,
        total=total,
        correct=correct,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def train_probe(
    model: nn.Module,
    train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: ProbeConfig,
    test_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    named_test_data: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    eval_only: bool = False,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    set_seed(cfg.train.seed)
    train_data = apply_train_label_control(
        train_data,
        mode=cfg.train.train_label_control,
        seed=cfg.train.seed,
    )

    pairwise = bool(getattr(model, "is_pairwise", False))
    eval_batch_size = (
        int(cfg.train.eval_batch_size)
        if cfg.train.eval_batch_size is not None and int(cfg.train.eval_batch_size) > 0
        else int(cfg.train.batch_size)
    )

    position_residualization_meta = None
    position_residualizer = None
    train_data, val_data, test_data, named_test_data, position_residualization_meta, position_residualizer = apply_position_residualization_to_splits(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        named_test_data=named_test_data,
        train_cfg=cfg.train,
    )

    if pairwise:
        eval_all_pairs = cfg.train.pair_eval_all_pairs
        train_dataset = PairTokenProbeDataset(
            *train_data,
            num_pairs=cfg.train.pair_train_samples,
            same_group=cfg.train.pair_same_group,
            balance=cfg.train.pair_balance,
            label_mode=cfg.train.pair_label_mode,
            max_tokens_per_group=cfg.train.pair_tokens_per_group,
            seed=cfg.train.seed,
            anchor_local_index=cfg.train.pair_anchor_local_index,
            force_anchor_static_label=cfg.train.pair_force_anchor_static_label,
            min_anchor_static_pairs_per_batch=cfg.train.pair_min_anchor_static_pairs_per_batch,
            min_anchor_dynamic_pairs_per_batch=cfg.train.pair_min_anchor_dynamic_pairs_per_batch,
            batch_size_hint=cfg.train.batch_size,
        )
        val_dataset = PairTokenProbeDataset(
            *val_data,
            num_pairs=cfg.train.pair_val_samples,
            same_group=cfg.train.pair_same_group,
            balance=False,
            label_mode=cfg.train.pair_label_mode,
            max_tokens_per_group=cfg.train.pair_tokens_per_group,
            seed=cfg.train.seed + 1,
            all_pairs=eval_all_pairs,
            anchor_local_index=cfg.train.pair_anchor_local_index,
            force_anchor_static_label=cfg.train.pair_force_anchor_static_label,
        )
        test_dataset = (
            PairTokenProbeDataset(
                *test_data,
                num_pairs=cfg.train.pair_test_samples,
                same_group=cfg.train.pair_same_group,
                balance=False,
                label_mode=cfg.train.pair_label_mode,
                max_tokens_per_group=cfg.train.pair_tokens_per_group,
                seed=cfg.train.seed + 2,
                all_pairs=eval_all_pairs,
                anchor_local_index=cfg.train.pair_anchor_local_index,
                force_anchor_static_label=cfg.train.pair_force_anchor_static_label,
            )
            if test_data is not None
            else None
        )
        named_test_datasets = {}
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
                    anchor_local_index=cfg.train.pair_anchor_local_index,
                    force_anchor_static_label=cfg.train.pair_force_anchor_static_label,
                )
    else:
        train_dataset = TokenProbeDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = TokenProbeDataset(val_data[0], val_data[1], val_data[2])
        test_dataset = TokenProbeDataset(test_data[0], test_data[1], test_data[2]) if test_data is not None else None
        named_test_datasets = {}
        if named_test_data is not None:
            for test_name, test_tuple in named_test_data.items():
                named_test_datasets[test_name] = TokenProbeDataset(
                    test_tuple[0], test_tuple[1], test_tuple[2]
                )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=not getattr(train_dataset, "enforce_anchor_batch_constraints", False),
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if test_dataset is not None
        else None
    )
    named_test_loaders = {
        test_name: DataLoader(
            named_test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for test_name, named_test_dataset in named_test_datasets.items()
    }

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = None
    if cfg.train.use_step_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.lr_step_size,
            gamma=cfg.train.lr_gamma,
        )
    pos_weight = torch.tensor([cfg.train.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    out_dir = resolve_probe_out_dir(cfg.output.out_dir, cfg.train)
    out_dir.mkdir(parents=True, exist_ok=True)
    if position_residualizer is not None:
        save_position_residualizer(out_dir / "position_residualizer.pt", position_residualizer)
    best_path = out_dir / cfg.output.save_name
    if eval_only:
        if checkpoint_path is not None:
            best_path = Path(checkpoint_path)
        if not best_path.exists():
            raise FileNotFoundError(f"Eval-only checkpoint not found: {best_path}")

    selection_metric = cfg.train.selection_metric
    valid_selection_metrics = {"loss", "accuracy", "precision", "recall", "f1"}
    if selection_metric not in valid_selection_metrics:
        raise ValueError(
            f"Unsupported selection_metric '{selection_metric}'. "
            f"Expected one of {sorted(valid_selection_metrics)}."
        )
    minimize_selection = selection_metric == "loss"
    best_selection_value = float("inf") if minimize_selection else -float("inf")

    best_train_metrics: Dict[str, float] = {}
    best_val_metrics: Dict[str, float] = {}
    best_test_metrics: Optional[Dict[str, float]] = None
    best_named_test_metrics: Dict[str, Dict[str, float]] = {}
    last_metrics: Dict[str, object] = {}

    for epoch in range(cfg.train.epochs if not eval_only else 0):
        model.train()
        running_loss = 0.0
        running_n = 0
        if pairwise and cfg.train.pair_resample_each_epoch:
            train_dataset.resample(seed=cfg.train.seed + epoch)

        for batch in train_loader:
            if pairwise:
                features_i, features_j, labels = batch
                features_i = features_i.to(device)
                features_j = features_j.to(device)
                labels = labels.to(device)
                logits = model(features_i, features_j)
            else:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_n = labels.numel()
            running_loss += loss.item() * batch_n
            running_n += batch_n

        train_loss = running_loss / max(running_n, 1)
        if pairwise and cfg.train.pair_eval_batchwise:
            val_metrics = evaluate_pairwise_batchwise(
                model,
                val_dataset,
                device=device,
                batch_size=eval_batch_size,
            )
        else:
            val_metrics = evaluate(model, val_loader, device=device, pairwise=pairwise)
        val_metrics["train_loss"] = float(train_loss)
        val_metrics["epoch"] = float(epoch + 1)
        val_metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])

        last_metrics = {
            "train": None,
            "val": val_metrics,
            "test": None,
            "tests": None,
        }

        current_selection_value = float(val_metrics[selection_metric])
        is_better = (
            current_selection_value < best_selection_value
            if minimize_selection
            else current_selection_value > best_selection_value
        )
        if is_better:
            best_selection_value = current_selection_value
            best_val_metrics = dict(val_metrics)
            torch.save(model.state_dict(), best_path)

        if scheduler is not None:
            scheduler.step()

    if not best_path.exists():
        if eval_only:
            raise FileNotFoundError(f"Eval-only checkpoint not found: {best_path}")
        torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=device))

    if pairwise and cfg.train.pair_eval_batchwise:
        best_train_metrics = evaluate_pairwise_batchwise(
            model, train_dataset, device=device, batch_size=eval_batch_size
        )
        best_val_eval_metrics = evaluate_pairwise_batchwise(
            model, val_dataset, device=device, batch_size=eval_batch_size
        )
        best_test_metrics = (
            evaluate_pairwise_batchwise(
                model, test_dataset, device=device, batch_size=eval_batch_size
            )
            if test_dataset is not None
            else None
        )
        best_named_test_metrics = {
            test_name: evaluate_pairwise_batchwise(
                model,
                named_test_dataset,
                device=device,
                batch_size=eval_batch_size,
            )
            for test_name, named_test_dataset in named_test_datasets.items()
        }
    else:
        best_train_metrics = evaluate(model, train_eval_loader, device=device, pairwise=pairwise)
        best_val_eval_metrics = evaluate(model, val_loader, device=device, pairwise=pairwise)
        best_test_metrics = (
            evaluate(model, test_loader, device=device, pairwise=pairwise)
            if test_loader is not None
            else None
        )
        best_named_test_metrics = {
            test_name: evaluate(model, named_test_loader, device=device, pairwise=pairwise)
            for test_name, named_test_loader in named_test_loaders.items()
        }

    best_epoch = float(best_val_metrics.get("epoch", cfg.train.epochs))
    best_lr = float(best_val_metrics.get("learning_rate", optimizer.param_groups[0]["lr"]))
    best_train_loss = float(best_val_metrics.get("train_loss", 0.0))
    for metrics in (
        [best_train_metrics, best_val_eval_metrics]
        + ([best_test_metrics] if best_test_metrics is not None else [])
        + list(best_named_test_metrics.values())
    ):
        metrics["epoch"] = best_epoch
        metrics["learning_rate"] = best_lr
        metrics["train_loss"] = best_train_loss

    if (not minimize_selection and best_selection_value == -float("inf")) or (
        minimize_selection and best_selection_value == float("inf")
    ):
        best_selection_value = float(best_val_eval_metrics[selection_metric])
    best_val_metrics = best_val_eval_metrics

    merged_best_tests = dict(best_named_test_metrics)
    if best_test_metrics is not None:
        merged_best_tests.setdefault("test", best_test_metrics)

    report = {
        "best": best_val_metrics,
        "best_train": best_train_metrics,
        "best_val": best_val_metrics,
        "best_tests": merged_best_tests,
        "last": last_metrics,
        "selection": {
            "split": "val",
            "metric": selection_metric,
            "value": best_selection_value,
        },
        "config": asdict(cfg),
        "checkpoint": str(best_path),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset) if test_dataset is not None else 0,
        "test_samples_by_name": {
            test_name: len(named_test_dataset)
            for test_name, named_test_dataset in named_test_datasets.items()
        },
        "pairwise": pairwise,
        "eval_only": bool(eval_only),
    }
    if position_residualization_meta is not None:
        report["position_residualization"] = position_residualization_meta

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report
