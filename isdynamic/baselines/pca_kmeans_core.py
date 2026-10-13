from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
except ImportError as exc:  # pragma: no cover - import error handled at runtime
    PCA = None
    KMeans = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


NORM_CHOICES = ("l2", "zscore_l2")
PREPROCESS_SCOPE_CHOICES = ("target_frame_only", "frame_group")
BLOCK_MODE_CHOICES = ("single_block", "multi_block_aggregate")
CLUSTER_RULE_CHOICES = (
    "anchor",
    "minority",
    "non_anchor_minority",
    "farthest_from_anchor",
    "oracle",
)


@dataclass(frozen=True)
class PCAKMeansConfig:
    norm: str = "zscore_l2"
    preprocess_scope: str = "frame_group"
    pca_dim: int = 8
    num_clusters: int = 2
    cluster_rule: str = "anchor"
    anchor_idx: int = 0
    random_state: int = 0
    n_init: int = 10

    def validate(self) -> None:
        if self.norm not in NORM_CHOICES:
            raise ValueError(f"Unsupported norm: {self.norm}")
        if self.preprocess_scope not in PREPROCESS_SCOPE_CHOICES:
            raise ValueError(f"Unsupported preprocess_scope: {self.preprocess_scope}")
        if self.cluster_rule not in CLUSTER_RULE_CHOICES:
            raise ValueError(f"Unsupported cluster_rule: {self.cluster_rule}")
        if self.pca_dim <= 0:
            raise ValueError("pca_dim must be positive.")
        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be positive.")
        if self.n_init <= 0:
            raise ValueError("n_init must be positive.")


@dataclass(frozen=True)
class PCAKMeansFitResult:
    projected_target: np.ndarray
    cluster_labels: np.ndarray
    centroids: np.ndarray
    effective_num_clusters: int
    preprocess_mean: np.ndarray | None
    preprocess_std: np.ndarray | None
    pca_components: np.ndarray
    pca_mean: np.ndarray
    pca_dim_used: int


def _require_sklearn() -> None:
    if PCA is None or KMeans is None:
        raise ImportError(
            "scikit-learn is required for the pca_kmeans baseline."
        ) from _SKLEARN_IMPORT_ERROR


def _safe_l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


def _fit_pca_numpy(features: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(features, dtype=np.float32)
    mean = centered.mean(axis=0, keepdims=True).astype(np.float32)
    centered = centered - mean
    if centered.shape[0] == 0:
        raise ValueError("Cannot fit PCA on empty features.")
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = np.asarray(vt[:n_components], dtype=np.float32)
    return components, mean.reshape(-1)


def _transform_pca_numpy(features: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    centered = np.asarray(features, dtype=np.float32) - mean.reshape(1, -1).astype(np.float32)
    return (centered @ components.T).astype(np.float32)


def _run_kmeans_numpy(data: np.ndarray, n_clusters: int, random_state: int, n_init: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=np.float32)
    n, d = x.shape
    if n == 0:
        raise ValueError("Cannot run k-means on empty data.")
    if n_clusters <= 1 or n == 1:
        return np.zeros(n, dtype=np.int64), x[:1].copy()

    rng = np.random.default_rng(random_state)
    best_inertia = None
    best_labels = None
    best_centroids = None
    max_iter = 100
    n_init_eff = max(1, int(n_init))

    for init_idx in range(n_init_eff):
        if n_clusters >= n:
            centroid_indices = np.arange(n, dtype=np.int64)
        else:
            centroid_indices = rng.choice(n, size=n_clusters, replace=False)
        centroids = x[centroid_indices].copy()

        for _ in range(max_iter):
            distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1).astype(np.int64)
            new_centroids = centroids.copy()
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                if np.any(mask):
                    new_centroids[cluster_id] = x[mask].mean(axis=0)
                else:
                    farthest_idx = int(np.argmax(distances.min(axis=1)))
                    new_centroids[cluster_id] = x[farthest_idx]
            if np.allclose(new_centroids, centroids, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids

        distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1).astype(np.int64)
        inertia = float(np.take_along_axis(distances, labels[:, None], axis=1).sum())
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    assert best_labels is not None and best_centroids is not None
    unique_ids = np.unique(best_labels)
    remap = {int(cluster_id): idx for idx, cluster_id in enumerate(unique_ids.tolist())}
    remapped_labels = np.array([remap[int(x)] for x in best_labels], dtype=np.int64)
    remapped_centroids = np.stack([x[remapped_labels == idx].mean(axis=0) for idx in range(len(unique_ids))], axis=0)
    return remapped_labels, np.asarray(remapped_centroids, dtype=np.float32)


def _fit_pca_kmeans_backend(fit_norm: np.ndarray, target_norm: np.ndarray, cfg: PCAKMeansConfig, pca_dim_used: int):
    if PCA is not None and KMeans is not None:
        pca = PCA(n_components=pca_dim_used, svd_solver="full", random_state=cfg.random_state)
        pca.fit(fit_norm)
        projected_target = pca.transform(target_norm).astype(np.float32)
        pca_components = np.asarray(pca.components_, dtype=np.float32)
        pca_mean = np.asarray(pca.mean_, dtype=np.float32)
    else:
        pca_components, pca_mean = _fit_pca_numpy(fit_norm, pca_dim_used)
        projected_target = _transform_pca_numpy(target_norm, pca_components, pca_mean)

    k_used = int(min(cfg.num_clusters, max(1, projected_target.shape[0])))
    if k_used <= 1 or np.allclose(projected_target, projected_target[:1], atol=1e-8):
        labels = np.zeros(projected_target.shape[0], dtype=np.int64)
        centroids = projected_target[:1].copy()
        effective_num_clusters = 1
    else:
        if KMeans is not None:
            kmeans = KMeans(n_clusters=k_used, random_state=cfg.random_state, n_init=cfg.n_init)
            labels = kmeans.fit_predict(projected_target).astype(np.int64)
            centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
            effective_num_clusters = int(len(np.unique(labels)))
            if effective_num_clusters < centroids.shape[0]:
                unique_ids = np.unique(labels)
                centroids = np.stack([projected_target[labels == cluster_id].mean(axis=0) for cluster_id in unique_ids], axis=0)
                remapped = {int(cluster_id): idx for idx, cluster_id in enumerate(unique_ids.tolist())}
                labels = np.array([remapped[int(x)] for x in labels], dtype=np.int64)
        else:
            labels, centroids = _run_kmeans_numpy(projected_target, k_used, cfg.random_state, cfg.n_init)
            effective_num_clusters = int(len(np.unique(labels)))

    return projected_target, labels, centroids.astype(np.float32), effective_num_clusters, pca_components, pca_mean


def preprocess_features(
    *,
    target_features: np.ndarray,
    fit_features: np.ndarray,
    norm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    target = np.asarray(target_features, dtype=np.float32)
    fit = np.asarray(fit_features, dtype=np.float32)
    if target.ndim != 2 or fit.ndim != 2:
        raise ValueError(f"Expected 2D features, got target={target.shape}, fit={fit.shape}")
    if target.shape[1] != fit.shape[1]:
        raise ValueError(f"Feature dim mismatch: target={target.shape}, fit={fit.shape}")

    if norm == "l2":
        return _safe_l2_normalize(target), _safe_l2_normalize(fit), None, None

    if norm == "zscore_l2":
        mean = fit.mean(axis=0, keepdims=True)
        std = fit.std(axis=0, keepdims=True)
        std = np.where(std > 1e-6, std, 1.0)
        target_norm = _safe_l2_normalize((target - mean) / std)
        fit_norm = _safe_l2_normalize((fit - mean) / std)
        return target_norm, fit_norm, mean.reshape(-1).astype(np.float32), std.reshape(-1).astype(np.float32)

    raise ValueError(f"Unsupported norm: {norm}")


def fit_pca_kmeans(
    *,
    target_features: np.ndarray,
    fit_features: np.ndarray,
    cfg: PCAKMeansConfig,
) -> PCAKMeansFitResult:
    cfg.validate()

    target_norm, fit_norm, mean, std = preprocess_features(
        target_features=target_features,
        fit_features=fit_features,
        norm=cfg.norm,
    )
    if target_norm.shape[0] == 0:
        raise ValueError("target_features is empty.")
    if fit_norm.shape[0] == 0:
        raise ValueError("fit_features is empty.")

    pca_dim_used = int(min(cfg.pca_dim, fit_norm.shape[0], fit_norm.shape[1]))
    if pca_dim_used <= 0:
        raise ValueError(
            f"Invalid PCA dimension after clipping: pca_dim={cfg.pca_dim}, fit={fit_norm.shape}"
        )

    projected_target, labels, centroids, effective_num_clusters, pca_components, pca_mean = _fit_pca_kmeans_backend(
        fit_norm,
        target_norm,
        cfg,
        pca_dim_used,
    )

    return PCAKMeansFitResult(
        projected_target=projected_target,
        cluster_labels=labels,
        centroids=centroids.astype(np.float32),
        effective_num_clusters=effective_num_clusters,
        preprocess_mean=mean,
        preprocess_std=std,
        pca_components=np.asarray(pca_components, dtype=np.float32),
        pca_mean=np.asarray(pca_mean, dtype=np.float32),
        pca_dim_used=pca_dim_used,
    )


def _cluster_ids(cluster_labels: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(cluster_labels, dtype=np.int64))


def _labels_from_selected_clusters(cluster_labels: np.ndarray, dynamic_clusters: Iterable[int]) -> np.ndarray:
    dynamic_set = {int(cluster_id) for cluster_id in dynamic_clusters}
    return np.array([1 if int(cluster_id) in dynamic_set else 0 for cluster_id in cluster_labels], dtype=np.uint8)


def labels_from_rule(
    *,
    fit_result: PCAKMeansFitResult,
    rule: str,
    anchor_idx: int = 0,
    oracle_selector=None,
) -> np.ndarray:
    labels = fit_result.cluster_labels
    cluster_ids = _cluster_ids(labels)
    if labels.size == 0:
        return np.zeros((0,), dtype=np.uint8)
    if not (0 <= int(anchor_idx) < labels.shape[0]):
        raise IndexError(f"anchor_idx out of range: {anchor_idx} for {labels.shape[0]} tokens")

    anchor_cluster = int(labels[int(anchor_idx)])

    if rule == "anchor":
        return _labels_from_selected_clusters(labels, [cluster_id for cluster_id in cluster_ids if int(cluster_id) != anchor_cluster])

    counts = {int(cluster_id): int(np.sum(labels == cluster_id)) for cluster_id in cluster_ids.tolist()}

    if rule == "minority":
        dynamic_cluster = min(counts, key=lambda cluster_id: (counts[cluster_id], cluster_id))
        return _labels_from_selected_clusters(labels, [dynamic_cluster])

    if rule == "non_anchor_minority":
        candidate_clusters = [cluster_id for cluster_id in counts if cluster_id != anchor_cluster]
        if not candidate_clusters:
            return np.zeros_like(labels, dtype=np.uint8)
        dynamic_cluster = min(candidate_clusters, key=lambda cluster_id: (counts[cluster_id], cluster_id))
        return _labels_from_selected_clusters(labels, [dynamic_cluster])

    if rule == "farthest_from_anchor":
        candidate_clusters = [cluster_id for cluster_id in counts if cluster_id != anchor_cluster]
        if not candidate_clusters:
            return np.zeros_like(labels, dtype=np.uint8)
        anchor_centroid = fit_result.centroids[anchor_cluster]
        dynamic_cluster = max(
            candidate_clusters,
            key=lambda cluster_id: (
                float(np.linalg.norm(fit_result.centroids[int(cluster_id)] - anchor_centroid)),
                int(cluster_id),
            ),
        )
        return _labels_from_selected_clusters(labels, [dynamic_cluster])

    if rule == "oracle":
        if oracle_selector is None:
            raise ValueError("oracle_selector is required for rule='oracle'")
        dynamic_clusters = oracle_selector(cluster_ids.tolist(), labels, fit_result)
        return _labels_from_selected_clusters(labels, dynamic_clusters)

    raise ValueError(f"Unsupported rule: {rule}")


def fit_pca_kmeans_assignments(
    *,
    features: np.ndarray,
    cfg: PCAKMeansConfig,
) -> PCAKMeansFitResult:
    cfg.validate()

    normalized_target, normalized_fit, mean, std = preprocess_features(
        target_features=features,
        fit_features=features,
        norm=cfg.norm,
    )
    if normalized_target.shape[0] == 0:
        raise ValueError("features is empty.")

    pca_dim_used = int(min(cfg.pca_dim, normalized_fit.shape[0], normalized_fit.shape[1]))
    if pca_dim_used <= 0:
        raise ValueError(
            f"Invalid PCA dimension after clipping: pca_dim={cfg.pca_dim}, fit={normalized_fit.shape}"
        )

    projected_target, labels, centroids, effective_num_clusters, pca_components, pca_mean = _fit_pca_kmeans_backend(
        normalized_fit,
        normalized_target,
        cfg,
        pca_dim_used,
    )

    return PCAKMeansFitResult(
        projected_target=projected_target,
        cluster_labels=labels,
        centroids=centroids.astype(np.float32),
        effective_num_clusters=effective_num_clusters,
        preprocess_mean=mean,
        preprocess_std=std,
        pca_components=np.asarray(pca_components, dtype=np.float32),
        pca_mean=np.asarray(pca_mean, dtype=np.float32),
        pca_dim_used=pca_dim_used,
    )


def all_cluster_subsets(cluster_ids: Sequence[int]) -> list[tuple[int, ...]]:
    ordered = [int(cluster_id) for cluster_id in cluster_ids]
    subsets: list[tuple[int, ...]] = [tuple()]
    for r in range(1, len(ordered) + 1):
        subsets.extend(tuple(int(x) for x in subset) for subset in combinations(ordered, r))
    return subsets
