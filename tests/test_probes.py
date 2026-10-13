import numpy as np
import pytest
import torch

from isdynamic.core.common_models import PairwiseLowRankQuadraticProbe
from isdynamic.core.common_trainer import PairTokenProbeDataset
from isdynamic.core.label_controls import apply_train_label_control


def test_pairs_stay_in_frame_and_use_same_motion_state():
    features = torch.arange(6, dtype=torch.float32).reshape(-1, 1)
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    groups = torch.tensor([10, 10, 10, 20, 20, 99])
    valid = torch.tensor([True, True, True, True, True, False])
    pairs = PairTokenProbeDataset(features, labels, valid, groups, num_pairs=4,
                                 same_group=True, balance=False, all_pairs=True)
    assert len(pairs) == 4
    observed = set()
    for x, y, z in pairs:
        i, j = int(x.item()), int(y.item())
        assert groups[i] == groups[j]
        assert valid[i] and valid[j]
        assert i != j
        assert int(z) == int(labels[i] == labels[j])
        observed.add((i, j, int(z)))
    assert observed == {(0, 1, 1), (0, 2, 0), (1, 2, 0), (3, 4, 0)}


def test_native_sampler_preserves_dynamic_token_zero():
    labels = torch.tensor([1, 0, 0, 1])
    pairs = PairTokenProbeDataset(torch.eye(4), labels, torch.ones(4, dtype=torch.bool),
                                 torch.zeros(4, dtype=torch.long), num_pairs=8, balance=False)
    assert pairs.labels[0] == 1
    assert torch.equal(labels, torch.tensor([1, 0, 0, 1]))


def test_shuffle_preserves_frame_counts_invalid_tokens_and_features():
    labels = torch.tensor([0, 1] * 20)
    groups = torch.repeat_interleave(torch.tensor([10, 20]), 20)
    valid = torch.ones(40, dtype=torch.bool)
    valid[0] = False
    features = torch.arange(80).reshape(40, 2).float()
    data = features, labels, valid, groups
    result = apply_train_label_control(data, "within_group_shuffle", 42)
    repeated = apply_train_label_control(data, "within_group_shuffle", 42)
    assert torch.equal(result[1], repeated[1])
    assert not torch.equal(result[1], labels)
    assert result[1][0] == labels[0]
    assert result[0] is features and result[2] is valid and result[3] is groups
    assert torch.equal(labels, torch.tensor([0, 1] * 20))
    for group in (10, 20):
        keep = (groups == group) & valid
        assert torch.equal(torch.bincount(result[1][keep]), torch.bincount(labels[keep]))
    assert apply_train_label_control(data, "none", 42) is data


def test_lowrank_score_matches_paper_equation_and_is_symmetric():
    torch.manual_seed(42)
    probe = PairwiseLowRankQuadraticProbe(input_dim=6, rank=2)
    x, y = torch.randn(5, 6), torch.randn(5, 6)
    matrix = (probe.u.T @ probe.v + probe.v.T @ probe.u) / (2 * np.sqrt(6))
    expected = torch.einsum("bi,ij,bj->b", x, matrix, y) + probe.b
    torch.testing.assert_close(probe(x, y), expected)
    torch.testing.assert_close(probe(x, y), probe(y, x))
    assert sum(parameter.numel() for parameter in probe.parameters()) == 2 * 2 * 6 + 1


@pytest.mark.parametrize("seed", [42, 13, 7])
def test_pair_sampling_reproducible(seed):
    args = (torch.eye(8), torch.tensor([0, 1] * 4), torch.ones(8, dtype=torch.bool),
            torch.tensor([0] * 4 + [1] * 4))
    a = PairTokenProbeDataset(*args, num_pairs=32, seed=seed, balance=False)
    b = PairTokenProbeDataset(*args, num_pairs=32, seed=seed, balance=False)
    torch.testing.assert_close(a.pair_i, b.pair_i)
    torch.testing.assert_close(a.pair_j, b.pair_j)
    torch.testing.assert_close(a.pair_labels, b.pair_labels)
