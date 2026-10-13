# Reproduction checks with pytest

Use an environment with PyTorch and this workspace's dependencies installed:

```bash
python -m pip install -r requirements-dev.txt
bash scripts/test.sh
```

`scripts/test.sh` uses `TEST_PYTHON_BIN`, or the configured analysis interpreter. It hides CUDA devices for the test process. No backbone repository, checkpoint, downloaded dataset, or GPU is required; fixtures are small tensors, PNG masks, and JSON logs under pytest's temporary directory.

The tests verify within-frame pair sampling and labels, deterministic shuffled-label controls, the paper's low-rank score equation, bilinear interpolation and threshold conventions, IoU/J-Recall, PNG-mask evaluation, NG/association statistics, averaging across all selected blocks, source provenance, native output isolation, model CLI forwarding, and seeded subset manifests. They also check that dry runs create no artifacts and that non-safeguard runs cannot silently reuse enforced-anchor metrics.

These checks validate the implementation's contracts. Full paper performance still requires the datasets, frozen checkpoints, exports, training, and benchmark inference described in `docs/main.md` and `docs/experiments.md`.
