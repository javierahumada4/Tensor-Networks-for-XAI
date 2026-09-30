# Born-MPS core for anomaly explanation experiments

This branch contains the dataset-agnostic model core for the new paper project.
It is intentionally separated from the original TFG pipeline: there is no
NSL-KDD encoder, dataset-specific training/evaluation script, plotting pipeline,
or legacy project artifact.

The current scope is deliberately small:

- `mps.py`: discrete Born-MPS model, stable likelihood/NLL evaluation,
  canonicalization, and two-site merge/SVD-split operations.
- `dmrg_trainer.py`: two-site maximum-likelihood training.
- No encoder is included. The paper encoder will be designed separately.
- The interaction-explanation implementation will be added separately rather
  than carrying over the TFG-specific explainability pipeline.

## Training design

### Empirical-marginal initialization

Training starts from the independent empirical model. For every site `i` and
state `a`,

```
p_i(a) = (n_i(a) + alpha) / (N + alpha * d_i)
A_i(a) = sqrt(p_i(a))
```

so every bond starts at dimension 1 and the initial Born distribution is

```
P_0(x) = product_i p_i(x_i).
```

`alpha` is the optional `pseudocount` (default `1e-6`) used only to give
unobserved states non-zero support.

### Fixed maximum bond dimension

There are no bond-dimension growth phases. `max_bond_dim` is a fixed
computational cap for the entire run. Local ranks can grow or shrink after each
two-site update as determined by the SVD.

### Discarded-weight SVD truncation

The SVD keeps the smallest rank `r` satisfying

```
sum_{j > r} sigma_j^2 / sum_j sigma_j^2 <= epsilon_trunc
```

unless `max_bond_dim` is reached first. If the hard cap is binding, the actual
discarded weight can exceed `epsilon_trunc`; the trainer records the observed
maximum discarded weight in the training history.

## Minimal usage

```python
import torch

from mps import MPS
from dmrg_trainer import DMRGConfig, dmrg_train

# train_data: LongTensor [n_samples, n_sites]
# physical_dims: number of discrete states at each site
model = MPS.from_empirical_frequencies(
    train_data,
    physical_dims=physical_dims,
    dtype=torch.float64,
    pseudocount=1e-6,
)

config = DMRGConfig(
    max_bond_dim=64,
    epsilon_trunc=1e-6,
    lr=8e-4,
    num_loops=100,
    batch_size=1024,
    metric_for_stopping="val_nll",
    early_stopping_patience=15,
    seed=123,
)

history = dmrg_train(model, train_data, val_data, config=config)
scores = model.anomaly_score(test_data)
```

The learning-rate plateau schedule from the previous implementation is still
present for now; it is independent of model capacity and can be reconsidered in
a later training ablation.
