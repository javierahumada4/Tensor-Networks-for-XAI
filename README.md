# Born-MPS core for anomaly explanation experiments

This branch contains the dataset-agnostic model core for the new paper project.
It is intentionally separated from the original TFG pipeline: there is no
NSL-KDD encoder, dataset-specific training/evaluation script, plotting pipeline,
or legacy project artifact.

The current scope is deliberately small:

- `mps.py`: discrete Born-MPS model, stable likelihood/NLL evaluation,
  canonicalization, and two-site merge/SVD-split operations.
- `dmrg_trainer.py`: two-site maximum-likelihood training.
- `encoder.py`: dataset-agnostic tabular encoder; one original feature maps to
  one MPS site.
- `data_artifacts.py`: reproducible one-class train/validation/test partition
  and persistent encoded artifacts.
- `prepare_dataset.py`: CLI for CSV or ADBench-style NPZ inputs.
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


## Data encoding and persistent splits

The encoder is fitted **only on normal training rows**. The one-class split is:

- normal rows -> train / validation / held-out normal test;
- anomaly rows -> test only;
- binary artifact labels -> `0 = normal`, `1 = anomaly`.

With the default split, 70% of normal rows are used for training, 15% for
validation, and the remaining 15% are combined with every anomaly in the test
set. A fixed seed makes the row partition reproducible.

### Feature encoding

Each original variable remains one MPS site.

Continuous features use quantile discretization fitted on normal training data.
Repeated quantiles are collapsed. Three explicit states are appended:

- `LOW`: below the minimum observed in normal training;
- `HIGH`: above the maximum observed in normal training;
- `MISSING`: missing value.

Categorical features keep one state per category observed in normal training,
plus:

- `UNKNOWN`: unseen non-missing category;
- `MISSING`: missing value.

High-cardinality categorical variables are rejected by default instead of being
silently hashed. They must be explicitly transformed or dropped.

### Artifact layout

Running the data preparation step creates:

```
artifacts/<dataset>/
├── manifest.json
├── encoder.json
├── split_indices.pt
├── train.pt
├── val.pt
└── test.pt
```

Each split file contains:

```python
{
    "x": LongTensor[num_rows, num_features],
    "y": LongTensor[num_rows],
    "row_indices": LongTensor[num_rows],
}
```

`split_indices.pt` stores the exact original row positions used for every
split. Downstream scripts should load these artifacts rather than recreate the
split.

Example:

```bash
python prepare_dataset.py data.csv artifacts/my_dataset \
  --label-column label \
  --normal-label 0 \
  --categorical protocol,service \
  --drop id \
  --bins 8 \
  --seed 123
```

ADBench-style `.npz` files containing arrays named `X` and `y` are also
accepted:

```bash
python prepare_dataset.py annthyroid.npz artifacts/annthyroid \
  --normal-label 0 \
  --bins 8 \
  --seed 123
```

Consumers use a single loader:

```python
from data_artifacts import load_encoded_bundle

data = load_encoded_bundle("artifacts/annthyroid")

train_x = data.train.x
val_x = data.val.x
test_x = data.test.x
test_y = data.test.y

physical_dims = data.physical_dims
feature_names = data.feature_names
```

This guarantees that training, evaluation, and explanation scripts consume the
same encoded variables and the same train/validation/test partition.
