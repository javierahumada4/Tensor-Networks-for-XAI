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


## Paper real-data suite: ADBench

The current real-data suite is pinned to ADBench commit
`3dac8221081e190f157d78e93bfa8867f90d0965` and contains:

| Dataset | Rows | Features | Anomalies |
|---|---:|---:|---:|
| annthyroid | 7,200 | 6 | 534 |
| cardio | 1,831 | 21 | 176 |
| cover | 286,048 | 10 | 2,747 |
| mammography | 11,183 | 6 | 260 |
| shuttle | 49,097 | 9 | 3,511 |
| vowels | 1,456 | 12 | 50 |

All six use the same preparation configuration:

- seed: `123`
- continuous quantile bins: `8`
- normal train fraction: `0.70`
- normal validation fraction: `0.15`
- remaining normals: test
- all anomalies: test only
- normal label: `0`

Generate the entire suite with:

```bash
python prepare_adbench.py artifacts/adbench
```

The script downloads the exact pinned source files, records SHA-256 hashes,
creates the persistent train/validation/test bundles, and writes
`adbench_suite_manifest.json`.

The workflow `.github/workflows/prepare-adbench.yml` runs the same pipeline,
validates every generated bundle, and publishes
`adbench-encoded-seed123-bins8` as a CI artifact.


## Frozen final MPS configuration

Hyperparameter selection was performed using **normal validation NLL only**.
Test anomaly labels were not used to select the model.

The final common configuration is stored in `paper_mps_config.json`:

- `max_bond_dim = 16`
- `epsilon_trunc = 0`
- initial learning rate `2e-3`
- batch size `1024`
- maximum 30 full-data loops
- learning-rate patience `5`, shrink factor `0.5`
- early-stopping patience `10`
- seed `123`

The final confirmation explicitly compared the strongest positive-tolerance
configurations against matched `epsilon_trunc=0` variants. The selected
configuration minimized mean relative normal-validation-NLL regret across the
six datasets.

`epsilon_trunc=0` means that the discarded-weight truncation machinery remains
implemented and tested, but the selected final model does not truncate by a
positive discarded-weight tolerance; local ranks are limited by the fixed
`max_bond_dim=16` cap.

The six final training artifacts were produced by workflow run
`36734660642`. Their artifact IDs and SHA-256 digests are recorded in
`final_training_manifest.json`.

## Frozen detection results

Threshold-free test evaluation is implemented in
`evaluate_adbench_final.py`. The required baseline is the independent
empirical-marginal model used to initialize the MPS.

| Dataset | MPS AUROC | Independent AUROC | MPS AUPRC | Independent AUPRC |
|---|---:|---:|---:|---:|
| annthyroid | 0.490 | 0.630 | 0.388 | 0.456 |
| cardio | 0.891 | 0.892 | 0.897 | 0.902 |
| cover | 0.543 | 0.619 | 0.337 | 0.324 |
| mammography | 0.848 | 0.871 | 0.562 | 0.577 |
| shuttle | 0.995 | 0.991 | 0.994 | 0.988 |
| vowels | 0.778 | 0.538 | 0.388 | 0.222 |

Macro averages:

- MPS AUROC: `0.7574`
- independent AUROC: `0.7568`
- MPS AUPRC: `0.5942`
- independent AUPRC: `0.5782`

The MPS improves normal validation likelihood on all six datasets, but this does
**not** translate into uniformly better anomaly ranking. This baseline is kept
explicit rather than selecting hyperparameters from test anomaly labels.

The exact table and evaluation provenance are versioned under `results/`.
