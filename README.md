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


## Raw interaction explanation experiments

The paper explanation experiments use the **raw** local interaction
decomposition only. No entropy-centered baseline is used.

For a sample `x` and non-empty feature subset `S`:

```
h_x(S) = -log p_S(x_S)

I_x(S) = h_x(S) - sum_{T proper non-empty subset of S} I_x(T)
```

When every interaction order is available,

```
NLL(x) = sum_{S non-empty} I_x(S).
```

The implementation lives in `mps_interactions.py`. Marginals `p_S(x_S)` are
computed exactly from the Born-MPS double layer; `raw_interactions()` uses
reused prefix contractions so that all subsets up to a chosen order are not
contracted independently from scratch. Unit tests compare these marginals
against exhaustive enumeration on small MPS models and verify exact NLL
reconstruction.

### Experiment 1 — Explanation fidelity vs interaction order

For all interactions up to order `m`,

```
A_hat_m(x) = sum_{1 <= |S| <= m} I_x(S)

c_m(x) = |NLL(x) - A_hat_m(x)| / NLL(x).
```

At `m=1`, this is the direct generalization of the previous first-order
correlation-share residual. Because raw interactions can have either sign,
`c_m` is not required to decrease monotonically.

### Experiment 2 — Explanation sparsity vs number of interactions

Interactions are separated by sign and ranked by magnitude within each sign.
At step `k`, the explanation retains the `k` largest positive interactions
and the `k` most negative interactions. Fidelity is measured with the same
relative NLL reconstruction residual `c_k`.

For aggregate curves, the common horizontal axis is the maximum budget `2k`,
because it is identical across samples even when one sign has fewer than `k`
available terms. Per-sample threshold tables report the **actual** number of
retained interactions. As with `c_m`, `c_k` is not assumed monotone.

The final real-data experiment uses uniform sampling without replacement from
the labeled test anomalies, seed `123`, and at most 100 anomalies per dataset
(all 50 anomalies for `vowels`). The maximum interaction order is fixed by a
predeclared computational budget of at most 5,000 subsets per sample, not by
the observed residuals.

| Dataset | Samples | Features | Max order | Complete decomposition | Subsets/sample |
|---|---:|---:|---:|---:|---:|
| annthyroid | 100 | 6 | 6 | yes | 63 |
| cardio | 100 | 21 | 3 | no | 1,561 |
| cover | 100 | 10 | 10 | yes | 1,023 |
| mammography | 100 | 6 | 6 | yes | 63 |
| shuttle | 100 | 9 | 9 | yes | 511 |
| vowels | 50 | 12 | 12 | yes | 4,095 |

The `cardio` results are therefore explicitly **order-truncated**. Its residual
after order 3 cannot be interpreted as evidence that higher-order interactions
are absent.

### Final interaction results

The table below reports the robust "stable" threshold: the first order (or
explanation size) after which every subsequently computed residual stays below
the requested tolerance. This definition is used because raw signed
interactions can temporarily worsen reconstruction through cancellation.

| Dataset | Median c1 | Stable order <=10% | Stable order <=5% | Interactions <=10% | Interactions <=5% |
|---|---:|---:|---:|---:|---:|
| annthyroid | 0.163 | 3 | 4 | 17 | 32 |
| cardio* | 0.068 | 3 (47/100 reached) | 3 (25/100 reached) | 100 | 250 |
| cover | 0.244 | 3 | 4 | 22 | 35 |
| mammography | 0.207 | 5 | 5 | 34 | 50 |
| shuttle | 0.128 | 3 | 4 | 36 | 74 |
| vowels | 0.017 | 2 | 8 | 73 | 272 |

`cardio*` reports medians only among samples that reached the corresponding
threshold within orders 1--3.

Two behaviors are particularly important for interpretation:

- low first-order residual does not imply that higher-order terms are
  negligible. For `vowels`, median `c_1` is about 1.7%, but the residual
  increases at intermediate orders before decreasing again;
- explanation complexity varies materially across datasets. For example,
  `mammography` generally needs order 5 for stable 5% reconstruction, while
  `annthyroid` and `cover` typically need order 4.

The final experiment provenance and workflow artifact hashes are stored in
`final_interactions_manifest.json`; the compact numerical summary is in
`results/interaction_experiments_summary.csv`.

Non-monotonicity is common rather than exceptional: 88% of `annthyroid`,
99% of `cover`, 93% of `mammography`, 85% of `shuttle`, and 100% of the
sampled `vowels` anomalies show at least one increase in `c_m` between
successive orders. This empirically supports the stable-threshold definition.
The corresponding audit is stored in `results/interaction_curve_audit.csv`.

For every dataset with a complete decomposition, the corrected full-order
reconstruction closes numerically with maximum relative residual below
`2.7e-16`. `cardio` is excluded from that closure statement because it is
intentionally truncated at order 3.

Generate paper figures from downloaded experiment outputs with:

```bash
python -m pip install -r requirements-plot.txt
python plot_interaction_results.py outputs/interactions outputs/figures
```


## Synthetic planted-order validation

The raw interaction method is also validated end-to-end on four synthetic
one-class problems with known interaction structure:

| Case | Planted order | Median c before planted order | Median stable order <=10% | Median stable order <=5% |
|---|---:|---:|---:|---:|
| marginal | 1 | - | 1 | 1 |
| pairwise | 2 | 0.686 | 2 | 2 |
| parity3 | 3 | 0.605 | 3 | 3 |
| parity4 | 4 | 0.535 | 4 | 4 |

Each dependency case uses a 2% normal violation probability so anomalous
configurations remain inside the support of the normal distribution. For parity
orders 2--4, all proper subsets of the planted variables have uniform
population marginals; the dependency first appears at the planted order.

For all 100 sampled anomalies in every case, the stable explanation order at
both 5% and 10% residual matches the planted order. The complete decomposition
closes at the planted/full order.

Reproduce the datasets with `prepare_synthetic.py`. The frozen validation
provenance is stored in `synthetic_interactions_manifest.json`, and the compact
table is in `results/synthetic_interaction_order.csv`.

## Publication figure artifact

The corrected final real-data interaction results are rendered by
`plot_interaction_results.py`. The frozen corrected figure workflow run is
`36749699314`, artifact `11113579074`, with digest:

```
sha256:994bfc7de6c8c6806b825f37f3694834845b0e0c2905d6f8d76de452d270cf96
```

The linear views do not impose an upper y-limit, so non-monotone residual
excursions and IQR bands are not clipped. Horizontal reference lines mark
5% and 10% relative reconstruction residual.

The main fidelity figure reports median `c_m` with IQR bands. The sparsity
figure uses the common maximum retained-interaction budget `2k` on the
horizontal axis, while the threshold tables report the actual number of retained
terms. `cardio` is shown with a dashed curve because its interaction expansion
is truncated at order 3.
