import json

import numpy as np
import pandas as pd
import pytest
import torch

from data_artifacts import (
    OneClassSplitConfig,
    load_encoded_bundle,
    prepare_one_class_artifacts,
)
from encoder import TabularEncoder
from mps import MPS
from prepare_dataset import _load_input


def test_continuous_special_states_and_categorical_unknown_missing() -> None:
    train = pd.DataFrame(
        {
            "value": np.arange(8, dtype=float),
            "kind": ["a", "a", "b", "b", "a", "b", "a", "b"],
        }
    )
    encoder = TabularEncoder(
        n_bins=4,
        categorical_columns=["kind"],
        continuous_columns=["value"],
    ).fit(train)

    transformed = encoder.transform(
        pd.DataFrame(
            {
                "value": [-1.0, 8.0, np.nan, 3.5],
                "kind": ["a", "new", None, "b"],
            }
        )
    )

    value_spec = next(spec for spec in encoder.specs if spec.name == "value")
    kind_spec = next(spec for spec in encoder.specs if spec.name == "kind")

    assert transformed[0, 0].item() == value_spec.low_code
    assert transformed[1, 0].item() == value_spec.high_code
    assert transformed[2, 0].item() == value_spec.missing_code
    assert 0 <= transformed[3, 0].item() < value_spec.regular_states

    assert transformed[1, 1].item() == kind_spec.unknown_code
    assert transformed[2, 1].item() == kind_spec.missing_code



def test_binary_numeric_feature_preserves_both_states() -> None:
    frame = pd.DataFrame({"binary": [0, 1, 0, 1, 0, 1]})
    encoder = TabularEncoder().fit(frame)
    spec = encoder.specs[0]

    assert spec.kind == "discrete_numeric"
    assert spec.regular_states == 2
    assert torch.unique(encoder.transform(frame)).numel() == 2

    transformed = encoder.transform(
        pd.DataFrame({"binary": [0, 1, 2, np.nan]})
    ).squeeze(1)
    assert transformed[0].item() != transformed[1].item()
    assert transformed[2].item() == spec.unknown_code
    assert transformed[3].item() == spec.missing_code


def test_low_cardinality_integer_numeric_is_discrete_unless_forced_continuous() -> None:
    frame = pd.DataFrame({"x": [0, 1, 2, 0, 1, 2]})

    inferred = TabularEncoder(max_discrete_numeric_states=8).fit(frame)
    assert inferred.specs[0].kind == "discrete_numeric"
    assert inferred.specs[0].regular_states == 3

    forced = TabularEncoder(
        n_bins=2,
        continuous_columns=["x"],
    ).fit(frame)
    assert forced.specs[0].kind == "continuous"


def test_constant_continuous_feature_is_preserved() -> None:
    frame = pd.DataFrame({"constant": [5.0, 5.0, 5.0, 5.0]})
    encoder = TabularEncoder().fit(frame)
    spec = encoder.specs[0]

    assert spec.kind == "continuous"
    assert spec.regular_states == 1
    assert spec.physical_dim == 4

    encoded = encoder.transform(
        pd.DataFrame({"constant": [4.0, 5.0, 6.0, np.nan]})
    ).squeeze(1)

    assert encoded.tolist() == [
        spec.low_code,
        0,
        spec.high_code,
        spec.missing_code,
    ]


def test_high_cardinality_categorical_requires_explicit_decision() -> None:
    frame = pd.DataFrame({"id_like": [f"id-{i}" for i in range(10)]})

    with pytest.raises(ValueError, match="high-cardinality|exceeding"):
        TabularEncoder(
            categorical_columns=["id_like"],
            max_categories=4,
        ).fit(frame)


def test_encoder_json_roundtrip_preserves_transform(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "c": ["red", "blue", "red", "green"],
        }
    )
    encoder = TabularEncoder(
        n_bins=2,
        categorical_columns=["c"],
    ).fit(frame)

    path = tmp_path / "encoder.json"
    encoder.save(path)
    loaded = TabularEncoder.load(path)

    torch.testing.assert_close(
        loaded.transform(frame),
        encoder.transform(frame),
    )
    assert loaded.feature_names == encoder.feature_names
    assert loaded.physical_dims == encoder.physical_dims


def _example_dataset():
    n = 24
    features = pd.DataFrame(
        {
            "continuous": np.linspace(-2.0, 2.0, n),
            "category": [
                "normal-a" if i % 2 == 0 else "normal-b"
                for i in range(n)
            ],
        }
    )
    labels = pd.Series([0] * 18 + [1] * 6)
    features.loc[18:, "category"] = "anomaly-only"
    return features, labels


def test_one_class_artifacts_are_disjoint_complete_and_reproducible(tmp_path) -> None:
    features, labels = _example_dataset()
    config = OneClassSplitConfig(
        train_fraction=0.60,
        val_fraction=0.20,
        seed=7,
    )

    bundle_a = prepare_one_class_artifacts(
        features,
        labels,
        tmp_path / "a",
        normal_label=0,
        dataset_name="toy",
        split_config=config,
        n_bins=4,
        categorical_columns=["category"],
    )
    bundle_b = prepare_one_class_artifacts(
        features,
        labels,
        tmp_path / "b",
        normal_label=0,
        dataset_name="toy",
        split_config=config,
        n_bins=4,
        categorical_columns=["category"],
    )

    assert torch.equal(bundle_a.train.row_indices, bundle_b.train.row_indices)
    assert torch.equal(bundle_a.val.row_indices, bundle_b.val.row_indices)
    assert torch.equal(bundle_a.test.row_indices, bundle_b.test.row_indices)

    train_idx = set(bundle_a.train.row_indices.tolist())
    val_idx = set(bundle_a.val.row_indices.tolist())
    test_idx = set(bundle_a.test.row_indices.tolist())

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert train_idx | val_idx | test_idx == set(range(len(features)))

    assert bundle_a.train.y.sum().item() == 0
    assert bundle_a.val.y.sum().item() == 0
    assert bundle_a.test.y.sum().item() == 6

    anomaly_rows = set(range(18, 24))
    assert anomaly_rows.issubset(test_idx)
    assert anomaly_rows.isdisjoint(train_idx)
    assert anomaly_rows.isdisjoint(val_idx)


def test_encoder_is_fitted_only_on_normal_training_rows(tmp_path) -> None:
    features, labels = _example_dataset()
    bundle = prepare_one_class_artifacts(
        features,
        labels,
        tmp_path / "bundle",
        normal_label=0,
        split_config=OneClassSplitConfig(
            train_fraction=0.60,
            val_fraction=0.20,
            seed=3,
        ),
        categorical_columns=["category"],
    )

    category_spec = next(
        spec for spec in bundle.encoder.specs if spec.name == "category"
    )
    assert "anomaly-only" not in (category_spec.category_labels or [])

    anomaly_mask = bundle.test.y == 1
    category_site = bundle.encoder.feature_names.index("category")
    anomaly_codes = bundle.test.x[anomaly_mask, category_site]
    assert torch.all(anomaly_codes == category_spec.unknown_code)


def test_persisted_bundle_roundtrip_is_exact(tmp_path) -> None:
    features, labels = _example_dataset()
    root = tmp_path / "bundle"

    created = prepare_one_class_artifacts(
        features,
        labels,
        root,
        normal_label=0,
        dataset_name="toy",
        split_config=OneClassSplitConfig(seed=11),
        n_bins=4,
        categorical_columns=["category"],
    )
    loaded = load_encoded_bundle(root)

    for original, restored in [
        (created.train, loaded.train),
        (created.val, loaded.val),
        (created.test, loaded.test),
    ]:
        assert torch.equal(original.x, restored.x)
        assert torch.equal(original.y, restored.y)
        assert torch.equal(original.row_indices, restored.row_indices)

    split_indices = torch.load(
        root / "split_indices.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert torch.equal(split_indices["train"], created.train.row_indices)
    assert torch.equal(split_indices["val"], created.val.row_indices)
    assert torch.equal(split_indices["test"], created.test.row_indices)

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["feature_names"] == created.encoder.feature_names
    assert manifest["physical_dims"] == created.encoder.physical_dims
    assert manifest["counts"]["test_anomaly"] == 6



def test_persisted_bundle_can_initialize_mps_directly(tmp_path) -> None:
    features, labels = _example_dataset()
    root = tmp_path / "bundle"
    prepare_one_class_artifacts(
        features,
        labels,
        root,
        normal_label=0,
        categorical_columns=["category"],
        n_bins=4,
    )
    data = load_encoded_bundle(root)

    model = MPS.from_empirical_frequencies(
        data.train.x,
        physical_dims=data.physical_dims,
        dtype=torch.float64,
    )

    assert model.physical_dims == data.physical_dims
    assert model.num_sites == len(data.feature_names)
    assert torch.isfinite(model.nll(data.val.x))


def test_prepare_dataset_npz_loader_uses_stable_feature_names(tmp_path) -> None:
    path = tmp_path / "toy.npz"
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([0] * 8 + [1] * 2)
    np.savez(path, X=X, y=y)

    features, labels = _load_input(path, label_column=None)

    assert features.columns.tolist() == ["x0", "x1"]
    np.testing.assert_array_equal(features.to_numpy(), X)
    np.testing.assert_array_equal(labels.to_numpy(), y)
