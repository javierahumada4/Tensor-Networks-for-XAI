"""Dataset-agnostic tabular encoder for discrete Born-MPS models.

One original feature maps to one MPS site. Continuous variables are discretized
with quantile bins fitted only on the training data and receive explicit LOW,
HIGH, and MISSING states. Categorical variables keep one state per observed
training category plus UNKNOWN and MISSING states.

The encoder is deliberately label-agnostic. One-class split construction lives
in data_artifacts.py and fits this encoder only on normal training rows.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


def _python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _category_key(value) -> str:
    """Stable typed representation used internally for categorical lookup."""
    value = _python_scalar(value)
    type_name = type(value).__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        payload = json.dumps(str(value), ensure_ascii=False)
    return f"{type_name}:{payload}"


def _category_label(value) -> str:
    value = _python_scalar(value)
    return str(value)


@dataclass
class FeatureSpec:
    name: str
    kind: str
    physical_dim: int
    regular_states: int
    missing_code: int
    low_code: Optional[int] = None
    high_code: Optional[int] = None
    unknown_code: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    inner_edges: Optional[List[float]] = None
    category_keys: Optional[List[str]] = None
    category_labels: Optional[List[str]] = None


class TabularEncoder:
    """Encode heterogeneous tabular data into integer MPS configurations.

    Parameters
    ----------
    n_bins:
        Target number of quantile bins for continuous variables. Repeated
        quantiles are collapsed, so the actual number of regular states may be
        smaller.
    categorical_columns / continuous_columns:
        Optional explicit type declarations. Columns not listed in either group
        are inferred: numeric -> continuous, everything else -> categorical.
    drop_columns:
        Features explicitly excluded before fitting.
    max_categories:
        Maximum number of observed training categories allowed for one
        categorical feature. Higher-cardinality variables must be explicitly
        transformed or dropped rather than silently hashed.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        n_bins: int = 8,
        categorical_columns: Optional[Sequence[str]] = None,
        continuous_columns: Optional[Sequence[str]] = None,
        drop_columns: Optional[Sequence[str]] = None,
        max_categories: int = 64,
    ) -> None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        if max_categories < 1:
            raise ValueError(
                f"max_categories must be >= 1, got {max_categories}"
            )
        self.n_bins = int(n_bins)
        self.categorical_columns = list(categorical_columns or [])
        self.continuous_columns = list(continuous_columns or [])
        self.drop_columns = list(drop_columns or [])
        self.max_categories = int(max_categories)

        overlap = set(self.categorical_columns) & set(self.continuous_columns)
        if overlap:
            raise ValueError(
                f"columns cannot be both categorical and continuous: "
                f"{sorted(overlap)}"
            )

        self.feature_names: List[str] = []
        self.specs: List[FeatureSpec] = []
        self._fitted = False

    @property
    def physical_dims(self) -> List[int]:
        self._require_fitted()
        return [spec.physical_dim for spec in self.specs]

    @property
    def feature_types(self) -> Dict[str, str]:
        self._require_fitted()
        return {spec.name: spec.kind for spec in self.specs}

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("encoder has not been fitted")

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        declared = (
            set(self.categorical_columns)
            | set(self.continuous_columns)
            | set(self.drop_columns)
        )
        missing = sorted(declared - set(frame.columns))
        if missing:
            raise ValueError(f"declared columns not present in data: {missing}")

    def _kind_for_column(self, frame: pd.DataFrame, name: str) -> str:
        if name in self.categorical_columns:
            return "categorical"
        if name in self.continuous_columns:
            return "continuous"
        if pd.api.types.is_numeric_dtype(frame[name].dtype) and not pd.api.types.is_bool_dtype(
            frame[name].dtype
        ):
            return "continuous"
        return "categorical"

    @staticmethod
    def _numeric_series(series: pd.Series, name: str) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        invalid = series.notna() & numeric.isna()
        if invalid.any():
            example = series[invalid].iloc[0]
            raise ValueError(
                f"continuous feature {name!r} contains non-numeric value "
                f"{example!r}"
            )
        return numeric.astype(float)

    def _fit_continuous(self, series: pd.Series, name: str) -> FeatureSpec:
        numeric = self._numeric_series(series, name)
        observed = numeric.dropna().to_numpy(dtype=float)
        if observed.size == 0:
            raise ValueError(
                f"continuous feature {name!r} is entirely missing in training; "
                "drop or transform it explicitly"
            )

        min_value = float(np.min(observed))
        max_value = float(np.max(observed))

        if min_value == max_value:
            inner_edges: List[float] = []
        else:
            quantiles = np.quantile(
                observed,
                np.linspace(0.0, 1.0, self.n_bins + 1),
            )
            inner = quantiles[1:-1]
            inner = inner[(inner > min_value) & (inner < max_value)]
            inner_edges = [
                float(value)
                for value in np.unique(inner)
            ]

        regular_states = len(inner_edges) + 1
        low_code = regular_states
        high_code = regular_states + 1
        missing_code = regular_states + 2

        return FeatureSpec(
            name=name,
            kind="continuous",
            physical_dim=regular_states + 3,
            regular_states=regular_states,
            missing_code=missing_code,
            low_code=low_code,
            high_code=high_code,
            min_value=min_value,
            max_value=max_value,
            inner_edges=inner_edges,
        )

    def _fit_categorical(self, series: pd.Series, name: str) -> FeatureSpec:
        observed = series[series.notna()]
        if len(observed) == 0:
            raise ValueError(
                f"categorical feature {name!r} is entirely missing in training; "
                "drop or transform it explicitly"
            )

        keyed = {}
        for value in observed.tolist():
            key = _category_key(value)
            keyed.setdefault(key, _category_label(value))

        category_keys = sorted(keyed)
        if len(category_keys) > self.max_categories:
            raise ValueError(
                f"categorical feature {name!r} has {len(category_keys)} "
                f"training categories, exceeding max_categories="
                f"{self.max_categories}. Explicitly transform or drop "
                "high-cardinality identifiers instead of hashing them."
            )

        category_labels = [keyed[key] for key in category_keys]
        regular_states = len(category_keys)
        unknown_code = regular_states
        missing_code = regular_states + 1

        return FeatureSpec(
            name=name,
            kind="categorical",
            physical_dim=regular_states + 2,
            regular_states=regular_states,
            missing_code=missing_code,
            unknown_code=unknown_code,
            category_keys=category_keys,
            category_labels=category_labels,
        )

    def fit(self, frame: pd.DataFrame) -> "TabularEncoder":
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if len(frame) == 0:
            raise ValueError("cannot fit encoder on an empty DataFrame")

        self._validate_columns(frame)
        feature_names = [
            name for name in frame.columns if name not in self.drop_columns
        ]
        if not feature_names:
            raise ValueError("no features remain after drop_columns")

        specs: List[FeatureSpec] = []
        for name in feature_names:
            kind = self._kind_for_column(frame, name)
            if kind == "continuous":
                specs.append(self._fit_continuous(frame[name], name))
            else:
                specs.append(self._fit_categorical(frame[name], name))

        self.feature_names = feature_names
        self.specs = specs
        self._fitted = True
        return self

    @staticmethod
    def _transform_continuous(series: pd.Series, spec: FeatureSpec) -> np.ndarray:
        numeric = TabularEncoder._numeric_series(series, spec.name)
        values = numeric.to_numpy(dtype=float)
        output = np.empty(len(values), dtype=np.int64)

        missing = np.isnan(values)
        low = (~missing) & (values < spec.min_value)
        high = (~missing) & (values > spec.max_value)
        regular = ~(missing | low | high)

        output[missing] = int(spec.missing_code)
        output[low] = int(spec.low_code)
        output[high] = int(spec.high_code)

        inner_edges = np.asarray(spec.inner_edges or [], dtype=float)
        if regular.any():
            output[regular] = np.searchsorted(
                inner_edges,
                values[regular],
                side="right",
            ).astype(np.int64)
        return output

    @staticmethod
    def _transform_categorical(series: pd.Series, spec: FeatureSpec) -> np.ndarray:
        mapping = {
            key: index
            for index, key in enumerate(spec.category_keys or [])
        }
        output = np.empty(len(series), dtype=np.int64)
        values = series.tolist()
        for index, value in enumerate(values):
            if pd.isna(value):
                output[index] = int(spec.missing_code)
            else:
                output[index] = mapping.get(
                    _category_key(value),
                    int(spec.unknown_code),
                )
        return output

    def transform(self, frame: pd.DataFrame) -> torch.Tensor:
        self._require_fitted()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")

        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"input is missing fitted features: {missing}")

        columns: List[np.ndarray] = []
        for spec in self.specs:
            if spec.kind == "continuous":
                encoded = self._transform_continuous(frame[spec.name], spec)
            else:
                encoded = self._transform_categorical(frame[spec.name], spec)
            columns.append(encoded)

        matrix = np.column_stack(columns)
        return torch.from_numpy(matrix).long()

    def fit_transform(self, frame: pd.DataFrame) -> torch.Tensor:
        return self.fit(frame).transform(frame)

    def to_dict(self) -> Dict:
        self._require_fitted()
        return {
            "format_version": self.FORMAT_VERSION,
            "config": {
                "n_bins": self.n_bins,
                "categorical_columns": self.categorical_columns,
                "continuous_columns": self.continuous_columns,
                "drop_columns": self.drop_columns,
                "max_categories": self.max_categories,
            },
            "feature_names": self.feature_names,
            "specs": [asdict(spec) for spec in self.specs],
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "TabularEncoder":
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(
                f"unsupported encoder format version "
                f"{payload.get('format_version')!r}"
            )
        config = payload["config"]
        encoder = cls(**config)
        encoder.feature_names = list(payload["feature_names"])
        encoder.specs = [FeatureSpec(**spec) for spec in payload["specs"]]
        encoder._fitted = True
        return encoder

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TabularEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
