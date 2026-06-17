"""NSL-KDD encoder: raw connection records -> discrete MPS sites.

Step 1 of the pipeline. Each MPS site is a categorical variable taking a value
in ``[0, d_k)``, so every NSL-KDD feature has to be mapped to a small integer
alphabet. The policy, decided per feature by looking only at *normal* training
traffic, is:

* string categoricals (``protocol_type``, ``service``, ``flag``) keep their
  levels, plus an ``UNKNOWN`` slot for values unseen at fit time;
* a feature that is constant (or nearly so) over normal traffic collapses to a
  binary "equals the normal value vs differs" site — cheap, and exactly the
  signal an anomaly detector wants;
* a feature with only a handful of distinct normal values is treated as
  discrete, with an ``OTHER`` slot for anything else;
* remaining numeric features are bucketed into quantile bins fitted on normal
  traffic.

Fitting on the normal subset matters: the model is trained only on normal data,
so the encoding should describe what normal looks like and let attacks fall into
the ``UNKNOWN`` / ``OTHER`` / tail buckets. The fitted state is one
:class:`FeatureSpec` per feature; ``physical_dims`` is just their ``d`` values
and is what the MPS constructor needs.

Run as a script it reads ``KDDTrain+.txt`` / ``KDDTest+.txt`` from the data dir
and writes the encoded tensors, the metadata, and ``encoding_schema.json``.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------

COLUMNS: List[str] = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
CATEGORICAL_COLS = {"protocol_type", "service", "flag"}
DROP_COLS = {"num_outbound_cmds"}                  # zero-variance feature
META_COLS = {"label", "difficulty"}                # metadata, not model inputs

ATTACK_FAMILY: Dict[str, str] = {
    "normal": "normal",
    # DoS (10)
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos",
    # Probe (6)
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L (15)
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "sendmail": "r2l", "named": "r2l", "worm": "r2l",
    # U2R (8)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
    "httptunnel": "u2r",
}


# ----------------------------------------------------------------------
# Feature specification
# ----------------------------------------------------------------------

@dataclass
class FeatureSpec:
    """How a single feature is encoded into a discrete site of the MPS."""
    name: str
    kind: str                                # 'categorical' | 'discrete' | 'constant_normal' | 'numeric'
    d: int                                   # physical dim of this site
    vocab: Optional[List] = None             # for categorical and discrete features
    edges: Optional[List[float]] = None      # for numeric quantile bins
    normal_value: Optional[float] = None     # reference value for binary normal-vs-different encoding

    _vocab_map: Optional[Dict] = field(
        default=None, repr=False, compare=False,
    )


# ----------------------------------------------------------------------
#  Exceptions
# ----------------------------------------------------------------------

class EncodingError(ValueError):
    """Encoded data violates the schema (e.g. a column outside [0, d))."""

# ----------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------

class NSLKDDEncoder:
    """Discretizes each feature into an integer in [0, d_k) according
    to the policy described in the module docstring.

    Fitted state consists of one ``FeatureSpec`` per feature, in a
    deterministic order.  ``physical_dims`` is then just ``[s.d for s in
    specs]`` and is what the MPS constructor needs.
    """

    def __init__(
        self,
        target_d_numeric: int = 3,
        max_implicit_categorical: int = 8,
        quasi_constant_threshold: float = 0.95,
    ) -> None:
        """Configure the fitting thresholds.

        ``target_d_numeric`` is the number of quantile bins for genuinely numeric
        features. ``max_implicit_categorical`` is the cut-off below which a
        small-cardinality numeric column is treated as discrete rather than
        binned. ``quasi_constant_threshold`` is the mode share above which a
        feature is considered constant-in-normal and collapsed to a binary site.
        """
        if target_d_numeric < 2:
            raise ValueError("target_d_numeric must be >= 2")
        if max_implicit_categorical < 2:
            raise ValueError("max_implicit_categorical must be >= 2")
        if not (0.5 < quasi_constant_threshold < 1.0):
            raise ValueError(
                "quasi_constant_threshold must be in (0.5, 1.0); "
                f"got {quasi_constant_threshold}"
            )

        self.target_d_numeric = target_d_numeric
        self.max_implicit_categorical = max_implicit_categorical
        self.quasi_constant_threshold = quasi_constant_threshold
        self.specs: List[FeatureSpec] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> "NSLKDDEncoder":
        """Build one FeatureSpec per feature.

        Expects ``df_train`` with all NSL-KDD columns including ``label``.
        Uses ``label == 'normal'`` to slice the normal-only subset for
        deciding numeric bin edges.
        """
        normal_mask = (df_train["label"].str.rstrip(".") == "normal").to_numpy()
        df_normal = df_train.loc[normal_mask]

        specs: List[FeatureSpec] = []
        for col in COLUMNS:
            if col in DROP_COLS or col in META_COLS:
                continue
            specs.append(self._fit_one(col, df_normal[col]))

        self.specs = specs
        return self

    def _fit_one(
        self,
        col: str,
        x_normal: pd.Series,
    ) -> FeatureSpec:
        """Decide the encoding for a single column from its normal-traffic values.

        Walks the policy in order — string categorical, constant, quasi-constant,
        low-cardinality discrete, otherwise quantile-binned numeric — and returns
        the first :class:`FeatureSpec` that fits. ``x_normal`` is this column
        restricted to the normal rows of the training split.
        """
        # (1) Categorical string columns
        if col in CATEGORICAL_COLS:
            vocab_seen = sorted(x_normal.astype(str).unique().tolist())
            if "UNKNOWN" in vocab_seen:
                raise EncodingError(
                    f"'UNKNOWN' appears as a real category in column "
                    f"{col!r}; choose a different sentinel or rename "
                    f"the category before fitting."
                )
            vocab = vocab_seen + ["UNKNOWN"]
            return FeatureSpec(name=col, kind="categorical", d=len(vocab), vocab=vocab)

        # From here on, numeric features.
        x_normal = x_normal.astype(float)

        n_unique_normal = int(x_normal.nunique())

        # (2) Feature constant in train ∩ normal
        if n_unique_normal == 1:
            normal_value = float(x_normal.iloc[0])
            return FeatureSpec(
                name=col, kind="constant_normal", d=2,
                normal_value=normal_value,
            )

        # (3) Quasi-constant in normal
        normal_value_counts = x_normal.value_counts()
        mode_share = float(normal_value_counts.iloc[0]) / float(len(x_normal))
        if mode_share >= self.quasi_constant_threshold:
            mode_value = float(normal_value_counts.index[0])
            return FeatureSpec(
                name=col, kind="constant_normal", d=2,
                normal_value=mode_value,
            )

        # (4) Few distinct values in train∩normal
        if n_unique_normal <= self.max_implicit_categorical:
            vocab_int = sorted(x_normal.dropna().unique().tolist())
            if all(float(v).is_integer() for v in vocab_int):
                vocab_int = [int(v) for v in vocab_int]
            vocab_with_other = vocab_int + [None]
            return FeatureSpec(
                name=col, kind="discrete",
                d=len(vocab_with_other), vocab=vocab_with_other,
            )

        # (5) Numeric feature
        x_for_qcut = x_normal.to_numpy().astype(float)

        d = self.target_d_numeric
        try:
            _, edges_qcut = pd.qcut(
                x_for_qcut, q=d, retbins=True, duplicates="drop",
            )
            interior = edges_qcut[1:-1].tolist()
        except ValueError:
            interior = []

        if len(interior) == 0:
            median = float(np.median(x_for_qcut))
            return FeatureSpec(
                name=col, kind="numeric", d=2,
                edges=[-np.inf, median, np.inf],
            )

        effective_d = len(interior) + 1
        edges_full = [-np.inf] + interior + [np.inf]
        return FeatureSpec(
            name=col, kind="numeric", d=effective_d,
            edges=edges_full,
        )

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Map a DataFrame onto a (n_rows, n_features) LongTensor."""
        if not self.specs:
            raise RuntimeError("Encoder not fitted yet; call fit() first.")

        cols_out: List[np.ndarray] = []
        for spec in self.specs:
            col = df[spec.name]
            cols_out.append(self._transform_one(spec, col))

        arr = np.stack(cols_out, axis=1).astype(np.int64)
        return torch.from_numpy(arr)

    def _transform_one(self, spec: FeatureSpec, x: pd.Series) -> np.ndarray:
        """Apply one fitted ``FeatureSpec`` to a column, returning integer codes.

        Categorical/discrete columns go through the vocab lookup; constant
        features become a 0/1 same-vs-different flag; numeric features are cut at
        the fitted bin edges. A value that doesn't fit (unseen category, NaN bin)
        lands in the spec's catch-all slot or raises if there isn't one.
        """
        n = len(x)
        if spec.kind == "categorical":
            return self._encode_categorical(spec, x.astype(str))

        if spec.kind == "discrete":
            return self._encode_categorical(spec, x.astype(float))

        if spec.kind == "constant_normal":
            arr = x.astype(float).to_numpy()
            return (~np.isclose(arr, spec.normal_value)).astype(np.int64)

        if spec.kind == "numeric":
            arr = x.astype(float).to_numpy()
            out = pd.cut(arr, bins=spec.edges, labels=False, include_lowest=True).astype(np.int64)
            if np.isnan(out).any():
                raise EncodingError(
                    f"NaN bins for feature {spec.name!r}; edges={spec.edges}"
                )
            return out
        
        raise ValueError(f"unknown FeatureSpec.kind: {spec.kind!r}")

    def _encode_categorical(self, spec: FeatureSpec, x: pd.Series) -> np.ndarray:
        """Map values to vocab indices, routing misses to the catch-all slot.

        Builds (and caches) the value->index map on first use. For ``discrete``
        specs the "miss" slot is ``OTHER`` and float values are matched to the
        nearest vocab entry within tolerance; for string ``categorical`` specs it
        is ``UNKNOWN``. Either way an out-of-vocab value never raises here — it
        gets the catch-all index, which is the whole point of those slots.
        """
        if spec._vocab_map is None:
            spec._vocab_map = {v: i for i, v in enumerate(spec.vocab)}
        mapping = spec._vocab_map

        if spec.kind == "discrete":
            real_values = [v for v in spec.vocab if v is not None]
            other_idx = mapping[None]
            sample = real_values[0]
            if isinstance(sample, int):
                values = x.astype(float).round().astype(int)
                codes = values.map(mapping)
                return codes.fillna(other_idx).astype(np.int64).to_numpy()
            else:
                arr = x.astype(float).to_numpy()
                vocab_arr = np.asarray(real_values, dtype=float)
                nearest = np.abs(arr[:, None] - vocab_arr[None, :]).argmin(axis=1)
                within_tol = np.isclose(arr, vocab_arr[nearest])
                return np.where(within_tol, nearest, other_idx).astype(np.int64)

        if spec.kind == "categorical":
            unknown_idx = mapping.get("UNKNOWN")
            if unknown_idx is None:
                raise RuntimeError(
                    f"Feature {spec.name!r} has kind='categorical' but no "
                    "UNKNOWN slot in its vocab; refit the encoder."
                )
            codes = x.map(mapping)
            return codes.fillna(unknown_idx).astype(np.int64).to_numpy()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def physical_dims(self) -> List[int]:
        """Per-site physical dimensions ``[s.d for s in specs]`` — the MPS shape."""
        return [s.d for s in self.specs]

    @property
    def feature_names(self) -> List[str]:
        """Feature name at each site, in MPS order."""
        return [s.name for s in self.specs]

    def schema_dict(self) -> Dict:
        """Serialisable description of the fitted encoding.

        One entry per site (name, kind, dimension, and whichever of vocab / bin
        edges / normal value applies) plus the top-level ``physical_dims``. This
        is what gets written to ``encoding_schema.json`` and later read back by
        training, evaluation and explainability so they agree on the layout.
        ``None`` vocab entries are surfaced as ``"OTHER"`` and infinite bin edges
        as ``null``.
        """
        out: List[Dict] = []
        for i, s in enumerate(self.specs):
            entry = {
                "site": i, "name": s.name, "kind": s.kind, "d": s.d,
            }
            if s.vocab is not None:
                entry["vocab"] = [
                    "OTHER" if v is None
                    else (v.item() if isinstance(v, np.generic) else v)
                    for v in s.vocab
                ]
            if s.edges is not None:
                entry["edges"] = [None if not np.isfinite(e) else float(e) for e in s.edges]
            if s.normal_value is not None:
                entry["normal_value"] = float(s.normal_value)
            out.append(entry)
        return {
            "n_features": len(self.specs),
            "physical_dims": self.physical_dims,
            "features": out,
        }

# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

def load_split(path: Path) -> pd.DataFrame:
    """Read one NSL-KDD split and attach the derived label columns.

    Loads the headerless CSV, names the columns, strips the trailing dot some
    labels carry, and adds ``family`` (dos/probe/r2l/u2r/normal) and the binary
    ``is_attack``. Raises if the column count is wrong or a label isn't in the
    known attack-family map.
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: esperadas {len(COLUMNS)} columnas, halladas {df.shape[1]}")
    df.columns = COLUMNS
    df["label"] = df["label"].str.rstrip(".")
    df["family"] = df["label"].map(ATTACK_FAMILY)
    if df["family"].isna().any():
        unknown = sorted(df.loc[df["family"].isna(), "label"].unique())
        raise ValueError(f"Unknown labels in {path.name}: {unknown}")
    df["is_attack"] = (df["family"] != "normal").astype(int)
    return df


def build_meta(df: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """Pack the labels/metadata side-by-side with the encoded X."""
    families_sorted = sorted(set(ATTACK_FAMILY.values()))
    family_to_code = {f: i for i, f in enumerate(families_sorted)}
    return {
        "is_attack": torch.tensor(df["is_attack"].to_numpy(), dtype=torch.long),
        "family_code": torch.tensor(
            df["family"].map(family_to_code).to_numpy(), dtype=torch.long,
        ),
        "family_names": families_sorted,
        "difficulty": torch.tensor(df["difficulty"].to_numpy(), dtype=torch.long),
        "label": df["label"].tolist(),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(data_dir: Path) -> None:
    """Fit the encoder on the train split and write all encoder artefacts.

    Reads ``KDDTrain+.txt`` / ``KDDTest+.txt`` from ``data_dir``, fits the
    encoding on training data, transforms both splits, sanity-checks that every
    encoded column stays inside its declared range, and writes the encoded
    tensors, per-split metadata and ``encoding_schema.json`` back into the same
    directory for the training step to pick up.
    """
    train = load_split(data_dir / "KDDTrain+.txt")
    test = load_split(data_dir / "KDDTest+.txt")
    logger.info("loaded: %d train rows, %d test rows", len(train), len(test))

    encoder = NSLKDDEncoder(
        target_d_numeric=4,
        max_implicit_categorical=8,
    )
    encoder.fit(train)

    train_X = encoder.transform(train)
    test_X = encoder.transform(test)

    physical_dims = encoder.physical_dims
    for split_name, X in (("train", train_X), ("test", test_X)):
        col_max = X.max(dim=0).values
        col_min = X.min(dim=0).values
        for k, (lo, hi, d) in enumerate(zip(col_min.tolist(), col_max.tolist(), physical_dims)):
            if lo < 0 or hi >= d:
                raise EncodingError(
                    f"{split_name}: site {k} ({encoder.feature_names[k]}) "
                    f"has range [{lo}, {hi}] outside [0, {d})"
                )
            
    logger.info(
        "schema: %d sites, sum(d)=%d, max(d)=%d",
        len(encoder.specs), sum(physical_dims), max(physical_dims),
    )

    kind_counts = Counter(s.kind for s in encoder.specs)
    logger.info("  by kind: %s", dict(kind_counts))

    torch.save(train_X, data_dir / "train_X.pt")
    torch.save(test_X, data_dir / "test_X.pt")
    torch.save(build_meta(train), data_dir / "train_meta.pt")
    torch.save(build_meta(test), data_dir / "test_meta.pt")

    schema_json = data_dir / "encoding_schema.json"
    schema_json.write_text(json.dumps(encoder.schema_dict(), indent=2))

    logger.info(
        "wrote artefacts to %s/ "
        "(train_X.pt, test_X.pt, train_meta.pt, test_meta.pt, encoding_schema.json)",
        data_dir,
    )
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./nsl_kdd")
    main(data_dir)
