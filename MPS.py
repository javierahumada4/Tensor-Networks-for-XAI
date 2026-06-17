"""Matrix Product State core: the model itself, nothing else.

This module holds the :class:`MPS` class and the handful of errors it can
raise. Everything here is about representing the state and answering the
questions training and scoring need: amplitudes, the norm, log-probabilities,
NLL, plus the canonical forms and two-site surgery (merge / SVD-split /
swap / permute) that DMRG relies on.

Two things were deliberately kept *out* of this file to stop it growing into
a god-class:

* sampling lives in ``mps_generative`` (:class:`MPSSampler`),
* reduced density matrices and information measures live in
  ``mps_explainability`` (:class:`MPSExplainer`).

Both read an ``MPS`` from the outside through its public methods and the
``_as_matrices`` helper, so the contract between them is small on purpose.

Conventions used throughout:

* Sites are 0-indexed; site ``k`` carries a tensor of shape
  ``(D_{k-1}, d_k, D_k)`` with open boundaries ``D_0 = D_N = 1``.
* "configurations" are integer tensors of shape ``(batch, num_sites)`` where
  column ``k`` takes values in ``[0, d_k)``.
* Real (float32/64) and complex (complex64/128) dtypes are both supported;
  the probability is the Born rule ``P(v) = |Psi(v)|^2 / Z``.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
#  Exceptions
# ----------------------------------------------------------------------

class MPSError(Exception):
    """Base class for all MPS errors."""

class MPSShapeError(MPSError, ValueError):
    """Tensor shape or index does not match the MPS configuration."""

class MPSNumericalError(MPSError, RuntimeError):
    """A numerical pathology was detected (NaN, overflow, vanishing norm)."""

# ----------------------------------------------------------------------
#  Checkpoint format
# ----------------------------------------------------------------------

_DTYPE_MAP: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
_REVERSE_DTYPE_MAP: Dict[torch.dtype, str] = {value: key for key, value in _DTYPE_MAP.items()}

class MPS(nn.Module):
    """An open-boundary Matrix Product State used as a Born machine.

    The state is a chain of ``num_sites`` rank-3 tensors. Site ``k`` has shape
    ``(D_{k-1}, d_k, D_k)``; the outer bonds are pinned to 1, so the whole
    contraction collapses to a scalar amplitude ``Psi(v)`` for any
    configuration ``v``. Probabilities follow the Born rule,
    ``P(v) = |Psi(v)|^2 / Z`` with ``Z = <Psi|Psi>``.

    Physical dimensions may differ from site to site (``physical_dims`` as a
    list), which is what lets a single chain model a row of mixed
    NSL-KDD features, each discretised to its own number of levels.

    The tensors are stored as an ``nn.ParameterList`` so the object is a normal
    ``nn.Module`` (movable with ``.to``, saveable, etc.), but training is done by
    DMRG sweeps rather than autograd — the trainer writes directly into
    ``site_tensors[k].data``.

    Notes on numerics: amplitudes are computed with per-site rescaling and the
    running scale tracked in log space, so chains of dozens of sites don't
    under/overflow. ``float64`` is the safe default for long chains.
    """

    # Above this fraction of discarded SVD weight, truncations log a warning.
    _discarded_weight_warn_threshold: float = 0.1

    def __init__(
            self,
            num_sites: int,
            bond_dim: int,
            physical_dims: Union[int, Sequence[int]] = 2,
            dtype: torch.dtype = torch.float32,
            init_std: Optional[float] = None,
            *,
            _skip_init: bool = False,
    ) -> None:
        """Build a fresh MPS.

        ``physical_dims`` is either a single int (same ``d`` everywhere) or one
        value per site. ``init_std`` controls the spread of the random Gaussian
        initialisation; left as ``None`` it defaults to ``1/sqrt(bond_dim)``,
        which keeps the initial amplitudes from blowing up with chain length.

        ``_skip_init`` is for internal use by :meth:`load`: it allocates the
        parameter list with zeros so the saved tensors can be copied straight
        in, skipping the (wasted) random init.
        """
        super().__init__()

        if num_sites < 2:
            raise MPSShapeError(f"num_sites must be >= 2, got {num_sites}")
        if bond_dim < 1:
            raise MPSShapeError(f"bond_dim must be >= 1, got {bond_dim}")
        if dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            raise TypeError(
                f"Unsupported dtype: {dtype}. Use float32/float64/complex64/complex128."
            )

        self.num_sites = num_sites
        self.bond_dim = bond_dim
        self.dtype = dtype

        self.physical_dims: List[int] = self._normalise_physical_dims(physical_dims)

        if _skip_init:
            self.site_tensors = self._empty_init()
        else:
            self.site_tensors = self._normal_init(init_std)

    def _randn(self, *shape) -> torch.Tensor:
        """Gaussian noise matching ``self.dtype``.

        For complex dtypes the real and imaginary parts are drawn
        independently and scaled by ``1/sqrt(2)`` so that ``E[|z|^2] = 1``,
        i.e. a complex entry has the same expected magnitude as a real one.
        """
        if self.dtype in (torch.complex64, torch.complex128):
            base_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
            real_part = torch.randn(*shape, dtype=base_dtype)
            imag_part = torch.randn(*shape, dtype=base_dtype)
            complex_tensor = (real_part + 1j * imag_part) / math.sqrt(2)
            return complex_tensor.to(self.dtype)
        else:
            return torch.randn(*shape, dtype=self.dtype)

    def _normal_init(self, init_std: Optional[float] = None) -> nn.ParameterList:
        """Random Gaussian site tensors with the correct boundary shapes.

        The first and last tensors have a trivial outer bond (1), the bulk
        tensors are square in their bonds. Default ``init_std`` of
        ``1/sqrt(bond_dim)`` roughly normalises the per-site transfer so the
        amplitude of a random state stays in a sane range.
        """
        if init_std is None:
            init_std = 1.0 / math.sqrt(self.bond_dim)

        tensor_list: List[nn.Parameter] = []

        left_tensor = self._randn(1, self.physical_dims[0], self.bond_dim) * init_std
        tensor_list.append(nn.Parameter(left_tensor))

        for k in range(1, self.num_sites-1):
            bulk_tensor = self._randn(self.bond_dim, self.physical_dims[k], self.bond_dim) * init_std
            tensor_list.append(nn.Parameter(bulk_tensor))

        right_tensor = self._randn(self.bond_dim, self.physical_dims[-1], 1) * init_std
        tensor_list.append(nn.Parameter(right_tensor))

        return nn.ParameterList(tensor_list)
    
    def _empty_init(self) -> nn.ParameterList:
        """Zero-filled site tensors with the right shapes.

        Used only by :meth:`load`, which immediately overwrites the data with
        the saved tensors. Avoids paying for a random init that gets thrown away.
        """
        tensor_list: List[nn.Parameter] = []

        left_tensor = torch.zeros(1, self.physical_dims[0], self.bond_dim, dtype=self.dtype)
        tensor_list.append(nn.Parameter(left_tensor))

        for k in range(1, self.num_sites-1):
            bulk_tensor = torch.zeros(self.bond_dim, self.physical_dims[k], self.bond_dim, dtype=self.dtype)
            tensor_list.append(nn.Parameter(bulk_tensor))

        right_tensor = torch.zeros(self.bond_dim, self.physical_dims[-1], 1, dtype=self.dtype)
        tensor_list.append(nn.Parameter(right_tensor))

        return nn.ParameterList(tensor_list)
    
    def _normalise_physical_dims(self, physical_dim: Union[int, Sequence[int]] = 2) -> List[int]:
        """Turn the ``physical_dims`` argument into a per-site list.

        Accepts a single int (broadcast to every site) or an explicit sequence
        of length ``num_sites``. Every dimension must be at least 2 — a site
        with a single level carries no information and would break the SVD
        bookkeeping.
        """
        if isinstance(physical_dim, int):
            if physical_dim < 2:
                raise MPSShapeError(f"physical_dim must be >= 2, got {physical_dim}")
            physical_dims: List[int] = [physical_dim] * self.num_sites
        else:
            physical_dims = list(physical_dim)
            if len(physical_dims) != self.num_sites:
                raise MPSShapeError(
                    f"physical_dim sequence has length {len(physical_dims)}, "
                    f"expected {self.num_sites}"
                )
            for k, d in enumerate(physical_dims):
                if not isinstance(d, int):
                    raise TypeError(
                        f"physical_dim[{k}]={d!r} must be int, got {type(d).__name__}"
                    )
                if d < 2:
                    raise MPSShapeError(
                        f"physical_dim[{k}]={d} must be >= 2"
                    )
        return physical_dims
    
    # ------------------------------------------------------------------
    #  Input validation helpers
    # ------------------------------------------------------------------

    def _validate_configurations(self, configurations: torch.Tensor) -> None:
        """Check shape and value range of a configurations tensor."""
        if configurations.dim() != 2:
            raise MPSShapeError(
                "configurations must be 2D with shape (batch_size, num_sites), "
                f"got shape {tuple(configurations.shape)}"
            )
        if configurations.shape[1] != self.num_sites:
            raise MPSShapeError(
                f"Expected {self.num_sites} sites, got {configurations.shape[1]}"
            )
        if configurations.numel() == 0:
            return
       
        if self.is_homogeneous:
            physical_dim = self.physical_dims[0]
            min_value = configurations.min().item()
            max_value = configurations.max().item()
            if min_value < 0 or max_value >= physical_dim:
                raise MPSShapeError(
                    f"configurations values must be in [0, {physical_dim}), "
                    f"got range [{min_value}, {max_value}]"
                )
            return
        col_min = configurations.min(dim=0).values
        col_max = configurations.max(dim=0).values
        dims = torch.tensor(
            self.physical_dims, device=configurations.device, dtype=col_max.dtype
        )
        out_of_range = (col_min < 0) | (col_max >= dims)
        if out_of_range.any():
            bad_sites = out_of_range.nonzero(as_tuple=False).flatten().tolist()
            details = "; ".join(
                f"site {k}: range [{col_min[k].item()}, {col_max[k].item()}] "
                f"outside [0, {self.physical_dims[k]})"
                for k in bad_sites
            )
            raise MPSShapeError(f"configurations out of range -- {details}")

    def _validate_truncation(
        self, max_bond_dim: Optional[int], cutoff: float
    ) -> None:
        """Check SVD truncation hyperparameters."""
        if max_bond_dim is not None and max_bond_dim < 1:
            raise MPSShapeError(f"max_bond_dim must be >= 1 or None, got {max_bond_dim}")
        if cutoff < 0:
            raise ValueError(f"cutoff must be >= 0, got {cutoff}")
        
    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_matrices(A: torch.Tensor) -> torch.Tensor:
        """View a site tensor ``(D_l, d, D_r)`` as ``d`` matrices ``(d, D_l, D_r)``.

        Most contractions are cleaner when the physical index is on the outside:
        slicing ``[v]`` then gives the transfer matrix for physical value ``v``.
        This is just a ``permute`` (a view), not a copy.
        """
        return A.permute(1, 0, 2)
    
    def select_matrices(
        self, site: int, values: torch.Tensor
    ) -> torch.Tensor:
        """Return the per-sample transfer matrices of ``site`` for ``values``.

        For each entry ``v`` in ``values``, picks the slice ``A[:, v, :]`` of
        this site's tensor (shape ``D_{site-1} x D_site``) and stacks them
        along a leading batch axis.
        """
        if not 0 <= site < self.num_sites:
            raise IndexError(
                f"site {site} out of range [0, {self.num_sites})"
            )
        if values.dim() != 1:
            raise ValueError(
                f"values must be 1-D, got shape {tuple(values.shape)}"
            )
        if values.dtype != torch.long:
            values = values.long()
        return self._as_matrices(self.site_tensors[site][:, values, :])
    
    # ------------------------------------------------------------------
    #  Device / dtype movement
    # ------------------------------------------------------------------
        
    def to(self, *args, **kwargs):
        """
        Move and/or cast the MPS, respecting the configured dtype.
        """
        new_dtype: Optional[torch.dtype] = kwargs.pop("dtype", None)
        new_device = kwargs.pop("device", None)
        positional: List = []
        for argument in args:
            if isinstance(argument, torch.dtype):
                if new_dtype is None:
                    new_dtype = argument
            elif isinstance(argument, (torch.device, str)):
                if new_device is None:
                    new_device = argument
            else:
                positional.append(argument)
 
        if new_dtype is not None:
            self_is_complex = self.dtype in (torch.complex64, torch.complex128)
            new_is_complex = new_dtype in (torch.complex64, torch.complex128)
            if self_is_complex != new_is_complex:
                raise TypeError(
                    f"Cannot change between real and complex dtype via .to() "
                    f"({self.dtype!r} -> {new_dtype!r}). "
                    "Construct a new MPS instead."
                )
            self.dtype = new_dtype
 
        rebuilt_kwargs = dict(kwargs)
        if new_dtype is not None:
            rebuilt_kwargs["dtype"] = new_dtype
        if new_device is not None:
            rebuilt_kwargs["device"] = new_device

        return super().to(*positional, **rebuilt_kwargs)
    
    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------
    
    def save(self, path: str) -> None:
        """
        Serialise the full MPS (config + site tensors) to disk.
        """
        torch.save(
            {
                "config": {
                    "num_sites": self.num_sites,
                    "bond_dim": self.bond_dim,
                    "physical_dims": list(self.physical_dims),
                    "dtype": _REVERSE_DTYPE_MAP[self.dtype],
                },
                "tensors": [site_tensor.detach().cpu().clone() for site_tensor in self.site_tensors],
            },
            path,
        )
 
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "MPS":
        """
        Reconstruct an MPS previously saved with :meth:`save`.
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        config = checkpoint["config"]
        tensors: List[torch.Tensor] = checkpoint["tensors"]

        raw_dtype = config["dtype"]
        if isinstance(raw_dtype, str):
            dtype = _DTYPE_MAP[raw_dtype]
        else:
            dtype = raw_dtype
 
        if len(tensors) != config["num_sites"]:
            raise MPSShapeError(
                f"Checkpoint has {len(tensors)} tensors but config "
                f"declares num_sites={config['num_sites']}"
            )
 
        model = cls(
            num_sites=config["num_sites"],
            bond_dim=config["bond_dim"],
            physical_dims=config["physical_dims"],
            dtype=dtype,
            _skip_init=True,
        )

        for dst, src in zip(model.site_tensors, tensors):
            src_on_device = src.to(device=dst.device, dtype=dtype)
            dst.data = src_on_device.clone()
        return model
 
    
    # ----------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------
    
    @property
    def bond_dims(self) -> List[int]:
        """Internal bond dimensions, length ``num_sites - 1``."""
        return [self.site_tensors[k].shape[2] for k in range(self.num_sites - 1)]
    
    @property
    def full_bond_dims(self) -> List[int]:
        """All bond dimensions including boundaries D_0=D_N=1  (length N+1)."""
        return [1] + self.bond_dims + [1]

    @property
    def num_parameters(self) -> int:
        """Total real parameter count (complex tensors counted as 2 reals)."""
        num_real_parameters = sum(site_tensor.numel() for site_tensor in self.site_tensors)
        if self.dtype in (torch.complex64, torch.complex128):
            num_real_parameters *= 2
        return num_real_parameters
    
    @property
    def _numerical_floor(self) -> float:
        """Smallest denominator allowed before clamping, dtype-dependent."""
        if self.dtype in (torch.float32, torch.complex64):
            return 1e-15
        return 1e-30
    
    @property
    def _log_floor(self) -> float:
        """Smallest |psi|^2 used to clamp before log(), dtype-dependent.
        """
        if self.dtype in (torch.float32, torch.complex64):
            return 1e-30
        return 1e-300
    
    @property
    def is_homogeneous(self) -> bool:
        """True iff every site has the same physical dimension."""
        d0 = self.physical_dims[0]
        return all(d == d0 for d in self.physical_dims)
    
    # ----------------------------------------------------------------------
    # Amplitudes, norms, probabilities
    # ----------------------------------------------------------------------
    
    def log_amplitude_squared(self, configurations: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable log |Psi(v)|^2 with per-site rescaling.
        """
        if configurations.dtype != torch.long:
            configurations = configurations.long()
        self._validate_configurations(configurations)
        batch_size = configurations.shape[0]
 
        device = configurations.device
 
        tensor = self.site_tensors[0]
        values = configurations[:, 0]
        env = self._as_matrices(tensor[:, values, :]).squeeze(1)
 
        log_scale = torch.zeros(batch_size, dtype=torch.float64, device=device)
 
        env_abs_max = env.abs().amax(dim=1).clamp_min(self._numerical_floor)
        env = env / env_abs_max.unsqueeze(1).to(env.dtype)
        log_scale = log_scale + env_abs_max.double().log()
 
        for site in range(1, self.num_sites):
            tensor = self.site_tensors[site]
            values = configurations[:, site]
            selected_matrices = self._as_matrices(tensor[:, values, :])
            env = torch.bmm(env.unsqueeze(1), selected_matrices).squeeze(1)
 
            env_abs_max = env.abs().amax(dim=1).clamp_min(self._numerical_floor)
            env = env / env_abs_max.unsqueeze(1).to(env.dtype)
            log_scale = log_scale + env_abs_max.double().log()
 
        psi_rescaled = env.squeeze(1)
        if psi_rescaled.is_complex():
            abs2 = (psi_rescaled.real.square() + psi_rescaled.imag.square()).clamp_min(self._log_floor)
        else:
            abs2 = psi_rescaled.square().clamp_min(self._log_floor)
 
        log_abs2 = abs2.double().log() + 2.0 * log_scale
 
        real_dtype = (
            torch.float32 if self.dtype in (torch.float32, torch.complex64)
            else torch.float64
        )
        return log_abs2.to(real_dtype)
    
    def log_norm(self) -> torch.Tensor:
        """ 
        Computes log Z = log <psi|psi>.
        """

        env = torch.ones(1, 1, dtype=self.dtype, device=self.site_tensors[0].device)
        log_scale = torch.zeros((), dtype=torch.float64, device=env.device)

        for site in range(self.num_sites):
            tensor = self.site_tensors[site]
            matrices = self._as_matrices(tensor)

            contracted = torch.matmul(env, matrices)
            matrices_dagger = matrices.conj().transpose(1, 2)
            env = torch.matmul(matrices_dagger, contracted).sum(dim=0)

            scale = env.abs().max().clamp_min(self._numerical_floor)
            env   = env / scale
            log_scale = log_scale + scale.double().log()
        
        z_value = env.squeeze()
        real_dtype = (
            torch.float32 if self.dtype in (torch.float32, torch.complex64)
            else torch.float64
        )
        return (z_value.real.clamp_min(self._numerical_floor).double().log() + log_scale).to(real_dtype)

    def log_prob(self, configurations: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Computes log P(v) = log |Psi(v)|^2 - log Z
        """
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1 or None, got {batch_size}")
 
        log_z = self.log_norm()
        if batch_size is None or len(configurations) <= batch_size:
            return self.log_amplitude_squared(configurations) - log_z
 
        chunks: List[torch.Tensor] = []
        for start in range(0, len(configurations), batch_size):
            end = start + batch_size
            chunks.append(self.log_amplitude_squared(configurations[start:end]))
        return torch.cat(chunks) - log_z

    def nll(self, configurations: torch.Tensor, reduction: str = "mean", batch_size: Optional[int] = None,) -> torch.Tensor:
        """
        Negative log-likelihood:
            NLL(v) = -log P(v)

        reduction:
          - "none": returns shape (batch_size,)
          - "mean": scalar
          - "sum" : scalar
        """
        nll_values = -self.log_prob(configurations, batch_size=batch_size)

        if reduction == "none":
            return nll_values
        if reduction == "mean":
            return nll_values.mean()
        if reduction == "sum":
            return nll_values.sum()

        raise ValueError(f"Unsupported reduction: {reduction!r}. Use 'mean', 'sum', or 'none'.")
    
    @torch.no_grad()
    def anomaly_score(self, configurations, batch_size: Optional[int] = None):
        """
        Per-sample anomaly score, defined as the negative log-likelihood:
 
            score(v) = -log P(v)
 
        Higher scores correspond to less probable configurations under the
        learned model.  Used as the raw signal for thresholding in
        anomaly-detection pipelines.
        """
        return -self.log_prob(configurations, batch_size=batch_size)
    
    @torch.no_grad()
    def normalize_state(self) -> None:
        """Rescale every site so that ``<Psi|Psi> = 1``.

        The total norm is spread evenly across sites (each tensor is multiplied
        by ``exp(-log_z / (2N))``) rather than dumped on one of them, which keeps
        the individual tensors well-scaled. Called after each DMRG loop to stop
        the amplitude drifting.
        """
        log_z = self.log_norm()
        scale = torch.exp(-0.5 * log_z / self.num_sites)
        for site_parameter in self.site_tensors:
            site_parameter.data = site_parameter.data * scale

    # ----------------------------------------------------------------------
    # Canonicalization and tensor manipulation
    # ----------------------------------------------------------------------

    def _truncation_rank(
        self,
        singular_values: torch.Tensor,
        max_bond_dim: Optional[int],
        cutoff: float,
    ) -> int:
        """How many singular values survive a truncation.

        Two independent caps, whichever bites first: a relative ``cutoff`` (drop
        anything below ``cutoff * sigma_max``) and a hard ``max_bond_dim``. At
        least one value is always kept so a bond never collapses to zero.
        """
        rank_to_keep = len(singular_values)
        if cutoff > 0:
            singular_values_max = singular_values[0].abs().clamp_min(self._numerical_floor)
            rank_to_keep = max(int((singular_values / singular_values_max >= cutoff).sum().item()), 1)
        if max_bond_dim is not None:
            rank_to_keep = min(rank_to_keep, max_bond_dim)
        return rank_to_keep
    
    def _log_discarded_weight(
        self, singular_values: torch.Tensor, num_kept: int, where: str
    ) -> None:
        """Emit a warning if the discarded weight at a truncation exceeds the
        configured threshold.

        Short-circuits when the logger is disabled at WARNING level so that
        the (cheap but non-zero) ``square().sum()`` and the host sync are
        avoided in production runs where the warning is filtered out.
        """
        if not logger.isEnabledFor(logging.WARNING):
            return
        if num_kept >= len(singular_values):
            return
        kept = singular_values[:num_kept].square().sum()
        total = singular_values.square().sum().clamp_min(1e-30)
        discarded = (1.0 - kept / total).item()
        if discarded > self._discarded_weight_warn_threshold:
            logger.warning(
                "%s: discarded %.1f%% weight (rank %d -> %d)",
                where, 100.0 * discarded, len(singular_values), num_kept,
            )
    
    @torch.no_grad()
    def left_canonicalize(
        self,
        up_to: Optional[int] = None,
        truncate: bool = False,
        max_bond_dim: Optional[int] = None,
        cutoff: float = 0.0,
    ) -> Optional[List[torch.Tensor]]:
        """Sweep left-to-right putting sites into left-canonical form.

        Walks sites ``0 .. up_to-1``, factorising each one and pushing the
        remainder onto its right neighbour, so every processed tensor becomes an
        isometry (``A^dag A = I``). ``up_to`` defaults to the last bond, leaving
        the chain fully left-canonical with the norm collected on the final site.

        With ``truncate=False`` it uses QR (exact, no rank loss) and returns
        ``None``. With ``truncate=True`` it uses SVD, honours ``max_bond_dim`` /
        ``cutoff``, and returns the kept singular values per bond — which is
        exactly what :meth:`MPSExplainer.bond_entropies` consumes.
        """
        if up_to is None:
            up_to = self.num_sites - 1
        if not (0 <= up_to <= self.num_sites - 1):
            raise MPSShapeError(
                f"up_to={up_to} out of range [0, {self.num_sites - 1}]"
            )
        if truncate:
            self._validate_truncation(max_bond_dim, cutoff)

        if not truncate:
            for site in range(up_to):
                tensor = self.site_tensors[site].data
                bond_dim_left, physical_dim, bond_dim_right = tensor.shape
 
                Q, R = torch.linalg.qr(tensor.reshape(bond_dim_left * physical_dim, bond_dim_right))
                new_bond_dim = Q.shape[-1]
                self.site_tensors[site].data = Q.reshape(bond_dim_left, physical_dim, new_bond_dim)
 
                next_tensor = self.site_tensors[site + 1].data
                _, physical_dim_next, bond_dim_right_next = next_tensor.shape
 
                self.site_tensors[site + 1].data = (
                    R @ next_tensor.reshape(bond_dim_right, physical_dim_next * bond_dim_right_next)
                ).reshape(new_bond_dim, physical_dim_next, bond_dim_right_next)
            return None
 
        singular_values_per_bond: List[torch.Tensor] = []
        for site in range(up_to):
            tensor = self.site_tensors[site].data
            bond_dim_left, physical_dim, bond_dim_right = tensor.shape
 
            U, singular_values, Vh = torch.linalg.svd(tensor.reshape(bond_dim_left * physical_dim, bond_dim_right), full_matrices=False)
            rank_kept = self._truncation_rank(singular_values, max_bond_dim, cutoff)
            self._log_discarded_weight(
                singular_values, rank_kept, where=f"left_canonicalize@bond_{site}"
            )
            U, singular_values, Vh = U[:, :rank_kept], singular_values[:rank_kept], Vh[:rank_kept, :]
 
            singular_values_per_bond.append(singular_values.detach().clone())
            self.site_tensors[site].data = U.reshape(bond_dim_left, physical_dim, rank_kept)
 
            SV = singular_values.unsqueeze(1) * Vh
            next_tensor = self.site_tensors[site + 1].data
            _, physical_dim_next, bond_dim_right_next = next_tensor.shape
 
            self.site_tensors[site + 1].data = (
                SV @ next_tensor.reshape(bond_dim_right, physical_dim_next * bond_dim_right_next)
            ).reshape(rank_kept, physical_dim_next, bond_dim_right_next)
 
        return singular_values_per_bond

    @torch.no_grad()
    def right_canonicalize(
        self,
        from_site: Optional[int] = None,
        truncate: bool = False,
        max_bond_dim: Optional[int] = None,
        cutoff: float = 0.0,
    ) -> Optional[List[torch.Tensor]]:
        """Sweep right-to-left putting sites into right-canonical form.

        Mirror image of :meth:`left_canonicalize`: processes sites from the end
        down to ``from_site``, leaving each as a right isometry and carrying the
        remainder leftward. ``from_site`` defaults to 1, so the whole chain
        (except site 0, which ends up holding the norm) becomes right-canonical.

        QR when ``truncate=False`` (returns ``None``), SVD with truncation
        otherwise (returns the kept singular values per bond, ordered from the
        left). DMRG uses this to prime the chain before the first sweep.
        """
        if from_site is None:
            from_site = 1
        if not (1 <= from_site <= self.num_sites):
            raise MPSShapeError(
                f"from_site={from_site} out of range [1, {self.num_sites}]"
            )
        if truncate:
            self._validate_truncation(max_bond_dim, cutoff)
 
        if not truncate:
            for site in range(self.num_sites - 1, from_site - 1, -1):
                tensor = self.site_tensors[site].data
                bond_dim_left, physical_dim, bond_dim_right = tensor.shape
 
                Q, R = torch.linalg.qr(tensor.reshape(bond_dim_left, physical_dim * bond_dim_right).conj().T)
                new_bond_dim = Q.shape[1]
                self.site_tensors[site].data = Q.conj().T.reshape(new_bond_dim, physical_dim, bond_dim_right)
 
                previous_tensor = self.site_tensors[site - 1].data
                R_dagger = R.conj().T
                bond_dim_left_previous, physical_dim_previous, _ = previous_tensor.shape
 
                self.site_tensors[site - 1].data = (
                    previous_tensor.reshape(bond_dim_left_previous * physical_dim_previous, bond_dim_left) @ R_dagger
                ).reshape(bond_dim_left_previous, physical_dim_previous, new_bond_dim)
            return None
 
        singular_values_per_bond: List[torch.Tensor] = []
        for site in range(self.num_sites - 1, from_site - 1, -1):
            tensor = self.site_tensors[site].data
            bond_dim_left, physical_dim, bond_dim_right = tensor.shape
 
            U, singular_values, Vh = torch.linalg.svd(tensor.reshape(bond_dim_left, physical_dim * bond_dim_right), full_matrices=False)
            rank_kept = self._truncation_rank(singular_values, max_bond_dim, cutoff)
            self._log_discarded_weight(
                singular_values, rank_kept, where=f"right_canonicalize@bond_{site}"
            )
            U, singular_values, Vh = U[:, :rank_kept], singular_values[:rank_kept], Vh[:rank_kept, :]
 
            singular_values_per_bond.append(singular_values.detach().clone())
            self.site_tensors[site].data = Vh.reshape(rank_kept, physical_dim, bond_dim_right)
 
            US = U * singular_values.unsqueeze(0)
            previous_tensor = self.site_tensors[site - 1].data
            bond_dim_left_previous, physical_dim_previous, _ = previous_tensor.shape
            self.site_tensors[site - 1].data = (
                previous_tensor.reshape(bond_dim_left_previous * physical_dim_previous, bond_dim_left) @ US
            ).reshape(bond_dim_left_previous, physical_dim_previous, rank_kept)
 
        singular_values_per_bond.reverse()
        return singular_values_per_bond
    
    @torch.no_grad()
    def merge_sites(self, k: int) -> torch.Tensor:
        """Contract sites ``k`` and ``k+1`` into one rank-4 block.

        Returns ``theta`` of shape ``(D_{k-1}, d_k, d_{k+1}, D_{k+1})``. This is
        the two-site object DMRG updates in one shot; afterwards
        :meth:`split_and_truncate` factorises it back into two sites.
        """
        if not (0 <= k < self.num_sites - 1):
            raise MPSShapeError(f"Invalid bond index k={k}; expected 0 <= k < {self.num_sites - 1}")

        site_tensor_first  = self.site_tensors[k].data
        site_tensor_second = self.site_tensors[k + 1].data

        bond_dim_left, physical_dim_first, bond_dim_middle = site_tensor_first.shape
        _, physical_dim_second, bond_dim_right   = site_tensor_second.shape

        return (site_tensor_first.reshape(bond_dim_left * physical_dim_first, bond_dim_middle) @ site_tensor_second.reshape(bond_dim_middle, physical_dim_second * bond_dim_right)).reshape(bond_dim_left, physical_dim_first, physical_dim_second, bond_dim_right)
    
    @torch.no_grad()
    def split_and_truncate(
        self,
        k: int,
        merged_tensor: torch.Tensor,
        direction: str,
        max_bond_dim: int,
        cutoff: float = 0.0,
    ) -> torch.Tensor:
        """Split a merged two-site block back into two sites via SVD.

        Inverse of :meth:`merge_sites`. SVD across the ``(D_l*d_k | d_{k+1}*D_r)``
        cut, truncate to ``max_bond_dim`` / ``cutoff``, and absorb the singular
        values into one side depending on ``direction``: ``"right"`` leaves site
        ``k`` as a left-isometry and pushes the weight onto ``k+1`` (used on a
        left-to-right sweep), ``"left"`` does the opposite. The new bond
        dimension is whatever survived truncation, so the chain grows or shrinks
        adaptively. Returns the kept singular values.
        """
        if not (0 <= k < self.num_sites - 1):
            raise MPSShapeError(
                f"Invalid bond index k={k}; expected 0 <= k < {self.num_sites - 1}"
            )
        if direction not in ("right", "left"):
            raise ValueError(
                f"direction must be 'right' or 'left', got {direction!r}"
            )
        if merged_tensor.dim() != 4:
            raise MPSShapeError(
                f"theta must be rank-4 with shape (D_l, d, d, D_r), got shape {tuple(merged_tensor.shape)}"
            )
        bond_dim_left, physical_dim_first, physical_dim_second, bond_dim_right = merged_tensor.shape
        expected_physical_dim_first = self.physical_dims[k]
        expected_physical_dim_second = self.physical_dims[k + 1]
        if physical_dim_first != expected_physical_dim_first or physical_dim_second != expected_physical_dim_second:
            raise MPSShapeError(
                f"theta physical dims must be ({expected_physical_dim_first}, {expected_physical_dim_second}), got ({physical_dim_first}, {physical_dim_second})"
            )
        expected_bond_dim_left = self.site_tensors[k].shape[0]
        expected_bond_dim_right = self.site_tensors[k + 1].shape[2]
        if bond_dim_left != expected_bond_dim_left or bond_dim_right != expected_bond_dim_right:
            raise MPSShapeError(
                f"theta bond dims ({bond_dim_left}, {bond_dim_right}) do not match adjacent sites "
                f"({expected_bond_dim_left}, {expected_bond_dim_right})"
            )
        self._validate_truncation(max_bond_dim, cutoff)

        U, singular_values, Vh = torch.linalg.svd(merged_tensor.reshape(bond_dim_left * physical_dim_first, physical_dim_second * bond_dim_right), full_matrices=False)
        rank_kept = self._truncation_rank(singular_values, max_bond_dim, cutoff)
        self._log_discarded_weight(singular_values, rank_kept, where=f"split_and_truncate@bond_{k}")
        U, singular_values, Vh = U[:, :rank_kept], singular_values[:rank_kept], Vh[:rank_kept, :]

        if direction == "right":
            self.site_tensors[k].data = U.reshape(bond_dim_left, physical_dim_first, rank_kept)
            self.site_tensors[k + 1].data = (singular_values.unsqueeze(1) * Vh).reshape(rank_kept, physical_dim_second, bond_dim_right)
        else:
            self.site_tensors[k].data = (U * singular_values.unsqueeze(0)).reshape(bond_dim_left, physical_dim_first, rank_kept)
            self.site_tensors[k + 1].data = Vh.reshape(rank_kept, physical_dim_second, bond_dim_right)

        return singular_values.detach().clone()
    
    @torch.no_grad()
    def swap_adjacent(
        self,
        k: int,
        max_bond_dim: Optional[int] = None,
        cutoff: float = 0.0,
    ) -> None:
        """Swap the physical indices of sites ``k`` and ``k+1`` in place.

        Merges the two sites, transposes the two physical legs, and splits again,
        so the features at those positions trade places while the state stays
        exactly the same up to truncation. The building block for
        :meth:`permute_sites`; with ``max_bond_dim=None`` the split keeps full
        rank (lossless).
        """
        if not (0 <= k < self.num_sites - 1):
            raise MPSShapeError(f"Invalid bond index k={k}; expected 0 <= k < {self.num_sites - 1}")
        self._validate_truncation(max_bond_dim, cutoff)
 
        merged_tensor = self.merge_sites(k)
        merged_tensor_swapped = merged_tensor.permute(0, 2, 1, 3).contiguous()
 
        bond_dim_left, physical_dim_first, physical_dim_second, bond_dim_right = merged_tensor.shape
        if max_bond_dim is None:
            effective_cap = min(bond_dim_left * physical_dim_second, physical_dim_first * bond_dim_right)
        else:
            effective_cap = max_bond_dim

        self.physical_dims[k], self.physical_dims[k + 1] = (
            self.physical_dims[k + 1],
            self.physical_dims[k],
        )
            
        self.split_and_truncate(
            k, merged_tensor_swapped, direction="right",
            max_bond_dim=effective_cap,
            cutoff=cutoff,
        )
 
    @torch.no_grad()
    def permute_sites(
        self,
        permutation: List[int],
        max_bond_dim: Optional[int] = None,
        cutoff: float = 0.0,
    ) -> None:
        """Reorder the physical sites in place to match ``permutation``.

        ``permutation[k]`` is the current site that should end up at position
        ``k``. Implemented as a bubble sort of adjacent swaps, so the cost (and
        the entanglement the bond dimensions have to absorb) grows with how far
        sites travel. Handy for trying feature orderings that keep strongly
        correlated columns close together — see the mutual-information matrix in
        the explainability module.
        """
        if sorted(permutation) != list(range(self.num_sites)):
            raise ValueError(
                f"permutation must be a permutation of range({self.num_sites}), got {permutation}"
            )
        self._validate_truncation(max_bond_dim, cutoff)
 
        target = [0] * self.num_sites
        for k, src in enumerate(permutation):
            target[src] = k
 
        current = list(range(self.num_sites))
        for k in range(self.num_sites):
            wanted = target[k]
            j = current.index(wanted)
            while j > k:
                self.swap_adjacent(j - 1, max_bond_dim=max_bond_dim, cutoff=cutoff)
                current[j - 1], current[j] = current[j], current[j - 1]
                j -= 1