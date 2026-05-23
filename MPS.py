from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    """
    Matrix Product State with open boundary
    """

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

        self._cached_left: Optional[List[torch.Tensor]] = None
        self._cached_right: Optional[List[torch.Tensor]] = None
        self._cache_valid: bool = False

    def _randn(self, *shape) -> torch.Tensor:
        """
        Generates real or complex Gaussian tensors depending on dtype.
        Ensures E[|z|^2] = 1 for complex tensors.
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

    def _validate_site(self, site: int, name: str = "site") -> None:
        """Check that ``site`` is a valid site index."""
        if not (0 <= site < self.num_sites):
            raise MPSShapeError(f"{name}={site} out of range [0, {self.num_sites})")

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
        """
        Re-index a site tensor as a stack of matrices keyed by physical value.
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
    
    @staticmethod
    def _abs_squared(x: torch.Tensor) -> torch.Tensor:
        """
        ``|x|^2`` for real or complex tensors
        """
        if x.is_complex():
            return x.real.square() + x.imag.square()
        return x.square()
    
    @staticmethod
    def _check_valid_probabilities(
        probabilities: torch.Tensor, site: int, context: str = "sample"
    ) -> None:
        """Verify that ``probs`` is a valid (multinomial-feedable) distribution.

        ``torch.multinomial`` errors out cryptically on non-finite inputs or
        on distributions whose row-sum is zero; both happen in practice when
        the MPS has over-/underflowed or has been catastrophically
        truncated.  We prefer a clear ``MPSNumericalError`` pointing at the
        actual cause.

        ``probs`` may be 1D (single distribution) or 2D (batch of
        distributions, one per row).
        """
        if not torch.isfinite(probabilities).all():
            raise MPSNumericalError(
                f"Non-finite conditional probabilities at site {site} during "
                f"{context}. The MPS likely under/overflowed; call "
                f"normalize_state() first or check init_std."
            )
        row_sums = probabilities.sum(dim=-1) if probabilities.dim() > 1 else probabilities.sum()
        if (row_sums <= 0).any():
            raise MPSNumericalError(
                f"Degenerate (all-zero) probabilities at site {site} during "
                f"{context}. The MPS state has likely been over-truncated or "
                "lies in an annihilated subspace for the current conditioning."
            )
    
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

        if new_device is not None:
            self.invalidate_environment_cache()

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
    def anomaly_score(self, configurations: torch.Tensor) -> torch.Tensor:
        """
        Per-sample anomaly score, defined as the negative log-likelihood:
 
            score(v) = -log P(v)
 
        Higher scores correspond to less probable configurations under the
        learned model.  Used as the raw signal for thresholding in
        anomaly-detection pipelines.
        """
        return -self.log_prob(configurations)
    
    @torch.no_grad()
    def normalize_state(self) -> None:
        """
        Rescale the MPS so that <psi|psi> = 1.
        """
        log_z = self.log_norm()
        scale = torch.exp(-0.5 * log_z / self.num_sites)
        for site_parameter in self.site_tensors:
            site_parameter.data = site_parameter.data * scale
        self.invalidate_environment_cache()

    # ----------------------------------------------------------------------
    # Canonicalization and tensor manipulation
    # ----------------------------------------------------------------------

    def _truncation_rank(
        self,
        singular_values: torch.Tensor,
        max_bond_dim: Optional[int],
        cutoff: float,
    ) -> int:
        """
        Determine how many singular values to keep.
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
        """
        """
        if up_to is None:
            up_to = self.num_sites - 1
        if not (0 <= up_to <= self.num_sites - 1):
            raise MPSShapeError(
                f"up_to={up_to} out of range [0, {self.num_sites - 1}]"
            )
        if truncate:
            self._validate_truncation(max_bond_dim, cutoff)

        self.invalidate_environment_cache()

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
        """
        """
        if from_site is None:
            from_site = 1
        if not (1 <= from_site <= self.num_sites):
            raise MPSShapeError(
                f"from_site={from_site} out of range [1, {self.num_sites}]"
            )
        if truncate:
            self._validate_truncation(max_bond_dim, cutoff)

        self.invalidate_environment_cache()
 
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
        """
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
        """
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

        self.invalidate_environment_cache()

        return singular_values.detach().clone()
    
    @torch.no_grad()
    def swap_adjacent(
        self,
        k: int,
        max_bond_dim: Optional[int] = None,
        cutoff: float = 0.0,
    ) -> None:
        """
        Swap the physical indices of sites k and k+1 in place.
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
        """
        Permute the physical sites of the MPS in place.
        """
        if sorted(permutation) != list(range(self.num_sites)):
            raise ValueError(
                f"permutation must be a permutation of range({self.num_sites}), got {permutation}"
            )
        self._validate_truncation(max_bond_dim, cutoff)

        self.invalidate_environment_cache()
 
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
 
    
    # ----------------------------------------------------------------------
    # Transfer environments and RDM kernels
    # ----------------------------------------------------------------------
    
    @torch.no_grad()
    def _apply_transfer_left(self, left_environment: torch.Tensor, site_tensor: torch.Tensor) -> torch.Tensor:
        matrices = self._as_matrices(site_tensor)
        left_times_conjugate = torch.matmul(left_environment, matrices.conj())
        per_site = torch.matmul(matrices.transpose(1, 2), left_times_conjugate)
        return per_site.sum(dim = 0)
    
    @torch.no_grad()
    def _apply_transfer_right(self, right_environment: torch.Tensor, site_tensor: torch.Tensor) -> torch.Tensor:
        matrices = self._as_matrices(site_tensor)
        matrices_times_right = torch.matmul(matrices, right_environment)
        per_site = torch.matmul(matrices_times_right, matrices.conj().transpose(1, 2))
        return per_site.sum(dim = 0)
    
    @torch.no_grad()
    def transfer_environments_left(self) -> List[torch.Tensor]:
        """
        Build left transfer matrices for every bond
        """
        device = self.site_tensors[0].device
        envs: List[torch.Tensor] = [torch.ones(1, 1, dtype=self.dtype, device=device)]
        for k in range(self.num_sites):
            envs.append(self._apply_transfer_left(envs[k], self.site_tensors[k].data))
        return envs
    
    @torch.no_grad()
    def transfer_environments_right(self) -> List[torch.Tensor]:
        """
        Build right transfer matrices for every bond.
        """
        N = self.num_sites
        device = self.site_tensors[0].device
        envs: List[Optional[torch.Tensor]] = [None] * N
        envs[N - 1] = torch.ones(1, 1, dtype=self.dtype, device=device)
        for k in range(N - 1, 0, -1):
            envs[k - 1] = self._apply_transfer_right(envs[k], self.site_tensors[k].data)
        return envs
    
    @torch.no_grad()
    def precompute_environments(self) -> None:
        """Cache the full left/right transfer environments."""
        self._cached_left = self.transfer_environments_left()
        self._cached_right = self.transfer_environments_right()
        self._cache_valid = True

    @torch.no_grad()
    def invalidate_environment_cache(self) -> None:
        """Drop the cached transfer environments."""
        self._cached_left = None
        self._cached_right = None
        self._cache_valid = False

    def _cached_environments(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return cached envs if valid, otherwise build fresh ones (uncached)."""
        if self._cache_valid:
            return self._cached_left, self._cached_right
        return self.transfer_environments_left(), self.transfer_environments_right()
    
    @torch.no_grad()
    def _open_site_rdm(self, left_environment: torch.Tensor, site_tensor: torch.Tensor, right_environment: torch.Tensor) -> torch.Tensor:
        """
        Single-site RDM kernel (un-normalised).
        """
        physical_dim = site_tensor.shape[1]
        matrices = self._as_matrices(site_tensor)

        intermediate = torch.matmul(torch.matmul(left_environment.transpose(0, 1), matrices), right_environment)
        conjugate_flattened = matrices.conj().reshape(physical_dim, -1)
        intermediate_flattened = intermediate.reshape(physical_dim, -1)

        return torch.matmul(intermediate_flattened, conjugate_flattened.T)
    
    @torch.no_grad()
    def _open_two_sites_tensor(self, left_environment: torch.Tensor, site_tensor: torch.Tensor) -> torch.Tensor:
        """
        First step of two-site RDM: leave one physical index pair open.
        """
        matrices = self._as_matrices(site_tensor)
        conjugate_matrices = matrices.conj()
        left_times_matrices = torch.matmul(left_environment.transpose(0, 1), matrices)
        return torch.matmul(left_times_matrices.permute(0, 2, 1).unsqueeze(1), conjugate_matrices.unsqueeze(0))
    
    @torch.no_grad()
    def _propagate_open_two_site_tensor(self, open_two_site_tensor: torch.Tensor, site_tensor: torch.Tensor) -> torch.Tensor:
        """
        Propagate the open two-index tensor M through an intermediate site
        by tracing over its physical index (transfer matrix).
        """
        matrices = self._as_matrices(site_tensor)

        return torch.einsum(
            "pca,xycd,pdb->xyab", matrices, open_two_site_tensor, matrices.conj()
        )

    # ----------------------------------------------------------------------
    # Reduced density matrices: public API
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def single_site_rdm(self, site: int) -> torch.Tensor:
        """
        Reduced density matrix for a single site (feature).
 
            ρ_k = Tr_{≠k}( |Ψ⟩⟨Ψ| )
 
        Returns a (d, d) Hermitian matrix normalised to trace 1.
        The diagonal entries give P(v_k = s) for each physical value s.
        """
        self._validate_site(site)
        
        left, right = self._cached_environments()
 
        rdm = self._open_site_rdm(left[site], self.site_tensors[site].data, right[site])
 
        trace = rdm.diagonal().real.sum().clamp_min(self._numerical_floor)
        return rdm / trace
    
    @torch.no_grad()
    def all_single_site_rdms(self) -> List[torch.Tensor]:
        """
        Single-site RDMs for every site.
        """
        left, right = self._cached_environments()
        rdms: List[torch.Tensor] = []
        for k in range(self.num_sites):
            rdm = self._open_site_rdm(left[k], self.site_tensors[k].data, right[k])
            trace = rdm.diagonal().real.sum().clamp_min(self._numerical_floor)
            rdms.append(rdm / trace)
        return rdms
    
    @torch.no_grad()
    def two_site_rdm(self, site_i: int, site_j: int) -> torch.Tensor:
        """
        Reduced density matrix for two sites (features).
 
            ρ_{ij} = Tr_{≠i,j}( |Ψ⟩⟨Ψ| )
 
        Returns a (d, d, d, d) tensor with index order [s_i, s_j, t_i, t_j],
        normalised so that  Σ_{s_i, s_j} ρ[s_i, s_j, s_i, s_j] = 1.
        """
        self._validate_site(site_i, "site_i")
        self._validate_site(site_j, "site_j")
        if site_i >= site_j:
            raise MPSShapeError(
                f"Need site_i < site_j, got ({site_i}, {site_j})"
            )
 
        left, right = self._cached_environments()
 
        left_environment = left[site_i]
        right_environment = right[site_j]
        site_tensor_i = self.site_tensors[site_i].data
        site_tensor_j = self.site_tensors[site_j].data
 
        open_two_site_tensor = self._open_two_sites_tensor(left_environment, site_tensor_i)
 
        for m in range(site_i + 1, site_j):
            open_two_site_tensor = self._propagate_open_two_site_tensor(open_two_site_tensor, self.site_tensors[m].data)
 
        matrices_j = self._as_matrices(site_tensor_j)
        conjugate_j = matrices_j.conj()
        matrices_j_times_right = torch.matmul(matrices_j, right_environment)
 
        rdm = torch.einsum("xyab,sac,tbc->xsyt", open_two_site_tensor, matrices_j_times_right, conjugate_j)
 
        trace = torch.einsum("stst->", rdm).real.clamp_min(self._numerical_floor)
        return rdm / trace
    
    @torch.no_grad()
    def conditional_rdm(
        self,
        site_i: int,
        site_j: int,
        value_j: int,
    ) -> torch.Tensor:
        """
        RDM at site i conditioned on site j having a fixed value.
 
        Returns a (d, d) matrix.  Diagonal entries give P(v_i | v_j = value_j).
        """
        self._validate_site(site_i, "site_i")
        self._validate_site(site_j, "site_j")
        if site_i == site_j:
            raise MPSShapeError("site_i and site_j must differ")
        if not (0 <= value_j < self.physical_dims[site_j]):
            raise MPSShapeError(
                f"value_j={value_j} out of range [0, {self.physical_dims[site_j]}) for site {site_j}"
            )
 
        lower, higher = min(site_i, site_j), max(site_i, site_j)
 
        left, right = self._cached_environments()
 
        left_environment = left[lower]
        right_environment = right[higher]
        site_tensor_lower = self.site_tensors[lower].data
        site_tensor_higher = self.site_tensors[higher].data
 
        if site_i < site_j:
            open_two_site_tensor = self._open_two_sites_tensor(left_environment, site_tensor_lower)
 
            for m in range(lower + 1, higher):
                open_two_site_tensor = self._propagate_open_two_site_tensor(open_two_site_tensor, self.site_tensors[m].data)
 
            fixed_value_j = site_tensor_higher[:, value_j, :]
            right_conditioned = fixed_value_j @ right_environment @ fixed_value_j.conj().T
            rdm = (open_two_site_tensor * right_conditioned).sum(dim=(-2, -1))
        else:
            fixed_value_j = site_tensor_lower[:, value_j, :]
            left_conditioned = fixed_value_j.T @ left_environment @ fixed_value_j.conj()
 
            for m in range(lower + 1, higher):
                left_conditioned = self._apply_transfer_left(left_conditioned, self.site_tensors[m].data)
 
            rdm = self._open_site_rdm(left_conditioned, site_tensor_higher, right_environment)
 
        trace = rdm.diagonal().real.sum().clamp_min(self._numerical_floor)
        return rdm / trace
    
    # ----------------------------------------------------------------------
    # Marginals and entropies
    # ----------------------------------------------------------------------
    
    @torch.no_grad()
    def feature_probabilities(self, site: int) -> torch.Tensor:
        """
        Marginal probability distribution P(v_k) for a single site.
 
        Equivalent to the diagonal of the single-site RDM.
        Returns a real (d,) tensor that sums to 1.
        """
        rdm = self.single_site_rdm(site)
        return rdm.diagonal().real
    
    @torch.no_grad()
    def all_feature_probabilities(self) -> List[torch.Tensor]:
        """
        Marginal probabilities P(v_k = s) for every site k and value s.
 
        Returns a real (num_sites, physical_dim) tensor whose rows sum to 1.
        Faster than a Python loop over `feature_probabilities(k)`.
        """
        rdms = self.all_single_site_rdms()
        return [rdm.diagonal().real for rdm in rdms]
    
    @torch.no_grad()
    def site_entropies(self) -> torch.Tensor:
        """
        Single-site von Neumann entropy at every site:
 
            S(ρ_k) = −Tr(ρ_k log ρ_k)
 
        where ρ_k is the reduced density matrix of site k.  Returns a
        ``(num_sites,)`` real tensor.
        """
        rdms = self.all_single_site_rdms()
        out = torch.zeros(self.num_sites, dtype=torch.float64)
        for k, rdm in enumerate(rdms):
            eigenvalues = torch.linalg.eigvalsh(rdm.real).clamp_min(self._numerical_floor)
            out[k] = -(eigenvalues * eigenvalues.log()).sum().item()
        return out
    
    @torch.no_grad()
    def bond_entropies(
        self,
        max_bond_dim: Optional[int] = None,
        cutoff: float = 0.0,
        preserve_state: bool = True,
    ) -> List[float]:
        """
         Bipartite von Neumann entropy at every bond:
 
            S(k) = −Σ_i p_i ln p_i,    p_i = σ_i² / Σ σ_j²
 
        where σ_i are the singular values at bond k.  Returns ``num_sites - 1`` values.
        """
        if preserve_state:
            tensor_backup = [parameter.data.clone() for parameter in self.site_tensors]
            try:
                singular_values_per_bond  = self.left_canonicalize(
                    truncate=True, max_bond_dim=max_bond_dim, cutoff=cutoff
                )
            finally:
                for parameter, backed_up_data in zip(self.site_tensors, tensor_backup):
                    parameter.data = backed_up_data
        else:
            singular_values_per_bond = self.left_canonicalize(
                truncate=True, max_bond_dim=max_bond_dim, cutoff=cutoff
            )

        entropies: List[float] = []
        for singular_values in singular_values_per_bond:
            probabilities = singular_values.square()
            probabilities = probabilities / probabilities.sum().clamp_min(self._numerical_floor)
            entropy = -(probabilities * probabilities.clamp_min(self._numerical_floor).log()).sum()
            entropies.append(entropy.item())
        return entropies
    
    # ----------------------------------------------------------------------
    # Information theory
    # ----------------------------------------------------------------------
 
    @torch.no_grad()
    def mutual_information(self, site_i: int, site_j: int) -> float:
        """
        Mutual information between two sites:
 
            I(i; j) = S(ρ_i) + S(ρ_j) − S(ρ_{ij})
 
        Quantifies total (including non-linear) correlation between two
        features.  Used to build the MI heatmap for feature
        clustering and ordering optimisation.
        """
        self._validate_site(site_i, "site_i")
        self._validate_site(site_j, "site_j")
        if site_i == site_j:
            raise MPSShapeError("site_i and site_j must differ")

        lower_site, higher_site = min(site_i, site_j), max(site_i, site_j)
 
        rdm_i = self.single_site_rdm(lower_site)
        rdm_j = self.single_site_rdm(higher_site)
        eigenvalues_i = torch.linalg.eigvalsh(rdm_i.real).clamp_min(self._numerical_floor)
        eigenvalues_j = torch.linalg.eigvalsh(rdm_j.real).clamp_min(self._numerical_floor)
        entropy_i = -(eigenvalues_i * eigenvalues_i.log()).sum().item()
        entropy_j = -(eigenvalues_j * eigenvalues_j.log()).sum().item()
 
        rdm_ij = self.two_site_rdm(lower_site, higher_site)
        physical_dim_i = self.physical_dims[lower_site]
        physical_dim_j = self.physical_dims[higher_site]
        density_matrix = rdm_ij.reshape(physical_dim_i * physical_dim_j, physical_dim_i * physical_dim_j)
        eigenvalues = torch.linalg.eigvalsh(density_matrix.real)
        eigenvalues = eigenvalues.clamp_min(self._numerical_floor)
        entropy_ij = -(eigenvalues * eigenvalues.log()).sum().item()
 
        return entropy_i + entropy_j - entropy_ij
    
    @torch.no_grad()
    def mutual_information_matrix(self) -> torch.Tensor:
        """
        Full N×N mutual-information matrix in one pass.
        """
        N = self.num_sites
 
        left, right = self._cached_environments()
 
        single_site_entropies = torch.zeros(N, dtype=torch.float64)
        for k in range(N):
            rdm = self._open_site_rdm(left[k], self.site_tensors[k].data, right[k])
            trace = rdm.diagonal().real.sum().clamp_min(self._numerical_floor)
            rdm = rdm / trace
            eigenvalues = torch.linalg.eigvalsh(rdm.real).clamp_min(self._numerical_floor)
            single_site_entropies[k] = -(eigenvalues * eigenvalues.log()).sum().item()
 
        mutual_information_values = torch.zeros(N, N, dtype=torch.float64)
        for i in range(N):
            mutual_information_values[i, i] = single_site_entropies[i]
 
        for i in range(N):
            site_tensor_i = self.site_tensors[i].data
            open_two_site_tensor = self._open_two_sites_tensor(left[i], site_tensor_i)
 
            for j in range(i + 1, N):
                if j > i + 1:
                    open_two_site_tensor = self._propagate_open_two_site_tensor(open_two_site_tensor, self.site_tensors[j - 1].data)
 
                site_tensor_j = self.site_tensors[j].data
                right_environment = right[j]
                matrices_j = self._as_matrices(site_tensor_j)
                conjugate_j = matrices_j.conj()
                matrices_j_times_right = torch.matmul(matrices_j, right_environment)
 
                rdm = torch.einsum("xyab,sac,tbc->xsyt", open_two_site_tensor, matrices_j_times_right, conjugate_j)

                trace = torch.einsum("stst->", rdm).real.clamp_min(
                    self._numerical_floor
                )
                rdm = rdm / trace
 
                physical_dim_i = self.physical_dims[i]
                physical_dim_j = self.physical_dims[j]
                density_matrix = rdm.reshape(physical_dim_i * physical_dim_j, physical_dim_i * physical_dim_j)
                eigenvalues = torch.linalg.eigvalsh(density_matrix.real).clamp_min(self._numerical_floor)
                entropy_ij = -(eigenvalues * eigenvalues.log()).sum().item()
 
                mutual_information_ij = single_site_entropies[i].item() + single_site_entropies[j].item() - entropy_ij
                mutual_information_values[i, j] = mutual_information_ij
                mutual_information_values[j, i] = mutual_information_ij
 
        return mutual_information_values
    
    # ----------------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------------
    
    @torch.no_grad()
    def sample(self, num_samples: int = 1, preserve_state: bool = False) -> torch.Tensor:
        """
        Draw exact, independent samples from P(v) = |Ψ(v)|² / Z.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        
        if preserve_state:
            tensor_backup = [parameter.data.clone() for parameter in self.site_tensors]
            try:
                return self._sample_left_canonical(num_samples)
            finally:
                for parameter, backed_up_data in zip(self.site_tensors, tensor_backup):
                    parameter.data = backed_up_data
        return self._sample_left_canonical(num_samples)
    
    def _sample_left_canonical(self, num_samples: int) -> torch.Tensor:
        self.left_canonicalize()
 
        device = self.site_tensors[0].device
        N = self.num_sites
 
        samples = torch.zeros(num_samples, N, dtype=torch.long, device=device)
 
        site_tensor_last = self.site_tensors[N - 1].data
        matrices = self._as_matrices(site_tensor_last).squeeze(2)
 
        squared_norms = self._abs_squared(matrices)
        probabilities = squared_norms.sum(dim=1)
        probabilities = probabilities / probabilities.sum().clamp_min(self._numerical_floor)
        self._check_valid_probabilities(probabilities, site=N - 1, context="sample")
 
        chosen = torch.multinomial(
            probabilities.unsqueeze(0).expand(num_samples, -1), 1
        ).squeeze(1)
        samples[:, N - 1] = chosen
 
        x = matrices[chosen]
 
        for k in range(N - 2, -1, -1):
            site_tensor_k = self.site_tensors[k].data
            matrices = self._as_matrices(site_tensor_k)
 
            candidates = torch.matmul(matrices, x.T)
            candidates = candidates.permute(2, 0, 1)
 
            squared_amplitudes = self._abs_squared(candidates)
            conditional_probabilities = squared_amplitudes.sum(dim=2)
            conditional_probabilities = conditional_probabilities / conditional_probabilities.sum(dim=1, keepdim=True).clamp_min(self._numerical_floor)
            self._check_valid_probabilities(conditional_probabilities, site=k, context="sample")
 
            chosen = torch.multinomial(conditional_probabilities, 1).squeeze(1)
            samples[:, k] = chosen
 
            gather_indices = chosen.unsqueeze(1).unsqueeze(2).expand(
                num_samples, 1, candidates.shape[2]
            )
            x = candidates.gather(1, gather_indices).squeeze(1)
 
        return samples
    
    @torch.no_grad()
    def sample_conditional(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int = 1,
        preserve_state: bool = False,
    ) -> torch.Tensor:
        """
        Conditional sampling: generate completions for partially known
        configurations.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        if known.dim() != 1 or known.shape[0] != self.num_sites:
            raise MPSShapeError(
                f"known must be 1D with {self.num_sites} entries, "
                f"got shape {tuple(known.shape)}"
            )
        if mask.dim() != 1 or mask.shape[0] != self.num_sites:
            raise MPSShapeError(
                f"mask must be 1D with {self.num_sites} entries, "
                f"got shape {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}")
        
        if preserve_state:
            tensor_backup = [parameter.data.clone() for parameter in self.site_tensors]
            try:
                return self._sample_conditional_dispatch(known, mask, num_samples)
            finally:
                for parameter, backed_up_data in zip(self.site_tensors, tensor_backup):
                    parameter.data = backed_up_data
        return self._sample_conditional_dispatch(known, mask, num_samples)
 
    def _sample_conditional_dispatch(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        N = self.num_sites
        device = self.site_tensors[0].device

        known = known.to(device).long()
        mask = mask.to(device)

        if mask.any():
            fixed_positions_check = mask.nonzero(as_tuple=False).flatten()
            for pos in fixed_positions_check.tolist():
                physical_dim = self.physical_dims[pos]
                value = int(known[pos].item())
                if value < 0 or value >= physical_dim:
                    raise MPSShapeError(
                        f"known[{pos}]={value} out of range [0, {physical_dim}) "
                        f"for that site's physical dim"
                    )
 
        free_positions = (~mask).nonzero(as_tuple=False).flatten()
        fixed_positions = mask.nonzero(as_tuple=False).flatten()
 
        if fixed_positions.numel() == 0:
            return self.sample(num_samples)
        if free_positions.numel() == 0:
            return known.long().unsqueeze(0).expand(num_samples, N).clone()
 
        if fixed_positions.min().item() > free_positions.max().item():
            return self._sample_conditional_right_to_left(known, mask, num_samples)
        if fixed_positions.max().item() < free_positions.min().item():
            return self._sample_conditional_left_to_right(known, mask, num_samples)
 
        return self._sample_conditional_scattered(known, mask, num_samples)
 
    @torch.no_grad()
    def _sample_conditional_right_to_left(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Conditional sampling with fixed bits at the right end of the chain."""
        self.left_canonicalize()
 
        device = self.site_tensors[0].device
        N = self.num_sites
 
        samples = torch.zeros(num_samples, N, dtype=torch.long, device=device)
 
        site_tensor_last = self.site_tensors[N - 1].data
        matrices = self._as_matrices(site_tensor_last).squeeze(2)
 
        if mask[N - 1]:
            chosen = known[N - 1].expand(num_samples)
        else:
            squared_norms = self._abs_squared(matrices)
            probabilities = squared_norms.sum(dim=1)
            probabilities = probabilities / probabilities.sum().clamp_min(self._numerical_floor)
            self._check_valid_probabilities(probabilities, site=N - 1, context="sample_conditional_RL")
            chosen = torch.multinomial(
                probabilities.unsqueeze(0).expand(num_samples, -1), 1
            ).squeeze(1)
 
        samples[:, N - 1] = chosen
        x = matrices[chosen]
 
        for k in range(N - 2, -1, -1):
            site_tensor_k = self.site_tensors[k].data
            matrices = self._as_matrices(site_tensor_k)
 
            candidates = torch.matmul(matrices, x.T).permute(2, 0, 1)
 
            if mask[k]:
                chosen = known[k].expand(num_samples)
            else:
                squared_amplitudes = self._abs_squared(candidates)
                conditional_probabilities = squared_amplitudes.sum(dim=2)
                conditional_probabilities = conditional_probabilities / conditional_probabilities.sum(dim=1, keepdim=True).clamp_min(self._numerical_floor)
                self._check_valid_probabilities(conditional_probabilities, site=k, context="sample_conditional_RL")
                chosen = torch.multinomial(conditional_probabilities, 1).squeeze(1)
 
            samples[:, k] = chosen
            gather_indices = chosen.unsqueeze(1).unsqueeze(2).expand(num_samples, 1, candidates.shape[2])
            x = candidates.gather(1, gather_indices).squeeze(1)
 
        return samples
 
    @torch.no_grad()
    def _sample_conditional_left_to_right(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Conditional sampling with fixed bits at the left end of the chain.
        """
        self.right_canonicalize(from_site=1)
 
        device = self.site_tensors[0].device
        N = self.num_sites
 
        samples = torch.zeros(num_samples, N, dtype=torch.long, device=device)
 
        site_tensor_first = self.site_tensors[0].data
        matrices = self._as_matrices(site_tensor_first).squeeze(1)
 
        if mask[0]:
            chosen = known[0].expand(num_samples)
        else:
            squared_norms = self._abs_squared(matrices)
            probabilities = squared_norms.sum(dim=1)
            probabilities = probabilities / probabilities.sum().clamp_min(self._numerical_floor)
            self._check_valid_probabilities(probabilities, site=0, context="sample_conditional_LR")
            chosen = torch.multinomial(
                probabilities.unsqueeze(0).expand(num_samples, -1), 1
            ).squeeze(1)
 
        samples[:, 0] = chosen
        x = matrices[chosen]
 
        for k in range(1, N):
            site_tensor_k = self.site_tensors[k].data
            matrices = self._as_matrices(site_tensor_k)
 
            candidates = torch.einsum('sa,vab->svb', x, matrices)
 
            if mask[k]:
                chosen = known[k].expand(num_samples)
            else:
                squared_amplitudes = self._abs_squared(candidates)
                conditional_probabilities = squared_amplitudes.sum(dim=2)
                conditional_probabilities = conditional_probabilities / conditional_probabilities.sum(dim=1, keepdim=True).clamp_min(self._numerical_floor)
                self._check_valid_probabilities(conditional_probabilities, site=k, context="sample_conditional_LR")
                chosen = torch.multinomial(conditional_probabilities, 1).squeeze(1)
 
            samples[:, k] = chosen
            gather_indices = chosen.unsqueeze(1).unsqueeze(2).expand(num_samples, 1, candidates.shape[2])
            x = candidates.gather(1, gather_indices).squeeze(1)
 
        return samples
    
    @torch.no_grad()
    def _sample_conditional_scattered(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Conditional sampling for scattered masks via ladder contraction.
        """
        device = self.site_tensors[0].device
        N = self.num_sites
        is_complex = self.dtype in (torch.complex64, torch.complex128)

        right_masked: List[Optional[torch.Tensor]] = [None] * N
        right_masked[N - 1] = torch.ones(1, 1, dtype=self.dtype, device=device)
 
        for k in range(N - 1, 0, -1):
            site_tensor_k = self.site_tensors[k].data
            matrices = self._as_matrices(site_tensor_k)
            right_environment_next = right_masked[k]
 
            if mask[k]:
                fixed_value = int(known[k].item())
                site_tensor_at_value = matrices[fixed_value]
                right_masked[k - 1] = site_tensor_at_value @ right_environment_next @ site_tensor_at_value.conj().T
            else:
                matrices_times_right = torch.matmul(matrices, right_environment_next)
                right_masked[k - 1] = torch.matmul(
                    matrices_times_right, matrices.conj().transpose(1, 2)
                ).sum(dim=0)
 
        samples = torch.zeros(num_samples, N, dtype=torch.long, device=device)
 
        x = torch.ones(num_samples, 1, dtype=self.dtype, device=device)
 
        for k in range(N):
            site_tensor_k = self.site_tensors[k].data
            matrices = self._as_matrices(site_tensor_k)
            right_environment_next = right_masked[k]
 
            candidates = torch.einsum('sa,vab->svb', x, matrices)
 
            if mask[k]:
                fixed_value = int(known[k].item())
                chosen = torch.full(
                    (num_samples,), fixed_value, dtype=torch.long, device=device,
                )
            else:
                weighted_candidates = torch.einsum('svb,bc->svc', candidates, right_environment_next)
                weights = (weighted_candidates * candidates.conj()).sum(dim=2)
                if is_complex:
                    weights = weights.real
                weights = weights.clamp_min(self._numerical_floor)
                conditional_probabilities = weights / weights.sum(dim=1, keepdim=True).clamp_min(self._numerical_floor)
                self._check_valid_probabilities(conditional_probabilities, site=k, context="sample_conditional_scattered")
                chosen = torch.multinomial(conditional_probabilities, 1).squeeze(1)
 
            samples[:, k] = chosen
            gather_indices = chosen.unsqueeze(1).unsqueeze(2).expand(
                num_samples, 1, candidates.shape[2]
            )
            x = candidates.gather(1, gather_indices).squeeze(1)
 
        return samples



