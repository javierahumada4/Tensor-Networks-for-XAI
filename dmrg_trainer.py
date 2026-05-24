from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

@dataclass
class DMRGConfig:
    """Hyperparameters for DMRG training.
    """

    num_descent_steps: int = 1
    max_bond_dim: int = 100
    max_bond_dim_schedule: Optional[Callable[[int], int]] = None
    svd_cutoff: float = 1e-8
    lr: float = 0.01
    num_loops: int = 20
    batch_size: int = 256
    lr_shrink: float = 0.5
    lr_min: float = 1e-6
    patience: int = 5
    improvement_threshold: float = 1e-4
    abort_after_dead_loops: int = 3
    batches_per_loop: int = 0
    metric_for_stopping: str = "train_nll"
    seed: Optional[int] = None
    log_path: Optional[str] = None


class DMRGTrainer:
    def __init__(self, mps: nn.Module, config: Optional[DMRGConfig] = None):
        self.mps = mps
        self.config = config or DMRGConfig()

        if self.config.metric_for_stopping not in ("train_nll", "val_nll"):
            raise ValueError(
                f"metric_for_stopping must be 'train_nll' or 'val_nll', "
                f"got {self.config.metric_for_stopping!r}"
            )
        if self.config.max_bond_dim < 1:
            raise ValueError(
                f"max_bond_dim must be >= 1, got {self.config.max_bond_dim}"
            )
        if self.config.abort_after_dead_loops < 0:
            raise ValueError(
                f"abort_after_dead_loops must be >= 0, got "
                f"{self.config.abort_after_dead_loops}"
            )
        
        try:
            device = next(self.mps.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self._generator_device = (
            device if device.type == "cuda" else torch.device("cpu")
        )
        self._generator = torch.Generator(device=self._generator_device.type)
        if self.config.seed is not None:
            self._generator.manual_seed(int(self.config.seed))

        self._log_file = None

    # ------------------------------------------------------------------
    #  Environment construction and updates
    # ------------------------------------------------------------------
    
    def _build_left_environments(self, configurations: torch.Tensor) -> List[torch.Tensor]:
        """
        left_envs[k] = contraction of sites 0..k-1 with data.
        Shape: (batch, D_{k-1}).
        """
        batch_size, num_sites = configurations.shape
        environments: List[torch.Tensor] = [None] * num_sites
        environments[0] = torch.ones(batch_size, 1, dtype=self.mps.dtype, device=configurations.device)

        for site in range(num_sites - 1):
            selected_matrices = self.mps.select_matrices(site, configurations[:, site])

            environments[site + 1] = torch.bmm(environments[site].unsqueeze(1), selected_matrices).squeeze(1)

        return environments
    
    def _build_right_environments(self, configurations: torch.Tensor) -> List[torch.Tensor]:
        """
        right_envs[k] = contraction of sites k+1..N-1 with data.
        Shape: (batch, D_k).
        """
        batch_size, num_sites = configurations.shape
        environments: List[torch.Tensor] = [None] * num_sites
        environments[num_sites - 1] = torch.ones(batch_size, 1, dtype=self.mps.dtype, device=configurations.device)

        for site in range(num_sites - 1, 0, -1):
            selected_matrices = self.mps.select_matrices(site, configurations[:, site])

            environments[site - 1] = torch.bmm(selected_matrices, environments[site].unsqueeze(2)).squeeze(2)

        return environments

    def _update_left_environment(self, left_environment: torch.Tensor, site: int, configurations: torch.Tensor) -> torch.Tensor:
        selected_matrices = self.mps.select_matrices(site, configurations[:, site])
        return torch.bmm(left_environment.unsqueeze(1), selected_matrices).squeeze(1)

    def _update_right_environment(self, right_environment: torch.Tensor, site: int, configurations: torch.Tensor) -> torch.Tensor:
        selected_matrices = self.mps.select_matrices(site, configurations[:, site])
        return torch.bmm(selected_matrices, right_environment.unsqueeze(2)).squeeze(2)
    
    # ------------------------------------------------------------------
    #  Numerical helpers
    # ------------------------------------------------------------------
    
    @staticmethod
    def _safe_psi(psi_v: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
        """
        """
        abs_psi = psi_v.abs()
        is_near_zero = abs_psi < eps
        
        if psi_v.is_complex():
            safe_abs = torch.where(
                abs_psi > 0, abs_psi, torch.ones_like(abs_psi)
            )
            phase = psi_v / safe_abs.to(psi_v.dtype)
            unit_phase = torch.ones_like(psi_v)
            phase = torch.where(abs_psi > 0, phase, unit_phase)
            near_zero_replacement = phase * eps
            return torch.where(is_near_zero, near_zero_replacement, psi_v)
        else:
            sign = torch.where(psi_v >= 0, torch.ones_like(psi_v), -torch.ones_like(psi_v))
            return torch.where(is_near_zero, sign * eps, psi_v)
        
    @staticmethod
    def _z_floor(dtype: torch.dtype) -> float:
        """Smallest allowed denominator for the partition function, by dtype."""
        if dtype in (torch.float32, torch.complex64):
            return 1e-15
        return 1e-30
    
    # ------------------------------------------------------------------
    #  Gradient
    # ------------------------------------------------------------------
    
    def _compute_gradient(
        self,
        k: int,
        merged_tensor: torch.Tensor,
        left_environment: torch.Tensor,
        right_environment: torch.Tensor,
        configurations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient of NLL w.r.t. merged tensor θ (Eq. B2).

        ∂L/∂θ = 2θ/Z − (2/|B|) Σ_v [outer(L_v, R_v)/Ψ(v)]
        """

        physical_dim_first = self.mps.physical_dims[k]
        physical_dim_second = self.mps.physical_dims[k + 1]
        batch_size = configurations.shape[0]

        z_floor = self._z_floor(merged_tensor.dtype)
        Z = (merged_tensor.conj() * merged_tensor).real.sum().to(torch.float64)
        Z_safe = Z.clamp_min(z_floor).to(merged_tensor.real.dtype if merged_tensor.is_complex() else merged_tensor.dtype)
        partition_function_term = 2.0 * merged_tensor / Z_safe

        configuration_values_first = configurations[:, k]
        configuration_values_second = configurations[:, k + 1]

        merged_tensor_selected = merged_tensor[:, configuration_values_first, configuration_values_second, :].permute(1, 0, 2)
        psi_value = torch.einsum("ba,bac,bc->b", left_environment, merged_tensor_selected, right_environment)
        
        psi_safe = self._safe_psi(psi_value, eps=z_floor)
 
        bond_dim_left, _, _, bond_dim_right = merged_tensor.shape
 
        if merged_tensor.is_complex():
            left_weighted = left_environment.conj() / psi_safe.conj().unsqueeze(1)
            right_weighted = right_environment.conj()
        else:
            left_weighted = left_environment / psi_safe.unsqueeze(1)
            right_weighted = right_environment
 
        contributions = left_weighted.unsqueeze(2) * right_weighted.unsqueeze(1)
 
        flattened_indices = configuration_values_first * physical_dim_second + configuration_values_second
        data_term_flattened = torch.zeros(physical_dim_first * physical_dim_second, bond_dim_left, bond_dim_right,
                                 dtype=merged_tensor.dtype, device=merged_tensor.device)
        data_term_flattened.index_add_(0, flattened_indices, contributions)
 
        data_term = (data_term_flattened
                 .view(physical_dim_first, physical_dim_second, bond_dim_left, bond_dim_right)
                 .permute(2, 0, 1, 3)
                 .contiguous())
        data_term = (2.0 / batch_size) * data_term
 
        return partition_function_term - data_term
    
    # ------------------------------------------------------------------
    #  Sweep
    # ------------------------------------------------------------------
    
    @torch.no_grad()
    def _sweep(
        self,
        configurations: torch.Tensor,
        direction: str,
        lr: float,
        left_environments: List[torch.Tensor],
        right_environments: List[torch.Tensor],
        max_bond_dim: int,
    ) -> Dict[str, Any]:
        num_sites = self.mps.num_sites
        cfg = self.config

        bond_indices = (
            range(num_sites - 2, -1, -1) if direction == "left"
            else range(0, num_sites - 1)
        )

        gradient_norms: List[torch.Tensor] = []
        num_skipped_nan = 0
        num_updates = 0

        for k in bond_indices:
            merged_tensor = self.mps.merge_sites(k)
            left_environment = left_environments[k]
            right_environment = right_environments[k + 1]

            was_updated = False
            for _ in range(cfg.num_descent_steps):
                gradient = self._compute_gradient(k, merged_tensor, left_environment, right_environment, configurations)

                if not torch.isfinite(gradient).all():
                    num_skipped_nan += 1
                    continue

                gradient_norm = gradient.norm()
                gradient_norms.append(gradient_norm)

                merged_tensor = merged_tensor - lr * gradient
                was_updated = True
                num_updates += 1

            if was_updated:
                self.mps.split_and_truncate(
                    k, merged_tensor, direction, max_bond_dim, cfg.svd_cutoff
                )

            if direction == "right" and k + 1 < num_sites - 1:
                left_environments[k + 1] = self._update_left_environment(left_environments[k], k, configurations)
            elif direction == "left" and k > 0:
                right_environments[k] = self._update_right_environment(right_environments[k + 1], k + 1, configurations)

        max_gradient_norm = (
            torch.stack(gradient_norms).max().item() if gradient_norms else 0.0
        )
        return {
            "max_gradient_norm": max_gradient_norm,
            "num_skipped_nan": num_skipped_nan,
            "num_updates": num_updates,
        }
    
    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    
    def _control_nll(self, data: torch.Tensor) -> float:
        """Exact NLL over the entire dataset."""
        return self.mps.nll(data, batch_size=self.config.batch_size).item()
    
    def _randperm_like(self, num_elements: int, device: torch.device) -> torch.Tensor:
        """Reproducible randperm honoring ``self._generator``."""
        if self._generator_device.type == device.type:
            return torch.randperm(num_elements, generator=self._generator, device=device)
        indices = torch.randperm(
            num_elements, generator=self._generator, device=self._generator_device
        )
        return indices.to(device)
    
    # ------------------------------------------------------------------
    #  Bond-dim schedule
    # ------------------------------------------------------------------
    
    def _bond_dim_for_loop(self, loop: int) -> int:
        schedule = self.config.max_bond_dim_schedule
        if schedule is None:
            return int(self.config.max_bond_dim)
        value = int(schedule(loop))
        if value < 1:
            raise ValueError(
                f"max_bond_dim_schedule returned {value} at loop={loop}; "
                "must be >= 1"
            )
        return value
    
    # ------------------------------------------------------------------
    #  Logging
    # ------------------------------------------------------------------
    
    def _open_log(self) -> None:
        if self.config.log_path is None:
            return
        Path(self.config.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.config.log_path, "w", encoding="utf-8")
 
    def _close_log(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.flush()
            finally:
                self._log_file.close()
                self._log_file = None
 
    def _write_log(self, record: Dict) -> None:
        if self._log_file is None:
            return
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    # ------------------------------------------------------------------
    #  Train loop
    # ------------------------------------------------------------------
    
    @torch.no_grad()
    def train(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        cfg = self.config
        train_data, val_data = self._prepare_data(train_data, val_data)

        metric = cfg.metric_for_stopping
        if metric == "val_nll" and val_data is None:
            logger.warning(
                "metric_for_stopping='val_nll' but val_data is None; "
                "falling back to 'train_nll'."
            )
            metric = "train_nll"

        self.mps.normalize_state()
        self.mps.right_canonicalize()

        loop_start = 0
        lr = cfg.lr
        wait = 0
        best_metric = float("inf")

        history: List[Dict] = []
        consecutive_dead_loops = 0

        if cfg.batches_per_loop > 0:
            num_batches = cfg.batches_per_loop
        else:
            num_batches = max(1, (len(train_data) + cfg.batch_size - 1) // cfg.batch_size)

        natural_batches_per_epoch = max(
            1, (len(train_data) + cfg.batch_size - 1) // cfg.batch_size
        )

        self._open_log()
        t_start = time.monotonic()

        try:
            for loop in range(loop_start, cfg.num_loops):
                t_loop_start = time.monotonic()
                max_bond_dim = self._bond_dim_for_loop(loop)

                permutation = self._randperm_like(len(train_data), train_data.device)
                num_skipped_nan = 0
                num_updates = 0

                for batch_index in range(num_batches):
                    if batch_index > 0 and batch_index % natural_batches_per_epoch == 0:
                        permutation = self._randperm_like(
                            len(train_data), train_data.device
                        )

                    batch_start = (batch_index % natural_batches_per_epoch) * cfg.batch_size
                    batch_indices = permutation[batch_start:batch_start + cfg.batch_size]
                    if len(batch_indices) < 2:
                        continue
                    batch = train_data[batch_indices]

                    left_environments = self._build_left_environments(batch)
                    right_environments = self._build_right_environments(batch)
                    stats_right_sweep = self._sweep(batch, "right", lr,left_environments, right_environments, max_bond_dim)

                    left_environments = self._build_left_environments(batch)
                    right_environments = self._build_right_environments(batch)
                    stats_left_sweep = self._sweep(batch, "left", lr, left_environments, right_environments, max_bond_dim)

                    num_skipped_nan += (
                        stats_right_sweep["num_skipped_nan"] + stats_left_sweep["num_skipped_nan"]
                    )
                    num_updates += (
                        stats_right_sweep["num_updates"] + stats_left_sweep["num_updates"]
                    )
                    
                if num_skipped_nan > 0:
                    logger.warning(
                        "loop %d: skipped %d non-finite gradient updates.",
                        loop, num_skipped_nan,
                    )

                self.mps.normalize_state()
                self.mps.right_canonicalize()
    
                train_nll = self._control_nll(train_data)
    
                record: Dict[str, Any] = {
                    "loop": loop,
                    "train_nll": train_nll,
                    "lr": lr,
                    "bond_dims": list(self.mps.bond_dims),
                    "max_bond_dim_cap": max_bond_dim,
                    "num_skipped_nan": num_skipped_nan,
                    "num_updates": num_updates,
                    "elapsed_s": time.monotonic() - t_loop_start,
                    "wallclock_s": time.monotonic() - t_start,
                }
                if val_data is not None:
                    record["val_nll"] = self._control_nll(val_data)
    
                history.append(record)
                self._write_log(record)

                if not math.isfinite(train_nll):
                    logger.error(
                        "loop %d: train_nll is non-finite (%s). "
                        "Aborting; the model has diverged.",
                        loop, train_nll,
                    )
                    break
    
                monitor_value = record.get(metric, train_nll)
                if monitor_value < best_metric - cfg.improvement_threshold:
                    best_metric = monitor_value
                    wait = 0
                else:
                    wait += 1
                    if wait >= cfg.patience:
                        lr *= cfg.lr_shrink
                        wait = 0
                        if lr < cfg.lr_min:
                            logger.info(
                                "lr %.2e fell below lr_min %.2e; "
                                "stopping early.", lr, cfg.lr_min,
                            )
                            break
                if num_updates == 0:
                    consecutive_dead_loops += 1
                    logger.error(
                        "loop %d: 0 gradient updates applied "
                        "(%d non-finite gradients skipped); the model did "
                        "not change. Consecutive dead loops: %d.",
                        loop, num_skipped_nan, consecutive_dead_loops,
                    )
                    if (cfg.abort_after_dead_loops > 0
                            and consecutive_dead_loops >= cfg.abort_after_dead_loops):
                        logger.error(
                            "Aborting: %d consecutive dead loops "
                            "(abort_after_dead_loops=%d). Every gradient was "
                            "non-finite; training cannot progress.",
                            consecutive_dead_loops, cfg.abort_after_dead_loops,
                        )
                        break
                else:
                    consecutive_dead_loops = 0
        finally:
            self._close_log()

        return history
    
            
    
    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_sites = self.mps.num_sites
        if train_data.dim() != 2:
            raise ValueError(
                f"train_data must be 2D (batch, num_sites), "
                f"got shape {tuple(train_data.shape)}"
            )
        if train_data.shape[1] != num_sites:
            raise ValueError(
                f"train_data has {train_data.shape[1]} sites, expected {num_sites}"
            )
        if len(train_data) < 2:
            raise ValueError(
                f"train_data has {len(train_data)} samples; DMRG needs at "
                "least 2 to form a usable minibatch."
            )

        device = next(self.mps.parameters()).device
        train_data = train_data.to(device)
        if val_data is not None:
            if val_data.dim() != 2:
                raise ValueError(
                    f"val_data must be 2D (batch, num_sites), "
                    f"got shape {tuple(val_data.shape)}"
                )
            if val_data.shape[1] != num_sites:
                raise ValueError(
                    f"val_data has {val_data.shape[1]} sites, expected {num_sites}"
                )
            val_data = val_data.to(device)
        if train_data.dtype != torch.long:
            train_data = train_data.long()
        if val_data is not None and val_data.dtype != torch.long:
            val_data = val_data.long()
        return train_data, val_data

# ----------------------------------------------------------------------
#  Functional entry point
# ----------------------------------------------------------------------

def dmrg_train(
    mps: nn.Module,
    train_data: torch.Tensor,
    val_data: Optional[torch.Tensor] = None,
    *,
    config: Optional[DMRGConfig] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Train an MPS Born Machine with DMRG two-site updates.

    Either pass a fully-built ``config`` or pass any subset of
    :class:`DMRGConfig` fields as keyword arguments.  ``kwargs`` override
    fields in ``config`` if both are provided.

    Example
    -------
        from mps import MPS
        from dmrg_trainer import dmrg_train

        model = MPS(num_sites=30, bond_dim=2, physical_dim=2)
        history = dmrg_train(model, train_data, max_bond_dim=60, num_loops=40)
    """
    if config is None:
        config = DMRGConfig(**kwargs)
    elif kwargs:
        config = dataclasses.replace(config, **kwargs)
    return DMRGTrainer(mps, config).train(train_data, val_data)