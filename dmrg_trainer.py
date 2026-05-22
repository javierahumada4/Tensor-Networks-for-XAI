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

_LOG_RECORD_VERSION: int = 1
_TRAINER_STATE_VERSION: int = 1

@dataclass
class DMRGConfig:
    """Hyperparameters for DMRG training.

    Parameters
    ----------
    num_descent_steps : int
        Number of gradient steps per bond per sweep.
    max_bond_dim : int
        Constant cap on the SVD truncation rank.  Used unless
        ``max_bond_dim_schedule`` overrides it.
    max_bond_dim_schedule : Optional[Callable[[int], int]]
        Optional callable that returns the bond-dim cap for a given loop
        index.  When provided, overrides ``max_bond_dim``.
    svd_cutoff : float
        Relative singular-value cutoff used inside the MPS SVD.
    lr : float
        Initial learning rate for the bond-wise gradient step.
    num_loops : int
        Number of full DMRG sweeps (right + left counts as one loop).
    batch_size : int
        Minibatch size used during sweeps.
    lr_shrink, lr_min, patience : float, float, int
        Patience-based LR scheduler.  When the monitored metric fails to
        improve by ``improvement_threshold`` for ``patience`` consecutive
        loops, ``lr`` is multiplied by ``lr_shrink``.  Training stops when
        ``lr`` falls below ``lr_min``.
    improvement_threshold : float
        Minimum decrease in the monitored metric counted as an
        improvement.
    adaptive_lr : bool
        If True, locally boost ``lr`` by ``plateau_factor`` when the
        relative gradient norm at a bond drops below ``plateau_threshold``.
        Disabled by default; see notes in `_sweep`.
    plateau_factor, plateau_threshold : float, float
        Parameters of the per-bond adaptive boost.
    batches_per_loop : int
        Number of minibatches per loop.  Zero means use the natural number
        of batches that covers the training set once.
    stochastic : bool
        If True, only one minibatch per loop is used (legacy switch).
    metric_for_stopping : str
        Either "train_nll" or "val_nll".  Used by the patience-based
        scheduler.  Falls back to "train_nll" if val_data is None.
    eval_max_samples : int
        Subsampling cap for NLL evaluation.  0 disables subsampling.
    seed : Optional[int]
        Seed for the RNG used in evaluation subsampling and minibatch
        permutation.  When set, training is reproducible (modulo CUDA's
        non-determinism in matmul/SVD; see torch.use_deterministic_algorithms).
    checkpoint_every : int
        If > 0, save the MPS and the trainer state every ``checkpoint_every``
        loops.  Trainer state captures lr, patience counter, best metric and
        the RNG state, so training can be resumed exactly with
        :meth:`DMRGTrainer.resume`.
    checkpoint_dir : Optional[str]
        Directory for checkpoint files.  Caller is responsible for
        validating this path; no sanitization is performed.
    log_path : Optional[str]
        Path to a JSONL log file.  Caller is responsible for validating
        this path.
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
    adaptive_lr: bool = True
    plateau_factor: float = 10.0
    plateau_threshold: float = 1e-4
    batches_per_loop: int = 0
    stochastic: bool = False
    metric_for_stopping: str = "train_nll"
    eval_max_samples: int = 2048
    seed: Optional[int] = None
    checkpoint_every: int = 0
    checkpoint_dir: Optional[str] = None
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
            tensor = self.mps.site_tensors[site].data
            values = configurations[:, site]
            selected_matrices = tensor[:, values, :].permute(1, 0, 2)

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
            tensor = self.mps.site_tensors[site].data
            values = configurations[:, site]
            selected_matrices = tensor[:, values, :].permute(1, 0, 2)

            environments[site - 1] = torch.bmm(selected_matrices, environments[site].unsqueeze(2)).squeeze(2)

        return environments

    def _update_left_environment(self, left_environment: torch.Tensor, site: int, configurations: torch.Tensor) -> torch.Tensor:
        tensor = self.mps.site_tensors[site].data
        selected_matrices = tensor[:, configurations[:, site], :].permute(1, 0, 2)
        return torch.bmm(left_environment.unsqueeze(1), selected_matrices).squeeze(1)

    def _update_right_environment(self, right_environment: torch.Tensor, site: int, configurations: torch.Tensor) -> torch.Tensor:
        tensor = self.mps.site_tensors[site].data
        selected_matrices = tensor[:, configurations[:, site], :].permute(1, 0, 2)
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
        z_floor = self._z_floor(self.mps.dtype)

        if cfg.adaptive_lr:
            plateau_factor_tensor = torch.tensor(
                cfg.plateau_factor, device=configurations.device,
                dtype=self.mps.dtype,
            )
            unit_tensor = torch.tensor(
                1.0, device=configurations.device, dtype=self.mps.dtype,
            )

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

                if cfg.adaptive_lr:
                    merged_tensor_norm = merged_tensor.norm().clamp_min(z_floor)
                    relative_gradient_norm = gradient_norm / merged_tensor_norm
                    
                    plateau_boost = torch.where(
                        relative_gradient_norm < cfg.plateau_threshold,
                        plateau_factor_tensor,
                        unit_tensor,
                    )
                    merged_tensor = merged_tensor - (lr * plateau_boost) * gradient
                else:
                    merged_tensor = merged_tensor - lr * gradient
                was_updated = True

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
        return {"max_gradient_norm": max_gradient_norm, "num_skipped_nan": num_skipped_nan}
    
    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_nll(self, data: torch.Tensor) -> float:
        max_samples_cap = self.config.eval_max_samples
        if max_samples_cap <= 0 or len(data) <= max_samples_cap:
            return self.mps.nll(data, batch_size=self.config.batch_size).item()
 
        sampled_indices = self._randperm_like(len(data), data.device)[:max_samples_cap]
        return self.mps.nll(data[sampled_indices], batch_size=self.config.batch_size).item()
    
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
    
    def _open_log(self, resuming: bool = False) -> None:
        if self.config.log_path is None:
            return
        Path(self.config.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.config.log_path, "a" if resuming else "w", encoding="utf-8")
 
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
        record = {"format_version": _LOG_RECORD_VERSION, **record}
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    # ------------------------------------------------------------------
    #  Checkpoints (full trainer state, not just MPS)
    # ------------------------------------------------------------------
 
    def _maybe_checkpoint(
        self,
        loop: int,
        lr: float,
        wait: int,
        best_metric: float,
    ) -> Optional[Dict[str, str]]:
        cfg = self.config
        if cfg.checkpoint_every <= 0 or cfg.checkpoint_dir is None:
            return None
        if (loop + 1) % cfg.checkpoint_every != 0:
            return None

        checkpoint_directory = Path(cfg.checkpoint_dir)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        mps_checkpoint_path = checkpoint_directory / f"mps_loop_{loop:04d}.pt"
        trainer_state_path = checkpoint_directory / f"trainer_loop_{loop:04d}.pt"

        self.mps.save(str(mps_checkpoint_path))

        payload = {
            "format_version": _TRAINER_STATE_VERSION,
            "loop": loop,
            "lr": lr,
            "wait": wait,
            "best_metric": best_metric,
            "config": dataclasses.asdict(self._serializable_config()),
            "generator_state": self._generator.get_state(),
            "generator_device": self._generator_device.type,
        }
        torch.save(payload, str(trainer_state_path))
        return {"mps_checkpoint_path": str(mps_checkpoint_path), "trainer_state_path": str(trainer_state_path)}
    
    def _serializable_config(self) -> DMRGConfig:
        """Return a copy of the config with non-picklable fields stripped.

        ``max_bond_dim_schedule`` may be a closure that does not pickle
        cleanly; we replace it with None in the serialized config and rely
        on the caller passing the same callable when resuming.
        """
        return dataclasses.replace(self.config, max_bond_dim_schedule=None)
    
    @classmethod
    def resume(
        cls,
        mps: nn.Module,
        trainer_state_path: str,
        max_bond_dim_schedule: Optional[Callable[[int], int]] = None,
    ) -> Tuple["DMRGTrainer", Dict[str, Any]]:
        """Reconstruct a trainer from a saved trainer-state file.

        The MPS must be loaded separately (via ``MPS.load``) and passed in.
        ``max_bond_dim_schedule`` cannot be serialized; pass the same
        callable used originally if you want to keep the schedule.
        """
        payload = torch.load(trainer_state_path, weights_only=True)
        if payload.get("format_version") != _TRAINER_STATE_VERSION:
            raise ValueError(
                f"Unknown trainer-state format_version "
                f"{payload.get('format_version')!r}; expected "
                f"{_TRAINER_STATE_VERSION}."
            )

        config_dict = dict(payload["config"])
        config_dict["max_bond_dim_schedule"] = max_bond_dim_schedule
        config = DMRGConfig(**config_dict)
        trainer = cls(mps, config)
        trainer._generator.set_state(payload["generator_state"])

        resume_state: Dict[str, Any] = {
            "loop_start": int(payload["loop"]) + 1,
            "lr": float(payload["lr"]),
            "wait": int(payload["wait"]),
            "best_metric": float(payload["best_metric"]),
        }
        return trainer, resume_state
    
    # ------------------------------------------------------------------
    #  Train loop
    # ------------------------------------------------------------------
    
    @torch.no_grad()
    def train(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        *,
        resume_state: Optional[Dict[str, Any]] = None,
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

        self.mps.right_canonicalize()

        if resume_state is not None:
            loop_start = resume_state["loop_start"]
            lr = resume_state["lr"]
            wait = resume_state["wait"]
            best_metric = resume_state["best_metric"]
        else:
            loop_start = 0
            lr = cfg.lr
            wait = 0
            best_metric = float("inf")

        history: List[Dict] = []

        if cfg.batches_per_loop > 0:
            num_batches = cfg.batches_per_loop
        else:
            num_batches = max(1, (len(train_data) + cfg.batch_size - 1) // cfg.batch_size)

        natural_batches_per_epoch = max(
            1, (len(train_data) + cfg.batch_size - 1) // cfg.batch_size
        )

        self._open_log()
        t_start = time.monotonic(resuming=(resume_state is not None))

        try:
            for loop in range(loop_start, cfg.num_loops):
                t_loop_start = time.monotonic()
                max_bond_dim = self._bond_dim_for_loop(loop)

                permutation = self._randperm_like(len(train_data), train_data.device)
                stochastic_max_gradient_norm = 0.0
                num_skipped_nan = 0

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

                    stochastic_max_gradient_norm = max(stochastic_max_gradient_norm, stats_right_sweep["max_gradient_norm"], stats_left_sweep["max_gradient_norm"])

                    num_skipped_nan += (
                        stats_right_sweep["num_skipped_nan"] + stats_left_sweep["num_skipped_nan"]
                    )

                    if cfg.stochastic:
                            break
                    
                if num_skipped_nan > 0:
                    logger.warning(
                        "loop %d: skipped %d non-finite gradient updates.",
                        loop, num_skipped_nan,
                    )
    
                train_nll = self._evaluate_nll(train_data)
    
                record: Dict[str, Any] = {
                    "loop": loop,
                    "train_nll": train_nll,
                    "lr": lr,
                    "bond_dims": list(self.mps.bond_dims),
                    "max_bond_dim_cap": max_bond_dim,
                    "max_gradient_norm": stochastic_max_gradient_norm,
                    "num_skipped_nan": num_skipped_nan,
                    "elapsed_s": time.monotonic() - t_loop_start,
                    "wallclock_s": time.monotonic() - t_start,
                }
                if val_data is not None:
                    record["val_nll"] = self._evaluate_nll(val_data)
    
                checkpoint_paths = self._maybe_checkpoint(loop, lr, wait, best_metric)
                if checkpoint_paths is not None:
                    record.update(checkpoint_paths)
    
                history.append(record)
                self._write_log(record)

                logger.info(
                    "loop %d/%d  train_nll=%.4f  lr=%.2e  max_gradient_norm=%.2e  bond_dims=%s",
                    loop, cfg.num_loops - 1, train_nll, lr,
                    stochastic_max_gradient_norm, self.mps.bond_dims,
                )

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