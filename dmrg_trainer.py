"""DMRG training for the MPS Born machine.

Fits an :class:`mps.MPS` to data by maximum likelihood using two-site DMRG: at
each bond the two neighbouring tensors are merged, nudged down the NLL gradient,
then split back with an SVD that also sets the new bond dimension. Sweeping
right then left covers every bond per loop.

What the trainer adds on top of the bare sweep is the bookkeeping you actually
need to get a clean run:

* an **adaptive bond cap** that only grows when truncation is genuinely losing
  weight, so the chain doesn't balloon early;
* **learning-rate annealing** on a plateau, with early stopping;
* a **best-model snapshot** that is restored at the end, so a noisy late loop
  can't undo a good fit;
* guards for the ways MPS training goes wrong in practice — non-finite
  gradients, dead loops, a diverging NLL — each with a clear log line.

Typical use is the :func:`dmrg_train` one-liner; :class:`DMRGTrainer` is there if
you want to drive the loop yourself. Everything runs under ``torch.no_grad`` —
gradients are derived in closed form, not by autograd.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

@dataclass
class DMRGConfig:
    """Everything that controls a DMRG run, with sensible defaults.

    Training schedule
        ``num_loops`` is the maximum number of full (right+left) sweeps.
        ``num_descent_steps`` is how many gradient steps to take on each merged
        two-site block before splitting it.

    Bond-dimension growth
        The cap starts at ``init_bond_cap`` and is allowed to climb towards
        ``max_bond_dim``. It only grows after the cap looks *binding* for
        ``grow_confirm_loops`` loops in a row — either a bond hit the cap or the
        truncation threw away more than ``discarded_weight_threshold`` of the
        weight. When it grows it multiplies by ``bond_growth_factor``.
        ``svd_cutoff`` is the relative singular-value floor for every split.

    Learning rate and stopping
        Start at ``lr``; after ``patience`` loops with no real improvement
        (better than ``improvement_threshold``) multiply by ``lr_shrink``. Drop
        below ``lr_min`` and training stops. Independently,
        ``early_stopping_patience`` (0 = off) stops if the monitored metric
        hasn't improved for that many loops.

    Minibatching
        ``batch_size`` rows per update. ``batches_per_loop`` overrides how many
        batches make up a loop; 0 means one pass over the data.

    Bookkeeping
        ``metric_for_stopping`` is ``"train_nll"`` or ``"val_nll"``.
        ``abort_after_dead_loops`` bails out if every gradient is non-finite for
        that many loops. ``seed`` makes the minibatch shuffling reproducible.
        ``log_path`` (if set) receives one JSON record per loop.
    """

    num_descent_steps: int = 1
    max_bond_dim: int = 100
    init_bond_cap: int = 4
    bond_growth_factor: float = 2.0
    discarded_weight_threshold: float = 1e-4
    grow_confirm_loops: int = 4
    svd_cutoff: float = 1e-8
    lr: float = 0.01
    num_loops: int = 20
    batch_size: int = 256
    lr_shrink: float = 0.5
    lr_min: float = 1e-6
    patience: int = 5
    improvement_threshold: float = 1e-4
    early_stopping_patience: int = 0
    abort_after_dead_loops: int = 3
    batches_per_loop: int = 0
    metric_for_stopping: str = "train_nll"
    seed: Optional[int] = None
    log_path: Optional[str] = None


class DMRGTrainer:
    """Drives the DMRG optimisation of one :class:`mps.MPS`.

    Holds the model, the :class:`DMRGConfig`, a seeded RNG for reproducible
    shuffling and the open log file. The interesting entry point is
    :meth:`train`; most other methods are the pieces it calls (environment
    construction, the closed-form gradient, a single sweep, the snapshot logic).
    The constructor validates the config up front so bad hyperparameters fail
    immediately rather than mid-run.
    """

    def __init__(self, mps: nn.Module, config: Optional[DMRGConfig] = None):
        """Bind the model and config, validating every hyperparameter up front."""
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
        if self.config.init_bond_cap < 1:
            raise ValueError(
                f"init_bond_cap must be >= 1, got {self.config.init_bond_cap}"
            )
        if self.config.bond_growth_factor <= 1.0:
            raise ValueError(
                "bond_growth_factor must be > 1.0 (set init_bond_cap == "
                "max_bond_dim to disable growth); got "
                f"{self.config.bond_growth_factor}"
            )
        if self.config.discarded_weight_threshold < 0.0:
            raise ValueError(
                "discarded_weight_threshold must be >= 0, got "
                f"{self.config.discarded_weight_threshold}"
            )
        if self.config.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be >= 0, got "
                f"{self.config.early_stopping_patience}"
            )
        if self.config.abort_after_dead_loops < 0:
            raise ValueError(
                f"abort_after_dead_loops must be >= 0, got "
                f"{self.config.abort_after_dead_loops}"
            )
        if self.config.grow_confirm_loops < 1:
            raise ValueError(
                f"grow_confirm_loops must be >= 1, got "
                f"{self.config.grow_confirm_loops}"
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
        """Extend a left environment by one site instead of rebuilding it.

        During a sweep the active bond moves one step at a time, so the
        already-contracted left part only needs the new site folded in — far
        cheaper than calling :meth:`_build_left_environments` again.
        """
        selected_matrices = self.mps.select_matrices(site, configurations[:, site])
        return torch.bmm(left_environment.unsqueeze(1), selected_matrices).squeeze(1)

    def _update_right_environment(self, right_environment: torch.Tensor, site: int, configurations: torch.Tensor) -> torch.Tensor:
        """Extend a right environment by one site (mirror of the left version)."""
        selected_matrices = self.mps.select_matrices(site, configurations[:, site])
        return torch.bmm(selected_matrices, right_environment.unsqueeze(2)).squeeze(2)
    
    # ------------------------------------------------------------------
    #  Numerical helpers
    # ------------------------------------------------------------------
    
    @staticmethod
    def _safe_psi(psi_v: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
        """Nudge amplitudes away from zero without changing their phase/sign.

        The gradient divides by ``Psi(v)``, so a sample the model currently
        assigns near-zero amplitude would blow it up. This floors ``|Psi|`` at
        ``eps`` while preserving sign (real) or phase (complex), keeping the
        division finite instead of producing inf/NaN.
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
        """One pass over every bond in a given direction.

        For each bond: merge the two sites, take ``num_descent_steps`` gradient
        steps on the merged block, split it back with truncation, then slide the
        environment one site over so the next bond is ready. ``direction`` is
        ``"right"`` (bonds ``0 -> N-2``) or ``"left"`` (the reverse), which also
        decides which side the singular values land on at the split.

        Non-finite gradients are skipped rather than applied. Returns a small
        stats dict (max gradient norm, max discarded weight, counts of skipped
        and applied updates) that the main loop aggregates and logs.
        """
        num_sites = self.mps.num_sites
        cfg = self.config

        bond_indices = (
            range(num_sites - 2, -1, -1) if direction == "left"
            else range(0, num_sites - 1)
        )

        gradient_norms: List[torch.Tensor] = []
        discarded_weights: List[torch.Tensor] = []
        num_skipped_nan = 0
        num_updates = 0
        z_floor = self._z_floor(self.mps.dtype)

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
                kept_singular_values = self.mps.split_and_truncate(
                    k, merged_tensor, direction, max_bond_dim, cfg.svd_cutoff
                )
                total_weight = merged_tensor.norm().pow(2)
                kept_weight = kept_singular_values.square().sum()
                discarded = (1.0 - kept_weight / total_weight.clamp_min(z_floor))
                discarded_weights.append(discarded.clamp_min(0.0))

            if direction == "right" and k + 1 < num_sites - 1:
                left_environments[k + 1] = self._update_left_environment(left_environments[k], k, configurations)
            elif direction == "left" and k > 0:
                right_environments[k] = self._update_right_environment(right_environments[k + 1], k + 1, configurations)

        max_gradient_norm = (
            torch.stack(gradient_norms).max().item() if gradient_norms else 0.0
        )
        max_discarded_weight = (
            torch.stack(discarded_weights).max().item() if discarded_weights else 0.0
        )
        return {
            "max_gradient_norm": max_gradient_norm,
            "max_discarded_weight": max_discarded_weight,
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
    #  Best-model snapshot
    # ------------------------------------------------------------------

    def _snapshot_mps(self) -> List[torch.Tensor]:
        """Detached CPU clone of every site tensor.
        """
        return [t.detach().cpu().clone() for t in self.mps.site_tensors]

    def _restore_mps(self, snapshot: List[torch.Tensor]) -> None:
        """Overwrite the live MPS with a snapshot taken by ``_snapshot_mps``.
        """
        for parameter, saved in zip(self.mps.site_tensors, snapshot):
            parameter.data = saved.to(
                device=parameter.device, dtype=parameter.dtype
            ).clone()

    # ------------------------------------------------------------------
    #  Dynamic bond-dim cap
    # ------------------------------------------------------------------
    
    def _cap_is_binding(self, cap: int, discarded_weight: float) -> Optional[str]:
        """Whether the truncation cap limited the model this loop.

        Returns a short human-readable reason, or None if the cap is not
        binding.  The cap is binding when either a bond actually reached
        the cap (the SVD wanted more rank) or the truncation discarded a
        non-negligible amount of weight.
        """
        if self.mps.bond_dims and max(self.mps.bond_dims) >= cap:
            return "a bond reached the cap"
        if discarded_weight > self.config.discarded_weight_threshold:
            return f"discarded weight {discarded_weight:.2e} over threshold"
        return None
    
    # ------------------------------------------------------------------
    #  Logging
    # ------------------------------------------------------------------
    
    def _open_log(self) -> None:
        """Open the JSONL log file (creating parent dirs) if a path was given."""
        if self.config.log_path is None:
            return
        Path(self.config.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.config.log_path, "w", encoding="utf-8")
 
    def _close_log(self) -> None:
        """Flush and close the log file if one is open."""
        if self._log_file is not None:
            try:
                self._log_file.flush()
            finally:
                self._log_file.close()
                self._log_file = None
 
    def _write_log(self, record: Dict) -> None:
        """Append one loop record as a JSON line and flush (so a crash keeps it)."""
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
        """Run the full optimisation and return the per-loop history.

        Primes the chain (normalise + right-canonicalise), then loops:
        right sweep, left sweep, renormalise, measure NLL, update the LR /
        bond-cap / early-stopping state, and snapshot the model whenever the
        monitored metric improves. The best snapshot is restored before
        returning, so the model you get back is the best one seen, not
        necessarily the last.

        ``val_data`` is optional; if absent and ``metric_for_stopping`` was
        ``"val_nll"`` it quietly falls back to ``"train_nll"``. The returned list
        has one dict per loop (NLL, lr, bond dims, timings, diagnostics) and is
        also streamed to ``config.log_path`` if set.
        """
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
        last_loop = loop_start - 1
        lr = cfg.lr
        wait = 0
        best_metric = float("inf")
        best_loop = -1
        best_snapshot: Optional[List[torch.Tensor]] = None
        loops_since_best = 0
        bond_cap = min(cfg.init_bond_cap, cfg.max_bond_dim)
        binding_streak = 0

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
                last_loop = loop
                t_loop_start = time.monotonic()
                max_bond_dim = bond_cap

                permutation = self._randperm_like(len(train_data), train_data.device)
                loop_max_gradient_norm = 0.0
                loop_max_discarded_weight = 0.0
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

                    loop_max_gradient_norm = max(loop_max_gradient_norm, stats_right_sweep["max_gradient_norm"], stats_left_sweep["max_gradient_norm"])
                    loop_max_discarded_weight = max(loop_max_discarded_weight, stats_right_sweep["max_discarded_weight"], stats_left_sweep["max_discarded_weight"])

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
                    "max_gradient_norm": loop_max_gradient_norm,
                    "max_discarded_weight": loop_max_discarded_weight,
                    "num_skipped_nan": num_skipped_nan,
                    "num_updates": num_updates,
                    "elapsed_s": time.monotonic() - t_loop_start,
                    "wallclock_s": time.monotonic() - t_start,
                }
                if val_data is not None:
                    record["val_nll"] = self._control_nll(val_data)
    
                history.append(record)
                self._write_log(record)

                monitored = record.get(metric, train_nll)
                improved = monitored < best_metric - cfg.improvement_threshold
                best_display = monitored if improved else best_metric
                wait_display = 0 if improved else wait + 1

                log_parts = [
                    f"loop {loop}/{cfg.num_loops - 1}",
                    f"train_nll={train_nll:.4f}",
                ]
                if "val_nll" in record:
                    val_nll = record["val_nll"]
                    log_parts.append(f"val_nll={val_nll:.4f}")
                    log_parts.append(f"gap={val_nll - train_nll:+.4f}")
                best_str = (
                    f"{best_display:.4f}" if math.isfinite(best_display) else "--"
                )
                log_parts.append(f"best_{metric}={best_str}")
                log_parts.append(f"wait={wait_display}/{cfg.patience}")
                log_parts.append(f"lr={lr:.2e}")
                log_parts.append(f"disc_w={loop_max_discarded_weight:.2e}")
                log_parts.append(f"|grad|={loop_max_gradient_norm:.2e}")
                log_parts.append(f"cap={bond_cap}/{cfg.max_bond_dim}")
                log_parts.append(f"bond_dims={list(self.mps.bond_dims)}")
                logger.info("  ".join(log_parts))

                if not math.isfinite(train_nll):
                    logger.error(
                        "loop %d: train_nll is non-finite (%s). "
                        "Aborting; the model has diverged.",
                        loop, train_nll,
                    )
                    break
    
                monitor_value = record.get(metric, train_nll)
                if improved:
                    best_metric = monitored
                    best_loop = loop
                    best_snapshot = self._snapshot_mps()
                    loops_since_best = 0
                    wait = 0
                else:
                    loops_since_best += 1
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
                    if (cfg.early_stopping_patience > 0
                            and loops_since_best >= cfg.early_stopping_patience):
                        logger.info(
                            "early stopping: %s has not improved for %d loops "
                            "(best=%.4f at loop %d).",
                            metric, loops_since_best, best_metric, best_loop,
                        )
                        break

                if bond_cap < cfg.max_bond_dim:
                    cap_reason = self._cap_is_binding(
                        bond_cap, loop_max_discarded_weight
                    )
                    binding_streak = (
                        binding_streak + 1 if cap_reason is not None else 0
                    )
                    if binding_streak >= cfg.grow_confirm_loops:
                        new_cap = min(
                            cfg.max_bond_dim,
                            math.ceil(bond_cap * cfg.bond_growth_factor),
                        )
                        if new_cap > bond_cap:
                            logger.info(
                                "loop %d: bond cap %d -> %d (%s, "
                                "binding %d loop(s)).",
                                loop, bond_cap, new_cap, cap_reason,
                                binding_streak,
                            )
                            bond_cap = new_cap
                            binding_streak = 0
                else:
                    binding_streak = 0

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
                
            if best_snapshot is not None and best_loop != last_loop:
                logger.info(
                    "restoring best model: loop %d (%s=%.4f), "
                    "discarding %d later loop(s).",
                    best_loop, metric, best_metric, last_loop - best_loop,
                )
                self._restore_mps(best_snapshot)
            elif best_snapshot is None:
                logger.warning(
                    "no loop improved on the initial metric; "
                    "keeping the last model."
                )
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
        """Validate shapes/dtype and move the data onto the model's device.

        Checks both tensors are ``(batch, num_sites)`` with the right number of
        sites, insists on at least two training rows (a minibatch needs them),
        and casts to ``long`` since configurations index into the site tensors.
        """
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