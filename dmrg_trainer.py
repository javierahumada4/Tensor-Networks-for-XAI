from typing import Optional, List, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class DMRGConfig:
    """Hyperparameters for DMRG training."""
    num_descent_steps: int = 1
    safe_threshold: float = 1e6
    max_bond_dim: int = 100
    svd_cutoff: float = 1e-8
    lr: float = 0.01
    num_loops: int = 20
    batch_size: int = 256
    lr_shrink: float = 0.5
    lr_min: float = 1e-6
    patience: int = 5
    adaptive_lr: bool = True
    plateau_factor: float = 10.0
    plateau_threshold: float = 1e-4
    lr_cap: float = 1.0
    batches_per_loop: int = 0 


class DMRGTrainer:
    def __init__(self, mps: nn.Module, config: Optional[DMRGConfig] = None):
        self.mps = mps
        self.config = config or DMRGConfig()
    
    def _build_left_envs(self, configurations: torch.Tensor) -> List[torch.Tensor]:
        """
        left_envs[k] = contraction of sites 0..k-1 with data.
        Shape: (batch, D_{k-1}).
        """
        batch_size, num_sites = configurations.shape
        assert num_sites == self.mps.num_sites

        environments = [None] * num_sites
        environments[0] = torch.ones(batch_size, 1, dtype=self.mps.dtype, device=configurations.device)

        for site in range(num_sites-1):
            tensor = self.mps.site_tensors[site].data
            values = configurations[:, site]
            selected_matrices = tensor[:, values, :].permute(1, 0, 2)

            environments[site + 1] = torch.bmm(environments[site].unsqueeze(1), selected_matrices).squeeze(1)

        return environments
    
    def _build_right_envs(self, configurations: torch.Tensor) -> List[torch.Tensor]:
        """
        right_envs[k] = contraction of sites k+1..N-1 with data.
        Shape: (batch, D_k).
        """
        batch_size, num_sites = configurations.shape
        assert num_sites == self.mps.num_sites

        environments = [None] * num_sites
        environments[num_sites - 1] = torch.ones(batch_size, 1, dtype=self.mps.dtype, device=configurations.device)

        for site in range(num_sites-1, 0, -1):
            tensor = self.mps.site_tensors[site].data
            values = configurations[:, site]
            selected_matrices = tensor[:, values, :].permute(1, 0, 2)

            environments[site - 1] = torch.bmm(selected_matrices, environments[site].unsqueeze(2)).squeeze(2)

        return environments

    def _update_left_env(self, left_env: torch.Tensor, site: int, configs: torch.Tensor) -> torch.Tensor:
        tensor = self.mps.site_tensors[site].data
        selected_matrices = tensor[:, configs[:, site], :].permute(1, 0, 2)
        return torch.bmm(left_env.unsqueeze(1), selected_matrices).squeeze(1)

    def _update_right_env(self, right_env: torch.Tensor, site: int, configs: torch.Tensor) -> torch.Tensor:
        tensor = self.mps.site_tensors[site].data
        selected_matrices = tensor[:, configs[:, site], :].permute(1, 0, 2)
        return torch.bmm(selected_matrices, right_env.unsqueeze(2)).squeeze(2)
    
    @staticmethod
    def _safe_psi(psi_v: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
        """
        """
        abs_psi = psi_v.abs()
        small = abs_psi < eps
        if psi_v.is_complex():
            phase = torch.where(abs_psi > 0, psi_v / abs_psi.clamp_min(eps), torch.ones_like(psi_v))
            return torch.where(small, phase * eps, psi_v)
        else:
            sign = torch.where(psi_v >= 0, torch.ones_like(psi_v), -torch.ones_like(psi_v))
            return torch.where(small, sign * eps, psi_v)
    
    def _compute_gradient(
        self,
        k: int,
        theta: torch.Tensor,
        left_env: torch.Tensor,
        right_env: torch.Tensor,
        configurations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient of NLL w.r.t. merged tensor θ (Eq. B2).

        ∂L/∂θ = 2θ/Z − (2/|B|) Σ_v [outer(L_v, R_v)/Ψ(v)]
        """

        physical_dim = self.mps.physical_dim
        batch_size = configurations.shape[0]

        Z = (theta.conj() * theta).real.sum()
        term1 = 2.0 * theta / Z.clamp_min(1e-30)

        v_k  = configurations[:, k]
        v_k1 = configurations[:, k + 1]

        theta_selected = theta[:, v_k, v_k1, :].permute(1, 0, 2)

        psi_v = torch.bmm(left_env.unsqueeze(1),
                          torch.bmm(theta_selected, right_env.unsqueeze(2))).reshape(batch_size)
        
        psi_safe = self._safe_psi(psi_v)
 
        D_l, _, _, D_r = theta.shape
 
        if theta.is_complex():
            L_w = left_env.conj() / psi_safe.conj().unsqueeze(1)
            R_w = right_env.conj()
        else:
            L_w = left_env / psi_safe.unsqueeze(1)
            R_w = right_env
 
        contributions = L_w.unsqueeze(2) * R_w.unsqueeze(1)
 
        flat_idx = v_k * physical_dim + v_k1
        term2_flat = torch.zeros(physical_dim * physical_dim, D_l, D_r,
                                 dtype=theta.dtype, device=theta.device)
        term2_flat.index_add_(0, flat_idx, contributions)
 
        term2 = (term2_flat
                 .view(physical_dim, physical_dim, D_l, D_r)
                 .permute(2, 0, 1, 3)
                 .contiguous())
        term2 = (2.0 / batch_size) * term2
 
        return term1 - term2
    
    @torch.no_grad()
    def _sweep(
        self,
        configurations: torch.Tensor,
        direction: str,
        lr: float,
        left_envs: List[torch.Tensor],
        right_envs: List[torch.Tensor],
    ) -> None:
        num_sites = self.mps.num_sites
        cfg = self.config

        bonds = (
            range(num_sites - 2, -1, -1) if direction == "left"
            else range(0, num_sites - 1)
        )

        for k in bonds:
            theta = self.mps.merge_sites(k)
            left_env = left_envs[k]
            right_env = right_envs[k + 1]

            for _ in range(cfg.num_descent_steps):
                grad = self._compute_gradient(k, theta, left_env, right_env, configurations)
                grad_norm = grad.norm().item()

                if grad_norm > cfg.safe_threshold:
                    grad = grad * (cfg.safe_threshold / grad_norm)
                    grad_norm = cfg.safe_threshold

                bond_lr = lr
                if cfg.adaptive_lr:
                    theta_norm = theta.norm().item()
                    relative_grad = grad_norm / max(theta_norm, 1e-30)
                    if relative_grad < cfg.plateau_threshold:
                        bond_lr = lr * cfg.plateau_factor
                bond_lr = min(bond_lr, cfg.lr_cap)

                theta = theta - bond_lr * grad

            self.mps.split_and_truncate(
                k, theta, direction, cfg.max_bond_dim, cfg.svd_cutoff
            )

            if direction == "right" and k + 1 < num_sites - 1:
                left_envs[k + 1] = self._update_left_env(left_envs[k], k, configurations)
            elif direction == "left" and k > 0:
                right_envs[k] = self._update_right_env(right_envs[k + 1], k + 1, configurations)

    @torch.no_grad()
    def _evaluate_nll(self, data: torch.Tensor, max_samples: int = 2048) -> float:
        n = min(len(data), max_samples)
        if n == len(data):
            return self.mps.nll(data).item()
        idx = torch.randperm(len(data), device=data.device)[:n]
        return self.mps.nll(data[idx]).item()

    @torch.no_grad()
    def train(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
    ) -> List[Dict]:
        cfg = self.config
        num_sites = self.mps.num_sites
        assert train_data.shape[1] == num_sites

        device = next(self.mps.parameters()).device
        train_data = train_data.to(device)
        if val_data is not None:
            val_data = val_data.to(device)
        if train_data.dtype != torch.long:
            train_data = train_data.long()

        self.mps.left_canonicalize()
        self.mps.right_canonicalize()

        lr = cfg.lr
        best_nll = float('inf')
        wait = 0
        history: List[Dict] = []

        if cfg.batches_per_loop > 0:
            n_batches = cfg.batches_per_loop
        else:
            n_batches = max(1, (len(train_data) + cfg.batch_size - 1) // cfg.batch_size)

        for loop in range(cfg.num_loops):
            perm = torch.randperm(len(train_data), device=device)
            for b in range(n_batches):
                start = (b * cfg.batch_size) % len(train_data)
                idx = perm[start:start + cfg.batch_size]
                if len(idx) < 2:
                    continue
                batch = train_data[idx]

                left_env = self._build_left_envs(batch)
                right_env = self._build_right_envs(batch)
                self._sweep(batch, "right", lr, left_env, right_env)

                left_env = self._build_left_envs(batch)
                right_env = self._build_right_envs(batch)
                self._sweep(batch, "left", lr, left_env, right_env)
 
            train_nll = self._evaluate_nll(train_data)
 
            record: Dict = {
                "loop": loop,
                "train_nll": train_nll,
                "lr": lr,
                "bond_dims": self.mps.bond_dims,
            }
            if val_data is not None:
                record["val_nll"] = self._evaluate_nll(val_data)
 
            history.append(record)
 
            if train_nll < best_nll - 1e-4:
                best_nll = train_nll
                wait = 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    lr *= cfg.lr_shrink
                    wait = 0
                    if lr < cfg.lr_min:
                        break
 
        return history

def dmrg_train(
    mps: nn.Module,
    train_data: torch.Tensor,
    val_data: Optional[torch.Tensor] = None,
    *,
    max_bond_dim: int = 100,
    svd_cutoff: float = 1e-8,
    lr: float = 0.01,
    num_loops: int = 20,
    num_descent_steps: int = 1,
    batch_size: int = 256,
    lr_shrink: float = 0.5,
    lr_min: float = 1e-6,
    patience: int = 5,
    adaptive_lr: bool = True,
    plateau_factor: float = 10.0,
    plateau_threshold: float = 1e-4,
    lr_cap: float = 1.0,
    batches_per_loop: int = 0,
) -> List[Dict]:
    """
    Train an MPS Born Machine with DMRG two-site updates.

    Example:
        from mps import MPS
        from dmrg_trainer import dmrg_train

        model = MPS(num_sites=30, bond_dim=2, physical_dim=2)
        history = dmrg_train(model, train_data, max_bond_dim=60, num_loops=40)
    """
    config = DMRGConfig(
        num_descent_steps=num_descent_steps,
        max_bond_dim=max_bond_dim,
        svd_cutoff=svd_cutoff,
        lr=lr,
        num_loops=num_loops,
        batch_size=batch_size,
        lr_shrink=lr_shrink,
        lr_min=lr_min,
        patience=patience,
        adaptive_lr=adaptive_lr,
        plateau_factor=plateau_factor,
        plateau_threshold=plateau_threshold,
        lr_cap=lr_cap,
        batches_per_loop=batches_per_loop,
    )
    return DMRGTrainer(mps, config).train(train_data, val_data)