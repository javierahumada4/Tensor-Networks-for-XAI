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
        bond_dim = configurations.shape[0]

        Z = (theta.conj() * theta).real.sum()
        term1 = 2.0 * theta / Z.clamp_min(1e-30)

        v_k  = configurations[:, k]
        v_k1 = configurations[:, k + 1]

        theta_selected = theta[:, v_k, v_k1, :].permute(1, 0, 2)

        psi_v = torch.bmm(left_env.unsqueeze(1),
                          torch.bmm(theta_selected, right_env.unsqueeze(2))).reshape(bond_dim)
        
        psi_safe = psi_v.clone()
        psi_safe[psi_safe.abs() < 1e-30] = 1e-30

        term2 = torch.zeros_like(theta)
        for s in range(physical_dim):
            for t in range(physical_dim):
                mask = (v_k == s) & (v_k1 == t)
                if not mask.any():
                    continue
                left_env_masked = left_env[mask]
                right_env_masked = right_env[mask]
                psi_inv = 1.0 / psi_safe[mask]

                if theta.is_complex():
                    contribution = (left_env_masked.conj() * psi_inv.unsqueeze(1)).T @ right_env_masked
                else:
                    contribution = (left_env_masked * psi_inv.unsqueeze(1)).T @ right_env_masked

                term2[:, s, t, :] = contribution

        term2 = (2.0 / bond_dim) * term2

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
                theta = theta - lr * grad

            self.mps.split_and_truncate(
                k, theta, direction, cfg.max_bond_dim, cfg.svd_cutoff
            )

            if direction == "right" and k + 1 < num_sites - 1:
                left_envs[k + 1] = self._update_left_env(left_envs[k], k, configurations)
            elif direction == "left" and k > 0:
                right_envs[k] = self._update_right_env(right_envs[k + 1], k + 1, configurations)

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

        self.mps.left_canonicalize()

        lr = cfg.lr

        for _ in range(cfg.num_loops):
            idx = torch.randint(0, len(train_data), (cfg.batch_size,), device=device)
            batch = train_data[idx]
            left_env = self._build_left_envs(batch)
            right_env = self._build_right_envs(batch)
            self._sweep(batch, "left", lr, left_env, right_env)

            idx = torch.randint(0, len(train_data), (cfg.batch_size,), device=device)
            batch = train_data[idx]
            left_env = self._build_left_envs(batch)
            right_env = self._build_right_envs(batch)
            self._sweep(batch, "right", lr, left_env, right_env)

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
    )
    return DMRGTrainer(mps, config).train(train_data, val_data)