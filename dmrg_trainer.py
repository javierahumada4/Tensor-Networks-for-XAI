from typing import Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class DMRGConfig:
    """Hyperparameters for DMRG training."""

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
        environments[num_sites-1] = torch.ones(batch_size, 1, dtype=self.mps.dtype, device=configurations.device)

        for site in range(num_sites-1, 0, -1):
            tensor = self.mps.site_tensors[site].data
            values = configurations[:, site]
            selected_matrices = tensor[:, values, :].permute(1, 0, 2)

            environments[site + 1] = torch.bmm(selected_matrices, environments[site].unsqueeze(2)).squeeze(2)

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
                    contribution = (right_env_masked * psi_inv.unsqueeze(1)).T @ right_env_masked

                term2[:, s, t, :] = contribution
                
        term2 = (2.0 / bond_dim) * term2

        return term1 - term2