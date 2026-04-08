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