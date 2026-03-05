import math
from typing import Optional, List

import torch
import torch.nn as nn

class MPS(nn.Module):
    """
    Matrix Product State with open boundary
    """

    def __init__(
            self,
            num_sites: int,
            bond_dim: int,
            physical_dim: int = 2,
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device] = None,
            init_std: Optional[float] = None,
    ):
        super().__init__()

        assert num_sites >= 2, "num_sites must be >= 2"
        assert bond_dim >= 1, "bond_dim must be >= 1"
        assert physical_dim >= 2, "physical_dim must be >= 2"

        assert dtype in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128
        ), f"Unsupported dtype: {dtype}"

        self.num_sites = num_sites
        self.bond_dim = bond_dim
        self.physical_dim = physical_dim
        self.dtype = dtype
        self.device = device

        self.site_tensors = self.normal_init(init_std)

    def _randn(self, *shape):
        """
        Generates real or complex Gaussian tensors depending on dtype.
        Ensures E[|z|^2] = 1 for complex tensors.
        """
        if self.dtype in (torch.complex64, torch.complex128):
            base_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
            real_part = torch.randn(*shape, dtype=base_dtype, device=self.device)
            imag_part = torch.randn(*shape, dtype=base_dtype, device=self.device)
            complex_tensor = (real_part + 1j * imag_part) / math.sqrt(2)
            return complex_tensor.to(self.dtype)
        else:
            return torch.randn(*shape, dtype=self.dtype, device=self.device)

    def normal_init(self, init_std: Optional[float] = None) -> nn.ParameterList:
        if init_std is None:
            init_std = 1.0 / math.sqrt(self.bond_dim)

        tensor_list: List[nn.Parameter] = []

        left_tensor = self._randn(self.physical_dim, self.bond_dim) * init_std
        tensor_list.append(nn.Parameter(left_tensor))

        for _ in range(1, self.num_sites-1):
            bulk_tensor = self._randn(self.bond_dim, self.physical_dim, self.bond_dim) * init_std
            tensor_list.append(nn.Parameter(bulk_tensor))

        right_tensor = self._randn(self.bond_dim, self.physical_dim) * init_std
        tensor_list.append(nn.Parameter(right_tensor))

        return nn.ParameterList(tensor_list)
    
    def psi(self, configurations: torch.Tensor) -> torch.Tensor:
        batch_size, num_sites = configurations.shape
        assert num_sites == self.num_sites

        first_tensor = self.site_tensors[0]
        first_site_values = configurations[:, 0]

        left_env = first_tensor.index_select(dim=0, index=first_site_values)

        for site in range(1, num_sites-1):
            site_tensor = self.site_tensors[site]
            site_values = configurations[:, site]

            selected_slices = site_tensor.index_select(dim=1, index=site_values)

            site_matrix_batch = selected_slices.permute(1, 0, 2).contiguous()

            left_env = torch.bmm(left_env.unsqueeze(1), site_matrix_batch).squeeze(1)

        last_tensor = self.site_tensors[-1]
        last_site_values = configurations[:, -1]
        selected_columns = last_tensor.index_select(dim=1, index=last_site_values)
        selected_columns = selected_columns.transpose(0,1).contiguous()

        psi_values = (left_env * selected_columns).sum(dim=1)
        return psi_values

