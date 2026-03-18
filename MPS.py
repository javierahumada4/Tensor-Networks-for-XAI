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

        self.site_tensors = self._normal_init(init_std)

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

    def _normal_init(self, init_std: Optional[float] = None) -> nn.ParameterList:
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
        """
        Computes the MPS amplitude Psi(v) for a batch of configurations.
        """
        batch_size, num_sites = configurations.shape
        assert num_sites == self.num_sites

        first_tensor = self.site_tensors[0]
        first_site_values = configurations[:, 0]
        left_env = first_tensor.index_select(dim=0, index=first_site_values)

        for site in range(1, num_sites-1):
            site_tensor = self.site_tensors[site]
            site_values = configurations[:, site]
            selected_slices = site_tensor.index_select(dim=1, index=site_values)
            selected_matrices = selected_slices.permute(1, 0, 2)

            left_env = torch.bmm(left_env.unsqueeze(1), selected_matrices).squeeze(1)

        last_tensor = self.site_tensors[-1]
        last_site_values = configurations[:, -1]
        selected_slices = last_tensor.index_select(dim=1, index=last_site_values)
        selected_columns = selected_slices.transpose(0,1)

        psi_values = (left_env * selected_columns).sum(dim=1)
        return psi_values
    
    def amplitude_squared(self, configurations: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Returns |Psi(v)|^2 for each configuration.
        """
        psi_values = self.psi(configurations)
        if psi_values.is_complex():
            abs_sq = psi_values.real.square() + psi_values.imag.square() 
        else:
            abs_sq = psi_values.square()
        return abs_sq.clamp_min(eps)
    
    def log_norm(self) -> torch.Tensor:
        """ 
        Computes log Z = log <psi|psi>.
        """

        first_tensor = self.site_tensors[0]

        env = first_tensor.conj().transpose(0,1) @ first_tensor

        log_scale = 0.0

        for site in range(1, self.num_sites - 1):
            site_tensor = self.site_tensors[site]
            site_matrices = site_tensor.permute(1, 0, 2)

            env_times_site_matrices = torch.matmul(env, site_matrices)
            site_matrices_dag = site_matrices.conj().transpose(1, 2)
            env = torch.matmul(site_matrices_dag, env_times_site_matrices).sum(dim=0)

            scale = env.abs().max().clamp_min(1e-30)
            env   = env / scale
            log_scale += scale.log().item()

        last_tensor = self.site_tensors[-1]

        env_times_last_tensor = env @ last_tensor
        
        z_value = (last_tensor.conj() * env_times_last_tensor).sum()

        return torch.log(z_value.real.clamp_min(1e-30)) + log_scale
    
    def norm(self) -> torch.Tensor:
        """
        Returns Z = <psi|psi>.
        """
        return self.log_norm().exp()

    def log_prob(self, configurations: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Computes log P(v) = log |Psi(v)|^2 - log Z
        """
        abs_sq = self.amplitude_squared(configurations, eps=eps)
        log_abs_sq = torch.log(abs_sq)
        log_z = self.log_norm()
        return log_abs_sq - log_z

    def nll(self, configurations: torch.Tensor, reduction: str = "mean", eps: float = 1e-12) -> torch.Tensor:
        """
        Negative log-likelihood:
            NLL(v) = -log P(v)

        reduction:
          - "none": returns shape (batch_size,)
          - "mean": scalar
          - "sum" : scalar
        """
        nll_values = -self.log_prob(configurations, eps=eps)

        if reduction == "none":
            return nll_values
        if reduction == "mean":
            return nll_values.mean()
        if reduction == "sum":
            return nll_values.sum()

        raise ValueError(f"Unsupported reduction: {reduction}")
    
    @torch.no_grad()
    def left_canonicalize(self, up_to: Optional[int] = None) -> None:
        """
        """
        if up_to is None:
            up_to = self.num_sites - 1
        
        for site in range(up_to):
            site_tensor = self.site_tensors[site].data

            if site == 0:
                Q, R = torch.linalg.qr(site_tensor)

                self.site_tensors[site].data = Q
            else:
                D_l, d, D_r = site_tensor.shape
                Q, R = torch.linalg.qr(site_tensor.flatten(0, 1))
                
                self.site_tensors[site].data = Q.reshape(D_l, d, -1)
            
            next_tensor = self.site_tensors[site + 1].data
            if site + 1 == self.num_sites - 1:
                self.site_tensors[site + 1].data = R @ next_tensor
            else:
                D_l, d, D_r = next_tensor.shape
                self.site_tensors[site + 1].data = (
                    R @ next_tensor.reshape(D_l, d * D_r)
                ).reshape(-1, d, D_r)

    @torch.no_grad()
    def right_canonicalize(self, from_site: Optional[int] = None) -> None:
        if from_site is None:
            from_site = 1

        for site in range(self.num_sites - 1, from_site - 1, -1):
            site_tensor = self.site_tensors[site].data

            if site == self.num_sites - 1:
                Q, R = torch.linalg.qr(site_tensor.conj().T)
                self.site_tensors[site].data = Q.conj().T
            else:
                D_l, d, D_r = site_tensor.shape
                Q, R = torch.linalg.qr(site_tensor.reshape(D_l, d * D_r).conj().T)
                self.site_tensors[site].data = Q.conj().T.reshape(-1, d, D_r)

            previous_tensor = self.site_tensors[site - 1].data
            Rdag = R.conj().T
            if site - 1 == 0:
                self.site_tensors[site - 1].data = previous_tensor @ Rdag
            else:
                D_l, d, D_r = previous_tensor.shape
                self.site_tensors[site - 1].data = (
                    previous_tensor.reshape(D_l * d, D_r) @ Rdag
                ).reshape(D_l, d, -1)

    @torch.no_grad()
    def mixed_canonicalize(self, center: int) -> None:
        assert 0 <= center < self.num_sites
        self.left_canonicalize(up_to=center)
        self.right_canonicalize(from_site=center + 1)
        self._center = center

    def log_norm_from_center(self) -> torch.Tensor:
        """
        """
        assert hasattr(self, "_center"), (
            "Call mixed_canonicalize(k) before using log_norm_from_center()"
        )

        Gamma = self.site_tensors[self._center]
        sq_norm = (Gamma.conj() * Gamma).real.sum()
        return torch.log(sq_norm.clamp_min(1e-30))
