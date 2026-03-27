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

        self._is_canonical = False

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

        left_tensor = self._randn(1, self.physical_dim, self.bond_dim) * init_std
        tensor_list.append(nn.Parameter(left_tensor))

        for _ in range(1, self.num_sites-1):
            bulk_tensor = self._randn(self.bond_dim, self.physical_dim, self.bond_dim) * init_std
            tensor_list.append(nn.Parameter(bulk_tensor))

        right_tensor = self._randn(self.bond_dim, self.physical_dim, 1) * init_std
        tensor_list.append(nn.Parameter(right_tensor))

        return nn.ParameterList(tensor_list)
    
    def psi(self, configurations: torch.Tensor) -> torch.Tensor:
        """
        Computes the MPS amplitude Psi(v) for a batch of configurations.
        """
        batch_size, num_sites = configurations.shape
        assert num_sites == self.num_sites

        tensor = self.site_tensors[0]
        values = configurations[:, 0]
        env = tensor[:, values, :].permute(1, 0, 2)

        for site in range(1, num_sites):
            tensor = self.site_tensors[site]
            values = configurations[:, site]
            selected_matrices = tensor[:, values, :].permute(1, 0, 2)

            env = torch.bmm(env, selected_matrices)

        return env.reshape(batch_size)
    
    def amplitude_squared(self, configurations: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
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

        env = torch.ones(1, 1, dtype=self.dtype, device=self.site_tensors[0].device)
        log_scale = torch.zeros((), dtype=torch.float64, device=env.device)

        for site in range(self.num_sites):
            tensor = self.site_tensors[site]
            matrices = tensor.permute(1, 0, 2)

            contracted = torch.matmul(env, matrices)
            matrices_dagger = matrices.conj().transpose(1, 2)
            env = torch.matmul(matrices_dagger, contracted).sum(dim=0)

            scale = env.abs().max().clamp_min(1e-30)
            env   = env / scale
            log_scale = log_scale + scale.double().log()
        
        z_value = env.squeeze()
        real_dtype = (
            torch.float32 if self.dtype in (torch.float32, torch.complex64)
            else torch.float64
        )
        return (z_value.real.clamp_min(1e-30).double().log() + log_scale).to(real_dtype)
    
    def norm(self) -> torch.Tensor:
        """
        Returns Z = <psi|psi>.
        """
        return self.log_norm().exp()

    def log_prob(self, configurations: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
        """
        Computes log P(v) = log |Psi(v)|^2 - log Z
        """
        abs_sq = self.amplitude_squared(configurations, eps=eps)
        log_abs_sq = torch.log(abs_sq)
        log_z = self.log_norm()
        return log_abs_sq - log_z

    def nll(self, configurations: torch.Tensor, reduction: str = "mean", eps: float = 1e-30) -> torch.Tensor:
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
            tensor = self.site_tensors[site].data
            D_l, d, D_r = tensor.shape

            Q, R = torch.linalg.qr(tensor.reshape(D_l * d, D_r))
            new_D = Q.shape[-1]
            self.site_tensors[site].data = Q.reshape(D_l, d, new_D)
            
            next_tensor = self.site_tensors[site + 1].data
            _, d_n, D_r2 = next_tensor.shape

            self.site_tensors[site + 1].data = (
                R @ next_tensor.reshape(D_r, d_n * D_r2)
            ).reshape(new_D, d_n, D_r2)

    @torch.no_grad()
    def right_canonicalize(self, from_site: Optional[int] = None) -> None:
        if from_site is None:
            from_site = 1

        for site in range(self.num_sites - 1, from_site - 1, -1):
            tensor = self.site_tensors[site].data
            D_l, d, D_r = tensor.shape

            Q, R = torch.linalg.qr(tensor.reshape(D_l, d * D_r).conj().T)
            new_D = Q.shape[1]
            self.site_tensors[site].data = Q.conj().T.reshape(new_D, d, D_r)

            previous_tensor = self.site_tensors[site - 1].data
            Rdag = R.conj().T
            D_l2, d_p, _ = previous_tensor.shape

            self.site_tensors[site - 1].data = (
                previous_tensor.reshape(D_l2 * d_p, D_l) @ Rdag
            ).reshape(D_l2, d_p, new_D)

    @torch.no_grad()
    def mixed_canonicalize(self, center: int) -> None:
        assert 0 <= center < self.num_sites
        self.left_canonicalize(up_to=center)
        self.right_canonicalize(from_site=center + 1)
        self._center = center
        self._is_canonical = True

    def _truncation_rank(
        self,
        S: torch.Tensor,
        max_bond_dim: Optional[int],
        cutoff: float,
    ) -> int:
        """
        Determine how many singular values to keep.
        """
        n = len(S)
        if cutoff > 0:
            S_max = S[0].abs().clamp_min(1e-30)
            n = max(int((S / S_max >= cutoff).sum().item()), 1)
        if max_bond_dim is not None:
            n = min(n, max_bond_dim)
        return n

    @torch.no_grad()
    def left_canonicalize_svd(
        self,
        up_to: Optional[int] = None,
        max_bond_dim: Optional[int] = None,
        cutoff: float = 0.0
    ) -> List[torch.Tensor]:
        """
        """
        if up_to is None:
            up_to = self.num_sites - 1
        
        singular_values = []

        for site in range(up_to):
            tensor = self.site_tensors[site].data
            D_l, d, D_r = tensor.shape

            U, S, Vh = torch.linalg.svd(tensor.reshape(D_l * d, D_r), full_matrices=False)
            n = self._truncation_rank(S, max_bond_dim, cutoff)
            U, S, Vh = U[:, :n], S[:n], Vh[:n, :]

            singular_values.append(S.detach().clone())

            self.site_tensors[site].data = U.reshape(D_l, d, n)

            SV = S.unsqueeze(1) * Vh
            next_tensor = self.site_tensors[site + 1].data
            _, d_n, D_r2 = next_tensor.shape

            self.site_tensors[site + 1].data = (
                SV @ next_tensor.reshape(D_r, d_n * D_r2)
            ).reshape(n, d_n, D_r2)
        
        return singular_values
    
    @torch.no_grad()
    def right_canonicalize_svd(
        self,
        from_site: Optional[int] = None,
        max_bond_dim: Optional[int] = None,
        cutoff: float = 0.0,
    ) -> List[torch.Tensor]:
        """
        """
        if from_site is None:
            from_site = 1

        singular_values: List[torch.Tensor] = []

        for site in range(self.num_sites - 1, from_site - 1, -1):
            tensor = self.site_tensors[site].data
            D_l, d, D_r = tensor.shape

            U, S, Vh = torch.linalg.svd(tensor.reshape(D_l, d * D_r), full_matrices=False)
            n = self._truncation_rank(S, max_bond_dim, cutoff)
            U, S, Vh = U[:, :n], S[:n], Vh[:n, :]

            singular_values.append(S.detach().clone())

            self.site_tensors[site].data = Vh.reshape(n, d, D_r)

            US = U * S.unsqueeze(0)
            previous_tensor = self.site_tensors[site - 1].data
            D_l2, d_p, _ = previous_tensor.shape
            self.site_tensors[site - 1].data = (
                previous_tensor.reshape(D_l2 * d_p, D_l) @ US
            ).reshape(D_l2, d_p, n)

        singular_values.reverse()
        return singular_values

    def log_norm_from_center(self) -> torch.Tensor:
        """
        """
        assert self._is_canonical and hasattr(self, "_center"), (
            "Call mixed_canonicalize(k) before using log_norm_from_center()"
        )

        Gamma = self.site_tensors[self._center]
        sq_norm = (Gamma.conj() * Gamma).real.sum()
        return torch.log(sq_norm.clamp_min(1e-30))
