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
    
    @torch.no_grad()
    def normalize_state(self) -> None:
        """
        Rescale the MPS so that <psi|psi> = 1.
        """
        log_z = self.log_norm()
        scale = torch.exp(-0.5 * log_z / self.num_sites)
        for p in self.site_tensors:
            p.data = p.data * scale
    
    def psi(self, configurations: torch.Tensor) -> torch.Tensor:
        """
        Computes the MPS amplitude Psi(v) for a batch of configurations.
        """
        if configurations.dtype != torch.long:
            configurations = configurations.long()
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
    
    def log_amplitude_squared(self, configurations: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable log |Psi(v)|^2 with per-site rescaling.
        """
        if configurations.dtype != torch.long:
            configurations = configurations.long()
        batch_size, num_sites = configurations.shape
        assert num_sites == self.num_sites
 
        device = configurations.device
 
        tensor = self.site_tensors[0]
        values = configurations[:, 0]
        env = tensor[:, values, :].permute(1, 0, 2).squeeze(1)
 
        log_scale = torch.zeros(batch_size, dtype=torch.float64, device=device)
 
        env_abs_max = env.abs().amax(dim=1).clamp_min(1e-30)
        env = env / env_abs_max.unsqueeze(1).to(env.dtype)
        log_scale = log_scale + env_abs_max.double().log()
 
        for site in range(1, num_sites):
            tensor = self.site_tensors[site]
            values = configurations[:, site]
            selected_matrices = tensor[:, values, :].permute(1, 0, 2)
            env = torch.bmm(env.unsqueeze(1), selected_matrices).squeeze(1)
 
            env_abs_max = env.abs().amax(dim=1).clamp_min(1e-30)
            env = env / env_abs_max.unsqueeze(1).to(env.dtype)
            log_scale = log_scale + env_abs_max.double().log()
 
        psi_rescaled = env.squeeze(1)
        if psi_rescaled.is_complex():
            abs2 = (psi_rescaled.real.square() + psi_rescaled.imag.square()).clamp_min(1e-300)
        else:
            abs2 = psi_rescaled.square().clamp_min(1e-300)
 
        log_abs2 = abs2.double().log() + 2.0 * log_scale
 
        real_dtype = (
            torch.float32 if self.dtype in (torch.float32, torch.complex64)
            else torch.float64
        )
        return log_abs2.to(real_dtype)
    
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
        log_abs_sq = self.log_amplitude_squared(configurations, eps=eps)
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
    
    def merge_sites(self, k: int) -> torch.Tensor:
        """
        """
        assert 0 <= k < self.num_sites - 1

        A_k  = self.site_tensors[k].data
        A_k1 = self.site_tensors[k + 1].data

        D_l, d1, D_mid = A_k.shape
        _, d2, D_r   = A_k1.shape

        return (A_k.reshape(D_l * d1, D_mid) @ A_k1.reshape(D_mid, d2 * D_r)).reshape(D_l, d1, d2, D_r)
    
    @torch.no_grad()
    def split_and_truncate(
        self,
        k: int,
        theta: torch.Tensor,
        direction: str,
        max_bond_dim: int,
        cutoff: float = 0.0,
    ) -> torch.Tensor:
        """
        """
        D_l, d1, d2, D_r = theta.shape
        U, S, Vh = torch.linalg.svd(theta.reshape(D_l * d1, d2 * D_r), full_matrices=False)

        n = self._truncation_rank(S, max_bond_dim, cutoff)
        U, S, Vh = U[:, :n], S[:n], Vh[:n, :]

        if direction == "right":
            self.site_tensors[k].data = U.reshape(D_l, d1, n)
            self.site_tensors[k + 1].data = (S.unsqueeze(1) * Vh).reshape(n, d2, D_r)
        else:
            self.site_tensors[k].data = (U * S.unsqueeze(0)).reshape(D_l, d1, n)
            self.site_tensors[k + 1].data = Vh.reshape(n, d2, D_r)

        return S.detach().clone()
    
    @torch.no_grad()
    def entanglement_entropy(
        self,
        max_bond_dim: Optional[int] = None,
        cutoff: float = 0.0,
    ) -> List[float]:
        """
        Von Neumann entanglement entropy S(k) = −Σ_i p_i ln p_i at each bond,
        where p_i = σ_i² / Σ σ_j².
        """
        svs = self.left_canonicalize_svd(max_bond_dim=max_bond_dim, cutoff=cutoff)
        entropies: List[float] = []
        for S in svs:
            p = S.square()
            p = p / p.sum().clamp_min(1e-30)
            ent = -(p * p.clamp_min(1e-30).log()).sum()
            entropies.append(ent.item())
        return entropies
    
    @property
    def bond_dims(self) -> List[int]:
        return [self.site_tensors[k].shape[2] for k in range(self.num_sites - 1)]
    
    @property
    def full_bond_dims(self) -> List[int]:
        """All bond dimensions including boundaries D_0=D_N=1  (length N+1)."""
        return [1] + self.bond_dims + [1]

    @property
    def num_parameters(self) -> int:
        n = sum(t.numel() for t in self.site_tensors)
        if self.dtype in (torch.complex64, torch.complex128):
            n *= 2
        return n
    
    def _apply_transfer_left(self, L: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        matrices = A.permute(1, 0, 2)
        L_times_conj = torch.matmul(L, matrices.conj())
        per_s = torch.matmul(matrices.transpose(1, 2), L_times_conj)
        return per_s.sum(dim = 0)
    
    def _apply_transfer_right(self, R: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        matrices = A.permute(1, 0, 2)
        M_times_R = torch.matmul(matrices, R)
        per_s = torch.matmul(M_times_R, matrices.conj().transpose(1, 2))
        return per_s.sum(dim = 0)
    
    def _left_transfer_envs(self) -> List[torch.Tensor]:
        """
        Build left transfer matrices for every bond
        """
        device = self.site_tensors[0].device
        envs: List[torch.Tensor] = [torch.ones(1, 1, dtype=self.dtype, device=device)]
        for k in range(self.num_sites):
            envs.append(self._apply_transfer_left(envs[k], self.site_tensors[k].data))
        return envs
    
    def _right_transfer_envs(self) -> List[torch.Tensor]:
        """
        Build right transfer matrices for every bond.
        """
        N = self.num_sites
        device = self.site_tensors[0].device
        envs: List[Optional[torch.Tensor]] = [None] * N
        envs[N - 1] = torch.ones(1, 1, dtype=self.dtype, device=device)
        for k in range(N - 1, 0, -1):
            envs[k - 1] = self._apply_transfer_right(envs[k], self.site_tensors[k].data)
        return envs
    
    def _open_site_rdm(self, L: torch.Tensor, A: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Single-site RDM kernel (un-normalised).
        """
        physical_dim = A.shape[1]
        matrices = A.permute(1, 0, 2)

        temp = torch.matmul(torch.matmul(L.conj().T, matrices), R)
        conj_flat = matrices.conj().reshape(physical_dim, -1)
        temp_flat = temp.reshape(physical_dim, -1)

        return torch.matmul(temp_flat, conj_flat.T)
    
    def _open_two_sites_M(self, L: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        First step of two-site RDM: leave one physical index pair open.
        """
        matrices = A.permute(1, 0, 2)
        conj = matrices.conj()
        LA = torch.matmul(L.conj().T, matrices)
        return torch.matmul(LA.permute(0, 2, 1).unsqueeze(1), conj.unsqueeze(0))
    
    def _propagate_M(self, M: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Propagate the open two-index tensor M through an intermediate site
        by tracing over its physical index (transfer matrix).
        """
        rdm_dim = M.shape[0]
        physical_dim = A.shape[1]
        bond_dim_right = A.shape[2]

        matrices = A.permute(1, 0, 2)
        conj = matrices.conj()

        M_flat = M.reshape(rdm_dim * rdm_dim, M.shape[2], M.shape[3])
        M_new = torch.zeros(rdm_dim * rdm_dim, bond_dim_right, bond_dim_right, dtype=M.dtype, device=M.device)

        for p in range(physical_dim):
            temp = torch.matmul(M_flat, conj[p]) 
            M_new = M_new + torch.matmul(matrices[p].T, temp) 
        
        return M_new.reshape(rdm_dim, rdm_dim, bond_dim_right, bond_dim_right)
    
    @torch.no_grad()
    def single_site_rdm(self, site: int) -> torch.Tensor:
        """
        Reduced density matrix for a single site (feature).
 
            ρ_k = Tr_{≠k}( |Ψ⟩⟨Ψ| )
 
        Returns a (d, d) Hermitian matrix normalised to trace 1.
        The diagonal entries give P(v_k = s) for each physical value s.
        """
        assert 0 <= site < self.num_sites
        left = self._left_transfer_envs()
        right = self._right_transfer_envs()
 
        rdm = self._open_site_rdm(left[site], self.site_tensors[site].data, right[site])
 
        tr = rdm.diagonal().real.sum().clamp_min(1e-30)
        return rdm / tr
    
    @torch.no_grad()
    def two_site_rdm(self, site_i: int, site_j: int) -> torch.Tensor:
        """
        Reduced density matrix for two sites (features).
 
            ρ_{ij} = Tr_{≠i,j}( |Ψ⟩⟨Ψ| )
 
        Returns a (d, d, d, d) tensor with index order [s_i, s_j, t_i, t_j],
        normalised so that  Σ_{s_i, s_j} ρ[s_i, s_j, s_i, s_j] = 1.
        """
        assert 0 <= site_i < site_j < self.num_sites
        d = self.physical_dim
 
        left = self._left_transfer_envs()
        right = self._right_transfer_envs()
 
        L = left[site_i]
        R = right[site_j]
        A_i = self.site_tensors[site_i].data
        A_j = self.site_tensors[site_j].data
 
        M = self._open_two_sites_M(L, A_i)
 
        for m in range(site_i + 1, site_j):
            M = self._propagate_M(M, self.site_tensors[m].data)
 
        matrices_j = A_j.permute(1, 0, 2)
        conj_j     = matrices_j.conj()
        AjR = torch.matmul(matrices_j, R)
 
        rdm = torch.zeros(d, d, d, d, dtype=M.dtype, device=M.device)
        for ti in range(d):
            for tj in range(d):
                temp = torch.matmul(M[:, ti], conj_j[tj])
                rdm[:, :, ti, tj] = torch.matmul(temp.reshape(d, -1), AjR.reshape(d, -1).T)
 
        trace = sum(rdm[s, t, s, t] for s in range(d) for t in range(d)).real.clamp_min(1e-30)
        return rdm / trace
    
    @torch.no_grad()
    def conditional_rdm(
        self,
        site_i: int,
        site_j: int,
        value_j: int,
    ) -> torch.Tensor:
        """
        RDM at site i conditioned on site j having a fixed value.
 
        Returns a (d, d) matrix.  Diagonal entries give P(v_i | v_j = value_j).
        """
        assert 0 <= site_i < self.num_sites
        assert 0 <= site_j < self.num_sites
        assert site_i != site_j
        assert 0 <= value_j < self.physical_dim
 
        lower, higher = min(site_i, site_j), max(site_i, site_j)
 
        left = self._left_transfer_envs()
        right = self._right_transfer_envs()
 
        L = left[lower]
        R = right[higher]
        A_lower = self.site_tensors[lower].data
        A_higher = self.site_tensors[higher].data
 
        if site_i < site_j:
            M = self._open_two_sites_M(L, A_lower)
 
            for m in range(lower + 1, higher):
                M = self._propagate_M(M, self.site_tensors[m].data)
 
            Fj = A_higher[:, value_j, :]
            RC = Fj @ R @ Fj.conj().T
            rdm = (M * RC).sum(dim=(-2, -1))
        else:
            Fj = A_lower[:, value_j, :]
            LC = Fj.conj().T @ L @ Fj
 
            for m in range(lower + 1, higher):
                LC = self._apply_transfer_left(LC, self.site_tensors[m].data)
 
            rdm = self._open_site_rdm(LC, A_higher, R)
 
        trace = rdm.diagonal().real.sum().clamp_min(1e-30)
        return rdm / trace
    
    @torch.no_grad()
    def feature_probabilities(self, site: int) -> torch.Tensor:
        """
        Marginal probability distribution P(v_k) for a single site.
 
        Equivalent to the diagonal of the single-site RDM.
        Returns a real (d,) tensor that sums to 1.
        """
        rdm = self.single_site_rdm(site)
        return rdm.diagonal().real
    
    @torch.no_grad()
    def all_single_site_rdms(self) -> List[torch.Tensor]:
        """
        Single-site RDMs for every site.
        """
        left = self._left_transfer_envs()
        right = self._right_transfer_envs()
        rdms: List[torch.Tensor] = []
        for k in range(self.num_sites):
            rdm = self._open_site_rdm(left[k], self.site_tensors[k].data, right[k])
            tr = rdm.diagonal().real.sum().clamp_min(1e-30)
            rdms.append(rdm / tr)
        return rdms
    
    @torch.no_grad()
    def all_feature_probabilities(self) -> torch.Tensor:
        """
        Marginal probabilities P(v_k = s) for every site k and value s.
 
        Returns a real (num_sites, physical_dim) tensor whose rows sum to 1.
        Faster than a Python loop over `feature_probabilities(k)`.
        """
        rdms = self.all_single_site_rdms()
        out = torch.stack([r.diagonal().real for r in rdms], dim=0)
        return out
 
    @torch.no_grad()
    def von_neumann_entropy_rdm(self, site: int) -> float:
        """
        Von Neumann entropy of the single-site RDM:
 
            S = −Tr(ρ log ρ)
 
        Measures how entangled the feature at *site* is with all the
        other features
        """
        rdm = self.single_site_rdm(site)
        eigenvalues = torch.linalg.eigvalsh(rdm.real)
        eigenvalues = eigenvalues.clamp_min(1e-30)
        return -(eigenvalues * eigenvalues.log()).sum().item()
    
    @torch.no_grad()
    def all_single_site_entropies(self) -> torch.Tensor:
        """
        Von Neumann entropies S(ρ_k) for every site k.
        """
        rdms = self.all_single_site_rdms()
        out = torch.zeros(self.num_sites, dtype=torch.float64)
        for k, r in enumerate(rdms):
            eigs = torch.linalg.eigvalsh(r.real).clamp_min(1e-30)
            out[k] = -(eigs * eigs.log()).sum().item()
        return out
 
    @torch.no_grad()
    def mutual_information(self, site_i: int, site_j: int) -> float:
        """
        Mutual information between two sites:
 
            I(i; j) = S(ρ_i) + S(ρ_j) − S(ρ_{ij})
 
        Quantifies total (including non-linear) correlation between two
        features.  Used to build the MI heatmap for feature
        clustering and ordering optimisation.
        """
        lo, hi = min(site_i, site_j), max(site_i, site_j)
 
        S_i = self.von_neumann_entropy_rdm(lo)
        S_j = self.von_neumann_entropy_rdm(hi)
 
        rdm_ij = self.two_site_rdm(lo, hi)
        d = self.physical_dim
        rho_matrix = rdm_ij.reshape(d * d, d * d)
        eigs = torch.linalg.eigvalsh(rho_matrix.real)
        eigs = eigs.clamp_min(1e-30)
        S_ij = -(eigs * eigs.log()).sum().item()
 
        return S_i + S_j - S_ij
    
    @torch.no_grad()
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Draw exact, independent samples from P(v) = |Ψ(v)|² / Z.
        """
        self.left_canonicalize()
 
        device = self.site_tensors[0].device
        N = self.num_sites
        d = self.physical_dim
        is_complex = self.dtype in (torch.complex64, torch.complex128)
 
        samples = torch.zeros(num_samples, N, dtype=torch.long, device=device)
 
        A_last = self.site_tensors[N - 1].data
        matrices = A_last.permute(1, 0, 2).squeeze(2)

        if is_complex:
            sq_norms = matrices.real.square() + matrices.imag.square()
        else:
            sq_norms = matrices.square()
        probs = sq_norms.sum(dim=1)
        probs = probs / probs.sum().clamp_min(1e-30)
 
        chosen = torch.multinomial(
            probs.unsqueeze(0).expand(num_samples, -1), 1
        ).squeeze(1)
        samples[:, N - 1] = chosen
 
        x = matrices[chosen]
 
        for k in range(N - 2, -1, -1):
            A_k = self.site_tensors[k].data
            mats = A_k.permute(1, 0, 2)

            candidates = torch.matmul(mats, x.T)
            candidates = candidates.permute(2, 0, 1)
 
            if is_complex:
                sq = candidates.real.square() + candidates.imag.square()
            else:
                sq = candidates.square()
            cond_probs = sq.sum(dim=2)
            cond_probs = cond_probs / cond_probs.sum(dim=1, keepdim=True).clamp_min(1e-30)
 
            chosen = torch.multinomial(cond_probs, 1).squeeze(1)
            samples[:, k] = chosen

            idx = chosen.unsqueeze(1).unsqueeze(2).expand(
                num_samples, 1, candidates.shape[2]
            )
            x = candidates.gather(1, idx).squeeze(1)
 
        return samples
    
    @torch.no_grad()
    def sample_conditional(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Conditional sampling: generate completions for partially known
        configurations.
 
        For sites where ``mask[k] == True`` the value in ``known[k]`` is
        kept fixed; the remaining sites are sampled from
        P(v_free | v_fixed).
        """
        assert known.shape[0] == self.num_sites
        assert mask.shape[0] == self.num_sites
 
        self.left_canonicalize()
 
        device = self.site_tensors[0].device
        N = self.num_sites
        d = self.physical_dim
        is_complex = self.dtype in (torch.complex64, torch.complex128)
 
        known = known.to(device)
        mask = mask.to(device)
        samples = torch.zeros(num_samples, N, dtype=torch.long, device=device)
 
        A_last = self.site_tensors[N - 1].data
        matrices = A_last.permute(1, 0, 2).squeeze(2)
 
        if mask[N - 1]:
            chosen = known[N - 1].expand(num_samples)
        else:
            if is_complex:
                sq_norms = matrices.real.square() + matrices.imag.square()
            else:
                sq_norms = matrices.square()
            probs = sq_norms.sum(dim=1)
            probs = probs / probs.sum().clamp_min(1e-30)
            chosen = torch.multinomial(
                probs.unsqueeze(0).expand(num_samples, -1), 1
            ).squeeze(1)
 
        samples[:, N - 1] = chosen
        x = matrices[chosen]
 
        for k in range(N - 2, -1, -1):
            A_k = self.site_tensors[k].data
            mats = A_k.permute(1, 0, 2)
 
            candidates = torch.matmul(mats, x.T)
            candidates = candidates.permute(2, 0, 1)
 
            if mask[k]:
                chosen = known[k].expand(num_samples)
            else:
                if is_complex:
                    sq = candidates.real.square() + candidates.imag.square()
                else:
                    sq = candidates.square()
                cond_probs = sq.sum(dim=2)
                cond_probs = cond_probs / cond_probs.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-30)
                chosen = torch.multinomial(cond_probs, 1).squeeze(1)
 
            samples[:, k] = chosen
 
            idx = chosen.unsqueeze(1).unsqueeze(2).expand(
                num_samples, 1, candidates.shape[2]
            )
            x = candidates.gather(1, idx).squeeze(1)
 
        return samples



