"""Explainability tools for a trained :class:`mps.MPS`.

Reduced density matrices, marginals, entropies and mutual information.
Split out of ``mps.py`` so the core class stays focused on training and
scoring.  Usage::

    from mps import MPS
    from mps_explainability import MPSExplainer

    expl = MPSExplainer(mps)
    rho = expl.single_site_rdm(0)
    mi = expl.mutual_information_matrix()

The transfer-environment cache lives on the explainer, not on the MPS.
It starts empty; every query rebuilds the environments unless
``precompute_environments()`` has been called.  If you mutate the MPS
after caching, call ``invalidate_environment_cache()`` or build a fresh
explainer -- the explainer cannot observe in-place changes to the MPS.

Public API
----------
Cache control        : precompute_environments, invalidate_environment_cache
Reduced density mats : single_site_rdm, all_single_site_rdms,
                       two_site_rdm, conditional_rdm
Probabilities        : feature_probabilities, all_feature_probabilities,
                       conditional_probabilities, joint_probabilities
Information measures : site_entropies, bond_entropies,
                       mutual_information, mutual_information_matrix

Everything prefixed with ``_`` is internal machinery (validation,
transfer-environment construction and un-normalised RDM kernels) and is
not part of the supported interface.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from mps import MPS, MPSShapeError


class MPSExplainer:
    """Reduced density matrices and information measures for a trained MPS.

    Everything here is read-only diagnostics: given the converged state, what did
    it learn about each feature and about the correlations between features? The
    building block is the reduced density matrix (RDM) for one or two sites,
    obtained by contracting the rest of the chain into left/right transfer
    environments. From the RDMs come marginal and joint probabilities,
    von Neumann and bond entropies, and the mutual-information matrix used to
    judge (and potentially reorder) the feature layout.

    The transfer environments are cached on the explainer, not the MPS, so they
    can't notice if you mutate the underlying state. After any in-place change
    call :meth:`invalidate_environment_cache` or build a new explainer.
    """

    # ==================================================================
    # Construction & cache management
    # ==================================================================
    def __init__(self, mps: MPS) -> None:
        """Wrap an MPS; the environment cache starts empty (lazy)."""
        self.mps = mps
        self._cached_left: Optional[List[torch.Tensor]] = None
        self._cached_right: Optional[List[torch.Tensor]] = None
        self._cache_valid: bool = False

    @torch.no_grad()
    def precompute_environments(self) -> None:
        """Cache the full left/right transfer environments."""
        self._cached_left = self._transfer_environments_left()
        self._cached_right = self._transfer_environments_right()
        self._cache_valid = True

    @torch.no_grad()
    def invalidate_environment_cache(self) -> None:
        """Drop the cached transfer environments."""
        self._cached_left = None
        self._cached_right = None
        self._cache_valid = False

    def _validate_site(self, site: int, name: str = "site") -> None:
        """Check that ``site`` is a valid site index."""
        if not (0 <= site < self.mps.num_sites):
            raise MPSShapeError(f"{name}={site} out of range [0, {self.mps.num_sites})")

    def _cached_environments(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return cached envs if valid, otherwise build fresh ones (uncached)."""
        if self._cache_valid:
            return self._cached_left, self._cached_right
        return self._transfer_environments_left(), self._transfer_environments_right()

    # ==================================================================
    # Transfer environments
    # ==================================================================
    @torch.no_grad()
    def _apply_transfer_left(self, left_environment: torch.Tensor, site_tensor: torch.Tensor) -> torch.Tensor:
        """Push a left environment through one site (sum over its physical leg).

        This is the transfer-matrix step ``E -> sum_s A_s^dag E A_s`` that grows
        the contracted left part of ``<Psi|Psi>`` by one site.
        """
        matrices = self.mps._as_matrices(site_tensor)
        left_times_conjugate = torch.matmul(left_environment, matrices.conj())
        per_site = torch.matmul(matrices.transpose(1, 2), left_times_conjugate)
        return per_site.sum(dim = 0)

    @torch.no_grad()
    def _apply_transfer_right(self, right_environment: torch.Tensor, site_tensor: torch.Tensor) -> torch.Tensor:
        """Push a right environment through one site (mirror of the left step)."""
        matrices = self.mps._as_matrices(site_tensor)
        matrices_times_right = torch.matmul(matrices, right_environment)
        per_site = torch.matmul(matrices_times_right, matrices.conj().transpose(1, 2))
        return per_site.sum(dim = 0)

    @torch.no_grad()
    def _transfer_environments_left(self) -> List[torch.Tensor]:
        """Build left transfer matrices for every bond."""
        device = self.mps.site_tensors[0].device
        envs: List[torch.Tensor] = [torch.ones(1, 1, dtype=self.mps.dtype, device=device)]
        for k in range(self.mps.num_sites):
            envs.append(self._apply_transfer_left(envs[k], self.mps.site_tensors[k].data))
        return envs

    @torch.no_grad()
    def _transfer_environments_right(self) -> List[torch.Tensor]:
        """Build right transfer matrices for every bond."""
        N = self.mps.num_sites
        device = self.mps.site_tensors[0].device
        envs: List[Optional[torch.Tensor]] = [None] * N
        envs[N - 1] = torch.ones(1, 1, dtype=self.mps.dtype, device=device)
        for k in range(N - 1, 0, -1):
            envs[k - 1] = self._apply_transfer_right(envs[k], self.mps.site_tensors[k].data)
        return envs

    # ==================================================================
    # RDM kernels
    # ==================================================================
    @torch.no_grad()
    def _open_site_rdm(self, left_environment: torch.Tensor, site_tensor: torch.Tensor, right_environment: torch.Tensor) -> torch.Tensor:
        """Single-site RDM kernel (un-normalised)."""
        physical_dim = site_tensor.shape[1]
        matrices = self.mps._as_matrices(site_tensor)

        intermediate = torch.matmul(torch.matmul(left_environment.transpose(0, 1), matrices), right_environment)
        conjugate_flattened = matrices.conj().reshape(physical_dim, -1)
        intermediate_flattened = intermediate.reshape(physical_dim, -1)

        return torch.matmul(intermediate_flattened, conjugate_flattened.T)

    @torch.no_grad()
    def _open_two_sites_tensor(self, left_environment: torch.Tensor, site_tensor: torch.Tensor) -> torch.Tensor:
        """First step of two-site RDM: leave one physical index pair open."""
        matrices = self.mps._as_matrices(site_tensor)
        conjugate_matrices = matrices.conj()
        left_times_matrices = torch.matmul(left_environment.transpose(0, 1), matrices)
        return torch.matmul(left_times_matrices.permute(0, 2, 1).unsqueeze(1), conjugate_matrices.unsqueeze(0))

    @torch.no_grad()
    def _propagate_open_two_site_tensor(self, open_two_site_tensor: torch.Tensor, site_tensor: torch.Tensor) -> torch.Tensor:
        """
        Propagate the open two-index tensor M through an intermediate site
        by tracing over its physical index (transfer matrix).
        """
        matrices = self.mps._as_matrices(site_tensor)

        return torch.einsum(
            "pca,xycd,pdb->xyab", matrices, open_two_site_tensor, matrices.conj()
        )

    # ==================================================================
    # Reduced density matrices
    # ==================================================================
    @torch.no_grad()
    def single_site_rdm(self, site: int) -> torch.Tensor:
        """
        Reduced density matrix for a single site (feature).

            ρ_k = Tr_{≠k}( |Ψ⟩⟨Ψ| )

        Returns a (d, d) Hermitian matrix normalised to trace 1.
        The diagonal entries give P(v_k = s) for each physical value s.
        """
        self._validate_site(site)

        left, right = self._cached_environments()

        rdm = self._open_site_rdm(left[site], self.mps.site_tensors[site].data, right[site])

        trace = rdm.diagonal().real.sum().clamp_min(self.mps._numerical_floor)
        return rdm / trace

    @torch.no_grad()
    def all_single_site_rdms(self) -> List[torch.Tensor]:
        """Single-site RDMs for every site."""
        left, right = self._cached_environments()
        rdms: List[torch.Tensor] = []
        for k in range(self.mps.num_sites):
            rdm = self._open_site_rdm(left[k], self.mps.site_tensors[k].data, right[k])
            trace = rdm.diagonal().real.sum().clamp_min(self.mps._numerical_floor)
            rdms.append(rdm / trace)
        return rdms

    @torch.no_grad()
    def two_site_rdm(self, site_i: int, site_j: int) -> torch.Tensor:
        """
        Reduced density matrix for two sites (features).

            ρ_{ij} = Tr_{≠i,j}( |Ψ⟩⟨Ψ| )

        Returns a (d, d, d, d) tensor with index order [s_i, s_j, t_i, t_j],
        normalised so that  Σ_{s_i, s_j} ρ[s_i, s_j, s_i, s_j] = 1.
        """
        self._validate_site(site_i, "site_i")
        self._validate_site(site_j, "site_j")
        if site_i >= site_j:
            raise MPSShapeError(
                f"Need site_i < site_j, got ({site_i}, {site_j})"
            )

        left, right = self._cached_environments()

        left_environment = left[site_i]
        right_environment = right[site_j]
        site_tensor_i = self.mps.site_tensors[site_i].data
        site_tensor_j = self.mps.site_tensors[site_j].data

        open_two_site_tensor = self._open_two_sites_tensor(left_environment, site_tensor_i)

        for m in range(site_i + 1, site_j):
            open_two_site_tensor = self._propagate_open_two_site_tensor(open_two_site_tensor, self.mps.site_tensors[m].data)

        matrices_j = self.mps._as_matrices(site_tensor_j)
        conjugate_j = matrices_j.conj()
        matrices_j_times_right = torch.matmul(matrices_j, right_environment)

        rdm = torch.einsum("xyab,sac,tbc->xsyt", open_two_site_tensor, matrices_j_times_right, conjugate_j)

        trace = torch.einsum("stst->", rdm).real.clamp_min(self.mps._numerical_floor)
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
        self._validate_site(site_i, "site_i")
        self._validate_site(site_j, "site_j")
        if site_i == site_j:
            raise MPSShapeError("site_i and site_j must differ")
        if not (0 <= value_j < self.mps.physical_dims[site_j]):
            raise MPSShapeError(
                f"value_j={value_j} out of range [0, {self.mps.physical_dims[site_j]}) for site {site_j}"
            )

        lower, higher = min(site_i, site_j), max(site_i, site_j)

        left, right = self._cached_environments()

        left_environment = left[lower]
        right_environment = right[higher]
        site_tensor_lower = self.mps.site_tensors[lower].data
        site_tensor_higher = self.mps.site_tensors[higher].data

        if site_i < site_j:
            open_two_site_tensor = self._open_two_sites_tensor(left_environment, site_tensor_lower)

            for m in range(lower + 1, higher):
                open_two_site_tensor = self._propagate_open_two_site_tensor(open_two_site_tensor, self.mps.site_tensors[m].data)

            fixed_value_j = site_tensor_higher[:, value_j, :]
            right_conditioned = fixed_value_j @ right_environment @ fixed_value_j.conj().T
            rdm = (open_two_site_tensor * right_conditioned).sum(dim=(-2, -1))
        else:
            fixed_value_j = site_tensor_lower[:, value_j, :]
            left_conditioned = fixed_value_j.T @ left_environment @ fixed_value_j.conj()

            for m in range(lower + 1, higher):
                left_conditioned = self._apply_transfer_left(left_conditioned, self.mps.site_tensors[m].data)

            rdm = self._open_site_rdm(left_conditioned, site_tensor_higher, right_environment)

        trace = rdm.diagonal().real.sum().clamp_min(self.mps._numerical_floor)
        return rdm / trace

    # ==================================================================
    # Probability distributions
    # ==================================================================
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
    def all_feature_probabilities(self) -> List[torch.Tensor]:
        """
        Marginal probabilities P(v_k = s) for every site k and value s.

        Returns a real (num_sites, physical_dim) tensor whose rows sum to 1.
        Faster than a Python loop over `feature_probabilities(k)`.
        """
        rdms = self.all_single_site_rdms()
        return [rdm.diagonal().real for rdm in rdms]

    @torch.no_grad()
    def conditional_probabilities(
        self, site_i: int, site_j: int, value_j: int
    ) -> torch.Tensor:
        """
        Conditional probability distribution P(v_i | v_j = value_j).

        Equivalent to the diagonal of the conditional RDM at site i.
        Returns a real (d_i,) tensor that sums to 1.
        """
        rdm = self.conditional_rdm(site_i, site_j, value_j)
        return rdm.diagonal().real

    @torch.no_grad()
    def joint_probabilities(self, site_i: int, site_j: int) -> torch.Tensor:
        """
        Joint probability distribution P(v_i, v_j) for two sites.

        Equivalent to the generalised diagonal of the two-site RDM.
        Returns a real (d_i, d_j) tensor that sums to 1, where entry
        [s_i, s_j] is P(v_i = s_i, v_j = s_j).
        """
        self._validate_site(site_i, "site_i")
        self._validate_site(site_j, "site_j")
        if site_i == site_j:
            raise MPSShapeError("site_i and site_j must differ")
        if site_i < site_j:
            rdm = self.two_site_rdm(site_i, site_j)
            return torch.einsum("ijij->ij", rdm).real
        rdm = self.two_site_rdm(site_j, site_i)
        return torch.einsum("ijij->ij", rdm).real.T

    # ==================================================================
    # Information-theoretic measures
    # ==================================================================
    @torch.no_grad()
    def site_entropies(self) -> torch.Tensor:
        """
        Single-site von Neumann entropy at every site:

            S(ρ_k) = −Tr(ρ_k log ρ_k)

        where ρ_k is the reduced density matrix of site k.  Returns a
        ``(num_sites,)`` real tensor.
        """
        rdms = self.all_single_site_rdms()
        out = torch.zeros(self.mps.num_sites, dtype=torch.float64)
        for k, rdm in enumerate(rdms):
            eigenvalues = torch.linalg.eigvalsh(rdm.real).clamp_min(self.mps._numerical_floor)
            out[k] = -(eigenvalues * eigenvalues.log()).sum().item()
        return out

    @torch.no_grad()
    def bond_entropies(
        self,
        max_bond_dim: Optional[int] = None,
        cutoff: float = 0.0,
        preserve_state: bool = True,
    ) -> List[float]:
        """
        Bipartite von Neumann entropy at every bond:

            S(k) = −Σ_i p_i ln p_i,    p_i = σ_i² / Σ σ_j²

        where σ_i are the singular values at bond k.  Returns ``num_sites - 1`` values.
        """
        if preserve_state:
            tensor_backup = [parameter.data.clone() for parameter in self.mps.site_tensors]
            try:
                singular_values_per_bond  = self.mps.left_canonicalize(
                    truncate=True, max_bond_dim=max_bond_dim, cutoff=cutoff
                )
            finally:
                for parameter, backed_up_data in zip(self.mps.site_tensors, tensor_backup):
                    parameter.data = backed_up_data
        else:
            singular_values_per_bond = self.mps.left_canonicalize(
                truncate=True, max_bond_dim=max_bond_dim, cutoff=cutoff
            )

        entropies: List[float] = []
        for singular_values in singular_values_per_bond:
            probabilities = singular_values.square()
            probabilities = probabilities / probabilities.sum().clamp_min(self.mps._numerical_floor)
            entropy = -(probabilities * probabilities.clamp_min(self.mps._numerical_floor).log()).sum()
            entropies.append(entropy.item())
        return entropies

    @torch.no_grad()
    def mutual_information(self, site_i: int, site_j: int) -> float:
        """
        Mutual information between two sites:

            I(i; j) = S(ρ_i) + S(ρ_j) − S(ρ_{ij})

        Quantifies total (including non-linear) correlation between two
        features.  Used to build the MI heatmap for feature
        clustering and ordering optimisation.
        """
        self._validate_site(site_i, "site_i")
        self._validate_site(site_j, "site_j")
        if site_i == site_j:
            raise MPSShapeError("site_i and site_j must differ")

        lower_site, higher_site = min(site_i, site_j), max(site_i, site_j)

        rdm_i = self.single_site_rdm(lower_site)
        rdm_j = self.single_site_rdm(higher_site)
        eigenvalues_i = torch.linalg.eigvalsh(rdm_i.real).clamp_min(self.mps._numerical_floor)
        eigenvalues_j = torch.linalg.eigvalsh(rdm_j.real).clamp_min(self.mps._numerical_floor)
        entropy_i = -(eigenvalues_i * eigenvalues_i.log()).sum().item()
        entropy_j = -(eigenvalues_j * eigenvalues_j.log()).sum().item()

        rdm_ij = self.two_site_rdm(lower_site, higher_site)
        physical_dim_i = self.mps.physical_dims[lower_site]
        physical_dim_j = self.mps.physical_dims[higher_site]
        density_matrix = rdm_ij.reshape(physical_dim_i * physical_dim_j, physical_dim_i * physical_dim_j)
        eigenvalues = torch.linalg.eigvalsh(density_matrix.real)
        eigenvalues = eigenvalues.clamp_min(self.mps._numerical_floor)
        entropy_ij = -(eigenvalues * eigenvalues.log()).sum().item()

        return entropy_i + entropy_j - entropy_ij

    @torch.no_grad()
    def mutual_information_matrix(self) -> torch.Tensor:
        """Full N×N mutual-information matrix in one pass."""
        N = self.mps.num_sites

        left, right = self._cached_environments()

        single_site_entropies = torch.zeros(N, dtype=torch.float64)
        for k in range(N):
            rdm = self._open_site_rdm(left[k], self.mps.site_tensors[k].data, right[k])
            trace = rdm.diagonal().real.sum().clamp_min(self.mps._numerical_floor)
            rdm = rdm / trace
            eigenvalues = torch.linalg.eigvalsh(rdm.real).clamp_min(self.mps._numerical_floor)
            single_site_entropies[k] = -(eigenvalues * eigenvalues.log()).sum().item()

        mutual_information_values = torch.zeros(N, N, dtype=torch.float64)
        for i in range(N):
            mutual_information_values[i, i] = single_site_entropies[i]

        for i in range(N):
            site_tensor_i = self.mps.site_tensors[i].data
            open_two_site_tensor = self._open_two_sites_tensor(left[i], site_tensor_i)

            for j in range(i + 1, N):
                if j > i + 1:
                    open_two_site_tensor = self._propagate_open_two_site_tensor(open_two_site_tensor, self.mps.site_tensors[j - 1].data)

                site_tensor_j = self.mps.site_tensors[j].data
                right_environment = right[j]
                matrices_j = self.mps._as_matrices(site_tensor_j)
                conjugate_j = matrices_j.conj()
                matrices_j_times_right = torch.matmul(matrices_j, right_environment)

                rdm = torch.einsum("xyab,sac,tbc->xsyt", open_two_site_tensor, matrices_j_times_right, conjugate_j)

                trace = torch.einsum("stst->", rdm).real.clamp_min(
                    self.mps._numerical_floor
                )
                rdm = rdm / trace

                physical_dim_i = self.mps.physical_dims[i]
                physical_dim_j = self.mps.physical_dims[j]
                density_matrix = rdm.reshape(physical_dim_i * physical_dim_j, physical_dim_i * physical_dim_j)
                eigenvalues = torch.linalg.eigvalsh(density_matrix.real).clamp_min(self.mps._numerical_floor)
                entropy_ij = -(eigenvalues * eigenvalues.log()).sum().item()

                mutual_information_ij = single_site_entropies[i].item() + single_site_entropies[j].item() - entropy_ij
                mutual_information_values[i, j] = mutual_information_ij
                mutual_information_values[j, i] = mutual_information_ij

        return mutual_information_values