"""Generative sampling for a trained :class:`mps.MPS`.

Split out of ``mps.py`` so the core class stays focused on training and
scoring.  Usage::

    from mps import MPS
    from mps_generative import MPSSampler

    sampler = MPSSampler(mps)
    draws = sampler.sample(100)

The sampler holds a reference to the MPS; it reads its tensors and may
canonicalize it in place (see ``preserve_state``).
"""

from __future__ import annotations

from typing import List, Optional

import torch

from mps import MPS, MPSShapeError, MPSNumericalError


class MPSSampler:
    """Exact, autoregressive sampler for ``P(v) = |Psi(v)|^2 / Z``.

    Sampling an MPS is exact (no MCMC): canonicalise the chain so the norm sits
    at one end, then draw sites one at a time, each from the correct conditional
    given the sites already chosen. The per-sample boundary vector is carried
    along so a whole batch is drawn in lockstep.

    The sampler keeps a reference to the MPS and *canonicalises it in place*. Pass
    ``preserve_state=True`` to any public method to snapshot and restore the
    tensors so the caller's MPS comes back untouched.
    """

    def __init__(self, mps: MPS) -> None:
        """Wrap the MPS to sample from; methods may canonicalise it in place."""
        self.mps = mps

    @staticmethod
    def _abs_squared(x: torch.Tensor) -> torch.Tensor:
        """
        ``|x|^2`` for real or complex tensors
        """
        if x.is_complex():
            return x.real.square() + x.imag.square()
        return x.square()

    @staticmethod
    def _check_valid_probabilities(
        probabilities: torch.Tensor, site: int, context: str = "sample"
    ) -> None:
        """Verify that ``probs`` is a valid (multinomial-feedable) distribution.

        ``torch.multinomial`` errors out cryptically on non-finite inputs or
        on distributions whose row-sum is zero; both happen in practice when
        the MPS has over-/underflowed or has been catastrophically
        truncated.  We prefer a clear ``MPSNumericalError`` pointing at the
        actual cause.

        ``probs`` may be 1D (single distribution) or 2D (batch of
        distributions, one per row).
        """
        if not torch.isfinite(probabilities).all():
            raise MPSNumericalError(
                f"Non-finite conditional probabilities at site {site} during "
                f"{context}. The MPS likely under/overflowed; call "
                f"normalize_state() first or check init_std."
            )
        row_sums = probabilities.sum(dim=-1) if probabilities.dim() > 1 else probabilities.sum()
        if (row_sums <= 0).any():
            raise MPSNumericalError(
                f"Degenerate (all-zero) probabilities at site {site} during "
                f"{context}. The MPS state has likely been over-truncated or "
                "lies in an annihilated subspace for the current conditioning."
            )

    @torch.no_grad()
    def sample(self, num_samples: int = 1, preserve_state: bool = False) -> torch.Tensor:
        """Draw ``num_samples`` independent samples from ``P(v) = |Psi(v)|^2 / Z``.

        Returns a ``(num_samples, num_sites)`` long tensor. Set
        ``preserve_state=True`` if you need the MPS left exactly as it was — the
        draw left-canonicalises it in place otherwise.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        
        if preserve_state:
            tensor_backup = [parameter.data.clone() for parameter in self.mps.site_tensors]
            try:
                return self._sample_left_canonical(num_samples)
            finally:
                for parameter, backed_up_data in zip(self.mps.site_tensors, tensor_backup):
                    parameter.data = backed_up_data
        return self._sample_left_canonical(num_samples)

    def _sample_left_canonical(self, num_samples: int) -> torch.Tensor:
        """Unconditional draw: left-canonicalise, then sample right-to-left.

        After left-canonicalisation the right boundary carries the norm, so the
        last site's marginal is read straight off its tensor. Each earlier site
        is then sampled from its conditional given the already-chosen suffix,
        whose contribution is summarised by the running vector ``x``.
        """
        self.mps.left_canonicalize()
 
        device = self.mps.site_tensors[0].device
        N = self.mps.num_sites
 
        samples = torch.zeros(num_samples, N, dtype=torch.long, device=device)
 
        site_tensor_last = self.mps.site_tensors[N - 1].data
        matrices = self.mps._as_matrices(site_tensor_last).squeeze(2)
 
        squared_norms = self._abs_squared(matrices)
        probabilities = squared_norms.sum(dim=1)
        probabilities = probabilities / probabilities.sum().clamp_min(self.mps._numerical_floor)
        self._check_valid_probabilities(probabilities, site=N - 1, context="sample")
 
        chosen = torch.multinomial(
            probabilities.unsqueeze(0).expand(num_samples, -1), 1
        ).squeeze(1)
        samples[:, N - 1] = chosen
 
        x = matrices[chosen]
 
        for k in range(N - 2, -1, -1):
            site_tensor_k = self.mps.site_tensors[k].data
            matrices = self.mps._as_matrices(site_tensor_k)
 
            candidates = torch.matmul(matrices, x.T)
            candidates = candidates.permute(2, 0, 1)
 
            squared_amplitudes = self._abs_squared(candidates)
            conditional_probabilities = squared_amplitudes.sum(dim=2)
            conditional_probabilities = conditional_probabilities / conditional_probabilities.sum(dim=1, keepdim=True).clamp_min(self.mps._numerical_floor)
            self._check_valid_probabilities(conditional_probabilities, site=k, context="sample")
 
            chosen = torch.multinomial(conditional_probabilities, 1).squeeze(1)
            samples[:, k] = chosen
 
            gather_indices = chosen.unsqueeze(1).unsqueeze(2).expand(
                num_samples, 1, candidates.shape[2]
            )
            x = candidates.gather(1, gather_indices).squeeze(1)
 
        return samples

    @torch.no_grad()
    def sample_conditional(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int = 1,
        preserve_state: bool = False,
    ) -> torch.Tensor:
        """Complete partially-observed configurations by sampling the rest.

        ``known`` holds a value for every site but only the positions flagged
        ``True`` in ``mask`` are treated as fixed; the rest are drawn from their
        conditional ``P(free | fixed)``. Returns ``(num_samples, num_sites)``
        with the fixed columns copied through unchanged.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        if known.dim() != 1 or known.shape[0] != self.mps.num_sites:
            raise MPSShapeError(
                f"known must be 1D with {self.mps.num_sites} entries, "
                f"got shape {tuple(known.shape)}"
            )
        if mask.dim() != 1 or mask.shape[0] != self.mps.num_sites:
            raise MPSShapeError(
                f"mask must be 1D with {self.mps.num_sites} entries, "
                f"got shape {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}")
        
        if preserve_state:
            tensor_backup = [parameter.data.clone() for parameter in self.mps.site_tensors]
            try:
                return self._sample_conditional_dispatch(known, mask, num_samples)
            finally:
                for parameter, backed_up_data in zip(self.mps.site_tensors, tensor_backup):
                    parameter.data = backed_up_data
        return self._sample_conditional_dispatch(known, mask, num_samples)

    def _sample_conditional_dispatch(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Pick the cheapest conditional-sampling routine for this mask.

        Three cases, in increasing cost. If the fixed sites all sit to one side
        of the free ones, a single canonicalisation puts the chain in the right
        gauge and one ordinary sweep does the job
        (:meth:`_sample_conditional_right_to_left` /
        ``_left_to_right``). Fixed sites interleaved with free ones need the
        general ladder contraction (:meth:`_sample_conditional_scattered`).
        Degenerate masks (nothing fixed / everything fixed) short-circuit.
        """
        N = self.mps.num_sites
        device = self.mps.site_tensors[0].device

        known = known.to(device).long()
        mask = mask.to(device)

        if mask.any():
            fixed_positions_check = mask.nonzero(as_tuple=False).flatten()
            for pos in fixed_positions_check.tolist():
                physical_dim = self.mps.physical_dims[pos]
                value = int(known[pos].item())
                if value < 0 or value >= physical_dim:
                    raise MPSShapeError(
                        f"known[{pos}]={value} out of range [0, {physical_dim}) "
                        f"for that site's physical dim"
                    )
 
        free_positions = (~mask).nonzero(as_tuple=False).flatten()
        fixed_positions = mask.nonzero(as_tuple=False).flatten()
 
        if fixed_positions.numel() == 0:
            return self.sample(num_samples)
        if free_positions.numel() == 0:
            return known.long().unsqueeze(0).expand(num_samples, N).clone()
 
        if fixed_positions.min().item() > free_positions.max().item():
            return self._sample_conditional_right_to_left(known, mask, num_samples)
        if fixed_positions.max().item() < free_positions.min().item():
            return self._sample_conditional_left_to_right(known, mask, num_samples)
 
        return self._sample_conditional_scattered(known, mask, num_samples)

    @torch.no_grad()
    def _sample_conditional_right_to_left(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Conditional draw when every fixed site is at the right end.

        Left-canonicalise, then sweep right-to-left as in the unconditional case,
        except that at a fixed site we skip the multinomial and clamp the value
        to ``known``. The suffix vector ``x`` still threads through so free sites
        see the correct conditional.
        """
        self.mps.left_canonicalize()
 
        device = self.mps.site_tensors[0].device
        N = self.mps.num_sites
 
        samples = torch.zeros(num_samples, N, dtype=torch.long, device=device)
 
        site_tensor_last = self.mps.site_tensors[N - 1].data
        matrices = self.mps._as_matrices(site_tensor_last).squeeze(2)
 
        if mask[N - 1]:
            chosen = known[N - 1].expand(num_samples)
        else:
            squared_norms = self._abs_squared(matrices)
            probabilities = squared_norms.sum(dim=1)
            probabilities = probabilities / probabilities.sum().clamp_min(self.mps._numerical_floor)
            self._check_valid_probabilities(probabilities, site=N - 1, context="sample_conditional_RL")
            chosen = torch.multinomial(
                probabilities.unsqueeze(0).expand(num_samples, -1), 1
            ).squeeze(1)
 
        samples[:, N - 1] = chosen
        x = matrices[chosen]
 
        for k in range(N - 2, -1, -1):
            site_tensor_k = self.mps.site_tensors[k].data
            matrices = self.mps._as_matrices(site_tensor_k)
 
            candidates = torch.matmul(matrices, x.T).permute(2, 0, 1)
 
            if mask[k]:
                chosen = known[k].expand(num_samples)
            else:
                squared_amplitudes = self._abs_squared(candidates)
                conditional_probabilities = squared_amplitudes.sum(dim=2)
                conditional_probabilities = conditional_probabilities / conditional_probabilities.sum(dim=1, keepdim=True).clamp_min(self.mps._numerical_floor)
                self._check_valid_probabilities(conditional_probabilities, site=k, context="sample_conditional_RL")
                chosen = torch.multinomial(conditional_probabilities, 1).squeeze(1)
 
            samples[:, k] = chosen
            gather_indices = chosen.unsqueeze(1).unsqueeze(2).expand(num_samples, 1, candidates.shape[2])
            x = candidates.gather(1, gather_indices).squeeze(1)
 
        return samples

    @torch.no_grad()
    def _sample_conditional_left_to_right(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Conditional draw when every fixed site is at the left end.

        Mirror of the right-to-left case: right-canonicalise from site 1 and
        sweep left-to-right, clamping fixed sites and sampling the rest.
        """
        self.mps.right_canonicalize(from_site=1)
 
        device = self.mps.site_tensors[0].device
        N = self.mps.num_sites
 
        samples = torch.zeros(num_samples, N, dtype=torch.long, device=device)
 
        site_tensor_first = self.mps.site_tensors[0].data
        matrices = self.mps._as_matrices(site_tensor_first).squeeze(1)
 
        if mask[0]:
            chosen = known[0].expand(num_samples)
        else:
            squared_norms = self._abs_squared(matrices)
            probabilities = squared_norms.sum(dim=1)
            probabilities = probabilities / probabilities.sum().clamp_min(self.mps._numerical_floor)
            self._check_valid_probabilities(probabilities, site=0, context="sample_conditional_LR")
            chosen = torch.multinomial(
                probabilities.unsqueeze(0).expand(num_samples, -1), 1
            ).squeeze(1)
 
        samples[:, 0] = chosen
        x = matrices[chosen]
 
        for k in range(1, N):
            site_tensor_k = self.mps.site_tensors[k].data
            matrices = self.mps._as_matrices(site_tensor_k)
 
            candidates = torch.einsum('sa,vab->svb', x, matrices)
 
            if mask[k]:
                chosen = known[k].expand(num_samples)
            else:
                squared_amplitudes = self._abs_squared(candidates)
                conditional_probabilities = squared_amplitudes.sum(dim=2)
                conditional_probabilities = conditional_probabilities / conditional_probabilities.sum(dim=1, keepdim=True).clamp_min(self.mps._numerical_floor)
                self._check_valid_probabilities(conditional_probabilities, site=k, context="sample_conditional_LR")
                chosen = torch.multinomial(conditional_probabilities, 1).squeeze(1)
 
            samples[:, k] = chosen
            gather_indices = chosen.unsqueeze(1).unsqueeze(2).expand(num_samples, 1, candidates.shape[2])
            x = candidates.gather(1, gather_indices).squeeze(1)
 
        return samples

    @torch.no_grad()
    def _sample_conditional_scattered(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """General conditional draw for fixed sites scattered through the chain.

        No single gauge makes this a plain sweep, so we precompute right
        environments that already have the fixed sites projected onto their known
        values (the ``right_masked`` ladder), then sweep left-to-right. At each
        free site the weight of every candidate value is its squared amplitude
        folded against that masked right environment, giving the exact
        conditional; fixed sites are clamped.
        """
        device = self.mps.site_tensors[0].device
        N = self.mps.num_sites
        is_complex = self.mps.dtype in (torch.complex64, torch.complex128)

        right_masked: List[Optional[torch.Tensor]] = [None] * N
        right_masked[N - 1] = torch.ones(1, 1, dtype=self.mps.dtype, device=device)
 
        for k in range(N - 1, 0, -1):
            site_tensor_k = self.mps.site_tensors[k].data
            matrices = self.mps._as_matrices(site_tensor_k)
            right_environment_next = right_masked[k]
 
            if mask[k]:
                fixed_value = int(known[k].item())
                site_tensor_at_value = matrices[fixed_value]
                right_masked[k - 1] = site_tensor_at_value @ right_environment_next @ site_tensor_at_value.conj().T
            else:
                matrices_times_right = torch.matmul(matrices, right_environment_next)
                right_masked[k - 1] = torch.matmul(
                    matrices_times_right, matrices.conj().transpose(1, 2)
                ).sum(dim=0)
 
        samples = torch.zeros(num_samples, N, dtype=torch.long, device=device)
 
        x = torch.ones(num_samples, 1, dtype=self.mps.dtype, device=device)
 
        for k in range(N):
            site_tensor_k = self.mps.site_tensors[k].data
            matrices = self.mps._as_matrices(site_tensor_k)
            right_environment_next = right_masked[k]
 
            candidates = torch.einsum('sa,vab->svb', x, matrices)
 
            if mask[k]:
                fixed_value = int(known[k].item())
                chosen = torch.full(
                    (num_samples,), fixed_value, dtype=torch.long, device=device,
                )
            else:
                weighted_candidates = torch.einsum('svb,bc->svc', candidates, right_environment_next)
                weights = (weighted_candidates * candidates.conj()).sum(dim=2)
                if is_complex:
                    weights = weights.real
                weights = weights.clamp_min(self.mps._numerical_floor)
                conditional_probabilities = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.mps._numerical_floor)
                self._check_valid_probabilities(conditional_probabilities, site=k, context="sample_conditional_scattered")
                chosen = torch.multinomial(conditional_probabilities, 1).squeeze(1)
 
            samples[:, k] = chosen
            gather_indices = chosen.unsqueeze(1).unsqueeze(2).expand(
                num_samples, 1, candidates.shape[2]
            )
            x = candidates.gather(1, gather_indices).squeeze(1)
 
        return samples
