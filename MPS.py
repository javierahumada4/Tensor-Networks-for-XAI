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
            N: int,
            D: int,
            d: int = 2,
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device] = None,
            init_std: Optional[float] = None,
    ):
        super().__init__()

        assert N >= 2, "N must be >= 2"
        assert D >= 1, "D must be >= 1"
        assert d >= 2, "d must be >= 2"

        assert dtype in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128
        ), f"Unsupported dtype: {dtype}"

        self.N = N
        self.D = D
        self.d = d
        self.dtype = dtype
        self.device = device

        self.A = self.normal_init(init_std)

    def _randn(self, *shape):
        """
        Generates real or complex Gaussian tensors depending on dtype.
        Ensures E[|z|^2] = 1 for complex tensors.
        """
        if self.dtype in (torch.complex64, torch.complex128):
            base = torch.float64 if self.dtype == torch.complex128 else torch.float32
            real = torch.randn(*shape, dtype=base, device=self.device)
            imag = torch.randn(*shape, dtype=base, device=self.device)
            z = (real + 1j * imag) / math.sqrt(2)
            return z.to(self.dtype)
        else:
            return torch.randn(*shape, dtype=self.dtype, device=self.device)


    def normal_init(self, init_std: Optional[float] = None) -> nn.ParameterList:
        if init_std is None:
            init_std = 1.0 / math.sqrt(self.D)

        A_list: List[nn.Parameter] = []

        A0 = self._randn(self.d, self.D) * init_std
        A_list.append(nn.Parameter(A0))

        for _ in range(1, self.N-1):
            Ai = self._randn(self.D, self.d, self.D) * init_std
            A_list.append(nn.Parameter(Ai))

        AN = self._randn(self.D, self.d) * init_std
        A_list.append(nn.Parameter(AN))

        A = nn.ParameterList(A_list)

        return A