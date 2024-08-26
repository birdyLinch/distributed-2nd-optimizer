"""Helper wrappers for supported PyTorch modules."""

from __future__ import annotations

from typing import cast
from typing import List, Tuple

import torch

from kfac.layers.utils import append_bias_ones
from kfac.layers.utils import get_cov
from .default_modules import ModuleHelper

class E3nnLayerModuleHelper(ModuleHelper):
    """ ModuleHelper for e3nn.nn._fc._Layer modules."""

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        """Get shape of A factor.

        A shape = (in_features + int(has_bias), in_features + int(has_bias))
        """
        x = self.module.weight.size(0) # type: ignore
        return (x, x)

    @property
    def g_factor_shape(self) -> tuple[int, int]:
        return (
                self.module.weight.size(1), # type: ignore 
                self.module.weight.size(1), # type: ignore
                )

    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        a = a.view(-1, a.size(-1))
        return get_cov(a)

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        g = g.reshape(-1, g.size(-1))
        return get_cov(g)

