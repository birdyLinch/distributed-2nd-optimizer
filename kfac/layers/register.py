"""Utilities for registering PyTorch modules to KFAC layers."""

from __future__ import annotations

import re
from typing import Any

import torch
from .fcn_tools import _Linear

from kfac.layers.base import KFACBaseLayer
from kfac.layers.modules import Conv2dModuleHelper
from kfac.layers.modules import LinearModuleHelper
from kfac.layers.modules import ModuleHelper
from kfac.layers.modules import E3nnLayerModuleHelper
from kfac.layers.modules import E3nnTPModuleHelper
import logging

from e3nn import o3

KNOWN_MODULES = {'linear', 'conv2d', '_Linear'}
LINEAR_TYPES: tuple[type[torch.nn.Module], ...] = (torch.nn.Linear,)
CONV2D_TYPES: tuple[type[torch.nn.Module], ...] = (torch.nn.Conv2d,)
E3NN_LAYER_TYPES: tuple[type[torch.nn.Module], ...] = (o3.FullyConnectedTensorProduct, _Linear)

# TODO: support o3.linear
#       support o3.FullyConnectedTensorProduct
#       support mace.symmetric_contraction.Contract

def get_flattened_modules(
    root: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """Returns flattened view of leaves of module tree."""
    # FIXME: This assumes only leaves contains weights.
    modules = []
    for n, m in root.named_modules():
        all_num_param = len(list(m.parameters()))
        if all_num_param == 0:
            continue
        ch_num_param = 0
        for c in m.children():
            ch_num_param += len(list(c.parameters()))
        if all_num_param > ch_num_param:
            modules.append((n, m))

    return modules


#def get_flattened_modules(
#    root: torch.nn.Module,
#) -> list[tuple[str, torch.nn.Module]]:
#    """Returns flattened view of leaves of module tree."""
#    # FIXME: This assumes only leaves contains weights.
#    return [
#        (name, module)
#        for name, module in root.named_modules()
#        if len(list(module.children())) == 0
#    ]


def requires_grad(module: torch.nn.Module) -> bool:
    """Return False if any module param has requires_grad=False."""
    return all([p.requires_grad for p in module.parameters()])


def get_module_helper(module: torch.nn.Module) -> ModuleHelper | None:
    """Return KFAC module helper that wraps a PyTorch module."""
    if isinstance(module, LINEAR_TYPES):
        return LinearModuleHelper(module)
    elif isinstance(module, CONV2D_TYPES):
        return Conv2dModuleHelper(module)  # type: ignore
    elif isinstance(module, E3NN_LAYER_TYPES):
        if isinstance(module, _Linear):
            return E3nnLayerModuleHelper(module)
        elif isinstance(module, o3.Linear):
            return None
        elif isinstance(module, o3.FullyConnectedTensorProduct):
            return E3nnTPModuleHelper(module)
    else:
        return None


def any_match(query: str, patterns: list[str]) -> bool:
    """Check if a query string matches any pattern in a list.

    Note:
        `search()` is used rather than `match()` so True will be returned
        if there is a match anywhere in the query string.
    """
    regexes = [re.compile(p) for p in patterns]
    return any(regex.search(query) for regex in regexes)


def register_modules(
    model: torch.nn.Module,
    kfac_layer_type: type[KFACBaseLayer],
    skip_layers: list[str],
    **layer_kwargs: Any,
) -> dict[torch.nn.Module, tuple[str, KFACBaseLayer]]:
    """Register supported modules in model with a KFACLayer.

    Args:
        model (torch.nn.Module): model to scan for modules to register.
        kfac_layer_type (type[KFACBaseLayer]): type of subclass of
            KFACBaseLayer to use.
        skip_layers (list[str]): regex patterns that if matched, will cause
            the layer to not be registered. The patterns will be applied
            against the layer's name and class name.
        **layer_kwargs (dict[str, Any]): optional keyword arguments to
            pass to the kfac_layer_type constructor.
    """
    modules = get_flattened_modules(model)

    kfac_layers: dict[torch.nn.Module, tuple[str, KFACBaseLayer]] = {}
    for name, module in modules:
        print(f"{name} --> {type(module)}")
        if (
            not any_match(name, skip_layers)
            and not any_match(module.__class__.__name__, skip_layers)
            and requires_grad(module)
        ):
            module_helper = get_module_helper(module)
            if module_helper is None:
                continue

            kfac_layer = kfac_layer_type(module_helper, **layer_kwargs)

            # get_flattened_modules() should never give us modules with the
            # same name
            assert module not in kfac_layers
            kfac_layers[module] = (name, kfac_layer)

    # Assumes the kfac layers only contains one parameter tensor
    param_kv = {k:v for k,v in model.named_parameters()}
    param_kfac = []
    
    for module, (name, kfac_layer) in kfac_layers.items():
        param = next(iter(module.parameters()))
        for k,v in param_kv.items():
            if v is param:
                param_kfac.append(k)

    set_param_kfac = set(param_kfac)
    set_param_all = set(list(param_kv.keys()))
    set_param_rest = set_param_rest = set_param_all - set_param_kfac

    nparam_tot = 0
    for p in set_param_all:
        nparam_tot += param_kv[p].numel()
    logging.info(f"total number of parameters -> {nparam_tot}")

    # Set the widths for alignment
    num_params_width = 10  # Width for the number of parameters column
    percentage_width = 8   # Width for the percentage column
    
    logging.info(f"kfac parameters:")
    for p in param_kfac:
        num_params = param_kv[p].numel()
        percentage = (num_params / nparam_tot) * 100  # Calculate percentage
        logging.info(f"\t{p:<80} -> {num_params:<{num_params_width}} {percentage:>{percentage_width}.2f}%")
    
    logging.info(f"non-kfac parameters:")
    for p in set_param_rest:
        num_params = param_kv[p].numel()
        percentage = (num_params / nparam_tot) * 100  # Calculate percentage
        logging.info(f"\t{p:<80} -> {num_params:<{num_params_width}} {percentage:>{percentage_width}.2f}%")
    #print(f"kfac parameters: ")
    #for p in param_kfac:
    #    num_params = param_kv[p].numel()
    #    percentage = (num_params / nparam_tot) * 100
    #    print(f"\t{p} -> {num_params} \t {percentage:.2f}%")
    #print(f"non-kfac parameters: ")
    #for p in set_param_rest:
    #    num_params = param_kv[p].numel()
    #    percentage = (num_params / nparam_tot) * 100
    #    print(f"\t{p} -> {num_params} \t {percentage:.2f}%")

    return kfac_layers
