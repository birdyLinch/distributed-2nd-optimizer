"""Helper wrappers for supported PyTorch modules."""

from __future__ import annotations

from typing import cast
from typing import List, Tuple

import torch

from kfac.layers.utils import append_bias_ones
from kfac.layers.utils import get_cov

from e3nn.util import prod

from torch import fx
from e3nn import o3
from e3nn.o3._tensor_product._instruction import Instruction
from opt_einsum_fx import optimize_einsums_full
from collections import OrderedDict
from math import sqrt

from .default_modules import ModuleHelper


class E3nnTPModuleHelper(ModuleHelper):
    """ ModuleHelper for e3nn.nn._fc._Layer modules."""
    def __init__(self, module: torch.nn.Module):
        """Init ModuleHelper.

        Args:
            module (torch.nn.Module): module in model to wrap.
        """
        self.module = module
        self.cogen_a_factor = codegen_tensor_product_a_factor(
            irreps_in1=self.module.irreps_in1,
            irreps_in2=self.module.irreps_in2,
            irreps_out=self.module.irreps_out,
            shared_weights=self.module.shared_weights,
            instructions=self.module.instructions,
            specialized_code=self.module._specialized_code,
            optimize_einsums=self.module._optimize_einsums
        )
        self.cogen_g_factor = codegen_tensor_product_g_factor(
            irreps_in1=self.module.irreps_in1,
            irreps_in2=self.module.irreps_in2,
            irreps_out=self.module.irreps_out,
            shared_weights=self.module.shared_weights,
            instructions=self.module.instructions,
            specialized_code=self.module._specialized_code,
            optimize_einsums=self.module._optimize_einsums
        )
    

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        """Get shape of A factor.

        A shape = (in_features + int(has_bias), in_features + int(has_bias))
        """
        x = self.a_size_from_instructions(self.module.instructions) # type: ignore
        return (x, x)

    @property
    def g_factor_shape(self) -> tuple[int, int]:
        x = self.g_size_from_instructions(self.module.instructions)
        return (x, x)

    def get_a_factor(self, a: List[torch.Tensor]) -> List[torch.Tensor]:
        with torch.no_grad():
            out = self.cogen_a_factor(*a)
        return out

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.cogen_g_factor(g)
        return out

    def g_size_from_instructions(self, instructions):
        self.out_size = []
        for ins in instructions:
            num_out = ins.path_shape[-1]
            self.out_size.append(num_out)
        return sum(self.out_size)


    def a_size_from_instructions(self, instructions):
        self.in_size = []
        for ins in instructions:
            num_in = prod(ins.path_shape[:-1])
            self.in_size.append(num_in)
        return sum(self.in_size)


def codegen_tensor_product_g_factor(
        irreps_in1: o3.irreps,
        irreps_in2: o3.irreps,
        irreps_out: o3.irreps,
        shared_weights: bool= False,
        instructions: List[Instruction]=None,
        specialized_code: bool = True,
        optimize_einsums: bool = True,
        mode: str = 'sum_pre',
) -> tuple[fx.GraphModule, fx.GraphModule]:
    # TODO:
    #     reshape and extract gs for individual insturction
    #     multiplyer: path weight
    #     sum over m or avg over m???
    graph = fx.Graph()

    tracer = fx.proxy.GraphAppendingTracer(graph)
    constants = OrderedDict()

    assert shared_weights, "shared_weights must be ture if tensor product layer contians internal weights"

    # = function definition =
    g_res =  fx.Proxy(graph.placeholder('g_res', torch.Tensor), tracer=tracer)

    instructions = [ins for ins in instructions if 0 not in ins.path_shape]
    assert len(instructions) > 0, "no instructions, please check network config."

    batch_numel = g_res.shape[0]

    g_res_list = [
            g_res[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps_out.slices(), irreps_out)
        ]

    # fixme: here to concat gg^t
    outputs = []

    for ins in instructions:
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        g_path = g_res_list[ins.i_out].permute(0, 2, 1) * ins.path_weight # cancel the path weight affect

        assert ins.connection_mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv', 'uvu<v', 'u<vw']

        l1l2l3 = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

        if ins.connection_mode == 'uvw':
            if specialized_code and l1l2l3 == (0, 0, 0):
                g = g_path
            elif specialized_code and mul_ir_in1.ir.l == 0:
                g = g_path #/ sqrt(mul_ir_out.ir.dim) # sqrt is divided in aa^t
            elif specialized_code and mul_ir_in2.ir.l == 0:
                g = g_path #/sqrt(mul_ir_out.ir.dim)
            elif specialized_code and mul_ir_out.ir.l == 0:
                g = g_path #/ sqrt(mul_ir_in1.ir.dim) 
            else:
                g = g_path
        else:
            raise NotImplementedError("not supporting yet")


        # aggregate g factor
        if mode == 'avg_pre':
            g = g.reshape(batch_numel, mul_ir_out.ir.dim, -1) 
            g = torch.mean(g, dim=1, keepdim=False)
            ggt = g.T @ (g / batch_numel)
        if mode == 'sum_pre':
            g = g.reshape(batch_numel, mul_ir_out.ir.dim, -1) 
            g = torch.sum(g, dim=1, keepdim=False)
            ggt = g.T @ (g / batch_numel)
        if mode == 'avg_after':
            g = g.reshape(batch_numel * mul_ir_out.ir.dim, -1)
            ggt = g.T @ (g / (batch_numel * mul_ir_out.ir.dim))
        if mode == 'sum_after':
            g = g.reshape(batch_numel * mul_ir_out.ir.dim, -1) 
            ggt = g.T @ (g / batch_numel)
        
        ggt = (ggt.T + ggt) / 2.0 # numerical stability
        outputs += [ggt.node,]

    graph.output(outputs, List[torch.Tensor])

    graph.lint()

    constants_root = torch.nn.Module()
    for key, value in constants.items():
        constants_root.register_buffer(key, value)
    graphmod = fx.GraphModule(constants_root, graph, class_name="tp_afactor_forward")
    if optimize_einsums:
        batchdim = 4
        example_inputs = (
                torch.zeros((batchdim, irreps_out.dim)),
            )
        graphmod = optimize_einsums_full(graphmod, example_inputs)
    
    return graphmod

def codegen_tensor_product_a_factor(
        irreps_in1: o3.irreps,
        irreps_in2: o3.irreps,
        irreps_out: o3.irreps,
        shared_weights: bool=False,
        instructions: List[Instruction]=None,
        specialized_code: bool = True,
        optimize_einsums: bool = True,
        mode: str = 'sum_pre',
) -> tuple[fx.GraphModule, fx.GraphModule]:
    graph = fx.Graph()

    tracer = fx.proxy.GraphAppendingTracer(graph)
    constants = OrderedDict()

    assert shared_weights, "shared_weights must be ture if tensor product layer contians internal weights"

    # = function definition =
    x1s = fx.Proxy(graph.placeholder('x1', torch.Tensor), tracer=tracer)
    x2s = fx.Proxy(graph.placeholder('x2', torch.Tensor), tracer=tracer)

    empty = fx.Proxy(graph.call_function(torch.empty, ((),), dict(device='cpu')), tracer=tracer)
    output_shape = torch.broadcast_tensors(empty.expand(x1s.shape[:-1]), empty.expand(x2s.shape[:-1]))[0].shape
    del empty

    instructions = [ins for ins in instructions if 0 not in ins.path_shape]
    assert len(instructions) > 0, "no instructions, please check network config."

    x1s, x2s = x1s.broadcast_to(output_shape + (-1,)), x2s.broadcast_to(output_shape + (-1,))

    output_shape = output_shape + (irreps_out.dim,)
    x1s = x1s.reshape(-1, irreps_in1.dim)
    x2s = x2s.reshape(-1, irreps_in2.dim)
    batch_numel = x1s.shape[0]
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)

    if len(irreps_in1) == 1:
        x1_list = [x1s.reshape(batch_numel, irreps_in1[0].mul, irreps_in1[0].ir.dim)]
    else:
        x1_list = [
            x1s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps_in1.slices(), irreps_in1)
        ]
    x2_list = []
    if len(irreps_in2) == 1:
        x2_list.append(
             x2s.reshape(batch_numel, irreps_in2[0].mul, irreps_in2[0].ir.dim)
        )
    else:
        for i, mul_ir in zip(irreps_in2.slices(), irreps_in2):
            x2_list.append(
                x2s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
            )
    
    z = ''
    
    xx_dict = dict()

    flat_weight_index = 0

    # fixme: here to concat aa^t
    outputs = []

    for ins in instructions:
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        x1 = x1_list[ins.i_in1]
        x2 = x2_list[ins.i_in2]

        assert ins.connection_mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv', 'uvu<v', 'u<vw']

        # fixme: for each instructions, the aa^t and gg^t is seperate and should be inverted seperatly
        key = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])
        if key not in xx_dict:
            if ins.connection_mode[:2] == 'uu':
                xx_dict[key] = torch.einsum('zui,zuj->zuij', x1, x2)
            else:
                xx_dict[key] = torch.einsum('zui,zvj->zuvij', x1, x2)
        xx = xx_dict[key]
        del key

        # obtain cg coefficents
        w3j_name = f"_w3j_{mul_ir_in1.ir.l}_{mul_ir_in2.ir.l}_{mul_ir_out.ir.l}"
        w3j = fx.Proxy(graph.get_attr(w3j_name), tracer=tracer)

        l1l2l3 = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

        if ins.connection_mode == 'uvw':
            if specialized_code and l1l2l3 == (0, 0, 0):
                a = torch.einsum(f"zu,zv->zuv", x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim))
            elif specialized_code and mul_ir_in1.ir.l == 0:
                a = torch.einsum(f"zu,zvj->zjuv", x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
            elif specialized_code and mul_ir_in2.ir.l == 0:
                a = torch.einsum(f"zui,zv->ziuv", x1, x2.reshape(batch_numel, mul_ir_in2.dim)) /sqrt(mul_ir_out.ir.dim)
            elif specialized_code and mul_ir_out.ir.l == 0:
                a = torch.einsum(f"zui,zvi->zuv", x1, x2) / sqrt(mul_ir_in1.ir.dim) 
            else:
                a = torch.einsum(f"ijk,zuvij->zkuv", w3j, xx)
        else:
            raise NotImplementedError("not supporting yet")

        # aggregate a factor
        if mode == 'avg_pre':
            a = a.reshape(batch_numel, mul_ir_out.ir.dim, -1) 
            a = torch.mean(a, dim=1, keepdim=False)
            aat = a.T @ (a / batch_numel)
        if mode == 'sum_pre':
            a = a.reshape(batch_numel, mul_ir_out.ir.dim, -1) 
            a = torch.sum(a, dim=1, keepdim=False)
            aat = a.T @ (a / batch_numel)
        if mode == 'avg_after':
            a = a.reshape(batch_numel * mul_ir_out.ir.dim, -1)
            aat = a.T @ (a / (batch_numel * mul_ir_out.ir.dim))
        if mode == 'sum_after':
            a = a.reshape(batch_numel * mul_ir_out.ir.dim, -1) 
            aat = a.T @ (a / batch_numel)

        aat = (aat.T + aat) / 2.0
        outputs += [aat.node,]
        
        # remove unused w3j
        if len(w3j.node.users) == 0:
            graph.erase_node(w3j.node)
        else:
            if w3j_name not in constants:
                constants[w3j_name] = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

    graph.output(outputs, List[torch.Tensor])

    graph.lint()

    constants_root = torch.nn.Module()
    for key, value in constants.items():
        constants_root.register_buffer(key, value)
    graphmod = fx.GraphModule(constants_root, graph, class_name="tp_afactor_forward")
    if optimize_einsums:
        batchdim = 4
        example_inputs = (
                torch.zeros((batchdim, irreps_in1.dim)),
                torch.zeros((batchdim, irreps_in2.dim)),
                torch.zeros(
                    1 if shared_weights else batchdim,
                    flat_weight_index,
                ),
            )
        graphmod = optimize_einsums_full(graphmod, example_inputs)
    
    return graphmod

