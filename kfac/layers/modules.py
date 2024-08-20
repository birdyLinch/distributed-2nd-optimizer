"""Helper wrappers for supported PyTorch modules."""

from __future__ import annotations

from typing import cast
from typing import List, Tuple

import torch

from kfac.layers.utils import append_bias_ones
from kfac.layers.utils import get_cov

from torch.func import grad
from e3nn.util import prod

from torch import fx
from e3nn import o3
from e3nn.o3._tensor_product._instruction import Instruction
from opt_einsum_fx import optimize_einsums_full
from collections import OrderedDict
from math import sqrt


class ModuleHelper:
    """PyTorch module helper.

    This base class provides the interface which the KFACBaseLayer expects
    as input. Namely, the interface provides methods to compute the factors
    of a module, get the shapes of the factors, and get and set the gradients.
    """

    def __init__(self, module: torch.nn.Module):
        """Init ModuleHelper.

        Args:
            module (torch.nn.Module): module in model to wrap.
        """
        self.module = module

    def __repr__(self) -> str:
        """Representation of the ModuleHelper instance."""
        return f'{self.__class__.__name__}({repr(self.module)})'

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        """Get shape of A factor."""
        raise NotImplementedError

    @property
    def g_factor_shape(self) -> tuple[int, int]:
        """Get shape of G factor."""
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Get device that the modules parameters are on."""
        return next(self.module.parameters()).device

    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        """Compute A factor with the input from the forward pass."""
        raise NotImplementedError

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        """Compute G factor with the gradient w.r.t. the output."""
        raise NotImplementedError

    def get_grad(self) -> torch.Tensor:
        """Get formatted gradients (weight and bias) of module.

        Returns:
            gradient of shape If bias != None,
            concats bias.
        """
        g = cast(torch.Tensor, self.module.weight.grad)
        if self.has_bias():
            g = torch.cat(
                [g, self.module.bias.grad.view(-1, 1)],  # type: ignore
                1,
            )
        return g

    def get_bias_grad(self) -> torch.Tensor:
        """Get the gradient of the bias."""
        return cast(torch.Tensor, self.module.bias.grad)

    def get_weight_grad(self) -> torch.Tensor:
        """Get the gradient of the weight."""
        return cast(torch.Tensor, self.module.weight.grad)

    def has_bias(self) -> bool:
        """Check if module has a bias parameter."""
        return hasattr(self.module, 'bias') and self.module.bias is not None

    def has_symmetric_factors(self) -> bool:
        """Check if module has symmetric factors."""
        return True

    def set_grad(self, grad: torch.Tensor) -> None:
        """Update the gradient of the module."""
        if self.has_bias():
            weight_grad = grad[:, :-1].view(self.get_weight_grad().size())
            bias_grad = grad[:, -1:].view(self.get_bias_grad().size())
        else:
            weight_grad = grad.view(self.get_weight_grad().size())

        if self.has_bias():
            self.module.bias.grad = bias_grad.contiguous()
        self.module.weight.grad = weight_grad.contiguous()



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
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            out = self.cogen_a_factor(*a)
        return out

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        #import ipdb; ipdb.set_trace()
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
        mode: str = 'sum_after',
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
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)

    flat_weight_index = 0

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
            notimplementederror("not supporting yet")


        # aggregate g factor
        if mode == 'avg_pre':
            g = g.reshape(batch_numel, mul_ir_out.ir.dim, -1) 
            g = torch.mean(g, dim=1, keepdim=False)
            scale = 1. / batch_numel
            ggt = scale * g.T @ g
        if mode == 'sum_pre':
            g = g.reshape(batch_numel, mul_ir_out.ir.dim, -1) 
            g = torch.sum(g, dim=1, keepdim=False)
            scale = 1. / batch_numel
            ggt = scale * g.T @ g
        if mode == 'avg_after':
            g = g.reshape(batch_numel * mul_ir_out.ir.dim, -1) 
            scale = 1. / (batch_numel * mul_ir_out.ir.dim)
            ggt = scale * g.T @ g
        if mode == 'sum_after':
            g = g.reshape(batch_numel * mul_ir_out.ir.dim, -1) 
            scale = 1. / (batch_numel * mul_ir_out.ir.dim)
            ggt = scale * g.T @ g * mul_ir_out.ir.dim


        outputs += [ggt.node,]
    
    #print(outputs)

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
        mode: str = 'sum_after',
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
            NotImplementedError("not supporting yet")

        # aggregate a factor
        if mode == 'avg_pre':
            a = a.reshape(batch_numel, mul_ir_out.ir.dim, -1) 
            a = torch.mean(a, dim=1, keepdim=False)
            scale = 1. / batch_numel
            aat = scale * a.T @ a
        if mode == 'sum_pre':
            a = a.reshape(batch_numel, mul_ir_out.ir.dim, -1) 
            a = torch.sum(a, dim=1, keepdim=False)
            scale = 1. / batch_numel
            aat = scale * a.T @ a
        if mode == 'avg_after':
            a = a.reshape(batch_numel * mul_ir_out.ir.dim, -1) 
            scale = 1. / (batch_numel * mul_ir_out.ir.dim)
            aat = scale * a.T @ a
        if mode == 'sum_after':
            a = a.reshape(batch_numel * mul_ir_out.ir.dim, -1) 
            scale = 1. / (batch_numel * mul_ir_out.ir.dim)
            aat = scale * a.T @ a * mul_ir_out.ir.dim

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


class LinearModuleHelper(ModuleHelper):
    """ModuleHelper for torch.nn.Linear modules."""

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        """Get shape of A factor.

        A shape = (in_features + int(has_bias), in_features + int(has_bias))
        """
        x = self.module.weight.size(1) + int(self.has_bias())  # type: ignore
        return (x, x)

    @property
    def g_factor_shape(self) -> tuple[int, int]:
        """Get shape of G factor.

        G shape = (out_features, out_features)
        """
        return (
            self.module.weight.size(0),  # type: ignore
            self.module.weight.size(0),  # type: ignore
        )

    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        """Compute A factor with the input from the forward pass.

        Args:
            a (torch.Tensor): tensor with shape batch_size * in_dim.
        """
        a = a.view(-1, a.size(-1))
        if self.has_bias():
            a = append_bias_ones(a)
        return get_cov(a)

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        """Compute G factor with the gradient w.r.t. the output.

        Args:
            g (torch.Tensor): tensor with shape batch_size * out_dim.
        """
        g = g.reshape(-1, g.size(-1))
        return get_cov(g)


class Conv2dModuleHelper(ModuleHelper):
    """ModuleHelper for torch.nn.Conv2d layers."""

    def __init__(self, module: torch.nn.Conv2d):
        """Init ModuleHelper.

        Args:
            module (torch.nn.Conv2d): Conv2d module in model to wrap.
        """
        self.module = module

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        """Get shape of A factor."""
        ksize0: int = self.module.kernel_size[0]  # type: ignore
        ksize1: int = self.module.kernel_size[1]  # type: ignore
        in_ch: int = self.module.in_channels  # type: ignore
        x = in_ch * ksize0 * ksize1 + int(self.has_bias())
        return (x, x)

    @property
    def g_factor_shape(self) -> tuple[int, int]:
        """Get shape of G factor."""
        out_ch: int = self.module.out_channels  # type: ignore
        return (out_ch, out_ch)

    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        """Compute A factor with the input from the forward pass."""
        a = self._extract_patches(a)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if self.has_bias():
            a = append_bias_ones(a)
        a = a / spatial_size
        return get_cov(a)

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        """Compute G factor with the gradient w.r.t. the output.

        Args:
            g (torch.Tensor): tensor with shape batch_size * n_filters *
                out_h * out_w n_filters is actually the output dimension
                (analogous to Linear layer).
        """
        spatial_size = g.size(2) * g.size(3)
        g = g.transpose(1, 2).transpose(2, 3)
        g = g.reshape(-1, g.size(-1))
        g = g / spatial_size
        return get_cov(g)

    def get_grad(self) -> torch.Tensor:
        """Get formmated gradients (weight and bias) of module."""
        grad = cast(
            torch.Tensor,
            self.module.weight.grad.view(  # type: ignore
                self.module.weight.grad.size(0),  # type: ignore
                -1,
            ),
        )
        if self.has_bias():
            grad = torch.cat(
                [grad, self.module.bias.grad.view(-1, 1)],  # type: ignore
                1,
            )
        return grad

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches from convolutional layer.

        Args:
            x (torch.Tensor): input feature maps with shape
                (batch_size, in_c, h, w).

        Returns:
            tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
        """
        padding = cast(List[int], self.module.padding)
        kernel_size = cast(List[int], self.module.kernel_size)
        stride = cast(List[int], self.module.stride)
        if padding[0] + padding[1] > 0:
            x = torch.nn.functional.pad(
                x,
                (padding[1], padding[1], padding[0], padding[0]),
            ).data
        x = x.unfold(2, kernel_size[0], stride[0])
        x = x.unfold(3, kernel_size[1], stride[1])
        x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
        x = x.view(
            x.size(0),
            x.size(1),
            x.size(2),
            x.size(3) * x.size(4) * x.size(5),
        )
        return x
