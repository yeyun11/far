import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t, Tensor
from typing import Union


class FARLinear(nn.Linear):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True, 
                 device=None, 
                 dtype=None
                 ) -> None:
        super().__init__(
            in_features, 
            out_features, 
            bias, 
            device, 
            dtype
        )
        kernels = self._calculate_kernels(in_features)

        self.register_buffer('kernels', kernels)
        self.weight.data.copy_(self._calculate_weight_by_projection(self.weight.data))
    
    def forward(self, t: Tensor) -> Tensor:
        weight = (self.weight[..., None] * self.kernels).sum(dim=1)
        return F.linear(t, weight, self.bias)

    def _calculate_weight_by_projection(self, linear_weight):
        with torch.no_grad():
            kernel_projections = (linear_weight[:, None, :] * self.kernels[...]).sum(dim=-1)
        return kernel_projections

    def initialize_by_linear_(self, linear) -> None:
        with torch.no_grad():
            self.weight.data.copy_(self._calculate_weight_by_projection(linear.weight.data))
            if self.bias is not None:
                self.bias.data.copy_(linear.bias.data)


    @staticmethod
    def _calculate_kernels(kernel_size):
        kernels = torch.empty(kernel_size, kernel_size)
        with torch.no_grad():
            for i in torch.arange(kernel_size):
                kernels[i] = torch.cos((2 * torch.arange(kernel_size) + 1) * math.pi / (2 * kernel_size) * i) * \
                    math.sqrt((1 if i == 0 else 2) / kernel_size)
        return kernels

class FARConv2d(nn.Conv2d):
    """Following conv def of PyTorch_
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: _size_2_t, 
        stride: _size_2_t = 1, 
        padding: Union[str, _size_2_t] = 0, 
        dilation: _size_2_t = 1, groups: int = 1, 
        bias: bool = True, 
        padding_mode: str = 'zeros', 
        device=None, 
        dtype=None
    ) -> None:
        # initialize as the original conv
        super().__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation, 
            groups, 
            bias, 
            padding_mode, 
            device, 
            dtype
        )

        # calculate kernels
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size        

        kernels = self._calculate_kernels(kernel_h, kernel_w)

        self.register_buffer('kernels', kernels)

        # replace the original conv weight
        self.weight = nn.Parameter(self._calculate_weight_by_projection(self.weight.data))

    def forward(self, t: Tensor) -> Tensor:
        weight = (self.weight * self.kernels).sum(dim=2)
        return self._conv_forward(t, weight, self.bias)
    
    @staticmethod
    def _calculate_kernels(kernel_height, kernel_width):
        kernels = torch.empty(kernel_height * kernel_width, kernel_height, kernel_width)
        kx, ky = torch.meshgrid(torch.arange(kernel_height), torch.arange(kernel_width), indexing='ij')
        ixiy = [(i // kernel_width, i % kernel_width) for i in range(0, kernel_height * kernel_width)]
        ixiy = sorted(ixiy, key=lambda x: sum(x))
        with torch.no_grad():
            for i, (ix, iy) in enumerate(ixiy):
                c = math.sqrt((1 if iy == 0 else 2) / kernel_height) * math.sqrt((1 if ix == 0 else 2) / kernel_width)
                base_kernel = torch.cos((2 * kx + 1) * math.pi / (2 * kernel_width) * ix) * \
                    torch.cos((2 * ky + 1) * math.pi / (2 * kernel_height) * iy)
                kernels[i] = c * base_kernel
        return kernels

    def _calculate_weight_by_projection(self, conv_weight):
        far_weight = torch.empty(
            self.out_channels, 
            self.in_channels // self.groups, 
            self.kernel_size[0] * self.kernel_size[1], 
            1, 
            1
        )    
        with torch.no_grad():
            kernel_projections = (conv_weight.unsqueeze(2) * self.kernels).flatten(3).sum(dim=-1)
            kernel_projections = kernel_projections.view(*kernel_projections.size(), 1, 1)
            far_weight.copy_(kernel_projections)#.permute(2, 0, 1, 3, 4))
        
        return far_weight

    def initialize_by_conv_(self, conv) -> None:
        with torch.no_grad():
            self.weight.data.copy_(self._calculate_weight_by_projection(conv.weight.data))
            if self.bias is not None:
                self.bias.data.copy_(conv.bias.data)
