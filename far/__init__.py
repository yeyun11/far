import torch.nn as nn
from .module import FARConv2d, FARLinear


def monkey_patch_(m):
    for n, subm in m._modules.items():
        if isinstance(subm, nn.Conv2d) and (subm.weight.size(2) > 1 or subm.weight.size(3) > 1):
            m._modules[n] = FARConv2d(
                in_channels=subm.in_channels, out_channels=subm.out_channels, 
                kernel_size=subm.kernel_size, stride=subm.stride, padding=subm.padding, 
                groups=subm.groups, bias=True if subm.bias is not None else False, 
                dilation=subm.dilation, padding_mode=subm.padding_mode
            )
            m._modules[n].initialize_by_conv_(subm)
        elif isinstance(subm, nn.Linear) and subm.weight.size(1) > 1:
            m._modules[n] = FARLinear(
                in_features=subm.in_features, 
                out_features=subm.out_features, 
                bias= True if subm.bias is not None else False, 
            )
            m._modules[n].initialize_by_linear_(subm)
        else:
            monkey_patch_(subm)
