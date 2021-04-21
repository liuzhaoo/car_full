from resnet import resnet18

import torch
from thop import profile
# 增加可读性
from thop import clever_format
inputs = torch.rand(1,3,240, 240)
model = resnet18(num_classes=27249)

flops, params = profile(model, inputs=(inputs, ))
flops, params = clever_format([flops, params], "%.3f")

print('flops:', flops)
print('params:', params)
print('Total params: %.2fM' % (sum(p.numel()
                                       for p in model.parameters()) / 1000000.0))