from Conv4d import Conv4d
import torch

input = torch.randn(2, 16, 5, 5, 32, 32).cuda()

net = Conv4d(16, 32,
             kernel_size=(3, 1, 1, 1),
             padding=(1, 1, 1, 1),
             stride=(1, 1, 1, 1),
             dilation=(1, 1, 1, 1),
             bias=True).cuda()
out = net(input)

print(out.shape)