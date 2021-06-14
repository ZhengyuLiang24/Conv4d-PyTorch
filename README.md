# **Conv4d for PyTorch**


# Introduction
This repository was inspired by the [pytorch-conv4d](https://github.com/timothygebhard/pytorch-conv4d) repository. It consists of an easy-to-use 4-dimensional convolution class (Conv4d) for PyTorch, in which, 4-dimensional convolution is disassembled into a number of official PyTorch 3-dimensional convolutions. It works by performing and stacking several 3D convolutions under proper conditions (see the original repository for a more detailed explanations).

However, the solution of [pytorch-conv4d](https://github.com/timothygebhard/pytorch-conv4d) repository is restricted. Its ***stride*** and ***dilation*** are constrainedly set to 1. And its ***bias*** does not work perfectly. Moreover, the weight initialization does not achieve the uniform distribution in four dimensions. In this repository, some key code has been rewritten to support the arbitrary values for ***stride***, ***dilation*** and ***bias***. The code for weight initialization has also been rewritten to realize Kaiming Initialization.


## Validation
To validate the correctness of this solution, we design 2D and 3D convolutions (validate_2d/validate_3d) equivalent to the official PyTorch version (torch.nn.Conv2d/Conv3d), and compare the outputs of ours and official ones. In our validation, all of values of ***kernel_size***, ***padding***, ***stride*** and ***dilation*** are set to reasonable random integers, and the error less than 1e-5 is considered qualified. You can directly run `validate_2d.py` and `validate_3d.py` to verify again.


## Usage
An example of Conv4d in `main.py`
```python
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
```


## ToDo
This work is not perfect, but it is sufficient for most application for deep learning. The implementation of ***groups*** greater than 1 is still work in progress. And I hope that all ***padding_mode***s can be implemented in the future.


## Contact
Questions or advice for improvements are very welcome!


## Peace & Love
If this work is useful to you, please consider starring it, so that it can reach a broader audience of like-minded people.
