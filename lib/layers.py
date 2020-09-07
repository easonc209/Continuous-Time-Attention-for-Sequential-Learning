# adapted from ODE function implementation: https://github.com/rtqichen/ffjord

import torch
import torch.nn as nn

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

    
class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

    
class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear_v2, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1))


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1)))
