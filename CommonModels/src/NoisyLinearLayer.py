"""
NoisyLinear Layer as in
https://arxiv.org/pdf/1706.10295
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinearLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(NoisyLinearLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.std_init = 0.1
        self.weight_mu = nn.Parameter(torch.randn(output_size, input_size))
        self.weight_sigma = nn.Parameter(torch.randn(output_size, input_size))
        self.bias_mu = nn.Parameter(torch.randn(output_size))
        self.bias_sigma = nn.Parameter(torch.randn(output_size))
        self.register_buffer('weight_eps', torch.ones_like(self.weight_mu))
        self.register_buffer('bias_eps', torch.ones_like(self.bias_mu))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma * self.weight_eps
        bias = self.bias_mu + self.bias_sigma * self.bias_eps
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.input_size)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.input_size))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.output_size))

    def reset_noise(self):
        eps_in = torch.randn(self.input_size, device=self.weight_eps.device)
        eps_out = torch.randn(self.output_size, device=self.weight_eps.device)

        f_in = eps_in.sign() * eps_in.abs().sqrt()
        f_out = eps_out.sign() * eps_out.abs().sqrt()

        self.weight_eps.copy_(f_out.unsqueeze(1) * f_in.unsqueeze(0))
        self.bias_eps.copy_(f_out)
