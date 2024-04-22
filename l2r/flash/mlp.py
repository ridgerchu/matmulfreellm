# -*- coding: utf-8 -*-

import torch.nn as nn
from transformers.activations import ACT2FN

from fla.modules.activations import swiglu


class FlashMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        # the final number of params is 4d^2, where d is the hidden size
        # `intermediate_size` is chosen to be (roughly) 2/3 of `hidden_size * hidden_ratio`, and a multiple of 256
        hidden_ratio, multiple_of = 4, 256
        intermediate_size = int(self.hidden_size * hidden_ratio * 2 / 3)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        y = swiglu(gate, y)
        y = self.down_proj(y)
        return y
