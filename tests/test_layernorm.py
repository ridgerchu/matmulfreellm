import torch
from mmfreelm.modules.layernorm import layer_norm_ref, rms_norm_ref

def test_layer_norm_ref():
    x = torch.randn(10, 10)
    weight = torch.ones(10)
    bias = torch.zeros(10)
    result = layer_norm_ref(x, weight, bias)
    assert result is not None  # Add more specific assertions based on expected behavior

def test_rms_norm_ref():
    x = torch.randn(10, 10)
    weight = torch.ones(10)
    bias = torch.zeros(10)
    result = rms_norm_ref(x, weight, bias)
    assert result is not None  # Add more specific assertions based on expected behavior
