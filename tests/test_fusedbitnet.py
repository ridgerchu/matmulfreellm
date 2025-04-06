import torch
from mmfreelm.ops.fusedbitnet import activation_quant, weight_quant

def test_activation_quant():
    x = torch.tensor([1.0, 2.0, 3.0])
    result = activation_quant(x)
    assert result is not None  # Add more specific assertions based on expected behavior

def test_weight_quant():
    w = torch.tensor([1.0, 2.0, 3.0])
    result = weight_quant(w)
    assert result is not None  # Add more specific assertions based on expected behavior
