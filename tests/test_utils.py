import torch
from mmfreelm.utils import contiguous

def test_contiguous():
    @contiguous
    def dummy_fn(ctx, x):
        return x

    x = torch.randn(10, 10).t()  # Non-contiguous tensor
    result = dummy_fn(None, x)
    assert result.is_contiguous()
