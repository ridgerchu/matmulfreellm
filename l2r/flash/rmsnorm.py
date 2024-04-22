# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from .utils import contiguous


@triton.jit
def rmsnorm_fwd_kernel_r(
    x,
    r,
    eps,
    stride_xb,
    stride_xt,
    stride_xd,
    stride_rb,
    T,
    D,
    BT: tl.constexpr,
    BD: tl.constexpr
):
    i_t, i_b = tl.program_id(0), tl.program_id(1)
    p_r = tl.make_block_ptr(r + i_b * stride_rb, (T,), (stride_xd,), (i_t * BT,), (BT,), (0,))

    # [BT,]
    b_m = tl.zeros([BT,], dtype=tl.float32)
    for i in range(0, D, BD):
        p_x = tl.make_block_ptr(x + i_b * stride_xb, (T, D), (stride_xt, stride_xd), (i_t * BT, i), (BT, BD), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1))
        b_m += tl.sum(tl.math.pow(b_x.to(tl.float32), 2), 1)
    # [BT,]
    b_m = b_m / D
    b_r = tl.math.rsqrt(b_m + eps)

    tl.store(p_r, b_r, boundary_check=(0,))


@triton.jit
def rmsnorm_fwd_kernel_y(
    x,
    z,
    y,
    r,
    w,
    stride_xb,
    stride_xt,
    stride_xd,
    stride_rb,
    T,
    D,
    BT: tl.constexpr,
    BD: tl.constexpr
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_x = tl.make_block_ptr(x + i_b * stride_xb, (T, D), (stride_xt, stride_xd), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    p_z = tl.make_block_ptr(z + i_b * stride_xb, (T, D), (stride_xt, stride_xd), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    p_y = tl.make_block_ptr(y + i_b * stride_xb, (T, D), (stride_xt, stride_xd), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    p_r = tl.make_block_ptr(r + i_b * stride_rb, (T,), (stride_xd,), (i_t * BT,), (BT,), (0,))
    p_w = tl.make_block_ptr(w, (D,), (stride_xd,), (i_d * BD,), (BD,), (0,))

    # [BT,]
    b_r = tl.load(p_r, boundary_check=(0,))
    # [BT, BD]
    b_x = tl.load(p_x, boundary_check=(0, 1))
    # [BD,]
    b_w = tl.load(p_w, boundary_check=(0,))

    b_z = (b_x.to(tl.float32) * b_r[:, None]).to(b_x.dtype)
    tl.store(p_z, b_z, boundary_check=(0, 1))
    tl.store(p_y, b_z * b_w, boundary_check=(0, 1))


@triton.jit
def rmsnorm_bwd_kernel_s(
    z,
    s,
    w,
    dy,
    stride_xb,
    stride_xt,
    stride_xd,
    stride_rb,
    T,
    D,
    BT: tl.constexpr,
    BD: tl.constexpr
):
    i_t, i_b = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_b * stride_rb, (T,), (stride_xd,), (i_t * BT,), (BT,), (0,))

    b_s = tl.zeros([BT,], dtype=tl.float32)
    for i in range(0, D, BD):
        p_z = tl.make_block_ptr(z + i_b * stride_xb, (T, D), (stride_xt, stride_xd), (i_t * BT, i), (BT, BD), (1, 0))
        p_w = tl.make_block_ptr(w, (D,), (stride_xd,), (i,), (BD,), (0,))
        p_dy = tl.make_block_ptr(dy + i_b * stride_xb, (T, D), (stride_xt, stride_xd), (i_t * BT, i), (BT, BD), (1, 0))

        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_w = tl.load(p_w, boundary_check=(0,))
        b_dy = tl.load(p_dy, boundary_check=(0,))
        b_s += tl.sum(b_z * b_dy * b_w[None, :], 1)
    tl.store(p_s, (b_s / D).to(p_s.dtype.element_ty), boundary_check=(0,))


@triton.jit
def rmsnorm_bwd_kernel_d(
    z,
    r,
    s,
    w,
    dy,
    dx,
    dw,
    stride_xb,
    stride_xt,
    stride_xd,
    stride_rb,
    stride_dw,
    T,
    D,
    BT: tl.constexpr,
    BD: tl.constexpr
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_z = tl.make_block_ptr(z + i_b * stride_xb, (T, D), (stride_xt, stride_xd), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    p_r = tl.make_block_ptr(r + i_b * stride_rb, (T,), (stride_xd,), (i_t * BT,), (BT,), (0,))
    p_s = tl.make_block_ptr(s + i_b * stride_rb, (T,), (stride_xd,), (i_t * BT,), (BT,), (0,))
    p_w = tl.make_block_ptr(w, (D,), (stride_xd,), (i_d * BD,), (BD,), (0,))
    p_dy = tl.make_block_ptr(dy + i_b * stride_xb, (T, D), (stride_xt, stride_xd), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    p_dx = tl.make_block_ptr(dx + i_b * stride_xb, (T, D), (stride_xt, stride_xd), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    p_dw = tl.make_block_ptr(dw + i_b * stride_dw, (tl.cdiv(T, BT) * D,), (stride_xd), (i_t * D + i_d * BD,), (BD,), (0,))

    # [BT,]
    b_r = tl.load(p_r, boundary_check=(0,))
    b_s = tl.load(p_s, boundary_check=(0,))
    # [BT, BD]
    b_z = tl.load(p_z, boundary_check=(0, 1))
    # [BD,]
    b_w = tl.load(p_w, boundary_check=(0,))
    # [BT, BD]
    b_dy = tl.load(p_dy, boundary_check=(0, 1))

    # [BT, BD]
    b_dx = (b_dy * b_w[None, :] - b_s[:, None] * b_z) * b_r[:, None]
    # [BD,]
    b_dw = tl.sum(b_dy * b_z, 0)

    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), boundary_check=(0,))


class FlashRMSNormFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, w, eps):
        if not x.is_contiguous():
            raise ValueError("data must be contiguous")
        batch_size, seq_len, hidden_size = x.shape
        BT, BD = 128, 128
        NT, ND = triton.cdiv(seq_len, BT), triton.cdiv(hidden_size, BD)
        r = x.new_empty(batch_size, seq_len, dtype=torch.float)
        grid = (NT, batch_size)
        rmsnorm_fwd_kernel_r[grid](
            x,
            r,
            eps,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            r.stride(0),
            seq_len,
            hidden_size,
            BT=BT,
            BD=BD,
            num_stages=3,
            num_warps=4
        )
        z, y = torch.empty_like(x), torch.empty_like(x)
        grid = (ND, NT, batch_size)
        rmsnorm_fwd_kernel_y[grid](
            x,
            z,
            y,
            r,
            w,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            r.stride(0),
            seq_len,
            hidden_size,
            BT=BT,
            BD=BD,
            num_stages=3,
            num_warps=4
        )
        ctx.save_for_backward(z, r, w)
        return y

    @staticmethod
    @contiguous
    def backward(ctx, dy):
        z, r, w = ctx.saved_tensors
        batch_size, seq_len, hidden_size = z.shape
        BT, BD = 128, 128
        NT, ND = triton.cdiv(seq_len, BT), triton.cdiv(hidden_size, BD)
        s, dx, dw = torch.empty_like(r), torch.empty_like(z), z.new_empty(batch_size, NT, hidden_size)
        grid = (NT, batch_size)
        rmsnorm_bwd_kernel_s[grid](
            z,
            s,
            w,
            dy,
            z.stride(0),
            z.stride(1),
            z.stride(2),
            r.stride(0),
            seq_len,
            hidden_size,
            BT=BT,
            BD=BD,
            num_stages=3,
            num_warps=4
        )
        grid = (ND, NT, batch_size)
        rmsnorm_bwd_kernel_d[grid](
            z,
            r,
            s,
            w,
            dy,
            dx,
            dw,
            z.stride(0),
            z.stride(1),
            z.stride(2),
            r.stride(0),
            dw.stride(0),
            seq_len,
            hidden_size,
            BT=BT,
            BD=BD,
            num_stages=3,
            num_warps=4
        )
        dw = dw.sum((0, 1))
        return dx, dw, None


class FlashRMSNorm(LlamaRMSNorm):
    """
    RMS Normalization layer along the last dimension.

    This is similar to torch.nn.functional.normalize but with eps being added
    instead of max.

    Expects contiguous input of shape (..., dim), and returns normalized data
    of the same shape. For each dim-length vector x, the result has

        x / sqrt( x*x.sum() + eps)

    If weights are included, they are a parameter of length dim which multiplies
    the result.

    This functionality is experimental. Its API might be changed without warnings.
    Use it at your own risk.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        from flash_attn.ops.triton.layernorm import rms_norm_fn
        return rms_norm_fn(x,
                           self.weight,
                           None,
                           residual=None,
                           eps=self.variance_epsilon,
                           prenorm=False,
                           residual_in_fp32=False)


class NaiveRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


if __name__ == "__main__":
    dtype = torch.float
    torch.random.manual_seed(0)
    batch_size, seq_len, hidden_size = 8, 1024, 3200
    naive_rmsnorm = NaiveRMSNorm(hidden_size).to('cuda').train()
    flash_rmsnorm = FlashRMSNorm(hidden_size).to('cuda').train()
    w = torch.randn_like(naive_rmsnorm.weight)
    naive_rmsnorm.weight.data.copy_(w)
    flash_rmsnorm.weight.data.copy_(w)

    if dtype == torch.bfloat16:
        naive_rmsnorm = naive_rmsnorm.bfloat16()
        flash_rmsnorm = flash_rmsnorm.bfloat16()
    if dtype == torch.float:
        naive_rmsnorm = naive_rmsnorm.float()
        flash_rmsnorm = flash_rmsnorm.float()
    if dtype == torch.float16:
        naive_rmsnorm = naive_rmsnorm.half()
        flash_rmsnorm = flash_rmsnorm.half()

    print('Testing')
    x = torch.randn((batch_size, seq_len, hidden_size), device='cuda', dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)
    ref = naive_rmsnorm(x)
    ref.backward(dy)
    ref_dw, naive_rmsnorm.weight.grad = naive_rmsnorm.weight.grad.clone(), None
    ref_dx, x.grad = x.grad.clone(), None

    tri = flash_rmsnorm(x)
    tri.backward(dy)
    tri_dw, flash_rmsnorm.weight.grad = flash_rmsnorm.weight.grad.clone(), None
    tri_dx, x.grad = x.grad.clone(), None

    assert ref.allclose(tri, 0, 1e-2), breakpoint()
    assert ref_dx.allclose(tri_dx, 0, 1e-2), breakpoint()
    assert ref_dw.allclose(tri_dw, 0, 1e-2), breakpoint()
    print('Done!')

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=['seq_len'],
            # different possible values for `x_name`
            x_vals=[128 * 2 ** i for i in range(0, 10)],
            # argument name whose value corresponds to a different line in the plot
            line_arg='provider',
            # possible values for `line_arg``
            line_vals=['naive', 'flash', 'naive_bwd', 'flash_bwd'],
            # label name for the lines
            line_names=['naive', 'flash', 'naive_bwd', 'flash_bwd'],
            # line styles
            styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
            ylabel="Execution Time (ms)",  # label name for the y-axis
            # name for the plot. Used also as a file name for saving the plot.
            plot_name="Performance",
            args={},
        )
    )
    def benchmark(seq_len, provider):
        device = 'cuda'
        dtype = torch.bfloat16
        requires_grad = True
        batch_size, hidden_size = 2, 3200

        naive_rmsnorm = NaiveRMSNorm(hidden_size).to(device)
        flash_rmsnorm = FlashRMSNorm(hidden_size).to(device)
        if dtype == torch.bfloat16:
            naive_rmsnorm = naive_rmsnorm.bfloat16()
            flash_rmsnorm = flash_rmsnorm.bfloat16()
        if dtype == torch.float:
            naive_rmsnorm = naive_rmsnorm.float()
            flash_rmsnorm = flash_rmsnorm.float()
        if dtype == torch.float16:
            naive_rmsnorm = naive_rmsnorm.half()
            flash_rmsnorm = flash_rmsnorm.half()

        x = torch.ones(batch_size, seq_len, hidden_size, requires_grad=requires_grad, dtype=dtype, device=device)
        dy = torch.randn_like(x)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'naive':
            results = triton.testing.do_bench(lambda: naive_rmsnorm(x), quantiles=quantiles)
        elif provider == 'flash':
            results = triton.testing.do_bench(lambda: flash_rmsnorm(x), quantiles=quantiles)
        elif provider == 'naive_bwd':
            results = triton.testing.do_bench(lambda: naive_rmsnorm(x).backward(dy), quantiles=quantiles)
        elif provider == 'flash_bwd':
            results = triton.testing.do_bench(lambda: flash_rmsnorm(x).backward(dy), quantiles=quantiles)
        return results
    benchmark.run(print_data=True)
