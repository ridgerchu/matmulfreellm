# mypy: disable-error-code="import, no-untyped-def, override"
# ruff: noqa: ANN001, ANN201, ANN202, N803, N806
"""Defines Triton kernels for the log-space RWKV forward and backward passes."""

import math
from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable

EPS = 1e-4


def get_block_size_c(chans: int) -> int:
    if chans < 32:
        return 32
    if chans < 64:
        return 64
    return 256


AUTOTUNE_CONFIGS: list[triton.Config] = [
    triton.Config({"BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"BLOCK_SIZE_C": 128}, num_warps=4),
    triton.Config({"BLOCK_SIZE_C": 1024}, num_warps=8),
]


@triton.jit
def logaddexp(a, b):
    max_ab = tl.maximum(a, b)
    return max_ab + tl.log(tl.exp(a - max_ab) + tl.exp(b - max_ab))


@triton.jit
def logsubexp(a, b, log_eps: tl.constexpr):
    max_ab = tl.maximum(tl.maximum(a, b), log_eps)
    return max_ab + tl.log(tl.exp(a - max_ab) - tl.exp(b - max_ab))


# @triton.autotune(configs=AUTOTUNE_CONFIGS, key=["chans"])
@triton.jit
def wkv_triton_log_space_forward_kernel(
    # W
    w_ptr,
    w_s_c,
    # U
    u_ptr,
    u_s_c,
    # K
    k_ptr,
    k_s_b,
    k_s_t,
    k_s_c,
    # V
    v_ptr,
    v_s_b,
    v_s_t,
    v_s_c,
    # State
    state_ptr,
    state_s_b,
    state_s_abe,
    state_s_c,
    # WKV
    wkv_ptr,
    wkv_s_b,
    wkv_s_t,
    wkv_s_c,
    # Output state
    state_out_ptr,
    state_out_s_b,
    state_out_s_abe,
    state_out_s_t,
    state_out_s_c,
    # Params
    chans,
    tsz,
    eps: tl.constexpr,
    log_eps: tl.constexpr,
    normalize: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize over the batch dimension.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    cs = (c_idx * BLOCK_SIZE_C) + tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    ln_alpha_p_ptr = state_ptr + b_idx * state_s_b
    ln_alpha_m_ptr = state_ptr + b_idx * state_s_b + state_s_abe
    ln_beta_ptr = state_ptr + b_idx * state_s_b + 2 * state_s_abe

    # Pointers to the batch (and possibly channel) for the output tensors.
    wkv_ptr = wkv_ptr + b_idx * wkv_s_b
    ln_alpha_p_out_ptr = state_out_ptr + b_idx * state_out_s_b
    ln_alpha_m_out_ptr = state_out_ptr + b_idx * state_out_s_b + state_out_s_abe
    ln_beta_out_ptr = state_out_ptr + b_idx * state_out_s_b + 2 * state_out_s_abe

    # Loads parameters.
    ln_alpha_p = tl.load(ln_alpha_p_ptr + cs * state_s_c, mask=cmask).to(tl.float32)
    ln_alpha_m = tl.load(ln_alpha_m_ptr + cs * state_s_c, mask=cmask).to(tl.float32)
    ln_beta = tl.load(ln_beta_ptr + cs * state_s_c, mask=cmask).to(tl.float32)
    w = tl.load(w_ptr + cs * w_s_c, mask=cmask).to(tl.float32)
    u = tl.load(u_ptr + cs * u_s_c, mask=cmask).to(tl.float32)

    for t in range(tsz):
        kt = tl.load(k_ptr + t * k_s_t + cs * k_s_c, mask=cmask).to(tl.float32)
        vt = tl.load(v_ptr + t * v_s_t + cs * v_s_c, mask=cmask).to(tl.float32)
        vt_p = tl.maximum(vt, 0) + eps
        vt_m = tl.maximum(-vt, 0) + eps
        ln_v_p = tl.log(vt_p)
        ln_v_m = tl.log(vt_m)

        if normalize:
            ln_alpha_pm = tl.minimum(ln_alpha_p, ln_alpha_m) - eps
            ln_alpha_p = logsubexp(ln_alpha_p, ln_alpha_pm, log_eps)
            ln_alpha_m = logsubexp(ln_alpha_m, ln_alpha_pm, log_eps)

        ln_wkv_p = logaddexp(u + kt + ln_v_p, ln_alpha_p) - logaddexp(u + kt, ln_beta)
        ln_wkv_m = logaddexp(u + kt + ln_v_m, ln_alpha_m) - logaddexp(u + kt, ln_beta)

        wkv = tl.exp(ln_wkv_p) - tl.exp(ln_wkv_m)
        tl.store(wkv_ptr + t * wkv_s_t + cs * wkv_s_c, wkv, mask=cmask)

        ln_alpha_p = logaddexp(w + ln_alpha_p, kt + ln_v_p)
        ln_alpha_m = logaddexp(w + ln_alpha_m, kt + ln_v_m)
        ln_beta = logaddexp(w + ln_beta, kt)
        tl.store(ln_alpha_p_out_ptr + t * state_out_s_t + cs * state_out_s_c, ln_alpha_p, mask=cmask)
        tl.store(ln_alpha_m_out_ptr + t * state_out_s_t + cs * state_out_s_c, ln_alpha_m, mask=cmask)
        tl.store(ln_beta_out_ptr + t * state_out_s_t + cs * state_out_s_c, ln_beta, mask=cmask)


def wkv_triton_log_space_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    eps: float = EPS,
    normalize: bool = False,
) -> tuple[Tensor, Tensor]:
    (bsz, tsz, chans), device = k.shape, k.device

    # Checks tensor shapes.
    assert v.shape == (bsz, tsz, chans), f"{v.shape} != {(bsz, tsz, chans)}"
    assert state.shape == (bsz, 3, 1, chans), f"{state.shape} != {(bsz, 3, 1, chans)}"
    assert w.shape == (chans,), f"{w.shape} != {(chans,)}"
    assert u.shape == (chans,), f"{u.shape} != {(chans,)}"

    # Checks tensor devices.
    for t in (v, state, w, u):
        assert t.device == device, f"{t.device} != {device}"

    # New tensors to output.
    wkvs = k.new_empty(bsz, tsz, chans)
    state_out = k.new_empty(bsz, 3, tsz, chans)

    # Constants.
    block_size_c = get_block_size_c(chans)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (bsz, triton.cdiv(chans, meta["BLOCK_SIZE_C"]))

    wkv_triton_log_space_forward_kernel[grid](
        # W
        w,
        w.stride(0),
        # U
        u,
        u.stride(0),
        # K
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        # V
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        # State
        state,
        state.stride(0),
        state.stride(1),
        state.stride(3),
        # WKV
        wkvs,
        wkvs.stride(0),
        wkvs.stride(1),
        wkvs.stride(2),
        # Output state
        state_out,
        state_out.stride(0),
        state_out.stride(1),
        state_out.stride(2),
        state_out.stride(3),
        # Params
        chans,
        tsz,
        eps=eps,
        log_eps=math.log(eps),
        normalize=normalize,
        BLOCK_SIZE_C=block_size_c,
    )

    state_out = torch.cat((state, state_out), dim=2)

    return wkvs, state_out


# @triton.autotune(configs=AUTOTUNE_CONFIGS, key=["chans"])
@triton.jit
def wkv_triton_log_space_backward_kernel(
    # W
    w_ptr,
    w_s_c,
    # U
    u_ptr,
    u_s_c,
    # K
    k_ptr,
    k_s_b,
    k_s_t,
    k_s_c,
    # V
    v_ptr,
    v_s_b,
    v_s_t,
    v_s_c,
    # State
    state_ptr,
    state_s_b,
    state_s_abe,
    state_s_t,
    state_s_c,
    # WKV grad
    gwkv_ptr,
    gwkv_s_b,
    gwkv_s_t,
    gwkv_s_c,
    # Output state grad
    gstate_out_ptr,
    gstate_out_s_b,
    gstate_out_s_abe,
    gstate_out_s_c,
    # W grad
    gw_ptr,
    gw_s_c,
    # U grad
    gu_ptr,
    gu_s_c,
    # K grad
    gk_ptr,
    gk_s_b,
    gk_s_t,
    gk_s_c,
    # V grad
    gv_ptr,
    gv_s_b,
    gv_s_t,
    gv_s_c,
    # State grad
    gstate_ptr,
    gstate_s_b,
    gstate_s_abe,
    gstate_s_c,
    # Params
    tsz,
    chans,
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize over the batch dimension.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    cs = (c_idx * BLOCK_SIZE_C) + tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_p_ptr = state_ptr + b_idx * state_s_b
    alpha_m_ptr = state_ptr + b_idx * state_s_b + state_s_abe
    beta_ptr = state_ptr + b_idx * state_s_b + 2 * state_s_abe

    # Pointers to the batch (and possibly channel) for the output tensors.
    gk_ptr = gk_ptr + b_idx * gk_s_b
    gv_ptr = gv_ptr + b_idx * gv_s_b

    # Pointers to gradients which were recieved by the function.
    gwkv_ptr = gwkv_ptr + b_idx * gwkv_s_b
    galpha_p_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b
    galpha_m_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b + gstate_out_s_abe
    gbeta_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b + 2 * gstate_out_s_abe

    # Loads parameters.
    gln_alpha_p = tl.load(galpha_p_out_ptr + gstate_out_s_c * cs, mask=cmask).to(tl.float32)
    gln_alpha_m = tl.load(galpha_m_out_ptr + gstate_out_s_c * cs, mask=cmask).to(tl.float32)
    gln_beta = tl.load(gbeta_out_ptr + gstate_out_s_c * cs, mask=cmask).to(tl.float32)
    w = tl.load(w_ptr + w_s_c * cs, mask=cmask).to(tl.float32)
    u = tl.load(u_ptr + u_s_c * cs, mask=cmask).to(tl.float32)

    # Gradient accumulators.
    gw = tl.zeros_like(w)
    gu = tl.zeros_like(u)

    for t in range(tsz):
        tc = tsz - t - 1

        kt = tl.load(k_ptr + tc * k_s_t + k_s_c * cs, mask=cmask).to(tl.float32)
        vt = tl.load(v_ptr + tc * v_s_t + v_s_c * cs, mask=cmask).to(tl.float32)
        vt_p = tl.maximum(vt, 0) + eps
        vt_m = tl.maximum(-vt, 0) + eps
        ln_v_p = tl.log(vt_p)
        ln_v_m = tl.log(vt_m)

        ln_alpha_p_prev = tl.load(alpha_p_ptr + tc * state_s_t + state_s_c * cs, mask=cmask).to(tl.float32)
        ln_alpha_m_prev = tl.load(alpha_m_ptr + tc * state_s_t + state_s_c * cs, mask=cmask).to(tl.float32)
        ln_beta_prev = tl.load(beta_ptr + tc * state_s_t + state_s_c * cs, mask=cmask).to(tl.float32)

        uk = u + kt
        ukv_p = uk + ln_v_p
        ukv_m = uk + ln_v_m

        ukb = logaddexp(uk, ln_beta_prev)
        wkv_p = tl.exp(logaddexp(ukv_p, ln_alpha_p_prev) - ukb)
        wkv_m = tl.exp(logaddexp(ukv_m, ln_alpha_m_prev) - ukb)

        gwkvt = tl.load(gwkv_ptr + tc * gwkv_s_t + gwkv_s_c * cs, mask=cmask).to(tl.float32)
        gln_wkv_p = gwkvt * wkv_p
        gln_wkv_m = gwkvt * -wkv_m

        # Backpropagates wkv gradients.
        e_num_p = tl.exp(ln_alpha_p_prev - ukv_p)
        e_num_m = tl.exp(ln_alpha_m_prev - ukv_m)
        e_den = tl.exp(ln_beta_prev - uk)
        grad_wkv_den_p = gln_wkv_p / (1 + e_den)
        grad_wkv_den_m = gln_wkv_m / (1 + e_den)
        gkv_p = gln_wkv_p / (1 + e_num_p)
        gkv_m = gln_wkv_m / (1 + e_num_m)
        grad_uk = gkv_p + gkv_m - grad_wkv_den_p - grad_wkv_den_m
        gu += grad_uk
        gk = grad_uk
        gv = tl.where(vt > 0, gkv_p / vt_p, gkv_m / -vt_m)

        gln_alpha_wkv_p = gln_wkv_p / (1 + (1 / e_num_p))
        gln_alpha_wkv_m = gln_wkv_m / (1 + (1 / e_num_m))
        gln_beta_wkv = -gln_wkv_p / (1 + (1 / e_den)) - gln_wkv_m / (1 + (1 / e_den))

        # Backpropagates alpha gradients.
        e_alpha_p = tl.exp(kt + ln_v_p - (w + ln_alpha_p_prev))
        e_alpha_m = tl.exp(kt + ln_v_m - (w + ln_alpha_m_prev))
        gwa_p = gln_alpha_p / (1 + e_alpha_p)
        gwa_m = gln_alpha_m / (1 + e_alpha_m)
        gkv_p = gln_alpha_p / (1 + (1 / e_alpha_p))
        gkv_m = gln_alpha_m / (1 + (1 / e_alpha_m))
        gw += gwa_p + gwa_m
        gk += gkv_p + gkv_m
        gv += tl.where(vt > 0, gkv_p / vt_p, -gkv_m / vt_m)

        # Backpropagates beta gradients.
        e_beta = tl.exp(kt - (w + ln_beta_prev))
        gwb = gln_beta / (1 + e_beta)
        gw += gwb
        gk += gln_beta / (1 + (1 / e_beta))

        # Stores the gradients for k and v.
        tl.store(gk_ptr + tc * gk_s_t + gk_s_c * cs, gk, mask=cmask)
        tl.store(gv_ptr + tc * gv_s_t + gv_s_c * cs, gv, mask=cmask)

        # Computes new gradients for alpha and beta.
        gln_alpha_p = gwa_p + gln_alpha_wkv_p
        gln_alpha_m = gwa_m + gln_alpha_wkv_m
        gln_beta = gwb + gln_beta_wkv

    # Stores final gradients for alpha and beta.
    gln_alpha_p_ptr = gstate_ptr + b_idx * gstate_s_b
    gln_alpha_m_ptr = gstate_ptr + b_idx * gstate_s_b + gstate_s_abe
    gln_beta_ptr = gstate_ptr + b_idx * gstate_s_b + 2 * gstate_s_abe
    tl.store(gln_alpha_p_ptr + gstate_s_c * cs, gln_alpha_p, mask=cmask)
    tl.store(gln_alpha_m_ptr + gstate_s_c * cs, gln_alpha_m, mask=cmask)
    tl.store(gln_beta_ptr + gstate_s_c * cs, gln_beta, mask=cmask)

    # Stores final gradients for w and u.
    gw_temp = tl.load(gw_ptr + gw_s_c * cs, mask=cmask).to(tl.float32)
    gw_temp += gw
    tl.store(gw_ptr + gw_s_c * cs, gw_temp, mask=cmask)
    gu_temp = tl.load(gu_ptr + gu_s_c * cs, mask=cmask).to(tl.float32)
    gu_temp += gu
    tl.store(gu_ptr + gu_s_c * cs, gu_temp, mask=cmask)


def wkv_triton_log_space_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
    eps: float = EPS,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    (bsz, tsz, chans), device = k.shape, k.device

    # Checks tensor shapes.
    assert v.shape == (bsz, tsz, chans), f"{v.shape} != {(bsz, tsz, chans)}"
    assert state.shape == (bsz, 3, tsz, chans), f"{state.shape} != {(bsz, 3, tsz, chans)}"
    assert w.shape == (chans,), f"{w.shape} != {(chans,)}"
    assert u.shape == (chans,), f"{u.shape} != {(chans,)}"
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 3, 1, chans)

    # Checks tensor devices.
    for t in (v, state, w, u, grad_wkv, grad_state):
        assert t.device == device, f"{t.device} != {device}"

    # New tensors to output.
    gw = torch.zeros_like(w)
    gu = torch.zeros_like(u)
    gk = torch.empty_like(k)
    gv = torch.empty_like(v)
    gstate = k.new_empty(bsz, 3, 1, chans)

    # Constants.
    block_size_c = get_block_size_c(chans)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (bsz, triton.cdiv(chans, meta["BLOCK_SIZE_C"]))

    wkv_triton_log_space_backward_kernel[grid](
        # W
        w,
        w.stride(0),
        # U
        u,
        u.stride(0),
        # K
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        # V
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        # State
        state,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        # WKV grad
        grad_wkv,
        grad_wkv.stride(0),
        grad_wkv.stride(1),
        grad_wkv.stride(2),
        # Output state grad
        grad_state,
        grad_state.stride(0),
        grad_state.stride(1),
        grad_state.stride(3),
        # W grad
        gw,
        gw.stride(0),
        # U grad
        gu,
        gu.stride(0),
        # K grad
        gk,
        gk.stride(0),
        gk.stride(1),
        gk.stride(2),
        # V grad
        gv,
        gv.stride(0),
        gv.stride(1),
        gv.stride(2),
        # State grad
        gstate,
        gstate.stride(0),
        gstate.stride(1),
        gstate.stride(3),
        # Params
        tsz,
        chans,
        eps=eps,
        BLOCK_SIZE_C=block_size_c,
    )

    return gw, gu, gk, gv, gstate


class WKVLogSpaceTritonFunction(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor,
        eps: float,
        normalize: bool,
    ) -> tuple[Tensor, Tensor]:
        if (
            w.device.type != "cuda"
            or u.device.type != "cuda"
            or k.device.type != "cuda"
            or v.device.type != "cuda"
        ):
            raise ValueError(
                "Calling the CUDA kernel for wkv attention requires all tensors to be on CUDA devices."
            )

        w = -torch.exp(w.float().contiguous())
        if k.dtype == torch.float16:
            u = u.float()
            k = k.float()
            v = v.float()
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        wkv, state_out = wkv_triton_log_space_forward(w, u, k, v, state, eps, normalize)
        ctx.normalize = normalize
        ctx.eps = eps
        ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        gwkv: Tensor,
        gstate: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, None, None]:
        if ctx.normalize:
            raise NotImplementedError("Backward pass for normalized operation is incorrect")
        w, u, k, v, state = cast(tuple[Tensor, ...], ctx.saved_tensors)
        gw, gu, gk, gv, gstate = wkv_triton_log_space_backward(w, u, k, v, state, gwkv, gstate, ctx.eps)
        return gw, gu, gk, gv, gstate, None, None


def wkv_triton_with_eps(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    eps: float = EPS,
    normalize: bool = False,
) -> tuple[Tensor, Tensor]:
    return WKVLogSpaceTritonFunction.apply(w, u, k, v, state, eps, normalize)
