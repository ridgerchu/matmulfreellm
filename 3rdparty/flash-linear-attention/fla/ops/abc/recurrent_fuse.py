# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024, Yu Zhang, Songlin Yang

import torch
import triton
import triton.language as tl

from fla.ops.abc.utils import logcumsumexp_fwd_kernel
from fla.ops.utils import contiguous


@triton.jit
def fused_recurrent_abc_fwd_kernel_K(
    q,
    k,
    pk,
    zk,
    K,
    s,
    s_qh,
    s_qd,
    s_skh,
    s_skm,
    s_Kh,
    s_Kd,
    s_Km,
    T,
    D,
    M,
    BD: tl.constexpr,
    BM: tl.constexpr
):
    i_bh = tl.program_id(0)
    p_q = tl.make_block_ptr(q + i_bh * s_qh, (T * D,), (s_qd,), (0,), (BD,), (0,))
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (T * D,), (s_qd,), (0,), (BD,), (0,))
    p_pk = tl.make_block_ptr(pk + i_bh * s_skh, (T * M,), (s_skm,), (0,), (BM,), (0,))
    p_zk = tl.make_block_ptr(zk + i_bh * s_skh, (T * M,), (s_skm,), (0,), (BM,), (0,))
    p_K = tl.make_block_ptr(K + i_bh * s_Kh, (D, M), (s_Kd, s_Km), (0, 0), (BD, BM), (1, 0))
    p_s = tl.make_block_ptr(s + i_bh * s_skh, (T * M,), (s_skm,), (0,), (BM,), (0,))
    b_pzk, b_K = tl.full([BM,], float('-inf'), dtype=tl.float32), tl.zeros([BD, BM], dtype=tl.float32)

    for _ in range(T):
        b_k, b_pk, b_zk = tl.load(p_k), tl.load(p_pk), tl.load(p_zk)
        # [BM,]
        b_tk, b_pzk = tl.exp(b_pzk - b_zk), b_zk
        # [BD, BM]
        b_K = b_K * b_tk[None, :] + b_pk[None, :] * b_k[:, None]

        # [BD,]
        b_q = tl.load(p_q)
        # [BM,]
        b_s = tl.sum(b_q[:, None] * b_K, 0)

        tl.store(p_s, b_s.to(p_s.dtype.element_ty))

        p_q = tl.advance(p_q, (BD,))
        p_k = tl.advance(p_k, (BD,))
        p_pk = tl.advance(p_pk, (BM,))
        p_zk = tl.advance(p_zk, (BM,))
        p_s = tl.advance(p_s, (BM,))
    tl.store(p_K, b_K.to(p_K.dtype.element_ty))


@triton.jit
def abc_fwd_kernel_o(
    v,
    pv,
    zv,
    V,
    p,
    o,
    s_qh,
    s_qd,
    s_skh,
    s_skm,
    s_Kh,
    s_Kd,
    s_Km,
    T,
    D,
    M,
    BD: tl.constexpr,
    BM: tl.constexpr
):
    i_bh = tl.program_id(0)
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T * D,), (s_qd,), (0,), (BD,), (0,))
    p_pv = tl.make_block_ptr(pv + i_bh * s_skh, (T * M,), (s_skm,), (0,), (BM,), (0,))
    p_zv = tl.make_block_ptr(zv + i_bh * s_skh, (T * M,), (s_skm,), (0,), (BM,), (0,))
    p_V = tl.make_block_ptr(V + i_bh * s_Kh, (D, M), (s_Kd, s_Km), (0, 0), (BD, BM), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_skh, (T * M,), (s_skm,), (0,), (BM,), (0,))
    p_o = tl.make_block_ptr(o + i_bh * s_qh, (T * D,), (s_qd,), (0,), (BD,), (0,))
    b_pzv, b_V = tl.full([BM,], float('-inf'), dtype=tl.float32), tl.zeros([BD, BM], dtype=tl.float32)

    for _ in range(T):
        b_v, b_pv, b_zv = tl.load(p_v), tl.load(p_pv), tl.load(p_zv)
        # [BM,]
        b_tv, b_pzv = tl.exp(b_pzv - b_zv), b_zv
        # [BD, BM]
        b_V = b_V * b_tv[None, :] + b_pv[None, :] * b_v[:, None]

        # [BM,]
        b_p = tl.load(p_p)
        # [BD,]
        b_o = tl.sum(b_p[None, :] * b_V, 1)
        tl.store(p_p, b_p.to(p_p.dtype.element_ty))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty))

        p_v = tl.advance(p_v, (BD,))
        p_pv = tl.advance(p_pv, (BM,))
        p_zv = tl.advance(p_zv, (BM,))
        p_p = tl.advance(p_p, (BM,))
        p_o = tl.advance(p_o, (BD,))
    tl.store(p_V, b_V.to(p_V.dtype.element_ty))


@triton.jit
def abc_bwd_kernel(
    q,
    k,
    v,
    sk,
    sv,
    zk,
    zv,
    K,
    V,
    p,
    o,
    do,
    dq,
    dk,
    dv,
    dsk,
    dsv,
    s_qh,
    s_qd,
    s_skh,
    s_skm,
    s_Kh,
    s_Kd,
    s_Km,
    T,
    D,
    M,
    scale,
    BD: tl.constexpr,
    BM: tl.constexpr
):
    i_bh = tl.program_id(0)
    p_q = tl.make_block_ptr(q + i_bh * s_qh, (T * D,), (s_qd,), ((T - 1) * D,), (BD,), (0,))
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (T * D,), (s_qd,), ((T - 1) * D,), (BD,), (0,))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T * D,), (s_qd,), ((T - 1) * D,), (BD,), (0,))
    p_sk = tl.make_block_ptr(sk + i_bh * s_skh, (T * M,), (s_skm,), ((T - 1) * M,), (BM,), (0,))
    p_sv = tl.make_block_ptr(sv + i_bh * s_skh, (T * M,), (s_skm,), ((T - 1) * M,), (BM,), (0,))
    p_zk = tl.make_block_ptr(zk + i_bh * s_skh, (T * M,), (s_skm,), ((T - 1) * M,), (BM,), (0,))
    p_zv = tl.make_block_ptr(zv + i_bh * s_skh, (T * M,), (s_skm,), ((T - 1) * M,), (BM,), (0,))
    p_K = tl.make_block_ptr(K + i_bh * s_Kh, (D, M), (s_Kd, s_Km), (0, 0), (BD, BM), (1, 0))
    p_V = tl.make_block_ptr(V + i_bh * s_Kh, (D, M), (s_Kd, s_Km), (0, 0), (BD, BM), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_skh, (T * M,), (s_skm,), ((T - 1) * M,), (BM,), (0,))
    p_o = tl.make_block_ptr(o + i_bh * s_qh, (T * D,), (s_qd,), ((T - 1) * D,), (BD,), (0,))
    p_do = tl.make_block_ptr(do + i_bh * s_qh, (T * D,), (s_qd,), ((T - 1) * D,), (BD,), (0,))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qh, (T * D,), (s_qd,), ((T - 1) * D,), (BD,), (0,))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qh, (T * D,), (s_qd,), ((T - 1) * D,), (BD,), (0,))
    p_dv = tl.make_block_ptr(dv + i_bh * s_qh, (T * D,), (s_qd,), ((T - 1) * D,), (BD,), (0,))
    p_dsk = tl.make_block_ptr(dsk + i_bh * s_skh, (T * M,), (s_skm,), ((T - 1) * M,), (BM,), (0,))
    p_dsv = tl.make_block_ptr(dsv + i_bh * s_skh, (T * M,), (s_skm,), ((T - 1) * M,), (BM,), (0,))

    b_K, b_V = tl.load(p_K), tl.load(p_V)
    b_pzk, b_pzv = tl.load(p_zk), tl.load(p_zv)
    b_zdK, b_zdKK = tl.zeros_like(b_K), tl.zeros([BM,], dtype=b_K.dtype)
    b_zdV, b_zdVV = tl.zeros_like(b_V), tl.zeros([BM,], dtype=b_V.dtype)

    for _ in range(T):
        b_o, b_do, b_p, b_q = tl.load(p_o), tl.load(p_do), tl.load(p_p), tl.load(p_q)
        b_k, b_sk, b_zk = tl.load(p_k), tl.load(p_sk), tl.load(p_zk)
        b_v, b_sv, b_zv = tl.load(p_v), tl.load(p_sv), tl.load(p_zv)
        b_K, b_pzk = b_K * tl.exp(b_pzk - b_zk)[None, :], b_zk
        b_V, b_pzv = b_V * tl.exp(b_pzv - b_zv)[None, :], b_zv
        # [BM,]
        b_dp = tl.sum(b_do[:, None] * b_V, 0)
        b_ds = b_p * (b_dp - tl.sum(b_do * b_o, 0)) * scale
        # [BD,]
        b_dq = tl.sum(b_K * b_ds[None, :], 1)

        b_ds = b_ds * tl.exp(-b_zk)
        b_dK = b_q[:, None] * b_ds[None, :]
        b_pk = tl.exp(b_sk)
        b_zdK += b_dK
        b_zdKK += tl.sum(b_dK * b_K, 0)
        b_zdKpk = b_zdK * b_pk[None, :]
        b_dk = tl.sum(b_zdKpk, 1)
        b_dsk = tl.sum(b_k[:, None] * b_zdKpk, 0) - b_pk * b_zdKK

        b_p = b_p * tl.exp(-b_zv)
        b_pv = tl.exp(b_sv)
        b_zdV += b_do[:, None] * b_p[None, :]
        b_zdVV += b_dp * b_p
        b_zdVpv = b_zdV * b_pv[None, :]
        b_dv = tl.sum(b_zdVpv, 1)
        b_dsv = tl.sum(b_v[:, None] * b_zdVpv, 0) - b_pv * b_zdVV

        b_K = b_K - b_k[:, None] * tl.exp(b_sk - b_zk)[None, :]
        b_V = b_V - b_v[:, None] * tl.exp(b_sv - b_zv)[None, :]
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty))
        tl.store(p_dsk, b_dsk.to(p_dsk.dtype.element_ty))
        tl.store(p_dsv, b_dsv.to(p_dsk.dtype.element_ty))

        p_q = tl.advance(p_q, (-BD,))
        p_k = tl.advance(p_k, (-BD,))
        p_v = tl.advance(p_v, (-BD,))
        p_sk = tl.advance(p_sk, (-BM,))
        p_sv = tl.advance(p_sv, (-BM,))
        p_zk = tl.advance(p_zk, (-BM,))
        p_zv = tl.advance(p_zv, (-BM,))
        p_p = tl.advance(p_p, (-BM,))
        p_o = tl.advance(p_o, (-BD,))
        p_do = tl.advance(p_do, (-BD,))
        p_dq = tl.advance(p_dq, (-BD,))
        p_dk = tl.advance(p_dk, (-BD,))
        p_dv = tl.advance(p_dv, (-BD,))
        p_dsk = tl.advance(p_dsk, (-BM,))
        p_dsv = tl.advance(p_dsv, (-BM,))


class ABCAttention(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, s, initial_state, output_final_state):
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 3 if BK <= 64 else 2
        num_warps = 4
        grid = (NK, NV, B * H)
        assert not output_final_state

        def fwd_pre(s, B, H, T, S):
            BT = 64
            NT = triton.cdiv(T, BT)
            # keep cummulative normalizer in fp32
            z = torch.empty_like(s, dtype=torch.float)
            grid = (B * H,)
            logcumsumexp_fwd_kernel[grid](
                s, z,
                s.stride(1), s.stride(2), s.stride(3),
                T=T, S=S, BT=BT, NT=NT,
                num_warps=num_warps,
                num_stages=num_stages
            )
            return z

        z = fwd_pre(s, B, H, T, M)

        zk = torch.empty_like(s)
        fused_recurrent_abc_fwd_kernel_K[grid](
            q, k, zk, o,
            q.stride(1), q.stride(2), q.stride(3), s.stride(1), s.stride(2), s.stride(3),
            T, K, V, M,
            BK=BK, BV=BV, BM=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        p = p.softmax(-1)

        o = torch.empty_like(q)
        abc_fwd_kernel_o[grid](
            v, pv, zv, V, p, o,
            q.stride(1), q.stride(3), sk.stride(1), sk.stride(3), K.stride(1), K.stride(2), K.stride(3),
            T, BD, BM,
            BD=BD, BM=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v, s, z, p, o)
        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, sk, sv, zk, zv, K, V, p, o = ctx.saved_tensors
        grid = ctx.grid
        scale = ctx.scale

        BD, BM = q.shape[-1], sk.shape[-1]
        seq_len, d_head, n_slots = ctx.seq_len, ctx.d_head, ctx.n_slots
        num_stages = 3 if d_head <= 64 else 2
        num_warps = 4

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            do = pad(do, o.shape)
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        dsk, dsv = torch.zeros_like(sk), torch.zeros_like(sk)

        abc_bwd_kernel[grid](
            q, k, v, sk, sv, zk, zv, K, V, p, o, do, dq, dk, dv, dsk, dsv,
            q.stride(1), q.stride(3), sk.stride(1), sk.stride(3), K.stride(1), K.stride(2), K.stride(3),
            seq_len, BD, BM, scale,
            BD=BD, BM=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq[..., :d_head], dk[..., :d_head], dv[..., :d_head], dsk[..., :n_slots], dsv[..., :n_slots]
