# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl


@triton.jit
def chunk_retention_fwd_kernel(
    q,
    k,
    v,
    o,
    b,
    s_qh,
    s_qt,
    s_qd,
    s_oh,
    H,
    T,
    scale,
    BT: tl.constexpr,
    BD: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_b = b + i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.load(p_b)
    d_b, d_o, d_h = tl.math.exp2(BT * b_b), tl.math.exp2(o_i * b_b), tl.math.exp2((BT - o_i) * b_b)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    # [DK, DV]
    b_h = tl.zeros([DK, DV], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_k * DK), (BT, DK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (i_k * DK, i * BT), (DK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_v * DV), (BT, DV), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_oh + i_k * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_v * DV), (BT, DV), (1, 0))

        # [BT, DK]
        b_q = tl.load(p_q)
        b_q = (b_q * scale).to(b_q.dtype)
        # [DK, BT]
        b_k = tl.load(p_k)
        # [BT, DV]
        b_v = tl.load(p_v)

        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        # [BT, DV]
        b_o = tl.dot((b_q * d_o[:, None]).to(b_q.dtype), b_h.to(b_q.dtype), allow_tf32=False)
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        # [DK, DV]
        b_h = d_b * b_h + tl.dot(b_k, (b_v * d_h[:, None]).to(b_k.dtype), allow_tf32=False)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.jit
def chunk_retention_bwd_kernel(
    q,
    k,
    v,
    b,
    do,
    dq,
    dk,
    dv,
    s_qh,
    s_qt,
    s_qd,
    s_dk,
    s_dv,
    H,
    T,
    scale,
    BT: tl.constexpr,
    BD: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_b = b + i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.load(p_b)
    d_b = tl.math.exp2(BT * b_b)
    d_q, d_k = tl.math.exp2(o_i * b_b) * scale, tl.math.exp2((BT - o_i) * b_b)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    [DV, DK]
    b_h = tl.zeros([DV, DK], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_k * DK), (BT, DK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (BD, T), (s_qd, s_qt), (i_v * DV, i * BT), (DV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_v * DV), (BT, DV), (1, 0))
        p_dq = tl.make_block_ptr(dq + i_bh*s_dk + i_v*s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_k * DK), (BT, DK), (1, 0))

        # [BT, DK]
        b_k = tl.load(p_k)
        # [DV, BT]
        b_v = tl.load(p_v)
        # [BT, DV]
        b_do = tl.load(p_do)

        # [BT, BT]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = (b_ds * d_s).to(b_k.dtype)
        # [BT, DK]
        b_dq = tl.dot(b_do, b_h.to(b_k.dtype), allow_tf32=False) * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False)
        # [DV, DK]
        b_h = d_b * b_h + tl.dot((b_v * d_k[None, :]).to(b_k.dtype), b_k, allow_tf32=False)

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty))

    d_s = tl.trans(d_s)
    # [DK, DV]
    b_dh = tl.zeros([DK, DV], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (BD, T), (s_qd, s_qt), (i_k * DK, T - i * BT), (DK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_k * DK), (BT, DK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_v * DV), (BT, DV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_v * DV), (BT, DV), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh*s_dk + i_v*s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_k * DK), (BT, DK), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh*s_dv + i_k*s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_v * DV), (BT, DV), (1, 0))
        # [DK, BT]
        b_q = tl.load(p_q)
        # [BT, DK]
        b_k = tl.load(p_k)
        # [BT, DV]
        b_v = tl.load(p_v)
        b_do = tl.load(p_do)

        # [BT, BT]
        b_ds = (tl.dot(b_v, tl.trans(b_do), allow_tf32=False) * d_s).to(b_k.dtype)

        # [BT, BT]
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * d_s
        # [BT, DK]
        b_dk = tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False) * d_k[:, None]
        b_dk += tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        # [BT, DV]
        b_dv = tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False) * d_k[:, None]
        b_dv += tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        # [DK, DV]
        b_dh = d_b * b_dh + tl.dot(b_q, (b_do * d_q[:, None]).to(b_q.dtype), allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty))


class ChunkRetentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        batch_size, n_heads, seq_len, d_head = q.shape
        scale = d_head ** -0.5
        BD = triton.next_power_of_2(q.shape[-1])
        BT = 32 if BD > 64 else 64
        DK, DV = min(BD, 128), min(BD, 64)
        NK, NV = triton.cdiv(BD, DK), triton.cdiv(BD, DV)
        num_stages = 3 if d_head <= 64 else 2
        num_warps = 4

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            q, k, v = (pad(i, (batch_size, n_heads, seq_len, BD)) for i in (q, k, v))

        o = q.new_empty(batch_size, n_heads, NK * seq_len, BD)
        # NOTE: be careful about BF16 precision
        b = (1. - q.new_tensor(2., dtype=torch.float).pow(-5 - q.new_tensor(range(n_heads), dtype=torch.float))).log2()
        grid = (NV, NK, batch_size * n_heads)
        chunk_retention_fwd_kernel[grid](
            q, k, v, o, b,
            q.stride(1), q.stride(2), q.stride(3), o.stride(1),
            n_heads, seq_len, scale,
            BT=BT, BD=BD, DK=DK, DV=DV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        o = o.view(batch_size, n_heads, NK, seq_len, BD).sum(2)
        ctx.save_for_backward(q, k, v, b)
        ctx.batch_size = batch_size
        ctx.n_heads = n_heads
        ctx.seq_len = seq_len
        ctx.d_head = d_head
        ctx.scale = scale
        return o[..., :d_head]

    @staticmethod
    def backward(ctx, do):
        q, k, v, b = ctx.saved_tensors
        scale = ctx.scale
        BD = triton.next_power_of_2(q.shape[-1])
        BT = 64
        DK, DV = min(BD, 64), min(BD, 128)
        NK, NV = triton.cdiv(BD, DK), triton.cdiv(BD, DV)
        batch_size, n_heads, seq_len, d_head = ctx.batch_size, ctx.n_heads, ctx.seq_len, ctx.d_head
        num_stages = 3 if d_head <= 64 else 2
        num_warps = 4
        assert seq_len % BT == 0, f"seq_len {seq_len} must be divisible by block_size {BT}"

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            do = pad(do, q.shape)

        dq = q.new_empty(batch_size, n_heads, NV * seq_len, BD)
        dk = q.new_empty(batch_size, n_heads, NV * seq_len, BD)
        dv = q.new_empty(batch_size, n_heads, NK * seq_len, BD)

        grid = (NV, NK, batch_size * n_heads)
        chunk_retention_bwd_kernel[grid](
            q, k, v, b, do, dq, dk, dv,
            q.stride(1), q.stride(2), q.stride(3), dk.stride(1), dv.stride(1),
            n_heads, seq_len, scale,
            BT=BT, BD=BD, DK=DK, DV=DV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dq = dq.view(batch_size, n_heads, NV, seq_len, BD).sum(2)
        dk = dk.view(batch_size, n_heads, NV, seq_len, BD).sum(2)
        dv = dv.view(batch_size, n_heads, NK, seq_len, BD).sum(2)
        return dq[..., :d_head], dk[..., :d_head], dv[..., :d_head]


@triton.jit
def flash_retention_fwd_kernel(
    q,
    k,
    v,
    o,
    b,
    s_qh,
    s_qt,
    s_qd,
    H, T,
    scale,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr
):
    i_q, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, 0), (BD, BK), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (0, 0), (BK, BD), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_b = b + i_bh % H

    o_q, o_k = tl.arange(0, BQ) + i_q * BQ, tl.arange(0, BK)
    b_b = tl.load(p_b)
    # [BQ, BD]
    b_q = tl.load(p_q)
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BQ, BD], dtype=tl.float32)
    for _ in range(0, (i_q + 1) * BQ, BK):
        # [BD, BK]
        b_k = tl.load(p_k)
        # [BK, BD]
        b_v = tl.load(p_v)

        # [BQ, BK]
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_q[:, None] - o_k[None, :]) * b_b), 0)
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        # [BQ, BD]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)

        p_k = tl.advance(p_k, (0, BK))
        p_v = tl.advance(p_v, (BK, 0))
        o_k += BK
    tl.store(p_o, b_o.to(p_o.dtype.element_ty))


@triton.jit
def flash_retention_bwd_kernel_dq(
    k,
    v,
    b,
    do,
    dq,
    s_qh,
    s_qt,
    s_qd,
    H, T,
    scale,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr
):
    i_q, i_bh = tl.program_id(0), tl.program_id(1)
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (0, 0), (BK, BD), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, 0), (BD, BK), (0, 1))
    p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_b = b + i_bh % H

    o_q, o_k = tl.arange(0, BQ) + i_q * BQ, tl.arange(0, BK)
    b_b = tl.load(p_b)
    # [BQ, BD]
    b_do = tl.load(p_do)
    # [BQ, BD]
    b_dq = tl.zeros([BQ, BD], dtype=tl.float32)
    for _ in range(0, (i_q + 1) * BQ, BK):
        # [BK, BD]
        b_k = tl.load(p_k)
        # [BD, BK]
        b_v = tl.load(p_v)

        # [BQ, BK]
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_q[:, None] - o_k[None, :]) * b_b), 0) * scale
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_s
        # [BQ, BD]
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)

        p_k = tl.advance(p_k, (BK, 0))
        p_v = tl.advance(p_v, (0, BK))
        o_k += BK
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty))


@triton.jit
def flash_retention_bwd_kernel_dkv(
    q,
    k,
    v,
    b,
    do,
    dk,
    dv,
    s_qh,
    s_qt,
    s_qd,
    H, T,
    scale,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr
):
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_b = b + i_bh % H

    o_q, o_k = tl.arange(0, BQ) + i_k * BK, tl.arange(0, BK) + i_k * BK
    b_b = tl.load(p_b)
    # [BK, BD]
    b_k, b_v = tl.load(p_k), tl.load(p_v)
    # [BK, BD]
    b_dk, b_dv = tl.zeros([BK, BD], dtype=tl.float32), tl.zeros([BK, BD], dtype=tl.float32)
    for i in range(i_k * BK, T, BQ):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i), (BD, BQ), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i), (BD, BQ), (0, 1))

        # [BD, BQ]
        b_q = tl.load(p_q)
        b_do = tl.load(p_do)

        # [BK, BQ]
        m_s = o_k[:, None] <= o_q[None, :]
        d_s = tl.where(m_s, tl.math.exp2((-o_k[:, None] + o_q[None, :]) * b_b.to(tl.float32)), 0) * scale
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * d_s
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * d_s

        # [BK, BD]
        b_dk += tl.dot(b_ds.to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
        b_dv += tl.dot(b_s.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        o_q += BQ
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty))


class FlashRetentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        BQ, BK, BD = 64, 64, triton.next_power_of_2(q.shape[-1])
        batch_size, n_heads, seq_len, d_head = q.shape
        num_stages = 3 if d_head <= 64 else 2
        num_warps = 4
        grid = (triton.cdiv(seq_len, BQ), batch_size * n_heads)
        scale = d_head ** -0.5

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            q, k, v = (pad(i, (batch_size, n_heads, seq_len, BD)) for i in (q, k, v))

        o = torch.empty_like(q)
        # NOTE: be careful about BF16 precision
        b = (1. - q.new_tensor(2., dtype=torch.float).pow(-5 - q.new_tensor(range(n_heads), dtype=torch.float))).log2()
        flash_retention_fwd_kernel[grid](
            q, k, v, o, b,
            q.stride(1), q.stride(2), q.stride(3),
            n_heads, seq_len, scale,
            BQ=BQ, BK=BK, BD=BD,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v, b)
        ctx.batch_size = batch_size
        ctx.n_heads = n_heads
        ctx.seq_len = seq_len
        ctx.d_head = d_head
        ctx.scale = scale
        return o[..., :d_head]

    @staticmethod
    def backward(ctx, do):
        q, k, v, b = ctx.saved_tensors
        scale = ctx.scale
        BD = triton.next_power_of_2(q.shape[-1])
        batch_size, n_heads, seq_len, d_head = ctx.batch_size, ctx.n_heads, ctx.seq_len, ctx.d_head
        num_stages = 3 if d_head <= 64 else 2
        num_warps = 4

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            do = pad(do, q.shape)

        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        BQ, BK = 64, 64
        grid = (triton.cdiv(seq_len, BQ), batch_size * n_heads)
        flash_retention_bwd_kernel_dq[grid](
            k, v, b, do, dq,
            q.stride(1), q.stride(2), q.stride(3),
            n_heads, seq_len, scale,
            BQ=BQ, BK=BK, BD=BD,
            num_warps=num_warps,
            num_stages=num_stages
        )
        BK, BQ = 64, 64
        grid = (triton.cdiv(seq_len, BK), batch_size * n_heads)
        flash_retention_bwd_kernel_dkv[grid](
            q, k, v, b, do, dk, dv,
            q.stride(1), q.stride(2), q.stride(3),
            n_heads, seq_len, scale,
            BQ=BQ, BK=BK, BD=BD,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq[..., :d_head], dk[..., :d_head], dv[..., :d_head]


def naive_retention(q, k, v):
    _, n_heads, seq_len, d_head = q.shape
    s = (1 - q.new_tensor(2., dtype=torch.float).pow(-5. - q.new_tensor(range(n_heads), dtype=torch.float))).log2()
    n = q.new_tensor(range(seq_len), dtype=torch.float)
    n = torch.exp2((n.unsqueeze(-1) - n) * s.view(-1, 1, 1)) * n.unsqueeze(-1).ge(n)
    # GroupNorm is invarinant to input scaling
    s = torch.einsum('bhqd,bhkd,hqk->bhqk', q * d_head ** -0.5, k, n.to(q.dtype))
    o = torch.einsum('bhqk,bhkd->bhqd', s, v)
    return o


chunk_retention = ChunkRetentionFunction.apply
flash_retention = FlashRetentionFunction.apply


if __name__ == '__main__':
    B, H, T, D = 2, 8, 1024, 100
    dtype = torch.float
    torch.manual_seed(42)
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()

    print('Testing...')
    do = torch.randn_like(q)
    ref = naive_retention(q, k, v)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # triton implementation
    tri = chunk_retention(q, k, v)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    assert ref.allclose(tri, 0, 1e-2), breakpoint()
    assert ref_dq.allclose(tri_dq, 0, 1e-2), breakpoint()
    assert ref_dk.allclose(tri_dk, 0, 1e-2), breakpoint()
    assert ref_dv.allclose(tri_dv, 0, 1e-2), breakpoint()
    print('Done!')

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=['seq_len'],
            # different possible values for `x_name`
            x_vals=[128 * 2 ** i for i in range(0, 8)],
            # argument name whose value corresponds to a different line in the plot
            line_arg='provider',
            # possible values for `line_arg``
            line_vals=['torch', 'chunk', 'flash', 'torch_bwd', 'chunk_bwd', 'flash_bwd'],
            # label name for the lines
            line_names=['torch', 'chunk', 'flash', 'torch_bwd', 'chunk_bwd', 'flash_bwd'],
            # line styles
            styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':'), ('yellow', 'dotted'), ('black', 'dashed')],
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
        batch_size, n_heads, d_head = 8, 32, 100

        q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        do = torch.ones_like(q, dtype=dtype)

        quantiles = [0.5, 0.2, 0.8]
        results = 0, 0, 0
        if provider == 'torch':
            if seq_len > 3000:
                return results
            results = triton.testing.do_bench(lambda: naive_retention(q, k, v), quantiles=quantiles)
        elif provider == 'chunk':
            results = triton.testing.do_bench(lambda: chunk_retention(q, k, v), quantiles=quantiles)
        elif provider == 'flash':
            results = triton.testing.do_bench(lambda: flash_retention(q, k, v), quantiles=quantiles)
        elif provider == 'torch_bwd':
            if seq_len > 2000:
                return results
            results = triton.testing.do_bench(lambda: naive_retention(q, k, v).backward(do), quantiles=quantiles)
        elif provider == 'chunk_bwd':
            results = triton.testing.do_bench(lambda: chunk_retention(q, k, v).backward(do), quantiles=quantiles)
        elif provider == 'flash_bwd':
            results = triton.testing.do_bench(lambda: flash_retention(q, k, v).backward(do), quantiles=quantiles)
        return results
    benchmark.run(print_data=True)
