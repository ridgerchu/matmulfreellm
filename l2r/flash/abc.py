# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl

from .utils import contiguous


@triton.jit
def chunk_abc_fwd_kernel_s(
    q,
    k,
    s,
    ek,
    zk,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DM: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BD, DM]
    b_hk = tl.zeros([BD, DM], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i * BT), (BD, BT), (0, 1))
        p_s = tl.make_block_ptr(s + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_zk = tl.make_block_ptr(zk + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))

        # [BT, BD]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BD, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, DM]
        b_ek = tl.load(p_ek, boundary_check=(0, 1))
        b_zk = tl.load(p_zk, boundary_check=(0, 1))

        # [BT, DM]
        b_s = tl.dot(b_q, b_hk.to(b_q.dtype), allow_tf32=False)
        b_s += tl.dot(tl.where(m_s, tl.dot(b_q, b_k, allow_tf32=False), 0).to(b_q.dtype), b_ek, allow_tf32=False)
        b_s = b_s / b_zk
        # [BD, DM]
        b_hk += tl.dot(b_k, b_ek, allow_tf32=False)

        tl.store(p_s, b_s.to(p_s.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_fwd_kernel_o(
    v,
    o,
    p,
    ev,
    zv,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DV: tl.constexpr
):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BM, DV]
    b_hv = tl.zeros([BM, DV], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_v * DV), (BT, DV), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_v * DV), (BT, DV), (1, 0))
        p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, 0), (BT, BM), (1, 0))
        p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, i * BT), (BM, BT), (0, 1))
        p_zv = tl.make_block_ptr(zv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, 0), (BT, BM), (1, 0))

        # [BT, DV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BM]
        b_p = tl.load(p_p, boundary_check=(0, 1))
        # [BM, BT]
        b_ev = tl.load(p_ev, boundary_check=(0, 1))
        # [BT, BM]
        b_zv = tl.load(p_zv, boundary_check=(0, 1))

        b_p = (b_p / b_zv).to(b_v.dtype)
        # [BT, DV]
        b_o = tl.dot(b_p, b_hv.to(b_v.dtype), allow_tf32=False)
        b_o += tl.dot(tl.where(m_s, tl.dot(b_p, b_ev, allow_tf32=False), 0).to(b_v.dtype), b_v, allow_tf32=False)
        # [BM, DV]
        b_hv += tl.dot(b_ev, b_v, allow_tf32=False)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dsp(
    v,
    p,
    ev,
    zk,
    zv,
    do,
    doo,
    ds,
    dp,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    scale,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DM: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BD, DM]
    b_hv = tl.zeros([BD, DM], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i * BT), (BD, BT), (0, 1))
        p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_zk = tl.make_block_ptr(zk + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_zv = tl.make_block_ptr(zv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        p_doo = tl.make_block_ptr(doo + i_bh * T, (T,), (s_qd,), (i * BT,), (BT,), (0,))
        p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))
        p_dp = tl.make_block_ptr(dp + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, i_m * DM), (BT, DM), (1, 0))

        # [BD, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, DM]
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_ev = tl.load(p_ev, boundary_check=(0, 1))
        b_zk = tl.load(p_zk, boundary_check=(0, 1))
        b_zv = tl.load(p_zv, boundary_check=(0, 1))
        # [BT, BD]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BT,]
        b_doo = tl.load(p_doo, boundary_check=(0,))

        # [BT, DM]
        b_dp = tl.dot(b_do, b_hv.to(b_do.dtype), allow_tf32=False)
        b_dp += tl.dot(tl.where(m_s, tl.dot(b_do, b_v, allow_tf32=False), 0).to(b_v.dtype), b_ev, allow_tf32=False)
        b_dp = (b_dp / b_zv).to(b_p.dtype)
        b_ds = (b_p / b_zk * (b_dp - b_doo[:, None]) * scale).to(b_p.dtype)
        # [BD, DM]
        b_hv += tl.dot(b_v, b_ev, allow_tf32=False)

        tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_ds, b_ds.to(p_ds.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dq(
    k,
    ek,
    dq,
    ds,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DK: tl.constexpr
):
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BM, DK]
    b_hk = tl.zeros([BM, DK], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_k * DK), (BT, DK), (1, 0))
        p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, i * BT), (BM, BT), (0, 1))
        p_dq = tl.make_block_ptr(dq + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, i_k * DK), (BT, DK), (1, 0))
        p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i * BT, 0), (BT, BM), (1, 0))

        # [BT, DK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BM, BT]
        b_ek = tl.load(p_ek, boundary_check=(0, 1))
        # [BT, BM]
        b_ds = tl.load(p_ds, boundary_check=(0, 1))

        # [BT, DK]
        b_dq = tl.dot(b_ds, b_hk.to(b_k.dtype), allow_tf32=False)
        b_dq += tl.dot(tl.where(m_s, tl.dot(b_ds, b_ek, allow_tf32=False), 0).to(b_k.dtype), b_k, allow_tf32=False)
        # [BM, DK]
        b_hk += tl.dot(b_ek, b_k, allow_tf32=False)

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dkv(
    q,
    p,
    ek,
    ev,
    do,
    ds,
    dk,
    dv,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DK: tl.constexpr
):
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] <= o_i[None, :]

    # [BM, DK]
    b_dhk = tl.zeros([BM, DK], dtype=tl.float32)
    b_dhv = tl.zeros([BM, DK], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_k * DK), (BT, DK), (1, 0))
        p_p = tl.make_block_ptr(p + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, T - i * BT), (BM, BT), (0, 1))
        p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, 0), (BT, BM), (1, 0))
        p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, 0), (BT, BM), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_k * DK), (BT, DK), (1, 0))
        p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, T - i * BT), (BM, BT), (0, 1))
        p_dk = tl.make_block_ptr(dk + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_k * DK), (BT, DK), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, i_k * DK), (BT, DK), (1, 0))

        # [BT, BD]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BD]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BT, BM]
        b_ek = tl.load(p_ek, boundary_check=(0, 1))
        b_ev = tl.load(p_ev, boundary_check=(0, 1))
        # [BM, BT]
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_ds = tl.load(p_ds, boundary_check=(0, 1))

        # [BT, BD]
        b_dk = tl.dot(b_ek, b_dhk.to(b_ek.dtype), allow_tf32=False)
        b_dk += tl.dot(tl.where(m_s, tl.dot(b_ek, b_ds, allow_tf32=False), 0.).to(b_do.dtype), b_q, allow_tf32=False)

        # [BT, BD]
        b_dv = tl.dot(b_ev, b_dhv.to(b_ev.dtype), allow_tf32=False)
        b_dv += tl.dot(tl.where(m_s, tl.dot(b_ev, b_p, allow_tf32=False), 0.).to(b_do.dtype), b_do, allow_tf32=False)
        # [BM, DK]
        b_dhk += tl.dot(b_ds, b_q, allow_tf32=False)
        b_dhv += tl.dot(b_p, b_do, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dskv(
    q,
    k,
    v,
    s,
    p,
    ek,
    ev,
    do,
    ds,
    dp,
    dsk,
    dsv,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    DM: tl.constexpr
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # [BT, BT]
    m_s = o_i[:, None] <= o_i[None, :]

    # [BD, DM]
    b_dhk = tl.zeros([BD, DM], dtype=tl.float32)
    b_dhv = tl.zeros([BD, DM], dtype=tl.float32)
    # [DM,]
    b_zdss, b_zdpp = tl.zeros([DM,], dtype=tl.float32), tl.zeros([DM,], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, T - i * BT), (BD, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, 0), (BT, BD), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (T - i * BT, 0), (BT, BD), (1, 0))
        p_s = tl.make_block_ptr(s + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, T - i * BT), (BD, BT), (0, 1))
        p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_dp = tl.make_block_ptr(dp + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_dsk = tl.make_block_ptr(dsk + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))
        p_dsv = tl.make_block_ptr(dsv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (T - i * BT, i_m * DM), (BT, DM), (1, 0))

        # [BT, BD]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # [BD, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BT, DM]
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_ek = tl.load(p_ek, boundary_check=(0, 1))
        b_ev = tl.load(p_ev, boundary_check=(0, 1))
        b_ds = tl.load(p_ds, boundary_check=(0, 1))
        b_dp = tl.load(p_dp, boundary_check=(0, 1))

        # [BT, DM]
        b_dss = b_ds * b_s
        # [BT, BT]
        b_dsk = tl.dot(tl.where(m_s, tl.dot(b_k, b_q, allow_tf32=False), 0.).to(b_k.dtype), b_ds, allow_tf32=False)
        b_dsk -= tl.dot(tl.where(m_s, 1., 0.).to(b_k.dtype), b_dss.to(b_k.dtype), allow_tf32=False)
        b_dsk += tl.dot(b_k, b_dhk.to(b_k.dtype), allow_tf32=False) - b_zdss[None, :]

        # [BT, DM]
        b_dpp = b_dp * b_p
        # [BT, BT]
        b_dsv = tl.dot(tl.where(m_s, tl.dot(b_v, b_do, allow_tf32=False), 0.).to(b_v.dtype), b_p, allow_tf32=False)
        b_dsv -= tl.dot(tl.where(m_s, 1., 0.).to(b_v.dtype), b_dpp.to(b_v.dtype), allow_tf32=False)
        b_dsv += tl.dot(b_v, b_dhv.to(b_v.dtype), allow_tf32=False) - b_zdpp[None, :]
        # [BD, DM]
        b_dhk += tl.dot(b_q, b_ds, allow_tf32=False)
        b_dhv += tl.dot(b_do, b_p, allow_tf32=False)
        # [DM,]
        b_zdss += tl.sum(b_dss, 0)
        b_zdpp += tl.sum(b_dpp, 0)

        tl.store(p_dsk, (b_ek * b_dsk).to(p_dsk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dsv, (b_ev * b_dsv).to(p_dsv.dtype.element_ty), boundary_check=(0, 1))


class ChunkABCAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, sk, sv):
        batch_size, n_heads, seq_len, d_head, n_slots = *q.shape, sk.shape[-1]
        scale = d_head ** -0.5
        BT, BD, BM = 32, triton.next_power_of_2(q.shape[-1]), triton.next_power_of_2(sk.shape[-1])
        DV, DM = min(BD, 32), min(BM, 32)
        NV, NM = triton.cdiv(BD, DV), triton.cdiv(BM, DM)
        num_stages = 1
        num_warps = 2

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            q, k, v = (pad(i, (batch_size, n_heads, seq_len, BD)) for i in (q, k, v))
        o = torch.empty_like(q)
        s = torch.empty_like(sk)
        sk, sv = sk.float(), sv.float()
        zk, zv = sk.logcumsumexp(2), sv.logcumsumexp(2)
        ek, ev = (sk - zk[:, :, -1:]).exp().to(q.dtype), (sv - zv[:, :, -1:]).exp().to(q.dtype)
        zk, zv = (zk - zk[:, :, -1:]).exp().to(q.dtype), (zv - zv[:, :, -1:]).exp().to(q.dtype)

        grid = (NM, batch_size * n_heads)
        chunk_abc_fwd_kernel_s[grid](
            q * scale, k, s, ek, zk,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BT=BT, BD=BD, BM=BM, DM=DM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        p = s.softmax(-1, dtype=torch.float).to(q.dtype)
        grid = (NV, batch_size * n_heads)
        chunk_abc_fwd_kernel_o[grid](
            v, o, p, ev, zv,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BT=BT, BD=BD, BM=BM, DV=DV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v, o, s, p, ek, ev, zk, zv)
        ctx.batch_size = batch_size
        ctx.n_heads = n_heads
        ctx.seq_len = seq_len
        ctx.d_head = d_head
        ctx.n_slots = n_slots
        ctx.dtype = q.dtype
        ctx.scale = scale
        return o[..., :d_head]

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, o, s, p, ek, ev, zk, zv = ctx.saved_tensors
        BT, BD, BM = 16, q.shape[-1], ek.shape[-1]
        DK, DM = min(BD, 64), min(BM, 32)
        NK, NM = triton.cdiv(BD, DK), triton.cdiv(BM, DM)
        batch_size, n_heads, seq_len, d_head, n_slots = ctx.batch_size, ctx.n_heads, ctx.seq_len, ctx.d_head, ctx.n_slots
        scale = ctx.scale
        num_stages = 1
        num_warps = 4

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            do = pad(do, o.shape)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        ds, dp, dsk, dsv = torch.empty_like(ek), torch.empty_like(ek), torch.empty_like(ek), torch.empty_like(ek)

        doo = (o * do).sum(-1)
        grid = (NM, batch_size * n_heads)
        chunk_abc_bwd_kernel_dsp[grid](
            v, p, ev, zk, zv, do, doo, ds, dp,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len, scale,
            BT=BT, BD=BD, BM=BM, DM=DM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        grid = (NK, batch_size * n_heads)
        chunk_abc_bwd_kernel_dq[grid](
            k, ek, dq, ds,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BT=BT, BD=BD, BM=BM, DK=DK,
            num_warps=num_warps,
            num_stages=num_stages
        )
        s, p = s / scale, p / zv
        chunk_abc_bwd_kernel_dkv[grid](
            q, p, ek, ev, do, ds, dk, dv,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BT=BT, BD=BD, BM=BM, DK=DK,
            num_warps=num_warps,
            num_stages=num_stages
        )
        grid = (NM, batch_size * n_heads)
        chunk_abc_bwd_kernel_dskv[grid](
            q, k, v, s, p, ek, ev, do, ds, dp, dsk, dsv,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BT=BT, BD=BD, BM=BM, DM=DM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq[..., :d_head], dk[..., :d_head], dv[..., :d_head], dsk[..., :n_slots], dsv[..., :n_slots]


@triton.jit
def flash_abc_fwd_kernel(
    q,
    k,
    v,
    ek,
    ev,
    zk,
    zv,
    s,
    p,
    o,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr
):
    i_q, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, 0), (BD, BK), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (0, 0), (BK, BD), (1, 0))
    p_s = tl.make_block_ptr(s + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (T, BM), (s_skt, s_skm), (0, 0), (BK, BM), (1, 0))
    p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, 0), (BM, BK), (0, 1))
    p_zk = tl.make_block_ptr(zk + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_zv = tl.make_block_ptr(zv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))

    o_q, o_k = tl.arange(0, BQ) + i_q * BQ, tl.arange(0, BK)

    # [BQ, BD]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_o = tl.zeros([BQ, BD], dtype=tl.float32)
    # [BQ, BM]
    b_s = tl.zeros([BQ, BM], dtype=tl.float32)
    for i in range(0, (i_q + 1) * BQ, BK):
        # [BD, BK]
        b_k = tl.load(p_k)
        # [BK, BM]
        b_ek = tl.load(p_ek)

        # [BQ, BK]
        m_qk = o_q[:, None] >= (i + o_k[None, :])
        b_qk = tl.dot(b_q, b_k, allow_tf32=False)
        b_qk = tl.where(m_qk, b_qk, 0).to(b_q.dtype)
        # [BQ, BM]
        b_s += tl.dot(b_qk, b_ek, allow_tf32=False)

        p_k = tl.advance(p_k, (0, BK))
        p_ek = tl.advance(p_ek, (BK, 0))
    b_s = b_s / tl.load(p_zk)
    b_z = tl.exp(b_s - tl.max(b_s, 1)[:, None])
    b_p = tl.fdiv(b_z, tl.sum(b_z, 1)[:, None])

    tl.store(p_s, b_s.to(p_s.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_p, b_p.to(p_p.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    # [BQ, BM]
    b_p = (b_p / tl.load(p_zv, boundary_check=(0, 1))).to(b_q.dtype)
    for i in range(0, (i_q + 1) * BQ, BK):
        # [BK, BD]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BM, BK]
        b_ev = tl.load(p_ev, boundary_check=(0, 1))

        # [BQ, BK]
        m_qk = o_q[:, None] >= (i + o_k[None, :])
        # [BQ, BD]
        b_o += tl.dot(tl.where(m_qk, tl.dot(b_p, b_ev, allow_tf32=False), 0).to(b_v.dtype), b_v, allow_tf32=False)

        p_v = tl.advance(p_v, (BK, 0))
        p_ev = tl.advance(p_ev, (0, BK))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_abc_bwd_kernel_dqsp(
    k,
    v,
    ek,
    ev,
    zv,
    p,
    do,
    doo,
    dq,
    ds,
    dp,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    scale,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr
):
    i_q, i_bh = tl.program_id(0), tl.program_id(1)
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, 0), (BD, BK), (0, 1))
    p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (T, BM), (s_skt, s_skm), (0, 0), (BK, BM), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))

    o_q, o_k = tl.arange(0, BQ) + i_q * BQ, tl.arange(0, BK)

    # [BQ, BD]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BQ, BM]
    b_dp = tl.zeros([BQ, BM], dtype=tl.float32)
    for i in range(0, (i_q + 1) * BQ, BK):
        # [BD, BK]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BM]
        b_ev = tl.load(p_ev, boundary_check=(0, 1))

        # [BQ, BK]
        m_qk = o_q[:, None] >= (i + o_k[None, :])
        # [BQ, BM]
        b_dp += tl.dot(tl.where(m_qk, tl.dot(b_do, b_v, allow_tf32=False), 0).to(b_v.dtype), b_ev, allow_tf32=False)

        p_v = tl.advance(p_v, (0, BK))
        p_ev = tl.advance(p_ev, (BK, 0))

    tl.debug_barrier()

    p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (0, 0), (BK, BD), (1, 0))
    p_zv = tl.make_block_ptr(zv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (BM, T), (s_skm, s_skt), (0, 0), (BM, BK), (0, 1))
    p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_doo = tl.make_block_ptr(doo + i_bh * T, (T,), (s_qd,), (i_q * BQ,), (BQ,), (0,))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_q * BQ, 0), (BQ, BD), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))
    p_dp = tl.make_block_ptr(dp + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_q * BQ, 0), (BQ, BM), (1, 0))

    # [BQ, BM]
    b_p = tl.load(p_p, boundary_check=(0, 1))
    b_zv = tl.load(p_zv, boundary_check=(0, 1))
    b_dp = (b_dp / b_zv).to(b_p.dtype)
    # [BQ,]
    b_doo = tl.load(p_doo, boundary_check=(0,))
    # [BQ, BM]
    b_ds = (b_p * (b_dp - b_doo[:, None]) * scale).to(b_p.dtype)
    # [BQ, BD]
    b_dq = tl.zeros([BQ, BD], dtype=tl.float32)
    for i in range(0, (i_q + 1) * BQ, BK):
        # [BK, BD]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BM, BK]
        b_ek = tl.load(p_ek, boundary_check=(0, 1))

        # [BQ, BK]
        m_qk = o_q[:, None] >= (i + o_k[None, :])
        # [BQ, BD]
        b_dq += tl.dot(tl.where(m_qk, tl.dot(b_ds, b_ek, allow_tf32=False), 0).to(b_k.dtype), b_k, allow_tf32=False)

        p_k = tl.advance(p_k, (BK, 0))
        p_ek = tl.advance(p_ek, (0, BK))
    tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_ds, b_ds.to(p_ds.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_abc_bwd_kernel_dkv(
    q,
    k,
    v,
    ek,
    ev,
    s,
    p,
    do,
    ds,
    dp,
    dk,
    dv,
    dsk,
    dsv,
    s_qh,
    s_qt,
    s_qd,
    s_skh,
    s_skt,
    s_skm,
    T,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr
):
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    p_k = tl.make_block_ptr(k + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_ek = tl.make_block_ptr(ek + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_k * BK, 0), (BK, BM), (1, 0))
    p_ev = tl.make_block_ptr(ev + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_k * BK, 0), (BK, BM), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i_k * BK, 0), (BK, BD), (1, 0))
    p_dsk = tl.make_block_ptr(dsk + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_k * BK, 0), (BK, BM), (1, 0))
    p_dsv = tl.make_block_ptr(dsv + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i_k * BK, 0), (BK, BM), (1, 0))

    o_q, o_k = tl.arange(0, BQ), tl.arange(0, BK) + i_k * BK

    # [BK, BD]
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    # [BK, BM]
    b_ek, b_ev = tl.load(p_ek, boundary_check=(0, 1)), tl.load(p_ev, boundary_check=(0, 1))
    # [BK, BD]
    b_dk, b_dv = tl.zeros([BK, BD], dtype=tl.float32), tl.zeros([BK, BD], dtype=tl.float32)
    # [BK, BM]
    b_dsk, b_dsv = tl.zeros([BK, BM], dtype=tl.float32), tl.zeros([BK, BM], dtype=tl.float32)

    for i in range(i_k * BK, T, BQ):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i, 0), (BQ, BD), (1, 0))
        p_s = tl.make_block_ptr(s + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i, 0), (BQ, BM), (1, 0))
        p_p = tl.make_block_ptr(p + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i, 0), (BQ, BM), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i, 0), (BQ, BD), (1, 0))
        p_ds = tl.make_block_ptr(ds + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i, 0), (BQ, BM), (1, 0))
        p_dp = tl.make_block_ptr(dp + i_bh * s_skh, (T, BM), (s_skt, s_skm), (i, 0), (BQ, BM), (1, 0))

        # [BQ, BD]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BQ, BM]
        b_p = tl.load(p_p, boundary_check=(0, 1))
        # [BQ, BD]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BQ, BM]
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_ds = tl.load(p_ds, boundary_check=(0, 1))
        b_dp = tl.load(p_dp, boundary_check=(0, 1))

        # [BK, BQ]
        m_kq = o_k[:, None] <= (o_q + i)[None, :]

        bed = tl.where(m_kq, tl.dot(b_ek, tl.trans(b_ds), allow_tf32=False), 0.).to(b_do.dtype)
        # [BK, BQ]
        b_dk += tl.dot(bed, b_q, allow_tf32=False)
        # [BK, BQ]
        b_kq = tl.where(m_kq, tl.dot(b_k, tl.trans(b_q), allow_tf32=False), 0.).to(b_do.dtype)
        b_dsk += tl.dot(b_kq, b_ds, allow_tf32=False)
        b_dsk -= tl.dot(tl.where(m_kq, 1., 0.).to(b_do.dtype), b_ds * b_s, allow_tf32=False)

        b_ep = tl.where(m_kq, tl.dot(b_ev, tl.trans(b_p), allow_tf32=False), 0.).to(b_do.dtype)
        # [BK, BD]
        b_dv += tl.dot(b_ep, b_do, allow_tf32=False)
        # [BK, BQ]
        b_vdo = tl.where(m_kq, tl.dot(b_v, tl.trans(b_do), allow_tf32=False), 0.).to(b_do.dtype)
        b_dsv += tl.dot(b_vdo, b_p, allow_tf32=False)
        b_dsv -= tl.dot(tl.where(m_kq, 1., 0.).to(b_do.dtype), b_dp * b_p, allow_tf32=False)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dsk, (b_ek * b_dsk).to(p_dsk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dsv, (b_ev * b_dsv).to(p_dsv.dtype.element_ty), boundary_check=(0, 1))


class FlashABCAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, sk, sv):
        BQ, BK, BD, BM = 64, 64, triton.next_power_of_2(q.shape[-1]), triton.next_power_of_2(sk.shape[-1])
        batch_size, n_heads, seq_len, d_head, n_slots = *q.shape, sk.shape[-1]
        scale = d_head ** -0.5
        num_stages = 1
        num_warps = 4
        grid = (triton.cdiv(seq_len, BQ), batch_size * n_heads)

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            q, k, v = (pad(i, (batch_size, n_heads, seq_len, BD)) for i in (q, k, v))
        s = torch.empty_like(sk)
        p = torch.empty_like(s)
        o = torch.empty_like(q)
        sk, sv = (sk - sk.max(2, True)[0]).float(), (sv - sv.max(2, True)[0]).float()
        zk, zv = sk.logcumsumexp(2), sv.logcumsumexp(2)
        ek, zk = (sk - zk[:, :, -1:]).to(q.dtype).exp(), (zk - zk[:, :, -1:]).to(q.dtype).exp()
        ev, zv = (sv - zv[:, :, -1:]).to(q.dtype).exp(), (zv - zv[:, :, -1:]).to(q.dtype).exp()

        flash_abc_fwd_kernel[grid](
            q * scale, k, v, ek, ev, zk, zv, s, p, o,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BQ=BQ, BK=BK, BD=BD, BM=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v, ek, ev, zk, zv, s, p, o)
        ctx.seq_len = seq_len
        ctx.d_head = d_head
        ctx.n_slots = n_slots
        ctx.grid = grid
        ctx.dtype = q.dtype
        ctx.scale = scale
        return o[..., :d_head]

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, ek, ev, zk, zv, s, p, o = ctx.saved_tensors
        grid = ctx.grid
        scale = ctx.scale

        BQ, BK, BD, BM = 64, 64, q.shape[-1], ek.shape[-1]
        seq_len, d_head, n_slots = ctx.seq_len, ctx.d_head, ctx.n_slots
        num_stages = 1
        num_warps = 4

        def pad(x, sizes):
            p = x.new_zeros(sizes)
            p[tuple(slice(0, i) for i in x.shape)] = x
            return p
        if BD != d_head:
            do = pad(do, o.shape)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        ds, dp, dsk, dsv = torch.empty_like(ek), torch.empty_like(ek), torch.empty_like(ek), torch.empty_like(ek)

        doo = (o * do).sum(-1)
        flash_abc_bwd_kernel_dqsp[grid](
            k, v, ek, ev, zv, p / zk, do, doo, dq, ds, dp,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len, scale,
            BQ=BQ, BK=BK, BD=BD, BM=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        flash_abc_bwd_kernel_dkv[grid](
            q, k, v, ek, ev, s / scale, p / zv, do, ds, dp, dk, dv, dsk, dsv,
            q.stride(1), q.stride(2), q.stride(3), ek.stride(1), ek.stride(2), ek.stride(3),
            seq_len,
            BQ=BQ, BK=BK, BD=BD, BM=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq[..., :d_head], dk[..., :d_head], dv[..., :d_head], dsk[..., :n_slots], dsv[..., :n_slots]


def naive_attention(q, k, v, sk, sv):
    *_, d_head = q.shape
    dtype, scale = q.dtype, d_head ** -0.5
    # [batch_size, n_heads, seq_len, 64]
    sk = (sk - sk.max(2, True)[0]).to(torch.float).exp()
    sv = (sv - sv.max(2, True)[0]).to(torch.float).exp()
    zk, zv = sk.cumsum(2), sv.cumsum(2)
    sk, zk = (sk / zk[:, :, -1:]).to(dtype), (zk / zk[:, :, -1:]).to(dtype)
    sv, zv = (sv / zv[:, :, -1:]).to(dtype), (zv / zv[:, :, -1:]).to(dtype)
    # [batch_size, n_heads, seq_len, 64, d_head]
    K = (sk.unsqueeze(-1) * k.unsqueeze(-2)).cumsum(2) / zk.unsqueeze(-1)
    V = (sv.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(2) / zv.unsqueeze(-1)
    # [batch_size, n_heads, seq_len, 64]
    p = torch.einsum('...d,...md->...m', q * scale, K).softmax(-1, dtype=torch.float).to(dtype)
    # [batch_size, n_heads, seq_len, d_head]
    o = torch.einsum('...m,...md->...d', p, V)
    return o


chunk_abc_attention = ChunkABCAttentionFunction.apply
flash_abc_attention = FlashABCAttentionFunction.apply


if __name__ == '__main__':
    B, H, T, D, M = 8, 32, 1024, 64, 32
    dtype = torch.bfloat16
    torch.manual_seed(42)
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    # [batch_size, n_heads, seq_len, n_slots]
    sk = torch.randn((B, H, T, M), dtype=dtype, device='cuda').requires_grad_()
    sv = torch.randn((B, H, T, M), dtype=dtype, device='cuda').requires_grad_()

    print('Testing...')
    do = torch.randn_like(q)
    ref = flash_abc_attention(q, k, v, sk, sv)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dsk, sk.grad = sk.grad.clone(), None
    ref_dsv, sv.grad = sv.grad.clone(), None

    # triton implementation
    tri = chunk_abc_attention(q, k, v, sk, sv)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dsk, sk.grad = sk.grad.clone(), None
    tri_dsv, sv.grad = sv.grad.clone(), None
    assert ref.allclose(tri, 0, 1e-2), breakpoint()
    assert ref_dq.allclose(tri_dq, 0, 1e-2), breakpoint()
    assert ref_dk.allclose(tri_dk, 0, 1e-2), breakpoint()
    assert ref_dv.allclose(tri_dv, 0, 1e-2), breakpoint()
    assert ref_dsk.allclose(tri_dsk, 0, 1e-2), breakpoint()
    assert ref_dsv.allclose(tri_dsv, 0, 1e-2), breakpoint()
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
        batch_size, n_heads, d_head, n_slots = 8, 32, 100, 64

        q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        sk = torch.randn(batch_size, n_heads, seq_len, n_slots, device=device, requires_grad=requires_grad, dtype=dtype)
        sv = torch.randn(batch_size, n_heads, seq_len, n_slots, device=device, requires_grad=requires_grad, dtype=dtype)
        do = torch.ones_like(q, dtype=dtype)

        quantiles = [0.5, 0.2, 0.8]
        results = 0, 0, 0
        if provider == 'torch':
            if seq_len > 1000:
                return results
            results = triton.testing.do_bench(lambda: naive_attention(q, k, v, sk, sv), quantiles=quantiles)
        elif provider == 'chunk':
            if seq_len > 1000000:
                return 0, 0, 0
            results = triton.testing.do_bench(lambda: chunk_abc_attention(q, k, v, sk, sv), quantiles=quantiles)
        elif provider == 'flash':
            if seq_len > 10000:
                return 0, 0, 0
            results = triton.testing.do_bench(lambda: flash_abc_attention(q, k, v, sk, sv), quantiles=quantiles)
        elif provider == 'torch_bwd':
            if seq_len > 1000:
                return results
            results = triton.testing.do_bench(lambda: naive_attention(q, k, v, sk, sv).backward(do), quantiles=quantiles)
        elif provider == 'chunk_bwd':
            if seq_len > 100000:
                return 0, 0, 0
            results = triton.testing.do_bench(lambda: chunk_abc_attention(q, k, v, sk, sv).backward(do), quantiles=quantiles)
        elif provider == 'flash_bwd':
            if seq_len > 10000:
                return 0, 0, 0
            results = triton.testing.do_bench(lambda: flash_abc_attention(q, k, v, sk, sv).backward(do), quantiles=quantiles)
        return results
    benchmark.run(print_data=True)
