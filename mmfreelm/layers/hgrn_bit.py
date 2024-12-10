# -*- coding: utf-8 -*-

# "HGRN2: Gated Linear RNNs with State Expansion"[https://arxiv.org/abs/2404.07904]

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from mmfreelm.modules import FusedRMSNormSwishGate, ShortConvolution
from mmfreelm.modules.activations import swiglu
from mmfreelm.ops.hgrn.recurrent_fuse import fused_recurrent_hgrn

#from mmfreelm.ops.bitnet import BitLinear_Fuse as BitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear


class HGRNBitAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'fused_recurrent',
        hidden_size: int = 1024,
        num_heads: Optional[int] = None,
        expand_ratio: Optional[int] = 1,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        share_conv_kernel: bool = True,
        layernorm_eps: float = 1e-5,
        layer_idx: int = None
    ) -> HGRNAttention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.input_dim = int(hidden_size * expand_ratio)
        self.head_dim = self.input_dim // self.num_heads
        self.recurrent_state = None
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.share_conv_kernel = share_conv_kernel


        self.rms_norm_custom = torch.nn.RMSNorm(hidden_size)

        self.layer_idx = layer_idx

        assert mode in ['fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.hidden_size % num_heads == 0, f"hidden size must be divisible by num_heads of {num_heads}"

        self.i_proj = BitLinear(hidden_size, self.input_dim, bias=False)#i
        self.f_proj = BitLinear(hidden_size, self.input_dim, bias=False)#g
        self.g_proj = BitLinear(hidden_size, self.input_dim, bias=False)#o

        

        if use_short_conv:
            self.conv_size = conv_size
            if share_conv_kernel:
                self.h_conv1d = ShortConvolution(hidden_size, conv_size, activation='silu')
            else:
                self.q_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')
                self.f_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')
                self.i_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')
                self.g_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')

        self.g_norm = FusedRMSNormSwishGate(self.input_dim, layernorm_eps)
        self.o_proj = BitLinear(self.input_dim, hidden_size, bias=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, (nn.Linear, BitLinear)):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = past_key_values[self.layer_idx] if use_cache else None
        if past_key_values is not None:
            print("Past Key Values", len(past_key_values))
        if last_state is not None:
            print("last_state", last_state.shape)
       

        if self.use_short_conv:
            conv_state = last_state[0] if use_cache else None
            if self.share_conv_kernel:
                # conv state is updated inplace
                hidden_states = self.h_conv1d(hidden_states, attention_mask, conv_state)
                i = self.i_proj(hidden_states)
                f = self.f_proj(hidden_states)
                g = self.g_proj(hidden_states)
            else:
                conv_state_i = last_state[2] if use_cache else None
                conv_state_f = last_state[1] if use_cache else None
                conv_state_g = last_state[0] if use_cache else None #WWWHYYYYYYYY
                i = self.i_conv1d(self.i_proj(hidden_states), attention_mask, conv_state_i)
                f = self.f_conv1d(self.f_proj(hidden_states), attention_mask, conv_state_f)
                g = self.g_conv1d(self.g_proj(hidden_states), attention_mask, conv_state_g)
        else:
            i = self.i_proj(hidden_states)
            f = self.f_proj(hidden_states)
            g = self.g_proj(hidden_states)

      
        
        ########################################
        f = f.sigmoid()
        # the lower bound for the first layer is zero
        if lower_bound is not None and self.layer_idx > 0:
            f = lower_bound + (1 - lower_bound) * f
        ########################################

    


        # dealing with left-padding
        if attention_mask is not None:
            i = i.mul_(attention_mask.unsqueeze(-1))


        B, T, D = i.shape
        if self.recurrent_state is None:
            self.recurrent_state = torch.zeros((B,T,D,D), dtype=torch.float32, device=i.device)
        

        if mode == 'fused_recurrent':
            for _ in range(T):

                self.recurrent_state = torch.matmul(self.recurrent_state, torch.diag_embed(f)) + torch.einsum('...i,...j->...ij', i, (1 - f))
                o = torch.matmul(self.recurrent_state, g.unsqueeze(-1)).squeeze(-1)

        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
     
        if past_key_values is not None:
            if self.use_short_conv:
                if self.share_conv_kernel:
                    last_state = (conv_state, self.recurrent_state)
                else:
                    last_state = (conv_state_i, conv_state_f, self.recurrent_state)
            else:
                last_state = (self.recurrent_state,)
            past_key_values.update(last_state, self.layer_idx, i.shape[2])
        
        o = self.rms_norm_custom(o)
        o = self.o_proj(o)

        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            if self.share_conv_kernel:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
            else:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),
                          param.new_zeros(batch_size, self.hidden_size, self.conv_size),
                          param.new_zeros(batch_size, self.hidden_size, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.hidden_size
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size