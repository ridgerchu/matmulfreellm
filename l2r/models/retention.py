# -*- coding: utf-8 -*-


from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv
from transformers.utils import logging

from fla.modules import FusedRMSNormSwishGate, RotaryEmbedding
from fla.ops.retention import chunk_retention

logger = logging.get_logger(__name__)


class LlamaRetention(LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.g_norm = FusedRMSNormSwishGate(self.head_dim)
        self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2 ** -2.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        # [batch_size, seq_len, n_heads * d_head]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = v.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        q, k = self.rotary_emb(q, k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if past_key_value is not None:  # reuse k, v, self_attention
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v) if use_cache else None

        # cast to half precision
        input_dtype = q.dtype
        if input_dtype == torch.float:
            logger.warning_once("The input hidden states seems to be silently casted in float32.")
            q = q.to(self.config.torch_dtype)
            k = k.to(self.config.torch_dtype)
            v = v.to(self.config.torch_dtype)

        if getattr(self, "num_key_value_groups", None):
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        o = chunk_retention(q, k, v)
        o = rearrange(o, 'b h t d -> b t h d')
        g = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        o = rearrange(self.g_norm(o, g), 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if not output_attentions:
            p = None

        return o, p, past_key_value
