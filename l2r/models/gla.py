# -*- coding: utf-8 -*-


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv
from transformers.utils import logging

from fla.modules import FusedRMSNormSwishGate, RotaryEmbedding
from fla.ops.gla.chunk import chunk_gla

logger = logging.get_logger(__name__)


class LlamaGatedLinearAttention(LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gate_low_rank_dim = 16
        self.gate_logit_normalizer = 16
        self.g_norm = FusedRMSNormSwishGate(self.head_dim)
        self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gk_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.gate_low_rank_dim, bias=False),
            nn.Linear(self.gate_low_rank_dim, self.hidden_size, bias=True)
        )

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        q = rearrange(self.q_proj(hidden_states), 'b n (h d) -> b n h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b n (h d) -> b n h d', h=self.num_key_value_heads)
        v = rearrange(self.v_proj(hidden_states), 'b n (h d) -> b h n d', h=self.num_key_value_heads)
        gk = rearrange(self.gk_proj(hidden_states), 'b n (h d) -> b h n d', h=self.num_heads)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

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

        o = chunk_gla(q, k, v, gk)
        o = rearrange(o, 'b h t d -> b t h d')
        g = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        o = rearrange(self.g_norm(o, g), 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if not output_attentions:
            p = None

        return o, p, past_key_value
