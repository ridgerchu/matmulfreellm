# -*- coding: utf-8 -*-


from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (LlamaFlashAttention2,
                                                      rotate_half)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class XposRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, n_heads=None, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.n_heads = n_heads
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        gamma = 1 - torch.tensor(2).pow(-5 - torch.arange(n_heads)).to(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("gamma", gamma, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        scale = self.gamma.unsqueeze(1).pow(t)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len], self.scale


def apply_rotary_pos_emb(q, k, cos, sin, scale, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos + rotate_half(q) * sin) * scale.unsqueeze(-1)
    k_embed = (k * cos + rotate_half(k) * sin) / scale.unsqueeze(-1)
    return q_embed, k_embed


class LlamaXPosAttention(LlamaFlashAttention2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_rope(self):
        self.rotary_emb = XposRotaryEmbedding(self.head_dim,
                                              max_position_embeddings=self.max_position_embeddings,
                                              base=self.rope_theta,
                                              n_heads=self.num_heads)

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

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        q, k = apply_rotary_pos_emb(q, k, *self.rotary_emb(v, seq_len=kv_seq_len), position_ids)

        if past_key_value is not None:  # reuse k, v, self_attention
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v) if use_cache else None

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        input_dtype = q.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )
            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)

        o = self._flash_attention_forward(q, k, v, attention_mask, seq_len, dropout=dropout_rate)
        o = o.reshape(batch_size, seq_len, self.hidden_size)
        if hasattr(self, "g_proj"):
            o = self.g_proj(hidden_states) * self.g_norm(o.transpose(1, 2)).transpose(1, 2)
        o = self.o_proj(o)

        if not output_attentions:
            p = None

        return o, p, past_key_value
