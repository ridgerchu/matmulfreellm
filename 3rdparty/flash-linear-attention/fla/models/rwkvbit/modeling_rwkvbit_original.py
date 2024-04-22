import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
from fla.modules import FusedCrossEntropyLoss
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from fla.ops.bitnet import BitLinear
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_bitsandbytes_available,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)
import torch.nn.init as init
from fla.models.rwkvbit.configuration_rwkvbit import RwkvBitConfig
from fla.models.rwkvbit.vanilla import wkv_triton_with_eps

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "RWKV/rwkv-4-169m-pile"
_CONFIG_FOR_DOC = "RwkvBitConfig"
rwkv_cuda_kernel = None


def init_linear_weights(module, mean=0.0, std=0.02):
    """
    初始化指定模块中所有Linear层的权重。

    Args:
        module (nn.Module): 需要初始化权重的模块。
        mean (float): 权重初始化的均值。
        std (float): 权重初始化的标准差。
    """
    if isinstance(module, (BitLinear, nn.Conv1d, BitLinear)):
        init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            init.constant_(module.bias, 0)


class RwkvBitLinearAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, time_decay, time_first, key, value, state=None, return_state=False
    ):
        batch_size, seq_len, hidden_size = key.size()

        ctx.input_dtype = key.dtype

        if (
            time_decay.device.type != "cuda"
            or time_first.device.type != "cuda"
            or key.device.type != "cuda"
            or value.device.type != "cuda"
        ):
            raise ValueError(
                "Calling the CUDA kernel for wkv attention requires all tensors to be on CUDA devices."
            )

        time_decay = -torch.exp(time_decay.float().contiguous())
        if key.dtype == torch.float16:
            time_first = time_first.float()
            key = key.float()
            value = value.float()
        time_first = time_first.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        # The CUDA kernel will fill this tensor.

        return output.to(ctx.input_dtype), None


def rwkv_linear_attention_cpu(
    time_decay, time_first, key, value, state=None, return_state=False
):
    # For CPU fallback. Will be slower and probably take more memory than the custom CUDA kernel if not executed
    # within a torch.no_grad.
    _, seq_length, _ = key.size()
    output = torch.zeros_like(key)

    if state is None:
        num_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        den_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        max_state = torch.zeros_like(key[:, 0], dtype=torch.float32) - 1e38
    else:
        num_state, den_state, max_state = state
    # For numerical stability
    #    real_numerator_state = num_state * torch.exp(max_state)
    #    real_denominator_state = den_state * torch.exp(max_state)

    time_decay = -torch.exp(time_decay)

    for current_index in range(seq_length):
        current_key = key[:, current_index].float()
        current_value = value[:, current_index]

        # wkv computation at time t
        max_for_output = torch.maximum(max_state, current_key + time_first)
        e1 = torch.exp(max_state - max_for_output)
        e2 = torch.exp(current_key + time_first - max_for_output)
        numerator = e1 * num_state + e2 * current_value
        denominator = e1 * den_state + e2
        output[:, current_index] = (numerator / denominator).to(output.dtype)

        # Update state for next iteration
        max_for_state = torch.maximum(max_state + time_decay, current_key)
        e1 = torch.exp(max_state + time_decay - max_for_state)
        e2 = torch.exp(current_key - max_for_state)
        num_state = e1 * num_state + e2 * current_value
        den_state = e1 * den_state + e2
        max_state = max_for_state

    if return_state or state is not None:
        state = [num_state, den_state, max_state]

    return output, state


def rwkv_linear_attention(
    time_decay, time_first, key, value, state=None, return_state=False
):
    return wkv_triton_with_eps(
            time_decay,
            time_first,
            key,
            value,
            state,
        )


class RwkvBitSelfAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        attention_hidden_size = (
            config.attention_hidden_size
            if config.attention_hidden_size is not None
            else hidden_size
        )
        self.attention_hidden_size = attention_hidden_size

        self.time_decay = nn.Parameter(torch.empty(attention_hidden_size))
        self.time_first = nn.Parameter(torch.empty(attention_hidden_size))

        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_value = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.hidden_state = torch.zeros(1, 3, 1, hidden_size)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = BitLinear(hidden_size, attention_hidden_size, bias=False)
        self.value = BitLinear(hidden_size, attention_hidden_size, bias=False)
        self.receptance = BitLinear(hidden_size, attention_hidden_size, bias=False)
        self.output = BitLinear(attention_hidden_size, hidden_size, bias=False)

        self.apply(init_linear_weights)
        scale = math.sqrt(1.0 / 2 * self.config.num_hidden_layers)
        self.output.weight.data *= scale

    # TODO: maybe jit, otherwise move inside forward
    def extract_key_value(self, hidden, state=None):
        # Mix hidden with the previous timestep to produce key, value, receptance
        if hidden.size(1) == 1 and state is not None:
            shifted = state[1][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[1][:, :, self.layer_id]
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        value = hidden * self.time_mix_value + shifted * (1 - self.time_mix_value)
        receptance = hidden * self.time_mix_receptance + shifted * (
            1 - self.time_mix_receptance
        )

        key = self.key(key)
        value = self.value(value)
        receptance = torch.sigmoid(self.receptance(receptance))

        if state is not None:
            state[1][:, :, self.layer_id] = hidden[:, -1]
        return receptance, key, value, state

    def forward(self, hidden, state=None, use_cache=False):
        bsz, _, _ = hidden.shape
        receptance, key, value, state = self.extract_key_value(hidden, state=state)
        layer_state = self.hidden_state.repeat_interleave(bsz, dim=0).to(key.device)
        rwkv, layer_state = rwkv_linear_attention(
            self.time_decay,
            self.time_first,
            key,
            value,
            state=layer_state,
            return_state=use_cache,
        )
        layer_state = None
        if layer_state is not None:
            state[2][:, :, self.layer_id] = layer_state[0]
            state[3][:, :, self.layer_id] = layer_state[1]
            state[4][:, :, self.layer_id] = layer_state[2]

        return self.output(receptance * rwkv), state


class RwkvBitFeedForward(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        intermediate_size = (
            config.intermediate_size
            if config.intermediate_size is not None
            else 4 * config.hidden_size
        )

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = BitLinear(hidden_size, intermediate_size, bias=False)
        self.receptance = BitLinear(hidden_size, hidden_size, bias=False)
        self.value = BitLinear(intermediate_size, hidden_size, bias=False)

        self.apply(init_linear_weights)
        scale = math.sqrt(1.0 / 2 * self.config.num_hidden_layers)
        self.value.weight.data *= scale

    def forward(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            shifted = state[0][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[0][:, :, self.layer_id]
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        receptance = hidden * self.time_mix_receptance + shifted * (
            1 - self.time_mix_receptance
        )

        key = torch.square(torch.relu(self.key(key)))
        value = self.value(key)
        receptance = torch.sigmoid(self.receptance(receptance))

        if state is not None:
            state[0][:, :, self.layer_id] = hidden[:, -1]

        return receptance * value, state


class RwkvBitBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.attention = RwkvBitSelfAttention(config, layer_id)
        self.feed_forward = RwkvBitFeedForward(config, layer_id)

    def forward(self, hidden, state=None, use_cache=False, output_attentions=False):
        # if self.layer_id == 0:
        #     hidden = self.pre_ln(hidden)

        attention, state = self.attention(hidden, state=state, use_cache=use_cache)
        hidden = hidden + attention

        feed_forward, state = self.feed_forward(hidden, state=state)
        hidden = hidden + feed_forward

        outputs = (hidden, state)
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)

        return outputs


class RwkvBitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RwkvBitConfig
    base_model_prefix = "rwkv"
    _no_split_modules = ["RwkvBitBlock"]
    _keep_in_fp32_modules = ["time_decay", "time_first"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, RwkvBitSelfAttention):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            attention_hidden_size = module.attention_hidden_size

            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_mix_key.dtype,
                device=module.time_mix_key.device,
            )
            time_weight = time_weight[None, None, :]

            decay_speed = [
                -5 + 8 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                for h in range(attention_hidden_size)
            ]
            decay_speed = torch.tensor(
                decay_speed,
                dtype=module.time_decay.dtype,
                device=module.time_decay.device,
            )
            zigzag = (
                torch.tensor(
                    [(i + 1) % 3 - 1 for i in range(attention_hidden_size)],
                    dtype=module.time_first.dtype,
                    device=module.time_first.device,
                )
                * 0.5
            )
            if isinstance(module, (nn.Linear, nn.Conv1d, BitLinear)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                nn.init.normal_(
                    module.weight, mean=0.0, std=self.config.initializer_range
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight, mean=0.0, std=self.config.initializer_range
                )
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

            with torch.no_grad():
                module.time_decay.data = decay_speed
                module.time_first.data = torch.ones_like(
                    module.time_first * math.log(0.3) + zigzag
                )

                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_value.data = (
                    torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
                )
                module.time_mix_receptance.data = torch.pow(
                    time_weight, 0.5 * ratio_1_to_almost0
                )
        elif isinstance(module, RwkvBitFeedForward):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size

            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_mix_key.dtype,
                device=module.time_mix_key.device,
            )
            time_weight = time_weight[None, None, :]

            with torch.no_grad():
                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_receptance.data = torch.pow(
                    time_weight, ratio_1_to_almost0
                )


@dataclass
class RwkvBitOutput(ModelOutput):
    """
    Class for the RWKV model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class RwkvBitCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


RWKV_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RwkvBitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RWKV_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            This is currently not used by `RwkvBitModel`, but will be supported in the future.

            [What are attention masks?](../glossary#attention-mask)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        state (tuple of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the last state is returned and can be used to quickly generate the next logits.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare RWKV Model transformer outputting raw hidden-states without any specific head on top.",
    RWKV_START_DOCSTRING,
)
class RwkvBitModel(RwkvBitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(
            self.embeddings.weight, mean=0.0, std=self.config.initializer_range
        )

        self.blocks = nn.ModuleList(
            [
                RwkvBitBlock(config, layer_id=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.ln_out = nn.LayerNorm(config.hidden_size)

        self.layers_are_rescaled = False

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=RwkvBitOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, RwkvBitOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not self.training else False)
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.training == self.layers_are_rescaled:
            self._rescale_layers()

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and state is None:
            shape = (
                inputs_embeds.size(0),
                self.config.hidden_size,
                self.config.num_hidden_layers,
            )
            state = [
                torch.zeros(
                    *shape,
                    dtype=inputs_embeds.dtype if i <= 1 else torch.float32,
                    device=inputs_embeds.device
                )
                for i in range(5)
            ]
            state[4] -= 1e30

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        hidden_states = inputs_embeds

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for idx, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                hidden_states, state, attentions = self._gradient_checkpointing_func(
                    block.__call__, hidden_states, state, use_cache, output_attentions
                )
            else:
                hidden_states, state, attentions = block(
                    hidden_states,
                    state=state,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if (
                self.layers_are_rescaled
                and self.config.rescale_every > 0
                and (idx + 1) % self.config.rescale_every == 0
            ):
                hidden_states = hidden_states / 2

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        hidden_states = self.ln_out(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                x
                for x in [hidden_states, state, all_hidden_states, all_self_attentions]
                if x is not None
            )

        return RwkvBitOutput(
            last_hidden_state=hidden_states,
            state=state,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _rescale_layers(self):
        # Layers should be rescaled for inference only.
        if self.layers_are_rescaled == (not self.training):
            return
        if self.config.rescale_every > 0:
            with torch.no_grad():
                for block_id, block in enumerate(self.blocks):
                    if self.training:
                        block.attention.output.weight.mul_(
                            2 ** int(block_id // self.config.rescale_every)
                        )
                        block.feed_forward.value.weight.mul_(
                            2 ** int(block_id // self.config.rescale_every)
                        )
                    else:
                        # Deal with quantization statistics
                        if hasattr(block.attention.output.weight, "SCB"):
                            block.attention.output.weight.SCB.div_(
                                2 ** int(block_id // self.config.rescale_every)
                            )
                            block.feed_forward.value.weight.SCB.div_(
                                2 ** int(block_id // self.config.rescale_every)
                            )
                        elif hasattr(block.attention.output.weight, "quant_state"):
                            self._bnb_4bit_dequantize_and_rescale(
                                block.attention.output, block_id
                            )
                            self._bnb_4bit_dequantize_and_rescale(
                                block.feed_forward.value, block_id
                            )
                        else:
                            block.attention.output.weight.div_(
                                2 ** int(block_id // self.config.rescale_every)
                            )
                            block.feed_forward.value.weight.div_(
                                2 ** int(block_id // self.config.rescale_every)
                            )

        self.layers_are_rescaled = not self.training

    def _bnb_4bit_dequantize_and_rescale(self, target_layer, block_id):
        r"""
        Perform the dequantization and rescaling of the weights of a given layer. After that operation the layer will
        be quantized again.
        """
        if not is_bitsandbytes_available():
            raise ImportError("Please install bitsandbytes to use this method.")
        import bitsandbytes as bnb

        dequant_weights = bnb.functional.dequantize_4bit(
            target_layer.weight.data, target_layer.weight.quant_state
        )

        dequant_weights.div_(2 ** int(block_id // self.config.rescale_every))

        # re-quantize the model:
        # we need to put it first on CPU then back to the device
        # this will create an overhead :/
        # We set requires_grad=False as we cannot compute gradients on top of 4bit parameters anyway and to avoid
        # bugs with bnb
        quant_weight = bnb.nn.Params4bit(
            dequant_weights.to("cpu"), requires_grad=False
        ).to(dequant_weights.device)
        setattr(target_layer, "weight", quant_weight)


@add_start_docstrings(
    """
    The RWKV Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    RWKV_START_DOCSTRING,
)
class RwkvBitForCausalLM(RwkvBitPreTrainedModel):
    _tied_weights_keys = ["head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.rwkv = RwkvBitModel(config)
        self.head = BitLinear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(init_linear_weights)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def generate(self, *args, **kwargs):
        # Thin wrapper to raise exceptions when trying to generate with methods that manipulate `past_key_values`.
        # RWKV is one of the few models that don't have it (it has `state` instead, which has different properties and
        # usage).
        try:
            gen_output = super().generate(*args, **kwargs)
        except AttributeError as exc:
            # Expected exception: "AttributeError: '(object name)' object has no attribute 'past_key_values'"
            if "past_key_values" in str(exc):
                raise AttributeError(
                    "You tried to call `generate` with a decoding strategy that manipulates `past_key_values`. RWKV "
                    "doesn't have that attribute, try another generation strategy instead. For the available "
                    "generation strategies, check this doc: https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exc
        return gen_output

    def prepare_inputs_for_generation(
        self, input_ids, state=None, inputs_embeds=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and state is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["state"] = state
        return model_inputs

    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=RwkvBitCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, RwkvBitCausalLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        rwkv_outputs = self.rwkv(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = rwkv_outputs[0]

        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = FusedCrossEntropyLoss(inplace_backward=True)
            labels = labels.to(logits.device)
            # Adjust labels as per the new strategy: shift left and append ignore_index at the end
            labels = torch.cat(
                (
                    labels[..., 1:],
                    torch.full_like(labels[:, :1], loss_fct.ignore_index),
                ),
                1,
            )
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + rwkv_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=rwkv_outputs.state,
            hidden_states=rwkv_outputs.hidden_states,
            attentions=rwkv_outputs.attentions,
        )
