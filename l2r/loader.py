# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)
from transformers.integrations import is_deepspeed_zero3_enabled

import fla  # noqa
from llmtuner.extras.logging import get_logger
from llmtuner.extras.misc import count_parameters
from llmtuner.hparams import FinetuningArguments
from llmtuner.model.patcher import patch_config, patch_model
from llmtuner.model.utils import register_autoclass

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from llmtuner.hparams import ModelArguments


logger = get_logger(__name__)


def load_model_and_tokenizer(
    model_args: ModelArguments,
    finetuning_args: FinetuningArguments,
    is_trainable: Optional[bool] = False
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer,)
    if finetuning_args.finetuning_type == 'raw':
        logger.info("All model params are randomly initialized for from-scratch training.")
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        if model_args.flash_attn:
            patch_config(config, tokenizer, model_args, config_kwargs, is_trainable)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=model_args.compute_dtype)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        patch_config(config, tokenizer, model_args, config_kwargs, is_trainable)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=model_args.compute_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            **config_kwargs,
        )

    patch_model(model, tokenizer, model_args, is_trainable)
    register_autoclass(config, model, tokenizer)

    if not is_trainable:
        model.requires_grad_(False)
        model = model.to(model_args.compute_dtype) if not getattr(model, "quantization_method", None) else model
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    logger.info(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )

    if not is_trainable:
        logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")

    return model, tokenizer
