# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Literal, Optional

from llmtuner.extras.logging import get_logger
from llmtuner.hparams import FinetuningArguments, ModelArguments

logger = get_logger(__name__)


@dataclass
class ModelArguments(ModelArguments):

    abc: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable ABC Attention."}
    )
    retnet: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable Rentention."}
    )
    gla: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable Gated Linear Attention."}
    )
    tokenizer: str = field(
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        metadata={"help": "Name of the tokenizer to use."}
    )


@dataclass
class FinetuningArguments(FinetuningArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    finetuning_type: Optional[Literal["lora", "raw", "full"]] = field(
        default="lora", metadata={"help": "Which fine-tuning method to use."}
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.name_module_trainable = split_arg(self.name_module_trainable)
        self.lora_alpha = self.lora_alpha or self.lora_rank * 2
        self.lora_target = split_arg(self.lora_target)
        self.additional_target = split_arg(self.additional_target)

        assert self.finetuning_type in ["lora", "raw", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."


def get_train_args(*args, **kwargs):
    import llmtuner
    from llmtuner.hparams.parser import _TRAIN_ARGS, get_train_args
    llmtuner.hparams.ModelArguments = ModelArguments
    llmtuner.hparams.FinetuningArguments = FinetuningArguments
    _TRAIN_ARGS[0] = ModelArguments
    _TRAIN_ARGS[-2] = FinetuningArguments
    return get_train_args(*args, **kwargs)


def get_eval_args(*args, **kwargs):
    import llmtuner
    from llmtuner.hparams.parser import _EVAL_ARGS, get_eval_args
    llmtuner.hparams.ModelArguments = ModelArguments
    llmtuner.hparams.FinetuningArguments = FinetuningArguments
    _EVAL_ARGS[0] = ModelArguments
    _EVAL_ARGS[-1] = FinetuningArguments
    return get_eval_args(*args, **kwargs)
