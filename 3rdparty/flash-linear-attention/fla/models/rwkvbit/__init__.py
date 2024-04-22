# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.rwkvbit.configuration_rwkvbit import RwkvBitConfig
from fla.models.rwkvbit.modeling_rwkvbit import RwkvBitForCausalLM, RwkvBitModel

AutoConfig.register(RwkvBitConfig.model_type, RwkvBitConfig)
AutoModel.register(RwkvBitConfig, RwkvBitModel)
AutoModelForCausalLM.register(RwkvBitConfig, RwkvBitForCausalLM)


__all__ = ['RwkvBitConfig', 'RwkvBitForCausalLM', 'RwkvBitModel']
