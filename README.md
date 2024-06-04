<div align=center>
<img src="__assets__/logo.png" width="150px">
</div>
<h2 align="center">MatMul-Free LM</h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>
<h5 align="center"> This repo is adapted from <a href="https://github.com/sustcsonglin/flash-linear-attention">flash-linear-attention</a>. </h2>

<h5 align="center">

# Installation

The following requirements should be satisfied 
- [PyTorch](https://pytorch.org/) >= 2.0
- [Triton](https://github.com/openai/triton) >=2.2
- [einops](https://einops.rocks/)

```sh
pip install -U git+https://github.com/ridgerchu/matmulfreellm
```

# Usage

## Model

We provide the implementations of models that are compatible with 🤗 Transformers library. 
Here's an example of how to initialize a GLA model from the default configs in `matmulfreelm`:
This is a huggingface-compatible libary that you can use such command to initize the model with huggingface `AutoModel`:


```py
>>> from mmfreelm.models import HGRNBitConfig
>>> from transformers import AutoModel
>>> config = HGRNBitConfig()
>>> AutoModel.from_config(config)
HGRNBitModel(
  (embeddings): Embedding(32000, 2048)
  (layers): ModuleList(
    (0): HGRNBitBlock(
      (attn_norm): RMSNorm(2048, eps=1e-06)
      (attn): HGRNBitAttention(
        (i_proj): FusedBitLinear(
          in_features=2048, out_features=2048, bias=False
          (norm): RMSNorm(2048, eps=1e-08)
        )
        (f_proj): FusedBitLinear(
          in_features=2048, out_features=2048, bias=False
          (norm): RMSNorm(2048, eps=1e-08)
        )
        (g_proj): FusedBitLinear(
          in_features=2048, out_features=2048, bias=False
          (norm): RMSNorm(2048, eps=1e-08)
        )
        (g_norm): FusedRMSNormSwishGate()
        (o_proj): FusedBitLinear(
          in_features=2048, out_features=2048, bias=False
          (norm): RMSNorm(2048, eps=1e-08)
        )
      )
      (mlp_norm): RMSNorm(2048, eps=1e-06)
      (mlp): HGRNBitMLP(
        (gate_proj): FusedBitLinear(
          in_features=2048, out_features=11264, bias=False
          (norm): RMSNorm(2048, eps=1e-08)
        )
        (down_proj): FusedBitLinear(
          in_features=5632, out_features=2048, bias=False
          (norm): RMSNorm(5632, eps=1e-08)
        )
        (act_fn): SiLU()
      )
    )
    
)
>>> 

```

## Generation

Upon successfully pretraining a model, it becomes accessible for generating text using the 🤗 text generation APIs.
In the following, we give a generation example in `generate.py`:

```py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer
#Change here to our open-sourced model
name = ''
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).cuda().half()
input_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
outputs = model.generate(input_ids, max_length=32,  do_sample=True, top_p=0.4, temperature=0.6)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

All of the pretrained models currently available can be found in [`fla-hub`](https://huggingface.co/fla-hub).
```py
>>> from huggingface_hub import list_models
>>> for model in list_models(author='mmfreelm-hub'): print(model.id)
```


# Citation
If you find this repo useful, please consider citing our works:
```bib
@article{yang2023gated,
  title   = {Gated Linear Attention Transformers with Hardware-Efficient Training},
  author  = {Yang, Songlin and Wang, Bailin and Shen, Yikang and Panda, Rameswar and Kim, Yoon},
  journal = {arXiv preprint arXiv:2312.06635},
  year    = {2023}
}

@software{yang2024fla,
  title  = {FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism},
  author = {Yang, Songlin and Zhang, Yu},
  url    = {https://github.com/sustcsonglin/flash-linear-attention},
  month  = jan,
  year   = {2024}
}
```
