<div align=center>
<img src="__assets__/logo.png" width="150px">
</div>
<h2 align="center">MatMul-Free LLM</h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>

<h5 align="center">

# Installation

The following requirements should be satisfied 
- [PyTorch](https://pytorch.org/) >= 2.0
- [Triton](https://github.com/openai/triton) >=2.2
- [einops](https://einops.rocks/)

As `fla` is actively developed now, no released packages are provided at this time.
If you do need to use `fla` ops/modules and contemplate further explorations, an alternative way is to install the package from source
```sh
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```
or manage `fla` with submodules
```sh
git submodule add https://github.com/sustcsonglin/flash-linear-attention.git 3rdparty/flash-linear-attention
ln -s 3rdparty/flash-linear-attention/mmfreelm mmfreelm
```

> [!CAUTION]
> If you're not working with Triton v2.2 or its nightly release, it's important to be aware of potential issues with the `FusedChunk` implementation, detailed in this [issue](https://github.com/openai/triton/issues/2852). 
You can run the test `python tests/test_fused_chunk.py` to check if your version is affected by similar compiler problems. 
While we offer some fixes for Triton<=2.1, be aware that these may result in reduced performance.
>
> For both Triton 2.2 and earlier versions (up to 2.1), you can reliably use the `Chunk` version (with hidden states materialized into HBMs).
> After careful optimization, this version generally delivers high performance in most scenarios.

# Usage

## Token Mixing

We provide "token mixing" linear attention layers in `fla.layers` for you to use. 
You can replace the standard multihead attention layer in your model with other linear attention layers. 
Example usage is as follows:

```py
>> > import torch
>> > from mmfreelm.layers import MultiScaleRetention
>> > batch_size, num_heads, seq_len, hidden_size, = 32, 4, 2048, 1024
>> > device, dtype = 'cuda:0', torch.bfloat16
>> > retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)
>> > x = torch.randn(batch_size, seq_len, hidden_size).to(device=device, dtype=dtype)
>> > y, *_ = retnet(x)
>> > y.shape
torch.Size([32, 2048, 1024])
```

We provide the implementations of models that are compatible with 🤗 Transformers library. 
Here's an example of how to initialize a GLA model from the default configs in `fla`:

```py
>> > from mmfreelm.models import GLAConfig
>> > from transformers import AutoModel
>> > config = GLAConfig()
>> > config
GLAConfig
{
    "attn_mode": "fused_chunk",
    "bos_token_id": 1,
    "clamp_min": null,
    "conv_size": 4,
    "eos_token_id": 2,
    "expand_k": 0.5,
    "expand_v": 1,
    "fuse_cross_entropy": true,
    "fuse_norm": true,
    "hidden_act": "swish",
    "hidden_ratio": 4,
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": null,
    "max_position_embeddings": 2048,
    "model_type": "gla",
    "num_heads": 4,
    "num_hidden_layers": 24,
    "rms_norm_eps": 1e-06,
    "share_conv_kernel": true,
    "tie_word_embeddings": false,
    "transformers_version": "4.39.1",
    "use_cache": true,
    "use_gk": true,
    "use_gv": false,
    "use_short_conv": false,
    "vocab_size": 32000
}

>> > AutoModel.from_config(config)
GLAModel(
    (embed_tokens): Embedding(32000, 2048)
(layers): ModuleList(
    (0 - 23): 24
x
GLABlock(
    (attn_norm): RMSNorm()
(attn): GatedLinearAttention(
    (gate_fn): SiLU()
(q_proj): Linear(in_features=2048, out_features=1024, bias=False)
(k_proj): Linear(in_features=2048, out_features=1024, bias=False)
(v_proj): Linear(in_features=2048, out_features=2048, bias=False)
(g_proj): Linear(in_features=2048, out_features=2048, bias=False)
(gk_proj): Sequential(
    (0): Linear(in_features=2048, out_features=16, bias=False)
(1): Linear(in_features=16, out_features=1024, bias=True)
)
(o_proj): Linear(in_features=2048, out_features=2048, bias=False)
(g_norm_swish_gate): FusedRMSNormSwishGate()
)
(mlp_norm): RMSNorm()
(mlp): GLAMLP(
    (gate_proj): Linear(in_features=2048, out_features=11264, bias=False)
(down_proj): Linear(in_features=5632, out_features=2048, bias=False)
(act_fn): SiLU()
)
)
)
(norm): RMSNorm()
)

```

## Generation

Upon successfully pretraining a model, it becomes accessible for generating text using the 🤗 text generation APIs.
In the following, we give a generation example:

```py
>> > import mmfreelm
>> > from transformers import AutoModelForCausalLM, AutoTokenizer
>> > name = 'mmfreelm-hub/gla-340M-15B'
>> > tokenizer = AutoTokenizer.from_pretrained(name)
>> > model = AutoModelForCausalLM.from_pretrained(name).cuda()
>> > input_prompt = "Power goes with permanence. Impermanence is impotence. And rotation is castration."
>> > input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
>> > outputs = model.generate(input_ids, max_length=64)
>> > tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
```

We also provide a simple script [here](benchmarks/benchmark_generation.py) for benchmarking the generation speed.
Simply run it by:
```sh
$ python -m benchmarks.benchmark_generation \
  --path 'mmfreelm-hub/gla-340M-15B' \
  --repetition_penalty 2. \
  --prompt="Hello everyone, I'm Songlin Yang"

Prompt:
Hello everyone, I'm Songlin Yang
Generated:
Hello everyone, I'm Songlin Yang.
I am a 20 year old girl from China who is currently studying in the United States of America for my Master degree and also working as an English teacher at school here on campus since last summer (1st semester). My main goal to be able do well with this course so that we can have

Prompt length: 10, generation length: 64
Total prompt processing + decoding time: 4593ms
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
