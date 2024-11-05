import torch
import bitblas
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import time
import argparse

torch.set_grad_enabled(False)
bitblas.set_log_level("INFO")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def generate_text_batch(model, tokenizer, prompts, max_length=100):
    # Encode the input prompts as a batch
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)

    # Generate cos and sin values (commented out as not used in generation)
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    generation_config = GenerationConfig(
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )

    start_time = time.time()
    output_ids = model.generate(input_ids, generation_config=generation_config)
    end_time = time.time()

    # Decode the output ids to text
    generated_texts = [
        tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids
    ]

    generation_time = end_time - start_time
    num_tokens = sum(len(output_id) for output_id in output_ids)
    tokens_per_second = num_tokens / generation_time

    print(f"Generated {num_tokens} tokens in {generation_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return generated_texts


def profile(model, input_data):
    import numpy as np
    model = model.cuda()
    model.eval()

    def get_runtime(num_repeats=1):
        tic = time.time()
        for _ in range(num_repeats):
            _ = model(input_data)
        torch.cuda.synchronize()
        return (time.time() - tic) * 1000 / num_repeats

    with torch.no_grad():
        st = time.time()
        while time.time() - st < 1.0:
            get_runtime()  # warmup
        warmup_runtime = get_runtime()
        num_repeats = max(1, int(1000 / warmup_runtime))
        times = get_runtime(num_repeats)
    return np.mean(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default=16, type=int)
    parser.add_argument('--in_seq_len', default=32, type=int)
    parser.add_argument('--out_seq_len', default=128, type=int)
    parser.add_argument('--bitblas', action='store_true')
    parser.add_argument('--model_size', choices=['1.3B', '2.7B', '370M'], default='1.3B', help="Choose model size: 1.3B, 2.7B, or 370M")
    args = parser.parse_args()

    # Set model path based on selected model size
    model_paths = {
        '1.3B': 'ridger/MMfreeLM-1.3B',
        '2.7B': 'ridger/MMfreeLM-2.7B',
        '370M': 'ridger/MMfreeLM-370M'
    }
    model_path = model_paths[args.model_size]
    
    bs = args.bs
    in_seq_len = args.in_seq_len
    out_seq_len = args.out_seq_len
    is_bitblas = args.bitblas

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_flash_attention_2=False,
        torch_dtype=torch.float16,
    ).cuda().half()

    if is_bitblas:
        with torch.no_grad():
            model.quantize()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = " ".join(["Hello"] * in_seq_len)
    prompts = [prompt] * bs
    max_length = out_seq_len + in_seq_len
    print(generate_text_batch(model, tokenizer, prompts, max_length=max_length))


if __name__ == '__main__':
    main()
