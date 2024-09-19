# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch

import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', default='ridger/MMfreeLM-2.7B', type=str)
args = parser.parse_args()
def profile(model, input_data):
    import time

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
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_path,
        device_map='auto',
        torch_dtype=torch.float16,
    ).half()
    print(
        f"gpu memory: {torch.cuda.memory_allocated() / 1024 ** 3} GB"
    )
    with torch.no_grad():
        model.quantize()
        #model._post_process_weights()
    print(
        f"gpu memory BitBLAS: {torch.cuda.memory_allocated() / 1024 ** 3} GB"
    )

if __name__ == '__main__':
    main()
