import torch
from model import GPT, GPTConfig

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_vram(model, bits_per_param):
    params = count_parameters(model)
    bytes_per_param = bits_per_param / 8.0
    total_megabytes = (params * bytes_per_param) / (1024 ** 2)
    return total_megabytes

config = GPTConfig(vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384)
bit_model = GPT(config)

print("--- Memory Profiling Report ---")
params = count_parameters(bit_model)
print(f"Total Trainable Parameters: {params:,}")

fp32_mb = estimate_vram(bit_model, 32)
print(f"Estimated Model Size (Standard FP32): {fp32_mb:.2f} MB")

bit158_mb = estimate_vram(bit_model, 2)
print(f"Estimated Model Size (1.58-Bit Packed): {bit158_mb:.2f} MB")
print(f"Theoretical Memory Reduction: {(1 - (bit158_mb/fp32_mb))*100:.1f}%")