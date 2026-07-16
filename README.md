# Bit-Net-LLM_Proj

A modular, scientifically controlled implementation of the 1.58-bit ternary weight architecture (BitNet b1.58) applied to a character-level nanoGPT model. 

This repository serves as a laboratory for understanding Extreme LLM Quantization. By replacing continuous floating-point weights with discrete ternary values `{-1, 0, 1}`, this project explores the mathematical mechanics, optimization challenges, and hardware implications of the 1-bit LLM era.

## 🔬 Architecture & Methodology

Unlike monolithic implementations, this repository uses a **modular injection strategy** to isolate variables and ensure rigorous scientific control against an FP16 baseline. 

The core of the architecture is the custom `BitLinear` module, which enforces two strict rules during the forward pass:
1. **Weight Quantization (Ternary):** Weights are scaled by their absolute mean ($\gamma$) and clamped to `[-1, 0, 1]`.
2. **Activation Quantization (INT8):** Incoming activations are scaled by their absolute maximum ($\beta$) and clamped to 8-bit integers `[-128, 127]`.

### The Straight-Through Estimator (STE)
To solve the zero-gradient problem caused by the non-differentiable `round()` step function, this architecture employs the `.detach()` trick. This allows discrete ternary weights to be used during the forward pass while routing continuous gradients to the latent high-precision weights during backpropagation.

## 📂 Repository Structure

```text
BitNet-b1.58-BabyGPT/
├── configs/
│   └── train_shakespeare_char.py    # Hyperparameter configurations
├── data/
│   └── shakespeare_char/            # Tokenized character-level dataset
├── modules/
│   └── bitlinear.py                 # Custom STE and Quantization math
├── scripts/
│   ├── model.py                     # Standard FP16 Baseline Backbone
│   ├── model_bitnet.py              # Modified 1.58-bit Backbone
│   ├── train.py                     # Baseline Execution Engine
│   ├── train_bitnet.py              # Quantized Execution Engine
│   └── generate.py                  # Text prediction and dictionary cleanup
├── notebooks/
│   └── BitNet_LLM_B.ipynb           # Master Colab Workspace & Generation Scripts
└── .gitignore
