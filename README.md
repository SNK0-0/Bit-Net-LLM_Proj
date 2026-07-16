# 🧠 Bit-Net-LLM_Proj

A modular, scientifically controlled implementation of the 1.58-bit ternary weight architecture (BitNet b1.58) applied to a character-level nanoGPT model. 

This repository serves as a laboratory for understanding Extreme LLM Quantization. By replacing continuous floating-point weights with discrete ternary values `{-1, 0, 1}`, this project explores the mathematical mechanics, optimization challenges, and hardware implications of the 1-bit LLM era.

---

## 🔬 Architecture & Methodology

The core of the architecture is the custom `BitLinear` module, which replaces standard `nn.Linear` layers and enforces two strict rules during the forward pass:

1. **Weight Quantization (Ternary):** Weights are scaled by their absolute mean (γ) and clamped to `[-1, 0, 1]`.
2. **Activation Quantization (INT8):** Incoming activations are scaled by their absolute maximum (β) and clamped to signed 8-bit integers `[-128, 127]`.

### The Straight-Through Estimator (STE)
To solve the zero-gradient problem caused by the non-differentiable `round()` step function, this architecture employs the `.detach()` trick. This allows discrete ternary weights to be used during the forward pass while routing continuous gradients to the latent high-precision weights during backpropagation.

---

## 📊 Benchmarks & Results

To measure the impact of ternary quantization, a strict control test was performed. Both models were trained on the Shakespeare character dataset for exactly 1,500 iterations at a `1e-3` learning rate on an NVIDIA T4 GPU.

| Metric | FP16 Baseline | 1.58-Bit Quantized | Delta (Impact) |
| :--- | :--- | :--- | :--- |
| **Final Train Loss** | 1.1510 | 2.4712 | +1.3202 |
| **Final Val Loss** | 1.4739 | 2.4870 | +1.0131 |
| **Model MFU** | ~0.64% | 0.47% | -0.17% |

**Analysis (The Quantization Shock):** As mathematically expected, forcing the optimization space into a highly restricted discrete set creates an initial shock to the network. The 1.58-bit model exhibits a higher loss under standard hyperparameters because quantized gradients are highly noisy approximations. The drop in MFU (Model FLOPs Utilization) is due to relying on high-level PyTorch Python operations for quantization rather than highly optimized custom C++/CUDA kernels.

---

## 🚀 Usage

**1. Train the baseline FP16 model:**
```bash
python scripts/train.py configs/train_shakespeare_char.py
```

**2. Train the 1.58-bit Quantized model:**
```bash
python scripts/train_bitnet.py configs/train_shakespeare_char.py
```

**3. Generate text from a saved checkpoint:**
```bash
python scripts/generate.py
```
*(Note: The generation script includes a pre-load dictionary cleanup loop to strip the `_orig_mod.` prefix applied by PyTorch 2.0's `torch.compile()` optimization feature).*

---

## 🗺️ Development Roadmap

- [x] **Phase 1:** Establish FP16 continuous-weight baseline metrics.
- [x] **Phase 2:** Architect the `BitLinear` module (Ternary weights, 8-bit activations, STE).
- [x] **Phase 3:** Modular architecture integration (replacing standard Linear layers).
- [x] **Phase 4:** Quantized model training execution and benchmark logging.
- [x] **Phase 5:** Checkpoint recovery and character-level text generation.
- [ ] **Phase 6:** The Hyperparameter Hunt (Optimizing LR and iterations for ternary convergence).
- [ ] **Phase 7:** Integrating `RMSNorm` to match the official Microsoft BitNet specification.
- [ ] **Phase 8:** Scaling up to Sub-Word Tokens (Tiktoken).
