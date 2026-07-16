import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Linear):
    """
    Custom Linear layer for BitNet b1.58.
    Quantizes weights to ternary (-1, 0, 1) and activations to 8-bit (int8).
    """
    def __init__(self, in_features, out_features, bias=False, eps=1e-5):
        # We inherit from nn.Linear so self.weight and self.bias are handled automatically
        super().__init__(in_features, out_features, bias)
        self.eps = eps

    def forward(self, x):
        # -------------------------------------------------------------
        # 1. Weight Quantization (Ternary)
        # -------------------------------------------------------------
        w = self.weight

        # Calculate gamma: the mean absolute value of the weight matrix
        gamma = w.abs().mean()

        # Scale, round, and clamp to ternary range [-1, 0, 1]
        w_scaled = w / (gamma + self.eps)
        w_quant = torch.clamp(torch.round(w_scaled), -1, 1)

        # Straight-Through Estimator (STE) trick for weights
        # Forward pass uses w_quant, backward pass updates original w
        w_quant = w + (w_quant - w).detach()

        # -------------------------------------------------------------
        # 2. Activation Quantization (8-bit)
        # -------------------------------------------------------------
        # Calculate beta: the max absolute value per token/feature
        beta = x.abs().max(dim=-1, keepdim=True).values

        # Scale to 8-bit integer range [-128, 127], round, and clamp
        x_scaled = x * (127.0 / (beta + self.eps))
        x_quant = torch.clamp(torch.round(x_scaled), -128, 127)

        # Straight-Through Estimator (STE) trick for activations
        x_quant = x + (x_quant - x).detach()

        # -------------------------------------------------------------
        # 3. The Core Matrix Multiplication
        # -------------------------------------------------------------
        # F.linear performs: x_quant @ w_quant^T + bias
        output = F.linear(x_quant, w_quant, self.bias)

        # -------------------------------------------------------------
        # 4. Dequantization / Rescaling
        # -------------------------------------------------------------
        # We divided w by gamma and x by (beta/127), so we must multiply them back
        scale_factor = (gamma * beta) / 127.0
        output = output * scale_factor

        return output
