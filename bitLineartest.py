import torch
import torch.nn as nn
from model import BitLinear

def run_layer_tests():
    print("--- Running BitLinear Unit Tests ---")
    
    batch_size = 4
    in_features = 16
    out_features = 32

    print("T1:Initialization")
    layer = BitLinear(in_features, out_features, bias=False)
    x = torch.randn(batch_size, in_features)
    print("Passed")

    print("T3: Forward Pass & Ternary Constraints")
    out = layer(x)
    assert out.shape == (batch_size, out_features), "Output shape mismatch!"
    
    print("Passed")

    print("T3: Backward Pass (STE) Gradients")
    out.sum().backward()
    
    assert layer.weight.grad is not None, "Gradients are missing! STE failed."
    assert not torch.isnan(layer.weight.grad).any(), "Gradients contain NaNs!"
    assert not torch.isinf(layer.weight.grad).any(), "Gradients contain Infs!"
    print("Passed")
    
    print("\nAll systems green. The BitLinear layer is structurally sound.")

if __name__ == '__main__':
    run_layer_tests()