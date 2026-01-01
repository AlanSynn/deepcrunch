"""
Example 1: Dynamic Quantization - Simple Linear Model

This example demonstrates basic dynamic quantization on a simple PyTorch model.
Dynamic quantization is the easiest to use and works well for models with Linear layers.
"""

import torch
import torch.nn as nn
from deepcrunch.backend.backend_registry import BackendRegistry


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def main():
    print("=" * 80)
    print("Dynamic Quantization Example")
    print("=" * 80)

    # Create and initialize model
    model = SimpleModel()
    model.eval()

    # Create sample input
    sample_input = torch.randn(1, 100)

    # Get original model output
    print("\n1. Original Model")
    with torch.no_grad():
        original_output = model(sample_input)
    print(f"   Output shape: {original_output.shape}")
    print(f"   Sample output: {original_output[0, :5]}")

    # Calculate original model size
    original_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    print(f"   Model size: {original_size / 1024:.2f} KB")

    # Quantize the model
    print("\n2. Quantizing with Dynamic Quantization (qint8)")
    backend = BackendRegistry.get_backend("torch")
    backend.model = model

    quantized_model = backend.quantize(
        type="dynamic",
        dtype="qint8"
    )

    # Test quantized model
    print("\n3. Quantized Model")
    with torch.no_grad():
        quantized_output = quantized_model(sample_input)
    print(f"   Output shape: {quantized_output.shape}")
    print(f"   Sample output: {quantized_output[0, :5]}")

    # Calculate quantized model size
    quantized_size = sum(p.nelement() * p.element_size() for p in quantized_model.parameters())
    print(f"   Model size: {quantized_size / 1024:.2f} KB")

    # Compare results
    print("\n4. Comparison")
    print(f"   Size reduction: {(1 - quantized_size/original_size) * 100:.2f}%")

    # Calculate output difference
    diff = torch.abs(original_output - quantized_output).mean().item()
    print(f"   Output difference (MAE): {diff:.6f}")

    # Check if outputs are close
    are_close = torch.allclose(original_output, quantized_output, rtol=0.1, atol=0.1)
    print(f"   Outputs close (rtol=0.1): {are_close}")

    print("\n" + "=" * 80)
    print("✓ Dynamic quantization completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
