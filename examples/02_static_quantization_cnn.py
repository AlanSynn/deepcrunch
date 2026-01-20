"""
Example 2: Static Quantization - CNN Model

This example demonstrates static quantization on a CNN model for image classification.
Static quantization requires calibration data but provides better performance.
"""

import torch
import torch.nn as nn
from deepcrunch.backend.backend_registry import BackendRegistry


# Define a CNN model for image classification
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_calibration_data(num_batches=20):
    """Create calibration data for static quantization"""
    print(f"   Generating {num_batches} batches of calibration data...")
    for i in range(num_batches):
        # Generate random images (batch_size=4, channels=3, height=32, width=32)
        yield [torch.randn(4, 3, 32, 32)]


def main():
    print("=" * 80)
    print("Static Quantization Example - CNN Model")
    print("=" * 80)

    # Create and initialize model
    model = SimpleCNN(num_classes=10)
    model.eval()

    # Create sample input
    sample_input = torch.randn(1, 3, 32, 32)

    # Get original model output
    print("\n1. Original Model")
    with torch.no_grad():
        original_output = model(sample_input)
    print(f"   Output shape: {original_output.shape}")
    print(f"   Predicted class: {original_output.argmax(dim=1).item()}")

    # Calculate original model size
    original_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {original_params:,}")

    # Quantize the model
    print("\n2. Quantizing with Static Quantization")
    print("   This requires calibration data to observe activation ranges...")

    backend = BackendRegistry.get_backend("torch")
    backend.model = model

    # Create calibration data generator
    calibration_data = create_calibration_data(num_batches=20)

    quantized_model = backend.quantize(
        type="static",
        calibration_data=calibration_data,
    )

    # Test quantized model
    print("\n3. Quantized Model")
    with torch.no_grad():
        quantized_output = quantized_model(sample_input)
    print(f"   Output shape: {quantized_output.shape}")
    print(f"   Predicted class: {quantized_output.argmax(dim=1).item()}")

    # Calculate quantized model parameters
    quantized_params = sum(p.numel() for p in quantized_model.parameters())
    print(f"   Parameters: {quantized_params:,}")

    # Compare results
    print("\n4. Comparison")

    # Calculate output difference
    diff = torch.abs(original_output - quantized_output).mean().item()
    print(f"   Output difference (MAE): {diff:.6f}")

    # Check class predictions match
    orig_class = original_output.argmax(dim=1).item()
    quant_class = quantized_output.argmax(dim=1).item()
    print(f"   Same prediction: {orig_class == quant_class}")

    # Benchmark inference speed
    print("\n5. Performance Benchmark")

    # Original model
    import time
    iterations = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(sample_input)
    original_time = (time.time() - start) / iterations * 1000
    print(f"   Original model: {original_time:.4f} ms/image")

    # Quantized model
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            quantized_model(sample_input)
    quantized_time = (time.time() - start) / iterations * 1000
    print(f"   Quantized model: {quantized_time:.4f} ms/image")

    speedup = original_time / quantized_time
    print(f"   Speedup: {speedup:.2f}x")

    print("\n" + "=" * 80)
    print("✓ Static quantization completed successfully!")
    print("  Static quantization is ideal for CNNs and provides better performance")
    print("  than dynamic quantization, but requires calibration data.")
    print("=" * 80)


if __name__ == "__main__":
    main()
