"""
Example 3: LSTM Quantization

This example demonstrates dynamic quantization on LSTM models for sequence processing.
LSTMs benefit greatly from dynamic quantization.
"""

import torch
import torch.nn as nn
from deepcrunch.backend.backend_registry import BackendRegistry


# Define an LSTM model for sequence classification
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_layers=2, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use the last hidden state
        out = self.fc(lstm_out[:, -1, :])
        return out


def main():
    print("=" * 80)
    print("LSTM Quantization Example")
    print("=" * 80)

    # Model parameters
    input_size = 50
    hidden_size = 128
    num_layers = 2
    num_classes = 5
    seq_length = 20

    # Create and initialize model
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    model.eval()

    # Create sample input (batch_size=1, seq_length=20, input_size=50)
    sample_input = torch.randn(1, seq_length, input_size)

    # Get original model output
    print("\n1. Original LSTM Model")
    with torch.no_grad():
        original_output = model(sample_input)
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Output shape: {original_output.shape}")
    print(f"   Predicted class: {original_output.argmax(dim=1).item()}")

    # Calculate original model size
    original_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    original_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {original_params:,}")
    print(f"   Model size: {original_size / 1024:.2f} KB")

    # Quantize the model
    print("\n2. Quantizing LSTM with Dynamic Quantization")
    print("   Dynamic quantization is particularly effective for LSTMs!")

    backend = BackendRegistry.get_backend("torch")
    backend.model = model

    # Try different quantization dtypes
    dtypes = ["qint8", "float16"]

    for dtype in dtypes:
        print(f"\n   Testing with dtype: {dtype}")
        quantized_model = backend.quantize(
            type="dynamic",
            dtype=dtype
        )

        # Test quantized model
        with torch.no_grad():
            quantized_output = quantized_model(sample_input)

        # Calculate quantized model size
        quantized_size = sum(p.nelement() * p.element_size() for p in quantized_model.parameters())

        print(f"   └─ Model size: {quantized_size / 1024:.2f} KB")
        print(f"   └─ Size reduction: {(1 - quantized_size/original_size) * 100:.2f}%")

        # Calculate output difference
        diff = torch.abs(original_output - quantized_output).mean().item()
        print(f"   └─ Output difference (MAE): {diff:.6f}")

        # Check class predictions match
        orig_class = original_output.argmax(dim=1).item()
        quant_class = quantized_output.argmax(dim=1).item()
        print(f"   └─ Same prediction: {orig_class == quant_class}")

    # Benchmark with batch processing
    print("\n3. Batch Processing Benchmark")

    backend.model = model
    quantized_model = backend.quantize(type="dynamic", dtype="qint8")

    batch_sizes = [1, 8, 16, 32]

    for batch_size in batch_sizes:
        batch_input = torch.randn(batch_size, seq_length, input_size)

        # Original model
        import time
        iterations = 50
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                model(batch_input)
        original_time = (time.time() - start) / iterations * 1000

        # Quantized model
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                quantized_model(batch_input)
        quantized_time = (time.time() - start) / iterations * 1000

        speedup = original_time / quantized_time
        print(f"   Batch size {batch_size:2d}: Original={original_time:6.2f}ms, "
              f"Quantized={quantized_time:6.2f}ms, Speedup={speedup:.2f}x")

    print("\n" + "=" * 80)
    print("✓ LSTM quantization completed successfully!")
    print("  Key takeaways:")
    print("  • Dynamic quantization works great for RNNs/LSTMs")
    print("  • Significant size reduction with minimal accuracy loss")
    print("  • Better speedup with larger batch sizes")
    print("=" * 80)


if __name__ == "__main__":
    main()
