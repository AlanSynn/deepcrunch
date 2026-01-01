"""
Example 5: BERT Quantization (LLM Example)

This example demonstrates quantization of BERT model for NLP tasks.
Shows how to quantize transformer-based language models.
"""

import torch
import torch.nn as nn
from deepcrunch.backend.backend_registry import BackendRegistry


# Simplified BERT-like model for demonstration
class SimpleBERTModel(nn.Module):
    """Simplified BERT-like model with attention mechanism"""

    def __init__(self, vocab_size=30000, hidden_size=768, num_layers=6, num_heads=12, num_classes=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(512, hidden_size)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids):
        # input_ids shape: (batch_size, seq_length)
        batch_size, seq_length = input_ids.shape

        # Create position IDs
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        # Embeddings
        token_embeddings = self.embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        # Transformer
        transformer_output = self.transformer(embeddings)

        # Classification (use [CLS] token - first token)
        cls_output = transformer_output[:, 0, :]
        logits = self.classifier(cls_output)

        return logits


def main():
    print("=" * 80)
    print("BERT Quantization Example (LLM)")
    print("=" * 80)

    # Model configuration
    vocab_size = 30000
    hidden_size = 768
    num_layers = 6
    num_heads = 12
    num_classes = 2  # Binary classification (e.g., sentiment analysis)
    seq_length = 128

    print("\n1. Creating BERT-like Model")
    print(f"   Vocabulary size: {vocab_size:,}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Number of layers: {num_layers}")
    print(f"   Number of attention heads: {num_heads}")

    # Create model
    model = SimpleBERTModel(vocab_size, hidden_size, num_layers, num_heads, num_classes)
    model.eval()

    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {model_size:.2f} MB")

    # Create sample input (batch of token IDs)
    print("\n2. Creating Sample Input")
    batch_size = 4
    sample_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Sample token IDs: {sample_input[0, :10].tolist()}")

    # Original model inference
    print("\n3. Original Model Inference")
    with torch.no_grad():
        original_output = model(sample_input)

    print(f"   Output shape: {original_output.shape}")
    print(f"   Predictions: {original_output.argmax(dim=1).tolist()}")
    print(f"   Confidence scores:\n{original_output.softmax(dim=1)}")

    # Quantize with Dynamic Quantization
    print("\n4. Applying Dynamic Quantization")
    print("   Quantizing Linear layers to INT8...")
    print("   This is ideal for BERT/Transformer models!")

    backend = BackendRegistry.get_backend("torch")
    backend.model = model

    quantized_model = backend.quantize(
        type="dynamic",
        dtype="qint8"
    )

    # Quantized model size
    quantized_size = sum(p.nelement() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)
    print(f"\n   ✓ Quantization completed")
    print(f"   Quantized model size: {quantized_size:.2f} MB")
    print(f"   Size reduction: {(1 - quantized_size/model_size) * 100:.2f}%")

    # Quantized model inference
    print("\n5. Quantized Model Inference")
    with torch.no_grad():
        quantized_output = quantized_model(sample_input)

    print(f"   Output shape: {quantized_output.shape}")
    print(f"   Predictions: {quantized_output.argmax(dim=1).tolist()}")
    print(f"   Confidence scores:\n{quantized_output.softmax(dim=1)}")

    # Compare outputs
    print("\n6. Accuracy Comparison")
    output_diff = torch.abs(original_output - quantized_output).mean().item()
    print(f"   Mean absolute error: {output_diff:.6f}")

    # Check if predictions match
    orig_preds = original_output.argmax(dim=1)
    quant_preds = quantized_output.argmax(dim=1)
    accuracy = (orig_preds == quant_preds).float().mean().item() * 100
    print(f"   Prediction accuracy: {accuracy:.2f}%")

    # Benchmark inference speed
    print("\n7. Inference Speed Benchmark")
    print("   Testing with different batch sizes...\n")

    import time

    batch_sizes = [1, 4, 8, 16]

    for bs in batch_sizes:
        test_input = torch.randint(0, vocab_size, (bs, seq_length))

        # Original model
        iterations = 50
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                model(test_input)
        original_time = (time.time() - start) / iterations * 1000

        # Quantized model
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                quantized_model(test_input)
        quantized_time = (time.time() - start) / iterations * 1000

        speedup = original_time / quantized_time
        print(f"   Batch size {bs:2d}: Original={original_time:7.2f}ms, "
              f"Quantized={quantized_time:7.2f}ms, Speedup={speedup:.2f}x")

    # Memory efficiency test
    print("\n8. Memory Efficiency")

    # Calculate memory usage per inference
    def calculate_memory_mb(module):
        return sum(p.nelement() * p.element_size() for p in module.parameters()) / (1024 * 1024)

    original_memory = calculate_memory_mb(model)
    quantized_memory = calculate_memory_mb(quantized_model)

    print(f"   Original model memory: {original_memory:.2f} MB")
    print(f"   Quantized model memory: {quantized_memory:.2f} MB")
    print(f"   Memory saved: {original_memory - quantized_memory:.2f} MB ({(1 - quantized_memory/original_memory) * 100:.2f}%)")

    print("\n" + "=" * 80)
    print("✓ BERT quantization completed successfully!")
    print("\n  Key Insights:")
    print("  • Dynamic quantization is ideal for BERT/Transformer models")
    print("  • Quantizes Linear/Embedding layers to INT8")
    print("  • Significant memory reduction (~50-75%)")
    print("  • Minimal accuracy loss (<1% typically)")
    print("  • Faster inference, especially on CPU")
    print("\n  Real-world applications:")
    print("  • Sentiment analysis")
    print("  • Text classification")
    print("  • Question answering")
    print("  • Named entity recognition")
    print("=" * 80)


if __name__ == "__main__":
    main()
