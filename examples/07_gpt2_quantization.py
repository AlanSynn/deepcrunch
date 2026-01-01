"""
Example 7: GPT-2 Quantization - Real-world LLM

This example demonstrates quantization of GPT-2 language model.
Shows how to compress large language models for efficient deployment.

Note: This example works with or without HuggingFace transformers.
      With transformers: Uses real GPT-2 model
      Without: Uses GPT-2-like architecture for demonstration
"""

import torch
import torch.nn as nn
from deepcrunch.backend.backend_registry import BackendRegistry
import time


# GPT-2 like model for demonstration (if transformers not available)
class GPT2LikeModel(nn.Module):
    """Simplified GPT-2-like model for demonstration"""

    def __init__(self, vocab_size=50257, n_embd=768, n_layer=12, n_head=12, block_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                batch_first=True
            )
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape

        # Embeddings
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


def try_load_real_gpt2():
    """Try to load real GPT-2 from transformers library"""
    try:
        from transformers import GPT2LMHeadModel
        print("   Loading real GPT-2 from HuggingFace...")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        print("   ✓ Loaded GPT-2 (124M parameters)")
        return model, "real"
    except ImportError:
        print("   transformers not installed, using GPT-2-like model")
        return None, "demo"
    except Exception as e:
        print(f"   Could not load GPT-2: {e}")
        return None, "demo"


def main():
    print("=" * 80)
    print("GPT-2 Language Model Quantization")
    print("Large Language Model (LLM) Compression Example")
    print("=" * 80)

    # Try to load real GPT-2, otherwise use demo model
    print("\n1. Loading GPT-2 Model")
    real_model, model_type = try_load_real_gpt2()

    if model_type == "real":
        model = real_model
        vocab_size = 50257
    else:
        print("   Creating GPT-2-like model for demonstration...")
        model = GPT2LikeModel(
            vocab_size=50257,
            n_embd=768,
            n_layer=12,
            n_head=12,
            block_size=1024
        )
        print("   ✓ Created GPT-2-like model")
        vocab_size = 50257

    model.eval()

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    print(f"\n   Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {model_size_mb:.2f} MB")
    print(f"   Architecture: GPT-2 ({model_type})")

    # Create sample input (token IDs)
    print("\n2. Preparing Input")
    seq_length = 128
    batch_size = 1
    sample_input = torch.randint(0, vocab_size, (batch_size, seq_length))

    print(f"   Input shape: {sample_input.shape} (batch, sequence_length)")
    print(f"   Sample token IDs: {sample_input[0, :10].tolist()}")

    # Original model inference
    print("\n3. Original FP32 Model")
    print("   Running inference...")

    start = time.time()
    with torch.no_grad():
        original_output = model(sample_input)
    original_time = (time.time() - start) * 1000

    if model_type == "real":
        # HuggingFace GPT2LMHeadModel returns object with logits
        if hasattr(original_output, 'logits'):
            original_logits = original_output.logits
        else:
            original_logits = original_output
    else:
        original_logits = original_output

    print(f"   Inference time: {original_time:.2f} ms")
    print(f"   Output shape: {original_logits.shape}")
    print(f"   Output: (batch, sequence, vocab_size)")

    # Get next token predictions
    next_token_logits = original_logits[0, -1, :]  # Last token predictions
    top_k_tokens = next_token_logits.topk(5)[1]
    print(f"   Top-5 next token predictions: {top_k_tokens.tolist()}")

    # ============================================
    # Dynamic Quantization (Best for LLMs)
    # ============================================
    print("\n4. Applying Dynamic Quantization (INT8)")
    print("   Quantizing Linear layers to INT8...")
    print("   This is the RECOMMENDED method for LLMs!")

    backend = BackendRegistry.get_backend("torch")
    backend.model = model

    quantized_model = backend.quantize(
        type="dynamic",
        dtype="qint8"
    )

    # Quantized model size
    quantized_size_mb = sum(p.nelement() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)

    print(f"\n   ✓ Quantization completed")
    print(f"   Quantized model size: {quantized_size_mb:.2f} MB")
    print(f"   Size reduction: {(1 - quantized_size_mb/model_size_mb) * 100:.2f}%")
    print(f"   Memory saved: {model_size_mb - quantized_size_mb:.2f} MB")

    # Quantized model inference
    print("\n5. Quantized INT8 Model")
    print("   Running inference...")

    start = time.time()
    with torch.no_grad():
        quantized_output = quantized_model(sample_input)
    quantized_time = (time.time() - start) * 1000

    if model_type == "real":
        if hasattr(quantized_output, 'logits'):
            quantized_logits = quantized_output.logits
        else:
            quantized_logits = quantized_output
    else:
        quantized_logits = quantized_output

    print(f"   Inference time: {quantized_time:.2f} ms")
    print(f"   Speedup: {original_time/quantized_time:.2f}x")

    # Get next token predictions from quantized model
    quant_next_token_logits = quantized_logits[0, -1, :]
    quant_top_k_tokens = quant_next_token_logits.topk(5)[1]
    print(f"   Top-5 next token predictions: {quant_top_k_tokens.tolist()}")

    # ============================================
    # Accuracy Comparison
    # ============================================
    print("\n6. Accuracy Analysis")

    # Compare logits
    logits_diff = torch.abs(original_logits - quantized_logits).mean().item()
    print(f"   Mean absolute error (logits): {logits_diff:.6f}")

    # Compare top predictions
    top_k = 10
    orig_top_k = original_logits[0, -1, :].topk(top_k)[1]
    quant_top_k = quantized_logits[0, -1, :].topk(top_k)[1]

    matches = sum(1 for i in range(top_k) if orig_top_k[i] in quant_top_k[:top_k])
    print(f"   Top-{top_k} token overlap: {matches}/{top_k}")

    # ============================================
    # Benchmark with Different Sequence Lengths
    # ============================================
    print("\n7. Performance Benchmark")
    print("   Testing with different sequence lengths...\n")

    seq_lengths = [32, 64, 128, 256]

    print(f"   {'Seq Length':<12s} {'FP32 (ms)':<12s} {'INT8 (ms)':<12s} {'Speedup':<12s} {'Memory Saved'}")
    print("   " + "-" * 65)

    for seq_len in seq_lengths:
        test_input = torch.randint(0, vocab_size, (1, seq_len))

        # Original model
        start = time.time()
        with torch.no_grad():
            model(test_input)
        orig_time = (time.time() - start) * 1000

        # Quantized model
        start = time.time()
        with torch.no_grad():
            quantized_model(test_input)
        quant_time = (time.time() - start) * 1000

        speedup = orig_time / quant_time
        memory_saved = model_size_mb - quantized_size_mb

        print(f"   {seq_len:<12d} {orig_time:<12.2f} {quant_time:<12.2f} "
              f"{speedup:<12.2f}x {memory_saved:.2f} MB")

    # ============================================
    # Float16 Alternative
    # ============================================
    print("\n8. Float16 Alternative (FP16)")
    print("   Testing mixed-precision as comparison...")

    fp16_model = backend.quantize(type="dynamic", dtype="float16")
    fp16_size_mb = sum(p.nelement() * p.element_size() for p in fp16_model.parameters()) / (1024 * 1024)

    print(f"   FP16 model size: {fp16_size_mb:.2f} MB")
    print(f"   Size reduction: {(1 - fp16_size_mb/model_size_mb) * 100:.2f}%")

    # Test FP16
    start = time.time()
    with torch.no_grad():
        fp16_model(sample_input)
    fp16_time = (time.time() - start) * 1000

    print(f"   FP16 inference time: {fp16_time:.2f} ms")
    print(f"   FP16 speedup: {original_time/fp16_time:.2f}x")

    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 80)
    print("✓ GPT-2 quantization completed successfully!")
    print("\n  Compression Results:")
    print(f"  • FP32 (Original): {model_size_mb:.2f} MB, {original_time:.2f} ms")
    print(f"  • INT8 (Quantized): {quantized_size_mb:.2f} MB, {quantized_time:.2f} ms ({original_time/quantized_time:.2f}x faster)")
    print(f"  • FP16 (Mixed): {fp16_size_mb:.2f} MB, {fp16_time:.2f} ms ({original_time/fp16_time:.2f}x faster)")

    print("\n  Key Insights:")
    print("  • INT8 quantization reduces model size by ~50-75%")
    print("  • Inference speed improves by 1.5-3x on CPU")
    print("  • Minimal impact on text generation quality")
    print("  • Perfect for deployment on edge devices or mobile")

    print("\n  Best Practices for LLM Quantization:")
    print("  1. Use Dynamic Quantization for transformer models")
    print("  2. Test on your specific use case (generation, classification)")
    print("  3. Measure perplexity or other task-specific metrics")
    print("  4. Consider 4-bit quantization for even larger models (e.g., GPTQ)")
    print("  5. Combine with other techniques (pruning, distillation)")

    print("\n  Production Deployment:")
    print("  • Mobile apps: INT8 quantization essential")
    print("  • Cloud inference: Consider FP16 for GPUs")
    print("  • Edge devices: INT8 provides best size/speed tradeoff")
    print("  • Large-scale serving: Quantization reduces infrastructure costs")
    print("=" * 80)


if __name__ == "__main__":
    main()
