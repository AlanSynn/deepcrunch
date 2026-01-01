"""
Example 6: ResNet Quantization - Real-world Vision Model

This example demonstrates quantization of ResNet models for image classification.
Shows practical compression for production deployment.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from deepcrunch.backend.backend_registry import BackendRegistry
import time


def create_calibration_data(num_batches=10, batch_size=4):
    """Generate calibration data for static quantization"""
    print(f"   Generating {num_batches} batches of calibration images...")
    for i in range(num_batches):
        # Generate random images (ImageNet size: 224x224)
        yield [torch.randn(batch_size, 3, 224, 224)]


def benchmark_model(model, input_tensor, iterations=100, warmup=10):
    """Benchmark model inference speed"""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)

    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            output = model(input_tensor)
    elapsed = (time.time() - start) / iterations * 1000  # ms

    return elapsed, output


def main():
    print("=" * 80)
    print("ResNet-18 Quantization Example")
    print("Real-world Computer Vision Model")
    print("=" * 80)

    # Load pre-trained ResNet-18 (you can also use resnet50, resnet101, etc.)
    print("\n1. Loading Pre-trained ResNet-18")
    try:
        # Try to load pretrained model
        model = models.resnet18(pretrained=True)
        print("   ✓ Loaded pretrained ResNet-18")
    except:
        # If no internet, create random initialized model
        model = models.resnet18(pretrained=False)
        print("   ✓ Created ResNet-18 (random weights)")

    model.eval()

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {model_size_mb:.2f} MB")

    # Create sample input (ImageNet size)
    print("\n2. Preparing Input Data")
    batch_size = 1
    sample_input = torch.randn(batch_size, 3, 224, 224)
    print(f"   Input shape: {sample_input.shape} (batch, channels, height, width)")

    # Original model inference
    print("\n3. Original FP32 Model Inference")
    original_time, original_output = benchmark_model(model, sample_input)
    print(f"   Inference time: {original_time:.2f} ms")
    print(f"   Output shape: {original_output.shape}")
    print(f"   Top-5 predictions: {original_output[0].topk(5)[1].tolist()}")

    # ============================================
    # Method 1: Dynamic Quantization
    # ============================================
    print("\n4. Dynamic Quantization (INT8)")
    print("   Note: Dynamic quantization may not be optimal for CNNs")
    print("   It works best on Linear layers (FC layers in ResNet)")

    backend = BackendRegistry.get_backend("torch")
    backend.model = model

    try:
        dynamic_model = backend.quantize(type="dynamic", dtype="qint8")
        dynamic_size_mb = sum(p.nelement() * p.element_size() for p in dynamic_model.parameters()) / (1024 * 1024)

        print(f"   ✓ Dynamic quantization completed")
        print(f"   Model size: {dynamic_size_mb:.2f} MB")
        print(f"   Size reduction: {(1 - dynamic_size_mb/model_size_mb) * 100:.2f}%")

        # Test
        dynamic_time, dynamic_output = benchmark_model(dynamic_model, sample_input)
        print(f"   Inference time: {dynamic_time:.2f} ms (speedup: {original_time/dynamic_time:.2f}x)")
    except Exception as e:
        print(f"   ⚠ Dynamic quantization skipped: {str(e)[:50]}...")

    # ============================================
    # Method 2: Static Quantization (Recommended for CNNs)
    # ============================================
    print("\n5. Static Quantization (INT8) - Recommended for CNNs")
    print("   This quantizes both weights and activations")
    print("   Requires calibration data for best accuracy")

    backend.model = model

    # Generate calibration data
    calibration_data = create_calibration_data(num_batches=10, batch_size=4)

    try:
        static_model = backend.quantize(
            type="static",
            calibration_data=calibration_data,
        )

        print(f"   ✓ Static quantization completed")

        # Calculate size - static quantized models are optimized
        print(f"   Quantized model created (optimized for inference)")

        # Test
        static_time, static_output = benchmark_model(static_model, sample_input)
        print(f"   Inference time: {static_time:.2f} ms (speedup: {original_time/static_time:.2f}x)")

        # Check accuracy
        orig_preds = original_output[0].topk(5)[1]
        static_preds = static_output[0].topk(5)[1]
        top1_match = orig_preds[0] == static_preds[0]
        print(f"   Top-1 prediction match: {top1_match}")
        print(f"   Top-5 predictions: {static_preds.tolist()}")

    except Exception as e:
        print(f"   ⚠ Static quantization failed: {str(e)[:100]}...")
        static_model = None

    # ============================================
    # Method 3: QAT (Quantization-Aware Training)
    # ============================================
    print("\n6. Quantization-Aware Training (QAT)")
    print("   Note: QAT requires training and is best for fine-tuning")
    print("   This example shows the setup (actual training not performed)")

    backend.model = model

    try:
        qat_model = backend.quantize(type="qat")
        print(f"   ✓ QAT model prepared")
        print(f"   Ready for fine-tuning with quantization simulation")
        print(f"   After training, convert to quantized model for deployment")
    except Exception as e:
        print(f"   ⚠ QAT setup: {str(e)[:50]}...")

    # ============================================
    # Comprehensive Benchmark
    # ============================================
    print("\n7. Comprehensive Benchmark")
    print("   Testing with different batch sizes...\n")

    batch_sizes = [1, 4, 8, 16]

    print(f"   {'Batch Size':<12s} {'FP32 (ms)':<12s} {'INT8 (ms)':<12s} {'Speedup':<10s}")
    print("   " + "-" * 50)

    for bs in batch_sizes:
        test_input = torch.randn(bs, 3, 224, 224)

        # Original model
        orig_time, _ = benchmark_model(model, test_input, iterations=50)

        # Quantized model (if available)
        if static_model is not None:
            quant_time, _ = benchmark_model(static_model, test_input, iterations=50)
            speedup = orig_time / quant_time
            print(f"   {bs:<12d} {orig_time:<12.2f} {quant_time:<12.2f} {speedup:<10.2f}x")
        else:
            print(f"   {bs:<12d} {orig_time:<12.2f} {'N/A':<12s} {'N/A':<10s}")

    # ============================================
    # Summary and Recommendations
    # ============================================
    print("\n" + "=" * 80)
    print("✓ ResNet quantization demonstration completed!")
    print("\n  Summary:")
    print(f"  • Original model: {model_size_mb:.2f} MB, {original_time:.2f} ms/image")
    if static_model is not None:
        print(f"  • Quantized model: ~{model_size_mb/4:.2f} MB (estimated), {static_time:.2f} ms/image")
        print(f"  • Speedup: {original_time/static_time:.2f}x")

    print("\n  Recommendations for Production:")
    print("  1. Use Static Quantization for CNN models (ResNet, MobileNet, etc.)")
    print("  2. Collect representative calibration data from your dataset")
    print("  3. Use QAT for best accuracy (fine-tune with quantization)")
    print("  4. Test on target hardware (CPU/Mobile) for real speedup")
    print("  5. Validate accuracy on your validation set before deployment")

    print("\n  Typical compression results:")
    print("  • Model size: 4x smaller (FP32 → INT8)")
    print("  • Inference speed: 2-4x faster on CPU")
    print("  • Accuracy drop: <1% with proper calibration")
    print("=" * 80)


if __name__ == "__main__":
    main()
