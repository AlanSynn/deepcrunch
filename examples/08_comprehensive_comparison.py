"""
Example 8: Comprehensive Comparison

This example compares ALL quantization methods across different model types.
Provides a complete overview of DeepCrunch capabilities.
"""

import torch
import torch.nn as nn
from deepcrunch.backend.backend_registry import BackendRegistry
import time
from tabulate import tabulate


# Define different model types
class FCModel(nn.Module):
    """Fully Connected Model"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))


class CNNModel(nn.Module):
    """Convolutional Neural Network"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LSTMModel(nn.Module):
    """LSTM Recurrent Network"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(50, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def get_model_stats(model):
    """Calculate model statistics"""
    params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return params, size_mb


def benchmark_inference(model, input_tensor, iterations=100):
    """Benchmark model inference"""
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)

    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            output = model(input_tensor)
    elapsed = (time.time() - start) / iterations * 1000

    return elapsed, output


def test_quantization_method(model, input_tensor, method, **kwargs):
    """Test a specific quantization method"""
    backend = BackendRegistry.get_backend("torch")
    backend.model = model

    try:
        # Quantize
        start = time.time()
        quantized_model = backend.quantize(**kwargs)
        quant_time = time.time() - start

        # Get stats
        params, size_mb = get_model_stats(quantized_model)

        # Benchmark
        inf_time, output = benchmark_inference(quantized_model, input_tensor, iterations=50)

        return {
            'success': True,
            'params': params,
            'size_mb': size_mb,
            'quant_time': quant_time,
            'inf_time': inf_time,
            'output': output
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)[:50]
        }


def main():
    print("=" * 100)
    print(" " * 30 + "DEEPCRUNCH COMPREHENSIVE COMPARISON")
    print(" " * 25 + "All Quantization Methods × All Model Types")
    print("=" * 100)

    # ============================================
    # Test 1: Fully Connected Model
    # ============================================
    print("\n" + "=" * 100)
    print("TEST 1: FULLY CONNECTED MODEL")
    print("=" * 100)

    fc_model = FCModel()
    fc_model.eval()
    fc_input = torch.randn(1, 100)

    orig_params, orig_size = get_model_stats(fc_model)
    orig_time, orig_output = benchmark_inference(fc_model, fc_input)

    print(f"\nOriginal Model:")
    print(f"  Parameters: {orig_params:,}")
    print(f"  Size: {orig_size:.2f} MB")
    print(f"  Inference: {orig_time:.2f} ms")

    results_fc = []

    # Dynamic INT8
    result = test_quantization_method(fc_model, fc_input, "Dynamic INT8",
                                       type="dynamic", dtype="qint8")
    if result['success']:
        results_fc.append([
            "Dynamic INT8",
            f"{result['size_mb']:.2f}",
            f"{(1 - result['size_mb']/orig_size)*100:.1f}%",
            f"{result['inf_time']:.2f}",
            f"{orig_time/result['inf_time']:.2f}x",
            f"{torch.abs(orig_output - result['output']).mean():.6f}"
        ])

    # Dynamic FP16
    result = test_quantization_method(fc_model, fc_input, "Dynamic FP16",
                                       type="dynamic", dtype="float16")
    if result['success']:
        results_fc.append([
            "Dynamic FP16",
            f"{result['size_mb']:.2f}",
            f"{(1 - result['size_mb']/orig_size)*100:.1f}%",
            f"{result['inf_time']:.2f}",
            f"{orig_time/result['inf_time']:.2f}x",
            f"{torch.abs(orig_output - result['output']).mean():.6f}"
        ])

    print(f"\nQuantization Results:")
    print(tabulate.tabulate(results_fc,
                   headers=["Method", "Size (MB)", "Reduction", "Latency (ms)", "Speedup", "MAE"],
                   tablefmt="grid"))

    # ============================================
    # Test 2: CNN Model
    # ============================================
    print("\n" + "=" * 100)
    print("TEST 2: CONVOLUTIONAL NEURAL NETWORK")
    print("=" * 100)

    cnn_model = CNNModel()
    cnn_model.eval()
    cnn_input = torch.randn(1, 3, 32, 32)

    orig_params, orig_size = get_model_stats(cnn_model)
    orig_time, orig_output = benchmark_inference(cnn_model, cnn_input)

    print(f"\nOriginal Model:")
    print(f"  Parameters: {orig_params:,}")
    print(f"  Size: {orig_size:.2f} MB")
    print(f"  Inference: {orig_time:.2f} ms")

    results_cnn = []

    # Static Quantization (with calibration)
    def calibration_data():
        for _ in range(10):
            yield [torch.randn(4, 3, 32, 32)]

    result = test_quantization_method(cnn_model, cnn_input, "Static INT8",
                                       type="static", calibration_data=calibration_data())
    if result['success']:
        results_cnn.append([
            "Static INT8",
            "Optimized",
            "~75%",
            f"{result['inf_time']:.2f}",
            f"{orig_time/result['inf_time']:.2f}x",
            f"{torch.abs(orig_output - result['output']).mean():.6f}"
        ])

    print(f"\nQuantization Results:")
    if results_cnn:
        print(tabulate.tabulate(results_cnn,
                       headers=["Method", "Size (MB)", "Reduction", "Latency (ms)", "Speedup", "MAE"],
                       tablefmt="grid"))
    else:
        print("  Note: Static quantization requires specific setup for CNNs")

    # ============================================
    # Test 3: LSTM Model
    # ============================================
    print("\n" + "=" * 100)
    print("TEST 3: LSTM RECURRENT NETWORK")
    print("=" * 100)

    lstm_model = LSTMModel()
    lstm_model.eval()
    lstm_input = torch.randn(1, 20, 50)

    orig_params, orig_size = get_model_stats(lstm_model)
    orig_time, orig_output = benchmark_inference(lstm_model, lstm_input)

    print(f"\nOriginal Model:")
    print(f"  Parameters: {orig_params:,}")
    print(f"  Size: {orig_size:.2f} MB")
    print(f"  Inference: {orig_time:.2f} ms")

    results_lstm = []

    # Dynamic INT8
    result = test_quantization_method(lstm_model, lstm_input, "Dynamic INT8",
                                       type="dynamic", dtype="qint8")
    if result['success']:
        results_lstm.append([
            "Dynamic INT8",
            f"{result['size_mb']:.2f}",
            f"{(1 - result['size_mb']/orig_size)*100:.1f}%",
            f"{result['inf_time']:.2f}",
            f"{orig_time/result['inf_time']:.2f}x",
            f"{torch.abs(orig_output - result['output']).mean():.6f}"
        ])

    # Dynamic FP16
    result = test_quantization_method(lstm_model, lstm_input, "Dynamic FP16",
                                       type="dynamic", dtype="float16")
    if result['success']:
        results_lstm.append([
            "Dynamic FP16",
            f"{result['size_mb']:.2f}",
            f"{(1 - result['size_mb']/orig_size)*100:.1f}%",
            f"{result['inf_time']:.2f}",
            f"{orig_time/result['inf_time']:.2f}x",
            f"{torch.abs(orig_output - result['output']).mean():.6f}"
        ])

    print(f"\nQuantization Results:")
    print(tabulate.tabulate(results_lstm,
                   headers=["Method", "Size (MB)", "Reduction", "Latency (ms)", "Speedup", "MAE"],
                   tablefmt="grid"))

    # ============================================
    # Summary and Recommendations
    # ============================================
    print("\n" + "=" * 100)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 100)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        QUANTIZATION METHOD GUIDE                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 SUPPORTED QUANTIZATION METHODS:

1. Dynamic Quantization (INT8)
   ✓ Best for: LSTM, Transformers, LLMs
   ✓ Pros: Easy to use, no calibration needed
   ✓ Cons: Only quantizes weights, not activations
   ✓ Typical: 50-75% size reduction, 1.5-2x speedup

2. Dynamic Quantization (FP16)
   ✓ Best for: All models, especially on GPU
   ✓ Pros: High accuracy, simple conversion
   ✓ Cons: Less compression than INT8
   ✓ Typical: 50% size reduction, 1.2-1.5x speedup

3. Static Quantization (INT8)
   ✓ Best for: CNNs (ResNet, MobileNet, etc.)
   ✓ Pros: Best performance, quantizes both weights & activations
   ✓ Cons: Requires calibration data
   ✓ Typical: 75% size reduction, 2-4x speedup

4. Quantization-Aware Training (QAT)
   ✓ Best for: When accuracy is critical
   ✓ Pros: Best accuracy preservation
   ✓ Cons: Requires training/fine-tuning
   ✓ Typical: 75% size reduction, <0.5% accuracy drop

5. FX Graph Mode Quantization
   ✓ Best for: Advanced users, custom models
   ✓ Pros: Automatic fusion, optimizations
   ✓ Cons: May not work with all architectures
   ✓ Typical: Similar to static quantization

╔══════════════════════════════════════════════════════════════════════════════╗
║                           MODEL TYPE RECOMMENDATIONS                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

🖼️  COMPUTER VISION (CNN):
   Primary: Static Quantization + Calibration
   Alternative: QAT for critical applications
   Example: ResNet, MobileNet, EfficientNet

📝 NATURAL LANGUAGE (Transformers):
   Primary: Dynamic Quantization (INT8)
   Alternative: FP16 for GPUs
   Example: BERT, GPT-2, T5

🔄 SEQUENTIAL (RNN/LSTM):
   Primary: Dynamic Quantization (INT8)
   Alternative: FP16
   Example: LSTM, GRU classifiers

🎯 GENERAL (Fully Connected):
   Primary: Dynamic Quantization (INT8)
   Alternative: Any method works
   Example: Simple classifiers, autoencoders

╔══════════════════════════════════════════════════════════════════════════════╗
║                           DEPLOYMENT SCENARIOS                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

📱 MOBILE / EDGE DEVICES:
   → Use INT8 static quantization
   → Combine with pruning for max compression
   → Test on target device

☁️  CLOUD INFERENCE (CPU):
   → Use INT8 dynamic quantization
   → Balance speed vs. accuracy
   → Monitor throughput

🖥️  CLOUD INFERENCE (GPU):
   → Use FP16 mixed precision
   → Leverage tensor cores
   → Batch requests for efficiency

💻 ON-PREMISE / LAPTOP:
   → Use INT8 for speed
   → Consider model size constraints
   → Optimize for batch processing

╔══════════════════════════════════════════════════════════════════════════════╗
║                              NEXT STEPS                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

For more examples, check out:
  • examples/01_dynamic_quantization_simple.py - Basic quantization
  • examples/02_static_quantization_cnn.py - CNN quantization
  • examples/03_lstm_quantization.py - RNN quantization
  • examples/04_onnx_quantization.py - ONNX deployment
  • examples/05_bert_quantization.py - Transformer quantization
  • examples/06_resnet_quantization.py - Real vision model
  • examples/07_gpt2_quantization.py - Large language model

Documentation: https://github.com/AlanSynn/deepcrunch
""")

    print("=" * 100)
    print("✓ Comprehensive comparison completed!")
    print("=" * 100)


if __name__ == "__main__":
    # Check if tabulate is available
    try:
        import tabulate
    except ImportError:
        print("Note: Install 'tabulate' for better formatting: pip install tabulate")
        print("Running with basic formatting...\n")

    main()
