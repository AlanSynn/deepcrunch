"""
Example 4: ONNX Quantization

This example demonstrates how to quantize models in ONNX format.
ONNX quantization is ideal for deployment and cross-platform inference.
"""

import os
import tempfile
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from deepcrunch.backend.backend_registry import BackendRegistry


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 30)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(30, 5)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


def export_to_onnx(model, input_shape, onnx_path):
    """Export PyTorch model to ONNX format"""
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=13
    )
    return dummy_input


def get_model_size(path):
    """Get model file size in MB"""
    return os.path.getsize(path) / (1024 * 1024)


def main():
    print("=" * 80)
    print("ONNX Quantization Example")
    print("=" * 80)

    # Create temporary directory for ONNX files
    temp_dir = tempfile.mkdtemp()

    # Create and export PyTorch model to ONNX
    print("\n1. Exporting PyTorch Model to ONNX")
    model = SimpleModel()
    model.eval()

    original_onnx_path = os.path.join(temp_dir, "model_fp32.onnx")
    sample_input = export_to_onnx(model, (1, 10), original_onnx_path)

    print(f"   ✓ ONNX model saved to: {original_onnx_path}")
    print(f"   Model size: {get_model_size(original_onnx_path):.4f} MB")

    # Test original ONNX model
    print("\n2. Testing Original ONNX Model")
    session = ort.InferenceSession(original_onnx_path)
    input_name = session.get_inputs()[0].name
    sample_input_np = sample_input.numpy()

    original_output = session.run(None, {input_name: sample_input_np})[0]
    print(f"   Input shape: {sample_input_np.shape}")
    print(f"   Output shape: {original_output.shape}")
    print(f"   Sample output: {original_output[0, :3]}")

    # Method 1: Dynamic Quantization
    print("\n3. Dynamic Quantization (INT8)")
    print("   Converting weights to INT8...")

    backend = BackendRegistry.get_backend("onnx")
    backend.model = original_onnx_path

    dynamic_onnx_path = os.path.join(temp_dir, "model_dynamic_int8.onnx")
    backend.quantize(
        type="dynamic",
        output_path=dynamic_onnx_path,
        weight_type="QInt8"
    )

    dynamic_size = get_model_size(dynamic_onnx_path)
    print(f"   ✓ Quantized model saved to: {dynamic_onnx_path}")
    print(f"   Model size: {dynamic_size:.4f} MB")
    print(f"   Size reduction: {(1 - dynamic_size/get_model_size(original_onnx_path)) * 100:.2f}%")

    # Test dynamic quantized model
    dynamic_session = ort.InferenceSession(dynamic_onnx_path)
    dynamic_output = dynamic_session.run(None, {input_name: sample_input_np})[0]

    diff = np.abs(original_output - dynamic_output).mean()
    print(f"   Output difference (MAE): {diff:.6f}")

    # Method 2: Static Quantization
    print("\n4. Static Quantization with Calibration")
    print("   Requires calibration data for better accuracy...")

    # Create calibration data reader
    class CalibrationDataReader:
        def __init__(self, num_samples=20):
            self.data = [np.random.randn(1, 10).astype(np.float32) for _ in range(num_samples)]
            self.index = 0

        def get_next(self):
            if self.index < len(self.data):
                input_data = {"input": self.data[self.index]}
                self.index += 1
                return input_data
            return None

    backend.model = original_onnx_path
    static_onnx_path = os.path.join(temp_dir, "model_static_int8.onnx")

    calibration_reader = CalibrationDataReader(num_samples=20)
    backend.quantize(
        type="static",
        output_path=static_onnx_path,
        calibration_data_reader=calibration_reader
    )

    static_size = get_model_size(static_onnx_path)
    print(f"   ✓ Quantized model saved to: {static_onnx_path}")
    print(f"   Model size: {static_size:.4f} MB")
    print(f"   Size reduction: {(1 - static_size/get_model_size(original_onnx_path)) * 100:.2f}%")

    # Test static quantized model
    static_session = ort.InferenceSession(static_onnx_path)
    static_output = static_session.run(None, {input_name: sample_input_np})[0]

    diff = np.abs(original_output - static_output).mean()
    print(f"   Output difference (MAE): {diff:.6f}")

    # Method 3: Float16 Conversion
    print("\n5. Float16 Conversion")
    print("   Converting model to FP16 precision...")

    backend.model = original_onnx_path
    float16_onnx_path = os.path.join(temp_dir, "model_fp16.onnx")

    backend.quantize(
        type="float16",
        output_path=float16_onnx_path
    )

    float16_size = get_model_size(float16_onnx_path)
    print(f"   ✓ FP16 model saved to: {float16_onnx_path}")
    print(f"   Model size: {float16_size:.4f} MB")
    print(f"   Size reduction: {(1 - float16_size/get_model_size(original_onnx_path)) * 100:.2f}%")

    # Test float16 model
    float16_session = ort.InferenceSession(float16_onnx_path)
    float16_output = float16_session.run(None, {input_name: sample_input_np})[0]

    diff = np.abs(original_output - float16_output).mean()
    print(f"   Output difference (MAE): {diff:.6f}")

    # Benchmark inference speed
    print("\n6. Inference Speed Benchmark")
    iterations = 1000

    models = {
        "FP32 (Original)": session,
        "INT8 (Dynamic)": dynamic_session,
        "INT8 (Static)": static_session,
        "FP16": float16_session
    }

    print(f"   Running {iterations} iterations for each model...\n")

    import time
    for name, model_session in models.items():
        start = time.time()
        for _ in range(iterations):
            model_session.run(None, {input_name: sample_input_np})
        elapsed = (time.time() - start) / iterations * 1000

        print(f"   {name:20s}: {elapsed:.4f} ms/inference")

    # Summary
    print("\n7. Summary")
    print("   " + "-" * 60)
    print(f"   {'Method':<20s} {'Size (MB)':<12s} {'Size Reduction':<15s} {'Accuracy'}")
    print("   " + "-" * 60)

    original_size = get_model_size(original_onnx_path)
    print(f"   {'FP32 (Original)':<20s} {original_size:<12.4f} {'0.00%':<15s} {'Baseline'}")

    for method, path, output in [
        ("INT8 (Dynamic)", dynamic_onnx_path, dynamic_output),
        ("INT8 (Static)", static_onnx_path, static_output),
        ("FP16", float16_onnx_path, float16_output)
    ]:
        size = get_model_size(path)
        reduction = (1 - size/original_size) * 100
        diff = np.abs(original_output - output).mean()
        print(f"   {method:<20s} {size:<12.4f} {reduction:<14.2f}% {'MAE=' + f'{diff:.6f}'}")

    print("   " + "-" * 60)

    print("\n" + "=" * 80)
    print("✓ ONNX quantization completed successfully!")
    print("  Recommendations:")
    print("  • Use Dynamic INT8 for easy deployment with good compression")
    print("  • Use Static INT8 for best accuracy with calibration data")
    print("  • Use FP16 for GPU deployment and mixed-precision training")
    print("=" * 80)

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
