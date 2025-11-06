"""End-to-end tests for accuracy validation and performance measurement."""

import os
import time

import numpy as np
import onnxruntime as ort
import pytest
import torch

from deepcrunch.backend.backend_registry import BackendRegistry


class TestAccuracyValidation:
    """Test accuracy validation for quantized models."""

    def test_torch_dynamic_accuracy_on_dataset(self, simple_linear_model):
        """Test accuracy of dynamically quantized model on a dataset."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        # Create test dataset
        test_data = [torch.randn(1, 10) for _ in range(100)]

        # Get original outputs
        original_outputs = []
        for data in test_data:
            with torch.no_grad():
                output = simple_linear_model(data)
                original_outputs.append(output)

        # Quantize model
        quantized_model = backend.quantize(type="dynamic", dtype="qint8")

        # Get quantized outputs
        quantized_outputs = []
        for data in test_data:
            with torch.no_grad():
                output = quantized_model(data)
                quantized_outputs.append(output)

        # Calculate accuracy metrics
        mse_values = []
        relative_errors = []

        for orig, quant in zip(original_outputs, quantized_outputs):
            orig_np = orig.numpy()
            quant_np = quant.numpy()

            # Mean squared error
            mse = np.mean((orig_np - quant_np) ** 2)
            mse_values.append(mse)

            # Relative error
            rel_error = np.mean(np.abs(orig_np - quant_np) / (np.abs(orig_np) + 1e-8))
            relative_errors.append(rel_error)

        avg_mse = np.mean(mse_values)
        avg_rel_error = np.mean(relative_errors)

        print(f"\nAccuracy metrics:")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average Relative Error: {avg_rel_error:.6f}")

        # Assert reasonable accuracy
        assert avg_mse < 0.1, f"MSE too high: {avg_mse}"
        assert avg_rel_error < 0.05, f"Relative error too high: {avg_rel_error}"

    def test_onnx_dynamic_accuracy_on_dataset(self, onnx_model_path, temp_dir):
        """Test accuracy of dynamically quantized ONNX model on a dataset."""
        # Original model
        original_session = ort.InferenceSession(onnx_model_path)
        input_name = original_session.get_inputs()[0].name

        # Create test dataset
        test_data = [np.random.randn(1, 10).astype(np.float32) for _ in range(100)]

        # Get original outputs
        original_outputs = [
            original_session.run(None, {input_name: data})[0] for data in test_data
        ]

        # Quantize model
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path
        output_path = str(temp_dir / "quantized_accuracy_test.onnx")
        backend.quantize(type="dynamic", output_path=output_path)

        # Get quantized outputs
        quantized_session = ort.InferenceSession(output_path)
        quantized_outputs = [
            quantized_session.run(None, {input_name: data})[0] for data in test_data
        ]

        # Calculate accuracy metrics
        mse_values = []
        for orig, quant in zip(original_outputs, quantized_outputs):
            mse = np.mean((orig - quant) ** 2)
            mse_values.append(mse)

        avg_mse = np.mean(mse_values)
        print(f"\nONNX Accuracy - Average MSE: {avg_mse:.6f}")

        # Assert reasonable accuracy
        assert avg_mse < 0.1, f"MSE too high: {avg_mse}"

    def test_accuracy_threshold_validation(self, simple_linear_model):
        """Test that quantization maintains accuracy within threshold."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        # Test on multiple inputs
        test_inputs = [torch.randn(5, 10) for _ in range(20)]

        # Quantize model
        quantized_model = backend.quantize(type="dynamic", dtype="qint8")

        # Count predictions within threshold
        within_threshold = 0
        total = 0

        for test_input in test_inputs:
            with torch.no_grad():
                original_output = simple_linear_model(test_input)
                quantized_output = quantized_model(test_input)

            # Check if outputs are close
            if torch.allclose(original_output, quantized_output, rtol=0.1, atol=0.1):
                within_threshold += test_input.size(0)
            total += test_input.size(0)

        accuracy_rate = within_threshold / total
        print(f"\nAccuracy rate (within threshold): {accuracy_rate:.2%}")

        # At least 80% should be within threshold
        assert accuracy_rate >= 0.8, f"Accuracy rate too low: {accuracy_rate:.2%}"


class TestPerformanceMetrics:
    """Test performance metrics for quantized models."""

    def test_torch_inference_latency(self, simple_linear_model, sample_linear_input):
        """Test inference latency comparison between original and quantized model."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        # Warm up
        for _ in range(10):
            with torch.no_grad():
                simple_linear_model(sample_linear_input)

        # Measure original model latency
        original_times = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                simple_linear_model(sample_linear_input)
            end = time.perf_counter()
            original_times.append(end - start)

        # Quantize model
        quantized_model = backend.quantize(type="dynamic", dtype="qint8")

        # Warm up quantized model
        for _ in range(10):
            with torch.no_grad():
                quantized_model(sample_linear_input)

        # Measure quantized model latency
        quantized_times = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                quantized_model(sample_linear_input)
            end = time.perf_counter()
            quantized_times.append(end - start)

        avg_original = np.mean(original_times) * 1000  # Convert to ms
        avg_quantized = np.mean(quantized_times) * 1000

        print(f"\nLatency comparison:")
        print(f"Original model: {avg_original:.4f} ms")
        print(f"Quantized model: {avg_quantized:.4f} ms")
        print(f"Speedup: {avg_original / avg_quantized:.2f}x")

        # Both should complete in reasonable time
        assert avg_original < 1000  # Less than 1 second
        assert avg_quantized < 1000

    def test_onnx_inference_latency(self, onnx_model_path, temp_dir):
        """Test inference latency comparison for ONNX models."""
        # Original model
        original_session = ort.InferenceSession(onnx_model_path)
        input_name = original_session.get_inputs()[0].name
        sample_input = np.random.randn(1, 10).astype(np.float32)

        # Warm up
        for _ in range(10):
            original_session.run(None, {input_name: sample_input})

        # Measure original model latency
        original_times = []
        for _ in range(100):
            start = time.perf_counter()
            original_session.run(None, {input_name: sample_input})
            end = time.perf_counter()
            original_times.append(end - start)

        # Quantize model
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path
        output_path = str(temp_dir / "quantized_perf_test.onnx")
        backend.quantize(type="dynamic", output_path=output_path)

        # Quantized model
        quantized_session = ort.InferenceSession(output_path)

        # Warm up quantized model
        for _ in range(10):
            quantized_session.run(None, {input_name: sample_input})

        # Measure quantized model latency
        quantized_times = []
        for _ in range(100):
            start = time.perf_counter()
            quantized_session.run(None, {input_name: sample_input})
            end = time.perf_counter()
            quantized_times.append(end - start)

        avg_original = np.mean(original_times) * 1000  # Convert to ms
        avg_quantized = np.mean(quantized_times) * 1000

        print(f"\nONNX Latency comparison:")
        print(f"Original model: {avg_original:.4f} ms")
        print(f"Quantized model: {avg_quantized:.4f} ms")

        # Both should complete in reasonable time
        assert avg_original < 1000
        assert avg_quantized < 1000

    def test_model_size_metrics(self, simple_linear_model, temp_dir):
        """Test model size metrics for quantized models."""
        # Save original model
        original_path = temp_dir / "original_size_test.pt"
        torch.save(simple_linear_model.state_dict(), original_path)
        original_size = os.path.getsize(original_path)

        # Quantize model
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model
        quantized_model = backend.quantize(type="dynamic", dtype="qint8")

        # Save quantized model
        quantized_path = temp_dir / "quantized_size_test.pt"
        torch.save(quantized_model.state_dict(), quantized_path)
        quantized_size = os.path.getsize(quantized_path)

        # Calculate compression ratio
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0

        print(f"\nModel size metrics:")
        print(f"Original size: {original_size} bytes")
        print(f"Quantized size: {quantized_size} bytes")
        print(f"Compression ratio: {compression_ratio:.2f}x")

        # Quantized model should exist and be reasonable size
        assert quantized_size > 0
        assert quantized_size <= original_size * 1.5  # Allow some overhead

    def test_throughput_measurement(self, simple_linear_model):
        """Test throughput (samples/second) for quantized models."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        batch_size = 32
        num_batches = 100
        test_input = torch.randn(batch_size, 10)

        # Measure original model throughput
        start = time.perf_counter()
        for _ in range(num_batches):
            with torch.no_grad():
                simple_linear_model(test_input)
        original_time = time.perf_counter() - start
        original_throughput = (num_batches * batch_size) / original_time

        # Quantize model
        quantized_model = backend.quantize(type="dynamic", dtype="qint8")

        # Measure quantized model throughput
        start = time.perf_counter()
        for _ in range(num_batches):
            with torch.no_grad():
                quantized_model(test_input)
        quantized_time = time.perf_counter() - start
        quantized_throughput = (num_batches * batch_size) / quantized_time

        print(f"\nThroughput comparison:")
        print(f"Original: {original_throughput:.2f} samples/sec")
        print(f"Quantized: {quantized_throughput:.2f} samples/sec")

        # Both should process samples
        assert original_throughput > 0
        assert quantized_throughput > 0


class TestMemoryUsage:
    """Test memory usage for quantized models."""

    def test_model_memory_footprint(self, simple_linear_model):
        """Test memory footprint of quantized models."""
        import sys

        # Calculate original model size in memory
        def get_model_size(model):
            total_size = 0
            for param in model.parameters():
                total_size += param.nelement() * param.element_size()
            for buffer in model.buffers():
                total_size += buffer.nelement() * buffer.element_size()
            return total_size

        original_size = get_model_size(simple_linear_model)

        # Quantize model
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model
        quantized_model = backend.quantize(type="dynamic", dtype="qint8")

        quantized_size = get_model_size(quantized_model)

        print(f"\nMemory footprint:")
        print(f"Original: {original_size / 1024:.2f} KB")
        print(f"Quantized: {quantized_size / 1024:.2f} KB")
        print(f"Reduction: {(1 - quantized_size / original_size) * 100:.2f}%")

        # Quantized model should fit in memory
        assert quantized_size > 0


class TestBatchSizePerformance:
    """Test performance across different batch sizes."""

    def test_various_batch_sizes(self, simple_linear_model):
        """Test quantized model performance with various batch sizes."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        # Quantize model
        quantized_model = backend.quantize(type="dynamic", dtype="qint8")

        batch_sizes = [1, 8, 16, 32, 64]
        results = {}

        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 10)

            # Measure latency
            times = []
            for _ in range(50):
                start = time.perf_counter()
                with torch.no_grad():
                    quantized_model(test_input)
                end = time.perf_counter()
                times.append(end - start)

            avg_time = np.mean(times) * 1000  # ms
            throughput = batch_size / np.mean(times)

            results[batch_size] = {
                "latency": avg_time,
                "throughput": throughput,
            }

        print("\nBatch size performance:")
        for bs, metrics in results.items():
            print(
                f"Batch size {bs}: {metrics['latency']:.4f} ms, "
                f"{metrics['throughput']:.2f} samples/sec"
            )

        # All batch sizes should work
        assert all(r["latency"] > 0 for r in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
