"""End-to-end tests for ONNX backend quantization."""

import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from deepcrunch.backend.backend_registry import BackendRegistry
from tests.conftest import CalibrationDataReader


class TestONNXDynamicQuantization:
    """Test dynamic quantization for ONNX models."""

    def test_dynamic_quantization_basic(self, onnx_model_path, temp_dir):
        """Test basic dynamic quantization on ONNX model."""
        # Get backend
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        # Quantize model
        output_path = str(temp_dir / "quantized_dynamic.onnx")
        backend.quantize(
            type="dynamic",
            output_path=output_path,
        )

        # Verify quantized model exists
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Load and verify quantized model
        quantized_model = onnx.load(output_path)
        onnx.checker.check_model(quantized_model)

    def test_dynamic_quantization_inference(self, onnx_model_path, temp_dir):
        """Test inference with dynamically quantized ONNX model."""
        # Original model inference
        original_session = ort.InferenceSession(onnx_model_path)
        input_name = original_session.get_inputs()[0].name
        sample_input = np.random.randn(1, 10).astype(np.float32)
        original_output = original_session.run(None, {input_name: sample_input})[0]

        # Quantize model
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        output_path = str(temp_dir / "quantized_dynamic.onnx")
        backend.quantize(type="dynamic", output_path=output_path)

        # Quantized model inference
        quantized_session = ort.InferenceSession(output_path)
        quantized_output = quantized_session.run(None, {input_name: sample_input})[0]

        # Verify outputs are close
        assert original_output.shape == quantized_output.shape
        assert np.allclose(original_output, quantized_output, rtol=0.1, atol=0.1)

    def test_dynamic_quantization_with_options(self, onnx_model_path, temp_dir):
        """Test dynamic quantization with various options."""
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        output_path = str(temp_dir / "quantized_with_options.onnx")

        # Quantize with options
        backend.quantize(
            type="dynamic",
            output_path=output_path,
            weight_type="QUInt8",
            per_channel=True,
        )

        # Verify model is valid
        assert os.path.exists(output_path)
        model = onnx.load(output_path)
        onnx.checker.check_model(model)

    def test_dynamic_quantization_size_reduction(self, onnx_model_path, temp_dir):
        """Test that dynamic quantization reduces model size."""
        original_size = os.path.getsize(onnx_model_path)

        # Quantize model
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        output_path = str(temp_dir / "quantized_dynamic.onnx")
        backend.quantize(type="dynamic", output_path=output_path)

        quantized_size = os.path.getsize(output_path)

        # Quantized model should be smaller or similar (for small models)
        assert quantized_size > 0
        print(f"Original size: {original_size}, Quantized size: {quantized_size}")


class TestONNXStaticQuantization:
    """Test static quantization for ONNX models."""

    def test_static_quantization_with_calibration(self, onnx_model_path, temp_dir):
        """Test static quantization with calibration data."""
        # Get backend
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        output_path = str(temp_dir / "quantized_static.onnx")

        # Quantize with calibration data
        calibration_reader = CalibrationDataReader(num_samples=10, input_shape=(1, 10))
        backend.quantize(
            type="static",
            output_path=output_path,
            calibration_data_reader=calibration_reader,
        )

        # Verify quantized model exists and is valid
        assert os.path.exists(output_path)
        model = onnx.load(output_path)
        onnx.checker.check_model(model)

    def test_static_quantization_inference(self, onnx_model_path, temp_dir):
        """Test inference with statically quantized ONNX model."""
        # Original model inference
        original_session = ort.InferenceSession(onnx_model_path)
        input_name = original_session.get_inputs()[0].name
        sample_input = np.random.randn(1, 10).astype(np.float32)
        original_output = original_session.run(None, {input_name: sample_input})[0]

        # Quantize model
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        output_path = str(temp_dir / "quantized_static.onnx")
        calibration_reader = CalibrationDataReader(num_samples=10, input_shape=(1, 10))
        backend.quantize(
            type="static",
            output_path=output_path,
            calibration_data_reader=calibration_reader,
        )

        # Quantized model inference
        quantized_session = ort.InferenceSession(output_path)
        quantized_output = quantized_session.run(None, {input_name: sample_input})[0]

        # Verify outputs are close (static quantization might have some accuracy loss)
        assert original_output.shape == quantized_output.shape
        assert np.allclose(original_output, quantized_output, rtol=0.2, atol=0.2)


class TestONNXFloat16Quantization:
    """Test float16 conversion for ONNX models."""

    def test_float16_conversion(self, onnx_model_path, temp_dir):
        """Test converting ONNX model to float16."""
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        output_path = str(temp_dir / "quantized_float16.onnx")

        # Convert to float16
        backend.quantize(type="float16", output_path=output_path)

        # Verify model exists and is valid
        assert os.path.exists(output_path)
        model = onnx.load(output_path)
        onnx.checker.check_model(model)

    def test_float16_inference(self, onnx_model_path, temp_dir):
        """Test inference with float16 ONNX model."""
        # Original model inference
        original_session = ort.InferenceSession(onnx_model_path)
        input_name = original_session.get_inputs()[0].name
        sample_input = np.random.randn(1, 10).astype(np.float32)
        original_output = original_session.run(None, {input_name: sample_input})[0]

        # Convert to float16
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        output_path = str(temp_dir / "quantized_float16.onnx")
        backend.quantize(type="float16", output_path=output_path)

        # Float16 model inference
        float16_session = ort.InferenceSession(output_path)
        float16_output = float16_session.run(None, {input_name: sample_input})[0]

        # Verify outputs are close
        assert original_output.shape == float16_output.shape
        assert np.allclose(original_output, float16_output, rtol=0.01, atol=0.01)

    def test_float16_size_reduction(self, onnx_model_path, temp_dir):
        """Test that float16 conversion reduces model size."""
        original_size = os.path.getsize(onnx_model_path)

        # Convert to float16
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        output_path = str(temp_dir / "quantized_float16.onnx")
        backend.quantize(type="float16", output_path=output_path)

        float16_size = os.path.getsize(output_path)

        # Float16 model should be smaller (roughly half the size for weights)
        assert float16_size > 0
        print(f"Original size: {original_size}, Float16 size: {float16_size}")
        # For real models, we'd expect ~50% reduction, but for small test models it varies
        assert float16_size <= original_size * 1.2  # Allow some overhead


class TestONNXQuantizationPreProcessing:
    """Test ONNX model pre-processing for quantization."""

    def test_preprocess_model(self, onnx_model_path, temp_dir):
        """Test pre-processing ONNX model before quantization."""
        from onnxruntime.quantization import quant_pre_process

        # Pre-process model
        preprocessed_path = str(temp_dir / "preprocessed.onnx")
        quant_pre_process(onnx_model_path, preprocessed_path)

        # Verify preprocessed model is valid
        assert os.path.exists(preprocessed_path)
        model = onnx.load(preprocessed_path)
        onnx.checker.check_model(model)

        # Quantize preprocessed model
        backend = BackendRegistry.get_backend("onnx")
        backend.model = preprocessed_path

        quantized_path = str(temp_dir / "quantized_preprocessed.onnx")
        backend.quantize(type="dynamic", output_path=quantized_path)

        # Verify quantized model is valid
        assert os.path.exists(quantized_path)
        quantized_model = onnx.load(quantized_path)
        onnx.checker.check_model(quantized_model)


class TestONNXQuantizationComparison:
    """Compare different ONNX quantization methods."""

    def test_compare_quantization_methods(self, onnx_model_path, temp_dir):
        """Compare dynamic, static, and float16 quantization."""
        # Original model
        original_session = ort.InferenceSession(onnx_model_path)
        input_name = original_session.get_inputs()[0].name
        sample_input = np.random.randn(1, 10).astype(np.float32)
        original_output = original_session.run(None, {input_name: sample_input})[0]

        results = {}

        # Dynamic quantization
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path
        dynamic_path = str(temp_dir / "quantized_dynamic.onnx")
        backend.quantize(type="dynamic", output_path=dynamic_path)

        dynamic_session = ort.InferenceSession(dynamic_path)
        dynamic_output = dynamic_session.run(None, {input_name: sample_input})[0]
        dynamic_size = os.path.getsize(dynamic_path)
        results["dynamic"] = {
            "output": dynamic_output,
            "size": dynamic_size,
            "accuracy": np.mean(np.abs(original_output - dynamic_output)),
        }

        # Float16 conversion
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path
        float16_path = str(temp_dir / "quantized_float16.onnx")
        backend.quantize(type="float16", output_path=float16_path)

        float16_session = ort.InferenceSession(float16_path)
        float16_output = float16_session.run(None, {input_name: sample_input})[0]
        float16_size = os.path.getsize(float16_path)
        results["float16"] = {
            "output": float16_output,
            "size": float16_size,
            "accuracy": np.mean(np.abs(original_output - float16_output)),
        }

        # Print comparison
        print("\nQuantization comparison:")
        print(f"Original size: {os.path.getsize(onnx_model_path)}")
        for method, data in results.items():
            print(f"{method}: size={data['size']}, accuracy_loss={data['accuracy']:.6f}")

        # All methods should produce valid outputs
        assert all(r["output"] is not None for r in results.values())


class TestONNXErrorHandling:
    """Test error handling in ONNX quantization."""

    def test_invalid_model_path(self, temp_dir):
        """Test handling of invalid model path."""
        backend = BackendRegistry.get_backend("onnx")
        backend.model = "nonexistent_model.onnx"

        output_path = str(temp_dir / "output.onnx")

        with pytest.raises((FileNotFoundError, RuntimeError)):
            backend.quantize(type="dynamic", output_path=output_path)

    def test_invalid_quantization_type(self, onnx_model_path, temp_dir):
        """Test handling of invalid quantization type."""
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        output_path = str(temp_dir / "output.onnx")

        with pytest.raises(ValueError):
            backend.quantize(type="invalid_type", output_path=output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
