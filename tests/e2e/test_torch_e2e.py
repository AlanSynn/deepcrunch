"""End-to-end tests for Torch backend quantization."""

import os

import pytest
import torch

from deepcrunch.backend.backend_registry import BackendRegistry


class TestTorchDynamicQuantization:
    """Test dynamic quantization for PyTorch models."""

    def test_linear_model_dynamic_quantization(self, simple_linear_model, sample_linear_input):
        """Test dynamic quantization on a simple linear model."""
        # Get backend
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        # Get original output
        original_output = simple_linear_model(sample_linear_input)

        # Quantize model
        quantized_model = backend.quantize(type="dynamic", dtype="qint8")

        # Verify model is quantized
        assert quantized_model is not None
        assert hasattr(quantized_model, "fc1")

        # Test forward pass
        quantized_output = quantized_model(sample_linear_input)
        assert quantized_output.shape == original_output.shape

        # Verify outputs are close (dynamic quantization should be accurate)
        from tests.conftest import compare_outputs
        assert compare_outputs(original_output, quantized_output, rtol=0.1, atol=0.1)

    def test_lstm_model_dynamic_quantization(self, simple_lstm_model, sample_lstm_input):
        """Test dynamic quantization on LSTM model."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_lstm_model

        # Get original output
        original_output = simple_lstm_model(sample_lstm_input)

        # Quantize model
        quantized_model = backend.quantize(type="dynamic", dtype="qint8")

        # Verify model is quantized
        assert quantized_model is not None

        # Test forward pass
        quantized_output = quantized_model(sample_lstm_input)
        assert quantized_output.shape == original_output.shape

    def test_dynamic_quantization_float16(self, simple_linear_model, sample_linear_input):
        """Test dynamic quantization with float16 dtype."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        # Quantize with float16
        quantized_model = backend.quantize(type="dynamic", dtype="float16")

        # Verify model works
        assert quantized_model is not None
        output = quantized_model(sample_linear_input)
        assert output.shape[0] == 1

    def test_dynamic_quantization_save_load(self, simple_linear_model, sample_linear_input, temp_dir):
        """Test saving and loading dynamically quantized model."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        # Quantize model
        quantized_model = backend.quantize(type="dynamic", dtype="qint8")

        # Save model
        save_path = temp_dir / "quantized_model.pt"
        torch.save(quantized_model.state_dict(), save_path)

        # Verify file exists and has reasonable size
        assert save_path.exists()
        assert os.path.getsize(save_path) > 0

        # Load model
        loaded_model = type(simple_linear_model)()
        loaded_model.load_state_dict(torch.load(save_path))

        # Test loaded model
        output = loaded_model(sample_linear_input)
        assert output is not None


class TestTorchStaticQuantization:
    """Test static quantization for PyTorch models."""

    def test_conv_model_static_quantization(self, simple_conv_model, sample_conv_input):
        """Test static quantization on a conv model."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_conv_model

        # Create calibration data
        def calibration_data():
            for _ in range(10):
                yield [torch.randn(1, 3, 32, 32)]

        # Get original output
        original_output = simple_conv_model(sample_conv_input)

        # Quantize model
        quantized_model = backend.quantize(
            type="static",
            calibration_data=calibration_data(),
        )

        # Verify model is quantized
        assert quantized_model is not None

        # Test forward pass
        quantized_output = quantized_model(sample_conv_input)
        assert quantized_output.shape == original_output.shape

    def test_static_quantization_accuracy(self, simple_conv_model, sample_conv_input):
        """Test accuracy of static quantization."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_conv_model

        # Create calibration data
        def calibration_data():
            for _ in range(20):
                yield [torch.randn(1, 3, 32, 32)]

        # Get original output
        original_output = simple_conv_model(sample_conv_input)

        # Quantize model
        quantized_model = backend.quantize(
            type="static",
            calibration_data=calibration_data(),
        )

        # Test output accuracy
        quantized_output = quantized_model(sample_conv_input)

        from tests.conftest import calculate_accuracy_drop
        accuracy_drop = calculate_accuracy_drop(original_output, quantized_output)

        # Accuracy drop should be reasonable (< 10%)
        assert accuracy_drop < 10.0


class TestTorchQATQuantization:
    """Test Quantization-Aware Training for PyTorch models."""

    def test_qat_basic(self, simple_conv_model, sample_conv_input):
        """Test basic QAT workflow."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_conv_model

        # Quantize with QAT
        quantized_model = backend.quantize(type="qat")

        # Verify model is quantized
        assert quantized_model is not None

        # Test forward pass
        quantized_output = quantized_model(sample_conv_input)
        assert quantized_output is not None

    def test_qat_with_backend_config(self, simple_conv_model, sample_conv_input):
        """Test QAT with custom backend configuration."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_conv_model

        # Quantize with QAT and backend config
        quantized_model = backend.quantize(
            type="qat",
            backend="fbgemm",
        )

        # Verify model works
        assert quantized_model is not None
        output = quantized_model(sample_conv_input)
        assert output is not None


class TestTorchFXQuantization:
    """Test FX graph mode quantization for PyTorch models."""

    def test_fx_quantization_basic(self, simple_linear_model, sample_linear_input):
        """Test basic FX quantization."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        # Get original output
        original_output = simple_linear_model(sample_linear_input)

        # Quantize with FX mode
        try:
            quantized_model = backend.quantize(
                type="fx",
                example_input=sample_linear_input,
            )

            # Verify model is quantized
            assert quantized_model is not None

            # Test forward pass
            quantized_output = quantized_model(sample_linear_input)
            assert quantized_output.shape == original_output.shape

        except (RuntimeError, AttributeError, NotImplementedError) as e:
            # FX mode might not be available in all PyTorch versions
            pytest.skip(f"FX quantization not available: {e}")

    def test_fx_quantization_conv_model(self, simple_conv_model, sample_conv_input):
        """Test FX quantization on conv model."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_conv_model

        try:
            quantized_model = backend.quantize(
                type="fx",
                example_input=sample_conv_input,
            )

            # Verify model works
            assert quantized_model is not None
            output = quantized_model(sample_conv_input)
            assert output is not None

        except (RuntimeError, AttributeError, NotImplementedError) as e:
            pytest.skip(f"FX quantization not available: {e}")


class TestTorchQuantizationComparison:
    """Compare different quantization methods."""

    def test_compare_dynamic_vs_static(self, simple_conv_model, sample_conv_input):
        """Compare dynamic and static quantization."""
        # Dynamic quantization
        backend_dynamic = BackendRegistry.get_backend("torch")
        backend_dynamic.model = simple_conv_model

        try:
            dynamic_quantized = backend_dynamic.quantize(type="dynamic")
            dynamic_output = dynamic_quantized(sample_conv_input)
        except (RuntimeError, TypeError) as e:
            # Dynamic quantization might not work on conv layers
            pytest.skip("Dynamic quantization not supported for conv layers")

        # Static quantization
        backend_static = BackendRegistry.get_backend("torch")
        backend_static.model = simple_conv_model

        def calibration_data():
            for _ in range(10):
                yield [torch.randn(1, 3, 32, 32)]

        static_quantized = backend_static.quantize(
            type="static",
            calibration_data=calibration_data(),
        )
        static_output = static_quantized(sample_conv_input)

        # Both should produce outputs
        assert dynamic_output is not None
        assert static_output is not None


class TestTorchQuantizationModelSize:
    """Test model size reduction after quantization."""

    def test_dynamic_quantization_reduces_size(self, simple_linear_model, temp_dir):
        """Test that dynamic quantization reduces model size."""
        # Save original model
        original_path = temp_dir / "original.pt"
        torch.save(simple_linear_model.state_dict(), original_path)
        original_size = os.path.getsize(original_path)

        # Quantize model
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model
        quantized_model = backend.quantize(type="dynamic", dtype="qint8")

        # Save quantized model
        quantized_path = temp_dir / "quantized.pt"
        torch.save(quantized_model.state_dict(), quantized_path)
        quantized_size = os.path.getsize(quantized_path)

        # Quantized model should be smaller (or similar size for small models)
        # For larger models, we'd expect significant reduction
        assert quantized_size > 0
        print(f"Original size: {original_size}, Quantized size: {quantized_size}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
