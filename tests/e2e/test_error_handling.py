"""End-to-end tests for error handling and edge cases."""

import os

import pytest
import torch
import torch.nn as nn

from deepcrunch.backend.backend_registry import BackendRegistry


class TestTorchErrorHandling:
    """Test error handling for Torch backend."""

    def test_invalid_quantization_type(self, simple_linear_model):
        """Test handling of invalid quantization type."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        with pytest.raises((ValueError, KeyError, AttributeError)):
            backend.quantize(type="invalid_type")

    def test_invalid_dtype(self, simple_linear_model):
        """Test handling of invalid dtype."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        with pytest.raises((ValueError, KeyError, AttributeError)):
            backend.quantize(type="dynamic", dtype="invalid_dtype")

    def test_none_model(self):
        """Test handling of None model."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = None

        with pytest.raises((AttributeError, ValueError, TypeError)):
            backend.quantize(type="dynamic")

    def test_static_without_calibration_data(self, simple_conv_model):
        """Test static quantization without calibration data."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_conv_model

        # This should either fail or handle gracefully
        try:
            backend.quantize(type="static")
        except (ValueError, TypeError, AttributeError, RuntimeError):
            # Expected to fail without calibration data
            assert True
        else:
            # If it doesn't fail, it should have used default calibration
            pass

    def test_unsupported_model_architecture(self):
        """Test quantization on unsupported model architecture."""

        class UnsupportedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.custom_op = lambda x: x  # Not a standard operation

            def forward(self, x):
                return self.custom_op(x)

        model = UnsupportedModel()
        backend = BackendRegistry.get_backend("torch")
        backend.model = model

        # Some architectures might not be quantizable
        try:
            backend.quantize(type="dynamic")
            # If it succeeds, that's fine too
        except (RuntimeError, TypeError, AttributeError):
            # Expected for some unsupported architectures
            pass


class TestONNXErrorHandling:
    """Test error handling for ONNX backend."""

    def test_nonexistent_model_file(self, temp_dir):
        """Test loading non-existent ONNX model."""
        backend = BackendRegistry.get_backend("onnx")
        backend.model = "nonexistent_model.onnx"

        output_path = str(temp_dir / "output.onnx")

        with pytest.raises((FileNotFoundError, RuntimeError)):
            backend.quantize(type="dynamic", output_path=output_path)

    def test_invalid_output_path(self, onnx_model_path):
        """Test with invalid output path."""
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        # Try to write to a directory that doesn't exist
        invalid_path = "/nonexistent_directory/model.onnx"

        with pytest.raises((FileNotFoundError, OSError, PermissionError)):
            backend.quantize(type="dynamic", output_path=invalid_path)

    def test_corrupted_onnx_model(self, temp_dir):
        """Test loading corrupted ONNX model."""
        # Create a fake ONNX file with invalid content
        corrupted_path = temp_dir / "corrupted.onnx"
        with open(corrupted_path, "w") as f:
            f.write("This is not a valid ONNX model")

        backend = BackendRegistry.get_backend("onnx")
        backend.model = str(corrupted_path)

        output_path = str(temp_dir / "output.onnx")

        with pytest.raises((RuntimeError, ValueError, OSError)):
            backend.quantize(type="dynamic", output_path=output_path)

    def test_static_without_calibration_reader(self, onnx_model_path, temp_dir):
        """Test static quantization without calibration data reader."""
        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        output_path = str(temp_dir / "output.onnx")

        # Static quantization requires calibration data
        with pytest.raises((TypeError, ValueError, AttributeError)):
            backend.quantize(type="static", output_path=output_path)


class TestBackendRegistryErrorHandling:
    """Test error handling for backend registry."""

    def test_invalid_backend_name(self):
        """Test getting backend with invalid name."""
        with pytest.raises((ValueError, KeyError)):
            BackendRegistry.get_backend("invalid_backend")

    def test_register_duplicate_backend(self):
        """Test registering duplicate backend."""
        from deepcrunch.backend.engines.torch_ao import TorchPTQ

        # Try to register an existing backend
        with pytest.raises(ValueError):
            BackendRegistry.register("torch", TorchPTQ)

    def test_get_backend_class_invalid(self):
        """Test getting invalid backend class."""
        with pytest.raises((ValueError, KeyError)):
            BackendRegistry.get_backend_class("nonexistent_backend")


class TestEdgeCases:
    """Test edge cases in quantization."""

    def test_empty_model(self):
        """Test quantization of empty model."""

        class EmptyModel(nn.Module):
            def forward(self, x):
                return x

        model = EmptyModel()
        backend = BackendRegistry.get_backend("torch")
        backend.model = model

        # Empty model might not be quantizable
        try:
            result = backend.quantize(type="dynamic")
            # If it succeeds, verify it returns something
            assert result is not None
        except (RuntimeError, TypeError):
            # Expected for empty models
            pass

    def test_single_layer_model(self):
        """Test quantization of single-layer model."""

        class SingleLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        model = SingleLayerModel()
        model.eval()

        backend = BackendRegistry.get_backend("torch")
        backend.model = model

        # Single layer should be quantizable
        quantized = backend.quantize(type="dynamic")
        assert quantized is not None

        # Test forward pass
        test_input = torch.randn(1, 10)
        output = quantized(test_input)
        assert output.shape == (1, 5)

    def test_very_small_model(self):
        """Test quantization of very small model."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, x):
                return self.fc(x)

        model = TinyModel()
        model.eval()

        backend = BackendRegistry.get_backend("torch")
        backend.model = model

        quantized = backend.quantize(type="dynamic")
        assert quantized is not None

        test_input = torch.randn(1, 2)
        output = quantized(test_input)
        assert output.shape == (1, 2)

    def test_large_batch_size(self, simple_linear_model):
        """Test quantized model with large batch size."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        quantized = backend.quantize(type="dynamic")

        # Test with large batch
        large_batch = torch.randn(1000, 10)
        output = quantized(large_batch)
        assert output.shape == (1000, 5)

    def test_extreme_input_values(self, simple_linear_model):
        """Test quantized model with extreme input values."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        quantized = backend.quantize(type="dynamic")

        # Test with very large values
        large_input = torch.randn(1, 10) * 1e6
        output = quantized(large_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Test with very small values
        small_input = torch.randn(1, 10) * 1e-6
        output = quantized(small_input)
        assert not torch.isnan(output).any()

    def test_zero_input(self, simple_linear_model):
        """Test quantized model with zero input."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        quantized = backend.quantize(type="dynamic")

        # Test with all zeros
        zero_input = torch.zeros(1, 10)
        output = quantized(zero_input)
        assert output is not None
        assert not torch.isnan(output).any()

    def test_model_in_training_mode(self):
        """Test quantization of model in training mode."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
                self.bn = nn.BatchNorm1d(5)

            def forward(self, x):
                return self.bn(self.fc(x))

        model = SimpleModel()
        model.train()  # Explicitly set to training mode

        backend = BackendRegistry.get_backend("torch")
        backend.model = model

        # Quantization typically requires eval mode
        # Backend should handle this or raise appropriate error
        try:
            quantized = backend.quantize(type="dynamic")
            # If successful, test it
            test_input = torch.randn(2, 10)
            output = quantized(test_input)
            assert output is not None
        except (RuntimeError, ValueError):
            # Expected if backend requires eval mode
            pass


class TestConcurrentQuantization:
    """Test concurrent quantization operations."""

    def test_multiple_backends_simultaneously(self, simple_linear_model, onnx_model_path, temp_dir):
        """Test using multiple backends at the same time."""
        # Torch backend
        torch_backend = BackendRegistry.get_backend("torch")
        torch_backend.model = simple_linear_model
        torch_quantized = torch_backend.quantize(type="dynamic")

        # ONNX backend
        onnx_backend = BackendRegistry.get_backend("onnx")
        onnx_backend.model = onnx_model_path
        onnx_output = str(temp_dir / "concurrent_test.onnx")
        onnx_backend.quantize(type="dynamic", output_path=onnx_output)

        # Both should succeed
        assert torch_quantized is not None
        assert os.path.exists(onnx_output)

    def test_multiple_quantizations_same_model(self, simple_linear_model):
        """Test quantizing the same model multiple times."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model

        # Quantize multiple times
        quantized1 = backend.quantize(type="dynamic", dtype="qint8")
        quantized2 = backend.quantize(type="dynamic", dtype="qint8")

        # Both should work
        assert quantized1 is not None
        assert quantized2 is not None

        # Test both models
        test_input = torch.randn(1, 10)
        output1 = quantized1(test_input)
        output2 = quantized2(test_input)

        assert output1.shape == output2.shape


class TestResourceCleanup:
    """Test resource cleanup after quantization."""

    def test_temporary_files_cleaned(self, onnx_model_path, temp_dir):
        """Test that temporary files are cleaned up."""
        import gc

        backend = BackendRegistry.get_backend("onnx")
        backend.model = onnx_model_path

        output_path = str(temp_dir / "cleanup_test.onnx")

        # Perform quantization
        backend.quantize(type="dynamic", output_path=output_path)

        # Force garbage collection
        gc.collect()

        # Output file should exist
        assert os.path.exists(output_path)

        # Check for leaked temporary files (basic check)
        temp_files_before = len(os.listdir(temp_dir))

        # Perform another quantization
        output_path2 = str(temp_dir / "cleanup_test2.onnx")
        backend.quantize(type="dynamic", output_path=output_path2)

        gc.collect()

        temp_files_after = len(os.listdir(temp_dir))

        # Should only have the two output files plus any original files
        assert temp_files_after <= temp_files_before + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
