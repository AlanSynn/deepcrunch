import unittest
from deepcrunch.backend.backend_registry import BackendRegistry, NeuralCompressor, Torch, onnx
from deepcrunch.backend.types import BACKEND_TYPES


class TestBackendRegistry(unittest.TestCase):
    def test_register(self):
        # Test registering a new backend
        BackendRegistry.register("test_backend", NeuralCompressor)
        self.assertIn("test_backend", BackendRegistry._backends)

        # Test registering an existing backend
        with self.assertRaises(ValueError):
            BackendRegistry.register(BACKEND_TYPES.NEURAL_COMPRESSOR, NeuralCompressor)

    def test_get_backend(self):
        # Test getting a backend instance
        backend = BackendRegistry.get_backend("neural_compressor")
        self.assertIsInstance(backend, NeuralCompressor)

        # Test getting a backend instance with arguments
        backend = BackendRegistry.get_backend("torch")
        self.assertIsInstance(backend, Torch)

    def test_get_backend_class(self):
        # Test getting a backend class
        backend_class = BackendRegistry.get_backend_class("onnx")
        self.assertEqual(backend_class, onnx)