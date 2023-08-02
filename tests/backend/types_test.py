import unittest
from deepcrunch.backend.types import BACKEND_TYPES


class TestBackendTypes(unittest.TestCase):
    def test_neural_compressor(self):
        self.assertEqual(str(BACKEND_TYPES.NEURAL_COMPRESSOR), "neural_compressor")
        self.assertEqual(BACKEND_TYPES.from_str("neural_compressor"), BACKEND_TYPES.NEURAL_COMPRESSOR)

    def test_torch(self):
        self.assertEqual(str(BACKEND_TYPES.TORCH), "torch")
        self.assertEqual(BACKEND_TYPES.from_str("torch"), BACKEND_TYPES.TORCH)

    def test_onnx(self):
        self.assertEqual(str(BACKEND_TYPES.ONNX), "onnx")
        self.assertEqual(BACKEND_TYPES.from_str("onnx"), BACKEND_TYPES.ONNX)

    def test_disabled(self):
        self.assertEqual(str(BACKEND_TYPES.DISABLED), "disabled")
        self.assertEqual(BACKEND_TYPES.from_str("disabled"), BACKEND_TYPES.DISABLED)

    def test_invalid_string(self):
        with self.assertRaises(KeyError):
            BACKEND_TYPES.from_str("invalid_string")