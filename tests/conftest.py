"""Pytest fixtures and utilities for end-to-end testing."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


# Simple test models
class SimpleLinearModel(nn.Module):
    """Simple linear model for testing."""

    def __init__(self, input_size=10, hidden_size=50, num_classes=5):
        super(SimpleLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleConvModel(nn.Module):
    """Simple CNN model for testing."""

    def __init__(self):
        super(SimpleConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleLSTMModel(nn.Module):
    """Simple LSTM model for testing dynamic quantization."""

    def __init__(self, input_size=10, hidden_size=20, num_layers=2, num_classes=5):
        super(SimpleLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


@pytest.fixture
def simple_linear_model():
    """Fixture for simple linear model."""
    model = SimpleLinearModel()
    model.eval()
    return model


@pytest.fixture
def simple_conv_model():
    """Fixture for simple conv model."""
    model = SimpleConvModel()
    model.eval()
    return model


@pytest.fixture
def simple_lstm_model():
    """Fixture for simple LSTM model."""
    model = SimpleLSTMModel()
    model.eval()
    return model


@pytest.fixture
def sample_linear_input():
    """Fixture for sample linear model input."""
    return torch.randn(1, 10)


@pytest.fixture
def sample_conv_input():
    """Fixture for sample conv model input."""
    return torch.randn(1, 3, 32, 32)


@pytest.fixture
def sample_lstm_input():
    """Fixture for sample LSTM model input."""
    return torch.randn(1, 5, 10)  # batch_size, seq_len, input_size


@pytest.fixture
def sample_calibration_data():
    """Fixture for calibration data (iterator of sample inputs)."""
    def calibration_data_reader():
        for _ in range(10):
            yield {"input": torch.randn(1, 10).numpy()}
    return calibration_data_reader


@pytest.fixture
def temp_dir():
    """Fixture for temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def onnx_model_path(simple_linear_model, sample_linear_input, temp_dir):
    """Fixture for ONNX model path."""
    onnx_path = temp_dir / "model.onnx"
    torch.onnx.export(
        simple_linear_model,
        sample_linear_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    return str(onnx_path)


def get_model_size(model_path):
    """Get model size in bytes."""
    return os.path.getsize(model_path)


def compare_outputs(output1, output2, rtol=1e-2, atol=1e-2):
    """Compare two model outputs."""
    if isinstance(output1, torch.Tensor):
        output1 = output1.detach().numpy()
    if isinstance(output2, torch.Tensor):
        output2 = output2.detach().numpy()

    import numpy as np
    return np.allclose(output1, output2, rtol=rtol, atol=atol)


def calculate_accuracy_drop(original_output, quantized_output):
    """Calculate accuracy drop percentage."""
    if isinstance(original_output, torch.Tensor):
        original_output = original_output.detach().numpy()
    if isinstance(quantized_output, torch.Tensor):
        quantized_output = quantized_output.detach().numpy()

    import numpy as np
    diff = np.abs(original_output - quantized_output)
    relative_diff = diff / (np.abs(original_output) + 1e-8)
    return np.mean(relative_diff) * 100


class CalibrationDataReader:
    """Calibration data reader for ONNX quantization."""

    def __init__(self, num_samples=10, input_shape=(1, 10), input_name="input"):
        """Initialize calibration data reader.
        
        Args:
            num_samples: Number of calibration samples to generate.
            input_shape: Shape of input data.
            input_name: Name of the input tensor.
        """
        self.data = [np.random.randn(*input_shape).astype(np.float32) for _ in range(num_samples)]
        self.input_name = input_name
        self.index = 0

    def get_next(self):
        """Get next calibration sample."""
        if self.index < len(self.data):
            input_data = {self.input_name: self.data[self.index]}
            self.index += 1
            return input_data
        return None
