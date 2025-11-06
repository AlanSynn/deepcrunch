# DeepCrunch Testing Suite

This directory contains comprehensive end-to-end tests for the DeepCrunch model compression library.

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures and utilities
├── backend/                             # Backend-specific tests
│   ├── types_test.py                   # Backend type enum tests
│   └── test_backend_registry.py        # Backend registration tests
├── core/                                # Core functionality tests
│   └── test_core.py                    # Model/Trainer wrapper tests
├── utils/                               # Utility tests
│   └── test_dot_dict.py                # DotDict utility tests
└── e2e/                                 # End-to-end integration tests
    ├── test_torch_e2e.py               # Torch backend E2E tests
    ├── test_onnx_e2e.py                # ONNX backend E2E tests
    ├── test_accuracy_performance.py     # Accuracy & performance tests
    └── test_error_handling.py          # Error handling & edge cases
```

## Test Coverage

### Torch Backend Tests (`test_torch_e2e.py`)
- **Dynamic Quantization**: Tests for Linear, LSTM models with various dtypes (qint8, float16)
- **Static Quantization**: Tests for Conv models with calibration data
- **QAT (Quantization-Aware Training)**: Basic QAT workflow tests
- **FX Mode Quantization**: Graph-mode quantization tests
- **Model Size**: Tests for model size reduction after quantization
- **Comparison**: Tests comparing different quantization methods

### ONNX Backend Tests (`test_onnx_e2e.py`)
- **Dynamic Quantization**: Basic quantization, inference, and size reduction tests
- **Static Quantization**: Calibration-based quantization tests
- **Float16 Conversion**: FP16 model conversion and inference tests
- **Pre-processing**: Model pre-processing for large models (>2GB)
- **Comparison**: Tests comparing all ONNX quantization methods
- **Error Handling**: Tests for invalid inputs and edge cases

### Accuracy & Performance Tests (`test_accuracy_performance.py`)
- **Accuracy Validation**: MSE, relative error metrics on test datasets
- **Inference Latency**: Latency comparison between original and quantized models
- **Throughput Measurement**: Samples/second throughput tests
- **Model Size Metrics**: Compression ratio calculations
- **Memory Footprint**: Memory usage tests
- **Batch Size Performance**: Performance tests across various batch sizes

### Error Handling Tests (`test_error_handling.py`)
- **Invalid Inputs**: Tests for invalid quantization types, dtypes, paths
- **Edge Cases**: Empty models, single-layer models, extreme inputs
- **Resource Management**: Tests for proper cleanup of temporary files
- **Concurrent Operations**: Tests for simultaneous quantization operations

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Suite
```bash
# End-to-end tests only
pytest tests/e2e/

# Torch backend tests
pytest tests/e2e/test_torch_e2e.py

# ONNX backend tests
pytest tests/e2e/test_onnx_e2e.py

# Accuracy and performance tests
pytest tests/e2e/test_accuracy_performance.py

# Error handling tests
pytest tests/e2e/test_error_handling.py
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=deepcrunch --cov-report=html --cov-report=term
```

### Run Tests in Parallel
```bash
# Run with 4 parallel workers
pytest tests/ -n 4
```

### Run Tests with Verbose Output
```bash
pytest tests/ -v -s
```

### Run Specific Test Class or Method
```bash
# Run specific test class
pytest tests/e2e/test_torch_e2e.py::TestTorchDynamicQuantization

# Run specific test method
pytest tests/e2e/test_torch_e2e.py::TestTorchDynamicQuantization::test_linear_model_dynamic_quantization
```

## Test Fixtures

The `conftest.py` file provides shared fixtures used across tests:

### Model Fixtures
- `simple_linear_model`: Simple linear neural network
- `simple_conv_model`: Simple convolutional neural network
- `simple_lstm_model`: Simple LSTM neural network

### Input Fixtures
- `sample_linear_input`: Sample input for linear models
- `sample_conv_input`: Sample input for conv models (32x32 images)
- `sample_lstm_input`: Sample input for LSTM models

### Utility Fixtures
- `temp_dir`: Temporary directory for test file operations
- `onnx_model_path`: Pre-exported ONNX model path
- `sample_calibration_data`: Calibration data generator

### Helper Functions
- `get_model_size(model_path)`: Get model file size in bytes
- `compare_outputs(output1, output2, rtol, atol)`: Compare model outputs
- `calculate_accuracy_drop(original, quantized)`: Calculate accuracy drop percentage

## Testing Best Practices

### 1. Use Fixtures
Always use provided fixtures instead of creating models inline:
```python
def test_quantization(simple_linear_model, sample_linear_input):
    # Good: Uses fixtures
    backend = BackendRegistry.get_backend("torch")
    backend.model = simple_linear_model
    quantized = backend.quantize(type="dynamic")
```

### 2. Clean Up Resources
Use the `temp_dir` fixture for file operations:
```python
def test_save_model(simple_linear_model, temp_dir):
    save_path = temp_dir / "model.pt"  # Auto-cleaned
    torch.save(simple_linear_model, save_path)
```

### 3. Handle Expected Failures
Use pytest.raises for expected exceptions:
```python
def test_invalid_input():
    with pytest.raises(ValueError):
        backend.quantize(type="invalid")
```

### 4. Skip Unavailable Features
Use pytest.skip for environment-dependent tests:
```python
def test_fx_quantization(simple_linear_model):
    try:
        quantized = backend.quantize(type="fx")
    except Exception as e:
        pytest.skip(f"FX not available: {e}")
```

## Performance Benchmarking

Performance tests measure:
- **Latency**: Average inference time per sample (ms)
- **Throughput**: Samples processed per second
- **Model Size**: File size reduction (compression ratio)
- **Memory Usage**: Runtime memory footprint
- **Accuracy**: MSE and relative error metrics

Results are printed during test execution for manual inspection.

## Continuous Integration

Tests run automatically on:
- Push to main branch
- Pull request creation/updates
- GitHub Actions workflow: `.github/workflows/test.yml`

CI Configuration:
- Python 3.10
- Ubuntu Latest
- Coverage uploaded to Codecov

## Adding New Tests

When adding new tests:

1. **Add fixtures** to `conftest.py` if needed
2. **Follow naming conventions**:
   - Test files: `test_*.py`
   - Test classes: `Test*`
   - Test methods: `test_*`
3. **Document test purpose** in docstrings
4. **Include assertions** for all critical behaviors
5. **Clean up resources** using fixtures
6. **Test both success and failure** cases

Example:
```python
class TestNewFeature:
    """Test new quantization feature."""

    def test_feature_success(self, simple_linear_model):
        """Test successful feature execution."""
        backend = BackendRegistry.get_backend("torch")
        backend.model = simple_linear_model
        result = backend.new_feature()
        assert result is not None

    def test_feature_failure(self):
        """Test feature with invalid input."""
        backend = BackendRegistry.get_backend("torch")
        with pytest.raises(ValueError):
            backend.new_feature(invalid_param=True)
```

## Test Metrics

Target test metrics:
- **Code Coverage**: >80% for all modules
- **Test Success Rate**: 100% on main branch
- **Test Execution Time**: <5 minutes for full suite
- **Documentation**: Every test has a descriptive docstring

## Troubleshooting

### Tests Failing Locally

1. **Check dependencies**:
   ```bash
   pip install -e ".[test]"
   ```

2. **Clear pytest cache**:
   ```bash
   pytest --cache-clear
   ```

3. **Run in isolation**:
   ```bash
   pytest tests/e2e/test_torch_e2e.py -v
   ```

### Slow Tests

1. **Run in parallel**:
   ```bash
   pytest -n auto
   ```

2. **Skip slow tests**:
   ```bash
   pytest -m "not slow"
   ```

### Import Errors

Ensure DeepCrunch is installed in development mode:
```bash
pip install -e .
```

## Contributing

When contributing tests:
1. Ensure all tests pass locally
2. Add docstrings to all test functions
3. Update this README if adding new test categories
4. Run coverage report and aim for >80% coverage
5. Follow existing test structure and naming conventions

## Contact

For questions about testing:
- Open an issue on GitHub
- Check existing test examples in this directory
- Review the main README.rst for project information
