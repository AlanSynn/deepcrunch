# DeepCrunch Examples 📚

This directory contains comprehensive examples demonstrating all DeepCrunch capabilities.

## Quick Start

Run any example directly:

```bash
python examples/01_dynamic_quantization_simple.py
```

## 📋 Examples Overview

### Basic Examples (Start Here!)

#### 01. Dynamic Quantization - Simple
**File:** `01_dynamic_quantization_simple.py`
- **What:** Basic quantization tutorial
- **Models:** Simple fully connected network
- **Methods:** Dynamic INT8 quantization
- **Learn:** Basic quantization workflow, size/speed comparison

```bash
python examples/01_dynamic_quantization_simple.py
```

**Expected Output:**
```
Original Model: 0.45 MB
Quantized Model: 0.12 MB
Size reduction: 73.33%
Speedup: 2.1x
```

---

#### 02. Static Quantization - CNN
**File:** `02_static_quantization_cnn.py`
- **What:** CNN quantization with calibration
- **Models:** Convolutional neural network for images
- **Methods:** Static INT8 quantization
- **Learn:** Calibration data, performance benchmarking

```bash
python examples/02_static_quantization_cnn.py
```

**Expected Output:**
```
Static Quantization with 20 calibration batches
Speedup: 2.8x (CNN optimized for static quantization)
Same prediction: True
```

---

#### 03. LSTM Quantization
**File:** `03_lstm_quantization.py`
- **What:** Sequence model quantization
- **Models:** LSTM for sequence classification
- **Methods:** Dynamic INT8, FP16
- **Learn:** RNN quantization, batch size effects

```bash
python examples/03_lstm_quantization.py
```

**Expected Output:**
```
qint8:   Size reduction: 68%, Speedup: 2.3x
float16: Size reduction: 50%, Speedup: 1.4x
Better speedup with larger batch sizes
```

---

#### 04. ONNX Quantization
**File:** `04_onnx_quantization.py`
- **What:** ONNX model compression
- **Models:** PyTorch → ONNX conversion
- **Methods:** Dynamic INT8, Static INT8, FP16
- **Learn:** Cross-platform deployment, ONNX workflow

```bash
python examples/04_onnx_quantization.py
```

**Expected Output:**
```
Method          Size (MB)  Reduction  Latency (ms)
FP32 (Original) 0.0234     0.00%      0.0156
INT8 (Dynamic)  0.0062     73.50%     0.0089
INT8 (Static)   0.0058     75.21%     0.0082
FP16            0.0117     50.00%     0.0124
```

---

### Advanced Examples

#### 05. BERT Quantization (LLM)
**File:** `05_bert_quantization.py`
- **What:** Transformer/LLM quantization
- **Models:** BERT-like model (768M params)
- **Methods:** Dynamic INT8 for transformers
- **Learn:** Large model compression, attention mechanism quantization

```bash
python examples/05_bert_quantization.py
```

**Key Features:**
- Simulates BERT-base architecture
- 12 transformer layers, 12 attention heads
- Demonstrates typical LLM compression (4x smaller)
- Shows batch size effects on speedup

**Expected Output:**
```
Total parameters: 119,547,909
Original model: 456.23 MB
Quantized model: 114.89 MB (74.8% reduction)
Speedup: 2.4x on CPU
Prediction accuracy: 100%
```

---

#### 06. ResNet Quantization
**File:** `06_resnet_quantization.py`
- **What:** Real-world vision model quantization
- **Models:** ResNet-18 (can use pretrained)
- **Methods:** Dynamic, Static, QAT
- **Learn:** Production deployment, real model compression

```bash
python examples/06_resnet_quantization.py
```

**Key Features:**
- Loads pretrained ResNet-18 or creates random init
- Compares all quantization methods
- Production deployment recommendations
- Comprehensive benchmarking

**Expected Output:**
```
Model: ResNet-18
Parameters: 11,689,512
Original: 46 MB, 20 ms/image
Static INT8: ~11 MB, 8 ms/image
Result: 4x smaller, 2.5x faster
Top-1 prediction match: True
```

---

#### 07. GPT-2 Quantization
**File:** `07_gpt2_quantization.py`
- **What:** Large language model compression
- **Models:** GPT-2 (124M params) or GPT-2-like
- **Methods:** Dynamic INT8, FP16
- **Learn:** LLM deployment, text generation optimization

```bash
python examples/07_gpt2_quantization.py
```

**Key Features:**
- Tries to load real GPT-2 from HuggingFace
- Falls back to GPT-2-like architecture
- Shows sequence length effects
- Compares INT8 vs FP16

**Expected Output:**
```
Model: GPT-2 (124M parameters)
Original: 500 MB, 200 ms/sequence
Quantized INT8: 125 MB, 85 ms/sequence
FP16: 250 MB, 160 ms/sequence
Memory saved: 375 MB
```

---

#### 08. Comprehensive Comparison
**File:** `08_comprehensive_comparison.py`
- **What:** ALL methods × ALL model types
- **Models:** FC, CNN, LSTM
- **Methods:** All quantization methods
- **Learn:** Complete overview, best practices guide

```bash
python examples/08_comprehensive_comparison.py
```

**Key Features:**
- Tests FC, CNN, LSTM models
- Compares all quantization methods
- Beautiful formatted tables
- Comprehensive recommendations guide

**Expected Output:**
```
╔════════════════════════════════════════════════════════════╗
║              QUANTIZATION METHOD GUIDE                      ║
╚════════════════════════════════════════════════════════════╝

TEST 1: FULLY CONNECTED MODEL
┌───────────────┬───────────┬───────────┬──────────┬─────────┐
│ Method        │ Size (MB) │ Reduction │ Latency  │ Speedup │
├───────────────┼───────────┼───────────┼──────────┼─────────┤
│ Dynamic INT8  │ 0.12      │ 73.3%     │ 0.15 ms  │ 2.1x    │
│ Dynamic FP16  │ 0.23      │ 48.9%     │ 0.18 ms  │ 1.8x    │
└───────────────┴───────────┴───────────┴──────────┴─────────┘

[Full comparison tables for CNN and LSTM models]

RECOMMENDATIONS BY MODEL TYPE:
• CNNs: Static Quantization
• Transformers/LLMs: Dynamic INT8
• LSTMs: Dynamic INT8 or FP16
```

---

## 🎯 Use Case Selection Guide

**I want to...**

### Compress a Vision Model (ResNet, MobileNet)
→ Start with: `06_resnet_quantization.py`
→ Method: Static Quantization
→ Expected: 4x smaller, 2-4x faster

### Compress a Language Model (BERT, GPT)
→ Start with: `05_bert_quantization.py` or `07_gpt2_quantization.py`
→ Method: Dynamic INT8
→ Expected: 4x smaller, 2-3x faster

### Deploy to Mobile/Edge
→ Start with: `04_onnx_quantization.py`
→ Method: ONNX INT8
→ Expected: Cross-platform, optimized inference

### Learn the Basics
→ Start with: `01_dynamic_quantization_simple.py`
→ Then: `02_static_quantization_cnn.py`
→ Finally: `08_comprehensive_comparison.py`

### See All Methods
→ Run: `08_comprehensive_comparison.py`
→ Complete comparison with recommendations

---

## 📊 Performance Expectations

| Model Type | Best Method | Size Reduction | Speedup | Accuracy |
|------------|-------------|----------------|---------|----------|
| **FC Network** | Dynamic INT8 | 70-75% | 2-3x | >99% |
| **CNN** | Static INT8 | 75% | 2-4x | 98-99% |
| **LSTM/RNN** | Dynamic INT8 | 65-70% | 2-3x | >99% |
| **Transformer** | Dynamic INT8 | 70-75% | 2-3x | >99% |
| **Vision (ResNet)** | Static INT8 | 75% | 2-4x | 99% |
| **LLM (GPT)** | Dynamic INT8 | 75% | 2-3x | 99% |

---

## 🚀 Running Examples

### Run Single Example
```bash
python examples/01_dynamic_quantization_simple.py
```

### Run All Examples
```bash
for example in examples/*.py; do
    echo "Running $example..."
    python "$example"
    echo "---"
done
```

### Run Specific Examples
```bash
# LLM examples
python examples/05_bert_quantization.py
python examples/07_gpt2_quantization.py

# Vision examples
python examples/02_static_quantization_cnn.py
python examples/06_resnet_quantization.py

# Complete overview
python examples/08_comprehensive_comparison.py
```

---

## 💡 Tips & Best Practices

### 1. Choosing Quantization Method

**Dynamic Quantization** when:
- ✅ Model has many Linear/LSTM layers
- ✅ No calibration data available
- ✅ Quick deployment needed
- ✅ Working with transformers/LLMs

**Static Quantization** when:
- ✅ Model is CNN-based
- ✅ Have representative calibration data
- ✅ Need best performance
- ✅ Production deployment

**QAT** when:
- ✅ Accuracy is critical
- ✅ Can afford fine-tuning time
- ✅ Have training infrastructure
- ✅ Need <0.5% accuracy drop

### 2. Calibration Data Tips

```python
def good_calibration_data():
    """Use representative samples from your dataset"""
    for batch in dataloader:
        yield [batch]  # Use real data

def bad_calibration_data():
    """Don't use random data"""
    for _ in range(10):
        yield [torch.randn(...)]  # Not representative!
```

### 3. Accuracy Validation

Always validate on your test set:

```python
# Quantize
quantized_model = backend.quantize(...)

# Test on validation set
original_acc = validate(model, val_loader)
quantized_acc = validate(quantized_model, val_loader)

print(f"Accuracy drop: {original_acc - quantized_acc:.2f}%")
```

### 4. Performance Testing

Test on target hardware:

```python
# CPU test
device = torch.device("cpu")
model = model.to(device)
# Benchmark...

# GPU test (for FP16)
device = torch.device("cuda")
model = model.to(device).half()
# Benchmark...
```

---

## 🔧 Troubleshooting

### Issue: "Module not found"
```bash
# Install DeepCrunch
pip install -e .

# Or install dependencies
pip install torch torchvision
```

### Issue: "Static quantization failed"
- Ensure model is in eval mode: `model.eval()`
- Provide proper calibration data
- Check model architecture compatibility

### Issue: "No speedup observed"
- Benchmarks are hardware-dependent
- Best speedup on CPU, not GPU
- Larger models show better speedup
- Try larger batch sizes

### Issue: "Accuracy dropped significantly"
- Use static quantization instead of dynamic
- Try QAT (Quantization-Aware Training)
- Increase calibration data samples
- Check if model architecture is suitable

---

## 📚 Additional Resources

- **Main Documentation:** [../README.md](../README.md)
- **API Reference:** [../docs/](../docs/)
- **Testing Guide:** [../tests/README.md](../tests/README.md)
- **Jupyter Notebooks:** [../notebooks/](../notebooks/)

---

## 🤝 Contributing

Found a bug or have an improvement?

1. Check existing examples
2. Create a new example following the naming convention
3. Add documentation to this README
4. Submit a pull request

---

## 📞 Support

- **GitHub Issues:** Report bugs or request features
- **Discussions:** Ask questions, share tips
- **Email:** alan@alansynn.com

---

<div align="center">

**Happy Compressing! 🚀**

[⬆ Back to Top](#deepcrunch-examples-)

</div>
