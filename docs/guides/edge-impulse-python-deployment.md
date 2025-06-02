# Edge Impulse Python Deployment Guide

## Getting the Right Deployment Package

You received the **C++ SDK** package, but for Python/AI Camera integration, we need the **Python package**. Here's how to get it:

## Step 1: Get Python Deployment Package

**In Edge Impulse Studio:**

1. **Go to "Deployment" tab**
2. **Search for "Python"** (not "Raspberry Pi 4")
3. **Select "Python (Linux/macOS/Windows)"**
4. **Click "Build"**
5. **Download the .zip file**

**Alternative - Direct .tflite Model:**

1. **Go to "Deployment" tab**
2. **Search for "TensorFlow Lite"**
3. **Select "TensorFlow Lite (int8 quantized)"**
4. **Click "Build"**
5. **Download model.tflite**

## Step 2: Extract and Organize

### Option A: Python Package (Recommended)

```bash
# Extract the Python package
unzip python-deployment-package.zip

# Copy to our project
cp -r python-package/* models/edge_impulse/
```

### Option B: Direct .tflite File

```bash
# Copy .tflite file to our models directory
cp model.tflite models/edge_impulse/dice_classifier_v1.tflite
```

## Step 3: Test Python Integration

```bash
# Run our test script
python3 scripts/deployment/test_model_performance.py
```

## What You Have vs What We Need

### ✅ What You Downloaded (C++ SDK):

- **Purpose**: Embedded C++ applications
- **Files**: `tflite_learn_3_compiled.cpp` (compiled model)
- **Use case**: Arduino, embedded systems, custom hardware
- **Integration**: Requires C++ compilation

### 🎯 What We Need (Python Package):

- **Purpose**: Python applications with TensorFlow Lite
- **Files**: `model.tflite` + Python wrapper
- **Use case**: Raspberry Pi with Python, AI Camera integration
- **Integration**: Direct Python import

## Expected Python Package Structure

```
python-package/
├── model.tflite               # TensorFlow Lite model
├── ei_classifier.py           # Edge Impulse Python wrapper
├── requirements.txt           # Python dependencies
├── README.md                  # Python usage instructions
└── examples/
    ├── classify.py            # Classification example
    └── camera_example.py      # Camera integration example
```

## Python Integration Example

Once you have the Python package:

```python
# Import Edge Impulse classifier
import ei_classifier

# Load model
classifier = ei_classifier.EIClassifier("model.tflite")

# Classify image
result = classifier.classify(image_array)
print(f"Prediction: {result}")
```

## AI Camera Integration Strategy

### Option 1: Edge Impulse Python Wrapper (Easiest)

```python
# Use EI's Python SDK with picamera2
import ei_classifier
from picamera2 import Picamera2

classifier = ei_classifier.EIClassifier("model.tflite")
picam2 = Picamera2()

# Capture and classify
image = picam2.capture_array()
result = classifier.classify(image)
```

### Option 2: Direct TensorFlow Lite (More Control)

```python
# Use TFLite directly with picamera2
import tensorflow as tf
from picamera2 import Picamera2

interpreter = tf.lite.Interpreter("model.tflite")
interpreter.allocate_tensors()

# Custom inference pipeline
image = picam2.capture_array()
# Preprocess image, run inference, postprocess results
```

### Option 3: IMX500 Hardware Acceleration (Future)

```python
# Deploy model directly to IMX500 chip
# Requires conversion to .rpk format
# Hardware-accelerated inference with metadata output
```

## Next Steps

1. **Download Python package** from Edge Impulse Studio
2. **Extract to `models/edge_impulse/`**
3. **Test with our performance script**
4. **Integrate with AI camera pipeline**

## Troubleshooting

**Q: Can't find Python deployment option?**

- Try searching for "TensorFlow Lite" instead
- Download .tflite file directly
- We can create Python wrapper ourselves

**Q: Model accuracy still low?**

- Test with Python package first
- Real-world performance often differs from studio metrics
- Consider data augmentation or additional training data

**Q: Want to use C++ SDK anyway?**

- Possible but requires compiling C++ extensions
- Python integration more complex
- Better for final optimization, not initial testing
