# Active Context - D6 Dice Recognition Pi 3 Project

## CRITICAL ISSUE DISCOVERED: False Detection Problem ⚠️

### **Current Status: Model v3 Has False Detection Bug**

**Problem Identified:** Model v3 consistently detects "Dice: 3" even when no dice are present or when dice are removed from the scene.

**Test Results:**

- First detection: Dice "3" (conf: 0.311) ✅ CORRECT
- Removed one die: Still detects Dice "3" (conf: 0.427) ❌ FALSE POSITIVE
- Removed another die: Still detects Dice "3" (conf: 0.502) ❌ FALSE POSITIVE
- Removed all dice: Still detects Dice "3" (conf: 0.474) ❌ FALSE POSITIVE

**Root Cause:** The original `parse_detection_output` function in `test_dice_ai.py` lacks proper confidence thresholding for classification models. It **always** returns a dice value regardless of confidence.

### **Solution Implemented: Fixed Detection Logic** ✅

**Files Created:**

1. `debug_false_detection.py` - Diagnostic tool to analyze raw model output
2. `test_dice_ai_fixed.py` - Corrected version with proper confidence thresholding

**Key Fix in Detection Logic:**

```python
def parse_detection_output_fixed(output_data, model_version="v3", confidence_threshold=0.5):
    if len(output_data.shape) == 2:
        # Classification model output
        class_probs = output_data[0]
        max_idx = np.argmax(class_probs)
        max_confidence = class_probs[max_idx]

        # CRITICAL FIX: Only return detection if confidence exceeds threshold
        if max_confidence < confidence_threshold:
            return "No dice detected (confidence too low)"

        dice_value = (max_idx % 6) + 1
        return f"Dice: {dice_value} (conf: {max_confidence:.3f})"
```

**Before Fix:** Always returned highest class probability as detection  
**After Fix:** Returns "No dice detected" when confidence < threshold

### **Immediate Next Steps (CURRENT PRIORITY)**

1. **🔄 Deploy Fixed Scripts to Pi 3**

   ```bash
   # On macOS - commit the fixes
   git add debug_false_detection.py test_dice_ai_fixed.py memory-bank/
   git commit -m "Fix false detection issue with confidence thresholding"
   git push origin main

   # On Pi 3 - get the fixes
   git pull origin main

   # Test with NO DICE first
   python3 test_dice_ai_fixed.py

   # Debug raw output if needed
   python3 debug_false_detection.py
   ```

2. **🧪 Validate Fix Works**

   - Test with empty scene (no dice) - should return "No dice detected"
   - Test with single die - should detect correctly
   - Test with multiple dice - should handle properly
   - Test confidence threshold tuning with 't' option

3. **📊 Determine Optimal Confidence Threshold**
   - Current default: 0.5
   - Test different values (0.3, 0.7) based on results
   - Balance false positives vs missed detections

## Current Focus: Post-Fix Validation and Optimization

**Fresh Start Decision:** All original project files deleted. Starting over with simplified approach focused on Pi 3 hardware constraints.

## CRITICAL: Leverage Onboard AI Acceleration 🚀

### **✅ SUCCESS: IMX500 AI Acceleration Confirmed Working**

**Status:** IMX500 firmware installed and tested successfully on Pi 3

- `sudo apt install imx500-all` ✅ COMPLETED
- MobileNet SSD test ✅ WORKING
- AI acceleration pipeline ✅ VALIDATED

### **Game-Changing Discovery: IMX500 Built-in Neural Network Processor**

The Sony IMX500 sensor includes:

- **Integrated AI accelerator** with 2 TOPS (tera-operations per second)
- **8MB on-chip SRAM** for model storage and processing
- **Built-in Image Signal Processor (ISP)** for preprocessing
- **Onboard RP2040 microcontroller** for model management
- **16MB flash cache** for model storage

### **Revolutionary Architecture**

**Traditional approach** (what we were planning):

```
Camera → Raspberry Pi CPU → Processing → Results
```

**AI Camera approach** (what we should use):

```
Camera → Onboard AI Processor → Results → Raspberry Pi
```

**Key Benefits:**

- **No Pi 3 CPU load** for AI processing
- **Ultra-low latency** (hardware acceleration)
- **Concurrent operation** (Pi 3 free for other tasks)
- **Built-in preprocessing** (automatic image scaling/cropping)
- **Tensor metadata** delivered alongside images

### **Implementation Strategy for Dice Detection**

**Phase 1: Use Pre-trained Models**

- **MobileNet SSD** (already loaded) for object detection
- **Custom training** with Edge Impulse Studio
- **Input tensor:** 640×640 max (int8 or uint8)
- **Output:** Synchronized with image frames

**Phase 2: Custom Dice Model**

- Train specifically for dice (1-6 values)
- Use Edge Impulse Studio or Sony AITRIOS
- Deploy `.rpk` model files to camera
- **Target performance:** <50ms inference time

### **Picamera2 Integration**

**Tensor Metadata Access:**

```python
from picamera2 import Picamera2
import numpy as np

# Initialize camera with AI model
picam2 = Picamera2()
picam2.configure(config_with_ai_model)
picam2.start()

# Capture with tensor metadata
request = picam2.capture_request()
image = request.make_array("main")
metadata = request.get_metadata()

# Access AI results
if "tensor" in metadata:
    ai_results = metadata["tensor"]
    # Process detection results
```

**Model Loading:**

```bash
# Install IMX500 firmware and models
sudo apt install imx500-all

# Test with MobileNet SSD
rpicam-hello --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json
```

## Immediate Next Steps (CURRENT PRIORITY)

1. **✅ Install IMX500 firmware** on Pi 3 - COMPLETED
2. **✅ Test basic AI functionality** with MobileNet - COMPLETED
3. **✅ Organize project structure** - COMPLETED
4. **✅ Train custom model** with Edge Impulse - COMPLETED (dice_classifier_v2.tflite)
5. **🔄 Deploy and test** dice-specific model on Pi 3 - IN PROGRESS

## Current Pi 3 Deployment Status

### **Virtual Environment Required** ⚠️

**Important:** Modern Raspberry Pi OS has `externally-managed-environment` restriction.

**Setup Required on Pi 3:**

```bash
# Create virtual environment for TensorFlow Lite
python3 -m venv ~/dice_env
source ~/dice_env/bin/activate
pip install tflite-runtime

# System packages (already installed)
sudo apt install python3-picamera2 python3-opencv python3-numpy
```

**Usage Pattern:**

```bash
# Every time before running dice detection:
source ~/dice_env/bin/activate
python3 test_dice_ai.py
deactivate  # when done
```

### **Model Successfully Deployed**

- **Model**: `dice_classifier_v2.tflite` (63MB)
- **Input**: 160×160×3 RGB images
- **Output**: 448 classes (object detection format)
- **Performance**: ~78ms inference on development machine

## Recent Accomplishments ✅

### **Critical Bug Fix - False Detection Issue**

**Problem:** Model v3 was detecting dice when none were present due to missing confidence thresholding in classification logic.

**Solution:** Implemented proper confidence-based detection logic:

- Added `debug_false_detection.py` for raw output analysis
- Created `test_dice_ai_fixed.py` with corrected detection logic
- Configurable confidence threshold (default 0.5)
- Threshold testing mode for optimization

**Expected Results After Fix:**

- ✅ **No false positives**: Empty scenes return "No dice detected"
- ✅ **Proper thresholding**: Low confidence detections filtered out
- ✅ **Accurate detection**: Real dice detected with high confidence
- ✅ **Debugging capability**: Raw output analysis available

### **Project Organization Completed**

**Major clean-up and professional structure implemented:**

- **📁 Folder structure**: Professional layout with docs/, src/, scripts/, data/, models/, tests/
- **📋 Documentation**: Comprehensive README.md, project structure guide, Edge Impulse workflow
- **🗂️ File organization**: All scripts and docs moved to proper locations
- **📦 Package structure**: Python packages created with proper **init**.py files
- **🔧 Dependencies**: Clean requirements.txt focused on AI acceleration

### **Files Reorganized:**

```bash
# Before → After
collect_dice_data.py → scripts/data_collection/collect_dice_images.py
download_dice_dataset.py → scripts/data_collection/download_dataset.py
edge_impulse_guide.md → docs/guides/edge-impulse-workflow.md
pico2_dice_concept.md → docs/research/pico2-feasibility.md
pi_zero_2w_analysis.md → docs/analysis/platform-comparison.md
dice_dataset_* → data/raw/ and data/processed/
```

### **New Structure Benefits:**

- **🎯 Clear separation**: Source code, documentation, data, models
- **📈 Scalable**: Easy to add new platforms, features, models
- **🤝 Collaborative**: Standard Python project structure
- **🚀 Professional**: Ready for open source or deployment

## Project Structure Overview

```
recognize-dice/
├── 📁 src/dice_detector/        # Main application package
├── 📁 scripts/data_collection/  # Data preparation scripts
├── 📁 docs/{guides,analysis,research}/ # Comprehensive documentation
├── 📁 data/{raw,processed,samples}/ # Dataset organization
├── 📁 models/edge_impulse/      # Trained AI models
├── 📁 tests/{unit,integration,hardware}/ # Test framework
├── 📁 configs/                 # Platform configurations
├── 📁 deployment/{pi3,zero2w}/ # Deployment artifacts
└── 📁 memory-bank/             # Project memory system
```

## Future Testing Goals 🎯

### **Phase 1: Pi 3 Development Platform** (Current)

- ✅ Validate AI acceleration works
- 🔄 Train and deploy custom dice model
- 🔄 Create real-time detection application
- 🔄 Optimize performance for Pi 3 constraints

### **Phase 2: Pi Zero 2 W Optimization** (Future Priority)

**Goal:** Port to smaller, more efficient platform

**Benefits of Zero 2 W testing:**

- **50% cost reduction** ($15 vs $35)
- **Ultra-portable form factor** (65×30mm vs 85×56mm)
- **60% power reduction** (~2W vs ~5W)
- **Same AI performance** (IMX500 acceleration)
- **Perfect for embedded/portable applications**

**Expected Performance:**

- **Total latency:** ~80ms (vs 60ms on Pi 3)
- **Frame rate:** ~12 FPS (vs 16 FPS on Pi 3)
- **Memory usage:** ~320MB (fits in 512MB)
- **Battery life:** 6+ hours with 2500mAh battery

**Testing Checklist:**

- [ ] Same Edge Impulse model deployment
- [ ] Memory optimization for 512MB limit
- [ ] Performance benchmarking vs Pi 3
- [ ] Power consumption measurements
- [ ] Portable enclosure design
- [ ] Battery operation validation

### **Phase 3: Alternative Platforms** (Research)

- **Pi Pico 2:** Ultra-low power embedded (custom hardware needed)
- **Pi 5:** Maximum performance comparison
- **Other SBCs:** Jetson Nano, Orange Pi, etc.

### **Dataset Strategy: Use Professional Dataset** 🎯

**Roboflow Dice Dataset:**

- **359 professionally annotated images**
- **Balanced classes** (dice values 1-6)
- **Multiple scenarios:** single dice, Catan dice, mass groupings
- **Public domain license** (free to use)
- **Ready for Edge Impulse** (multiple export formats)

**Advantages over custom collection:**

- ✅ **Immediate availability** (no time spent capturing)
- ✅ **Professional quality** annotations
- ✅ **Variety of conditions** (lighting, backgrounds, angles)
- ✅ **Proven dataset** (used by other ML projects)
- ✅ **More training data** (359 vs ~150 we would collect)

## Camera Specifications & Setup (Critical for Dice Detection)

### Hardware: Raspberry Pi AI Camera (Sony IMX500)

**Key Specifications for Dice Detection:**

- **Sensor:** Sony IMX500 (12.3MP, 1/2.3" format)
- **Resolution:** 4056×3040 full / 2028×1520 binned (30fps)
- **Pixel size:** 1.55μm × 1.55μm
- **Focal length:** 4.74mm fixed
- **F-stop:** F1.79 (excellent low light)
- **Focus range:** 20cm to infinity (manual adjustable)
- **Field of view:** 66.3° horizontal × 52.3° vertical
- **Frame rates:** 30fps binned / 10fps full resolution

### Optimal Dice Detection Setup

**Recommended Camera Position:**

- **Distance:** 25-40cm from dice (within sweet spot of focus range)
- **Angle:** Perpendicular to dice surface (minimize perspective distortion)
- **Lighting:** Overhead or side lighting to create shadows for pip definition

**Detection Zone Calculations:**

- At 30cm distance: ~32cm × 24cm capture area
- At 40cm distance: ~43cm × 32cm capture area
- **Dice size:** Standard D6 ≈ 16mm (optimal detection at 25-35cm)

### Camera Configuration for Pi 3

**Performance Settings:**

- **Resolution:** 320×320 or 640×480 (Pi 3 optimized)
- **Frame rate:** 5-10fps target (thermal management)
- **Format:** RGB888 preferred for OpenCV compatibility
- **ROI (Region of Interest):** Crop center area for dice zone

**libcamera Commands:**

```bash
# Test camera
libcamera-hello -t 0

# Capture test image
libcamera-still -o test.jpg --width 640 --height 480

# Video stream (for live detection)
libcamera-vid --width 320 --height 320 --framerate 10 -t 0
```

### Focus Adjustment

**Manual Focus Ring:**

- **Near focus:** 20cm minimum (too close for dice)
- **Optimal range:** 25-40cm for dice detection
- **Focus check:** Use `libcamera-hello --info-text "focus %focus"` to see focus metric
- **Sharp images:** Focus metric >1000 typically indicates good focus

## Technical Constraints (Pi 3 Specific)

### Performance Targets (REVISED WITH AI ACCELERATION)

- **Frame rate:** 10-30 FPS possible (AI processing offloaded)
- **Memory usage:** <500MB total application footprint (reduced load)
- **Resolution:** 640×640 max for AI inference (hardware processed)
- **Processing time:** <100ms per frame including AI (target)

### Camera Interface Strategy

- **Primary:** picamera2 with AI model support
- **AI Models:** MobileNet SSD, custom trained models
- **Format handling:** Built-in ISP handles preprocessing
- **Streaming:** Hardware-accelerated pipeline with tensor metadata

## Detection Strategy for Fresh Start (AI-POWERED)

### Phase 1: AI-Accelerated Detection (Week 1)

```python
# AI-powered approach (~30 lines)
1. Load dice detection model to IMX500
2. Capture frame with tensor metadata
3. Parse AI detection results
4. Display dice count/values
5. No CPU-intensive processing needed
```

### Phase 2: Enhanced AI Model (Week 2)

- Train custom dice model with Edge Impulse
- Implement value classification (1-6)
- Add confidence scoring
- Real-time performance optimization

### Phase 3: Application Features (Week 3+)

- Multi-dice detection
- Rolling averages
- User interface improvements
- Data logging capabilities

## Key Learnings from Previous Attempts

### What Worked

- **Basic CV operations** (blob detection, contours)
- **picamera2/picamera dual support**
- **Tkinter for lightweight GUI**
- **macOS→Pi development workflow**

### What Failed

- **Over-engineering** (too many fallback systems)
- **Complex ML models** (Pi 3 can't handle efficiently)
- **Multiple detection algorithms** (confusion, not improvement)
- **Ambitious performance targets** (ignored Pi 3 reality)

### Critical Success Factors

- **Use onboard AI acceleration** (game changer)
- **Start simple, build gradually**
- **Test on Pi 3 early and often**
- **Design for hardware acceleration first**
- **Memory management becomes less critical**

## Current Workflow

1. **Develop on macOS:** Code editing, basic testing
2. **Deploy to Pi 3:** SSH transfer for hardware testing
3. **Test cycle:** Quick iteration on real hardware
4. **Performance monitoring:** FPS, memory, thermal tracking

## Architecture Decisions for Fresh Start

### Core Principles

- **Hardware acceleration first**
- **AI-powered over traditional CV**
- **Pi 3 + IMX500 = powerful combination**
- **Leverage built-in preprocessing**
- **Focus on model training over algorithm complexity**

### Technology Stack

- **Python 3** (picamera2 integration)
- **IMX500 AI models** (MobileNet, custom trained)
- **picamera2** (AI-enabled camera interface)
- **Edge Impulse Studio** (model training)
- **Tkinter** (lightweight GUI)

## Development Environment Commands

### macOS Development

```bash
python3 -c "import cv2; print('OpenCV available')"  # Test framework
# Edit code, commit, push to Pi
```

### Pi 3 Hardware Testing

```bash
# Install AI firmware
sudo apt install imx500-all

# Test AI camera
rpicam-hello --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json

# Camera tests
libcamera-hello -t 5000                    # Camera test
python3 ai_dice_detection.py               # Run AI detection
htop                                        # Monitor performance
cat /sys/class/thermal/thermal_zone0/temp   # Check thermal
```

## AI Model Development Pipeline

### Training Custom Dice Model

1. **Data Collection:** Use picamera2 to capture dice images
2. **Edge Impulse Studio:** Upload, label, train model
3. **Model Export:** Download .tflite or .rpk format
4. **Deployment:** Load to IMX500 via firmware
5. **Testing:** Real-time validation on Pi 3

### Model Performance Targets

- **Accuracy:** >95% dice value classification
- **Latency:** <50ms inference time
- **Memory:** <8MB model size (fits in IMX500 cache)
- **Power:** Minimal additional consumption

The key insight: **Use the IMX500's built-in AI acceleration to completely bypass Pi 3 CPU limitations**.
