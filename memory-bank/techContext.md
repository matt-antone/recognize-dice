# Technical Context: D6 Dice Recognition App

## Target Hardware

- **Raspberry Pi**: Model 3B/3B+ (1GB RAM, ARM Cortex-A53 @ 1.2GHz)
- **Raspberry Pi Camera**: V2 or HQ camera module (need to verify user's module)
- **Display**: HDMI monitor or Pi touchscreen for output
- **Storage**: 32GB+ microSD card (Class 10 minimum)

## Core Technologies

### Computer Vision & AI

- **TensorFlow Lite**: Optimized for Raspberry Pi inference
- **OpenCV**: Image processing and camera interface
- **NumPy**: Numerical operations
- **Reference Model**: Based on nell-byler/dice_detection project

### Development Stack

- **Python 3.8+**: Primary development language
- **picamera2**: Raspberry Pi camera interface (or picamera for older setups)
- **Tkinter**: Lightweight GUI framework (chosen over PyQt5 for Pi 3)
- **Raspberry Pi OS**: Target operating system

### Model Architecture

- **Base**: SSD MobileNet (from reference project)
- **Optimization**: Aggressive TensorFlow Lite quantization (INT8)
- **Input**: 320x320 RGB images (reduced from 640x640 for Pi 3)
- **Output**: Bounding boxes + classification (1-6)

## Technical Constraints

### Hardware Limitations (Pi 3 Specific)

- **Processing Power**: ARM Cortex-A53 @ 1.2GHz (significantly slower than Pi 4)
- **Memory**: Only 1GB RAM (major constraint vs 2-8GB on Pi 4)
- **Storage**: SD card I/O limitations
- **Power**: microUSB power (less power delivery than Pi 4)

### Performance Requirements (Adjusted for Pi 3)

- **Inference Time**: <300ms per frame (relaxed from <200ms)
- **Memory Usage**: <800MB total application footprint (reduced from <1GB)
- **Frame Rate**: 3-5 FPS target (reduced from 5-15 FPS)
- **Startup Time**: <15 seconds application launch (increased from <10s)

## Development Environment

### Local Setup

```bash
# Required system packages
sudo apt update
sudo apt install python3-pip python3-venv
sudo apt install libcamera-dev python3-opencv
sudo apt install python3-tk  # For Tkinter GUI

# Python environment
python3 -m venv dice_env
source dice_env/bin/activate
pip install -r requirements.txt
```

### Dependencies (Pi 3 Optimized)

```
tensorflow-lite-runtime>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
picamera2>=0.3.0  # or picamera if using older camera
pillow>=8.0.0
# Note: Avoiding PyQt5 to reduce memory footprint
```

### Model Pipeline

1. **Image Capture**: Pi Camera → Raw frame (lower resolution)
2. **Preprocessing**: Resize to 320x320, normalize, format for model
3. **Inference**: TensorFlow Lite model execution (heavily quantized)
4. **Post-processing**: Parse detections, apply confidence thresholds
5. **Display**: Render results on Tkinter GUI

## Key Technical Decisions

### Model Choice (Pi 3 Specific)

- **SSD MobileNet**: Best balance of speed/accuracy for limited hardware
- **TensorFlow Lite**: Essential for Pi 3 performance
- **INT8 Quantization**: Maximum speed optimization
- **Reduced Input Size**: 320x320 instead of 640x640

### Camera Integration

- **picamera2**: Preferred for newer setups
- **picamera**: Fallback for older camera modules
- **Resolution**: 320x320 for model input (significant reduction)
- **Format**: RGB24 for model compatibility

### Performance Optimizations (Critical for Pi 3)

- **Aggressive Model Quantization**: INT8 weights and activations
- **Frame Skipping**: Process every 2-3 frames to maintain responsiveness
- **Memory Streaming**: No frame buffering, process immediately
- **Single-threaded**: Avoid threading overhead on limited cores
- **Minimal GUI**: Tkinter instead of PyQt5 for lower memory usage

### Pi 3 Specific Adaptations

- **Reduced Model Complexity**: May need simpler architecture
- **Lower Resolution Processing**: 320x320 max input size
- **Simplified GUI**: Basic Tkinter interface
- **Conservative Memory Management**: Immediate cleanup of processed frames
- **Thermal Considerations**: Monitor for thermal throttling

## Fallback Strategies

- **Model Fallback**: Simple template matching if ML model too slow
- **Resolution Scaling**: Further reduce to 160x160 if needed
- **Frame Rate Adaptation**: Dynamic frame skipping based on performance
- **Confidence Thresholds**: Higher thresholds to reduce false positives
