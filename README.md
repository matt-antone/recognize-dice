# D6 Dice Recognition for Raspberry Pi 3

Real-time dice recognition application optimized for Raspberry Pi 3 with camera module. Detects and identifies the face value (1-6) of standard six-sided dice using computer vision and machine learning.

## Hardware Requirements

- **Raspberry Pi 3B/3B+** (1GB RAM)
- **Raspberry Pi Camera Module** (V2 or HQ camera)
- **Display**: HDMI monitor or touchscreen
- **Storage**: 32GB+ microSD card (Class 10)
- **Power**: 5V 2.5A microUSB power supply

## Features

- Real-time dice detection and classification
- Optimized for Raspberry Pi 3 performance constraints
- Simple, lightweight user interface
- Support for standard D6 dice
- Confidence scoring for reliable detection
- Fallback computer vision when ML model unavailable

## Performance Expectations (Pi 3)

- **Frame Rate**: 3-5 FPS
- **Accuracy**: >90% for well-lit conditions (with ML model)
- **Startup Time**: ~15 seconds
- **Memory Usage**: <800MB

## Quick Start

### 1. Setup Environment

**Option A: Automatic Installation (Recommended)**

```bash
# Update system (on Raspberry Pi)
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv python3-tk
sudo apt install libcamera-dev python3-opencv

# Clone repository
git clone <your-repo-url>
cd recognize-dice

# Create virtual environment
python3 -m venv dice_env
source dice_env/bin/activate

# Run platform-specific installer
python3 install_deps.py
```

**Option B: Manual Installation**

```bash
# Create virtual environment
python3 -m venv dice_env
source dice_env/bin/activate

# Install base requirements
pip install opencv-python numpy pillow psutil

# Platform-specific TensorFlow Lite:
# On Raspberry Pi 3:
pip install tensorflow-lite-runtime>=2.8.0

# On development machines (macOS/Windows/Linux):
pip install tensorflow>=2.8.0

# Camera support (Raspberry Pi only):
pip install picamera2  # or picamera for older setups
```

### 2. Test Installation

```bash
# Test dependencies and camera
python3 test_camera.py
```

### 3. Run Application

```bash
source dice_env/bin/activate
python3 main.py
```

## Platform Support

### Raspberry Pi 3 (Target Platform)

- **Full functionality**: Camera + ML model + GUI
- **Optimized performance**: Frame skipping, memory streaming
- **TensorFlow Lite**: Lightweight ML inference

### Development Platforms (macOS/Windows/Linux)

- **Limited functionality**: GUI + fallback detection only
- **No camera**: Uses simulated or imported images
- **Full TensorFlow**: For model development and testing

### Testing on Development Machines

Even without a Pi camera, you can test the GUI and fallback detection:

```bash
python3 main.py  # Will show GUI with fallback CV detection
```

## Project Structure

```
recognize-dice/
├── main.py              # Main application entry point
├── test_camera.py       # Camera testing utility
├── install_deps.py      # Platform-specific installer
├── src/
│   ├── camera/          # Camera interface modules
│   ├── detection/       # AI model and processing
│   ├── gui/            # User interface components
│   └── utils/          # Utility functions
├── models/             # TensorFlow Lite model files
├── memory-bank/        # Project documentation
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Configuration

### Camera Settings (Pi 3 Optimized)

- Resolution: 320x320 (reduced for performance)
- Format: RGB24
- Frame rate: 3-5 FPS actual (15 FPS max)

### Model Settings

- Input size: 320x320 pixels
- Quantization: INT8 for speed
- Confidence threshold: 0.5 (adjustable)

## Troubleshooting

### Installation Issues

**TensorFlow Lite not available:**

```bash
# The app will automatically use fallback detection
# This is normal on development platforms
python3 main.py  # Still works with computer vision fallback
```

**Camera not detected (Raspberry Pi):**

```bash
# Check camera connection
vcgencmd get_camera

# Enable camera interface
sudo raspi-config
# Navigate to Interface Options > Camera > Enable
```

**Import errors:**

```bash
# Re-run the installer
python3 install_deps.py

# Or check what's missing
python3 -c "import cv2, numpy, PIL, tkinter; print('Core deps OK')"
```

### Performance Issues

**Memory errors:**

- Reduce input resolution further (160x160)
- Close other applications
- Consider using swap file

**Poor performance:**

- Ensure adequate power supply (2.5A minimum)
- Check for thermal throttling: `vcgencmd measure_temp`
- Monitor CPU usage: `htop`

**Low accuracy:**

- Improve lighting conditions
- Use dice with clear, contrasting dots
- Adjust confidence thresholds in config

## Development

### Cross-Platform Development

1. **Develop on any platform**: macOS/Windows/Linux supported
2. **Test locally**: Use fallback detection for GUI testing
3. **Deploy to Pi**: Full functionality with camera and ML model

### Adding New Features

1. Check memory-bank/ documentation for project context
2. Follow established patterns in src/ modules
3. Test on actual Pi 3 hardware when possible

### Performance Optimization

- Profile with `cProfile` for bottlenecks
- Monitor memory usage with `psutil`
- Test thermal performance during extended use

## Model Integration

The app framework is ready for the TensorFlow Lite model from [nell-byler/dice_detection](https://github.com/nell-byler/dice_detection). Currently uses fallback computer vision detection until the model is converted and integrated.

### Current Status

- ✅ **Application framework**: Complete and working
- ✅ **Camera interface**: Pi 3 optimized with dual compatibility
- ✅ **GUI system**: Lightweight Tkinter interface
- ✅ **Fallback detection**: Computer vision using HoughCircles
- 🔄 **ML model**: Framework ready, needs model conversion

## License

BSD-3-Clause (following reference project)

## Contributing

1. Test changes on actual Pi 3 hardware when possible
2. Update memory-bank/ documentation for significant changes
3. Maintain performance targets for Pi 3 constraints
4. Use cross-platform development approach

---

**Note**: This application works on development platforms for testing the GUI and fallback detection, but full functionality (camera + ML model) requires Raspberry Pi 3 hardware.

# If using SSH, enable X11 forwarding:

ssh -X pi@your-pi-ip

# Or if running directly on Pi with monitor:

export DISPLAY=:0

# Or run GUI apps with:

DISPLAY=:0 python3 main.py
