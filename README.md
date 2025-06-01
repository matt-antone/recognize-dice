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

## Performance Expectations (Pi 3)

- **Frame Rate**: 3-5 FPS
- **Accuracy**: >90% for well-lit conditions
- **Startup Time**: ~15 seconds
- **Memory Usage**: <800MB

## Quick Start

### 1. Setup Environment

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install python3-pip python3-venv python3-tk
sudo apt install libcamera-dev python3-opencv

# Clone repository
git clone <your-repo-url>
cd recognize-dice

# Create virtual environment
python3 -m venv dice_env
source dice_env/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Test Camera

```bash
# Test camera functionality
python3 -c "from picamera2 import Picamera2; print('Camera test passed')"
```

### 3. Run Application

```bash
source dice_env/bin/activate
python3 main.py
```

## Project Structure

```
recognize-dice/
├── main.py              # Main application entry point
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

### Camera Settings

- Resolution: 320x320 (optimized for Pi 3)
- Format: RGB24
- Frame rate: Limited by processing speed

### Model Settings

- Input size: 320x320 pixels
- Quantization: INT8 for speed
- Confidence threshold: 0.5 (adjustable)

## Troubleshooting

### Common Issues

**Camera not detected:**

```bash
# Check camera connection
vcgencmd get_camera

# Enable camera interface
sudo raspi-config
# Navigate to Interface Options > Camera > Enable
```

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
- Adjust confidence thresholds

## Development

### Adding New Features

1. Check memory-bank/ documentation for project context
2. Follow established patterns in src/ modules
3. Test on actual Pi 3 hardware

### Performance Optimization

- Profile with `cProfile` for bottlenecks
- Monitor memory usage with `psutil`
- Test thermal performance during extended use

## Based on Reference Work

This project adapts techniques from [nell-byler/dice_detection](https://github.com/nell-byler/dice_detection) for Raspberry Pi 3 constraints, using TensorFlow Lite optimization and reduced model complexity.

## License

BSD-3-Clause (following reference project)

## Contributing

1. Test changes on actual Pi 3 hardware
2. Update memory-bank/ documentation for significant changes
3. Maintain performance targets for Pi 3 constraints

---

**Note**: This application is specifically optimized for Raspberry Pi 3. For Pi 4 or other hardware, consider adjusting performance targets and model complexity in the configuration.
