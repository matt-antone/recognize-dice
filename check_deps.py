#!/usr/bin/env python3
"""
Dependency Checker for Pi 3 AI Dice Detection
"""

import sys

print("🔍 Checking Dependencies for AI Dice Detection")
print("=" * 50)

# Check Python version
print(f"🐍 Python: {sys.version}")

# Check individual packages
packages = [
    ("picamera2", "Camera interface"),
    ("cv2", "OpenCV for image processing"),
    ("numpy", "Numerical operations"),
    ("tensorflow", "TensorFlow (full)"),
    ("tflite_runtime.interpreter", "TensorFlow Lite Runtime")
]

available = []
missing = []

for package, description in packages:
    try:
        __import__(package)
        print(f"✅ {package} - {description}")
        available.append(package)
    except ImportError:
        print(f"❌ {package} - {description}")
        missing.append(package)

print("\n📊 Summary:")
print(f"✅ Available: {len(available)}")
print(f"❌ Missing: {len(missing)}")

if missing:
    print("\n📦 Installation Suggestions:")
    
    if "picamera2" in missing:
        print("   sudo apt install python3-picamera2")
    
    if "cv2" in missing:
        print("   sudo apt install python3-opencv")
    
    if "numpy" in missing:
        print("   sudo apt install python3-numpy")
    
    if "tensorflow" in missing and "tflite_runtime.interpreter" in missing:
        print("   # Try system package first:")
        print("   sudo apt install python3-tflite-runtime")
        print("   # Or use virtual environment:")
        print("   python3 -m venv ~/dice_env")
        print("   source ~/dice_env/bin/activate")
        print("   pip install tflite-runtime")

else:
    print("\n🎉 All dependencies available! Ready to run AI dice detection!")

print("\n💡 Next step: python3 test_dice_ai.py") 