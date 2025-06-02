#!/bin/bash

# Pi 3 + Sony IMX500 AI Camera Dependency Setup

echo "🚀 Setting up Pi 3 + Sony IMX500 AI Camera dependencies..."

# Step 1: Install system packages
echo "📦 Installing system packages..."
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-numpy

# Step 2: Install IMX500 firmware (if not already done)
echo "🤖 Installing IMX500 AI firmware..."
sudo apt install imx500-all

# Step 3: Activate virtual environment and install Python packages
echo "🐍 Setting up virtual environment..."
source ~/dice_env/bin/activate
pip install tflite-runtime opencv-python numpy

# Step 4: Test the setup
echo "🧪 Testing dependencies..."
python3 -c "import picamera2; print('✅ picamera2 OK')"
python3 -c "import cv2; print('✅ OpenCV OK')"
python3 -c "import numpy; print('✅ NumPy OK')"
python3 -c "import tflite_runtime; print('✅ TFLite OK')"

# Step 5: Test AI acceleration
echo "🚀 Testing AI acceleration..."
rpicam-hello --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json

echo "✅ Setup complete! Your Pi 3 + Sony IMX500 is ready for dice detection." 