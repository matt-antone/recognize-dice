#!/usr/bin/env python3
"""
Quick Model Performance Test for Edge Impulse Dice Detection
Tests real-world accuracy vs training metrics
"""

import time
import json
from pathlib import Path
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("⚠️  Camera not available - will simulate testing")

def test_model_deployment():
    """Test Edge Impulse model deployment and performance"""
    
    print("🎲 Testing Edge Impulse Dice Model Performance")
    print("=" * 60)
    
    # Check for model files
    model_dir = Path("../../models/edge_impulse")
    model_files = list(model_dir.glob("*.tflite")) if model_dir.exists() else []
    
    if not model_files:
        print("❌ No .tflite model found in models/edge_impulse/")
        print("📥 Download guide:")
        print("   1. Edge Impulse Studio → 'Deployment' tab")
        print("   2. Search for 'TensorFlow Lite'")
        print("   3. Select 'TensorFlow Lite (int8 quantized)'")
        print("   4. Click 'Build' → Download model.tflite")
        print("   5. Move to models/edge_impulse/dice_model.tflite")
        print("")
        print("📋 Alternative: C++ Library deployment:")
        print("   1. Search for 'C++ Library'")
        print("   2. Select 'TensorFlow Lite / Unoptimized float32'")  
        print("   3. Extract .tflite from the library package")
        print("")
        print("💡 You have the C++ SDK at: deployment/pi3/edge_impulse_cpp_sdk/")
        print("   This contains the compiled model, but we need .tflite for Python")
        return
    
    model_file = model_files[0]
    print(f"✅ Found model: {model_file.name}")
    
    # Test AI acceleration compatibility
    test_ai_acceleration()
    
    # Performance testing
    if CAMERA_AVAILABLE:
        test_real_camera_performance(model_file)
    else:
        test_simulated_performance(model_file)

def test_ai_acceleration():
    """Test if AI acceleration is working"""
    print("\n🚀 Testing AI Acceleration...")
    
    try:
        import subprocess
        result = subprocess.run([
            "rpicam-hello", "--info-text", "AI Test", "-t", "2000",
            "--post-process-file", "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ IMX500 AI acceleration working!")
            return True
        else:
            print("⚠️  AI acceleration test failed")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not test AI acceleration: {e}")
        return False

def test_real_camera_performance(model_file):
    """Test with real camera hardware"""
    print(f"\n📸 Testing Real Camera Performance...")
    print(f"Model: {model_file.name}")
    
    try:
        # Test TensorFlow Lite import
        try:
            import tensorflow as tf
            print("✅ TensorFlow Lite available")
        except ImportError:
            print("❌ TensorFlow Lite not available - install with:")
            print("   pip install tensorflow-lite-runtime")
            return
        
        # Initialize camera (simplified for testing)
        picam2 = Picamera2()
        
        # Basic configuration for testing
        config = picam2.create_still_configuration(
            main={"size": (640, 640)},
            display="main"
        )
        picam2.configure(config)
        picam2.start()
        
        print("✅ Camera initialized")
        print("🎲 Place dice in view and press Enter to test...")
        input()
        
        # Load TensorFlow Lite model
        interpreter = tf.lite.Interpreter(str(model_file))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📊 Model info:")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        
        # Capture test images
        test_results = []
        
        for i in range(5):
            print(f"📷 Capture {i+1}/5...")
            
            start_time = time.time()
            
            # Capture image
            image_array = picam2.capture_array()
            capture_time = time.time() - start_time
            
            # Run actual TensorFlow Lite inference
            inference_start = time.time()
            dice_prediction = run_tflite_inference(interpreter, image_array, input_details, output_details)
            inference_time = time.time() - inference_start
            
            total_time = capture_time + inference_time
            
            result = {
                "capture": f"{capture_time*1000:.1f}ms",
                "inference": f"{inference_time*1000:.1f}ms", 
                "total": f"{total_time*1000:.1f}ms",
                "prediction": dice_prediction
            }
            
            test_results.append(result)
            print(f"   Result: {dice_prediction} | Total: {total_time*1000:.1f}ms")
            
            time.sleep(1)
        
        picam2.stop()
        
        # Performance summary
        print_performance_summary(test_results)
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        print("💡 Try: sudo usermod -a -G video $USER && reboot")

def run_tflite_inference(interpreter, image_array, input_details, output_details):
    """Run actual TensorFlow Lite inference"""
    import numpy as np
    import cv2
    
    try:
        # Get input shape and preprocess image
        input_shape = input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize and preprocess image for model
        if len(image_array.shape) == 3:
            # Convert RGB to grayscale if needed
            if image_array.shape[2] == 3:
                image_resized = cv2.resize(image_array, (width, height))
            else:
                image_resized = cv2.resize(image_array[:,:,0], (width, height))
        else:
            image_resized = cv2.resize(image_array, (width, height))
        
        # Normalize to 0-1 range
        if input_details[0]['dtype'] == np.float32:
            input_data = np.expand_dims(image_resized.astype(np.float32) / 255.0, axis=0)
        else:
            input_data = np.expand_dims(image_resized.astype(np.uint8), axis=0)
        
        # Ensure correct input shape
        if len(input_shape) == 4 and input_shape[3] == 3:
            # Model expects RGB
            if len(input_data.shape) == 3:
                input_data = np.expand_dims(input_data, axis=-1)
            if input_data.shape[3] == 1:
                input_data = np.repeat(input_data, 3, axis=3)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Parse prediction (assumes classification output)
        if len(output_data.shape) == 2:
            prediction_index = np.argmax(output_data[0])
            confidence = output_data[0][prediction_index]
            
            # Dice values typically 1-6
            dice_value = prediction_index + 1 if prediction_index < 6 else prediction_index
            
            return f"Dice: {dice_value} (conf: {confidence:.3f})"
        else:
            return f"Output shape: {output_data.shape} | Raw: {output_data}"
            
    except Exception as e:
        return f"Inference error: {e}"

def simulate_inference(image_array):
    """Simulate model inference (replace with actual TFLite)"""
    import random
    
    # Simulate processing time
    time.sleep(0.05)  # 50ms typical inference
    
    # Simulate dice detection
    dice_values = [1, 2, 3, 4, 5, 6]
    confidence = random.uniform(0.6, 0.95)
    
    if confidence > 0.8:
        return f"Dice: {random.choice(dice_values)} (conf: {confidence:.2f})"
    else:
        return f"Uncertain (conf: {confidence:.2f})"

def test_simulated_performance(model_file):
    """Test with simulated data (development environment)"""
    print(f"\n🖥️  Testing Simulated Performance...")
    print(f"Model: {model_file.name}")
    
    # Test loading TensorFlow Lite
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(str(model_file))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model info:")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Input type: {input_details[0]['dtype']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        print(f"   Output type: {output_details[0]['dtype']}")
        
    except ImportError:
        print("❌ TensorFlow Lite not available")
        print("📦 Install with: pip install tensorflow-lite-runtime")
        print("📊 Simulating performance instead...")
    except Exception as e:
        print(f"⚠️  Model loading error: {e}")
        print("📊 Simulating performance instead...")
    
    print("📊 Simulating 10 detection cycles...")
    
    results = []
    for i in range(10):
        start_time = time.time()
        
        # Simulate image processing
        time.sleep(0.02)  # 20ms capture simulation
        
        # Simulate inference
        prediction = simulate_inference(None)
        
        total_time = time.time() - start_time
        
        results.append({
            "total": f"{total_time*1000:.1f}ms",
            "prediction": prediction
        })
        
        print(f"   Test {i+1}: {prediction} | {total_time*1000:.1f}ms")
    
    print_performance_summary(results)

def print_performance_summary(results):
    """Print performance analysis"""
    print("\n📊 Performance Summary:")
    print("-" * 40)
    
    if not results:
        return
    
    # Extract timing data
    times = []
    for result in results:
        time_str = result["total"].replace("ms", "")
        times.append(float(time_str))
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    
    print(f"⏱️  Average latency: {avg_time:.1f}ms")
    print(f"⚡ Best case: {min_time:.1f}ms") 
    print(f"🐌 Worst case: {max_time:.1f}ms")
    
    # FPS calculation
    avg_fps = 1000 / avg_time if avg_time > 0 else 0
    print(f"🎥 Estimated FPS: {avg_fps:.1f}")
    
    # Performance assessment
    if avg_time < 100:
        print("✅ Excellent performance!")
    elif avg_time < 200:
        print("👍 Good performance")
    elif avg_time < 500:
        print("⚠️  Acceptable performance")
    else:
        print("❌ Poor performance - optimization needed")
    
    print("\n💡 Next Steps:")
    if avg_time > 200:
        print("   • Try smaller input resolution (320x320 → 160x160)")
        print("   • Enable INT8 quantization in Edge Impulse")
        print("   • Consider frame skipping (process every 2nd frame)")
    
    print("   • Test with actual dice in various lighting conditions")
    print("   • Compare accuracy with your training metrics")
    print("   • Document failure cases for model improvement")
    print("   • Consider IMX500 hardware acceleration for optimal performance")

if __name__ == "__main__":
    test_model_deployment() 