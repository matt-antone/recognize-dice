#!/usr/bin/env python3
"""
AI Dice Detection Test - Pi 3 + IMX500 AI Camera
Simple test script for TensorFlow Lite dice classification
"""

import time
import numpy as np
import cv2
from pathlib import Path

try:
    from picamera2 import Picamera2
    import tensorflow as tf
    CAMERA_AVAILABLE = True
    print("✅ Camera and TensorFlow Lite available")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📦 Install: sudo apt install python3-picamera2")
    print("📦 Install: pip install tensorflow-lite-runtime")
    exit(1)

def load_model():
    """Load the TensorFlow Lite dice model"""
    model_path = Path("models/edge_impulse/dice_classifier_v2.tflite")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("📥 Make sure you have the model file in models/edge_impulse/")
        return None
    
    try:
        interpreter = tf.lite.Interpreter(str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ Model loaded: {model_path.name}")
        print(f"📊 Input shape: {input_details[0]['shape']}")
        print(f"📊 Output shape: {output_details[0]['shape']}")
        
        return interpreter, input_details, output_details
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def preprocess_image(image, input_details):
    """Preprocess camera image for model input"""
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    
    # Resize image to model input size
    resized = cv2.resize(image, (width, height))
    
    # Normalize to 0-1 range if model expects float32
    if input_details[0]['dtype'] == np.float32:
        processed = resized.astype(np.float32) / 255.0
    else:
        processed = resized.astype(np.uint8)
    
    # Add batch dimension
    return np.expand_dims(processed, axis=0)

def run_inference(interpreter, image, input_details, output_details):
    """Run TensorFlow Lite inference on image"""
    try:
        # Preprocess image
        input_data = preprocess_image(image, input_details)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        return output_data
        
    except Exception as e:
        print(f"⚠️ Inference error: {e}")
        return None

def parse_detection_output(output_data):
    """Parse model output for dice detection"""
    # For object detection models, output often contains:
    # [batch, detections, 4_coords + confidence + class_scores]
    
    if len(output_data.shape) == 2:
        # Classification output: find highest confidence class
        class_idx = np.argmax(output_data[0])
        confidence = output_data[0][class_idx]
        
        # Assume classes 0-5 map to dice values 1-6
        dice_value = (class_idx % 6) + 1
        
        return f"Dice: {dice_value} (conf: {confidence:.3f})"
        
    elif len(output_data.shape) == 3:
        # Object detection output: multiple detections possible
        detections = []
        
        # Simple parsing - look for high confidence detections
        for detection in output_data[0]:
            if len(detection) >= 6:  # coords + conf + class
                confidence = detection[4]
                if confidence > 0.5:  # confidence threshold
                    class_idx = int(detection[5])
                    dice_value = (class_idx % 6) + 1
                    detections.append(f"Dice: {dice_value} ({confidence:.3f})")
        
        if detections:
            return " | ".join(detections)
        else:
            return "No dice detected"
    
    else:
        return f"Unknown output shape: {output_data.shape}"

def test_ai_acceleration():
    """Test IMX500 AI acceleration"""
    print("\n🚀 Testing IMX500 AI Acceleration...")
    
    try:
        import subprocess
        result = subprocess.run([
            "rpicam-hello", "--info-text", "AI Test", "-t", "3000",
            "--post-process-file", "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ IMX500 AI acceleration working!")
            return True
        else:
            print("⚠️ AI acceleration test failed - using CPU inference")
            return False
            
    except Exception as e:
        print(f"⚠️ Could not test AI acceleration: {e}")
        return False

def main():
    """Main testing function"""
    print("🎲 AI Dice Detection Test - Pi 3 + IMX500")
    print("=" * 50)
    
    # Test AI acceleration
    ai_acceleration = test_ai_acceleration()
    
    # Load model
    model_data = load_model()
    if not model_data:
        return
    
    interpreter, input_details, output_details = model_data
    
    # Initialize camera
    print("\n📸 Initializing AI Camera...")
    try:
        picam2 = Picamera2()
        
        # Configure for dice detection
        config = picam2.create_still_configuration(
            main={"size": (640, 640)},  # Good balance of quality/speed
            display="main"
        )
        picam2.configure(config)
        picam2.start()
        
        print("✅ Camera ready!")
        print("🎲 Place dice in view and press Enter to detect...")
        
        while True:
            input("Press Enter to capture and detect (Ctrl+C to exit): ")
            
            start_time = time.time()
            
            # Capture image
            image = picam2.capture_array()
            capture_time = time.time() - start_time
            
            # Run inference
            inference_start = time.time()
            output = run_inference(interpreter, image, input_details, output_details)
            inference_time = time.time() - inference_start
            
            # Parse results
            if output is not None:
                result = parse_detection_output(output)
                total_time = capture_time + inference_time
                
                print(f"📊 Result: {result}")
                print(f"⏱️ Timing: Capture {capture_time*1000:.1f}ms + Inference {inference_time*1000:.1f}ms = {total_time*1000:.1f}ms")
                print(f"🎥 FPS: {1/total_time:.1f}")
                
                if ai_acceleration:
                    print("🚀 Note: AI acceleration available for further optimization")
            else:
                print("❌ Inference failed")
            
            print("-" * 30)
            
    except KeyboardInterrupt:
        print("\n👋 Stopping dice detection...")
    except Exception as e:
        print(f"❌ Camera error: {e}")
    finally:
        try:
            picam2.stop()
        except:
            pass

if __name__ == "__main__":
    main() 