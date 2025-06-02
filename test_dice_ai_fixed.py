#!/usr/bin/env python3
"""
AI Dice Detection Test - FIXED VERSION
Fixed false detection issue with proper confidence thresholding
"""

import time
import numpy as np
import cv2
from pathlib import Path

try:
    from picamera2 import Picamera2
    try:
        import tensorflow as tf
    except ImportError:
        import tflite_runtime.interpreter as tflite
        class TFCompat:
            class lite:
                @staticmethod
                def Interpreter(model_path):
                    return tflite.Interpreter(model_path)
        tf = TFCompat()
    
    CAMERA_AVAILABLE = True
    print("✅ Camera and TensorFlow Lite available")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    exit(1)

def load_model(model_version="v3"):
    """Load the specified TensorFlow Lite dice model"""
    model_paths = {
        "v2": "models/edge_impulse/dice_classifier_v2.tflite",
        "v3": "models/edge_impulse/dice_classifier_v3.tflite"
    }
    
    model_path = Path(model_paths.get(model_version, model_paths["v3"]))
    
    if not model_path.exists():
        print(f"❌ Model {model_version} not found: {model_path}")
        return None
    
    try:
        interpreter = tf.lite.Interpreter(str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        print(f"✅ Model {model_version} loaded: {model_path.name}")
        print(f"📊 Model size: {model_size_mb:.1f}MB")
        print(f"📊 Input shape: {input_details[0]['shape']}")
        print(f"📊 Output shape: {output_details[0]['shape']}")
        
        return {
            'interpreter': interpreter,
            'input_details': input_details,
            'output_details': output_details,
            'version': model_version,
            'size_mb': model_size_mb,
            'path': str(model_path)
        }
        
    except Exception as e:
        print(f"❌ Model {model_version} loading failed: {e}")
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

def run_inference(model_data, image):
    """Run TensorFlow Lite inference on image"""
    try:
        interpreter = model_data['interpreter']
        input_details = model_data['input_details']
        output_details = model_data['output_details']
        
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

def parse_detection_output_fixed(output_data, model_version="v3", confidence_threshold=0.5):
    """
    FIXED: Parse model output with proper confidence thresholding
    This version prevents false detections when no dice are present
    """
    
    if len(output_data.shape) == 2:
        # Classification model output
        class_probs = output_data[0]
        max_idx = np.argmax(class_probs)
        max_confidence = class_probs[max_idx]
        
        # CRITICAL FIX: Only return detection if confidence exceeds threshold
        if max_confidence < confidence_threshold:
            return "No dice detected (confidence too low)"
        
        # Map class index to dice value (assuming 0-5 maps to 1-6)
        dice_value = (max_idx % 6) + 1
        
        return f"Dice: {dice_value} (conf: {max_confidence:.3f})"
        
    elif len(output_data.shape) == 3:
        # Object detection model output
        valid_detections = []
        
        for detection in output_data[0]:
            if len(detection) >= 6:  # coords + confidence + class
                x, y, w, h, confidence, class_id = detection[:6]
                
                # CRITICAL FIX: Apply confidence threshold
                if confidence > confidence_threshold:
                    dice_value = (int(class_id) % 6) + 1
                    valid_detections.append(f"Dice: {dice_value} ({confidence:.3f})")
        
        if valid_detections:
            return " | ".join(valid_detections)
        else:
            return "No dice detected"
    
    else:
        return f"Unknown output shape: {output_data.shape}"

def test_confidence_levels(output_data):
    """Test different confidence levels to help tune the threshold"""
    results = {}
    
    if len(output_data.shape) == 2:
        class_probs = output_data[0]
        max_idx = np.argmax(class_probs)
        max_confidence = class_probs[max_idx]
        dice_value = (max_idx % 6) + 1
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for thresh in thresholds:
            if max_confidence >= thresh:
                results[thresh] = f"Dice: {dice_value} (conf: {max_confidence:.3f})"
            else:
                results[thresh] = "No dice detected"
    
    return results

def main():
    """Main testing function with improved detection logic"""
    print("🎲 AI Dice Detection Test - FIXED VERSION")
    print("🔧 Now with proper confidence thresholding to prevent false detections")
    print("=" * 70)
    
    # Load model
    model_data = load_model("v3")
    if not model_data:
        return
    
    # Initialize camera
    print(f"\n📸 Initializing AI Camera...")
    try:
        picam2 = Picamera2()
        
        config = picam2.create_still_configuration(
            main={"size": (640, 640)},
            display="main"
        )
        picam2.configure(config)
        picam2.start()
        
        print(f"✅ Camera ready!")
        print(f"🎲 Using model: {model_data['version']} ({model_data['size_mb']:.1f}MB)")
        
        # Set confidence threshold
        confidence_threshold = 0.5  # Adjustable threshold
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        print(f"📋 Test with NO DICE first to verify no false detections!")
        
        while True:
            try:
                user_input = input("\nPress Enter to capture (or 't' for threshold test, Ctrl+C to exit): ")
                
                start_time = time.time()
                
                # Capture image
                image = picam2.capture_array()
                capture_time = time.time() - start_time
                
                # Run inference
                inference_start = time.time()
                output = run_inference(model_data, image)
                inference_time = time.time() - inference_start
                
                if output is not None:
                    # Use fixed detection logic
                    result = parse_detection_output_fixed(output, model_data['version'], confidence_threshold)
                    
                    total_time = capture_time + inference_time
                    
                    print(f"📊 Result: {result}")
                    print(f"⏱️ Timing: Capture {capture_time*1000:.1f}ms + Inference {inference_time*1000:.1f}ms = {total_time*1000:.1f}ms")
                    print(f"🎥 FPS: {1/total_time:.1f}")
                    print(f"🤖 Model: {model_data['version']} ({model_data['size_mb']:.1f}MB)")
                    
                    # Show threshold testing if requested
                    if user_input.lower() == 't':
                        print(f"\n🔍 THRESHOLD TESTING:")
                        threshold_results = test_confidence_levels(output)
                        for thresh, res in threshold_results.items():
                            print(f"   {thresh}: {res}")
                else:
                    print("❌ Inference failed")
                
                print("-" * 50)
                
            except EOFError:
                break
            
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