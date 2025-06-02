#!/usr/bin/env python3
"""
Debug False Detection - Understanding Model Output
Diagnostic script to analyze why dice are detected when none are present
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
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    exit(1)

def load_model():
    """Load the v3 TensorFlow Lite dice model"""
    model_path = Path("models/edge_impulse/dice_classifier_v3.tflite")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    try:
        interpreter = tf.lite.Interpreter(str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ Model loaded: {model_path.name}")
        print(f"📊 Input shape: {input_details[0]['shape']}")
        print(f"📊 Input dtype: {input_details[0]['dtype']}")
        print(f"📊 Output shape: {output_details[0]['shape']}")
        print(f"📊 Output dtype: {output_details[0]['dtype']}")
        
        return {
            'interpreter': interpreter,
            'input_details': input_details,
            'output_details': output_details
        }
        
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

def run_inference_with_debug(model_data, image):
    """Run inference and provide detailed output analysis"""
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

def analyze_output_in_detail(output_data):
    """Analyze model output in detail to understand false positives"""
    print(f"\n🔍 RAW OUTPUT ANALYSIS:")
    print(f"   Shape: {output_data.shape}")
    print(f"   Min value: {np.min(output_data):.6f}")
    print(f"   Max value: {np.max(output_data):.6f}")
    print(f"   Mean value: {np.mean(output_data):.6f}")
    
    if len(output_data.shape) == 2:
        # Classification model
        print(f"\n📊 CLASSIFICATION MODEL OUTPUT:")
        print(f"   Classes found: {output_data.shape[1]}")
        
        # Show all class probabilities
        class_probs = output_data[0]
        for i, prob in enumerate(class_probs):
            dice_value = (i % 6) + 1
            print(f"   Class {i} (Dice {dice_value}): {prob:.6f}")
        
        # Find highest probability
        max_idx = np.argmax(class_probs)
        max_prob = class_probs[max_idx]
        predicted_dice = (max_idx % 6) + 1
        
        print(f"\n🎯 HIGHEST PROBABILITY:")
        print(f"   Class {max_idx} (Dice {predicted_dice}): {max_prob:.6f}")
        
        # Apply different confidence thresholds
        print(f"\n🚪 CONFIDENCE THRESHOLDS:")
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for thresh in thresholds:
            if max_prob > thresh:
                result = f"Dice {predicted_dice} (conf: {max_prob:.3f})"
            else:
                result = "No dice detected"
            print(f"   Threshold {thresh}: {result}")
        
        return predicted_dice, max_prob
        
    elif len(output_data.shape) == 3:
        # Object detection model
        print(f"\n📊 OBJECT DETECTION MODEL OUTPUT:")
        print(f"   Detections: {output_data.shape[1]}")
        print(f"   Detection data size: {output_data.shape[2]}")
        
        detections = []
        print(f"\n🔍 ALL DETECTIONS:")
        for i, detection in enumerate(output_data[0]):
            if len(detection) >= 6:
                x, y, w, h, confidence, class_id = detection[:6]
                dice_value = (int(class_id) % 6) + 1
                print(f"   Detection {i}: x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}, conf={confidence:.6f}, dice={dice_value}")
                
                if confidence > 0.1:  # Very low threshold for analysis
                    detections.append((dice_value, confidence))
        
        # Apply different confidence thresholds
        print(f"\n🚪 CONFIDENCE THRESHOLDS:")
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for thresh in thresholds:
            valid_detections = [d for d in detections if d[1] > thresh]
            if valid_detections:
                result = " | ".join([f"Dice {d[0]} ({d[1]:.3f})" for d in valid_detections])
            else:
                result = "No dice detected"
            print(f"   Threshold {thresh}: {result}")
        
        if detections:
            best_detection = max(detections, key=lambda x: x[1])
            return best_detection[0], best_detection[1]
        else:
            return None, 0.0
    
    else:
        print(f"❓ Unknown output format: {output_data.shape}")
        return None, 0.0

def improved_detection_with_threshold(output_data, confidence_threshold=0.5):
    """Improved detection logic with proper confidence thresholding"""
    
    if len(output_data.shape) == 2:
        # Classification model - need minimum confidence
        class_probs = output_data[0]
        max_idx = np.argmax(class_probs)
        max_prob = class_probs[max_idx]
        
        if max_prob < confidence_threshold:
            return "No dice detected (low confidence)"
        
        predicted_dice = (max_idx % 6) + 1
        return f"Dice: {predicted_dice} (conf: {max_prob:.3f})"
        
    elif len(output_data.shape) == 3:
        # Object detection model
        detections = []
        
        for detection in output_data[0]:
            if len(detection) >= 6:
                confidence = detection[4]
                if confidence > confidence_threshold:
                    class_idx = int(detection[5])
                    dice_value = (class_idx % 6) + 1
                    detections.append(f"Dice: {dice_value} ({confidence:.3f})")
        
        if detections:
            return " | ".join(detections)
        else:
            return "No dice detected"
    
    return "Unknown model format"

def main():
    """Main diagnostic function"""
    print("🔍 Debug False Detection - Model Output Analysis")
    print("=" * 60)
    
    # Load model
    model_data = load_model()
    if not model_data:
        return
    
    # Initialize camera
    print(f"\n📸 Initializing Camera...")
    try:
        picam2 = Picamera2()
        
        config = picam2.create_still_configuration(
            main={"size": (640, 640)},
            display="main"
        )
        picam2.configure(config)
        picam2.start()
        
        print(f"✅ Camera ready!")
        print(f"🎯 Take images with NO DICE to debug false positives")
        print(f"📋 This script will show detailed model output analysis")
        
        while True:
            input("\n📸 Press Enter to capture and analyze (Ctrl+C to exit): ")
            
            # Capture image
            image = picam2.capture_array()
            
            # Run inference with debug
            output = run_inference_with_debug(model_data, image)
            
            if output is not None:
                # Detailed analysis
                predicted_dice, confidence = analyze_output_in_detail(output)
                
                # Test different confidence thresholds
                print(f"\n🎯 RECOMMENDED DETECTION LOGIC:")
                for threshold in [0.3, 0.5, 0.7]:
                    result = improved_detection_with_threshold(output, threshold)
                    print(f"   Threshold {threshold}: {result}")
                
                print("=" * 60)
            else:
                print("❌ Inference failed")
            
    except KeyboardInterrupt:
        print("\n👋 Stopping analysis...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        try:
            picam2.stop()
        except:
            pass

if __name__ == "__main__":
    main() 