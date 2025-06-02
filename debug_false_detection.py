#!/usr/bin/env python3
"""
Debug False Detection - Understanding Model Output
Diagnostic script to analyze why dice are detected when none are present
"""

import time
import numpy as np
import cv2
from pathlib import Path
import datetime

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
    return np.expand_dims(processed, axis=0), resized

def run_inference_with_debug(model_data, image):
    """Run inference and provide detailed output analysis"""
    try:
        interpreter = model_data['interpreter']
        input_details = model_data['input_details']
        output_details = model_data['output_details']
        
        # Preprocess image and get the model input version
        input_data, model_input_image = preprocess_image(image, input_details)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        return output_data, model_input_image
        
    except Exception as e:
        print(f"⚠️ Inference error: {e}")
        return None, None

def save_debug_images(original_image, model_input_image, confidence, predicted_dice, actual_dice_count):
    """Save images for debugging analysis"""
    # Create debug directory if it doesn't exist
    debug_dir = Path("debug_images")
    debug_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save original camera image
    original_path = debug_dir / f"original_{timestamp}_detected{predicted_dice}_actual{actual_dice_count}_conf{confidence:.3f}.jpg"
    cv2.imwrite(str(original_path), original_image)
    
    # Save model input image (what the model actually sees)
    if model_input_image is not None:
        # Convert back to uint8 for saving
        if model_input_image.dtype == np.float32:
            model_save = (model_input_image * 255).astype(np.uint8)
        else:
            model_save = model_input_image
        
        model_path = debug_dir / f"model_input_{timestamp}_detected{predicted_dice}_actual{actual_dice_count}_conf{confidence:.3f}.jpg"
        cv2.imwrite(str(model_path), model_save)
    
    print(f"💾 Debug images saved:")
    print(f"   📸 Original: {original_path}")
    print(f"   🤖 Model input: {model_path}")
    
    return original_path, model_path

def get_actual_dice_count():
    """Ask user how many dice are actually in the frame"""
    while True:
        try:
            user_input = input("🎲 How many dice are actually in the camera view? (0-6 or 'q' to quit): ").strip().lower()
            
            if user_input == 'q':
                return None
            
            dice_count = int(user_input)
            if 0 <= dice_count <= 6:
                return dice_count
            else:
                print("⚠️ Please enter a number between 0 and 6")
                
        except ValueError:
            print("⚠️ Please enter a valid number (0-6) or 'q' to quit")

def analyze_detection_accuracy(predicted_dice, confidence, actual_dice_count):
    """Analyze the accuracy of the detection"""
    print(f"\n📊 DETECTION ACCURACY ANALYSIS:")
    print(f"   🤖 Model detected: Dice {predicted_dice} (conf: {confidence:.3f})")
    print(f"   👁️ Actually in frame: {actual_dice_count} dice")
    
    if actual_dice_count == 0:
        if confidence > 0.5:
            accuracy = "❌ FALSE POSITIVE (High confidence)"
        else:
            accuracy = "⚠️ FALSE POSITIVE (Low confidence)"
    elif actual_dice_count == 1:
        accuracy = "🔍 SINGLE DICE - Check if value correct"
    else:
        accuracy = f"🔍 MULTIPLE DICE ({actual_dice_count}) - Model only detected 1"
    
    print(f"   📈 Result: {accuracy}")
    
    return accuracy

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
        
        # Find highest probability
        class_probs = output_data[0]
        max_idx = np.argmax(class_probs)
        max_prob = class_probs[max_idx]
        predicted_dice = (max_idx % 6) + 1
        
        print(f"\n🎯 HIGHEST PROBABILITY:")
        print(f"   Class {max_idx} (Dice {predicted_dice}): {max_prob:.6f}")
        
        # Show top 5 detections for analysis (reduced from 10 for cleaner output)
        print(f"\n🔝 TOP 5 DETECTIONS:")
        top_indices = np.argsort(class_probs)[-5:][::-1]  # Top 5 in descending order
        for i, idx in enumerate(top_indices):
            dice_value = (idx % 6) + 1
            prob = class_probs[idx]
            print(f"   {i+1}. Class {idx} (Dice {dice_value}): {prob:.6f}")
        
        # Apply different confidence thresholds
        print(f"\n🚪 CONFIDENCE THRESHOLDS:")
        thresholds = [0.3, 0.5, 0.7, 0.9]
        for thresh in thresholds:
            if max_prob > thresh:
                result = f"Dice {predicted_dice} (conf: {max_prob:.3f})"
            else:
                result = "No dice detected"
            print(f"   Threshold {thresh}: {result}")
        
        return predicted_dice, max_prob
        
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
    print("🔍 Debug False Detection - Model vs Reality Analysis")
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
        print(f"🎯 This script compares what the model detects vs reality")
        print(f"📋 You'll tell us how many dice are actually visible")
        print(f"💾 Images will be saved to debug_images/ folder")
        
        while True:
            try:
                input("\n📸 Press Enter to capture and analyze (Ctrl+C to exit): ")
                
                # Get actual dice count from user
                actual_dice_count = get_actual_dice_count()
                if actual_dice_count is None:
                    break  # User chose to quit
                
                # Capture image
                image = picam2.capture_array()
                
                # Run inference with debug
                output, model_input_image = run_inference_with_debug(model_data, image)
                
                if output is not None:
                    # Detailed analysis
                    predicted_dice, confidence = analyze_output_in_detail(output)
                    
                    # Analyze accuracy
                    accuracy = analyze_detection_accuracy(predicted_dice, confidence, actual_dice_count)
                    
                    # Save debug images with both detected and actual info
                    save_debug_images(image, model_input_image, confidence, predicted_dice, actual_dice_count)
                    
                    # Test different confidence thresholds
                    print(f"\n🎯 RECOMMENDED DETECTION LOGIC:")
                    for threshold in [0.3, 0.5, 0.7]:
                        result = improved_detection_with_threshold(output, threshold)
                        print(f"   Threshold {threshold}: {result}")
                    
                    print("=" * 60)
                    print(f"💾 Check debug_images/ folder for saved images")
                else:
                    print("❌ Inference failed")
                    
            except KeyboardInterrupt:
                print("\n👋 Stopping analysis...")
                break
            
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