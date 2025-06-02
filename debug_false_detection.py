#!/usr/bin/env python3
"""
Debug False Detection - Understanding Model Output
Diagnostic script to analyze why dice are detected when none are present
NOW WITH PREVIEW - See what the camera sees!
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

def save_debug_images(original_image, model_input_image, confidence, predicted_dice):
    """Save images for debugging analysis"""
    # Create debug directory if it doesn't exist
    debug_dir = Path("debug_images")
    debug_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save original camera image
    original_path = debug_dir / f"original_{timestamp}_dice{predicted_dice}_conf{confidence:.3f}.jpg"
    cv2.imwrite(str(original_path), original_image)
    
    # Save model input image (what the model actually sees)
    if model_input_image is not None:
        # Convert back to uint8 for saving
        if model_input_image.dtype == np.float32:
            model_save = (model_input_image * 255).astype(np.uint8)
        else:
            model_save = model_input_image
        
        model_path = debug_dir / f"model_input_{timestamp}_dice{predicted_dice}_conf{confidence:.3f}.jpg"
        cv2.imwrite(str(model_path), model_save)
    
    print(f"💾 Debug images saved:")
    print(f"   📸 Original: {original_path}")
    print(f"   🤖 Model input: {model_path}")
    
    return original_path, model_path

def show_preview_and_analysis(original_image, model_input_image, confidence, predicted_dice):
    """Show preview windows and analysis"""
    # Create preview windows
    cv2.namedWindow("Camera View (Original)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Model Input (160x160)", cv2.WINDOW_NORMAL)
    
    # Resize windows for better viewing
    cv2.resizeWindow("Camera View (Original)", 640, 480)
    cv2.resizeWindow("Model Input (160x160)", 320, 320)
    
    # Show original image
    display_original = original_image.copy()
    
    # Add detection info overlay
    text = f"DETECTED: Dice {predicted_dice} (conf: {confidence:.3f})"
    color = (0, 0, 255) if confidence > 0.5 else (0, 255, 255)  # Red if confident, yellow if not
    cv2.putText(display_original, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(display_original, "Press any key to continue", (10, display_original.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Camera View (Original)", display_original)
    
    # Show model input (what the AI actually analyzes)
    if model_input_image is not None:
        # Convert back to uint8 for display if needed
        if model_input_image.dtype == np.float32:
            display_model = (model_input_image * 255).astype(np.uint8)
        else:
            display_model = model_input_image
            
        # Add detection info to model input view
        cv2.putText(display_model, f"Dice {predicted_dice}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_model, f"{confidence:.3f}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Model Input (160x160)", display_model)
    
    # Wait for key press
    print(f"👁️ PREVIEW: Check the camera view windows!")
    print(f"   🎯 What do you see that might look like a dice?")
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()

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
        
        # Show top 10 detections for analysis
        print(f"\n🔝 TOP 10 DETECTIONS:")
        top_indices = np.argsort(class_probs)[-10:][::-1]  # Top 10 in descending order
        for i, idx in enumerate(top_indices):
            dice_value = (idx % 6) + 1
            prob = class_probs[idx]
            print(f"   {i+1}. Class {idx} (Dice {dice_value}): {prob:.6f}")
        
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
    print("🔍 Debug False Detection - Model Output Analysis WITH PREVIEW")
    print("=" * 70)
    
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
        print(f"👁️ PREVIEW MODE: You'll see what the camera sees!")
        print(f"🎯 Take images with NO DICE to debug false positives")
        print(f"📋 This script will show detailed model output analysis")
        print(f"💾 Images will be saved to debug_images/ folder")
        
        while True:
            input("\n📸 Press Enter to capture and analyze (Ctrl+C to exit): ")
            
            # Capture image
            image = picam2.capture_array()
            
            # Run inference with debug
            output, model_input_image = run_inference_with_debug(model_data, image)
            
            if output is not None:
                # Detailed analysis
                predicted_dice, confidence = analyze_output_in_detail(output)
                
                # Save debug images
                save_debug_images(image, model_input_image, confidence, predicted_dice)
                
                # Show preview and analysis
                show_preview_and_analysis(image, model_input_image, confidence, predicted_dice)
                
                # Test different confidence thresholds
                print(f"\n🎯 RECOMMENDED DETECTION LOGIC:")
                for threshold in [0.3, 0.5, 0.7]:
                    result = improved_detection_with_threshold(output, threshold)
                    print(f"   Threshold {threshold}: {result}")
                
                print("=" * 70)
                print(f"💡 TIP: Check the preview windows to see what the camera detected!")
                print(f"💾 Check debug_images/ folder for saved images")
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
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 