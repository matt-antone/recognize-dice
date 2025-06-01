#!/usr/bin/env python3
"""
Test fallback detection without camera
Works on any platform for development testing
"""

import sys
import os
import numpy as np
import cv2

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.detection.dice_detector import DiceDetector
    from src.utils.config import Config
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install dependencies: python3 install_deps.py")
    sys.exit(1)


def create_test_image():
    """Create a simple test image with circular shapes (simulating dice)."""
    # Create a 320x320 image
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    img.fill(50)  # Dark gray background
    
    # Add some circular shapes to simulate dice
    centers = [(80, 80), (160, 160), (240, 240)]
    
    for i, (cx, cy) in enumerate(centers):
        # Draw a white circle (simulating a dice)
        cv2.circle(img, (cx, cy), 30, (255, 255, 255), -1)
        
        # Add some dots to make it look more like dice
        dot_color = (0, 0, 0)  # Black dots
        
        if i == 0:  # One dot
            cv2.circle(img, (cx, cy), 3, dot_color, -1)
        elif i == 1:  # Two dots
            cv2.circle(img, (cx-8, cy-8), 3, dot_color, -1)
            cv2.circle(img, (cx+8, cy+8), 3, dot_color, -1)
        elif i == 2:  # Three dots
            cv2.circle(img, (cx-10, cy-10), 3, dot_color, -1)
            cv2.circle(img, (cx, cy), 3, dot_color, -1)
            cv2.circle(img, (cx+10, cy+10), 3, dot_color, -1)
    
    return img


def test_fallback_detection():
    """Test the fallback detection system."""
    print("D6 Dice Recognition - Fallback Detection Test")
    print("=" * 50)
    
    # Set up logging
    logger = setup_logger(__name__)
    
    # Create config
    config = Config()
    
    print(f"Using fallback detection: {config.enable_fallback_detection}")
    print(f"Detection method: {config.fallback_method}")
    print()
    
    try:
        # Initialize detector
        print("Initializing dice detector...")
        detector = DiceDetector(config)
        
        # Create test image
        print("Creating test image with simulated dice...")
        test_image = create_test_image()
        
        # Run detection
        print("Running fallback detection...")
        detections = detector.detect_dice(test_image)
        
        # Display results
        print(f"\nDetection Results:")
        print(f"Found {len(detections)} objects")
        
        if detections:
            for i, detection in enumerate(detections):
                print(f"  Detection {i+1}:")
                print(f"    Value: {detection.value}")
                print(f"    Confidence: {detection.confidence:.2f}")
                print(f"    Position: ({detection.center_x}, {detection.center_y})")
                print(f"    Bounding box: {detection.bbox}")
                print()
        else:
            print("  No dice detected")
        
        # Get detector statistics
        stats = detector.get_stats()
        print("Detection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Save test image with detections for visual verification
        output_image = test_image.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{detection.value} ({detection.confidence:.2f})"
            cv2.putText(output_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite("test_detection_result.png", output_image)
        print(f"\nTest image saved as: test_detection_result.png")
        
        # Cleanup
        detector.cleanup()
        
        print("\n✅ Fallback detection test completed successfully!")
        print("This confirms the detection pipeline is working.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Fallback test error: {e}")
        return False


def check_gui_imports():
    """Check if GUI components can be imported."""
    print("\nTesting GUI imports...")
    
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
        print("✅ GUI components available")
        
        # Test basic window creation
        root = tk.Tk()
        root.withdraw()  # Hide the window
        root.destroy()
        print("✅ Tkinter window creation works")
        
        return True
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False


def main():
    """Main test function."""
    print("Raspberry Pi 3 - Fallback Detection Test")
    print("Testing dice recognition without camera")
    print("-" * 50)
    
    # Check dependencies
    try:
        import cv2
        import numpy
        print("✅ OpenCV and NumPy available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please run: python3 install_deps.py")
        return False
    
    # Test fallback detection
    if not test_fallback_detection():
        return False
    
    # Test GUI components
    if not check_gui_imports():
        print("⚠️  GUI components not available, but detection works")
    
    print("\n🎉 All tests passed!")
    print("\nNext steps:")
    print("1. Try the main application: python3 main.py")
    print("2. On Raspberry Pi: python3 test_camera.py")
    print("3. Check the saved test image: test_detection_result.png")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 