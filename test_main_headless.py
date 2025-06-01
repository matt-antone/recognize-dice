#!/usr/bin/env python3
"""
Headless test for D6 Dice Recognition
Tests camera and detection without GUI for SSH/headless testing
"""

import sys
import os
import time
import cv2

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.camera.camera_interface import CameraInterface
    from src.detection.dice_detector import DiceDetector
    from src.utils.config import Config
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install dependencies: python3 install_deps.py")
    sys.exit(1)


def test_headless():
    """Test camera and detection without GUI."""
    print("D6 Dice Recognition - Headless Test")
    print("=" * 40)
    
    # Set up logging
    logger = setup_logger(__name__)
    
    # Create config
    config = Config()
    
    try:
        # Initialize camera
        print("Initializing AI camera...")
        camera = CameraInterface(config)
        
        # Initialize detector
        print("Initializing dice detector...")
        detector = DiceDetector(config)
        
        # Start camera
        print("Starting camera...")
        camera.start()
        
        print("\nCapturing and processing frames...")
        print("(Press Ctrl+C to stop)")
        
        frame_count = 0
        
        while frame_count < 10:  # Capture 10 frames for testing
            # Capture frame
            frame = camera.capture_frame()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Process frame
            start_time = time.time()
            detections = detector.detect_dice(frame)
            processing_time = time.time() - start_time
            
            # Display results
            print(f"\nFrame {frame_count}:")
            print(f"  Processing time: {processing_time*1000:.1f}ms")
            print(f"  Detections found: {len(detections)}")
            
            for i, detection in enumerate(detections):
                print(f"    Dice {i+1}: Value={detection.value}, "
                      f"Confidence={detection.confidence:.2f}, "
                      f"Position=({detection.center_x}, {detection.center_y})")
            
            # Save frame for inspection
            if frame_count == 1:
                # Save first frame with detections
                output_frame = frame.copy()
                for detection in detections:
                    x1, y1, x2, y2 = detection.bbox
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"{detection.value} ({detection.confidence:.2f})"
                    cv2.putText(output_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imwrite("headless_test_result.jpg", cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
                print(f"  Saved test image: headless_test_result.jpg")
            
            time.sleep(0.5)  # Small delay
        
        # Cleanup
        camera.stop()
        camera.cleanup()
        detector.cleanup()
        
        print(f"\n✅ Headless test completed successfully!")
        print(f"Processed {frame_count} frames")
        print("The app is ready for GUI testing when display is available.")
        
        return True
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Headless test error: {e}")
        return False


if __name__ == "__main__":
    success = test_headless()
    sys.exit(0 if success else 1) 