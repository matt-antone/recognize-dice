#!/usr/bin/env python3
"""
Quick test of the improved detection algorithm based on debug analysis.
"""

import sys
import os
import cv2
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.camera.camera_interface import CameraInterface
    from src.detection.fallback_detection import FallbackDetection
    from src.utils.config import Config
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_improved_detection():
    """Test the improved detection algorithm."""
    print("Testing Improved Dice Detection")
    print("=" * 40)
    
    logger = setup_logger(__name__)
    config = Config()
    
    try:
        # Initialize camera and detection
        print("Initializing AI camera...")
        camera = CameraInterface(config)
        camera.start()
        
        print("Initializing improved detection...")
        detector = FallbackDetection(config)
        
        print("\nPress Enter to test detection on current frame...")
        input("Position dice and press Enter...")
        
        # Capture and analyze frame
        frame = camera.capture_frame()
        if frame is None:
            print("❌ Failed to capture frame")
            return
        
        print(f"Captured frame: {frame.shape}")
        
        # Run improved detection
        detections = detector.detect_dice(frame)
        
        print(f"\n🎲 Detection Results:")
        print(f"Found {len(detections)} dice")
        
        for i, dice in enumerate(detections):
            print(f"  Dice {i+1}:")
            print(f"    Method: {dice['method']}")
            print(f"    Value: {dice['value']}")
            print(f"    Confidence: {dice['confidence']:.2f}")
            print(f"    Center: {dice['center']}")
            print(f"    Area: {dice['area']:.0f}")
        
        # Test multiple frames
        print(f"\nTesting detection stability...")
        for test_frame in range(3):
            frame = camera.capture_frame()
            if frame is not None:
                detections = detector.detect_dice(frame)
                print(f"Frame {test_frame + 1}: {len(detections)} dice detected")
            else:
                print(f"Frame {test_frame + 1}: Failed to capture")
        
        camera.stop()
        camera.cleanup()
        
        print(f"\n✅ Improved detection test complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}")


if __name__ == "__main__":
    test_improved_detection() 