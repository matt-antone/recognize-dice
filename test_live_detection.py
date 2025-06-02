#!/usr/bin/env python3
"""
Live dice detection preview with real-time visualization.
Shows detected dice with bounding boxes, values, and debug info.
"""

import sys
import os
import cv2
import numpy as np
import time

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


def draw_detection_overlay(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw detection results on the frame."""
    overlay_frame = frame.copy()
    
    # Convert RGB to BGR for OpenCV display
    if len(overlay_frame.shape) == 3 and overlay_frame.shape[2] == 3:
        overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)
    
    # Draw each detection
    for i, detection in enumerate(detections):
        x, y, w, h = detection['bbox']
        center = detection['center']
        value = detection['value']
        confidence = detection['confidence']
        method = detection['method']
        
        # Draw bounding box
        color = (0, 255, 0) if method == 'contour' else (255, 0, 0)  # Green for contour, Blue for blob
        cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw center point
        cv2.circle(overlay_frame, center, 3, (0, 0, 255), -1)
        
        # Draw value and info
        text = f"Dice {i+1}: {value}"
        cv2.putText(overlay_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw method and confidence
        info_text = f"{method} ({confidence:.2f})"
        cv2.putText(overlay_frame, info_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw frame info
    info_y = 30
    cv2.putText(overlay_frame, f"Detected: {len(detections)} dice", (10, info_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if len(detections) > 0:
        total_value = sum(d['value'] for d in detections)
        cv2.putText(overlay_frame, f"Total: {total_value}", (10, info_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw instructions
    cv2.putText(overlay_frame, "Press 'q' to quit, 's' to save screenshot", (10, overlay_frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return overlay_frame


def test_live_detection():
    """Run live dice detection with real-time preview."""
    print("Live Dice Detection Preview")
    print("=" * 40)
    print("- Shows real-time detection with overlays")
    print("- Green boxes: Contour detection")
    print("- Blue boxes: Blob detection") 
    print("- Press 'q' to quit")
    print("- Press 's' to save screenshot")
    print("- Press 'd' to toggle debug output")
    
    logger = setup_logger(__name__)
    config = Config()
    
    try:
        # Initialize camera and detection
        print("\nInitializing AI camera...")
        camera = CameraInterface(config)
        camera.start()
        
        print("Initializing detection...")
        detector = FallbackDetection(config)
        
        print("\nStarting live preview...")
        print("(Detection window should open)")
        
        frame_count = 0
        fps_start_time = time.time()
        show_debug = False
        
        while True:
            # Capture frame
            frame = camera.capture_frame()
            if frame is None:
                print("Failed to capture frame")
                break
            
            # Run detection
            detections = detector.detect_dice(frame)
            
            # Print debug info if enabled
            if show_debug:
                print(f"Frame {frame_count}: {len(detections)} dice detected")
                for i, d in enumerate(detections):
                    print(f"  Dice {i+1}: Value={d['value']}, Method={d['method']}, Confidence={d['confidence']:.2f}")
            
            # Draw overlay
            display_frame = draw_detection_overlay(frame, detections)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Live Dice Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"dice_detection_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('d'):
                # Toggle debug output
                show_debug = not show_debug
                print(f"Debug output: {'ON' if show_debug else 'OFF'}")
        
        # Cleanup
        cv2.destroyAllWindows()
        camera.stop()
        camera.cleanup()
        
        print("\n✅ Live detection ended")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        cv2.destroyAllWindows()
        if 'camera' in locals():
            camera.stop()
            camera.cleanup()
    except Exception as e:
        print(f"❌ Live detection failed: {e}")
        logger.error(f"Live detection error: {e}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_live_detection() 