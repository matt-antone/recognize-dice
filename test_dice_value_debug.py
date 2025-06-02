#!/usr/bin/env python3
"""
Debug dice value estimation to understand why it's reporting wrong values.
"""

import sys
import os
import cv2
import numpy as np
from datetime import datetime

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


def debug_dice_value_estimation(dice_region: np.ndarray, actual_value: int, debug_dir: str):
    """Debug the dice value estimation process step by step."""
    print(f"\n=== Debugging Dice Value Estimation ===")
    print(f"Actual dice value: {actual_value}")
    print(f"Dice region shape: {dice_region.shape}")
    
    if dice_region.size == 0:
        print("❌ Empty dice region!")
        return 1
    
    # Save original dice region
    cv2.imwrite(f"{debug_dir}/dice_region_original.jpg", dice_region)
    
    # Step 1: Analyze brightness
    avg_brightness = np.mean(dice_region)
    brightness_std = np.std(dice_region)
    print(f"Brightness: mean={avg_brightness:.1f}, std={brightness_std:.1f}")
    
    # Step 2: Apply threshold
    _, binary = cv2.threshold(dice_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{debug_dir}/dice_binary.jpg", binary)
    
    # Step 3: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"{debug_dir}/dice_cleaned.jpg", cleaned)
    
    # Step 4: Find potential dot contours
    dot_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debug image showing all contours
    debug_img = cv2.cvtColor(dice_region, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_img, dot_contours, -1, (0, 255, 0), 1)
    cv2.imwrite(f"{debug_dir}/dice_all_contours.jpg", debug_img)
    
    print(f"Found {len(dot_contours)} total contours")
    
    # Step 5: Analyze each contour
    dot_count = 0
    min_dot_area = 10
    max_dot_area = dice_region.size // 10
    
    valid_dots_img = cv2.cvtColor(dice_region, cv2.COLOR_GRAY2BGR)
    
    for i, contour in enumerate(dot_contours):
        area = cv2.contourArea(contour)
        print(f"  Contour {i}: area={area:.1f}", end="")
        
        if min_dot_area < area < max_dot_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                print(f", circularity={circularity:.2f}", end="")
                
                if circularity > 0.4:
                    dot_count += 1
                    # Draw valid dots in red
                    cv2.drawContours(valid_dots_img, [contour], -1, (0, 0, 255), 2)
                    print(" ✅ VALID DOT")
                else:
                    print(" ❌ not circular enough")
            else:
                print(" ❌ no perimeter")
        else:
            print(" ❌ wrong size")
    
    cv2.imwrite(f"{debug_dir}/dice_valid_dots.jpg", valid_dots_img)
    print(f"Valid dots counted: {dot_count}")
    
    # Step 6: Try inverted version
    binary_inv = cv2.bitwise_not(binary)
    cleaned_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"{debug_dir}/dice_inverted.jpg", binary_inv)
    cv2.imwrite(f"{debug_dir}/dice_cleaned_inv.jpg", cleaned_inv)
    
    dot_contours_inv, _ = cv2.findContours(cleaned_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dot_count_inv = 0
    valid_dots_inv_img = cv2.cvtColor(dice_region, cv2.COLOR_GRAY2BGR)
    
    print(f"Inverted: Found {len(dot_contours_inv)} contours")
    for i, contour in enumerate(dot_contours_inv):
        area = cv2.contourArea(contour)
        if min_dot_area < area < max_dot_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.4:
                    dot_count_inv += 1
                    cv2.drawContours(valid_dots_inv_img, [contour], -1, (255, 0, 0), 2)
    
    cv2.imwrite(f"{debug_dir}/dice_valid_dots_inv.jpg", valid_dots_inv_img)
    print(f"Valid inverted dots counted: {dot_count_inv}")
    
    # Step 7: Decision logic
    final_count = dot_count if 1 <= dot_count <= 6 else dot_count_inv
    print(f"Chosen dot count: {final_count}")
    
    if not (1 <= final_count <= 6):
        print("Dot counting failed, using brightness fallback")
        if avg_brightness < 60:
            final_count = 5
        elif avg_brightness < 80:
            final_count = 4
        elif avg_brightness < 100:
            final_count = 3
        elif avg_brightness < 120:
            final_count = 2
        else:
            final_count = 1
        print(f"Brightness fallback result: {final_count}")
    
    estimated_value = max(1, min(6, final_count))
    print(f"Final estimated value: {estimated_value}")
    print(f"Actual value: {actual_value}")
    print(f"Accuracy: {'✅ CORRECT' if estimated_value == actual_value else '❌ WRONG'}")
    
    return estimated_value


def test_dice_value_debug():
    """Debug dice value estimation with user input."""
    print("Dice Value Estimation Debug")
    print("=" * 40)
    
    # Create debug directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = f"dice_value_debug_{timestamp}"
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Debug files will be saved to: {debug_dir}/")
    
    logger = setup_logger(__name__)
    config = Config()
    
    try:
        # Initialize camera and detection
        camera = CameraInterface(config)
        camera.start()
        detector = FallbackDetection(config)
        
        print("\nPlace a dice in view and tell me its actual value...")
        actual_value = int(input("What value is the dice showing (1-6)? "))
        
        if not (1 <= actual_value <= 6):
            print("Invalid value! Must be 1-6")
            return
        
        # Capture frame and detect dice
        frame = camera.capture_frame()
        if frame is None:
            print("❌ Failed to capture frame")
            return
        
        detections = detector.detect_dice(frame)
        
        if len(detections) == 0:
            print("❌ No dice detected!")
            return
        
        if len(detections) > 1:
            print(f"⚠️ Multiple dice detected ({len(detections)}), using first one")
        
        # Get the first detected dice region
        dice = detections[0]
        x, y, w, h = dice['bbox']
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        dice_region = gray[y:y+h, x:x+w]
        
        print(f"Detected dice at ({dice['center'][0]}, {dice['center'][1]}) using {dice['method']}")
        print(f"Algorithm estimated value: {dice['value']}")
        
        # Debug the value estimation
        debug_dice_value_estimation(dice_region, actual_value, debug_dir)
        
        camera.stop()
        camera.cleanup()
        
        print(f"\n✅ Debug complete! Check {debug_dir}/ for detailed analysis")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        logger.error(f"Debug error: {e}")


if __name__ == "__main__":
    test_dice_value_debug() 