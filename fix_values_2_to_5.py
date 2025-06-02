#!/usr/bin/env python3
"""
Fix the values 2-5 detection issue.
Currently only 2.0% of detections are values 2-5 - this is severely under-represented.
"""

import sys
import os
import cv2
import numpy as np
import json
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.detection.fallback_detection import FallbackDetection
    from src.utils.config import Config
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def analyze_values_2_to_5_problem():
    """Analyze why values 2-5 are severely under-detected (only 2.0%)."""
    print("🔍 ANALYZING VALUES 2-5 DETECTION PROBLEM")
    print("=" * 50)
    
    # Load Kaggle results to understand the issue
    results_file = Path("kaggle_dataset_results_20250601_194236.json")
    if not results_file.exists():
        print("❌ No results file found. Run test_kaggle_dataset.py first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Analyze the value distribution problem
    total_detections = results['summary']['total_dice_detected']
    value_dist = results['summary']['dice_value_distribution']
    
    print(f"📊 CURRENT VALUE DISTRIBUTION (Total: {total_detections}):")
    for value, count in value_dist.items():
        percentage = (count / total_detections) * 100
        print(f"  Value {value}: {count} ({percentage:.1f}%)")
    
    # The problem is clear: we're over-detecting 6s and 1s, under-detecting 2-5
    values_2_5_total = sum(value_dist.get(str(i), 0) for i in range(2, 6))
    values_2_5_percent = (values_2_5_total / total_detections) * 100
    
    print(f"\n🚨 PROBLEM IDENTIFIED:")
    print(f"  Values 2-5: {values_2_5_total} ({values_2_5_percent:.1f}%) - SEVERELY LOW")
    print(f"  Value 6: {value_dist.get('6', 0)} ({(value_dist.get('6', 0) / total_detections) * 100:.1f}%) - OVER-DETECTED")
    print(f"  Value 1: {value_dist.get('1', 0)} ({(value_dist.get('1', 0) / total_detections) * 100:.1f}%) - OVER-DETECTED")
    
    print(f"\n🎯 ROOT CAUSE HYPOTHESIS:")
    print("  1. Value 6 fix is too aggressive - capturing values 2-5 as 6")
    print("  2. Value 1 detection is still over-triggering")
    print("  3. Values 2-5 patterns are not being recognized")
    print("  4. Need better discrimination between similar pip counts")
    
    return True


def test_enhanced_value_estimation():
    """Test improved value estimation for values 2-5."""
    print("\n🔧 TESTING ENHANCED VALUE 2-5 ESTIMATION")
    print("=" * 50)
    
    config = Config()
    detector = FallbackDetection(config)
    
    # Create test patterns for values 2-5
    test_patterns = create_test_patterns_2_to_5()
    
    for value, pattern in test_patterns.items():
        print(f"\nTesting value {value} pattern:")
        
        # Test current estimation
        detected_value = detector._estimate_dice_value(pattern)
        
        # Test enhanced estimation
        enhanced_value = enhanced_value_estimation(pattern, detector)
        
        print(f"  Current: {detected_value}")
        print(f"  Enhanced: {enhanced_value}")
        
        if enhanced_value == value:
            print(f"  ✅ CORRECT: Enhanced detection works!")
        elif detected_value == value:
            print(f"  ➖ Current works, enhanced doesn't help")
        else:
            print(f"  ❌ BOTH WRONG: Expected {value}")


def enhanced_value_estimation(dice_region: np.ndarray, detector) -> int:
    """Enhanced value estimation with better 2-5 discrimination."""
    if dice_region.size == 0:
        return 1
    
    # ENHANCED: Better discrimination for values 2-5
    
    # Step 1: Get all potential dots
    dots = detector._find_all_circular_dots(dice_region)
    dot_count = len(dots)
    
    # Step 2: Pattern analysis for specific values
    h, w = dice_region.shape
    
    # Value 2: Look for diagonal or opposite corner pattern
    if dot_count == 2:
        if is_diagonal_pattern(dots, w, h):
            return 2
    
    # Value 3: Look for diagonal line pattern
    if dot_count == 3:
        if is_diagonal_line_pattern(dots, w, h):
            return 3
    
    # Value 4: Look for 2x2 corner pattern
    if dot_count == 4:
        if is_four_corners_pattern(dots, w, h):
            return 4
    
    # Value 5: Look for 4 corners + center pattern
    if dot_count == 5:
        if is_five_pattern(dots, w, h):
            return 5
    
    # Value 6: Use existing detection (already working)
    if dot_count == 6:
        if detector._is_3x2_grid_pattern(dots, w, h):
            return 6
    
    # Value 1: Look for single center pip
    if dot_count == 1:
        if is_center_pattern(dots, w, h):
            return 1
    
    # Fallback: Conservative estimation based on dot count
    # But add some logic to prevent 6-bias
    if dot_count <= 1:
        return 1
    elif dot_count == 2:
        return 2
    elif dot_count == 3:
        return 3
    elif dot_count == 4:
        return 4
    elif dot_count == 5:
        return 5
    else:
        return 6


def is_diagonal_pattern(dots: list, width: int, height: int) -> bool:
    """Check if 2 dots form a diagonal pattern (value 2)."""
    if len(dots) != 2:
        return False
    
    centers = [dot['center'] for dot in dots]
    p1, p2 = centers
    
    # Check if dots are in opposite corners (diagonal)
    diagonal_threshold = min(width, height) * 0.3
    
    # Main diagonal or anti-diagonal
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    
    return dx > diagonal_threshold and dy > diagonal_threshold


def is_diagonal_line_pattern(dots: list, width: int, height: int) -> bool:
    """Check if 3 dots form a diagonal line (value 3)."""
    if len(dots) != 3:
        return False
    
    centers = [dot['center'] for dot in dots]
    
    # Sort by one coordinate to check alignment
    centers_by_x = sorted(centers, key=lambda p: p[0])
    
    # Check if they form a roughly diagonal line
    p1, p2, p3 = centers_by_x
    
    # Calculate if points are roughly collinear
    # Using cross product to check linearity
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p1[0], p3[1] - p1[1])
    
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    threshold = width * height * 0.01  # Allow some deviation
    
    return abs(cross_product) < threshold


def is_four_corners_pattern(dots: list, width: int, height: int) -> bool:
    """Check if 4 dots form a square pattern (value 4)."""
    if len(dots) != 4:
        return False
    
    centers = [dot['center'] for dot in dots]
    
    # Check if dots are roughly in the 4 corners
    corner_positions = [
        (width * 0.25, height * 0.25),  # Top-left
        (width * 0.75, height * 0.25),  # Top-right
        (width * 0.25, height * 0.75),  # Bottom-left
        (width * 0.75, height * 0.75)   # Bottom-right
    ]
    
    # Match actual dots to expected corner positions
    matched_corners = 0
    threshold = min(width, height) * 0.3
    
    for corner in corner_positions:
        for center in centers:
            dx = abs(center[0] - corner[0])
            dy = abs(center[1] - corner[1])
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < threshold:
                matched_corners += 1
                break
    
    return matched_corners >= 3  # At least 3 of 4 corners matched


def is_five_pattern(dots: list, width: int, height: int) -> bool:
    """Check if 5 dots form the value 5 pattern (4 corners + center)."""
    if len(dots) != 5:
        return False
    
    centers = [dot['center'] for dot in dots]
    
    # Look for center dot
    center_x, center_y = width // 2, height // 2
    center_threshold = min(width, height) * 0.2
    
    center_dots = 0
    corner_dots = 0
    
    for center in centers:
        # Check if it's near the center
        dx = abs(center[0] - center_x)
        dy = abs(center[1] - center_y)
        
        if dx < center_threshold and dy < center_threshold:
            center_dots += 1
        else:
            # Check if it's in a corner region
            corner_threshold = min(width, height) * 0.3
            
            if ((center[0] < corner_threshold or center[0] > width - corner_threshold) and
                (center[1] < corner_threshold or center[1] > height - corner_threshold)):
                corner_dots += 1
    
    # Value 5 should have 1 center dot and 4 corner dots
    return center_dots >= 1 and corner_dots >= 3


def is_center_pattern(dots: list, width: int, height: int) -> bool:
    """Check if single dot is in center (value 1)."""
    if len(dots) != 1:
        return False
    
    center = dots[0]['center']
    center_x, center_y = width // 2, height // 2
    
    threshold = min(width, height) * 0.3
    dx = abs(center[0] - center_x)
    dy = abs(center[1] - center_y)
    
    return dx < threshold and dy < threshold


def create_test_patterns_2_to_5() -> dict:
    """Create synthetic test patterns for values 2-5."""
    patterns = {}
    
    # Value 2: Diagonal pattern
    pattern2 = np.ones((60, 60), dtype=np.uint8) * 200
    cv2.circle(pattern2, (15, 15), 5, 50, -1)  # Top-left
    cv2.circle(pattern2, (45, 45), 5, 50, -1)  # Bottom-right
    patterns[2] = pattern2
    
    # Value 3: Diagonal line
    pattern3 = np.ones((60, 60), dtype=np.uint8) * 200
    cv2.circle(pattern3, (15, 15), 5, 50, -1)  # Top-left
    cv2.circle(pattern3, (30, 30), 5, 50, -1)  # Center
    cv2.circle(pattern3, (45, 45), 5, 50, -1)  # Bottom-right
    patterns[3] = pattern3
    
    # Value 4: Four corners
    pattern4 = np.ones((60, 60), dtype=np.uint8) * 200
    cv2.circle(pattern4, (15, 15), 5, 50, -1)  # Top-left
    cv2.circle(pattern4, (45, 15), 5, 50, -1)  # Top-right
    cv2.circle(pattern4, (15, 45), 5, 50, -1)  # Bottom-left
    cv2.circle(pattern4, (45, 45), 5, 50, -1)  # Bottom-right
    patterns[4] = pattern4
    
    # Value 5: Four corners + center
    pattern5 = np.ones((60, 60), dtype=np.uint8) * 200
    cv2.circle(pattern5, (15, 15), 5, 50, -1)  # Top-left
    cv2.circle(pattern5, (45, 15), 5, 50, -1)  # Top-right
    cv2.circle(pattern5, (15, 45), 5, 50, -1)  # Bottom-left
    cv2.circle(pattern5, (45, 45), 5, 50, -1)  # Bottom-right
    cv2.circle(pattern5, (30, 30), 5, 50, -1)  # Center
    patterns[5] = pattern5
    
    return patterns


if __name__ == "__main__":
    analyze_values_2_to_5_problem()
    test_enhanced_value_estimation() 