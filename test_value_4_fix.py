#!/usr/bin/env python3
"""
Test Value 4 Detection Fix
Validates the 2x2 grid pattern detection for dice value 4.
Addresses the toggle between 1 and 2 detections.
"""

import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detection.fallback_detection import FallbackDetection
from utils.config import Config

def create_value_4_pattern(size=80):
    """Create a synthetic value 4 pattern (2x2 grid of dark pips)."""
    pattern = np.ones((size, size), dtype=np.uint8) * 200  # Light background
    
    # Create 4 pips in 2x2 grid
    pip_radius = size // 12  # Pip size
    
    # Calculate positions for 2x2 grid
    quarter_x = size // 4
    quarter_y = size // 4
    three_quarter_x = 3 * size // 4
    three_quarter_y = 3 * size // 4
    
    # Draw 4 pips
    positions = [
        (quarter_x, quarter_y),        # Top-left
        (three_quarter_x, quarter_y),  # Top-right
        (quarter_x, three_quarter_y),  # Bottom-left
        (three_quarter_x, three_quarter_y)  # Bottom-right
    ]
    
    for x, y in positions:
        cv2.circle(pattern, (x, y), pip_radius, 50, -1)  # Dark pips
    
    return pattern

def test_value_4_detection():
    """Test the enhanced value 4 detection."""
    print("🎯 Testing Value 4 Detection Fix")
    print("=" * 50)
    
    # Initialize detection
    config = Config()
    detector = FallbackDetection(config)
    
    # Test 1: Synthetic value 4 pattern
    print("\n🔬 Test 1: Synthetic Value 4 Pattern")
    value_4_pattern = create_value_4_pattern(80)
    
    # Test the specific value 4 detection
    value_4_confidence = detector._detect_value_4_pattern(value_4_pattern)
    estimated_value = detector._estimate_dice_value(value_4_pattern)
    
    print(f"  Pattern size: 80x80 pixels")
    print(f"  Pattern: 2x2 grid (4 dark pips)")
    print(f"  Value 4 confidence: {value_4_confidence:.3f}")
    print(f"  Estimated value: {estimated_value}")
    
    # Test the individual methods
    dots_found = detector._find_all_circular_dots(value_4_pattern)
    is_2x2_grid = detector._is_2x2_grid_pattern(dots_found, 80, 80) if len(dots_found) == 4 else False
    spatial_score = detector._analyze_2x2_spatial_pattern(value_4_pattern)
    template_score = detector._template_match_value_4(value_4_pattern)
    
    print(f"\n  🔍 Detailed Analysis:")
    print(f"    Dots found: {len(dots_found)}")
    print(f"    Is 2x2 grid: {is_2x2_grid}")
    print(f"    Spatial score: {spatial_score:.3f}")
    print(f"    Template score: {template_score:.3f}")
    
    # Test 2: Multiple synthetic patterns
    print(f"\n🔬 Test 2: Multiple Value 4 Patterns")
    
    test_cases = [
        ("Small (60x60)", create_value_4_pattern(60)),
        ("Medium (80x80)", create_value_4_pattern(80)),
        ("Large (100x100)", create_value_4_pattern(100)),
    ]
    
    success_count = 0
    for name, pattern in test_cases:
        estimated = detector._estimate_dice_value(pattern)
        confidence = detector._detect_value_4_pattern(pattern)
        
        success = estimated == 4 and confidence > 0.7
        if success:
            success_count += 1
        
        status = "✅" if success else "❌"
        print(f"  {status} {name}: Value {estimated} (confidence: {confidence:.3f})")
    
    # Test 3: Compare with problematic patterns
    print(f"\n🔬 Test 3: Compare with Other Values")
    
    # Create patterns that might be confused with value 4
    single_pip = np.ones((80, 80), dtype=np.uint8) * 200
    cv2.circle(single_pip, (40, 40), 8, 50, -1)  # Single central pip
    
    # Test original problematic case
    value_1_est = detector._estimate_dice_value(single_pip)
    value_4_est = detector._estimate_dice_value(value_4_pattern)
    
    print(f"  Single pip pattern → Value {value_1_est}")
    print(f"  2x2 grid pattern → Value {value_4_est}")
    
    distinction = "✅ Correctly distinguished" if value_1_est != value_4_est else "❌ Still confused"
    print(f"  {distinction}")
    
    # Results summary
    print(f"\n🎉 Results Summary")
    print(f"=" * 30)
    
    overall_success = (estimated_value == 4 and 
                      value_4_confidence > 0.7 and 
                      success_count >= 2 and
                      value_1_est != value_4_est)
    
    if overall_success:
        print("✅ SUCCESS: Value 4 detection fix working!")
        print("  - Correctly detects 2x2 grid patterns")
        print("  - High confidence scores")
        print("  - Distinguishes from value 1")
        print("  - Should fix toggle issue")
        return True
    else:
        print("❌ FAILURE: Value 4 detection needs more work")
        print(f"  - Target value: 4, Got: {estimated_value}")
        print(f"  - Target confidence: >0.7, Got: {value_4_confidence:.3f}")
        print(f"  - Pattern tests passed: {success_count}/3")
        return False

if __name__ == "__main__":
    success = test_value_4_detection()
    
    print(f"\n💡 Next Steps:")
    if success:
        print("  1. Test with real dice on Pi 3 hardware")
        print("  2. Verify toggle issue is resolved")
        print("  3. Consider optimizing values 2, 3, 5 detection")
    else:
        print("  1. Debug detection parameters")
        print("  2. Adjust confidence thresholds") 
        print("  3. Test with different pip sizes/contrast")
    
    sys.exit(0 if success else 1) 