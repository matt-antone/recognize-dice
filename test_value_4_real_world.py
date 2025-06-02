#!/usr/bin/env python3
"""
Real-World Value 4 Toggle Fix Test
Simulates the actual issue where value 4 dice toggle between detecting as 1 and 2.
Tests the fix under realistic conditions.
"""

import numpy as np
import cv2
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detection.fallback_detection import FallbackDetection
from utils.config import Config

def create_realistic_dice_4(size=100, noise_level=0.1, blur_amount=1):
    """Create a realistic-looking dice face with value 4."""
    # Start with light dice background
    dice = np.ones((size, size), dtype=np.uint8) * 220
    
    # Add some texture/noise to simulate real dice surface
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 255, (size, size))
        dice = np.clip(dice + noise, 0, 255).astype(np.uint8)
    
    # Create 4 pips in 2x2 arrangement
    pip_size = max(6, size // 15)  # Realistic pip size
    
    # Position the 4 pips
    offset = size // 3
    positions = [
        (offset, offset),           # Top-left
        (size - offset, offset),    # Top-right  
        (offset, size - offset),    # Bottom-left
        (size - offset, size - offset)  # Bottom-right
    ]
    
    # Draw pips with slight imperfections
    for x, y in positions:
        # Add slight randomness to pip position and size
        actual_x = x + np.random.randint(-2, 3)
        actual_y = y + np.random.randint(-2, 3)
        actual_size = pip_size + np.random.randint(-1, 2)
        
        cv2.circle(dice, (actual_x, actual_y), actual_size, 40, -1)
    
    # Apply slight blur to simulate camera defocus
    if blur_amount > 0:
        dice = cv2.GaussianBlur(dice, (blur_amount*2+1, blur_amount*2+1), 0)
    
    return dice

def create_confusing_patterns():
    """Create patterns that might be confused with value 4."""
    patterns = {}
    
    # Pattern that might be detected as 1 (single central pip)
    single = np.ones((100, 100), dtype=np.uint8) * 220
    cv2.circle(single, (50, 50), 8, 40, -1)
    patterns['single_pip'] = single
    
    # Pattern that might be detected as 2 (diagonal pips)
    double = np.ones((100, 100), dtype=np.uint8) * 220
    cv2.circle(double, (30, 30), 8, 40, -1)
    cv2.circle(double, (70, 70), 8, 40, -1)
    patterns['diagonal_2'] = double
    
    # Pattern with 3 pips in triangle
    triple = np.ones((100, 100), dtype=np.uint8) * 220
    cv2.circle(triple, (50, 25), 7, 40, -1)
    cv2.circle(triple, (35, 75), 7, 40, -1)
    cv2.circle(triple, (65, 75), 7, 40, -1)
    patterns['triangle_3'] = triple
    
    return patterns

def simulate_toggle_scenario():
    """Simulate the scenario where value 4 toggles between 1 and 2."""
    print("🎯 Simulating Value 4 Toggle Scenario")
    print("=" * 50)
    
    config = Config()
    detector = FallbackDetection(config)
    
    # Test multiple realistic value 4 dice
    print("\n📱 Testing Multiple Value 4 Dice (Simulating Camera Frames)")
    
    results = []
    for i in range(10):
        # Create slightly different value 4 dice (simulating camera frames)
        dice_4 = create_realistic_dice_4(
            size=100, 
            noise_level=0.05 + i * 0.01,  # Increasing noise
            blur_amount=1 + (i % 3)       # Varying blur
        )
        
        estimated_value = detector._estimate_dice_value(dice_4)
        confidence_4 = detector._detect_value_4_pattern(dice_4)
        confidence_6 = detector._detect_value_6_pattern(dice_4)
        
        results.append(estimated_value)
        
        status = "✅" if estimated_value == 4 else "❌"
        print(f"  Frame {i+1:2}: {status} Detected value {estimated_value} "
              f"(4-conf: {confidence_4:.2f}, 6-conf: {confidence_6:.2f})")
    
    # Analyze consistency
    value_4_count = results.count(4)
    value_1_count = results.count(1)
    value_2_count = results.count(2)
    other_count = len(results) - value_4_count - value_1_count - value_2_count
    
    print(f"\n📊 Toggle Analysis:")
    print(f"  Value 4 detections: {value_4_count}/10 ({value_4_count*10}%)")
    print(f"  Value 1 detections: {value_1_count}/10 ({value_1_count*10}%)")
    print(f"  Value 2 detections: {value_2_count}/10 ({value_2_count*10}%)")
    print(f"  Other values: {other_count}/10 ({other_count*10}%)")
    
    # Check if toggle issue is fixed
    toggle_fixed = value_4_count >= 8 and (value_1_count + value_2_count) <= 2
    
    if toggle_fixed:
        print("  ✅ Toggle issue FIXED! Consistent value 4 detection")
    else:
        print("  ❌ Toggle issue persists! Inconsistent detection")
    
    return toggle_fixed, results

def test_disambiguation():
    """Test that value 4 is properly distinguished from other values."""
    print("\n🔍 Testing Value Disambiguation")
    print("=" * 40)
    
    config = Config()
    detector = FallbackDetection(config)
    
    # Test value 4 vs confusing patterns
    value_4_dice = create_realistic_dice_4(100, 0.05, 1)
    confusing_patterns = create_confusing_patterns()
    
    test_results = {}
    
    # Test actual value 4
    estimated_4 = detector._estimate_dice_value(value_4_dice)
    test_results['value_4'] = estimated_4
    print(f"  Actual value 4 → Detected: {estimated_4}")
    
    # Test confusing patterns
    for pattern_name, pattern in confusing_patterns.items():
        estimated = detector._estimate_dice_value(pattern)
        test_results[pattern_name] = estimated
        print(f"  {pattern_name} → Detected: {estimated}")
    
    # Check disambiguation success
    disambiguation_success = (
        test_results['value_4'] == 4 and
        test_results['single_pip'] == 1 and
        test_results['diagonal_2'] == 2 and
        test_results.get('triangle_3', 3) in [3, 4]  # 3 or 4 acceptable
    )
    
    return disambiguation_success, test_results

def main():
    """Run comprehensive value 4 toggle fix validation."""
    print("🚀 Value 4 Toggle Fix - Real World Validation")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test 1: Toggle scenario simulation
    toggle_fixed, toggle_results = simulate_toggle_scenario()
    
    # Test 2: Value disambiguation
    disambiguation_ok, disambiguation_results = test_disambiguation()
    
    # Overall assessment
    print(f"\n🎉 Final Assessment")
    print(f"=" * 30)
    
    overall_success = toggle_fixed and disambiguation_ok
    
    if overall_success:
        print("✅ SUCCESS: Value 4 detection fix is working excellently!")
        print("  ✓ Consistent value 4 detection across frames")
        print("  ✓ Proper disambiguation from other values")
        print("  ✓ Toggle issue between 1 and 2 resolved")
        print("\n  🎯 Real-world deployment ready!")
        
    else:
        print("❌ ISSUES: Value 4 detection needs refinement")
        if not toggle_fixed:
            print("  • Toggle consistency issue remains")
        if not disambiguation_ok:
            print("  • Value disambiguation needs improvement")
        
        print("\n  🔧 Suggested improvements:")
        print("    - Fine-tune detection thresholds")
        print("    - Enhance 2x2 grid pattern recognition")
        print("    - Test with more varied dice conditions")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Test completed in {elapsed_time:.2f} seconds")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    
    print(f"\n💡 Next Actions:")
    if success:
        print("  1. 🎯 Deploy to Pi 3 for hardware validation")
        print("  2. 📹 Test with real dice using camera")
        print("  3. 📊 Validate with users reporting toggle issues")
        print("  4. 🔧 Consider optimizing values 2, 3, 5 next")
    else:
        print("  1. 🐛 Debug specific failure patterns")
        print("  2. 🔬 Analyze failed test cases")
        print("  3. 🎛️  Adjust detection parameters")
        print("  4. 🔄 Re-test after improvements")
    
    sys.exit(0 if success else 1) 