#!/usr/bin/env python3
"""
Debug Value 4 Detection Issue
Investigate why realistic dice are detected as value 6 instead of 4.
"""

import numpy as np
import cv2
import sys
import os

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

def debug_detection_steps(dice_region, detector):
    """Debug each step of the detection process."""
    print("🔍 Debugging Detection Steps")
    print("-" * 40)
    
    h, w = dice_region.shape
    print(f"Input size: {w}x{h}")
    print(f"Average brightness: {np.mean(dice_region):.1f}")
    print(f"Brightness std: {np.std(dice_region):.1f}")
    
    # Step 1: Find dots
    print(f"\n🎯 Step 1: Find Circular Dots")
    dots_found = detector._find_all_circular_dots(dice_region)
    print(f"  Dots found: {len(dots_found)}")
    for i, dot in enumerate(dots_found):
        print(f"    Dot {i+1}: center={dot['center']}, area={dot['area']:.1f}, circularity={dot['circularity']:.3f}")
    
    # Step 2: Grid pattern analysis
    if len(dots_found) == 4:
        print(f"\n🎯 Step 2: 2x2 Grid Pattern Check")
        is_2x2 = detector._is_2x2_grid_pattern(dots_found, w, h)
        print(f"  Is 2x2 grid: {is_2x2}")
        
        if not is_2x2:
            # Debug why it's not detected as 2x2
            centers = [dot['center'] for dot in dots_found]
            mid_y = h / 2
            top_dots = [p for p in centers if p[1] < mid_y]
            bottom_dots = [p for p in centers if p[1] >= mid_y]
            print(f"    Top dots: {len(top_dots)} - {top_dots}")
            print(f"    Bottom dots: {len(bottom_dots)} - {bottom_dots}")
    
    # Step 3: Spatial pattern analysis
    print(f"\n🎯 Step 3: Spatial Pattern Analysis")
    spatial_score = detector._analyze_2x2_spatial_pattern(dice_region)
    print(f"  Spatial score: {spatial_score:.3f}")
    
    # Debug quadrants
    top_left = dice_region[0:h//2, 0:w//2]
    top_right = dice_region[0:h//2, w//2:w]
    bottom_left = dice_region[h//2:h, 0:w//2]
    bottom_right = dice_region[h//2:h, w//2:w]
    
    quadrants = [('top_left', top_left), ('top_right', top_right), 
                 ('bottom_left', bottom_left), ('bottom_right', bottom_right)]
    
    for name, quad in quadrants:
        if quad.size > 0:
            qh, qw = quad.shape
            center_region = quad[qh//3:2*qh//3, qw//3:2*qw//3]
            if center_region.size > 0:
                center_mean = np.mean(center_region)
                edge_mean = np.mean([
                    np.mean(quad[0:qh//4, :]) if qh//4 > 0 else center_mean,
                    np.mean(quad[3*qh//4:qh, :]) if qh - 3*qh//4 > 0 else center_mean,
                    np.mean(quad[:, 0:qw//4]) if qw//4 > 0 else center_mean,
                    np.mean(quad[:, 3*qw//4:qw]) if qw - 3*qw//4 > 0 else center_mean
                ])
                contrast = edge_mean - center_mean
                print(f"    {name}: center={center_mean:.1f}, edge={edge_mean:.1f}, contrast={contrast:.1f}")
    
    # Step 4: Template matching
    print(f"\n🎯 Step 4: Template Matching")
    template_score = detector._template_match_value_4(dice_region)
    print(f"  Template score: {template_score:.3f}")
    
    # Debug template matching with CORRECTED calculations
    avg_brightness = np.mean(dice_region)
    
    # Calculate brightness score correctly
    if 180 <= avg_brightness <= 230:
        brightness_score = 1.0 - abs(avg_brightness - 205) / 25.0
        brightness_range = "Light dice"
    elif 140 <= avg_brightness <= 179:
        brightness_score = 1.0 - abs(avg_brightness - 160) / 20.0
        brightness_range = "Medium dice"
    else:
        brightness_score = 0.0
        brightness_range = "Out of range"
    
    brightness_score = max(0.0, brightness_score)
    
    _, binary = cv2.threshold(dice_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (h * w) // 100
    significant_regions = [c for c in contours if cv2.contourArea(c) > min_area]
    
    region_count = len(significant_regions)
    if region_count == 4:
        region_score = 1.0
    elif region_count == 3:
        region_score = 0.8
    elif region_count == 5:
        region_score = 0.6
    elif region_count == 2:
        region_score = 0.4
    else:
        region_score = 0.0
    
    print(f"    Brightness: {avg_brightness:.1f} ({brightness_range})")
    print(f"    Brightness score: {brightness_score:.3f}")
    print(f"    Significant regions: {region_count}")
    print(f"    Region score: {region_score:.3f}")
    print(f"    Min area threshold: {min_area}")
    print(f"    Final template score: {(brightness_score * 0.7 + region_score * 0.3):.3f}")

def main():
    """Debug the value 4 detection issue."""
    print("🐛 Debug Value 4 Detection Issue")
    print("=" * 50)
    
    config = Config()
    detector = FallbackDetection(config)
    
    # Test cases
    test_cases = [
        ("Perfect synthetic", create_realistic_dice_4(100, 0.0, 0)),
        ("Light noise", create_realistic_dice_4(100, 0.05, 1)),
        ("Medium noise", create_realistic_dice_4(100, 0.1, 1)),
        ("Heavy noise", create_realistic_dice_4(100, 0.15, 2)),
    ]
    
    for test_name, dice_region in test_cases:
        print(f"\n{'='*60}")
        print(f"🎲 Test: {test_name}")
        print(f"{'='*60}")
        
        # Get overall detection result
        estimated_value = detector._estimate_dice_value(dice_region)
        value_4_conf = detector._detect_value_4_pattern(dice_region)
        value_6_conf = detector._detect_value_6_pattern(dice_region)
        
        print(f"📊 Result: Detected value {estimated_value}")
        print(f"    Value 4 confidence: {value_4_conf:.3f}")
        print(f"    Value 6 confidence: {value_6_conf:.3f}")
        
        # Debug the detection process
        debug_detection_steps(dice_region, detector)
        
        # Save debug image
        debug_filename = f"debug_{test_name.replace(' ', '_').lower()}.png"
        cv2.imwrite(debug_filename, dice_region)
        print(f"\n💾 Debug image saved: {debug_filename}")

if __name__ == "__main__":
    main() 