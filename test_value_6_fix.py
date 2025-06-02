#!/usr/bin/env python3
"""
Quick test to verify our value 6 fix against Kaggle dataset.
Focus on measuring improvement specifically for value 6.
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


def test_value_6_improvements():
    """Test value 6 improvements on Kaggle dataset."""
    print("🎯 TESTING VALUE 6 FIX ON KAGGLE DATASET")
    print("=" * 50)
    
    # Load previous results
    results_file = Path("kaggle_dataset_results.json")
    if not results_file.exists():
        print("❌ No previous Kaggle results found. Run test_kaggle_dataset.py first.")
        return
    
    with open(results_file, 'r') as f:
        previous_results = json.load(f)
    
    # Initialize detection system
    config = Config()
    detector = FallbackDetection(config)
    
    # Test images directory
    images_dir = Path("datasets/d6-dice")
    if not images_dir.exists():
        print("❌ Kaggle dataset not found. Run test_kaggle_dataset.py first.")
        return
    
    # Focus on value 6 test cases
    value_6_improvements = []
    value_6_files = []
    
    # Find value 6 test cases from previous results
    for result in previous_results.get('detailed_results', []):
        if result['expected_value'] == 6:
            value_6_files.append(result['filename'])
    
    if not value_6_files:
        print("❌ No value 6 test cases found in previous results.")
        return
    
    print(f"📊 Testing {len(value_6_files)} value 6 cases...")
    
    correct_before = 0
    correct_after = 0
    
    for filename in value_6_files[:10]:  # Test first 10 value 6 cases
        image_path = images_dir / filename
        if not image_path.exists():
            continue
        
        # Load and process image
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get the dice detection from the main system
        detections = detector.detect_dice(gray)
        
        if detections:
            detected_value = detections[0]['value']
            confidence = detections[0]['confidence']
            
            # Check if this was correct before (from previous results)
            prev_result = next((r for r in previous_results['detailed_results'] if r['filename'] == filename), None)
            if prev_result and prev_result['detected_value'] == 6:
                correct_before += 1
            
            # Check if correct now
            if detected_value == 6:
                correct_after += 1
                print(f"✅ {filename}: Detected {detected_value} (confidence: {confidence:.2f})")
            else:
                print(f"❌ {filename}: Detected {detected_value} (expected 6)")
        else:
            print(f"🔍 {filename}: No detection")
    
    print(f"\n📈 VALUE 6 IMPROVEMENT RESULTS:")
    print(f"Before fix: {correct_before}/{len(value_6_files[:10])} correct ({correct_before/len(value_6_files[:10])*100:.1f}%)")
    print(f"After fix:  {correct_after}/{len(value_6_files[:10])} correct ({correct_after/len(value_6_files[:10])*100:.1f}%)")
    
    improvement = correct_after - correct_before
    if improvement > 0:
        print(f"🎉 IMPROVEMENT: +{improvement} correct detections!")
    elif improvement == 0:
        print("➖ No change in accuracy")
    else:
        print(f"⚠️  Regression: -{abs(improvement)} fewer correct detections")


def test_specific_value_6_pattern():
    """Test our value 6 detection on a created pattern."""
    print("\n🔬 TESTING SYNTHETIC VALUE 6 PATTERN")
    print("=" * 40)
    
    # Create a synthetic value 6 pattern
    pattern = np.ones((60, 90), dtype=np.uint8) * 200  # Light background
    
    # Add 6 dark circles in 3x2 pattern
    positions = [
        (15, 20), (15, 45), (15, 70),  # Top row
        (45, 20), (45, 45), (45, 70)   # Bottom row
    ]
    
    for y, x in positions:
        cv2.circle(pattern, (x, y), 6, 40, -1)  # Dark circles
    
    # Test detection
    config = Config()
    detector = FallbackDetection(config)
    
    # Simulate a dice detection bbox around our pattern
    detections = []
    
    # Manually test the value estimation
    estimated_value = detector._estimate_dice_value(pattern)
    
    print(f"Synthetic 6-pattern result: {estimated_value}")
    if estimated_value == 6:
        print("✅ SUCCESS: Synthetic pattern correctly detected as 6!")
    else:
        print(f"❌ FAILED: Synthetic pattern detected as {estimated_value}")


if __name__ == "__main__":
    test_value_6_improvements()
    test_specific_value_6_pattern() 