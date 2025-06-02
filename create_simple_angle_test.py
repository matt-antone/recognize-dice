#!/usr/bin/env python3
"""
Simple angle diagnostic test for current dice detection system.
Helps identify specific failure points to fix.
"""

import sys
import os
import cv2
import numpy as np
import time
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


class AngleDiagnostic:
    """Simple diagnostic tool for angle detection performance."""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger(__name__)
        self.camera = None
        self.detector = None
        self.results = []
    
    def initialize_hardware(self):
        """Initialize camera and detection."""
        print("Initializing AI camera...")
        self.camera = CameraInterface(self.config)
        self.camera.start()
        
        print("Initializing detection...")
        self.detector = FallbackDetection(self.config)
        
        # Wait for camera to stabilize
        time.sleep(2)
        print("✅ Hardware ready")
    
    def test_angle_scenario(self, scenario_name, user_dice_value=None):
        """Test detection for a specific angle scenario."""
        print(f"\n=== Testing: {scenario_name} ===")
        
        if user_dice_value:
            print(f"Expected dice value: {user_dice_value}")
        
        input("Position dice and press Enter to test...")
        
        # Capture frame
        frame = self.camera.capture_frame()
        if frame is None:
            print("❌ Failed to capture frame")
            return None
        
        # Run detection
        start_time = time.time()
        detections = self.detector.detect_dice(frame)
        detection_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Analyze results
        result = {
            'scenario': scenario_name,
            'expected_value': user_dice_value,
            'detection_count': len(detections),
            'detection_time_ms': detection_time,
            'detections': detections,
            'success': len(detections) > 0
        }
        
        # Display results
        print(f"Detection time: {detection_time:.1f}ms")
        print(f"Dice detected: {len(detections)}")
        
        if len(detections) > 0:
            for i, dice in enumerate(detections):
                print(f"  Dice {i+1}:")
                print(f"    Method: {dice['method']}")
                print(f"    Value: {dice['value']}")
                print(f"    Confidence: {dice['confidence']:.2f}")
                print(f"    Center: {dice['center']}")
                print(f"    Area: {dice['area']:.0f}")
                
                if user_dice_value:
                    accuracy = "✅ CORRECT" if dice['value'] == user_dice_value else "❌ WRONG"
                    print(f"    Accuracy: {accuracy}")
        else:
            print("  ❌ No dice detected")
        
        self.results.append(result)
        return result
    
    def run_comprehensive_test(self):
        """Run comprehensive angle testing."""
        print("Comprehensive Angle Detection Test")
        print("=" * 50)
        print("This test will help identify where detection fails")
        print("You'll need ONE dice that you can position at different angles")
        print()
        
        # Get dice value from user
        dice_value = None
        while dice_value is None:
            try:
                dice_value = int(input("What value is your test dice showing (1-6)? "))
                if not (1 <= dice_value <= 6):
                    print("Please enter a value between 1 and 6")
                    dice_value = None
            except ValueError:
                print("Please enter a number between 1 and 6")
        
        print(f"\nTesting with dice showing: {dice_value}")
        print("\nWe'll test different angles and conditions...")
        
        # Test scenarios
        scenarios = [
            ("Straight Down (Baseline)", "Place dice flat, camera directly above"),
            ("Slight Tilt", "Tilt dice slightly (15-20 degrees)"),
            ("Medium Angle", "Tilt dice more (30-40 degrees)"),
            ("High Angle", "Tilt dice significantly (45+ degrees)"),
            ("Side View", "Turn dice so side is mostly visible"),
            ("Corner View", "Show two faces (corner view)"),
            ("Different Lighting", "Change lighting (add shadow, move light source)"),
            ("Different Background", "Change surface color/texture if possible")
        ]
        
        for scenario_name, instruction in scenarios:
            print(f"\n--- {scenario_name} ---")
            print(f"Instructions: {instruction}")
            
            # Allow user to skip scenarios
            skip = input("Press Enter to test, or 's' to skip: ").lower().strip()
            if skip == 's':
                print("Skipped")
                continue
            
            self.test_angle_scenario(scenario_name, dice_value)
        
        # Generate summary
        self.generate_summary()
    
    def run_quick_test(self):
        """Run quick 3-position test."""
        print("Quick Angle Test")
        print("=" * 30)
        print("Test 3 basic positions to identify main issues")
        
        dice_value = int(input("What value is your dice showing (1-6)? "))
        
        quick_scenarios = [
            ("Flat/Straight Down", "Place dice flat on surface"),
            ("Tilted", "Tilt dice at moderate angle"), 
            ("Side Visible", "Show the side of the dice")
        ]
        
        for scenario_name, instruction in quick_scenarios:
            print(f"\n--- {scenario_name} ---")
            print(f"Instructions: {instruction}")
            self.test_angle_scenario(scenario_name, dice_value)
        
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary and recommendations."""
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.results)
        successful_detections = sum(1 for r in self.results if r['success'])
        
        print(f"Total tests: {total_tests}")
        print(f"Successful detections: {successful_detections}/{total_tests}")
        print(f"Detection rate: {(successful_detections/total_tests)*100:.1f}%")
        
        # Performance analysis
        if self.results:
            avg_time = np.mean([r['detection_time_ms'] for r in self.results])
            print(f"Average detection time: {avg_time:.1f}ms")
        
        # Detailed results
        print("\nDetailed Results:")
        for i, result in enumerate(self.results, 1):
            status = "✅" if result['success'] else "❌"
            print(f"{i:2}. {status} {result['scenario']}")
            if result['success'] and result['detections']:
                dice = result['detections'][0]
                expected = result['expected_value']
                actual = dice['value']
                accuracy = "✅" if actual == expected else f"❌ (got {actual})"
                print(f"     Value: {accuracy}")
        
        # Identify problems
        print("\nIssue Analysis:")
        failed_scenarios = [r for r in self.results if not r['success']]
        
        if not failed_scenarios:
            print("✅ No detection failures - system working well!")
        else:
            print("❌ Detection failures in:")
            for result in failed_scenarios:
                print(f"   - {result['scenario']}")
        
        # Value accuracy analysis
        accurate_values = []
        for result in self.results:
            if result['success'] and result['expected_value']:
                dice = result['detections'][0]
                accurate_values.append(dice['value'] == result['expected_value'])
        
        if accurate_values:
            accuracy_rate = (sum(accurate_values) / len(accurate_values)) * 100
            print(f"\nValue accuracy: {accuracy_rate:.1f}%")
        
        # Recommendations
        self.generate_recommendations()
        
        # Save results
        self.save_results()
    
    def generate_recommendations(self):
        """Generate specific recommendations based on test results."""
        print("\nRECOMMENDATIONS:")
        
        failed_scenarios = [r['scenario'] for r in self.results if not r['success']]
        
        if 'Side View' in failed_scenarios or 'High Angle' in failed_scenarios:
            print("📐 ANGLE ISSUES DETECTED:")
            print("   - Current system struggles with angled dice")
            print("   - Recommend: Relax shape constraints")
            print("   - Recommend: Add perspective correction")
        
        if 'Different Lighting' in failed_scenarios:
            print("💡 LIGHTING ISSUES DETECTED:")
            print("   - Current system sensitive to lighting changes")  
            print("   - Recommend: Improve preprocessing")
            print("   - Recommend: Better edge detection parameters")
        
        detection_rates = [r['success'] for r in self.results]
        if detection_rates and sum(detection_rates) < len(detection_rates) * 0.7:
            print("🎯 LOW DETECTION RATE:")
            print("   - Consider reducing minimum area thresholds")
            print("   - Consider more permissive aspect ratios")
        
        # Check value accuracy
        accurate_values = []
        for result in self.results:
            if result['success'] and result['expected_value'] and result['detections']:
                dice = result['detections'][0]
                accurate_values.append(dice['value'] == result['expected_value'])
        
        if accurate_values and sum(accurate_values) < len(accurate_values) * 0.7:
            print("🔢 VALUE ACCURACY ISSUES:")
            print("   - Pip detection needs improvement")
            print("   - Consider different thresholding methods")
            print("   - Consider pattern-based recognition")
    
    def save_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"angle_diagnostic_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("Dice Detection Angle Diagnostic Results\n")
            f.write("=" * 50 + "\n\n")
            
            for result in self.results:
                f.write(f"Scenario: {result['scenario']}\n")
                f.write(f"Success: {result['success']}\n")
                f.write(f"Detection count: {result['detection_count']}\n")
                f.write(f"Time: {result['detection_time_ms']:.1f}ms\n")
                
                if result['detections']:
                    for i, dice in enumerate(result['detections']):
                        f.write(f"  Dice {i+1}: Value={dice['value']}, "
                               f"Method={dice['method']}, "
                               f"Confidence={dice['confidence']:.2f}\n")
                f.write("\n")
        
        print(f"\nResults saved to: {filename}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.camera:
            self.camera.stop()
            self.camera.cleanup()


def main():
    """Main diagnostic function."""
    diagnostic = AngleDiagnostic()
    
    try:
        diagnostic.initialize_hardware()
        
        print("\nChoose test type:")
        print("1. Quick test (3 positions)")
        print("2. Comprehensive test (8 scenarios)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            diagnostic.run_quick_test()
        else:
            diagnostic.run_comprehensive_test()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        diagnostic.cleanup()


if __name__ == "__main__":
    main() 