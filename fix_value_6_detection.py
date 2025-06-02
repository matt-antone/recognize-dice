#!/usr/bin/env python3
"""
Diagnose and fix the value 6 detection catastrophe.
Value 6 has unique 3x2 pip pattern that needs special handling.
"""

import sys
import os
import cv2
import numpy as np
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


class Value6DetectionFixer:
    """Fix the catastrophic value 6 detection failure."""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger(__name__)
    
    def create_enhanced_value_estimator(self, dice_region: np.ndarray) -> int:
        """
        Enhanced value estimation with specific fixes for value 6.
        Addresses the 3x2 pip pattern that standard algorithms miss.
        """
        if dice_region.size == 0:
            return 1
        
        # PRIORITY 1: Check for value 6 pattern specifically
        value_6_confidence = self._detect_value_6_pattern(dice_region)
        if value_6_confidence > 0.7:
            return 6
        
        # PRIORITY 2: Check for value 1 over-detection issue
        if self._is_likely_single_pip(dice_region):
            return 1
        
        # PRIORITY 3: Use original multi-method approach for other values
        return self._original_estimation_with_fixes(dice_region)
    
    def _detect_value_6_pattern(self, dice_region: np.ndarray) -> float:
        """
        Specifically detect the value 6 pattern (3x2 grid of pips).
        Returns confidence score 0.0-1.0.
        """
        h, w = dice_region.shape
        
        # METHOD 1: Check for 6 distinct circular regions
        dots_found = self._find_circular_dots(dice_region)
        if len(dots_found) == 6:
            # Verify they form a 3x2 grid pattern
            if self._is_3x2_grid_pattern(dots_found, w, h):
                return 0.9
        
        # METHOD 2: Check for distinctive 3x2 spatial pattern
        pattern_score = self._analyze_3x2_spatial_pattern(dice_region)
        if pattern_score > 0.7:
            return pattern_score
        
        # METHOD 3: Template matching for value 6
        template_score = self._template_match_value_6(dice_region)
        
        return max(0.0, template_score)
    
    def _find_circular_dots(self, dice_region: np.ndarray) -> list:
        """Find all circular dot-like regions in the dice."""
        # Try multiple thresholding methods
        potential_dots = []
        
        # Method 1: Adaptive threshold
        try:
            adaptive = cv2.adaptiveThreshold(
                dice_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            dots_adaptive = self._extract_dots_from_binary(adaptive, dice_region.size)
            potential_dots.extend(dots_adaptive)
        except:
            pass
        
        # Method 2: OTSU threshold
        try:
            _, otsu = cv2.threshold(dice_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dots_otsu = self._extract_dots_from_binary(otsu, dice_region.size)
            potential_dots.extend(dots_otsu)
        except:
            pass
        
        # Method 3: Inverted OTSU (for white pips on dark dice)
        try:
            _, otsu_inv = cv2.threshold(dice_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dots_inv = self._extract_dots_from_binary(otsu_inv, dice_region.size)
            potential_dots.extend(dots_inv)
        except:
            pass
        
        # Deduplicate dots that are too close
        return self._deduplicate_dots(potential_dots)
    
    def _extract_dots_from_binary(self, binary: np.ndarray, dice_area: int) -> list:
        """Extract circular dots from binary image."""
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        # OPTIMIZED: For real dice dimensions
        min_dot_area = 10  # Smaller minimum to catch value 6 pips
        max_dot_area = 120  # Larger maximum for various dice sizes
        relative_max = dice_area // 25  # More generous for value 6
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_dot_area <= area <= min(max_dot_area, relative_max):
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.2:  # Lenient for value 6
                        # Get center
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            dots.append({'center': (cx, cy), 'area': area, 'circularity': circularity})
        
        return dots
    
    def _deduplicate_dots(self, dots: list) -> list:
        """Remove duplicate dots that are too close together."""
        if len(dots) <= 1:
            return dots
        
        final_dots = []
        min_distance = 15  # Minimum distance between dot centers
        
        for dot in dots:
            too_close = False
            for existing in final_dots:
                dx = dot['center'][0] - existing['center'][0]
                dy = dot['center'][1] - existing['center'][1]
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                final_dots.append(dot)
        
        return final_dots
    
    def _is_3x2_grid_pattern(self, dots: list, width: int, height: int) -> bool:
        """Check if 6 dots form a 3x2 grid pattern (value 6)."""
        if len(dots) != 6:
            return False
        
        # Extract centers
        centers = [dot['center'] for dot in dots]
        
        # Sort by Y coordinate to find rows
        centers_by_y = sorted(centers, key=lambda p: p[1])
        
        # Check if we can form 2 rows of 3 dots each
        # Row 1: first 3 dots
        # Row 2: last 3 dots
        
        row1 = centers_by_y[:3]
        row2 = centers_by_y[3:]
        
        # Check if each row has 3 dots roughly aligned horizontally
        row1_y_var = np.var([p[1] for p in row1])
        row2_y_var = np.var([p[1] for p in row2])
        
        # Allow some variance but not too much
        max_y_variance = height * 0.1
        
        if row1_y_var < max_y_variance and row2_y_var < max_y_variance:
            # Check if rows are vertically separated
            row1_avg_y = np.mean([p[1] for p in row1])
            row2_avg_y = np.mean([p[1] for p in row2])
            
            vertical_separation = abs(row2_avg_y - row1_avg_y)
            min_separation = height * 0.2
            
            if vertical_separation > min_separation:
                return True
        
        return False
    
    def _analyze_3x2_spatial_pattern(self, dice_region: np.ndarray) -> float:
        """Analyze spatial patterns specific to value 6."""
        h, w = dice_region.shape
        
        # Divide into 6 regions (3x2 grid)
        region_h = h // 2
        region_w = w // 3
        
        regions = []
        for row in range(2):
            for col in range(3):
                y1 = row * region_h
                y2 = (row + 1) * region_h
                x1 = col * region_w
                x2 = (col + 1) * region_w
                
                region = dice_region[y1:y2, x1:x2]
                if region.size > 0:
                    regions.append(region)
        
        if len(regions) != 6:
            return 0.0
        
        # Check if each region has a dark spot (pip)
        dark_regions = 0
        overall_mean = np.mean(dice_region)
        
        for region in regions:
            region_min = np.min(region)
            region_mean = np.mean(region)
            
            # Check for dark spot in this region
            if region_min < overall_mean * 0.6 or region_mean < overall_mean * 0.8:
                dark_regions += 1
        
        # Value 6 should have dark spots in all 6 regions
        confidence = dark_regions / 6.0
        return confidence
    
    def _template_match_value_6(self, dice_region: np.ndarray) -> float:
        """Simple template matching for value 6 pattern."""
        # This is a simplified version - in practice, you'd have actual templates
        h, w = dice_region.shape
        
        # Check if the image has the right characteristics for value 6
        # Value 6 typically has:
        # - Multiple dark regions
        # - Distributed across the dice face
        # - Lower overall brightness (more pips = darker)
        
        # Check overall darkness (more pips = darker overall)
        avg_brightness = np.mean(dice_region)
        brightness_score = 1.0 - (avg_brightness / 255.0)
        
        # Check for multiple distinct dark regions
        _, binary = cv2.threshold(dice_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        distinct_regions = len([c for c in contours if cv2.contourArea(c) > 20])
        region_score = min(1.0, distinct_regions / 6.0)
        
        # Combine scores
        template_score = (brightness_score * 0.6 + region_score * 0.4)
        
        return template_score
    
    def _is_likely_single_pip(self, dice_region: np.ndarray) -> bool:
        """Check if this is likely a single pip (value 1) to reduce over-detection."""
        h, w = dice_region.shape
        
        # Look for a single central dark region
        center_region = dice_region[h//3:2*h//3, w//3:2*w//3]
        
        if center_region.size == 0:
            return False
        
        # Check if center is significantly darker than edges
        center_mean = np.mean(center_region)
        edge_mean = np.mean([
            np.mean(dice_region[0:h//4, :]),  # Top edge
            np.mean(dice_region[3*h//4:h, :]),  # Bottom edge
            np.mean(dice_region[:, 0:w//4]),  # Left edge
            np.mean(dice_region[:, 3*w//4:w])  # Right edge
        ])
        
        contrast = edge_mean - center_mean
        
        # Single pip should have strong central contrast
        return contrast > 30
    
    def _original_estimation_with_fixes(self, dice_region: np.ndarray) -> int:
        """Use original estimation but with fixes for known issues."""
        # This would use the existing multi-method approach
        # but with adjustments for the known over-detection of 1s
        
        # For now, simple fallback
        dots = self._find_circular_dots(dice_region)
        dot_count = len(dots)
        
        # Apply some heuristics to reduce false 1s
        if dot_count == 1:
            # Double-check this isn't a fragment of a larger pattern
            if self._is_likely_single_pip(dice_region):
                return 1
            else:
                return 2  # Conservative estimate
        
        return max(1, min(6, dot_count))
    
    def test_value_6_improvements(self):
        """Test the value 6 improvements on sample data."""
        print("🔧 TESTING VALUE 6 DETECTION IMPROVEMENTS")
        print("=" * 50)
        
        # Create test patterns for value 6
        test_patterns = self._create_test_patterns()
        
        for i, pattern in enumerate(test_patterns):
            print(f"\nTesting pattern {i+1}:")
            
            # Test original estimation
            original_detector = FallbackDetection(self.config)
            original_value = original_detector._estimate_dice_value(pattern)
            
            # Test improved estimation
            improved_value = self.create_enhanced_value_estimator(pattern)
            confidence = self._detect_value_6_pattern(pattern)
            
            print(f"  Original: {original_value}")
            print(f"  Improved: {improved_value}")
            print(f"  Value 6 confidence: {confidence:.2f}")
            
            if improved_value == 6:
                print("  ✅ IMPROVEMENT: Now detecting as value 6!")
            elif original_value == improved_value:
                print("  ➖ No change")
            else:
                print(f"  🔄 Changed from {original_value} to {improved_value}")
    
    def _create_test_patterns(self) -> list:
        """Create synthetic test patterns for value 6."""
        patterns = []
        
        # Pattern 1: 3x2 grid of dark circles
        pattern1 = np.ones((60, 90), dtype=np.uint8) * 200  # Light background
        
        # Add 6 dark circles in 3x2 pattern
        positions = [
            (15, 20), (15, 45), (15, 70),  # Top row
            (45, 20), (45, 45), (45, 70)   # Bottom row
        ]
        
        for y, x in positions:
            cv2.circle(pattern1, (x, y), 5, 50, -1)  # Dark circles
        
        patterns.append(pattern1)
        
        # Pattern 2: More realistic with noise
        pattern2 = pattern1.copy()
        noise = np.random.normal(0, 10, pattern2.shape).astype(np.uint8)
        pattern2 = cv2.add(pattern2, noise)
        patterns.append(pattern2)
        
        return patterns


def main():
    """Test and implement value 6 detection fixes."""
    print("🎲 VALUE 6 DETECTION FIXER")
    print("Addressing the 0.4% detection catastrophe")
    print("=" * 50)
    
    fixer = Value6DetectionFixer()
    
    # Test improvements
    fixer.test_value_6_improvements()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Integrate enhanced value estimation into fallback_detection.py")
    print("2. Test against Kaggle dataset to measure improvement")
    print("3. Verify value 1 over-detection is also reduced")


if __name__ == "__main__":
    main() 