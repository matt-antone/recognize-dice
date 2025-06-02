#!/usr/bin/env python3
"""
Fallback dice detection using computer vision when ML model is unavailable.
Based on debug analysis showing contour and blob detection work better than HoughCircles.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

class FallbackDetection:
    """Computer vision-based dice detection for when ML model is unavailable."""
    
    def __init__(self, config):
        self.config = config
        
        # Detection parameters optimized based on debug results
        # FIXED: Much stricter area limits to prevent pips being detected as dice
        self.min_dice_area = 800   # Minimum ~30x30 pixel dice (much larger than individual pips)
        self.max_dice_area = 8000  # Maximum dice size for reasonable camera distance
        self.min_circularity = 0.3
        self.min_convexity = 0.5
        
        # Edge filtering - exclude objects too close to frame boundaries
        self.edge_margin = 30  # Minimum distance from frame edge
        
        # Blob detector setup (uses strict area limits to prevent pip detection as dice)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self.min_dice_area  # Now 800+ pixels (prevents pips being detected as dice)
        params.maxArea = self.max_dice_area
        params.filterByCircularity = True
        params.minCircularity = self.min_circularity
        params.filterByConvexity = True
        params.minConvexity = self.min_convexity
        self.blob_detector = cv2.SimpleBlobDetector_create(params)
    
    def detect_dice(self, frame: np.ndarray) -> List[dict]:
        """
        Detect dice in frame using computer vision.
        Returns list of detected dice with their properties.
        """
        if frame is None:
            return []
        
        # Store frame dimensions for edge filtering
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Light preprocessing - avoid over-smoothing based on debug results
        # Skip CLAHE since it reduced contrast in debug
        filtered = cv2.bilateralFilter(gray, 5, 50, 50)  # Lighter filtering
        
        # Try multiple detection methods and combine results
        contour_dice = self._detect_by_contours(filtered)
        blob_dice = self._detect_by_blobs(filtered)
        
        # Combine and deduplicate detections
        all_detections = contour_dice + blob_dice
        
        # Filter out edge detections (containers, partial objects)
        edge_filtered = self._filter_edge_detections(all_detections)
        
        # Final deduplication
        final_detections = self._deduplicate_detections(edge_filtered)
        
        return final_detections
    
    def _is_too_close_to_edge(self, bbox: List[int], center: Tuple[int, int]) -> bool:
        """Check if detection is too close to frame edges (likely container/partial object)."""
        x, y, w, h = bbox
        center_x, center_y = center
        
        # Check if bounding box or center is too close to any edge
        if (x < self.edge_margin or 
            y < self.edge_margin or 
            x + w > self.frame_width - self.edge_margin or 
            y + h > self.frame_height - self.edge_margin or
            center_x < self.edge_margin or 
            center_y < self.edge_margin or
            center_x > self.frame_width - self.edge_margin or 
            center_y > self.frame_height - self.edge_margin):
            return True
        
        return False
    
    def _filter_edge_detections(self, detections: List[dict]) -> List[dict]:
        """Remove detections that are too close to frame edges."""
        filtered = []
        
        for detection in detections:
            if not self._is_too_close_to_edge(detection['bbox'], detection['center']):
                filtered.append(detection)
            # Debug info for edge filtering
            else:
                center = detection['center']
                method = detection['method']
                print(f"  Filtered out {method} detection at ({center[0]}, {center[1]}) - too close to edge")
        
        return filtered
    
    def _detect_by_contours(self, gray: np.ndarray) -> List[dict]:
        """Detect dice using contour detection (relaxed for real-world conditions)."""
        detections = []
        
        # Edge detection with conservative parameters
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (prevent pips being detected as dice)
            if not (self.min_dice_area < area < self.max_dice_area):
                continue
            
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # RELAXED: Much more permissive constraints for real dice
            
            # 1. Aspect ratio check (very permissive for angled views)
            aspect_ratio = float(w) / h
            if not (0.2 <= aspect_ratio <= 5.0):  # Very wide range for extreme angles
                continue
            
            # 2. REMOVED strict solidity check - was filtering rounded dice
            # 3. REMOVED strict extent check - was filtering angled dice
            
            # If it passes basic size and aspect tests, it's likely a dice
            dice_region = gray[y:y+h, x:x+w]
            dice_value = self._estimate_dice_value(dice_region)
            
            detections.append({
                'method': 'contour',
                'bbox': [x, y, w, h],
                'center': (x + w//2, y + h//2),
                'value': dice_value,
                'confidence': 0.7,
                'area': area
            })
        
        return detections
    
    def _detect_by_blobs(self, gray: np.ndarray) -> List[dict]:
        """Detect dice using blob detection (showed consistent results in debug)."""
        detections = []
        
        # Detect blobs
        keypoints = self.blob_detector.detect(gray)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            
            # Create bounding box
            bbox_x = max(0, x - radius)
            bbox_y = max(0, y - radius)
            bbox_w = min(gray.shape[1] - bbox_x, 2 * radius)
            bbox_h = min(gray.shape[0] - bbox_y, 2 * radius)
            
            # Extract dice region for value estimation
            dice_region = gray[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
            dice_value = self._estimate_dice_value(dice_region)
            
            detections.append({
                'method': 'blob',
                'bbox': [bbox_x, bbox_y, bbox_w, bbox_h],
                'center': (x, y),
                'value': dice_value,
                'confidence': 0.6,  # Moderate confidence for CV
                'area': np.pi * radius * radius
            })
        
        return detections
    
    def _estimate_dice_value(self, dice_region: np.ndarray) -> int:
        """
        Estimate dice value by analyzing dots/pips in the dice region.
        ENHANCED: Fixes value 6 detection catastrophe and value 1 over-detection.
        UPDATED: Adds specific value 4 detection to fix 1/2 toggle issue.
        REFINED: Improved priority logic to distinguish 4 vs 6.
        """
        if dice_region.size == 0:
            return 1
        
        # Get confidence scores for multiple patterns
        value_6_confidence = self._detect_value_6_pattern(dice_region)
        value_4_confidence = self._detect_value_4_pattern(dice_region)
        
        # PRIORITY 1: Check for value 6 pattern (3x2 grid) with higher threshold
        if value_6_confidence > 0.8 and value_6_confidence > value_4_confidence:
            return 6
        
        # PRIORITY 2: Check for value 4 pattern (2x2 grid) 
        if value_4_confidence > 0.7:
            return 4
        
        # PRIORITY 3: Check for value 6 with lower threshold if 4 is not confident
        if value_6_confidence > 0.7 and value_4_confidence < 0.5:
            return 6
        
        # PRIORITY 4: Check for value 1 over-detection issue (was 41.6% over-detected)
        if self._is_likely_single_pip(dice_region):
            return 1
        
        # PRIORITY 5: Use enhanced multi-method approach for other values
        return self._enhanced_multi_method_estimation(dice_region)
    
    def _detect_value_6_pattern(self, dice_region: np.ndarray) -> float:
        """
        Specifically detect the value 6 pattern (3x2 grid of pips).
        Returns confidence score 0.0-1.0.
        ENHANCED: More specific to avoid confusion with value 4 (2x2).
        """
        h, w = dice_region.shape
        
        # METHOD 1: Check for exactly 6 distinct circular regions (not 4!)
        dots_found = self._find_all_circular_dots(dice_region)
        if len(dots_found) == 6:
            # Verify they form a 3x2 grid pattern
            if self._is_3x2_grid_pattern(dots_found, w, h):
                return 0.95  # High confidence for perfect 3x2 grid
        elif len(dots_found) == 4:
            # 4 dots should NOT be considered value 6
            return 0.1  # Very low confidence
        
        # METHOD 2: Check for distinctive 3x2 spatial pattern
        pattern_score = self._analyze_3x2_spatial_pattern(dice_region)
        if pattern_score > 0.7:
            return pattern_score
        
        # METHOD 3: Template matching for value 6 (requires 6+ regions)
        template_score = self._template_match_value_6(dice_region)
        
        return max(0.0, template_score)
    
    def _find_all_circular_dots(self, dice_region: np.ndarray) -> list:
        """Find all circular dot-like regions in the dice using multiple methods."""
        potential_dots = []
        
        # Method 1: Adaptive threshold
        try:
            adaptive = cv2.adaptiveThreshold(
                dice_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            dots_adaptive = self._extract_dots_from_binary_enhanced(adaptive, dice_region.size)
            potential_dots.extend(dots_adaptive)
        except:
            pass
        
        # Method 2: OTSU threshold
        try:
            _, otsu = cv2.threshold(dice_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dots_otsu = self._extract_dots_from_binary_enhanced(otsu, dice_region.size)
            potential_dots.extend(dots_otsu)
        except:
            pass
        
        # Method 3: Inverted OTSU (for white pips on dark dice)
        try:
            _, otsu_inv = cv2.threshold(dice_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dots_inv = self._extract_dots_from_binary_enhanced(otsu_inv, dice_region.size)
            potential_dots.extend(dots_inv)
        except:
            pass
        
        # Deduplicate dots that are too close
        return self._deduplicate_dots(potential_dots)
    
    def _extract_dots_from_binary_enhanced(self, binary: np.ndarray, dice_area: int) -> list:
        """Extract circular dots from binary image with enhanced parameters."""
        # Clean up noise more aggressively
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        # Additional noise reduction
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        # ENHANCED: More strict parameters to avoid noise detection
        min_dot_area = 20   # Increased from 8 to filter noise
        max_dot_area = 200  # Reasonable maximum
        relative_max = dice_area // 25  # More restrictive relative size
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_dot_area <= area <= min(max_dot_area, relative_max):
                # Check circularity (more strict for noise reduction)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.4:  # More strict circularity requirement
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
        min_distance = 12  # Minimum distance between dot centers
        
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
        row1 = centers_by_y[:3]
        row2 = centers_by_y[3:]
        
        # Check if each row has 3 dots roughly aligned horizontally
        row1_y_var = np.var([p[1] for p in row1])
        row2_y_var = np.var([p[1] for p in row2])
        
        # Allow some variance but not too much
        max_y_variance = height * 0.15  # Slightly more lenient
        
        if row1_y_var < max_y_variance and row2_y_var < max_y_variance:
            # Check if rows are vertically separated
            row1_avg_y = np.mean([p[1] for p in row1])
            row2_avg_y = np.mean([p[1] for p in row2])
            
            vertical_separation = abs(row2_avg_y - row1_avg_y)
            min_separation = height * 0.15
            
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
        
        # Check if each region has a dark spot (pip) - MORE STRICT
        dark_regions = 0
        overall_mean = np.mean(dice_region)
        
        for region in regions:
            region_min = np.min(region)
            region_mean = np.mean(region)
            
            # ENHANCED: More strict criteria for value 6 detection
            # Require both strong minimum darkness AND significant mean darkness
            has_dark_pip = (
                region_min < overall_mean * 0.5 and  # Very dark minimum (was 0.65)
                region_mean < overall_mean * 0.8     # Moderately dark mean (was 0.85)
            )
            
            if has_dark_pip:
                dark_regions += 1
        
        # Value 6 should have dark spots in ALL 6 regions (strict requirement)
        if dark_regions == 6:
            confidence = 1.0
        elif dark_regions == 5:
            confidence = 0.7  # Might be missing one pip due to lighting
        elif dark_regions == 4:
            confidence = 0.1  # Likely value 4, not 6
        else:
            confidence = 0.0
            
        return confidence
    
    def _template_match_value_6(self, dice_region: np.ndarray) -> float:
        """Simple template matching for value 6 pattern."""
        h, w = dice_region.shape
        
        # Check overall darkness (more pips = darker overall)
        avg_brightness = np.mean(dice_region)
        brightness_score = 1.0 - (avg_brightness / 255.0)
        
        # Check for multiple distinct dark regions
        _, binary = cv2.threshold(dice_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        distinct_regions = len([c for c in contours if cv2.contourArea(c) > 15])
        
        # Value 6 should have 5-6 regions (strict requirement)
        if distinct_regions >= 5:
            region_score = min(1.0, distinct_regions / 6.0)
        elif distinct_regions == 4:
            region_score = 0.2  # Low score for 4 regions (likely value 4)
        else:
            region_score = 0.0
        
        # Combine scores - require both good brightness AND sufficient regions
        template_score = (brightness_score * 0.4 + region_score * 0.6)
        
        return template_score
    
    def _detect_value_4_pattern(self, dice_region: np.ndarray) -> float:
        """
        Specifically detect the value 4 pattern (2x2 grid of pips).
        Returns confidence score 0.0-1.0.
        Fixes toggle between 1 and 2 detections.
        """
        h, w = dice_region.shape
        
        # METHOD 1: Check for exactly 4 distinct circular regions
        dots_found = self._find_all_circular_dots(dice_region)
        if len(dots_found) == 4:
            # Verify they form a 2x2 grid pattern
            if self._is_2x2_grid_pattern(dots_found, w, h):
                return 0.95  # High confidence for perfect 2x2 grid
        
        # METHOD 2: Check for 2x2 spatial pattern characteristics
        pattern_score = self._analyze_2x2_spatial_pattern(dice_region)
        if pattern_score > 0.7:
            return pattern_score
        
        # METHOD 3: Template matching for value 4 (moderate darkness, 4 regions)
        template_score = self._template_match_value_4(dice_region)
        
        return max(0.0, template_score)
    
    def _is_2x2_grid_pattern(self, dots: list, width: int, height: int) -> bool:
        """Check if 4 dots form a 2x2 grid pattern (value 4)."""
        if len(dots) != 4:
            return False
        
        # Extract centers
        centers = [dot['center'] for dot in dots]
        
        # Sort by Y coordinate to find potential rows
        centers_by_y = sorted(centers, key=lambda p: p[1])
        
        # For 2x2, we need 2 dots in top half and 2 in bottom half
        mid_y = height / 2
        top_dots = [p for p in centers if p[1] < mid_y]
        bottom_dots = [p for p in centers if p[1] >= mid_y]
        
        if len(top_dots) != 2 or len(bottom_dots) != 2:
            return False
        
        # Check horizontal alignment within each row
        top_y_var = abs(top_dots[0][1] - top_dots[1][1])
        bottom_y_var = abs(bottom_dots[0][1] - bottom_dots[1][1])
        
        max_y_variance = height * 0.2  # Allow some variance
        
        if top_y_var < max_y_variance and bottom_y_var < max_y_variance:
            # Check vertical alignment (left and right columns)
            left_dots = sorted(centers, key=lambda p: p[0])[:2]
            right_dots = sorted(centers, key=lambda p: p[0])[2:]
            
            left_x_var = abs(left_dots[0][0] - left_dots[1][0])
            right_x_var = abs(right_dots[0][0] - right_dots[1][0])
            
            max_x_variance = width * 0.2
            
            if left_x_var < max_x_variance and right_x_var < max_x_variance:
                return True
        
        return False
    
    def _analyze_2x2_spatial_pattern(self, dice_region: np.ndarray) -> float:
        """Analyze spatial patterns specific to value 4 (2x2 grid)."""
        h, w = dice_region.shape
        
        # Divide dice into 4 quadrants
        top_left = dice_region[0:h//2, 0:w//2]
        top_right = dice_region[0:h//2, w//2:w]
        bottom_left = dice_region[h//2:h, 0:w//2]
        bottom_right = dice_region[h//2:h, w//2:w]
        
        quadrants = [top_left, top_right, bottom_left, bottom_right]
        
        # Each quadrant should have a dark pip
        pip_scores = []
        for quad in quadrants:
            if quad.size == 0:
                pip_scores.append(0.0)
                continue
            
            # Look for dark circular region in center of quadrant
            qh, qw = quad.shape
            center_region = quad[qh//3:2*qh//3, qw//3:2*qw//3]
            
            if center_region.size == 0:
                pip_scores.append(0.0)
                continue
            
            # Check if center is darker than edges
            center_mean = np.mean(center_region)
            edge_mean = np.mean([
                np.mean(quad[0:qh//4, :]) if qh//4 > 0 else center_mean,
                np.mean(quad[3*qh//4:qh, :]) if qh - 3*qh//4 > 0 else center_mean,
                np.mean(quad[:, 0:qw//4]) if qw//4 > 0 else center_mean,
                np.mean(quad[:, 3*qw//4:qw]) if qw - 3*qw//4 > 0 else center_mean
            ])
            
            contrast = edge_mean - center_mean
            pip_score = min(1.0, max(0.0, contrast / 30.0))  # Normalize contrast
            pip_scores.append(pip_score)
        
        # All 4 quadrants should have pips for value 4
        avg_pip_score = np.mean(pip_scores)
        min_pip_score = min(pip_scores)
        
        # Require all quadrants to have some pip evidence
        if min_pip_score > 0.3 and avg_pip_score > 0.6:
            return avg_pip_score * 0.9  # High confidence
        elif min_pip_score > 0.2 and avg_pip_score > 0.5:
            return avg_pip_score * 0.7  # Moderate confidence
        else:
            return 0.0
    
    def _template_match_value_4(self, dice_region: np.ndarray) -> float:
        """Template matching for value 4 pattern (moderate darkness, 4 distinct regions)."""
        h, w = dice_region.shape
        
        # Check overall darkness (4 pips = moderately dark)
        avg_brightness = np.mean(dice_region)
        # ENHANCED: Expand acceptable brightness range for real dice
        if 180 <= avg_brightness <= 230:
            # Light dice (common case)
            brightness_score = 1.0 - abs(avg_brightness - 205) / 25.0
        elif 140 <= avg_brightness <= 179:
            # Medium dice
            brightness_score = 1.0 - abs(avg_brightness - 160) / 20.0
        else:
            # Too dark or too light
            brightness_score = 0.0
        
        brightness_score = max(0.0, brightness_score)
        
        # Check for 4 distinct dark regions
        _, binary = cv2.threshold(dice_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant contours
        significant_regions = []
        min_area = (h * w) // 100  # Reduced threshold for better detection
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                significant_regions.append(contour)
        
        # Should have around 4 regions for value 4
        region_count = len(significant_regions)
        if region_count == 4:
            region_score = 1.0
        elif region_count == 3:
            region_score = 0.8  # Close but might be missed pip
        elif region_count == 5:
            region_score = 0.6  # Might be split pip
        elif region_count == 2:
            region_score = 0.4  # Some pips detected
        else:
            region_score = 0.0
        
        # Combine scores with emphasis on brightness for light dice
        template_score = (brightness_score * 0.7 + region_score * 0.3)
        
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
        return contrast > 25
    
    def _enhanced_multi_method_estimation(self, dice_region: np.ndarray) -> int:
        """Enhanced multi-method estimation for values 2-5."""
        # Try multiple approaches for robust detection
        dot_counts = []
        
        # Approach 1: Enhanced dot counting
        dots = self._find_all_circular_dots(dice_region)
        dot_count = len(dots)
        if 1 <= dot_count <= 6:
            dot_counts.append(dot_count)
        
        # Approach 2: Edge-based detection
        edges = cv2.Canny(dice_region, 30, 100)
        edge_count = self._count_dots_from_edges(edges, dice_region.size)
        if 1 <= edge_count <= 6:
            dot_counts.append(edge_count)
        
        # Approach 3: Multiple thresholding methods
        thresholding_counts = self._try_thresholding_methods(dice_region)
        dot_counts.extend(thresholding_counts)
        
        # Find consensus from valid counts
        if dot_counts:
            # Use most common count, or median if no clear winner
            final_count = max(set(dot_counts), key=dot_counts.count)
        else:
            # Fallback to brightness analysis
            final_count = self._brightness_fallback(dice_region)
        
        return max(1, min(6, final_count))
    
    def _count_dots_from_edges(self, edges: np.ndarray, dice_face_area: int) -> int:
        """Count dots using edge detection - works better for colored dice."""
        # Find circular contours in edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_dot_area = 10  # Smaller minimum for edge detection
        max_dot_area = 100  # Larger maximum for better coverage
        relative_max_area = dice_face_area // 30
        
        dot_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            max_area_limit = min(max_dot_area, relative_max_area)
            
            if min_dot_area < area < max_area_limit:
                # Check if roughly circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.2:  # Very lenient for edge detection
                        dot_count += 1
        
        return dot_count
    
    def _try_thresholding_methods(self, dice_region: np.ndarray) -> list:
        """Try multiple thresholding approaches and return valid counts."""
        valid_counts = []
        
        # Method 1: OTSU thresholding
        _, binary_otsu = cv2.threshold(dice_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        count = self._count_dots_in_binary(binary_otsu, dice_region.size)
        if 1 <= count <= 6:
            valid_counts.append(count)
        
        # Try inverted OTSU
        count_inv = self._count_dots_in_binary(cv2.bitwise_not(binary_otsu), dice_region.size)
        if 1 <= count_inv <= 6:
            valid_counts.append(count_inv)
        
        # Method 2: Adaptive threshold
        binary_adaptive = cv2.adaptiveThreshold(
            dice_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        count = self._count_dots_in_binary(binary_adaptive, dice_region.size)
        if 1 <= count <= 6:
            valid_counts.append(count)
        
        return valid_counts
    
    def _count_dots_in_binary(self, binary: np.ndarray, dice_face_area: int) -> int:
        """Count dots in a binary image using the optimized parameters."""
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours of potential dots
        dot_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # OPTIMIZED: Based on real dice dimensions (12mm-16mm cubes)
        min_dot_area = 15  # ~3-4 pixel radius (real pip size)
        max_dot_area = 80  # ~5-6 pixel radius (maximum pip size)
        relative_max_area = dice_face_area // 40  # Pip should be <1/40 of dice face
        
        dot_count = 0
        for contour in dot_contours:
            area = cv2.contourArea(contour)
            max_area_limit = min(max_dot_area, relative_max_area)
            
            if min_dot_area < area < max_area_limit:
                # Check if contour is roughly circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.35:
                        dot_count += 1
        
        return dot_count
    
    def _brightness_fallback(self, dice_region: np.ndarray) -> int:
        """Fallback estimation based on brightness and contrast."""
        avg_brightness = np.mean(dice_region)
        brightness_std = np.std(dice_region)
        
        # Higher contrast usually means more dots
        if brightness_std > 60:
            if avg_brightness < 100:
                return 6
            elif avg_brightness < 130:
                return 5  
            else:
                return 4
        elif brightness_std > 40:
            if avg_brightness < 120:
                return 3
            else:
                return 2
        else:
            return 1
    
    def _deduplicate_detections(self, detections: List[dict]) -> List[dict]:
        """Remove duplicate detections that are too close to each other."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence descending
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        min_distance = 50  # Minimum distance between dice centers
        
        for detection in detections:
            # Check if this detection is too close to any already accepted detection
            too_close = False
            for accepted in final_detections:
                dx = detection['center'][0] - accepted['center'][0]
                dy = detection['center'][1] - accepted['center'][1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                final_detections.append(detection)
        
        return final_detections 