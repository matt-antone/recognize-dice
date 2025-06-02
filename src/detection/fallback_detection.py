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
        Improved for colored dice, rounded dice, and angled views.
        """
        if dice_region.size == 0:
            return 1
        
        # Try multiple approaches for robust detection
        dot_counts = []
        
        # Approach 1: Edge-based detection (works better for colored dice)
        edges = cv2.Canny(dice_region, 30, 100)
        dot_count_edges = self._count_dots_from_edges(edges, dice_region.size)
        if 1 <= dot_count_edges <= 6:
            dot_counts.append(dot_count_edges)
        
        # Approach 2: Multiple thresholding (for traditional dice)
        thresholding_counts = self._try_thresholding_methods(dice_region)
        dot_counts.extend(thresholding_counts)
        
        # Approach 3: Template matching for common pip patterns
        template_count = self._estimate_from_pattern(dice_region)
        if 1 <= template_count <= 6:
            dot_counts.append(template_count)
        
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
    
    def _estimate_from_pattern(self, dice_region: np.ndarray) -> int:
        """Estimate dice value from overall pattern and distribution."""
        # Analyze the spatial distribution of dark/bright regions
        h, w = dice_region.shape
        
        # Divide dice face into regions and analyze distribution
        center_region = dice_region[h//3:2*h//3, w//3:2*w//3]
        corner_regions = [
            dice_region[0:h//3, 0:w//3],      # Top-left
            dice_region[0:h//3, 2*w//3:w],    # Top-right  
            dice_region[2*h//3:h, 0:w//3],    # Bottom-left
            dice_region[2*h//3:h, 2*w//3:w]   # Bottom-right
        ]
        
        # Count regions with significant dark spots
        threshold = np.mean(dice_region) * 0.7
        
        regions_with_dots = 0
        if np.min(center_region) < threshold:
            regions_with_dots += 1
        
        for corner in corner_regions:
            if corner.size > 0 and np.min(corner) < threshold:
                regions_with_dots += 1
        
        # Map region patterns to dice values
        if regions_with_dots <= 1:
            return 1
        elif regions_with_dots == 2:
            return 2
        elif regions_with_dots == 3:
            return 3
        elif regions_with_dots == 4:
            return 4
        elif regions_with_dots == 5:
            return 5
        else:
            return 6
    
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