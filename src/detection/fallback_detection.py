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
        self.min_dice_area = 100
        self.max_dice_area = 5000
        self.min_circularity = 0.3
        self.min_convexity = 0.5
        
        # Edge filtering - exclude objects too close to frame boundaries
        self.edge_margin = 30  # Minimum distance from frame edge
        
        # Blob detector setup (showed good results in debug)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self.min_dice_area
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
        """Detect dice using contour detection (improved for angled views and colors)."""
        detections = []
        
        # Edge detection with conservative parameters
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if not (self.min_dice_area < area < self.max_dice_area):
                continue
            
            # Approximate contour to polygon
            epsilon = 0.03 * cv2.arcLength(contour, True)  # More lenient approximation
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # IMPROVED: Accept more shapes for angled dice views
            if 3 <= len(approx) <= 10:  # Triangles to octagons (angled dice can be any of these)
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # RELAXED: Much more permissive aspect ratio for angled views
                aspect_ratio = float(w) / h
                if 0.3 <= aspect_ratio <= 3.3:  # Allow more distortion from perspective
                    # Estimate dice value by analyzing the region
                    dice_region = gray[y:y+h, x:x+w]
                    dice_value = self._estimate_dice_value(dice_region)
                    
                    detections.append({
                        'method': 'contour',
                        'bbox': [x, y, w, h],
                        'center': (x + w//2, y + h//2),
                        'value': dice_value,
                        'confidence': 0.7,  # Moderate confidence for CV
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
        Improved for colored dice and angled views.
        """
        if dice_region.size == 0:
            return 1
        
        # Try multiple thresholding approaches for colored dice
        dot_counts = []
        
        # Method 1: OTSU thresholding (works for black/white dice)
        _, binary_otsu = cv2.threshold(dice_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dot_count_otsu = self._count_dots_in_binary(binary_otsu, dice_region.size)
        dot_counts.append(('otsu', dot_count_otsu))
        
        # Method 2: Fixed threshold (works for colored dice with consistent lighting)
        mean_brightness = np.mean(dice_region)
        threshold_val = int(mean_brightness * 0.8)  # 80% of mean brightness
        _, binary_fixed = cv2.threshold(dice_region, threshold_val, 255, cv2.THRESH_BINARY)
        dot_count_fixed = self._count_dots_in_binary(binary_fixed, dice_region.size)
        dot_counts.append(('fixed', dot_count_fixed))
        
        # Method 3: Adaptive threshold (works for varying lighting)
        binary_adaptive = cv2.adaptiveThreshold(
            dice_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        dot_count_adaptive = self._count_dots_in_binary(binary_adaptive, dice_region.size)
        dot_counts.append(('adaptive', dot_count_adaptive))
        
        # Also try inverted versions for each method
        for method_name, count in dot_counts.copy():
            if method_name == 'otsu':
                binary_inv = cv2.bitwise_not(binary_otsu)
            elif method_name == 'fixed':
                binary_inv = cv2.bitwise_not(binary_fixed)
            else:  # adaptive
                binary_inv = cv2.bitwise_not(binary_adaptive)
            
            dot_count_inv = self._count_dots_in_binary(binary_inv, dice_region.size)
            dot_counts.append((f'{method_name}_inv', dot_count_inv))
        
        # Find the most reasonable count (1-6 range)
        valid_counts = [count for name, count in dot_counts if 1 <= count <= 6]
        
        if valid_counts:
            # Use the most common valid count, or median if tied
            final_count = max(set(valid_counts), key=valid_counts.count)
        else:
            # All methods failed, use improved brightness/pattern fallback
            avg_brightness = np.mean(dice_region)
            brightness_std = np.std(dice_region)
            
            # IMPROVED: Better fallback for colored dice
            if brightness_std > 60:  # High contrast - likely multiple dots
                if avg_brightness < 100:  # Dark overall - many dark dots
                    final_count = 6
                elif avg_brightness < 130:
                    final_count = 5  
                else:
                    final_count = 4
            elif brightness_std > 40:  # Medium contrast
                if avg_brightness < 120:
                    final_count = 3
                else:
                    final_count = 2
            else:  # Low contrast - likely single dot or empty
                final_count = 1
        
        return max(1, min(6, final_count))
    
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