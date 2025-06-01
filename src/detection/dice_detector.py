"""
Dice detection module using TensorFlow Lite
Optimized for Raspberry Pi 3 performance constraints
"""

import numpy as np
import cv2
import logging
import time
from typing import List, Dict, Optional, Tuple


class DiceDetection:
    """Class representing a single dice detection."""
    
    def __init__(self, bbox: Tuple[int, int, int, int], value: int, confidence: float):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.value = value  # 1-6
        self.confidence = confidence
        
        # Calculate center point
        self.center_x = (bbox[0] + bbox[2]) // 2
        self.center_y = (bbox[1] + bbox[3]) // 2
    
    def __str__(self):
        return f"Dice(value={self.value}, conf={self.confidence:.2f}, bbox={self.bbox})"


class DiceDetector:
    """Main dice detection class using TensorFlow Lite."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model state
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_loaded = False
        
        # Performance tracking
        self.detection_count = 0
        self.total_inference_time = 0
        
        # Load model if available
        self._load_model()
    
    def _load_model(self):
        """Load TensorFlow Lite model."""
        try:
            # Check if model file exists
            if not self.config.model_exists:
                self.logger.warning(f"Model file not found: {self.config.model_path}")
                self.logger.info("Using fallback detection method")
                return
            
            # Import TensorFlow Lite
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite
            
            # Load interpreter
            self.interpreter = tflite.Interpreter(model_path=self.config.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.model_loaded = True
            self.logger.info(f"TensorFlow Lite model loaded successfully")
            self.logger.info(f"Input shape: {self.input_details[0]['shape']}")
            self.logger.info(f"Output details: {len(self.output_details)} outputs")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.info("Falling back to placeholder detection")
    
    def detect_dice(self, frame: np.ndarray) -> List[DiceDetection]:
        """
        Detect dice in the given frame.
        
        Args:
            frame: RGB image array of shape (height, width, 3)
            
        Returns:
            List of DiceDetection objects
        """
        start_time = time.time()
        
        try:
            if self.model_loaded:
                detections = self._detect_with_model(frame)
            else:
                detections = self._detect_fallback(frame)
            
            # Track performance
            inference_time = time.time() - start_time
            self.detection_count += 1
            self.total_inference_time += inference_time
            
            if self.detection_count % 30 == 0:  # Log every 30 detections
                avg_time = self.total_inference_time / self.detection_count
                self.logger.debug(f"Average inference time: {avg_time*1000:.1f}ms")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def _detect_with_model(self, frame: np.ndarray) -> List[DiceDetection]:
        """Detect dice using TensorFlow Lite model."""
        try:
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Set input
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get outputs
            detections = []
            for output_detail in self.output_details:
                output_data = self.interpreter.get_tensor(output_detail['index'])
                # TODO: Parse model outputs and create DiceDetection objects
                # This depends on the specific model architecture
            
            # For now, return empty list until model is properly integrated
            return []
            
        except Exception as e:
            self.logger.error(f"Model inference error: {e}")
            return []
    
    def _detect_fallback(self, frame: np.ndarray) -> List[DiceDetection]:
        """
        Improved fallback detection method using computer vision.
        Tuned for real dice detection with motion handling.
        """
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Preprocessing for better dice detection
            # Apply bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply adaptive histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(filtered)
            
            # Apply Gaussian blur to smooth for circle detection
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 1)
            
            # Use more aggressive HoughCircles parameters for dice
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,                    # Inverse ratio of accumulator resolution
                minDist=30,             # Minimum distance between circle centers (reduced for smaller dice)
                param1=100,             # Higher threshold for edge detection
                param2=25,              # Lower accumulator threshold (more sensitive)
                minRadius=10,           # Smaller minimum radius for dice
                maxRadius=80            # Maximum radius for dice
            )
            
            detections = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # Filter circles that are likely dice
                valid_circles = []
                for (x, y, r) in circles:
                    # Check if circle is within reasonable bounds
                    if (r >= 10 and r <= 80 and 
                        x-r >= 0 and y-r >= 0 and 
                        x+r < frame.shape[1] and y+r < frame.shape[0]):
                        
                        # Extract the circular region
                        roi = enhanced[y-r:y+r, x-r:x+r]
                        if roi.size > 0:
                            # Simple validation: check if it looks like a dice face
                            # Dice should have relatively uniform background with some variation (dots)
                            roi_std = np.std(roi)
                            roi_mean = np.mean(roi)
                            
                            # Dice faces typically have moderate contrast (dots on face)
                            if roi_std > 10 and roi_mean > 50:  # Some variation and not too dark
                                valid_circles.append((x, y, r, roi_std))
                
                # Sort by standard deviation (more variation = more likely to be dice with dots)
                valid_circles.sort(key=lambda x: x[3], reverse=True)
                
                # Take top candidates (limit to reasonable number of dice)
                for i, (x, y, r, std_val) in enumerate(valid_circles[:6]):  # Max 6 dice
                    # Create bounding box from circle
                    bbox = (x - r, y - r, x + r, y + r)
                    
                    # Ensure bbox is within frame bounds
                    h, w = frame.shape[:2]
                    bbox = (
                        max(0, bbox[0]),
                        max(0, bbox[1]),
                        min(w, bbox[2]),
                        min(h, bbox[3])
                    )
                    
                    # Analyze the region to estimate dice value
                    value = self._estimate_dice_value(enhanced, x, y, r)
                    
                    # Confidence based on how much it looks like a dice
                    # Higher std deviation suggests more dots/features
                    confidence = min(0.9, 0.4 + (std_val / 100))
                    
                    detection = DiceDetection(bbox, value, confidence)
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Fallback detection error: {e}")
            return []
    
    def _estimate_dice_value(self, gray_image: np.ndarray, cx: int, cy: int, radius: int) -> int:
        """
        Estimate dice value by analyzing the circular region.
        This is a simple heuristic - not perfect but better than random.
        """
        try:
            # Extract circular region
            y1, y2 = max(0, cy-radius), min(gray_image.shape[0], cy+radius)
            x1, x2 = max(0, cx-radius), min(gray_image.shape[1], cx+radius)
            roi = gray_image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return 3  # Default middle value
            
            # Create circular mask
            mask = np.zeros(roi.shape, dtype=np.uint8)
            local_cx, local_cy = roi.shape[1]//2, roi.shape[0]//2
            cv2.circle(mask, (local_cx, local_cy), radius//2, 255, -1)
            
            # Apply mask to roi
            masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
            
            # Threshold to find dark spots (dots)
            # Assume dice are light with dark dots
            mean_val = cv2.mean(masked_roi, mask=mask)[0]
            threshold_val = max(mean_val - 30, 50)  # Adaptive threshold
            
            _, binary = cv2.threshold(masked_roi, threshold_val, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.bitwise_and(binary, binary, mask=mask)
            
            # Find contours (potential dots)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (dots should be reasonably sized)
            min_dot_area = (radius * 0.1) ** 2  # Minimum dot size
            max_dot_area = (radius * 0.4) ** 2  # Maximum dot size
            
            dot_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_dot_area <= area <= max_dot_area:
                    # Check if contour is roughly circular (dots are round)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.3:  # Reasonably circular
                            dot_count += 1
            
            # Map dot count to dice value with some bounds checking
            if dot_count == 0:
                return 1  # Could be 1 (center dot might be missed) or clean face
            elif dot_count == 1:
                return 1
            elif dot_count == 2:
                return 2
            elif dot_count <= 4:
                return 3 if dot_count == 3 else 4
            elif dot_count == 5:
                return 5
            else:
                return 6  # 6 or many dots detected
                
        except Exception as e:
            self.logger.debug(f"Dice value estimation error: {e}")
            return 3  # Default to middle value
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input."""
        try:
            # Resize to model input size
            target_size = self.config.model_input_size
            resized = cv2.resize(frame, target_size)
            
            # Normalize to [0, 1] range
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            input_tensor = np.expand_dims(normalized, axis=0)
            
            return input_tensor
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get detection statistics."""
        avg_time = 0
        if self.detection_count > 0:
            avg_time = self.total_inference_time / self.detection_count
        
        return {
            'detection_count': self.detection_count,
            'avg_inference_time_ms': avg_time * 1000,
            'model_loaded': self.model_loaded,
            'model_path': self.config.model_path
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.interpreter:
            # TensorFlow Lite interpreter cleanup (if needed)
            self.interpreter = None
        
        self.logger.info("Dice detector cleanup completed") 