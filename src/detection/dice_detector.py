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
        Fallback detection method using simple computer vision.
        This provides basic functionality while we work on the ML model.
        """
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Use HoughCircles to detect circular shapes (dice)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=100
            )
            
            detections = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
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
                    
                    # For fallback, assign random value and moderate confidence
                    # In a real implementation, this would analyze the dice face
                    import random
                    value = random.randint(1, 6)
                    confidence = 0.6  # Moderate confidence for fallback
                    
                    detection = DiceDetection(bbox, value, confidence)
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Fallback detection error: {e}")
            return []
    
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