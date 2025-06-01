"""
Camera interface for Raspberry Pi with optimizations for Pi 3
Supports both picamera2 (modern) and picamera (legacy) libraries
"""

import numpy as np
import cv2
import logging
import time
import threading
from typing import Optional, Tuple


class CameraInterface:
    """Camera interface optimized for Raspberry Pi 3."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Camera instance
        self.camera = None
        self.camera_type = None  # 'picamera2' or 'picamera'
        
        # State
        self.is_running = False
        self.last_frame = None
        self.frame_lock = threading.Lock()
        
        # Performance tracking
        self.frame_count = 0
        self.capture_errors = 0
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera with fallback support."""
        # Try picamera2 first (modern)
        if self._try_picamera2():
            self.camera_type = 'picamera2'
            self.logger.info("Using picamera2 interface")
            return
        
        # Fallback to picamera (legacy)
        if self._try_picamera():
            self.camera_type = 'picamera'
            self.logger.info("Using picamera interface")
            return
        
        # No camera available
        raise RuntimeError("No compatible camera interface found. "
                         "Please install picamera2 or picamera library.")
    
    def _try_picamera2(self):
        """Try to initialize picamera2."""
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            
            # For the new Raspberry Pi AI Camera (IMX500), we need special handling
            self.logger.info("Detected IMX500 AI Camera - using AI camera configuration")
            
            # Try multiple configuration approaches for AI camera
            configurations_to_try = [
                # Approach 1: AI camera specific configuration
                lambda: self.camera.create_preview_configuration(
                    main={"size": self.config.camera_resolution, "format": "RGB888"},
                    raw={"size": self.camera.sensor_resolution}
                ),
                # Approach 2: Simple preview without raw specification
                lambda: self.camera.create_preview_configuration(
                    main={"size": self.config.camera_resolution, "format": "RGB888"}
                ),
                # Approach 3: Use default configuration and modify
                lambda: self._create_ai_camera_config(),
                # Approach 4: Minimal configuration
                lambda: self.camera.create_preview_configuration()
            ]
            
            for i, config_func in enumerate(configurations_to_try):
                try:
                    self.logger.info(f"Trying AI camera configuration approach {i+1}...")
                    config = config_func()
                    self.camera.configure(config)
                    self.logger.info(f"AI camera initialized successfully with approach {i+1}")
                    self.logger.info(f"Camera resolution: {self.config.camera_resolution}")
                    return True
                except Exception as e:
                    self.logger.warning(f"AI camera configuration approach {i+1} failed: {e}")
                    continue
            
            # If all configurations failed
            self.logger.error("All AI camera configuration approaches failed")
            return False
            
        except ImportError:
            self.logger.debug("picamera2 not available")
            return False
        except Exception as e:
            self.logger.warning(f"Failed to initialize picamera2: {e}")
            return False
    
    def _create_ai_camera_config(self):
        """Create configuration specific for AI camera."""
        try:
            # Get the default configuration first
            config = self.camera.create_preview_configuration()
            
            # Modify for our needs without specifying raw
            config["main"]["size"] = self.config.camera_resolution
            config["main"]["format"] = "RGB888"
            
            # Remove raw configuration if it exists
            if "raw" in config:
                del config["raw"]
            
            return config
        except Exception as e:
            self.logger.error(f"Failed to create AI camera config: {e}")
            raise
    
    def _try_picamera(self):
        """Try to initialize legacy picamera."""
        try:
            import picamera
            import picamera.array
            
            self.camera = picamera.PiCamera()
            
            # Configure camera for Pi 3 optimization
            self.camera.resolution = self.config.camera_resolution
            self.camera.framerate = self.config.camera_framerate
            self.camera.rotation = self.config.camera_rotation
            
            # Warm up camera
            time.sleep(2)
            
            self.logger.info(f"Legacy picamera initialized with resolution: {self.config.camera_resolution}")
            return True
            
        except ImportError:
            self.logger.debug("picamera not available")
            return False
        except Exception as e:
            self.logger.warning(f"Failed to initialize picamera: {e}")
            return False
    
    def start(self):
        """Start camera capture."""
        if self.is_running:
            self.logger.warning("Camera already running")
            return
        
        try:
            if self.camera_type == 'picamera2':
                self.camera.start()
            elif self.camera_type == 'picamera':
                # picamera doesn't need explicit start for capture
                pass
            
            self.is_running = True
            self.logger.info("Camera started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            raise
    
    def stop(self):
        """Stop camera capture."""
        if not self.is_running:
            return
        
        try:
            if self.camera_type == 'picamera2':
                self.camera.stop()
            elif self.camera_type == 'picamera':
                # picamera cleanup handled in cleanup()
                pass
            
            self.is_running = False
            self.logger.info("Camera stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping camera: {e}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from camera.
        
        Returns:
            numpy.ndarray: RGB frame with shape (height, width, 3) or None if failed
        """
        if not self.is_running:
            return None
        
        try:
            if self.camera_type == 'picamera2':
                return self._capture_picamera2()
            elif self.camera_type == 'picamera':
                return self._capture_picamera()
            else:
                return None
                
        except Exception as e:
            self.capture_errors += 1
            self.logger.error(f"Frame capture error: {e}")
            return None
    
    def _capture_picamera2(self) -> Optional[np.ndarray]:
        """Capture frame using picamera2."""
        try:
            # Capture array directly
            frame = self.camera.capture_array()
            
            # Ensure RGB format
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Already RGB
                pass
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                # RGBA, convert to RGB
                frame = frame[:, :, :3]
            else:
                self.logger.warning(f"Unexpected frame shape: {frame.shape}")
                return None
            
            # Store last frame for debugging
            with self.frame_lock:
                self.last_frame = frame.copy()
            
            self.frame_count += 1
            return frame
            
        except Exception as e:
            self.logger.error(f"picamera2 capture error: {e}")
            return None
    
    def _capture_picamera(self) -> Optional[np.ndarray]:
        """Capture frame using legacy picamera."""
        try:
            import picamera.array
            
            with picamera.array.PiRGBArray(self.camera, size=self.config.camera_resolution) as output:
                self.camera.capture(output, format='rgb')
                frame = output.array
                
                # Store last frame for debugging
                with self.frame_lock:
                    self.last_frame = frame.copy()
                
                self.frame_count += 1
                return frame
                
        except Exception as e:
            self.logger.error(f"picamera capture error: {e}")
            return None
    
    def get_frame_size(self) -> Tuple[int, int]:
        """Get camera frame size (width, height)."""
        return self.config.camera_resolution
    
    def get_stats(self) -> dict:
        """Get camera statistics."""
        return {
            'frame_count': self.frame_count,
            'capture_errors': self.capture_errors,
            'error_rate': self.capture_errors / max(self.frame_count, 1),
            'camera_type': self.camera_type,
            'is_running': self.is_running
        }
    
    def adjust_settings(self, brightness=None, contrast=None, saturation=None):
        """Adjust camera settings if supported."""
        try:
            if self.camera_type == 'picamera' and self.camera:
                if brightness is not None:
                    self.camera.brightness = max(0, min(100, brightness))
                if contrast is not None:
                    self.camera.contrast = max(-100, min(100, contrast))
                if saturation is not None:
                    self.camera.saturation = max(-100, min(100, saturation))
                    
                self.logger.info(f"Camera settings adjusted: brightness={brightness}, "
                               f"contrast={contrast}, saturation={saturation}")
            else:
                self.logger.debug("Camera setting adjustment not supported for current interface")
                
        except Exception as e:
            self.logger.warning(f"Failed to adjust camera settings: {e}")
    
    def test_capture(self) -> bool:
        """Test camera capture functionality."""
        try:
            self.start()
            frame = self.capture_frame()
            self.stop()
            
            if frame is not None:
                self.logger.info(f"Camera test successful. Frame shape: {frame.shape}")
                return True
            else:
                self.logger.error("Camera test failed - no frame captured")
                return False
                
        except Exception as e:
            self.logger.error(f"Camera test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up camera resources."""
        try:
            self.stop()
            
            if self.camera:
                if self.camera_type == 'picamera2':
                    self.camera.close()
                elif self.camera_type == 'picamera':
                    self.camera.close()
                
                self.camera = None
            
            self.logger.info("Camera cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during camera cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup() 