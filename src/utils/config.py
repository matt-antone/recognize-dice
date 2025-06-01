"""
Configuration settings for D6 Dice Recognition App
Optimized for Raspberry Pi 3 constraints
"""

import os


class Config:
    """Configuration settings for the dice recognition application."""
    
    def __init__(self):
        # Hardware-specific settings for Pi 3
        self.hardware_platform = "raspberry_pi_3"
        
        # Camera settings (optimized for Pi 3)
        self.camera_resolution = (320, 320)  # Reduced from 640x640 for Pi 3
        self.camera_format = "RGB888"
        self.camera_framerate = 15  # Max framerate, actual will be lower
        self.camera_rotation = 0
        
        # Model settings
        self.model_input_size = (320, 320)  # Match camera resolution
        self.model_path = os.path.join("models", "dice_detection_pi3.tflite")
        self.labelmap_path = os.path.join("models", "labelmap.txt")
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.max_detections = 6  # Maximum dice to detect simultaneously
        self.nms_threshold = 0.4  # Non-maximum suppression
        
        # Performance optimization for Pi 3
        self.frame_skip = 2  # Process every 2nd frame
        self.max_processing_time = 0.3  # 300ms timeout per frame
        self.memory_limit_mb = 800  # Stay under 800MB total usage
        
        # GUI settings
        self.window_title = "D6 Dice Recognition - Pi 3"
        self.window_size = (800, 600)
        self.display_scale = 1.0
        
        # Logging settings
        self.log_level = "INFO"
        self.log_file = "dice_recognition.log"
        self.enable_file_logging = True
        
        # Thermal management (Pi 3 specific)
        self.thermal_throttle_temp = 80  # °C
        self.enable_thermal_monitoring = True
        
        # Fallback settings
        self.enable_fallback_detection = True
        self.fallback_method = "template_matching"
        
        # Development/debug settings
        self.debug_mode = False
        self.save_debug_images = False
        self.show_fps = True
        self.show_confidence = True
    
    @property
    def model_exists(self):
        """Check if the model file exists."""
        return os.path.exists(self.model_path)
    
    @property
    def labelmap_exists(self):
        """Check if the labelmap file exists."""
        return os.path.exists(self.labelmap_path)
    
    def validate_settings(self):
        """Validate configuration settings."""
        errors = []
        
        # Check required files
        if not self.model_exists:
            errors.append(f"Model file not found: {self.model_path}")
        
        if not self.labelmap_exists:
            errors.append(f"Labelmap file not found: {self.labelmap_path}")
        
        # Validate ranges
        if not (0.0 <= self.confidence_threshold <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        if self.frame_skip < 1:
            errors.append("Frame skip must be >= 1")
        
        if self.memory_limit_mb < 100:
            errors.append("Memory limit too low (minimum 100MB)")
        
        return errors
    
    def get_camera_config(self):
        """Get camera-specific configuration dictionary."""
        return {
            "main": {
                "size": self.camera_resolution,
                "format": self.camera_format
            },
            "controls": {
                "FrameRate": self.camera_framerate
            },
            "transform": {"rotation": self.camera_rotation}
        }
    
    def update_for_performance(self, current_fps, cpu_temp=None):
        """Dynamically adjust settings based on performance."""
        # Adjust frame skipping based on FPS
        if current_fps < 2.0:
            self.frame_skip = min(self.frame_skip + 1, 5)
        elif current_fps > 4.0 and self.frame_skip > 1:
            self.frame_skip = max(self.frame_skip - 1, 1)
        
        # Thermal throttling for Pi 3
        if cpu_temp and cpu_temp > self.thermal_throttle_temp:
            self.frame_skip = min(self.frame_skip + 1, 5)
            # Could also reduce resolution further if needed
    
    def __str__(self):
        """String representation of current configuration."""
        return f"""
        D6 Dice Recognition Configuration (Pi 3):
        - Camera Resolution: {self.camera_resolution}
        - Model Input Size: {self.model_input_size}
        - Confidence Threshold: {self.confidence_threshold}
        - Frame Skip: {self.frame_skip}
        - Memory Limit: {self.memory_limit_mb}MB
        - Model Path: {self.model_path}
        """ 