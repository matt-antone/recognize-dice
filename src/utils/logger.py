"""
Logging utilities for D6 Dice Recognition App
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime


def setup_logger(name, log_level="INFO", log_file="dice_recognition.log", enable_file_logging=True):
    """
    Set up logger with console and file output.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file path
        enable_file_logging: Whether to log to file
    
    Returns:
        logging.Logger: Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Set logging level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if enable_file_logging:
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "."
            os.makedirs(log_dir, exist_ok=True)
            
            # Use rotating file handler to prevent huge log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=3
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")
    
    return logger


def log_system_info(logger):
    """Log system information for debugging."""
    try:
        import platform
        import psutil
        
        logger.info("=== System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"CPU: {platform.processor()}")
        logger.info(f"CPU Count: {psutil.cpu_count()}")
        logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        logger.info("========================")
        
    except ImportError:
        logger.info("psutil not available for system info")
    except Exception as e:
        logger.warning(f"Could not log system info: {e}")


def log_performance_metrics(logger, fps, memory_usage, cpu_temp=None):
    """Log performance metrics."""
    logger.debug(f"Performance - FPS: {fps:.1f}, Memory: {memory_usage:.1f}MB", 
                 extra={"cpu_temp": cpu_temp})


class PerformanceLogger:
    """Class to track and log performance metrics."""
    
    def __init__(self, logger, log_interval=30):
        self.logger = logger
        self.log_interval = log_interval  # seconds
        self.last_log_time = 0
        self.frame_count = 0
        self.total_processing_time = 0
    
    def log_frame_processed(self, processing_time):
        """Log that a frame was processed."""
        self.frame_count += 1
        self.total_processing_time += processing_time
        
        current_time = datetime.now().timestamp()
        
        if current_time - self.last_log_time >= self.log_interval:
            self._log_metrics()
            self.last_log_time = current_time
            self.frame_count = 0
            self.total_processing_time = 0
    
    def _log_metrics(self):
        """Log accumulated metrics."""
        if self.frame_count > 0:
            avg_processing_time = self.total_processing_time / self.frame_count
            fps = self.frame_count / self.log_interval
            
            self.logger.info(f"Performance: {fps:.1f} FPS, "
                           f"Avg processing: {avg_processing_time*1000:.1f}ms")
    
    def log_memory_usage(self):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.logger.debug(f"Memory usage: {memory_mb:.1f}MB")
            return memory_mb
        except ImportError:
            return None
    
    def log_thermal_status(self):
        """Log thermal status (Pi specific)."""
        try:
            # Try to read CPU temperature on Raspberry Pi
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_raw = f.read().strip()
                temp_c = int(temp_raw) / 1000.0
                
                if temp_c > 70:
                    self.logger.warning(f"High CPU temperature: {temp_c:.1f}°C")
                else:
                    self.logger.debug(f"CPU temperature: {temp_c:.1f}°C")
                
                return temp_c
        except FileNotFoundError:
            # Not on Raspberry Pi or thermal zone not available
            return None
        except Exception as e:
            self.logger.debug(f"Could not read thermal status: {e}")
            return None 