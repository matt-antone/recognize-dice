#!/usr/bin/env python3
"""
D6 Dice Recognition for Raspberry Pi 3
Main application entry point

Optimized for Pi 3 constraints:
- 1GB RAM limitation
- ARM Cortex-A53 @ 1.2GHz
- Target 3-5 FPS performance
"""

import sys
import os
import logging
import tkinter as tk
from tkinter import messagebox
import threading
import time

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.camera.camera_interface import CameraInterface
    from src.detection.dice_detector import DiceDetector
    from src.gui.main_window import MainWindow
    from src.utils.config import Config
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class DiceRecognitionApp:
    """Main application class for dice recognition."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.config = Config()
        
        # Core components
        self.camera = None
        self.detector = None
        self.gui = None
        
        # Application state
        self.running = False
        self.processing_thread = None
        
        self.logger.info("Initializing D6 Dice Recognition App for Pi 3")
    
    def initialize_components(self):
        """Initialize camera, detector, and GUI components."""
        try:
            self.logger.info("Initializing camera interface...")
            self.camera = CameraInterface(self.config)
            
            self.logger.info("Initializing dice detector...")
            self.detector = DiceDetector(self.config)
            
            self.logger.info("Initializing GUI...")
            self.gui = MainWindow(self.config, self.on_start, self.on_stop)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            messagebox.showerror("Initialization Error", 
                               f"Failed to initialize application components:\n{e}")
            return False
    
    def on_start(self):
        """Start the dice recognition process."""
        if self.running:
            return
        
        try:
            self.logger.info("Starting dice recognition...")
            self.camera.start()
            self.running = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.processing_thread.start()
            
            self.gui.update_status("Running - Place dice in camera view")
            
        except Exception as e:
            self.logger.error(f"Failed to start recognition: {e}")
            self.gui.update_status(f"Error: {e}")
    
    def on_stop(self):
        """Stop the dice recognition process."""
        if not self.running:
            return
        
        self.logger.info("Stopping dice recognition...")
        self.running = False
        
        if self.camera:
            self.camera.stop()
        
        self.gui.update_status("Stopped")
    
    def processing_loop(self):
        """Main processing loop for dice detection."""
        frame_count = 0
        last_time = time.time()
        
        while self.running:
            try:
                # Capture frame
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                # Skip frames for performance (Pi 3 optimization)
                frame_count += 1
                if frame_count % self.config.frame_skip != 0:
                    continue
                
                # Process frame through detector
                detections = self.detector.detect_dice(frame)
                
                # Update GUI with results
                self.gui.update_detections(frame, detections)
                
                # Calculate and display FPS
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frame_count / (current_time - last_time)
                    self.gui.update_fps(fps)
                    frame_count = 0
                    last_time = current_time
                
                # Small delay to prevent overwhelming the Pi 3
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                # Continue processing despite errors
                continue
    
    def cleanup(self):
        """Clean up resources before exit."""
        self.logger.info("Cleaning up resources...")
        
        self.on_stop()
        
        if self.camera:
            self.camera.cleanup()
        
        if self.detector:
            self.detector.cleanup()
    
    def run(self):
        """Run the main application."""
        try:
            # Initialize all components
            if not self.initialize_components():
                return False
            
            self.logger.info("Starting GUI main loop...")
            
            # Set up cleanup on window close
            self.gui.root.protocol("WM_DELETE_WINDOW", self.on_exit)
            
            # Start GUI main loop
            self.gui.root.mainloop()
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
            return True
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            return False
        finally:
            self.cleanup()
    
    def on_exit(self):
        """Handle application exit."""
        self.cleanup()
        self.gui.root.destroy()


def main():
    """Main entry point."""
    print("D6 Dice Recognition for Raspberry Pi 3")
    print("Optimized for Pi 3 performance constraints")
    print("-" * 50)
    
    # Create and run application
    app = DiceRecognitionApp()
    
    try:
        success = app.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 