"""
Main GUI window for D6 Dice Recognition App
Lightweight Tkinter interface optimized for Pi 3
"""

import tkinter as tk
from tkinter import ttk, messagebox, Frame, Label, Button, Text
import tkinter.font as tkFont
from PIL import Image, ImageTk
import numpy as np
import cv2
import logging
from typing import List, Callable, Optional
import threading


class MainWindow:
    """Main application window using Tkinter."""
    
    def __init__(self, config, start_callback: Callable, stop_callback: Callable):
        self.config = config
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.logger = logging.getLogger(__name__)
        
        # GUI state
        self.is_running = False
        self.current_image = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(self.config.window_title)
        self.root.geometry(f"{self.config.window_size[0]}x{self.config.window_size[1]}")
        self.root.resizable(True, True)
        
        # Set up GUI components
        self._create_widgets()
        self._setup_layout()
        
        self.logger.info("Main window initialized")
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        self.main_frame = Frame(self.root)
        
        # Control panel
        self.control_frame = Frame(self.main_frame, relief="ridge", bd=2)
        
        # Start/Stop buttons
        self.start_button = Button(
            self.control_frame,
            text="Start Detection",
            command=self._on_start_clicked,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5
        )
        
        self.stop_button = Button(
            self.control_frame,
            text="Stop Detection",
            command=self._on_stop_clicked,
            bg="#f44336",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5,
            state="disabled"
        )
        
        # Status display
        self.status_label = Label(
            self.control_frame,
            text="Ready - Click Start to begin",
            font=("Arial", 10),
            fg="blue"
        )
        
        # FPS display
        self.fps_label = Label(
            self.control_frame,
            text="FPS: --",
            font=("Arial", 10),
            fg="green"
        )
        
        # Camera display frame
        self.camera_frame = Frame(self.main_frame, relief="sunken", bd=2)
        
        # Camera display label
        self.camera_label = Label(
            self.camera_frame,
            text="Camera view will appear here",
            font=("Arial", 12),
            bg="black",
            fg="white",
            width=40,
            height=20
        )
        
        # Detection results frame
        self.results_frame = Frame(self.main_frame, relief="ridge", bd=2)
        
        # Results title
        self.results_title = Label(
            self.results_frame,
            text="Detected Dice:",
            font=("Arial", 12, "bold")
        )
        
        # Results display
        self.results_text = Text(
            self.results_frame,
            height=8,
            width=30,
            font=("Courier", 10),
            state="disabled"
        )
        
        # Scrollbar for results
        self.results_scrollbar = ttk.Scrollbar(
            self.results_frame,
            orient="vertical",
            command=self.results_text.yview
        )
        self.results_text.configure(yscrollcommand=self.results_scrollbar.set)
        
        # Statistics frame
        self.stats_frame = Frame(self.main_frame, relief="ridge", bd=2)
        
        self.stats_title = Label(
            self.stats_frame,
            text="Statistics:",
            font=("Arial", 10, "bold")
        )
        
        self.stats_text = Text(
            self.stats_frame,
            height=6,
            width=30,
            font=("Courier", 9),
            state="disabled"
        )
    
    def _setup_layout(self):
        """Set up widget layout."""
        # Main frame
        self.main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Control panel at top
        self.control_frame.pack(fill="x", pady=(0, 5))
        
        # Buttons in control frame
        self.start_button.pack(side="left", padx=5)
        self.stop_button.pack(side="left", padx=5)
        self.status_label.pack(side="left", padx=20)
        self.fps_label.pack(side="right", padx=5)
        
        # Camera and results in main area
        content_frame = Frame(self.main_frame)
        content_frame.pack(fill="both", expand=True)
        
        # Camera view on left
        self.camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.camera_label.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Right panel for results and stats
        right_panel = Frame(content_frame)
        right_panel.pack(side="right", fill="y")
        
        # Results section
        self.results_frame.pack(fill="both", expand=True, pady=(0, 5))
        self.results_title.pack(anchor="w", padx=5, pady=(5, 2))
        
        # Results text with scrollbar
        text_frame = Frame(self.results_frame)
        text_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        self.results_text.pack(side="left", fill="both", expand=True)
        self.results_scrollbar.pack(side="right", fill="y")
        
        # Statistics section
        self.stats_frame.pack(fill="x")
        self.stats_title.pack(anchor="w", padx=5, pady=(5, 2))
        self.stats_text.pack(fill="x", padx=5, pady=(0, 5))
    
    def _on_start_clicked(self):
        """Handle start button click."""
        try:
            self.start_callback()
            self.is_running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
        except Exception as e:
            self.logger.error(f"Error starting detection: {e}")
            messagebox.showerror("Error", f"Failed to start detection:\n{e}")
    
    def _on_stop_clicked(self):
        """Handle stop button click."""
        try:
            self.stop_callback()
            self.is_running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            
        except Exception as e:
            self.logger.error(f"Error stopping detection: {e}")
            messagebox.showerror("Error", f"Failed to stop detection:\n{e}")
    
    def update_status(self, status: str):
        """Update status display."""
        if self.root:
            self.status_label.config(text=status)
            self.root.update_idletasks()
    
    def update_fps(self, fps: float):
        """Update FPS display."""
        if self.root:
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.root.update_idletasks()
    
    def update_detections(self, frame: np.ndarray, detections: List):
        """
        Update display with new frame and detections.
        
        Args:
            frame: RGB frame from camera
            detections: List of DiceDetection objects
        """
        if not self.root:
            return
        
        try:
            # Update camera display
            self._update_camera_display(frame, detections)
            
            # Update detection results
            self._update_detection_results(detections)
            
            # Update statistics (every 10 frames to reduce overhead)
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
            else:
                self._frame_count = 1
            
            if self._frame_count % 10 == 0:
                self._update_statistics()
            
        except Exception as e:
            self.logger.error(f"Error updating display: {e}")
    
    def _update_camera_display(self, frame: np.ndarray, detections: List):
        """Update camera display with frame and detection overlays."""
        try:
            # Make a copy to draw on
            display_frame = frame.copy()
            
            # Draw detection boxes and labels
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{detection.value} ({detection.confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background for text
                cv2.rectangle(
                    display_frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (0, 255, 0),
                    -1
                )
                
                # Text
                cv2.putText(
                    display_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
            
            # Convert to PIL Image
            pil_image = Image.fromarray(display_frame)
            
            # Resize for display (maintain aspect ratio)
            display_size = (480, 360)  # Smaller for Pi 3 performance
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.camera_label.config(image=photo, text="")
            self.camera_label.image = photo  # Keep reference
            
        except Exception as e:
            self.logger.error(f"Error updating camera display: {e}")
    
    def _update_detection_results(self, detections: List):
        """Update detection results text."""
        try:
            # Enable text widget for editing
            self.results_text.config(state="normal")
            
            # Clear existing content
            self.results_text.delete(1.0, tk.END)
            
            if not detections:
                self.results_text.insert(tk.END, "No dice detected\n")
            else:
                for i, detection in enumerate(detections):
                    result_line = (f"Dice {i+1}:\n"
                                 f"  Value: {detection.value}\n"
                                 f"  Confidence: {detection.confidence:.2f}\n"
                                 f"  Position: ({detection.center_x}, {detection.center_y})\n\n")
                    self.results_text.insert(tk.END, result_line)
            
            # Disable text widget
            self.results_text.config(state="disabled")
            
            # Scroll to top
            self.results_text.see(1.0)
            
        except Exception as e:
            self.logger.error(f"Error updating detection results: {e}")
    
    def _update_statistics(self):
        """Update statistics display."""
        try:
            # Enable text widget
            self.stats_text.config(state="normal")
            
            # Clear existing content
            self.stats_text.delete(1.0, tk.END)
            
            # Add statistics (placeholder for now)
            stats_text = (f"Frames processed: {getattr(self, '_frame_count', 0)}\n"
                         f"Status: {'Running' if self.is_running else 'Stopped'}\n"
                         f"Resolution: {self.config.camera_resolution}\n"
                         f"Frame skip: {self.config.frame_skip}\n"
                         f"Conf threshold: {self.config.confidence_threshold}\n")
            
            self.stats_text.insert(tk.END, stats_text)
            
            # Disable text widget
            self.stats_text.config(state="disabled")
            
        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}") 