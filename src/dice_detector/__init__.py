"""
D6 Dice Recognition with AI Camera

AI-powered dice detection using Raspberry Pi + Sony IMX500 AI Camera.
Leverages hardware AI acceleration for real-time performance.
"""

__version__ = "1.0.0"
__author__ = "AI Camera Dice Recognition Project"
__license__ = "MIT"

# Main application components
from .camera_interface import AICamera
from .ai_processor import DiceAIProcessor
from .dice_classifier import DiceClassifier

__all__ = [
    "AICamera",
    "DiceAIProcessor", 
    "DiceClassifier",
] 