#!/usr/bin/env python3
"""
Dice Dataset Collection Tool for AI Camera
Captures images of dice showing values 1-6 for custom model training
"""

from picamera2 import Picamera2
import time
import os
from datetime import datetime

def create_dataset_structure():
    """Create directory structure for dice dataset"""
    base_dir = "dice_dataset"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create directories for each dice value
    for value in range(1, 7):
        os.makedirs(f"{base_dir}/value_{value}", exist_ok=True)
    
    # Create background class (no dice)
    os.makedirs(f"{base_dir}/background", exist_ok=True)
    
    return base_dir

def capture_dice_images():
    """Interactive dice image capture session"""
    base_dir = create_dataset_structure()
    
    # Initialize camera
    picam2 = Picamera2()
    
    # Configure for consistent image capture
    config = picam2.create_still_configuration(
        main={"size": (640, 640)},  # Square format good for AI training
        lores={"size": (320, 320)}, # Preview size
    )
    picam2.configure(config)
    picam2.start()
    
    # Let camera stabilize
    time.sleep(2)
    
    print("🎲 Dice Dataset Collection Tool")
    print("=" * 40)
    print("Instructions:")
    print("1. Position dice clearly in camera view")
    print("2. Ensure good lighting (no shadows on pips)")
    print("3. Try different angles and backgrounds")
    print("4. Capture 20-30 images per dice value")
    print("5. Type 'done' when finished with a value")
    print("=" * 40)
    
    # Capture images for each dice value
    for dice_value in range(1, 8):  # 1-6 + background (7)
        if dice_value == 7:
            class_name = "background"
            print(f"\n📷 Capturing BACKGROUND images (no dice visible)")
        else:
            class_name = f"value_{dice_value}"
            print(f"\n🎲 Capturing dice showing VALUE {dice_value}")
        
        count = 0
        while True:
            user_input = input(f"Press ENTER to capture {class_name} image #{count+1} (or 'done' to finish): ").strip().lower()
            
            if user_input == 'done':
                break
            
            # Capture image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{class_name}_{timestamp}.jpg"
            filepath = os.path.join(base_dir, class_name, filename)
            
            picam2.capture_file(filepath)
            count += 1
            print(f"✅ Captured {filepath}")
        
        print(f"✅ Completed {class_name}: {count} images")
    
    picam2.stop()
    
    # Summary
    print("\n📊 Dataset Summary:")
    print("=" * 40)
    total_images = 0
    for value in range(1, 7):
        class_dir = os.path.join(base_dir, f"value_{value}")
        count = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
        print(f"Value {value}: {count} images")
        total_images += count
    
    bg_count = len([f for f in os.listdir(os.path.join(base_dir, "background")) if f.endswith('.jpg')])
    print(f"Background: {bg_count} images")
    total_images += bg_count
    
    print(f"Total: {total_images} images")
    print(f"\n🎯 Dataset ready for Edge Impulse training!")
    print(f"📁 Location: {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    try:
        capture_dice_images()
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}") 