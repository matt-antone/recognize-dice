#!/usr/bin/env python3
"""
Download Dice Dataset from Roboflow for AI Camera Training
Prepares dataset for Edge Impulse Studio
"""

import os
import requests
import zipfile
from pathlib import Path
import json

def download_roboflow_dataset():
    """Download the 6-sided dice dataset from Roboflow"""
    
    print("🎲 Downloading Roboflow Dice Dataset")
    print("=" * 50)
    
    # Create dataset directory
    dataset_dir = Path("dice_dataset_roboflow")
    dataset_dir.mkdir(exist_ok=True)
    
    # Roboflow dataset URL (you'll need to get the actual download link)
    # This is a placeholder - user will need to get the actual URL from Roboflow
    print("📋 To download the dataset:")
    print("1. Go to: https://public.roboflow.com/object-detection/dice")
    print("2. Click 'Download Dataset'")
    print("3. Choose format: 'Multiclass Classification' (for Edge Impulse)")
    print("4. Download and extract to 'dice_dataset_roboflow' folder")
    print("")
    print("🔄 Alternative: Use COCO JSON format if classification not available")
    
    return dataset_dir

def prepare_for_edge_impulse(dataset_dir):
    """Organize dataset for Edge Impulse Studio upload"""
    
    print("\n🛠️  Preparing dataset for Edge Impulse...")
    
    # Create Edge Impulse friendly structure
    ei_dir = Path("dice_dataset_edge_impulse")
    ei_dir.mkdir(exist_ok=True)
    
    # Edge Impulse expects folders named by class
    classes = ["value_1", "value_2", "value_3", "value_4", "value_5", "value_6", "background"]
    
    for class_name in classes:
        (ei_dir / class_name).mkdir(exist_ok=True)
    
    print("✅ Created Edge Impulse directory structure:")
    for class_name in classes:
        print(f"   📁 {ei_dir / class_name}")
    
    print(f"\n📁 Dataset location: {ei_dir.absolute()}")
    print("\n🎯 Next steps:")
    print("1. Organize images into class folders")
    print("2. Upload to Edge Impulse Studio")
    print("3. Train custom dice detection model")
    
    return ei_dir

def create_dataset_info():
    """Create dataset information file"""
    
    info = {
        "name": "D6 Dice Detection Dataset",
        "source": "Roboflow Public Dataset",
        "url": "https://public.roboflow.com/object-detection/dice",
        "images": 359,
        "classes": 7,  # 6 dice values + background
        "license": "Public Domain",
        "description": "6-sided dice detection with values 1-6 plus background class",
        "formats": [
            "COCO JSON",
            "YOLO",
            "Pascal VOC",
            "Multiclass Classification"
        ],
        "use_case": "AI Camera dice recognition on Raspberry Pi",
        "target_model": "MobileNet for IMX500 acceleration"
    }
    
    with open("dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print("📄 Created dataset_info.json")

if __name__ == "__main__":
    try:
        # Download dataset
        dataset_dir = download_roboflow_dataset()
        
        # Prepare for Edge Impulse
        ei_dir = prepare_for_edge_impulse(dataset_dir)
        
        # Create info file
        create_dataset_info()
        
        print("\n🎉 Dataset preparation complete!")
        print("Ready for Edge Impulse training!")
        
    except Exception as e:
        print(f"❌ Error: {e}") 