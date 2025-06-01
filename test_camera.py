#!/usr/bin/env python3
"""
Simple camera test for Raspberry Pi 3
Tests camera functionality before running the main application
"""

import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.camera.camera_interface import CameraInterface
    from src.utils.config import Config
    from src.utils.logger import setup_logger
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


def test_camera():
    """Test camera functionality."""
    print("D6 Dice Recognition - Camera Test")
    print("=" * 40)
    
    # Set up logging
    logger = setup_logger(__name__)
    
    # Create config
    config = Config()
    
    print(f"Target resolution: {config.camera_resolution}")
    print(f"Frame skip: {config.frame_skip}")
    print()
    
    try:
        # Initialize camera
        print("Initializing camera...")
        camera = CameraInterface(config)
        
        # Test camera capture
        print("Testing camera capture...")
        success = camera.test_capture()
        
        if not success:
            print("❌ Camera test failed!")
            return False
        
        print("✅ Camera test passed!")
        
        # Capture a few frames to test performance
        print("\nTesting frame capture performance...")
        camera.start()
        
        frame_times = []
        for i in range(10):
            start_time = time.time()
            frame = camera.capture_frame()
            capture_time = time.time() - start_time
            
            if frame is not None:
                frame_times.append(capture_time)
                print(f"Frame {i+1}: {capture_time*1000:.1f}ms, shape: {frame.shape}")
            else:
                print(f"Frame {i+1}: Failed to capture")
            
            time.sleep(0.1)  # Small delay between captures
        
        camera.stop()
        camera.cleanup()
        
        # Calculate statistics
        if frame_times:
            avg_time = np.mean(frame_times)
            max_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            print(f"\n📊 Performance Summary:")
            print(f"Average capture time: {avg_time*1000:.1f}ms")
            print(f"Theoretical max FPS: {max_fps:.1f}")
            print(f"Target FPS for Pi 3: 3-5 FPS")
            
            if max_fps >= 3:
                print("✅ Performance looks good for Pi 3!")
            else:
                print("⚠️  Performance may be slower than target")
        
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed with error: {e}")
        logger.error(f"Camera test error: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")
    
    required_modules = [
        'cv2',
        'numpy', 
        'PIL',
        'tkinter'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing.append(module)
    
    # Check camera libraries
    camera_available = False
    try:
        import picamera2
        print("✅ picamera2")
        camera_available = True
    except ImportError:
        print("❌ picamera2")
        
        try:
            import picamera
            print("✅ picamera (legacy)")
            camera_available = True
        except ImportError:
            print("❌ picamera")
    
    if not camera_available:
        missing.append('picamera2 or picamera')
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies available!")
    return True


def main():
    """Main test function."""
    print("Raspberry Pi 3 - Camera Test")
    print("Testing camera setup for dice recognition")
    print("-" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing dependencies before testing camera.")
        return False
    
    print()
    
    # Test camera
    if test_camera():
        print("\n🎉 Camera test completed successfully!")
        print("You can now run the main application with: python3 main.py")
        return True
    else:
        print("\n❌ Camera test failed!")
        print("Please check your camera connection and setup.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 