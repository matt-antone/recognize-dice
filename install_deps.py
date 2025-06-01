#!/usr/bin/env python3
"""
Platform-specific dependency installation for D6 Dice Recognition
Handles TensorFlow Lite installation differences between platforms
"""

import sys
import platform
import subprocess
import os


def detect_platform():
    """Detect the current platform and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    print(f"Detected platform: {system} {machine}")
    
    # Check if running on Raspberry Pi
    is_raspberry_pi = False
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                is_raspberry_pi = True
    except (FileNotFoundError, PermissionError):
        pass
    
    return {
        'system': system,
        'machine': machine,
        'is_raspberry_pi': is_raspberry_pi,
        'is_arm': machine.startswith('arm') or machine.startswith('aarch'),
        'is_macos': system == 'darwin',
        'is_linux': system == 'linux',
        'is_windows': system == 'windows'
    }


def install_base_requirements():
    """Install base requirements that work on all platforms."""
    base_requirements = [
        'opencv-python>=4.5.0',
        'numpy>=1.21.0',
        'pillow>=8.0.0',
        'setuptools>=45.0.0',
        'psutil>=5.8.0'
    ]
    
    print("Installing base requirements...")
    for req in base_requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
            return False
    
    return True


def install_tensorflow_lite(platform_info):
    """Install TensorFlow Lite based on platform."""
    print("\nInstalling TensorFlow Lite...")
    
    if platform_info['is_raspberry_pi']:
        # Try tensorflow-lite-runtime first (optimized for Pi)
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'tensorflow-lite-runtime>=2.8.0'
            ])
            print("✅ Installed tensorflow-lite-runtime (Pi optimized)")
            return True
        except subprocess.CalledProcessError:
            print("⚠️  tensorflow-lite-runtime not available, trying alternatives...")
    
    # For development or when tflite-runtime isn't available
    if not platform_info['is_raspberry_pi']:
        print("📝 Development platform detected - installing full TensorFlow")
        print("   (TensorFlow Lite runtime will be extracted from this)")
        
        try:
            # Try installing TensorFlow (includes TFLite)
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'tensorflow>=2.8.0,<2.16.0'  # Avoid newer versions that may have issues
            ])
            print("✅ Installed tensorflow (includes TFLite)")
            return True
        except subprocess.CalledProcessError:
            print("⚠️  TensorFlow installation failed, will use fallback detection only")
            return False
    
    # Fallback for Pi if tflite-runtime failed
    print("⚠️  No TensorFlow Lite available - will use fallback detection only")
    return False


def install_camera_deps(platform_info):
    """Install camera dependencies if on appropriate platform."""
    if not (platform_info['is_raspberry_pi'] or platform_info['is_linux']):
        print("📝 Camera dependencies skipped (not on Raspberry Pi)")
        return True
    
    print("\nInstalling camera dependencies...")
    
    # Try picamera2 first (modern)
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'picamera2>=0.3.0'])
        print("✅ Installed picamera2")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  picamera2 not available, will try legacy picamera")
    
    # Try legacy picamera
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'picamera>=1.13'])
        print("✅ Installed picamera (legacy)")
        return True
    except subprocess.CalledProcessError:
        print("❌ No camera library available")
        if platform_info['is_raspberry_pi']:
            print("   You may need to enable camera interface: sudo raspi-config")
        return False


def check_installation():
    """Check if critical components can be imported."""
    print("\n" + "="*50)
    print("INSTALLATION VERIFICATION")
    print("="*50)
    
    # Test imports
    components = [
        ('OpenCV', 'cv2'),
        ('NumPy', 'numpy'),
        ('PIL', 'PIL'),
        ('Tkinter', 'tkinter')
    ]
    
    success_count = 0
    for name, module in components:
        try:
            __import__(module)
            print(f"✅ {name}")
            success_count += 1
        except ImportError:
            print(f"❌ {name}")
    
    # Test TensorFlow Lite
    tf_available = False
    try:
        import tflite_runtime.interpreter as tflite
        print("✅ TensorFlow Lite Runtime")
        tf_available = True
    except ImportError:
        try:
            import tensorflow.lite as tflite
            print("✅ TensorFlow Lite (from full TensorFlow)")
            tf_available = True
        except ImportError:
            print("⚠️  TensorFlow Lite (will use fallback detection)")
    
    # Test camera (only on appropriate platforms)
    platform_info = detect_platform()
    if platform_info['is_raspberry_pi'] or platform_info['is_linux']:
        camera_available = False
        try:
            import picamera2
            print("✅ Picamera2")
            camera_available = True
        except ImportError:
            try:
                import picamera
                print("✅ Picamera (legacy)")
                camera_available = True
            except ImportError:
                print("❌ Camera library")
    else:
        print("📝 Camera check skipped (not on Pi)")
        camera_available = True  # Don't count as failure on dev platforms
    
    print("\n" + "="*50)
    if success_count >= 4:  # Core components working
        print("🎉 Installation completed successfully!")
        if not tf_available:
            print("📝 Note: Will use fallback detection (no TensorFlow Lite)")
        print("\nYou can now test the camera with:")
        print("  python3 test_camera.py")
        print("\nOr run the main application with:")
        print("  python3 main.py")
        return True
    else:
        print("❌ Installation incomplete - some components missing")
        return False


def main():
    """Main installation function."""
    print("D6 Dice Recognition - Dependency Installation")
    print("=" * 50)
    
    # Detect platform
    platform_info = detect_platform()
    
    if platform_info['is_macos']:
        print("📝 macOS detected - installing development dependencies")
        print("   (Full testing requires Raspberry Pi hardware)")
    elif platform_info['is_raspberry_pi']:
        print("🥧 Raspberry Pi detected - installing Pi-optimized dependencies")
    else:
        print("🖥️  Generic platform - installing compatible dependencies")
    
    print()
    
    # Install components
    success = True
    
    # Base requirements
    if not install_base_requirements():
        success = False
    
    # TensorFlow Lite
    install_tensorflow_lite(platform_info)  # Don't fail on this
    
    # Camera dependencies
    if platform_info['is_raspberry_pi'] or platform_info['is_linux']:
        install_camera_deps(platform_info)  # Don't fail on this either
    
    # Verify installation
    check_installation()
    
    if not success:
        print("\n⚠️  Some components failed to install")
        print("You may still be able to run the application with limited functionality")
        return False
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1) 