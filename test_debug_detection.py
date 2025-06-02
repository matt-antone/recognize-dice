#!/usr/bin/env python3
"""
Debug detection to understand what's happening with dice detection
Saves intermediate processing steps for analysis
"""

import sys
import os
import cv2
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.camera.camera_interface import CameraInterface
    from src.utils.config import Config
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def debug_detection_pipeline(frame: np.ndarray, frame_num: int):
    """Debug the detection pipeline step by step."""
    print(f"\n=== Debugging Frame {frame_num} ===")
    print(f"Frame shape: {frame.shape}")
    
    # Save original frame
    cv2.imwrite(f"debug_01_original_{frame_num}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(f"debug_02_gray_{frame_num}.jpg", gray)
    gray_mean, gray_std = np.mean(gray), np.std(gray)
    print(f"Gray stats: mean={gray_mean:.1f}, std={gray_std:.1f}")
    
    # Analyze brightness
    if gray_mean < 50:
        print("  → Image is VERY DARK - consider better lighting")
    elif gray_mean < 100:
        print("  → Image is somewhat dark - dice may be hard to detect")
    elif gray_mean > 200:
        print("  → Image is very bright - may be overexposed")
    else:
        print("  → Brightness looks reasonable")
    
    if gray_std < 20:
        print("  → Low contrast - dice may not stand out from background")
    else:
        print("  → Good contrast - dice should be detectable")
    
    # Step 2: Bilateral filter
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    cv2.imwrite(f"debug_03_filtered_{frame_num}.jpg", filtered)
    
    # Step 3: CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(filtered)
    cv2.imwrite(f"debug_04_enhanced_{frame_num}.jpg", enhanced)
    enhanced_mean, enhanced_std = np.mean(enhanced), np.std(enhanced)
    print(f"Enhanced stats: mean={enhanced_mean:.1f}, std={enhanced_std:.1f}")
    print(f"  → Contrast improvement: {enhanced_std - gray_std:.1f}")
    
    # Step 4: Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1)
    cv2.imwrite(f"debug_05_blurred_{frame_num}.jpg", blurred)
    
    # Step 5: Try multiple HoughCircles parameter sets
    param_sets = [
        # Original params
        {"name": "Original", "dp": 1, "minDist": 30, "param1": 100, "param2": 25, "minRadius": 10, "maxRadius": 80},
        # More sensitive
        {"name": "Sensitive", "dp": 1, "minDist": 25, "param1": 80, "param2": 20, "minRadius": 8, "maxRadius": 100},
        # Less sensitive
        {"name": "Conservative", "dp": 1, "minDist": 40, "param1": 120, "param2": 30, "minRadius": 15, "maxRadius": 60},
        # Very sensitive
        {"name": "Very Sensitive", "dp": 1, "minDist": 20, "param1": 60, "param2": 15, "minRadius": 5, "maxRadius": 120},
    ]
    
    best_circles = None
    best_count = 0
    best_params = None
    
    print(f"\nTesting HoughCircles with different parameters:")
    for i, params in enumerate(param_sets):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=params["dp"],
            minDist=params["minDist"],
            param1=params["param1"],
            param2=params["param2"],
            minRadius=params["minRadius"],
            maxRadius=params["maxRadius"]
        )
        
        circle_count = len(circles[0]) if circles is not None else 0
        print(f"  {params['name']}: Found {circle_count} circles")
        
        # Analyze circle sizes if found
        if circles is not None and circle_count > 0:
            radii = circles[0, :, 2]
            print(f"    Circle radii: min={np.min(radii):.1f}, max={np.max(radii):.1f}, mean={np.mean(radii):.1f}")
        
        # Draw circles for this parameter set
        debug_img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        if circles is not None:
            circles_int = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles_int:
                cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(debug_img, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(debug_img, f"r={r}", (x-20, y-r-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        cv2.imwrite(f"debug_06_circles_{params['name'].lower().replace(' ', '_')}_{frame_num}.jpg", debug_img)
        
        if circle_count > best_count:
            best_circles = circles
            best_count = circle_count
            best_params = params['name']
    
    print(f"  → Best result: {best_count} circles with {best_params} parameters")
    
    # Step 6: Try edge detection to see what edges are found
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite(f"debug_07_edges_{frame_num}.jpg", edges)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_percentage = (edge_pixels / total_pixels) * 100
    print(f"Edge detection: {edge_percentage:.1f}% of pixels are edges")
    
    if edge_percentage < 5:
        print("  → Very few edges detected - image may be too smooth/blurry")
    elif edge_percentage > 20:
        print("  → Many edges detected - image may be noisy")
    else:
        print("  → Reasonable amount of edges detected")
    
    # Step 7: Try simple blob detection
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 5000
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5
    
    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(enhanced)
    
    print(f"Blob detection: Found {len(keypoints)} blobs")
    if len(keypoints) > 0:
        sizes = [kp.size for kp in keypoints]
        print(f"  Blob sizes: min={np.min(sizes):.1f}, max={np.max(sizes):.1f}, mean={np.mean(sizes):.1f}")
    
    # Draw detected blobs
    blob_img = cv2.drawKeypoints(enhanced, keypoints, np.array([]), (0,0,255), 
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(f"debug_08_blobs_{frame_num}.jpg", blob_img)
    
    # Step 8: Try contour detection for rectangular shapes (dice are cubes)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} total contours")
    
    # Filter contours that might be dice
    dice_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:  # Reasonable size range
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly square/rectangular (4-6 sides)
            if 4 <= len(approx) <= 8:
                dice_contours.append(contour)
    
    print(f"Contour detection: Found {len(dice_contours)} potential dice-shaped contours")
    if len(dice_contours) > 0:
        areas = [cv2.contourArea(c) for c in dice_contours]
        print(f"  Contour areas: min={np.min(areas):.0f}, max={np.max(areas):.0f}, mean={np.mean(areas):.0f}")
    
    # Draw contours
    contour_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, dice_contours, -1, (0, 255, 0), 2)
    cv2.imwrite(f"debug_09_contours_{frame_num}.jpg", contour_img)
    
    # Overall assessment
    print(f"\n--- Frame {frame_num} Assessment ---")
    detection_methods = [
        ("HoughCircles", best_count),
        ("Blob Detection", len(keypoints)),
        ("Contour Detection", len(dice_contours))
    ]
    
    best_method = max(detection_methods, key=lambda x: x[1])
    print(f"Best detection method: {best_method[0]} found {best_method[1]} objects")
    
    if best_method[1] == 0:
        print("❌ No dice detected by any method!")
        print("Possible issues:")
        print("  - Dice too small or too large for current parameters")
        print("  - Poor lighting or contrast")
        print("  - Dice not clearly visible from background")
        print("  - Camera focus issues")
    elif best_method[1] > 6:
        print("⚠️  Too many objects detected - may be detecting noise")
    else:
        print("✅ Reasonable number of objects detected")
    
    return best_circles, len(keypoints), len(dice_contours)


def test_debug():
    """Run debug detection test."""
    print("D6 Dice Recognition - Debug Detection")
    print("=" * 40)
    
    logger = setup_logger(__name__)
    config = Config()
    
    try:
        # Initialize camera
        print("Initializing AI camera...")
        camera = CameraInterface(config)
        camera.start()
        
        print("\nCapturing frames for debug analysis...")
        print("Place dice in view and press Enter to capture debug frames")
        input("Press Enter when ready...")
        
        # Capture a few frames for analysis
        for i in range(3):
            print(f"\nCapturing debug frame {i+1}...")
            frame = camera.capture_frame()
            if frame is not None:
                circles, blobs, contours = debug_detection_pipeline(frame, i+1)
                print(f"Frame {i+1} summary:")
                print(f"  Circles: {len(circles[0]) if circles is not None else 0}")
                print(f"  Blobs: {blobs}")
                print(f"  Contours: {contours}")
            
            if i < 2:
                input("Press Enter for next frame...")
        
        camera.stop()
        camera.cleanup()
        
        print(f"\n✅ Debug analysis complete!")
        print("Check the debug_*.jpg files to see what the detection pipeline sees.")
        print("\nLook for:")
        print("- Are dice visible in the original image?")
        print("- Do the preprocessing steps help or hurt?")
        print("- Which detection method finds the most dice?")
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        logger.error(f"Debug test error: {e}")


if __name__ == "__main__":
    test_debug() 