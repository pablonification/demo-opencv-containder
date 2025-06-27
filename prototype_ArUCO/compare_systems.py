import cv2
import numpy as np
import time
from cv2 import aruco

# Import functions from both systems
# Note: You'll need to organize these into separate modules or copy the functions

def compare_detection_systems(image_path):
    """
    Compare blue rectangle vs ArUco marker detection systems
    """
    print("=== BOTTLE DETECTION SYSTEM COMPARISON ===")
    print(f"Testing image: {image_path}")
    print()
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from '{image_path}'")
        return
    
    # Create side-by-side comparison
    h, w = original_image.shape[:2]
    comparison_image = np.zeros((h, w*2, 3), dtype=np.uint8)
    
    # === TEST 1: BLUE RECTANGLE SYSTEM ===
    print("1. Testing BLUE RECTANGLE system...")
    start_time = time.time()
    
    # Blue rectangle detection parameters
    BLUE_LOWER = np.array([110, 10, 50])
    BLUE_UPPER = np.array([130, 60, 255])
    KNOWN_REF_OBJECT_HEIGHT_CM = 12
    
    try:
        # Simulate blue rectangle detection
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blue_rect_found = False
        blue_rect_ppm = None
        blue_rect_angle = None
        
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:
                    rect = cv2.minAreaRect(cnt)
                    (x,y), (w,h), angle = rect
                    box_points = cv2.boxPoints(rect)
                    box_points = np.intp(box_points)
                    
                    pixel_height = max(w, h)
                    blue_rect_ppm = pixel_height / KNOWN_REF_OBJECT_HEIGHT_CM
                    blue_rect_angle = angle
                    blue_rect_found = True
                    
                    # Draw on left side
                    comparison_image[:, :w] = original_image
                    cv2.drawContours(comparison_image[:, :w], [box_points], 0, (255, 100, 0), 2)
                    cv2.putText(comparison_image[:, :w], "BLUE RECT", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
                    break
        
        blue_rect_time = time.time() - start_time
        
        if blue_rect_found:
            print(f"   ✓ Blue rectangle detected")
            print(f"   PPM: {blue_rect_ppm:.2f}")
            print(f"   Angle: {blue_rect_angle:.1f}°")
            print(f"   Time: {blue_rect_time:.3f}s")
        else:
            print(f"   ✗ Blue rectangle NOT detected")
            comparison_image[:, :w] = original_image
            cv2.putText(comparison_image[:, :w], "NO BLUE RECT", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    except Exception as e:
        print(f"   ✗ Blue rectangle system failed: {e}")
        blue_rect_found = False
        blue_rect_time = time.time() - start_time
    
    print()
    
    # === TEST 2: ARUCO SYSTEM ===
    print("2. Testing ARUCO system...")
    start_time = time.time()
    
    # ArUco detection parameters (compatible with different OpenCV versions)
    try:
        ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)
        ARUCO_PARAMS = aruco.DetectorParameters_create()
    except AttributeError:
        ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        ARUCO_PARAMS = aruco.DetectorParameters()
    KNOWN_MARKER_SIZE_CM = 5.0
    CAMERA_MATRIX = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    DIST_COEFFS = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)
    
    try:
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
        
        aruco_found = False
        aruco_ppm = None
        aruco_angle = None
        aruco_distance = None
        
        if ids is not None:
            # Estimate pose (compatible with different OpenCV versions)
            try:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, KNOWN_MARKER_SIZE_CM, CAMERA_MATRIX, DIST_COEFFS
                )
            except TypeError:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, KNOWN_MARKER_SIZE_CM, CAMERA_MATRIX, DIST_COEFFS
                )
            
            marker_corners = corners[0][0]
            
            # Calculate pixel size
            side1 = np.linalg.norm(marker_corners[1] - marker_corners[0])
            side2 = np.linalg.norm(marker_corners[2] - marker_corners[1])
            side3 = np.linalg.norm(marker_corners[3] - marker_corners[2])
            side4 = np.linalg.norm(marker_corners[0] - marker_corners[3])
            avg_pixel_size = (side1 + side2 + side3 + side4) / 4.0
            
            # Calculate viewing angle
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            marker_normal = rotation_matrix[:, 2]
            camera_direction = np.array([0, 0, 1])
            cos_angle = np.dot(marker_normal, camera_direction)
            viewing_angle_deg = np.degrees(np.arccos(np.clip(np.abs(cos_angle), 0, 1)))
            
            # Perspective correction
            basic_ppm = avg_pixel_size / KNOWN_MARKER_SIZE_CM
            viewing_angle_rad = np.radians(viewing_angle_deg)
            perspective_correction = 1.0 / np.cos(viewing_angle_rad)
            aruco_ppm = basic_ppm * perspective_correction
            aruco_angle = viewing_angle_deg
            aruco_distance = np.linalg.norm(tvec)
            aruco_found = True
            
            # Draw on right side
            comparison_image[:, w:] = original_image
            corners_int = marker_corners.astype(int)
            cv2.polylines(comparison_image[:, w:], [corners_int], True, (0, 255, 0), 2)
            
            # Draw pose axes (compatible with different OpenCV versions)
            try:
                aruco.drawAxis(comparison_image[:, w:], CAMERA_MATRIX, DIST_COEFFS, 
                              rvec, tvec, KNOWN_MARKER_SIZE_CM * 0.5)
            except AttributeError:
                cv2.drawFrameAxes(comparison_image[:, w:], CAMERA_MATRIX, DIST_COEFFS, 
                                 rvec, tvec, KNOWN_MARKER_SIZE_CM * 0.5, 3)
            
            cv2.putText(comparison_image[:, w:], "ARUCO", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        aruco_time = time.time() - start_time
        
        if aruco_found:
            print(f"   ✓ ArUco marker detected (ID: {ids[0][0]})")
            print(f"   Basic PPM: {basic_ppm:.2f}")
            print(f"   Corrected PPM: {aruco_ppm:.2f}")
            print(f"   Viewing angle: {aruco_angle:.1f}°")
            print(f"   Distance: {aruco_distance:.1f}cm")
            print(f"   Time: {aruco_time:.3f}s")
        else:
            print(f"   ✗ ArUco marker NOT detected")
            comparison_image[:, w:] = original_image
            cv2.putText(comparison_image[:, w:], "NO ARUCO", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    except Exception as e:
        print(f"   ✗ ArUco system failed: {e}")
        aruco_found = False
        aruco_time = time.time() - start_time
    
    print()
    
    # === COMPARISON RESULTS ===
    print("=== COMPARISON RESULTS ===")
    
    detection_comparison = {
        "Blue Rectangle": {"found": blue_rect_found, "time": blue_rect_time},
        "ArUco": {"found": aruco_found, "time": aruco_time}
    }
    
    for system, result in detection_comparison.items():
        status = "✓ DETECTED" if result["found"] else "✗ NOT DETECTED"
        print(f"{system:15}: {status:12} ({result['time']:.3f}s)")
    
    if blue_rect_found and aruco_found:
        print()
        print("=== ACCURACY COMPARISON ===")
        print(f"Blue Rectangle PPM: {blue_rect_ppm:.2f}")
        print(f"ArUco PPM (corrected): {aruco_ppm:.2f}")
        print(f"Difference: {abs(aruco_ppm - blue_rect_ppm):.2f} PPM")
        print(f"Perspective correction factor: {aruco_ppm/basic_ppm:.3f}")
        
        if aruco_angle > 10:
            print(f"⚠️  SIGNIFICANT PERSPECTIVE DISTORTION DETECTED!")
            print(f"   Viewing angle: {aruco_angle:.1f}°")
            print(f"   ArUco correction is CRITICAL for accuracy")
        else:
            print(f"✓ Minimal perspective distortion ({aruco_angle:.1f}°)")
    
    elif aruco_found and not blue_rect_found:
        print()
        print("🎯 ARUCO WINS: ArUco detected when blue rectangle failed!")
        print("   This demonstrates ArUco's superior robustness")
    
    elif blue_rect_found and not aruco_found:
        print()
        print("📐 BLUE RECTANGLE WINS: Blue rectangle detected when ArUco failed")
        print("   Consider printing a new ArUco marker or improving lighting")
    
    else:
        print()
        print("❌ BOTH SYSTEMS FAILED: No reference objects detected")
        print("   Check image quality, lighting, and reference object presence")
    
    # Display comparison
    cv2.imshow("System Comparison: Blue Rectangle (Left) vs ArUco (Right)", 
               cv2.resize(comparison_image, (1200, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return detection_comparison

def test_multiple_angles():
    """
    Test system performance across multiple viewing angles
    """
    print("=== MULTI-ANGLE TESTING ===")
    print("This function would test both systems across various camera angles")
    print("Implementation requires multiple test images with known ground truth")
    
    # Placeholder for multiple image testing
    test_images = [
        "test_0_degrees.jpg",   # Straight on
        "test_15_degrees.jpg",  # Slight angle
        "test_30_degrees.jpg",  # Medium angle
        "test_45_degrees.jpg",  # High angle
    ]
    
    results = {
        "blue_rectangle": {"detected": 0, "total": 0, "avg_error": 0},
        "aruco": {"detected": 0, "total": 0, "avg_error": 0}
    }
    
    for image_path in test_images:
        print(f"Testing {image_path}...")
        # Would call compare_detection_systems(image_path) here
        # and aggregate results
    
    print("Results would show detection rates and accuracy across angles")

if __name__ == "__main__":
    print("=== DETECTION SYSTEM COMPARISON TOOL ===")
    print()
    
    # Test with your image
    image_path = "test1.jpg"  # GANTI dengan nama file foto yang sama
    
    compare_detection_systems(image_path)
    
    print()
    print("=== RECOMMENDATIONS ===")
    print("1. If ArUco consistently detects when blue rectangle fails:")
    print("   → Switch to ArUco system for better reliability")
    print()
    print("2. If viewing angles > 15° are common:")
    print("   → ArUco perspective correction is essential")
    print()
    print("3. If both systems work but give different measurements:")
    print("   → ArUco measurements are more accurate due to perspective correction")
    print()
    print("4. For production use:")
    print("   → Use ArUco system with proper camera calibration")
    print("   → Ensure good lighting and marker quality") 