"""
Depth Offset Measurement Tool
Tool untuk mengukur selisih jarak antara marker dan botol dari kamera
"""

import cv2
import numpy as np
from cv2 import aruco
from aruco_config import *

def measure_depth_offset(image_path=None, camera_index=0):
    """
    Mengukur selisih jarak marker dengan objek lain (botol)
    """
    # Setup ArUco
    try:
        ARUCO_DICT = aruco.Dictionary_get(ARUCO_DICT_TYPE)
        ARUCO_PARAMS = get_aruco_params()
    except AttributeError:
        ARUCO_DICT = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        ARUCO_PARAMS = get_aruco_params()
    
    CAMERA_MATRIX, DIST_COEFFS = get_camera_params()
    
    if image_path:
        # Static image mode
        print(f"=== DEPTH MEASUREMENT FROM IMAGE ===")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image {image_path}")
            return
        
        measure_from_image(image, ARUCO_DICT, ARUCO_PARAMS, CAMERA_MATRIX, DIST_COEFFS)
    
    else:
        # Live camera mode
        print(f"=== LIVE DEPTH MEASUREMENT ===")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return
        
        measure_from_camera(cap, ARUCO_DICT, ARUCO_PARAMS, CAMERA_MATRIX, DIST_COEFFS)
        cap.release()
    
    cv2.destroyAllWindows()

def detect_aruco_simple(image, aruco_dict, aruco_params, camera_matrix, dist_coeffs):
    """
    Deteksi ArUco sederhana untuk mendapatkan jarak
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    distances = []
    
    if ids is not None:
        try:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, KNOWN_MARKER_SIZE_CM, camera_matrix, dist_coeffs
            )
        except TypeError:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, KNOWN_MARKER_SIZE_CM, camera_matrix, dist_coeffs
            )
        
        for i, marker_id in enumerate(ids.flatten()):
            tvec = tvecs[i][0]
            distance = np.linalg.norm(tvec)
            distances.append({
                "id": marker_id,
                "distance": distance,
                "position": tvec
            })
    
    return distances

def measure_from_image(image, aruco_dict, aruco_params, camera_matrix, dist_coeffs):
    """
    Pengukuran dari gambar statis
    """
    print("Instructions:")
    print("1. Ensure you have multiple ArUco markers in the image")
    print("2. Place one marker where the bottle will be")
    print("3. Place another marker at reference position (wall/background)")
    print()
    
    distances = detect_aruco_simple(image, aruco_dict, aruco_params, camera_matrix, dist_coeffs)
    
    if len(distances) < 2:
        print(f"Found only {len(distances)} marker(s). Need at least 2 for depth measurement.")
        return
    
    print("Detected markers:")
    for i, marker in enumerate(distances):
        print(f"  {i+1}. ID {marker['id']}: {marker['distance']:.1f}cm from camera")
    
    print("\nSelect markers to compare:")
    try:
        ref_idx = int(input("Reference marker index (background/wall): ")) - 1
        bottle_idx = int(input("Bottle position marker index: ")) - 1
        
        if 0 <= ref_idx < len(distances) and 0 <= bottle_idx < len(distances):
            ref_dist = distances[ref_idx]['distance']
            bottle_dist = distances[bottle_idx]['distance']
            depth_offset = ref_dist - bottle_dist
            
            print(f"\n=== DEPTH MEASUREMENT RESULT ===")
            print(f"Reference marker distance: {ref_dist:.1f}cm")
            print(f"Bottle position distance: {bottle_dist:.1f}cm")
            print(f"Depth offset: {depth_offset:.1f}cm")
            
            if depth_offset > 0:
                print(f"→ Bottle is {depth_offset:.1f}cm CLOSER to camera")
                print(f"Use BOTTLE_DEPTH_OFFSET_CM = {depth_offset:.0f} in aruco_bottle_detector.py")
            elif depth_offset < 0:
                print(f"→ Bottle is {abs(depth_offset):.1f}cm FARTHER from camera")
                print(f"Use BOTTLE_DEPTH_OFFSET_CM = {depth_offset:.0f} in aruco_bottle_detector.py")
            else:
                print("→ Bottle and reference are at same distance")
                print("Use BOTTLE_DEPTH_OFFSET_CM = 0 in aruco_bottle_detector.py")
        
        else:
            print("Invalid marker indices")
    
    except ValueError:
        print("Invalid input")

def measure_from_camera(cap, aruco_dict, aruco_params, camera_matrix, dist_coeffs):
    """
    Pengukuran real-time dari kamera
    """
    print("Live measurement mode:")
    print("- Place markers at different positions")
    print("- Press 's' to save current measurements")
    print("- Press 'q' to quit")
    
    saved_measurements = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        distances = detect_aruco_simple(frame, aruco_dict, aruco_params, camera_matrix, dist_coeffs)
        display_frame = frame.copy()
        
        # Draw detection info
        for marker in distances:
            # Find marker corners for drawing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            
            if ids is not None:
                marker_idx = np.where(ids.flatten() == marker["id"])[0]
                if len(marker_idx) > 0:
                    idx = marker_idx[0]
                    corner = corners[idx][0]
                    center = tuple(map(int, np.mean(corner, axis=0)))
                    
                    # Draw marker
                    cv2.polylines(display_frame, [corner.astype(int)], True, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"ID:{marker['id']} {marker['distance']:.1f}cm", 
                               center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show instructions
        cv2.putText(display_frame, f"Detected: {len(distances)} markers", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 's' to save, 'q' to quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Depth Measurement', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(distances) >= 1:
            # Save current measurement
            saved_measurements.append(distances.copy())
            print(f"Saved measurement {len(saved_measurements)}: {len(distances)} markers")
            
        elif key == ord('q'):
            break
    
    # Analyze saved measurements
    if saved_measurements:
        print(f"\n=== ANALYSIS OF {len(saved_measurements)} MEASUREMENTS ===")
        
        # Find common markers across measurements
        all_ids = set()
        for measurement in saved_measurements:
            for marker in measurement:
                all_ids.add(marker["id"])
        
        print(f"Found markers with IDs: {sorted(all_ids)}")
        
        # Calculate average distances for each marker
        for marker_id in sorted(all_ids):
            distances_for_id = []
            for measurement in saved_measurements:
                for marker in measurement:
                    if marker["id"] == marker_id:
                        distances_for_id.append(marker["distance"])
            
            if distances_for_id:
                avg_dist = np.mean(distances_for_id)
                std_dist = np.std(distances_for_id)
                print(f"  ID {marker_id}: {avg_dist:.1f} ± {std_dist:.1f}cm ({len(distances_for_id)} measurements)")

def main():
    print("=== DEPTH OFFSET MEASUREMENT TOOL ===")
    print("Tool untuk mengukur selisih jarak marker dan botol")
    print()
    
    print("Mode:")
    print("1. Static image analysis")
    print("2. Live camera measurement")
    
    choice = input("Pilih mode (1-2): ").strip()
    
    if choice == "1":
        image_path = input("Image path (atau tekan Enter untuk test2.2.jpg): ").strip()
        if not image_path:
            image_path = "test2.2.jpg"
        measure_depth_offset(image_path=image_path)
    
    elif choice == "2":
        camera_index = input("Camera index (default 0): ").strip()
        camera_index = int(camera_index) if camera_index.isdigit() else 0
        measure_depth_offset(camera_index=camera_index)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 