import cv2
import numpy as np
import imutils
from cv2 import aruco

# --- 1. Configuration & Calibration ---

# == IMPORT CONFIGURATION ==
from aruco_config import *

# Setup ArUco dengan konfigurasi yang bisa di-tune
try:
    ARUCO_DICT = aruco.Dictionary_get(ARUCO_DICT_TYPE)
    ARUCO_PARAMS = get_aruco_params()
except AttributeError:
    ARUCO_DICT = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    ARUCO_PARAMS = get_aruco_params()

# Setup Camera dengan konfigurasi
CAMERA_MATRIX, DIST_COEFFS = get_camera_params()

# --- 2. ArUco Detection Functions ---

def detect_aruco_markers(image, dictionary, parameters):
    """
    Detects ArUco markers in the image and returns their information.
    Returns list of marker data including corners, IDs, and pose estimation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    
    marker_data = []
    
    if ids is not None:
        print(f"Found {len(ids)} ArUco marker(s) with IDs: {ids.flatten()}")
        
        # Estimate pose for each marker (compatible with different OpenCV versions)
        try:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, KNOWN_MARKER_SIZE_CM, CAMERA_MATRIX, DIST_COEFFS
            )
        except TypeError:
            # For newer OpenCV versions
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, KNOWN_MARKER_SIZE_CM, CAMERA_MATRIX, DIST_COEFFS
            )
        
        for i, marker_id in enumerate(ids.flatten()):
            # Get corner coordinates
            marker_corners = corners[i][0]  # Shape: (4, 2)
            
            # Calculate marker center
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            
            # Calculate marker size in pixels (average of sides)
            side1 = np.linalg.norm(marker_corners[1] - marker_corners[0])
            side2 = np.linalg.norm(marker_corners[2] - marker_corners[1])
            side3 = np.linalg.norm(marker_corners[3] - marker_corners[2])
            side4 = np.linalg.norm(marker_corners[0] - marker_corners[3])
            avg_pixel_size = (side1 + side2 + side3 + side4) / 4.0
            
            # Get pose information
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]
            
            # Calculate distance from camera to marker
            distance = np.linalg.norm(tvec)
            
            # Calculate viewing angle (deviation from perpendicular view)
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # The z-axis of the marker in camera coordinates
            marker_normal = rotation_matrix[:, 2]
            
            # Camera viewing direction is along positive z-axis
            camera_direction = np.array([0, 0, 1])
            
            # Calculate angle between camera direction and marker normal
            cos_angle = np.dot(marker_normal, camera_direction)
            viewing_angle_deg = np.degrees(np.arccos(np.clip(np.abs(cos_angle), 0, 1)))
            
            marker_info = {
                "id": marker_id,
                "corners": marker_corners,
                "center": (center_x, center_y),
                "pixel_size": avg_pixel_size,
                "rvec": rvec,
                "tvec": tvec,
                "distance": distance,
                "viewing_angle_deg": viewing_angle_deg,
                "rotation_matrix": rotation_matrix
            }
            
            marker_data.append(marker_info)
            
            print(f"  Marker ID {marker_id}:")
            print(f"    Pixel size: {avg_pixel_size:.2f}px")
            print(f"    Distance: {distance:.2f}cm")
            print(f"    Viewing angle: {viewing_angle_deg:.1f}°")
    
    return marker_data

def calculate_perspective_corrected_calibration(marker_data):
    """
    Calculates perspective-corrected pixels per cm using ArUco marker data.
    Takes into account viewing angle and distance for accurate calibration.
    """
    if not marker_data:
        print("No ArUco markers found for calibration")
        return None
    
    # Use the first detected marker for calibration
    # In a multi-marker setup, you could use the best-positioned marker
    marker = marker_data[0]
    
    # Basic calibration: pixels per cm at this distance and angle
    basic_ppm = marker["pixel_size"] / KNOWN_MARKER_SIZE_CM
    
    # Perspective correction factor
    # When viewing angle increases, the apparent size decreases
    viewing_angle_rad = np.radians(marker["viewing_angle_deg"])
    perspective_correction = 1.0 / np.cos(viewing_angle_rad)
    
    # Distance-based correction (optional, for more advanced calibration)
    # This accounts for lens distortion and can be calibrated empirically
    distance_correction = 1.0  # Keep simple for now
    
    corrected_ppm = basic_ppm * perspective_correction * distance_correction
    
    print(f"Calibration Details:")
    print(f"  Basic PPM: {basic_ppm:.2f}")
    print(f"  Viewing angle: {marker['viewing_angle_deg']:.1f}°")
    print(f"  Perspective correction: {perspective_correction:.3f}")
    print(f"  Corrected PPM: {corrected_ppm:.2f}")
    
    return corrected_ppm, marker

def preprocess_roi_for_bottle(roi):
    """
    Applies robust preprocessing to the Region of Interest (ROI) to isolate the bottle.
    """
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edged_roi = cv2.Canny(blurred_roi, 40, 120)
    cv2.imshow("4. ROI Canny Edges (Original)", edged_roi.copy())

    closing_kernel_size = (5, 5)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, closing_kernel_size)
    iterations_closing = 1
    closed_edges_roi = cv2.morphologyEx(edged_roi, cv2.MORPH_CLOSE, kernel_edge, iterations=iterations_closing)
    cv2.imshow("5. ROI Canny Edges (AFTER Closing)", closed_edges_roi)

    return closed_edges_roi

def find_bottle_in_roi(processed_roi, min_bottle_area_pixels=1000, min_bottle_aspect_ratio=1.3, max_bottle_tilt_degrees=15):
    """
    Finds bottle contours in the processed ROI with filtering criteria.
    """
    contours, hierarchy = cv2.findContours(processed_roi.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    valid_bottle = None
    max_area_found = 0

    print(f"--- In find_bottle_in_roi ---")
    if not contours:
        print("No contours found in processed ROI.")
    else:
        print(f"Found {len(contours)} contours in ROI before filtering.")
        roi_debug_contours_img = cv2.cvtColor(processed_roi, cv2.COLOR_GRAY2BGR)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            rect_debug = cv2.minAreaRect(cnt)
            (center_x, center_y), (rect_w_raw, rect_h_raw), angle_r = rect_debug
            box_pts_debug = cv2.boxPoints(rect_debug)
            box_pts_debug = np.intp(box_pts_debug)

            # Determine visual height and width
            current_rect_h = max(rect_w_raw, rect_h_raw)
            current_rect_w = min(rect_w_raw, rect_h_raw)
            aspect_ratio_debug = current_rect_h / current_rect_w if current_rect_w > 0 else 0

            print(f"  Contour {i}: Area={area:.0f}, AspectRatio={aspect_ratio_debug:.2f}, Angle={angle_r:.1f}")
            cv2.drawContours(roi_debug_contours_img, [cnt], -1, (0, 0, 255), 1)
            cv2.drawContours(roi_debug_contours_img, [box_pts_debug], 0, (255, 0, 0), 1)

            if area < min_bottle_area_pixels:
                print(f"    Contour {i} REJECTED: Area {area:.0f} too small (min: {min_bottle_area_pixels:.0f}).")
                continue

            if current_rect_w == 0 or current_rect_h == 0:
                print(f"    Contour {i} REJECTED: Zero width or height.")
                continue

            aspect_ratio = current_rect_h / current_rect_w
            if aspect_ratio < min_bottle_aspect_ratio:
                print(f"    Contour {i} REJECTED: Aspect ratio {aspect_ratio:.2f} too small (min: {min_bottle_aspect_ratio:.2f}).")
                continue

            # === Upright Bottle Filter ===
            is_upright = False
            if rect_h_raw >= rect_w_raw:
                if abs(angle_r) < max_bottle_tilt_degrees:
                    is_upright = True
                    print(f"    Contour {i} Upright Check: angle_r={angle_r:.1f}° vs max_tilt={max_bottle_tilt_degrees}°. PASSED.")
                else:
                    print(f"    Contour {i} Upright Check: angle_r={angle_r:.1f}° vs max_tilt={max_bottle_tilt_degrees}°. FAILED.")
            else:
                deviation_from_vertical = abs(90.0 - abs(angle_r))
                if deviation_from_vertical < max_bottle_tilt_degrees:
                    is_upright = True
                    print(f"    Contour {i} Upright Check: deviation={deviation_from_vertical:.1f}° vs max_tilt={max_bottle_tilt_degrees}°. PASSED.")
                else:
                    print(f"    Contour {i} Upright Check: deviation={deviation_from_vertical:.1f}° vs max_tilt={max_bottle_tilt_degrees}°. FAILED.")

            if not is_upright:
                print(f"    Contour {i} REJECTED: Not sufficiently upright.")
                continue

            if area > max_area_found:
                print(f"      Contour {i} is now the best candidate.")
                max_area_found = area
                valid_bottle = {
                    "contour": cnt,
                    "box_points": box_pts_debug,
                    "pixel_width": current_rect_w,
                    "pixel_height": current_rect_h,
                    "pixel_area": area
                }
        cv2.imshow("ROI Contours (Debug)", roi_debug_contours_img)
    print("--- Exiting find_bottle_in_roi ---")
    return valid_bottle

def estimate_and_classify(bottle_info, ppm, known_specs, tolerance_percent):
    """
    Estimates bottle volume and classifies it based on known specifications.
    """
    if ppm == 0 or ppm is None:
        print("Error: Pixels Per CM (ppm) is zero or None. Calibration error.")
        bottle_info["classification"] = "Calibration Error"
        bottle_info["confidence_percent"] = 0
        bottle_info["estimated_volume_ml"] = 0
        bottle_info["real_height_cm"] = 0
        bottle_info["real_diameter_cm"] = 0
        return bottle_info

    pixel_h = bottle_info["pixel_height"]
    pixel_w = bottle_info["pixel_width"]

    real_height_cm = pixel_h / ppm
    real_diameter_cm = pixel_w / ppm
    radius_cm = real_diameter_cm / 2.0
    estimated_volume_ml = np.pi * (radius_cm ** 2) * real_height_cm

    bottle_info["real_height_cm"] = real_height_cm
    bottle_info["real_diameter_cm"] = real_diameter_cm
    bottle_info["estimated_volume_ml"] = estimated_volume_ml

    print(f"  DEBUG: Bottle Real Height: {real_height_cm:.2f} cm, Real Diameter: {real_diameter_cm:.2f} cm")
    print(f"  DEBUG: Estimated Volume (mL): {estimated_volume_ml:.2f}")

    best_match_label = "Other"
    min_diff_percent = float('inf')

    for label, spec in known_specs.items():
        target_volume = spec["volume_ml"]
        volume_diff_percent = (abs(estimated_volume_ml - target_volume) / target_volume) * 100
        if volume_diff_percent < min_diff_percent:
            min_diff_percent = volume_diff_percent
            best_match_label = label

    if min_diff_percent <= tolerance_percent:
        bottle_info["classification"] = best_match_label
        bottle_info["confidence_percent"] = 100 - min_diff_percent
    else:
        bottle_info["classification"] = f"Other ({estimated_volume_ml:.0f}mL)"
        bottle_info["confidence_percent"] = max(0, 100 - min_diff_percent)
    return bottle_info

def draw_aruco_info(image, marker_data, calibration_result):
    """
    Draws ArUco marker information and pose on the image.
    """
    display_image = image.copy()
    
    for marker in marker_data:
        # Draw marker outline
        corners = marker["corners"].astype(int)
        cv2.polylines(display_image, [corners], True, (0, 255, 0), 2)
        
        # Draw marker ID
        center = marker["center"]
        cv2.putText(display_image, f"ID: {marker['id']}", 
                   (center[0] - 30, center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw pose axes (optional, compatible with different OpenCV versions)
        try:
            aruco.drawAxis(display_image, CAMERA_MATRIX, DIST_COEFFS, 
                          marker["rvec"], marker["tvec"], KNOWN_MARKER_SIZE_CM * 0.5)
        except AttributeError:
            cv2.drawFrameAxes(display_image, CAMERA_MATRIX, DIST_COEFFS, 
                             marker["rvec"], marker["tvec"], KNOWN_MARKER_SIZE_CM * 0.5, 3)
        
        # Draw calibration info
        info_text = f"Dist: {marker['distance']:.1f}cm, Angle: {marker['viewing_angle_deg']:.1f}°"
        cv2.putText(display_image, info_text,
                   (center[0] - 50, center[1] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
    
    if calibration_result:
        ppm, ref_marker = calibration_result
        cv2.putText(display_image, f"Corrected PPM: {ppm:.2f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    
    return display_image

# --- Main Execution ---
if __name__ == "__main__":
    image_path = "data_6.jpg"  # GANTI dengan nama file foto Anda

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from '{image_path}'. Please check the path.")
        # Create a placeholder image if loading fails
        original_image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(original_image, "Image not found", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    display_image = original_image.copy()
    current_pixels_per_cm = None
    ref_marker_data = None

    print("Detecting ArUco markers...")
    marker_data = detect_aruco_markers(original_image, ARUCO_DICT, ARUCO_PARAMS)

    if marker_data:
        # Calculate perspective-corrected calibration
        calibration_result = calculate_perspective_corrected_calibration(marker_data)
        
        if calibration_result:
            current_pixels_per_cm, ref_marker_data = calibration_result
            
            # Draw ArUco information on display image
            display_image = draw_aruco_info(display_image, marker_data, calibration_result)
            
            # Define ROI for bottle detection (area above the reference marker)
            ref_center = ref_marker_data["center"]
            ref_corners = ref_marker_data["corners"]
            
            # Find the topmost point of the marker
            ref_top_y = int(np.min(ref_corners[:, 1]))
            ref_left_x = int(np.min(ref_corners[:, 0]))
            ref_right_x = int(np.max(ref_corners[:, 0]))
            
            # Define ROI
            roi_y_start = 0
            separation_pixels = 10
            roi_y_end = max(0, ref_top_y - separation_pixels)
            roi_x_start = max(0, ref_left_x - 100)  # Extend search area
            roi_x_end = min(original_image.shape[1], ref_right_x + 100)
            
            if roi_y_end > roi_y_start and roi_x_end > roi_x_start:
                bottle_roi = original_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                
                if bottle_roi.size > 0:
                    cv2.rectangle(display_image, (roi_x_start, roi_y_start), 
                                 (roi_x_end, roi_y_end), (0, 255, 255), 2)
                    cv2.imshow("Bottle ROI", bottle_roi)
                    
                    print("Processing ROI for bottle detection...")
                    processed_bottle_roi = preprocess_roi_for_bottle(bottle_roi.copy())
                    
                    # Dynamic minimum area based on corrected calibration
                    min_area_cm_sq = 2
                    dynamic_min_bottle_area_pixels = (current_pixels_per_cm**2) * min_area_cm_sq
                    
                    bottle_data = find_bottle_in_roi(processed_bottle_roi,
                                                     min_bottle_area_pixels=dynamic_min_bottle_area_pixels,
                                                     min_bottle_aspect_ratio=1.2,
                                                     max_bottle_tilt_degrees=30)
                    
                    if bottle_data:
                        print("Bottle found in ROI.")
                        # Adjust coordinates back to original image frame
                        bottle_data["contour"][:, 0, 0] += roi_x_start
                        bottle_data["contour"][:, 0, 1] += roi_y_start
                        bottle_data["box_points"][:, 0] += roi_x_start
                        bottle_data["box_points"][:, 1] += roi_y_start
                        
                        cv2.drawContours(display_image, [bottle_data["contour"]], -1, (0, 255, 0), 2)
                        cv2.drawContours(display_image, [bottle_data["box_points"]], 0, (0, 0, 255), 2)
                        
                        classified_bottle_data = estimate_and_classify(bottle_data,
                                                                       current_pixels_per_cm,
                                                                       KNOWN_BOTTLE_SPECS,
                                                                       CLASSIFICATION_TOLERANCE_PERCENT)
                        
                        print(f"--- Bottle Analysis Result (ArUco Corrected) ---")
                        print(f"  Real Dimensions (HxW): {classified_bottle_data['real_height_cm']:.1f} x {classified_bottle_data['real_diameter_cm']:.1f} cm")
                        print(f"  Estimated Volume: {classified_bottle_data['estimated_volume_ml']:.0f} mL")
                        print(f"  Classified As: {classified_bottle_data['classification']}")
                        print(f"  Confidence: {classified_bottle_data['confidence_percent']:.0f}%")
                        print(f"  Perspective Correction Applied: {ref_marker_data['viewing_angle_deg']:.1f}° viewing angle")
                        print(f"-------------------------------")
                        
                        text = f"{classified_bottle_data['classification']} ({classified_bottle_data['estimated_volume_ml']:.0f}mL)"
                        text_pos = (bottle_data["box_points"][0][0], bottle_data["box_points"][0][1] - 10)
                        cv2.putText(display_image, text, text_pos,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    else:
                        print("No suitable bottle found in the defined ROI after filtering.")
                else:
                    print("Error: Bottle ROI is empty.")
            else:
                print("Could not define a valid ROI.")
        else:
            cv2.putText(display_image, "Calibration Error!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        print("No ArUco markers found. Cannot proceed with measurements.")
        cv2.putText(display_image, "No ArUco markers found!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display results
    cv2.imshow("Original Image", original_image)
    resized_display = cv2.resize(display_image, (800, 1000))
    cv2.imshow("ArUco Detections", resized_display)
    
    print("\nPress 'q' or ESC to exit, or close windows.")
    print("INSTRUCTIONS:")
    print("1. Print an ArUco marker (ID 0-249 from DICT_6X6_250)")
    print("2. Make sure the marker is exactly 5.0cm x 5.0cm")
    print("3. Place it as reference in your image")
    print("4. The system will automatically detect and correct for perspective!")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        try:
            if cv2.getWindowProperty("Original Image", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.getWindowProperty("ArUco Detections", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyAllWindows() 