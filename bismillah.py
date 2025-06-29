import cv2
import numpy as np
import imutils # For convenience functions like resizing

# --- 1. Configuration & Calibration ---

# == REFERENCE OBJECT (BLUE RECTANGLE) CONFIGURATION ==
KNOWN_REF_OBJECT_HEIGHT_CM = 16
BLUE_LOWER = np.array([0, 0, 0])
BLUE_UPPER = np.array([180, 255, 50])


# == BOTTLE CONFIGURATION (Same as before) ==
KNOWN_BOTTLE_SPECS = {
    "200mL": {"volume_ml": 200, "avg_diameter_cm": 5.0, "avg_height_cm": 13.0},
    "500mL": {"volume_ml": 500, "avg_diameter_cm": 6.5, "avg_height_cm": 20.0},
    "1000mL": {"volume_ml": 1000, "avg_diameter_cm": 8.0, "avg_height_cm": 25.0},
}
CLASSIFICATION_TOLERANCE_PERCENT = 30

# --- 2. Helper Functions ---

def detect_blue_rectangle(image, lower_blue, upper_blue):
    """
    Detects the largest blue rectangular object in the image.
    Returns its contour, minAreaRect box points, pixel width, and pixel height from minAreaRect.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imshow("Blue Mask", mask) # For debugging

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect_candidate = None
    max_area = 0

    if contours:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500: # Min area to be considered
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

            if len(approx) >= 4 and len(approx) <= 6 :
                rect = cv2.minAreaRect(cnt) # ((center_x, center_y), (width, height), angle)
                (x,y), (w,h), angle = rect
                if area > max_area:
                    max_area = area
                    box_points = cv2.boxPoints(rect)
                    box_points = np.intp(box_points)
                    # Ensure pixel_height is always the larger dimension and pixel_width the smaller for the reference
                    # This assumes the KNOWN_REF_OBJECT_HEIGHT_CM refers to its longest dimension if it's rectangular
                    # If the reference can be oriented such that its 'width' is known, adjust this logic.
                    ref_pixel_height = max(w, h)
                    ref_pixel_width = min(w, h)

                    best_rect_candidate = {
                        "contour": cnt,
                        "box_points": box_points,
                        "pixel_width": ref_pixel_width, # Using the shorter dimension as width
                        "pixel_height": ref_pixel_height, # Using the longer dimension as height for calibration
                        "center_x": int(x),
                        "center_y": int(y),
                        "angle": angle
                    }
    return best_rect_candidate

def preprocess_roi_for_bottle(roi):
    """
    Applies robust preprocessing to the Region of Interest (ROI) to isolate the bottle.
    Focuses on precise edge detection and shadow reduction.
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
            (center_x,center_y), (rect_w_raw,rect_h_raw), angle_r = rect_debug
            box_pts_debug = cv2.boxPoints(rect_debug)
            box_pts_debug = np.intp(box_pts_debug)

            # Determine visual height and width
            current_rect_h = max(rect_w_raw, rect_h_raw)
            current_rect_w = min(rect_w_raw, rect_h_raw)
            aspect_ratio_debug = current_rect_h / current_rect_w if current_rect_w > 0 else 0

            print(f"  Contour {i}: Area={area:.0f}, AspectRatio={aspect_ratio_debug:.2f}, Angle={angle_r:.1f}, RawDims(w,h)=({rect_w_raw:.1f},{rect_h_raw:.1f})")
            cv2.drawContours(roi_debug_contours_img, [cnt], -1, (0,0,255), 1)
            cv2.drawContours(roi_debug_contours_img, [box_pts_debug], 0, (255,0,0), 1)

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

            # === Upright Bottle Filter (MODIFIED) ===
            is_upright = False
            if rect_h_raw >= rect_w_raw: # If minAreaRect's 'height' (rect_h_raw) is the visual height.
                                         # This means rect_w_raw is the visual width.
                                         # For the bottle to be upright, rect_w_raw should be horizontal.
                                         # angle_r is the angle of rect_w_raw from the horizontal.
                                         # So, abs(angle_r) should be small.
                if abs(angle_r) < max_bottle_tilt_degrees:
                    is_upright = True
                    print(f"    Contour {i} Upright Check (h_raw is main): angle_r={angle_r:.1f} vs tilt={max_bottle_tilt_degrees}. PASSED.")
                else:
                    print(f"    Contour {i} Upright Check (h_raw is main): angle_r={angle_r:.1f} vs tilt={max_bottle_tilt_degrees}. FAILED.")
            else: # minAreaRect's 'width' (rect_w_raw) is the visual height.
                  # This means rect_h_raw is the visual width.
                  # For the bottle to be upright, rect_w_raw should be vertical.
                  # angle_r is the angle of rect_w_raw from the horizontal.
                  # So, abs(angle_r) should be close to 90 degrees.
                  # Deviation from vertical is abs(90 - abs(angle_r)).
                deviation_from_vertical = abs(90.0 - abs(angle_r))
                if deviation_from_vertical < max_bottle_tilt_degrees:
                    is_upright = True
                    print(f"    Contour {i} Upright Check (w_raw is main): angle_r={angle_r:.1f}, dev_from_vert={deviation_from_vertical:.1f} vs tilt={max_bottle_tilt_degrees}. PASSED.")
                else:
                    print(f"    Contour {i} Upright Check (w_raw is main): angle_r={angle_r:.1f}, dev_from_vert={deviation_from_vertical:.1f} vs tilt={max_bottle_tilt_degrees}. FAILED.")

            if not is_upright:
                print(f"    Contour {i} REJECTED: Not sufficiently upright. Angle={angle_r:.1f} (w_raw={rect_w_raw:.1f}, h_raw={rect_h_raw:.1f}), MaxTilt={max_bottle_tilt_degrees} deg.")
                continue
            # === End Upright Bottle Filter ===

            print(f"    Contour {i} PASSED filters. Current max_area_found: {max_area_found}")
            if area > max_area_found:
                print(f"      Contour {i} is now the best candidate.")
                max_area_found = area
                valid_bottle = {
                    "contour": cnt,
                    "box_points": box_pts_debug,
                    "pixel_width": current_rect_w, # Use the visual width
                    "pixel_height": current_rect_h, # Use the visual height
                    "pixel_area": area
                }
        cv2.imshow("ROI Contours (Debug)", roi_debug_contours_img)
    print("--- Exiting find_bottle_in_roi ---")
    return valid_bottle


def estimate_and_classify(bottle_info, ppm, known_specs, tolerance_percent):
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

# --- Main Execution ---
if __name__ == "__main__":
    image_path = "hitam.jpg" # GANTI dengan path gambar Anda / Gunakan nama gambar yang diunggah jika menguji dengannya

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from '{image_path}'. Please check the path.")
        # Create a placeholder image if loading fails
        original_image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(original_image, "Image not found", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        # Draw a dummy reference object and a dummy bottle area for testing flow if image fails
        cv2.rectangle(original_image, (200, 400), (200 + int(10*11.97), 400 + int(KNOWN_REF_OBJECT_HEIGHT_CM*11.97)), (255,0,0), -1) # Dummy ref
        cv2.rectangle(original_image, (150, 100), (350, 390), (200,200,200), -1) # Dummy bottle

    display_image = original_image.copy()
    current_pixels_per_cm = None

    print("Detecting blue reference object...")
    ref_object_data = detect_blue_rectangle(original_image, BLUE_LOWER, BLUE_UPPER)

    if ref_object_data:
        # For calibration, KNOWN_REF_OBJECT_HEIGHT_CM refers to the physical height.
        # The 'pixel_height' from detect_blue_rectangle is already the longer dimension.
        pixel_dim_for_calibration = ref_object_data['pixel_height']
        print(f"Blue reference object found. Using its pixel dimension (longest side): {pixel_dim_for_calibration:.2f}px for calibration.")
        cv2.drawContours(display_image, [ref_object_data["box_points"]], 0, (255, 100, 0), 2)
        cv2.putText(display_image, "Reference",
                    (ref_object_data["box_points"][0][0], ref_object_data["box_points"][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        if pixel_dim_for_calibration > 0:
            current_pixels_per_cm = pixel_dim_for_calibration / KNOWN_REF_OBJECT_HEIGHT_CM
            print(f"Calculated PIXELS_PER_CM: {current_pixels_per_cm:.2f}")
        else:
            print("Error: Reference object detected with zero for the calibration dimension.")
            current_pixels_per_cm = None
    else:
        print("Blue reference object not found. Cannot proceed with measurements.")
        current_pixels_per_cm = None

    if ref_object_data and current_pixels_per_cm is not None:
        ref_box_pts = ref_object_data["box_points"]
        all_x_coords = ref_box_pts[:, 0]
        all_y_coords = ref_box_pts[:, 1]

        ref_min_x = np.min(all_x_coords)
        ref_max_x = np.max(all_x_coords)
        ref_top_y = np.min(all_y_coords) # Topmost y-coordinate of the reference object

        # Define ROI for bottle (area above the reference object)
        roi_y_start = 0 # Start from the top of the image

        separation_pixels = 5 # Small gap above the reference object
        roi_y_end = ref_top_y - separation_pixels
        roi_y_end = max(0, roi_y_end) # Ensure roi_y_end is not negative

        # ROI horizontal span: slightly wider than the reference object
        # You might want to make this wider if bottles are not always centered over the reference
        roi_x_start = max(0, ref_min_x - 50)  # Extend 50 pixels to the left
        roi_x_end = min(original_image.shape[1], ref_max_x + 50) # Extend 50 pixels to the right
        roi_y_start = max(0, roi_y_start)

        if roi_y_end > roi_y_start and roi_x_end > roi_x_start :
            bottle_roi = original_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            if bottle_roi.size == 0:
                print("Error: Bottle ROI is empty. Check ROI coordinates and reference object detection.")
            else:
                cv2.rectangle(display_image, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0,255,255), 1)
                cv2.imshow("Bottle ROI", bottle_roi)

                print("Processing ROI for bottle detection...")
                processed_bottle_roi = preprocess_roi_for_bottle(bottle_roi.copy())

                min_area_cm_sq = 2 # Minimum physical area for a bottle to be considered (e.g., 2 cm^2)
                dynamic_min_bottle_area_pixels = (current_pixels_per_cm**2) * min_area_cm_sq if current_pixels_per_cm else 1000

                bottle_data = find_bottle_in_roi(processed_bottle_roi,
                                                 min_bottle_area_pixels=dynamic_min_bottle_area_pixels,
                                                 min_bottle_aspect_ratio=1.2, # Bottles are generally taller than wide
                                                 max_bottle_tilt_degrees=30) # Max allowed tilt from vertical in degrees

                if bottle_data:
                    print("Bottle found in ROI.")
                    # Adjust contour and box points coordinates back to the original image frame
                    bottle_data["contour"][:, 0, 0] += roi_x_start
                    bottle_data["contour"][:, 0, 1] += roi_y_start
                    bottle_data["box_points"][:, 0] += roi_x_start
                    bottle_data["box_points"][:, 1] += roi_y_start

                    cv2.drawContours(display_image, [bottle_data["contour"]], -1, (0, 255, 0), 2)
                    cv2.drawContours(display_image, [bottle_data["box_points"]], 0, (0,0,255), 2) # Rotated rect

                    classified_bottle_data = estimate_and_classify(bottle_data,
                                                                   current_pixels_per_cm,
                                                                   KNOWN_BOTTLE_SPECS,
                                                                   CLASSIFICATION_TOLERANCE_PERCENT)

                    print(f"--- Bottle Analysis Result ---")
                    print(f"  Real Dimensions (HxW): {classified_bottle_data['real_height_cm']:.1f} x {classified_bottle_data['real_diameter_cm']:.1f} cm")
                    print(f"  Estimated Volume: {classified_bottle_data['estimated_volume_ml']:.0f} mL")
                    print(f"  Classified As: {classified_bottle_data['classification']}")
                    print(f"  Confidence: {classified_bottle_data['confidence_percent']:.0f}%")
                    print(f"-------------------------------")

                    text = f"{classified_bottle_data['classification']} ({classified_bottle_data['estimated_volume_ml']:.0f}mL)"
                    # Position text above the bottle
                    text_y_candidate = bottle_data["box_points"][0][1] - 10 # Use one of the top points of the box
                    # Ensure text is within image bounds
                    if text_y_candidate < 20 and roi_y_start < original_image.shape[0] - 20 :
                        text_pos_y = bottle_data["box_points"][1][1] + 20 # Try below if too high and space allows
                    elif text_y_candidate < 20 :
                        text_pos_y = 20 # Default to top if still too high
                    else:
                        text_pos_y = text_y_candidate
                    text_pos_x = bottle_data["box_points"][0][0]
                    cv2.putText(display_image, text, (text_pos_x, text_pos_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                else:
                    print("No suitable bottle found in the defined ROI after filtering.")
        else:
            print("Could not define a valid ROI: ROI dimensions are not valid (e.g., roi_y_end <= roi_y_start or empty).")
    elif not ref_object_data:
        cv2.putText(display_image, "Reference not found!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    elif current_pixels_per_cm is None:
         cv2.putText(display_image, "Calibration Error!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)


    cv2.imshow("Original Image", original_image)
    resized_display =  cv2.resize(display_image, (800, 1000))
    cv2.imshow("Detections", resized_display)
    print("\nPress 'q' or ESC, or close 'Original Image'/'Detections' window to exit all windows.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC key
            print("Exit key pressed. Closing all windows...")
            break
        try:
            # Check if windows were closed by clicking the 'X'
            if cv2.getWindowProperty("Original Image", cv2.WND_PROP_VISIBLE) < 1:         
                print("'Original Image' window closed by user. Closing all windows...")
                break
            if cv2.getWindowProperty("Detections", cv2.WND_PROP_VISIBLE) < 1:
                print("'Detections' window closed by user. Closing all windows...")
                break
            # Add checks for other windows if they are critical
            if cv2.getWindowProperty("Bottle ROI", cv2.WND_PROP_AUTOSIZE) >= 0 and cv2.getWindowProperty("Bottle ROI", cv2.WND_PROP_VISIBLE) < 1: # Check if it was created then closed
                 print("'Bottle ROI' window closed by user. Closing all windows...")
                 break
            if cv2.getWindowProperty("4. ROI Canny Edges (Original)", cv2.WND_PROP_AUTOSIZE) >= 0 and cv2.getWindowProperty("4. ROI Canny Edges (Original)", cv2.WND_PROP_VISIBLE) < 1:
                 print("'ROI Canny Edges (Original)' window closed by user. Closing all windows...")
                 break
            if cv2.getWindowProperty("5. ROI Canny Edges (AFTER Closing)", cv2.WND_PROP_AUTOSIZE) >= 0 and cv2.getWindowProperty("5. ROI Canny Edges (AFTER Closing)", cv2.WND_PROP_VISIBLE) < 1:
                 print("'ROI Canny Edges (AFTER Closing)' window closed by user. Closing all windows...")
                 break
            if cv2.getWindowProperty("ROI Contours (Debug)", cv2.WND_PROP_AUTOSIZE) >= 0 and cv2.getWindowProperty("ROI Contours (Debug)", cv2.WND_PROP_VISIBLE) < 1:
                 print("'ROI Contours (Debug)' window closed by user. Closing all windows...")
                 break

        except cv2.error:
            # This handles cases where a window might not have been created yet (e.g., ROI window if no ref object)
            # or was already destroyed, preventing a crash.
            # If the main windows are gone, we should still break.
            try:
                if cv2.getWindowProperty("Original Image", cv2.WND_PROP_VISIBLE) < 1 or \
                   cv2.getWindowProperty("Detections", cv2.WND_PROP_VISIBLE) < 1:
                    print("Main window(s) closed or inaccessible. Closing all windows...")
                    break
            except cv2.error: # If even checking main windows fails, assume they're gone
                 print("Error checking window status or window no longer exists. Closing all windows...")
                 break
        if cv2.waitKey(1) == -1 and \
           cv2.getWindowProperty("Original Image", cv2.WND_PROP_VISIBLE) < 1 and \
           cv2.getWindowProperty("Detections", cv2.WND_PROP_VISIBLE) < 1:
            # Fallback if all windows are closed without explicit key press detection
            print("All main windows appear closed. Exiting.")
            break

    cv2.destroyAllWindows()