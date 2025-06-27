import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt

def generate_aruco_marker(marker_id, marker_size_pixels=200, save_image=True):
    """
    Generate ArUco marker and save as image file.
    
    Args:
        marker_id: ID of the marker (0-249 for DICT_6X6_250)
        marker_size_pixels: Size of the marker in pixels
        save_image: Whether to save the image file
    """
    # Get the ArUco dictionary (compatible with different OpenCV versions)
    try:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    except AttributeError:
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # Generate the marker (compatible with different OpenCV versions)
    try:
        marker_img = aruco.drawMarker(aruco_dict, marker_id, marker_size_pixels)
    except AttributeError:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
    
    if save_image:
        filename = f"aruco_marker_id_{marker_id}_size_{marker_size_pixels}px.png"
        cv2.imwrite(filename, marker_img)
        print(f"ArUco marker saved as: {filename}")
        print(f"Print this at exactly 5.0cm x 5.0cm for accurate measurements!")
    
    return marker_img

def generate_multiple_markers(num_markers=5, marker_size_pixels=200):
    """
    Generate multiple ArUco markers and display them.
    """
    try:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    except AttributeError:
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # Create a grid to display multiple markers
    cols = min(num_markers, 3)
    rows = (num_markers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(num_markers):
        try:
            marker_img = aruco.drawMarker(aruco_dict, i, marker_size_pixels)
        except AttributeError:
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size_pixels)
        
        # Save individual marker
        filename = f"aruco_marker_id_{i}.png"
        cv2.imwrite(filename, marker_img)
        
        # Display in subplot
        if i < len(axes):
            axes[i].imshow(marker_img, cmap='gray')
            axes[i].set_title(f'ArUco ID: {i}')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_markers, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('aruco_markers_collection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Generated {num_markers} ArUco markers")
    print("IMPORTANT: Print each marker at exactly 5.0cm x 5.0cm")

def create_marker_with_border(marker_id, marker_size_cm=5.0, dpi=300, border_ratio=0.2):
    """
    Create ArUco marker with white border for better detection.
    """
    try:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    except AttributeError:
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # Calculate pixel size based on DPI and desired physical size
    marker_size_pixels = int(marker_size_cm * dpi / 2.54)  # Convert cm to inches then to pixels
    
    # Generate the core marker (compatible with different OpenCV versions)
    try:
        marker_core = aruco.drawMarker(aruco_dict, marker_id, marker_size_pixels)
    except AttributeError:
        marker_core = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
    
    # Add white border
    border_pixels = int(marker_size_pixels * border_ratio)
    total_size = marker_size_pixels + 2 * border_pixels
    
    # Create white background
    marker_with_border = np.ones((total_size, total_size), dtype=np.uint8) * 255
    
    # Place marker in center
    start_pos = border_pixels
    end_pos = start_pos + marker_size_pixels
    marker_with_border[start_pos:end_pos, start_pos:end_pos] = marker_core
    
    filename = f"aruco_marker_id_{marker_id}_with_border_{marker_size_cm}cm.png"
    cv2.imwrite(filename, marker_with_border)
    
    print(f"ArUco marker with border saved as: {filename}")
    print(f"Total size: {marker_size_cm + 2*marker_size_cm*border_ratio:.1f}cm x {marker_size_cm + 2*marker_size_cm*border_ratio:.1f}cm")
    print(f"Print at {dpi} DPI for accurate size")
    
    return marker_with_border

if __name__ == "__main__":
    print("=== ArUco Marker Generator ===")
    print("This script generates ArUco markers for bottle detection calibration")
    print()
    
    # Generate a single marker (recommended)
    print("1. Generating single marker (ID 0)...")
    marker = generate_aruco_marker(marker_id=0, marker_size_pixels=400)
    
    print()
    print("2. Generating marker with border...")
    marker_with_border = create_marker_with_border(marker_id=0, marker_size_cm=5.0)
    
    print()
    print("3. Generating multiple markers for backup...")
    generate_multiple_markers(num_markers=5)
    
    print()
    print("=== PRINTING INSTRUCTIONS ===")
    print("1. Use the marker with border for best detection")
    print("2. Print at 300 DPI or higher")
    print("3. Measure printed marker to ensure it's exactly 5.0cm x 5.0cm")
    print("4. Use thick, matte paper (avoid glossy paper)")
    print("5. Ensure good lighting when using the marker")
    print()
    print("The markers are now ready for use with aruco_bottle_detector.py!") 