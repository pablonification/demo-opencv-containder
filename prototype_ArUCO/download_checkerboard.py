"""
Download/Generate Checkerboard Pattern for Camera Calibration
Script ini akan membuat checkerboard pattern untuk kalibrasi kamera
"""

import cv2
import numpy as np

def create_checkerboard_pattern(rows=7, cols=10, square_size_pixels=50, save_path="checkerboard_pattern.png"):
    """
    Create checkerboard pattern untuk kalibrasi kamera
    
    Args:
        rows: jumlah baris kotak (squares, bukan corners)
        cols: jumlah kolom kotak (squares, bukan corners)  
        square_size_pixels: ukuran setiap kotak dalam pixels
        save_path: path untuk menyimpan file
    """
    
    # Ukuran total image
    img_height = rows * square_size_pixels
    img_width = cols * square_size_pixels
    
    # Create checkerboard
    checkerboard = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            # Alternating pattern: putih (255) atau hitam (0)
            if (i + j) % 2 == 0:
                color = 255  # Putih
            else:
                color = 0    # Hitam
            
            # Fill square
            y_start = i * square_size_pixels
            y_end = (i + 1) * square_size_pixels
            x_start = j * square_size_pixels  
            x_end = (j + 1) * square_size_pixels
            
            checkerboard[y_start:y_end, x_start:x_end] = color
    
    # Add white border untuk printing
    border_size = 50
    bordered_checkerboard = np.ones((img_height + 2*border_size, img_width + 2*border_size), dtype=np.uint8) * 255
    bordered_checkerboard[border_size:border_size+img_height, border_size:border_size+img_width] = checkerboard
    
    # Save image
    cv2.imwrite(save_path, bordered_checkerboard)
    
    print(f"✓ Checkerboard pattern saved: {save_path}")
    print(f"  Size: {rows}x{cols} squares")
    print(f"  Internal corners: {cols-1}x{rows-1}")
    print(f"  Image size: {bordered_checkerboard.shape[1]}x{bordered_checkerboard.shape[0]} pixels")
    print(f"  Square size: {square_size_pixels} pixels")
    print()
    print("PRINTING INSTRUCTIONS:")
    print("1. Print this image at actual size (100% scale, no scaling)")
    print("2. Use thick matte paper (avoid glossy)")
    print("3. Measure one square dengan penggaris untuk menentukan ukuran fisik")
    print("4. Gunakan ukuran fisik ini dalam camera_calibration.py")
    
    return bordered_checkerboard

def main():
    print("=== CHECKERBOARD PATTERN GENERATOR ===")
    print("Membuat pattern untuk kalibrasi kamera")
    print()
    
    # Standard checkerboard configurations
    presets = {
        "1": {"rows": 7, "cols": 10, "desc": "7x10 squares (recommended)"},
        "2": {"rows": 6, "cols": 9, "desc": "6x9 squares (kompak)"},
        "3": {"rows": 8, "cols": 11, "desc": "8x11 squares (large)"},
        "4": {"rows": 5, "cols": 8, "desc": "5x8 squares (small)"}
    }
    
    print("Pilih ukuran checkerboard:")
    for key, preset in presets.items():
        print(f"{key}. {preset['desc']}")
    print("5. Custom size")
    
    choice = input("\nPilihan (1-5): ").strip()
    
    if choice in presets:
        rows = presets[choice]["rows"]
        cols = presets[choice]["cols"]
    elif choice == "5":
        try:
            rows = int(input("Jumlah baris squares: "))
            cols = int(input("Jumlah kolom squares: "))
        except ValueError:
            print("Input tidak valid, menggunakan default 7x10")
            rows, cols = 7, 10
    else:
        print("Pilihan tidak valid, menggunakan default 7x10")
        rows, cols = 7, 10
    
    # Square size
    square_size = input(f"Ukuran square dalam pixels (default 50): ").strip()
    square_size = int(square_size) if square_size.isdigit() else 50
    
    # Generate pattern
    pattern = create_checkerboard_pattern(rows, cols, square_size)
    
    # Show preview
    preview = cv2.resize(pattern, (600, int(600 * pattern.shape[0] / pattern.shape[1])))
    cv2.imshow("Checkerboard Pattern Preview", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n=== NEXT STEPS ===")
    print("1. Print checkerboard_pattern.png")
    print("2. Jalankan: python camera_calibration.py")
    print("3. Gunakan ukuran square yang terukur untuk kalibrasi")

if __name__ == "__main__":
    main() 