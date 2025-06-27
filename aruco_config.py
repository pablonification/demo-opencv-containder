"""
ArUco Configuration File for Fine-tuning
Semua parameter kalibrasi dan fine-tuning dikumpulkan di sini
"""
import numpy as np
from cv2 import aruco

# ========================================
# 1. MARKER CONFIGURATION
# ========================================

# Ukuran fisik marker yang sudah diprint (UKUR DENGAN PENGGARIS!)
KNOWN_MARKER_SIZE_CM = 5.0  # UBAH ini sesuai ukuran nyata marker Anda

# Dictionary ArUco yang digunakan
ARUCO_DICT_TYPE = aruco.DICT_6X6_250  # Bisa diganti ke DICT_4X4_50, DICT_7X7_250, dll

# ========================================
# 2. DETECTION PARAMETERS (Fine-tuning)
# ========================================

def get_aruco_params():
    """
    Konfigurasi detection parameters untuk fine-tuning
    """
    try:
        params = aruco.DetectorParameters_create()
    except AttributeError:
        params = aruco.DetectorParameters()
    
    # === ADAPTIVE THRESHOLD (untuk lighting berbeda) ===
    params.adaptiveThreshWinSizeMin = 3        # Min: 3, Max: 23
    params.adaptiveThreshWinSizeMax = 23       # Untuk marker kecil: turunkan ke 15
    params.adaptiveThreshWinSizeStep = 10      # Step size
    params.adaptiveThreshConstant = 7          # Threshold constant
    
    # === CONTOUR FILTERING ===
    params.minMarkerPerimeterRate = 0.03       # Min perimeter (turunkan jika marker kecil)
    params.maxMarkerPerimeterRate = 4.0        # Max perimeter
    params.polygonalApproxAccuracyRate = 0.03  # Polygon approximation
    
    # === CORNER DETECTION ===
    params.minCornerDistanceRate = 0.05        # Min distance between corners
    params.minDistanceToBorder = 3             # Min distance to image border
    
    # === MARKER VALIDATION ===
    params.minMarkerDistanceRate = 0.05        # Min distance between markers
    params.minOtsuStdDev = 5.0                # Otsu threshold std dev
    
    # === PERSPECTIVE REMOVAL ===
    params.perspectiveRemovePixelPerCell = 4   # Pixels per cell
    params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    
    # === ERROR CORRECTION ===
    params.maxErroneousBitsInBorderRate = 0.35 # Max error rate in border
    params.errorCorrectionRate = 0.6           # Error correction rate
    
    return params

# ========================================
# 3. CAMERA CALIBRATION
# ========================================

# Default camera parameters (HARUS dikalibrasi untuk akurasi tinggi!)
def get_camera_params():
    """
    Parameter kamera - kalibrasi dengan checkerboard untuk hasil terbaik
    """
    # Resolusi gambar tipikal
    image_width = 640
    image_height = 480
    
    # Focal length estimate (adjust based on your camera)
    fx = fy = 800  # Untuk webcam biasa: 500-1000
    
    # Principal point (biasanya di tengah gambar)
    cx = image_width / 2   # 320 untuk 640px width
    cy = image_height / 2  # 240 untuk 480px height
    
    camera_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], dtype=np.float32)
    
    # Distortion coefficients (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)
    
    return camera_matrix, dist_coeffs

# ========================================
# 4. BOTTLE SPECIFICATIONS
# ========================================

# Spesifikasi botol yang dikenali sistem
KNOWN_BOTTLE_SPECS = {
    "200mL": {
        "volume_ml": 200, 
        "avg_diameter_cm": 5.0, 
        "avg_height_cm": 13.0
    },
    "330mL": {  # Tambahan untuk kaleng
        "volume_ml": 330, 
        "avg_diameter_cm": 6.6, 
        "avg_height_cm": 12.3
    },
    "500mL": {
        "volume_ml": 500, 
        "avg_diameter_cm": 6.5, 
        "avg_height_cm": 20.0
    },
    "600mL": {
        "volume_ml": 600, 
        "avg_diameter_cm": 7.0, 
        "avg_height_cm": 22.0
    },
    "1000mL": {
        "volume_ml": 1000, 
        "avg_diameter_cm": 8.0, 
        "avg_height_cm": 25.0
    },
    "1500mL": {
        "volume_ml": 1500, 
        "avg_diameter_cm": 9.5, 
        "avg_height_cm": 30.0
    },
}

# Toleransi klasifikasi (%)
CLASSIFICATION_TOLERANCE_PERCENT = 30  # Bisa diturunkan ke 20 untuk lebih strict

# ========================================
# 5. ROI & DETECTION PARAMETERS
# ========================================

# Minimum area botol dalam cm² (untuk filtering)
MIN_BOTTLE_AREA_CM_SQ = 2

# Aspect ratio minimum untuk botol (tinggi/lebar)
MIN_BOTTLE_ASPECT_RATIO = 1.2  # Botol harus lebih tinggi dari lebar

# Maximum tilt yang diizinkan untuk botol (derajat)
MAX_BOTTLE_TILT_DEGREES = 30

# ROI extension (pixels) - area pencarian botol diperluas dari marker
ROI_EXTENSION_PIXELS = 100

# Separation pixels antara marker dan ROI
ROI_SEPARATION_PIXELS = 10

# ========================================
# 6. PERSPECTIVE CORRECTION
# ========================================

# Distance correction factor (experimental)
DISTANCE_CORRECTION_FACTOR = 1.0

# Maximum viewing angle yang masih bisa dikoreksi (derajat)
MAX_VIEWING_ANGLE_DEGREES = 60

# ========================================
# 7. FINE-TUNING PRESETS
# ========================================

def get_preset_config(preset_name):
    """
    Preset konfigurasi untuk kondisi berbeda
    """
    presets = {
        "indoor_bright": {
            "adaptiveThreshConstant": 7,
            "minMarkerPerimeterRate": 0.03,
            "marker_size_cm": 5.0
        },
        "indoor_dim": {
            "adaptiveThreshConstant": 10,
            "minMarkerPerimeterRate": 0.02,
            "marker_size_cm": 5.0
        },
        "outdoor": {
            "adaptiveThreshConstant": 5,
            "minMarkerPerimeterRate": 0.04,
            "marker_size_cm": 5.0
        },
        "phone_camera": {
            "fx": 1000, "fy": 1000,
            "cx": 320, "cy": 240,
            "marker_size_cm": 5.0
        },
        "webcam": {
            "fx": 600, "fy": 600,
            "cx": 320, "cy": 240,
            "marker_size_cm": 5.0
        }
    }
    
    return presets.get(preset_name, {})

# ========================================
# 8. QUICK TUNING HELPERS
# ========================================

def print_current_config():
    """
    Print konfigurasi saat ini untuk debugging
    """
    print("=== CURRENT ARUCO CONFIGURATION ===")
    print(f"Marker Size: {KNOWN_MARKER_SIZE_CM}cm")
    print(f"Dictionary: {ARUCO_DICT_TYPE}")
    print(f"Classification Tolerance: {CLASSIFICATION_TOLERANCE_PERCENT}%")
    print(f"Bottle Specs: {len(KNOWN_BOTTLE_SPECS)} types")
    print("===================================")

if __name__ == "__main__":
    print_current_config()
    
    print("\n=== QUICK TUNING GUIDE ===")
    print("1. KNOWN_MARKER_SIZE_CM = Ukur marker fisik dengan penggaris")
    print("2. Untuk marker tidak terdeteksi:")
    print("   - Turunkan minMarkerPerimeterRate ke 0.02")
    print("   - Naikkan adaptiveThreshConstant ke 10")
    print("3. Untuk akurasi tinggi:")
    print("   - Kalibrasi kamera dengan checkerboard")
    print("   - Update camera_matrix dan dist_coeffs")
    print("4. Untuk botol baru:")
    print("   - Tambah spesifikasi di KNOWN_BOTTLE_SPECS")
    print("==========================") 