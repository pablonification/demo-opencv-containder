"""
Camera Calibration Script for ArUco Bottle Detector
Gunakan script ini untuk mengkalibrasi kamera Anda dengan pattern checkerboard
Hasil kalibrasi akan meningkatkan akurasi pengukuran botol secara signifikan.

INSTRUKSI:
1. Print checkerboard pattern (tersedia di: https://github.com/opencv/opencv/blob/master/doc/pattern.png)
2. Atau gunakan: cv2.samples.findFile("data/left01.jpg") untuk contoh
3. Ambil 15-20 foto checkerboard dari berbagai sudut dan jarak
4. Jalankan script ini untuk mendapatkan parameter kamera
"""

import cv2
import numpy as np
import glob
import os
import json
from datetime import datetime

class CameraCalibrator:
    def __init__(self, checkerboard_size=(9, 6), square_size_mm=25.0):
        """
        Inisialisasi kalibrasi kamera
        
        Args:
            checkerboard_size: (width, height) dalam internal corners (bukan squares)
            square_size_mm: ukuran satu kotak dalam milimeter
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.square_size_cm = square_size_mm / 10.0
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size_cm  # Convert to cm
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane
        
        self.image_size = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
        
    def capture_calibration_images(self, camera_index=0, num_images=20):
        """
        Mengambil foto kalibrasi secara real-time
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return False
        
        print(f"=== CAMERA CALIBRATION IMAGE CAPTURE ===")
        print(f"Target: {num_images} images")
        print(f"Checkerboard size: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} internal corners")
        print(f"Square size: {self.square_size_mm}mm")
        print("\nINSTRUCTIONS:")
        print("- Hold checkerboard pattern in view")
        print("- Move it to different positions and angles")
        print("- Press SPACE when checkerboard is detected (corners shown)")
        print("- Press 'q' to quit early")
        print("- Try to fill the entire image area with different positions")
        
        captured_count = 0
        
        # Create directory for calibration images
        os.makedirs("calibration_images", exist_ok=True)
        
        while captured_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from camera")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = frame.copy()
            
            # Find checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(
                gray, self.checkerboard_size, 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret_corners:
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                
                # Draw corners
                cv2.drawChessboardCorners(display_frame, self.checkerboard_size, corners_refined, ret_corners)
                
                # Show instruction to capture
                cv2.putText(display_frame, f"DETECTED! Press SPACE to capture ({captured_count}/{num_images})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Move to different position/angle after capture", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(display_frame, f"Checkerboard NOT detected ({captured_count}/{num_images})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "Show checkerboard pattern to camera", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow('Camera Calibration - Capture Images', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and ret_corners:  # Space to capture
                # Save the image and corner data
                filename = f"calibration_images/calib_{captured_count:02d}.jpg"
                cv2.imwrite(filename, frame)
                
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners_refined)
                
                captured_count += 1
                print(f"Captured image {captured_count}/{num_images}: {filename}")
                
                # Show confirmation
                confirm_frame = display_frame.copy()
                cv2.putText(confirm_frame, f"SAVED! ({captured_count}/{num_images})", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                cv2.imshow('Camera Calibration - Capture Images', confirm_frame)
                cv2.waitKey(500)  # Show confirmation for 500ms
                
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_count < 10:
            print(f"\nWARNING: Only {captured_count} images captured. Recommend at least 10-15 for good calibration.")
        
        return captured_count > 0
    
    def load_calibration_images(self, image_pattern="calibration_images/*.jpg"):
        """
        Load dan proses gambar kalibrasi dari file
        """
        images = glob.glob(image_pattern)
        
        if not images:
            print(f"No images found with pattern: {image_pattern}")
            return False
        
        print(f"Found {len(images)} calibration images")
        successful_detections = 0
        
        for i, image_path in enumerate(images):
            print(f"Processing {image_path}...")
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"  Error loading {image_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.image_size is None:
                self.image_size = gray.shape[::-1]  # (width, height)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, self.checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret:
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners_refined)
                successful_detections += 1
                print(f"  ✓ Detected checkerboard corners")
                
                # Optionally visualize
                if i < 5:  # Show first 5 detections
                    img_corners = img.copy()
                    cv2.drawChessboardCorners(img_corners, self.checkerboard_size, corners_refined, ret)
                    cv2.imshow(f'Calibration Image {i}', cv2.resize(img_corners, (600, 400)))
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()
            else:
                print(f"  ✗ Could not detect checkerboard in {image_path}")
        
        print(f"\nSuccessfully processed: {successful_detections}/{len(images)} images")
        return successful_detections > 0
    
    def calibrate_camera(self):
        """
        Melakukan kalibrasi kamera
        """
        if len(self.objpoints) == 0:
            print("No calibration data available. Capture or load images first.")
            return False
        
        print(f"\n=== PERFORMING CAMERA CALIBRATION ===")
        print(f"Using {len(self.objpoints)} image sets")
        print(f"Image size: {self.image_size}")
        
        # Perform camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None
        )
        
        if not ret:
            print("Camera calibration failed!")
            return False
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        # Calculate reprojection error
        total_error = 0
        total_points = 0
        
        for i in range(len(self.objpoints)):
            imgpoints_proj, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
            total_error += error * len(imgpoints_proj)
            total_points += len(imgpoints_proj)
        
        self.calibration_error = total_error / total_points
        
        print(f"\n=== CALIBRATION RESULTS ===")
        print(f"Reprojection Error: {self.calibration_error:.3f} pixels")
        print(f"Camera Matrix (fx, fy, cx, cy):")
        print(f"  fx = {camera_matrix[0,0]:.2f}")
        print(f"  fy = {camera_matrix[1,1]:.2f}")
        print(f"  cx = {camera_matrix[0,2]:.2f}")
        print(f"  cy = {camera_matrix[1,2]:.2f}")
        print(f"Distortion Coefficients:")
        print(f"  k1 = {dist_coeffs[0,0]:.6f}")
        print(f"  k2 = {dist_coeffs[0,1]:.6f}")
        print(f"  p1 = {dist_coeffs[0,2]:.6f}")
        print(f"  p2 = {dist_coeffs[0,3]:.6f}")
        print(f"  k3 = {dist_coeffs[0,4]:.6f}")
        
        if self.calibration_error < 1.0:
            print("✓ EXCELLENT calibration quality (error < 1.0 pixel)")
        elif self.calibration_error < 2.0:
            print("✓ GOOD calibration quality (error < 2.0 pixels)")
        else:
            print("⚠ Calibration quality could be better (error >= 2.0 pixels)")
            print("  Consider taking more images with better distribution")
        
        return True
    
    def save_calibration(self, filename="camera_calibration.json"):
        """
        Simpan hasil kalibrasi ke file
        """
        if self.camera_matrix is None:
            print("No calibration data to save. Perform calibration first.")
            return False
        
        calibration_data = {
            "timestamp": datetime.now().isoformat(),
            "image_size": self.image_size,
            "checkerboard_size": self.checkerboard_size,
            "square_size_mm": self.square_size_mm,
            "num_images": len(self.objpoints),
            "reprojection_error": float(self.calibration_error),
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.tolist()
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"\n✓ Calibration saved to: {filename}")
        return True
    
    def load_calibration(self, filename="camera_calibration.json"):
        """
        Load kalibrasi dari file
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.camera_matrix = np.array(data["camera_matrix"])
            self.dist_coeffs = np.array(data["distortion_coefficients"])
            self.image_size = tuple(data["image_size"])
            self.calibration_error = data["reprojection_error"]
            
            print(f"✓ Calibration loaded from: {filename}")
            print(f"  Error: {self.calibration_error:.3f} pixels")
            print(f"  Images used: {data['num_images']}")
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def update_aruco_config(self, config_file="aruco_config.py"):
        """
        Update aruco_config.py dengan hasil kalibrasi
        """
        if self.camera_matrix is None:
            print("No calibration data available.")
            return False
        
        # Read current config
        try:
            with open(config_file, 'r') as f:
                config_content = f.read()
        except FileNotFoundError:
            print(f"Config file {config_file} not found!")
            return False
        
        # Generate new camera parameters function
        new_camera_params = f'''
def get_camera_params():
    """
    Parameter kamera - HASIL KALIBRASI OTOMATIS
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Reprojection error: {self.calibration_error:.3f} pixels
    """
    # Image size: {self.image_size[0]}x{self.image_size[1]}
    
    # Calibrated focal lengths and principal point
    fx = {self.camera_matrix[0,0]:.2f}
    fy = {self.camera_matrix[1,1]:.2f}
    cx = {self.camera_matrix[0,2]:.2f}
    cy = {self.camera_matrix[1,2]:.2f}
    
    camera_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], dtype=np.float32)
    
    # Calibrated distortion coefficients (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([{self.dist_coeffs[0,0]:.6f}, {self.dist_coeffs[0,1]:.6f}, {self.dist_coeffs[0,2]:.6f}, {self.dist_coeffs[0,3]:.6f}, {self.dist_coeffs[0,4]:.6f}], dtype=np.float32)
    
    return camera_matrix, dist_coeffs
'''
        
        # Replace the function in config
        import re
        pattern = r'def get_camera_params\(\):.*?return camera_matrix, dist_coeffs'
        
        if re.search(pattern, config_content, re.DOTALL):
            new_content = re.sub(pattern, new_camera_params.strip(), config_content, flags=re.DOTALL)
        else:
            print("Could not find get_camera_params function to replace!")
            return False
        
        # Backup original
        backup_file = f"{config_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_file, 'w') as f:
            f.write(config_content)
        
        # Write new config
        with open(config_file, 'w') as f:
            f.write(new_content)
        
        print(f"✓ Updated {config_file}")
        print(f"✓ Backup saved as {backup_file}")
        return True
    
    def test_calibration(self, camera_index=0):
        """
        Test kalibrasi dengan menampilkan undistorted image
        """
        if self.camera_matrix is None:
            print("No calibration data available.")
            return
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Cannot open camera {camera_index}")
            return
        
        print("Testing calibration... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Undistort image
            undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # Show side by side
            combined = np.hstack([frame, undistorted])
            combined = cv2.resize(combined, (1200, 400))
            
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combined, "Undistorted", (610, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Calibration Test: Original vs Undistorted', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("=== CAMERA CALIBRATION FOR ARUCO BOTTLE DETECTOR ===")
    print("This script will help you calibrate your camera for better accuracy")
    print()
    
    # Initialize calibrator
    calibrator = CameraCalibrator(
        checkerboard_size=(9, 6),  # 9x6 internal corners (10x7 squares)
        square_size_mm=25.0        # 2.5cm squares
    )
    
    while True:
        print("\n=== MENU ===")
        print("1. Capture new calibration images")
        print("2. Use existing images from 'calibration_images/' folder")
        print("3. Load existing calibration file")
        print("4. Test current calibration")
        print("5. Exit")
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == '1':
            print("\n=== CAPTURING CALIBRATION IMAGES ===")
            num_images = input("Number of images to capture (default 20): ").strip()
            num_images = int(num_images) if num_images.isdigit() else 20
            
            if calibrator.capture_calibration_images(num_images=num_images):
                if calibrator.calibrate_camera():
                    calibrator.save_calibration()
                    if input("Update aruco_config.py? (y/n): ").lower() == 'y':
                        calibrator.update_aruco_config()
        
        elif choice == '2':
            print("\n=== USING EXISTING IMAGES ===")
            if calibrator.load_calibration_images():
                if calibrator.calibrate_camera():
                    calibrator.save_calibration()
                    if input("Update aruco_config.py? (y/n): ").lower() == 'y':
                        calibrator.update_aruco_config()
            else:
                print("No valid calibration images found in 'calibration_images/' folder")
        
        elif choice == '3':
            print("\n=== LOADING EXISTING CALIBRATION ===")
            filename = input("Calibration file (default: camera_calibration.json): ").strip()
            filename = filename if filename else "camera_calibration.json"
            
            if calibrator.load_calibration(filename):
                if input("Update aruco_config.py? (y/n): ").lower() == 'y':
                    calibrator.update_aruco_config()
        
        elif choice == '4':
            print("\n=== TESTING CALIBRATION ===")
            calibrator.test_calibration()
        
        elif choice == '5':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option!")

if __name__ == "__main__":
    main() 