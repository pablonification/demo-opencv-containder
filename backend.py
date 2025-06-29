import base64
import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" artinya semua frontend boleh akses
    allow_credentials=True,
    allow_methods=["*"],  # Boleh GET, POST, OPTIONS, dst
    allow_headers=["*"],  # Boleh pake header apa aja
)

class Image(BaseModel):
    image: str

# --- Configuration ---
KNOWN_REF_OBJECT_HEIGHT_CM = 16
BLUE_LOWER = np.array([0, 0, 0])
BLUE_UPPER = np.array([180, 255, 50])

KNOWN_BOTTLE_SPECS = {
    "200mL": {"volume_ml": 200, "avg_diameter_cm": 5.0, "avg_height_cm": 13.0},
    "500mL": {"volume_ml": 500, "avg_diameter_cm": 6.5, "avg_height_cm": 20.0},
    "1000mL": {"volume_ml": 1000, "avg_diameter_cm": 8.0, "avg_height_cm": 25.0},
}
CLASSIFICATION_TOLERANCE_PERCENT = 30

def detect_blue_rectangle(image, lower_blue, upper_blue):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if 4 <= len(approx) <= 6:
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            if area > max_area:
                max_area = area
                ref_height = max(w, h)
                ref_width = min(w, h)
                best_candidate = {
                    "box_points": cv2.boxPoints(rect).astype(int),
                    "pixel_height": ref_height,
                    "pixel_width": ref_width,
                }
    return best_candidate

def preprocess_roi_for_bottle(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return closed

def find_bottle_in_roi(processed_roi, min_area=1000, min_aspect=1.3, max_tilt=15):
    contours, _ = cv2.findContours(processed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    max_area_found = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        h_vis, w_vis = max(w, h), min(w, h)
        aspect = h_vis / w_vis if w_vis > 0 else 0
        if aspect < min_aspect:
            continue
        upright = abs(angle) < max_tilt if h >= w else abs(90 - abs(angle)) < max_tilt
        if not upright:
            continue
        if area > max_area_found:
            max_area_found = area
            best = {
                "pixel_height": h_vis,
                "pixel_width": w_vis
            }
    return best

def estimate_and_classify(bottle_info, ppm, specs, tolerance):
    if not bottle_info or ppm == 0:
        return {
            "classification": "Calibration Error",
            "estimated_volume_ml": 0,
            "real_height_cm": 0,
            "real_diameter_cm": 0,
            "confidence_percent": 0
        }
    h_cm = bottle_info["pixel_height"] / ppm
    d_cm = bottle_info["pixel_width"] / ppm
    r_cm = d_cm / 2
    vol_ml = np.pi * (r_cm ** 2) * h_cm
    best_label = "Other"
    min_diff = float("inf")
    for label, spec in specs.items():
        diff = abs(vol_ml - spec["volume_ml"]) / spec["volume_ml"] * 100
        if diff < min_diff:
            min_diff = diff
            best_label = label
    if min_diff > tolerance:
        best_label = f"Other ({int(vol_ml)}mL)"
    return {
        "classification": best_label,
        "estimated_volume_ml": int(vol_ml),
        "real_height_cm": round(h_cm, 2),
        "real_diameter_cm": round(d_cm, 2),
        "confidence_percent": max(0, round(100 - min_diff, 2))
    }

@app.post("/")
async def classify_bottle(data: Image):
    try:
        base64_str = data.image.split(",")[-1]
        img_bytes = base64.b64decode(base64_str)
        np_array = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Invalid image data."}

        ref = detect_blue_rectangle(image, BLUE_LOWER, BLUE_UPPER)
        if not ref or ref["pixel_height"] == 0:
            return {"error": "Reference object not found."}

        ppm = ref["pixel_height"] / KNOWN_REF_OBJECT_HEIGHT_CM
        pts = ref["box_points"]
        x1, x2 = np.min(pts[:, 0]) - 50, np.max(pts[:, 0]) + 50
        y2 = np.min(pts[:, 1]) - 5
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        y2 = max(0, y2)
        roi = image[0:int(y2), int(x1):int(x2)]

        if roi.size == 0:
            return {"error": "ROI is empty."}

        processed = preprocess_roi_for_bottle(roi)
        min_area = ppm**2 * 2 if ppm else 1000
        bottle = find_bottle_in_roi(processed, min_area=min_area)

        if not bottle:
            return {"error": "No valid bottle detected."}

        result = estimate_and_classify(bottle, ppm, KNOWN_BOTTLE_SPECS, CLASSIFICATION_TOLERANCE_PERCENT)
        return result

    except Exception as e:
        return {"error": str(e)}
