"""
CHILI STRESS DETECTION BACKEND - PYTHON FLASK
UiTM ITT569 IoT Final Project - CDCS259
*Version: Dry-soil alert logic fix + Water daily limit (2x/day) + Fertilizer 7-day cooldown*
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import tensorflow as tf
import json
from datetime import datetime, timedelta
import os
import logging
import base64
import pytz  # To handle Malaysia timezone (MYT)

# ========== CONFIGURATION ==========
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Malaysia timezone
MY_TIMEZONE = pytz.timezone('Asia/Kuala_Lumpur')

def get_malaysia_time():
    return datetime.now(MY_TIMEZONE)

# ========== LOAD PRE-TRAINED MODEL ==========
MODEL_PATH = "plant_disease_model.h5"
CLASS_NAMES = [
    "Chili Bell Bacterial Spot",  # Index 0 (Diseased)
    "Chili Bell Healthy"          # Index 1 (Healthy)
]

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("TensorFlow model (.h5) loaded successfully!")
        model_available = True
    else:
        logger.warning("plant_disease_model.h5 not found. Using OpenCV fallback.")
        model_available = False
except Exception as e:
    logger.warning(f"Failed to load model: {str(e)}. Using OpenCV fallback.")
    model_available = False

# ========== SENSOR THRESHOLDS ==========
# NOTE: Following the new ESP32 mapping: a LOW percentage (< 30%) means dry soil
MOISTURE_THRESHOLD_DRY = 30.0

# ========== WATER PUMP: DAILY LIMIT (2x/DAY) ==========
# Water will be given AS LONG AS the soil is dry, REGARDLESS of whether the
# chili plant is detected in frame or not. But it's capped at a maximum of
# 2 times within a 1-day window (resets based on the Malaysia date).
WATER_DAILY_LIMIT = 2

water_tracker = {
    "date": None,          # Malaysia date this count applies to
    "count": 0,             # how many times water has been given today
    "history_times": []     # list of ISO timestamps water was given today
}

def _reset_water_tracker_if_new_day():
    today = get_malaysia_time().date()
    if water_tracker["date"] != today:
        water_tracker["date"] = today
        water_tracker["count"] = 0
        water_tracker["history_times"] = []

def can_water_now():
    _reset_water_tracker_if_new_day()
    return water_tracker["count"] < WATER_DAILY_LIMIT

def record_water_given():
    _reset_water_tracker_if_new_day()
    water_tracker["count"] += 1
    water_tracker["history_times"].append(get_malaysia_time().isoformat())

def get_water_tracker_info():
    _reset_water_tracker_if_new_day()
    return {
        "count_today": water_tracker["count"],
        "limit_daily": WATER_DAILY_LIMIT,
        "remaining_today": max(0, WATER_DAILY_LIMIT - water_tracker["count"]),
        "history_times": water_tracker["history_times"]
    }

def reset_water_tracker():
    water_tracker["date"] = None
    water_tracker["count"] = 0
    water_tracker["history_times"] = []

# ========== FERTILIZER PUMP: 7-DAY COOLDOWN ==========
# Once fertilizer is given, the system will "go quiet" (won't fertilize again)
# for 7 days, even if stress/disease is detected again, to avoid over-fertilizing.
FERTILIZE_COOLDOWN_DAYS = 7

fertilize_tracker = {
    "last_given": None  # datetime object (Malaysia time) of the last fertilizer dose
}

def can_fertilize_now():
    if fertilize_tracker["last_given"] is None:
        return True
    elapsed = get_malaysia_time() - fertilize_tracker["last_given"]
    return elapsed >= timedelta(days=FERTILIZE_COOLDOWN_DAYS)

def record_fertilize_given():
    fertilize_tracker["last_given"] = get_malaysia_time()

def get_fertilize_tracker_info():
    if fertilize_tracker["last_given"] is None:
        return {
            "last_given": None,
            "days_since": None,
            "cooldown_days": FERTILIZE_COOLDOWN_DAYS,
            "can_fertilize": True,
            "days_remaining": 0.0
        }
    elapsed = get_malaysia_time() - fertilize_tracker["last_given"]
    days_since = elapsed.total_seconds() / 86400.0
    days_remaining = max(0.0, FERTILIZE_COOLDOWN_DAYS - days_since)
    return {
        "last_given": fertilize_tracker["last_given"].isoformat(),
        "days_since": round(days_since, 2),
        "cooldown_days": FERTILIZE_COOLDOWN_DAYS,
        "can_fertilize": days_since >= FERTILIZE_COOLDOWN_DAYS,
        "days_remaining": round(days_remaining, 2)
    }

def reset_fertilize_tracker():
    fertilize_tracker["last_given"] = None

# ========== DATA STORAGE ==========
latest_image_base64 = None
latest_image_timestamp = None
sensor_history = []
max_history = 100

# Global state shared with the ESP32 Node (Pump actuator)
current_pump_status = {
    "action_water": False,
    "action_fertilize": False,
    "alert_disease": False,
    "decision_reason": "Waiting for initial data..."
}

# ========== ROUTES ==========

@app.route('/', methods=['GET'])
def index():
    return render_template('dashboard.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "OK",
        "model_loaded": model_available,
        "timestamp": get_malaysia_time().isoformat()
    }), 200

@app.route('/detect', methods=['POST'])
def detect_plant():
    try:
        global latest_image_base64, latest_image_timestamp, current_pump_status

        # Get sensor data from HTTP Headers (ESP32-CAM)
        soil_raw = request.headers.get('X-Soil-Raw', type=int, default=2500)
        soil_percent = request.headers.get('X-Soil-Percent', type=float, default=50.0)

        image_data = request.data
        if not image_data:
            return jsonify({"error": "No image received"}), 400

        # Decode image using OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # ========== STEP 1: LEAF COLOR ANALYSIS (OPENCV) ==========
        leaf_analysis = analyze_leaf_health(img)

        # ========== STEP 2: DISEASE DETECTION (AI MODEL / FALLBACK) ==========
        if model_available and leaf_analysis['is_plant_detected']:
            disease_result = detect_disease_with_model(img)
        else:
            disease_result = detect_disease_simple(img, leaf_analysis)

        diagnosis = disease_result['diagnosis']
        disease_confidence = disease_result['confidence']
        has_disease = disease_result['has_disease']

        # ========== STEP 3: SMART DECISION LOGIC ==========
        action_water = False
        action_fertilize = False
        alert_disease = False
        decision_reason = ""

        # NOTE: soil_percent below the threshold is considered fully dry
        soil_is_dry = soil_percent < MOISTURE_THRESHOLD_DRY
        leaf_is_stressed = leaf_analysis['stress_level'] > 0.5

        # -------- WATER PUMP LOGIC (ignores detection, capped at 2x/day) --------
        # Water is given AS LONG AS the soil is dry, REGARDLESS of whether the
        # chili plant is detected in the camera frame. Capped at 2 times per day.
        if soil_is_dry:
            if can_water_now():
                action_water = True
                record_water_given()
                w_info = get_water_tracker_info()
                decision_reason = f"Dry soil -> Water pump ON (dose {w_info['count_today']}/{WATER_DAILY_LIMIT} today)"
            else:
                action_water = False
                decision_reason = f"Soil is dry BUT the {WATER_DAILY_LIMIT}x/day water limit has been reached. Wait until tomorrow."
        else:
            decision_reason = "Soil is moist - watering not needed."

        # -------- FERTILIZER PUMP LOGIC (requires plant detection + 7-day cooldown) --------
        if leaf_analysis['is_plant_detected']:
            if not soil_is_dry and (leaf_is_stressed or has_disease):
                if can_fertilize_now():
                    action_fertilize = True
                    record_fertilize_given()
                    decision_reason += " | Plant stressed/diseased -> Fertilizer pump ON (7-day cooldown starts now)"
                else:
                    f_info = get_fertilize_tracker_info()
                    decision_reason += f" | Plant stress/disease detected BUT fertilizer is still on cooldown, {f_info['days_remaining']} day(s) left before it can be given again"

            if has_disease:
                alert_disease = True
                decision_reason += f" | DISEASE ALERT: {diagnosis}"

            if not action_water and not action_fertilize and not alert_disease:
                decision_reason = "All parameters normal - Chili plant is healthy!"
        else:
            # No plant detected (e.g. only the floor/wall is visible)
            # NOTE: the water pump can STILL run above since it ignores detection.
            diagnosis = "No Chili Plant Detected"
            disease_confidence = 1.0
            has_disease = False
            if not action_water:
                decision_reason = "No chili plant detected in the video frame."
            else:
                decision_reason += " | (No plant detected in frame, but water was still given because the soil is dry)"

        # ========== STEP 4: DRAW BOUNDING BOX ==========
        processed_img = draw_detection_box(img, leaf_analysis, diagnosis)

        # Convert processed image to Base64 string
        _, buffer = cv2.imencode('.jpg', processed_img)
        latest_image_base64 = base64.b64encode(buffer).decode('utf-8')
        latest_image_timestamp = get_malaysia_time().isoformat()

        # Save status to share with the ESP32 Node
        current_pump_status = {
            "action_water": action_water,
            "action_fertilize": action_fertilize,
            "alert_disease": alert_disease,
            "decision_reason": decision_reason
        }

        # Determine the plant's needs based on active pump status
        if action_water and action_fertilize:
            plant_needs = "Need Water & Fertilizer"
        elif action_water:
            plant_needs = "Need Water"
        elif action_fertilize:
            plant_needs = "Need Fertilizer"
        else:
            plant_needs = "Optimal (No Action)"

        # Save record to in-memory history
        record = {
            "timestamp": latest_image_timestamp,
            "soil_percent": soil_percent,
            "soil_raw": soil_raw,
            "diagnosis": plant_needs,  # <--- We send the plant's needs status to the frontend
            "disease_confidence": disease_confidence,
            "leaf_stress": leaf_analysis['stress_level'],
            "leaf_color_normal": not leaf_analysis['color_abnormal'],
            "action_water": action_water,
            "action_fertilize": action_fertilize,
            "alert_disease": alert_disease,
            "decision_reason": decision_reason
        }
        sensor_history.append(record)
        if len(sensor_history) > max_history:
            sensor_history.pop(0)

        return jsonify({"success": True, "message": "Analysis successful"}), 200

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/soil', methods=['POST'])
def receive_soil_data():
    """Receives soil moisture data directly from the ESP32 sensor node"""
    try:
        data = request.json
        soil_raw = data.get("soil_raw", 2500)
        soil_percent = data.get("soil_percent", 50.0)

        # Get the last diagnosis, if any
        last_diagnosis = sensor_history[-1]['diagnosis'] if sensor_history else "Waiting for camera"
        last_confidence = sensor_history[-1]['disease_confidence'] if sensor_history else 0.0
        last_stress = sensor_history[-1]['leaf_stress'] if sensor_history else 0.0

        # NOTE: soil_percent below the threshold is considered fully dry
        soil_is_dry = soil_percent < MOISTURE_THRESHOLD_DRY

        # -------- WATER PUMP LOGIC (ignores detection, capped at 2x/day) --------
        if soil_is_dry:
            if can_water_now():
                current_pump_status["action_water"] = True
                record_water_given()
                w_info = get_water_tracker_info()
                current_pump_status["decision_reason"] = f"Soil data: Dry -> Water pump ON (dose {w_info['count_today']}/{WATER_DAILY_LIMIT} today)"
            else:
                current_pump_status["action_water"] = False
                current_pump_status["decision_reason"] = f"Soil data: Dry BUT the {WATER_DAILY_LIMIT}x/day water limit has been reached. Wait until tomorrow."
        else:
            current_pump_status["action_water"] = False
            current_pump_status["decision_reason"] = "Soil data: Moist - watering not needed."

        # -------- FERTILIZER PUMP LOGIC (7-day cooldown) --------
        if "Bacterial Spot" in last_diagnosis or last_stress > 0.5:
            if not soil_is_dry:
                if can_fertilize_now():
                    current_pump_status["action_fertilize"] = True
                    record_fertilize_given()
                    current_pump_status["decision_reason"] += " | Plant stressed -> Fertilizer pump ON (7-day cooldown starts now)"
                else:
                    f_info = get_fertilize_tracker_info()
                    current_pump_status["action_fertilize"] = False
                    current_pump_status["decision_reason"] += f" | Plant stress detected BUT fertilizer is still on cooldown, {f_info['days_remaining']} day(s) left"
            else:
                current_pump_status["action_fertilize"] = False
        else:
            current_pump_status["action_fertilize"] = False

        record = {
            "timestamp": get_malaysia_time().isoformat(),
            "soil_percent": soil_percent,
            "soil_raw": soil_raw,
            "diagnosis": last_diagnosis,
            "disease_confidence": last_confidence,
            "leaf_stress": last_stress,
            "leaf_color_normal": True,
            "action_water": current_pump_status["action_water"],
            "action_fertilize": current_pump_status["action_fertilize"],
            "alert_disease": current_pump_status["alert_disease"],
            "decision_reason": current_pump_status["decision_reason"]
        }
        sensor_history.append(record)
        return jsonify({"status": "Success", "action": current_pump_status}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug/reset-trackers', methods=['GET', 'POST'])
def reset_trackers():
    """
    TESTING-ONLY ENDPOINT.
    Resets the water pump daily limit (2x/day) and/or the fertilizer pump
    7-day cooldown so you can re-test the logic instantly without waiting
    for the next day / next 7 days, and without restarting the whole server
    (history/stats stay intact).

    Usage:
      - GET  /debug/reset-trackers                 -> resets BOTH trackers
      - GET  /debug/reset-trackers?target=water     -> resets water pump only
      - GET  /debug/reset-trackers?target=fertilize -> resets fertilizer pump only

    This is also called by the "Reset Water Pump" / "Reset Fertilizer Pump"
    buttons on the dashboard.
    """
    target = request.args.get('target', 'all')

    if target in ('all', 'water'):
        reset_water_tracker()

    if target in ('all', 'fertilize'):
        reset_fertilize_tracker()

    return jsonify({
        "success": True,
        "message": f"Tracker '{target}' has been reset for testing.",
        "water_tracker": get_water_tracker_info(),
        "fertilize_tracker": get_fertilize_tracker_info()
    }), 200

@app.route('/pump-status', methods=['GET'])
def get_pump_status():
    """Endpoint read by the ESP32 Actuator Node for relay control + the Dashboard"""
    status = dict(current_pump_status)
    status["water_tracker"] = get_water_tracker_info()
    status["fertilize_tracker"] = get_fertilize_tracker_info()
    return jsonify(status), 200

@app.route('/latest-image-data', methods=['GET'])
def get_latest_image_data():
    if latest_image_base64 is None:
        return jsonify({"error": "No image captured yet"}), 404
    return jsonify({
        "image_base64": latest_image_base64,
        "timestamp": latest_image_timestamp,
        "status": "OK"
    }), 200

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({
        "count": len(sensor_history),
        "data": sensor_history[-20:]
    }), 200

@app.route('/stats', methods=['GET'])
def get_stats():
    if not sensor_history:
        return jsonify({"message": "No data yet"}), 200

    avg_soil = np.mean([r['soil_percent'] for r in sensor_history])
    avg_stress = np.mean([r['leaf_stress'] for r in sensor_history])
    disease_count = sum(1 for r in sensor_history if r['alert_disease'])

    return jsonify({
        "total_checks": len(sensor_history),
        "avg_soil_moisture": float(avg_soil),
        "avg_leaf_stress": float(avg_stress),
        "disease_alerts_count": disease_count,
        "last_check": sensor_history[-1]['timestamp']
    }), 200

# ========== PROCESSING FUNCTIONS ==========

def analyze_leaf_health(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green color range for chili leaves
    lower_green = np.array([35, 35, 35])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask_green)
    total_pixels = img.shape[0] * img.shape[1]
    green_ratio = green_pixels / total_pixels

    # Yellow color range (sign of stress)
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(mask_yellow)

    stress_level = (yellow_pixels / total_pixels) * 3
    stress_level = min(stress_level, 1.0)

    # Verify whether the object is actually a chili plant or just empty background
    is_plant_detected = green_ratio > 0.05 or (yellow_pixels / total_pixels) > 0.05
    color_abnormal = green_ratio < 0.20 if is_plant_detected else False

    return {
        "stress_level": float(stress_level),
        "color_abnormal": bool(color_abnormal),
        "green_ratio": float(green_ratio),
        "is_plant_detected": bool(is_plant_detected),
        "mask_green": mask_green
    }

def draw_detection_box(img, leaf_analysis, diagnosis):
    output = img.copy()
    h, w, _ = output.shape

    if leaf_analysis['is_plant_detected']:
        # Find the outer contour of the plant to draw the red detection box
        contours, _ = cv2.findContours(leaf_analysis['mask_green'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Take the largest contour
            c = max(contours, key=cv2.contourArea)
            x, y, box_w, box_h = cv2.boundingRect(c)

            # Draw the main red box around the detected leaf area
            cv2.rectangle(output, (x, y), (x + box_w, y + box_h), (0, 0, 255), 3)

            # Label the tag above the red box
            label = f"{diagnosis}"
            cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        # If no plant is detected, draw a warning border around the whole feed
        cv2.rectangle(output, (20, 20), (w - 20, h - 20), (0, 165, 255), 2)
        cv2.putText(output, "SCANNING: No Chili Plant Found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    return output

def detect_disease_with_model(img):
    try:
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        predictions = model.predict(img_batch, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])

        return {
            "diagnosis": CLASS_NAMES[class_idx],
            "confidence": confidence,
            "has_disease": class_idx == 0
        }
    except:
        return {"diagnosis": "AI Model Error", "confidence": 0.0, "has_disease": False}

def detect_disease_simple(img, leaf_analysis):
    if not leaf_analysis['is_plant_detected']:
        return {"diagnosis": "No Chili Plant Detected", "confidence": 1.0, "has_disease": False}

    if leaf_analysis['stress_level'] > 0.5:
        return {"diagnosis": "Chili Bell Bacterial Spot (Estimated)", "confidence": 0.70, "has_disease": True}
    return {"diagnosis": "Chili Bell Healthy", "confidence": 0.85, "has_disease": False}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)