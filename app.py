"""
CHILI STRESS DETECTION BACKEND - PYTHON FLASK
UiTM ITT569 IoT Final Project - CDCS259
*Version: Fixed fertilizer logic - single source of truth (ESP32-CAM /detect only)*
*/soil endpoint no longer overwrites action_fertilize -> prevents the two ESP32
nodes from fighting over the same shared status dict*
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
WATER_DAILY_LIMIT = 2

water_tracker = {
    "date": None,
    "count": 0,
    "history_times": []
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
    global watering_session_active
    water_tracker["date"] = None
    water_tracker["count"] = 0
    water_tracker["history_times"] = []
    watering_session_active = False

# True while a watering dose is actively "in progress" (soil still dry since
# it was triggered). Prevents the daily count from incrementing again on the
# very next poll (every 3s) before the soil has had time to actually absorb
# water - which was cutting doses off almost instantly.
watering_session_active = False

def handle_water_logic(soil_percent):
    """
    SINGLE SOURCE OF TRUTH for water pump decisions - called by BOTH /detect
    and /soil so the two ESP32 nodes never diverge or double-count doses.

    A daily "dose" is counted ONCE, when watering starts. The pump then stays
    ON continuously (without re-counting or re-blocking) for as long as the
    soil reports dry, and only turns OFF once the soil actually becomes
    moist - instead of counting a fresh dose on every 3-second poll before
    the water has had a chance to soak in.
    """
    global watering_session_active
    soil_is_dry = soil_percent < MOISTURE_THRESHOLD_DRY

    if soil_is_dry:
        if watering_session_active:
            return True, "Still dry - continuing current watering session (pump stays ON)"
        if can_water_now():
            watering_session_active = True
            record_water_given()
            w_info = get_water_tracker_info()
            return True, f"Dry soil -> Water pump ON (dose {w_info['count_today']}/{WATER_DAILY_LIMIT} today)"
        else:
            return False, f"Soil is dry BUT the {WATER_DAILY_LIMIT}x/day water limit has been reached. Wait until tomorrow."
    else:
        if watering_session_active:
            watering_session_active = False
            return False, "Soil now moist - watering session complete, pump OFF"
        return False, "Soil is moist - watering not needed."

# ========== FERTILIZER PUMP: 7-DAY COOLDOWN ==========
FERTILIZE_COOLDOWN_DAYS = 7

# How long the fertilizer relay should physically stay ON for one dose.
# This is a TIMER, separate from the decision logic below, so the pump
# doesn't get cut off almost instantly by the next /detect call (which can
# arrive every ~3-5s from the ESP32-CAM) before it had time to actually
# dispense anything.
FERTILIZE_PUMP_DURATION_SECONDS = 10

fertilize_tracker = {
    "last_given": None
}

# Set to True by /detect when stress/disease is seen but soil was too dry to
# fertilize at that moment. Cleared once fertilizer actually fires, so the
# next scan (once soil is wet enough) can still act on it instead of losing
# the window.
pending_fertilize = False

# Timestamp (Malaysia time) until which the fertilizer relay should stay ON.
# None / in the past = pump should be OFF.
fertilize_on_until = None

def trigger_fertilize_pump():
    """Start (or restart) the 10s ON window for the fertilizer relay."""
    global fertilize_on_until
    fertilize_on_until = get_malaysia_time() + timedelta(seconds=FERTILIZE_PUMP_DURATION_SECONDS)

def fertilize_pump_is_on():
    """The ACTUAL relay state the ESP32 actuator should follow - timer based,
    not tied to whatever the latest /detect decision cycle computed."""
    return fertilize_on_until is not None and get_malaysia_time() < fertilize_on_until

def fertilize_pump_seconds_remaining():
    if fertilize_on_until is None:
        return 0.0
    remaining = (fertilize_on_until - get_malaysia_time()).total_seconds()
    return max(0.0, round(remaining, 1))

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
            "days_remaining": 0.0,
            "pending": pending_fertilize
        }
    elapsed = get_malaysia_time() - fertilize_tracker["last_given"]
    days_since = elapsed.total_seconds() / 86400.0
    days_remaining = max(0.0, FERTILIZE_COOLDOWN_DAYS - days_since)
    return {
        "last_given": fertilize_tracker["last_given"].isoformat(),
        "days_since": round(days_since, 2),
        "cooldown_days": FERTILIZE_COOLDOWN_DAYS,
        "can_fertilize": days_since >= FERTILIZE_COOLDOWN_DAYS,
        "days_remaining": round(days_remaining, 2),
        "pending": pending_fertilize
    }

def reset_fertilize_tracker():
    global pending_fertilize, fertilize_on_until
    fertilize_tracker["last_given"] = None
    pending_fertilize = False
    fertilize_on_until = None

# ========== DATA STORAGE ==========
latest_image_base64 = None
latest_image_timestamp = None
sensor_history = []
max_history = 100

# Global state shared with the ESP32 Node (Pump actuator)
# NOTE: action_fertilize is ONLY ever written by /detect (ESP32-CAM).
# /soil (plain ESP32) is only allowed to touch action_water + soil fields.
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
        global latest_image_base64, latest_image_timestamp, current_pump_status, pending_fertilize

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

        soil_is_dry = soil_percent < MOISTURE_THRESHOLD_DRY
        leaf_is_stressed = leaf_analysis['stress_level'] > 0.5
        needs_fertilizer = leaf_is_stressed or has_disease

        # -------- WATER PUMP LOGIC (ignores detection, capped at 2x/day) --------
        # Uses the SAME shared function as /soil, so a dose triggered by one
        # node is recognised by the other and never double-counted.
        action_water, decision_reason = handle_water_logic(soil_percent)

        # -------- FERTILIZER PUMP LOGIC (requires plant detection + 7-day cooldown) --------
        # This block is the ONLY place in the whole app that is allowed to set
        # action_fertilize. /soil must never touch it.
        if leaf_analysis['is_plant_detected']:
            if needs_fertilizer:
                if soil_is_dry:
                    # Can't fertilize dry soil right now, but don't lose the
                    # signal - remember it so next wet-soil scan can still act.
                    pending_fertilize = True
                    decision_reason += " | Plant stressed/diseased but soil is dry -> fertilizer queued until soil is watered"
                else:
                    if can_fertilize_now():
                        action_fertilize = True
                        pending_fertilize = False
                        record_fertilize_given()
                        trigger_fertilize_pump()
                        decision_reason += f" | Plant stressed/diseased -> Fertilizer pump ON for {FERTILIZE_PUMP_DURATION_SECONDS}s (7-day cooldown starts now)"
                    else:
                        f_info = get_fertilize_tracker_info()
                        decision_reason += f" | Plant stress/disease detected BUT fertilizer is still on cooldown, {f_info['days_remaining']} day(s) left before it can be given again"
            elif pending_fertilize and not soil_is_dry:
                # Stress/disease from a PREVIOUS scan is still pending and soil
                # has since become wet enough -> act on it now.
                if can_fertilize_now():
                    action_fertilize = True
                    pending_fertilize = False
                    record_fertilize_given()
                    trigger_fertilize_pump()
                    decision_reason += f" | Soil now wet enough -> firing previously queued fertilizer dose for {FERTILIZE_PUMP_DURATION_SECONDS}s (7-day cooldown starts now)"
                else:
                    f_info = get_fertilize_tracker_info()
                    decision_reason += f" | Queued fertilizer dose still on cooldown, {f_info['days_remaining']} day(s) left"

            if has_disease:
                alert_disease = True
                decision_reason += f" | DISEASE ALERT: {diagnosis}"

            if not action_water and not action_fertilize and not alert_disease:
                decision_reason = "All parameters normal - Chili plant is healthy!"
        else:
            # No plant detected (e.g. only the floor/wall is visible)
            diagnosis = "No Chili Plant Detected"
            disease_confidence = 1.0
            has_disease = False
            if not action_water:
                decision_reason = "No chili plant detected in the video frame."
            else:
                decision_reason += " | (No plant detected in frame, but water was still given because the soil is dry)"

        # ========== STEP 4: DRAW BOUNDING BOX ==========
        processed_img = draw_detection_box(img, leaf_analysis, diagnosis)

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

        if action_water and action_fertilize:
            plant_needs = "Need Water & Fertilizer"
        elif action_water:
            plant_needs = "Need Water"
        elif action_fertilize:
            plant_needs = "Need Fertilizer"
        else:
            plant_needs = "Optimal (No Action)"

        # Save record to in-memory history.
        # NOTE: "diagnosis" keeps the RAW model/heuristic diagnosis (e.g.
        # "Chili Bell Bacterial Spot") so any future text-matching stays
        # valid. The human-friendly summary is stored separately as
        # "plant_needs" instead of overwriting diagnosis like before.
        record = {
            "timestamp": latest_image_timestamp,
            "soil_percent": soil_percent,
            "soil_raw": soil_raw,
            "diagnosis": diagnosis,          # raw diagnosis, e.g. "Chili Bell Bacterial Spot"
            "plant_needs": plant_needs,       # human-friendly summary for the dashboard
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
    """
    Receives soil moisture data directly from the plain ESP32 sensor node.
    IMPORTANT: this endpoint only controls action_water. It must NEVER set
    action_fertilize - that decision belongs solely to /detect (ESP32-CAM),
    otherwise the two nodes end up racing to overwrite the same shared
    current_pump_status dict.
    """
    try:
        data = request.json
        soil_raw = data.get("soil_raw", 2500)
        soil_percent = data.get("soil_percent", 50.0)

        last_diagnosis = sensor_history[-1]['diagnosis'] if sensor_history else "Waiting for camera"
        last_confidence = sensor_history[-1]['disease_confidence'] if sensor_history else 0.0
        last_stress = sensor_history[-1]['leaf_stress'] if sensor_history else 0.0

        # -------- WATER PUMP LOGIC (ignores detection, capped at 2x/day) --------
        # Uses the SAME shared function as /detect, so a dose triggered by
        # one node is recognised by the other and never double-counted.
        action_water, water_reason = handle_water_logic(soil_percent)
        current_pump_status["action_water"] = action_water
        current_pump_status["decision_reason"] = f"Soil data: {water_reason}"

        # action_fertilize and alert_disease are intentionally left untouched
        # here - they keep whatever /detect last decided.

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
    7-day cooldown (+ pending_fertilize flag) so you can re-test the logic
    instantly without waiting for the next day / next 7 days.

    Usage:
      - GET  /debug/reset-trackers                 -> resets BOTH trackers
      - GET  /debug/reset-trackers?target=water     -> resets water pump only
      - GET  /debug/reset-trackers?target=fertilize -> resets fertilizer pump only
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
    """Endpoint read by the ESP32 Actuator Node for relay control + the Dashboard.
    IMPORTANT: action_fertilize here is the TIMER-based live state (stays True
    for FERTILIZE_PUMP_DURATION_SECONDS after triggering), NOT just whatever
    the most recent /detect decision cycle happened to compute. This is what
    stops the relay from flicking OFF again before it had time to dispense."""
    status = dict(current_pump_status)
    status["action_fertilize"] = fertilize_pump_is_on()
    status["water_tracker"] = get_water_tracker_info()
    status["fertilize_tracker"] = get_fertilize_tracker_info()
    status["fertilize_tracker"]["seconds_remaining"] = fertilize_pump_seconds_remaining()
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

    lower_green = np.array([35, 35, 35])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask_green)
    total_pixels = img.shape[0] * img.shape[1]
    green_ratio = green_pixels / total_pixels

    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(mask_yellow)

    stress_level = (yellow_pixels / total_pixels) * 3
    stress_level = min(stress_level, 1.0)

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
        contours, _ = cv2.findContours(leaf_analysis['mask_green'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, box_w, box_h = cv2.boundingRect(c)

            cv2.rectangle(output, (x, y), (x + box_w, y + box_h), (0, 0, 255), 3)

            label = f"{diagnosis}"
            cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
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
