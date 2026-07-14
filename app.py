"""
CHILI STRESS DETECTION BACKEND - PYTHON FLASK
UiTM ITT569 IoT Final Project - CDCS259

*Version: Fixed fertilizer logic - single source of truth (ESP32-CAM /detect only)*
*/soil endpoint no longer overwrites action_fertilize -> prevents the two ESP32
nodes from fighting over the same shared status dict*
*Version: Multi-leaf detection - detects EACH leaf as its own bounding box
and reports a per-leaf + overall stressed-percentage instead of one box
for the whole plant*
*Version: Frame throttling - image preview updates on EVERY frame the
ESP32-CAM sends (fast, feels like live video), but the heavy YOLO analysis
only runs every ANALYZE_EVERY_N_FRAMES frames*
*Version: Switched to DIRECT YOLO OBJECT DETECTION on the full frame -
the model itself finds AND localizes each individual leaf (its own
bounding box) and classifies it in one pass. This REPLACES the old
OpenCV-green-contour-crop-then-classify approach, which caused:
  1. Non-leaf areas (walls, monitor, floor) getting boxed - the HSV
     green-mask was loose enough to treat gray/dark surfaces as "leaf".
  2. One giant box covering the whole plant - MORPH_CLOSE fused touching
     leaves into a single contour, and a bounding rect around an
     irregular blob always swallows the non-leaf gaps inside it.
Since the Roboflow model was trained on labelled bounding boxes (an
object-detection model, not a classifier), letting YOLO detect leaves
itself avoids both problems entirely.*
*Version: Class mapping LOCKED - class index 0 = Healthy, 1 = Stressed.
Matched by INDEX (box.cls), never by string name, since Roboflow's exact
class-name string/casing is not guaranteed. Double-checked at startup by
printing yolo_model.names - if this ever prints something other than
{0: 'Healthy', 1: 'Stressed'} (order/spelling may vary), fix CLASS_STRESSED
below to point at whichever index Roboflow assigned to the stressed class.*
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
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

MY_TIMEZONE = pytz.timezone('Asia/Kuala_Lumpur')

def get_malaysia_time():
    return datetime.now(MY_TIMEZONE)

# ========== LOAD YOLO DETECTION MODEL (.pt via ultralytics) ==========
# Running on Cloud Run with an NVIDIA L4 GPU + 16GB RAM, so we use the full
# ultralytics + torch stack with GPU acceleration for inference.
MODEL_PATH = "best.pt"  # <-- your trained weight, e.g. runs/detect/train/weights/best.pt
                        #     (NOT yolo26n.pt - that's the blank pretrained COCO weight)

# ---- CLASS MAPPING (LOCKED) ----
# 0 = Healthy, 1 = Stressed. This must match the order in your data.yaml
# "names:" list. Roboflow exports classes alphabetically by default, and
# "Healthy" < "Stressed" alphabetically, so 0/1 should already line up -
# but ALWAYS verify against the printed yolo_model.names log line below
# the first time you deploy a freshly retrained model.
CLASS_HEALTHY = 0
CLASS_STRESSED = 1

try:
    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
        import torch
        device = 0 if torch.cuda.is_available() else 'cpu'
        logger.info(f"YOLO detection model loaded successfully! Using device: {device}")
        logger.info(f"Model class names/mapping: {yolo_model.names}")
        # Sanity check: warn (don't crash) if the mapping doesn't look like we expect
        names_lower = {k: str(v).lower() for k, v in yolo_model.names.items()}
        if names_lower.get(CLASS_HEALTHY) != "healthy" or names_lower.get(CLASS_STRESSED) != "stressed":
            logger.warning(
                f"Class mapping mismatch! Expected {{0: 'healthy', 1: 'stressed'}}, "
                f"got {yolo_model.names}. Update CLASS_HEALTHY / CLASS_STRESSED above."
            )
        model_available = True
    else:
        logger.warning(f"{MODEL_PATH} not found. Detection will be disabled.")
        model_available = False
        device = 'cpu'
except Exception as e:
    logger.warning(f"Failed to load YOLO model: {str(e)}. Detection will be disabled.")
    model_available = False
    device = 'cpu'

# ========== DETECTION TUNING ==========
CONF_THRESHOLD = 0.5   # ignore detections the model is less than 50% sure about
IOU_THRESHOLD = 0.5    # merge/suppress heavily-overlapping duplicate boxes

# Whole-plant decision: fertilize if this % (or more) of DETECTED leaves are
# individually stressed.
STRESSED_LEAF_PERCENT_THRESHOLD = 40.0

# ========== SENSOR THRESHOLDS ==========
# NOTE: Following the ESP32 mapping: a LOW percentage (< 30%) means dry soil
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
FERTILIZE_PUMP_DURATION_SECONDS = 10

fertilize_tracker = {
    "last_given": None
}

pending_fertilize = False
fertilize_on_until = None

def trigger_fertilize_pump():
    """Start (or restart) the 10s ON window for the fertilizer relay."""
    global fertilize_on_until
    fertilize_on_until = get_malaysia_time() + timedelta(seconds=FERTILIZE_PUMP_DURATION_SECONDS)

def fertilize_pump_is_on():
    """The ACTUAL relay state the ESP32 actuator should follow - timer based."""
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

# ========== FRAME THROTTLING ==========
ANALYZE_EVERY_N_FRAMES = 3  # e.g. ESP32 sends every 1s -> full analysis runs every ~3s

frame_counter = 0
last_leaf_analysis = None
last_diagnosis = "Waiting for first analysis..."
last_disease_confidence = 0.0
last_has_disease = False

# Global state shared with the ESP32 Node (Pump actuator)
# NOTE: action_fertilize is ONLY ever written by /detect (ESP32-CAM).
# /soil (plain ESP32) is only allowed to touch action_water + soil fields.
current_pump_status = {
    "action_water": False,
    "action_fertilize": False,
    "alert_disease": False,
    "decision_reason": "Waiting for initial data..."
}

# ========== PROCESSING FUNCTIONS ==========

def analyze_leaf_health(img):
    """
    Runs the trained YOLO object-DETECTION model DIRECTLY on the full frame.
    The model finds AND localizes each individual leaf (its own bounding
    box) and classifies it (0 = Healthy, 1 = Stressed) in one pass.
    """
    leaves = []

    if model_available:
        try:
            result = yolo_model(
                img, verbose=False, device=device,
                conf=CONF_THRESHOLD, iou=IOU_THRESHOLD
            )[0]
            for box in result.boxes:
                pred_class_idx = int(box.cls[0])
                pred_conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                bw, bh = max(1, x2 - x1), max(1, y2 - y1)

                # Match by class INDEX (locked mapping: 0 = Healthy, 1 = Stressed)
                is_stressed = (pred_class_idx == CLASS_STRESSED)
                leaf_stress_level = pred_conf if is_stressed else (1.0 - pred_conf)

                leaves.append({
                    "bbox": (x1, y1, bw, bh),
                    "area": float(bw * bh),
                    "stress_level": float(leaf_stress_level),
                    "is_stressed": is_stressed
                })
        except Exception as e:
            logger.warning(f"YOLO detect failed on frame: {str(e)}")

    # Largest leaves first (usually the most reliable/least noisy)
    leaves.sort(key=lambda l: -l["area"])

    num_leaves = len(leaves)
    num_stressed = sum(1 for l in leaves if l["is_stressed"])
    stressed_percent = (num_stressed / num_leaves * 100.0) if num_leaves > 0 else 0.0
    avg_stress_level = (sum(l["stress_level"] for l in leaves) / num_leaves) if num_leaves > 0 else 0.0

    is_plant_detected = num_leaves > 0

    return {
        "leaves": leaves,
        "num_leaves": num_leaves,
        "num_stressed": num_stressed,
        "stressed_percent": float(stressed_percent),
        "stress_level": float(avg_stress_level),  # kept for backward-compat (used by /stats)
        "color_abnormal": stressed_percent > STRESSED_LEAF_PERCENT_THRESHOLD,
        "is_plant_detected": bool(is_plant_detected)
    }

def draw_detection_box(img, leaf_analysis, diagnosis):
    """Draws ONE box PER detected leaf (red = stressed, green = healthy),
    each labelled with that leaf's own stress %, plus a summary bar at the
    top showing total leaves and overall stressed percentage."""
    output = img.copy()
    h, w, _ = output.shape

    if leaf_analysis['is_plant_detected'] and leaf_analysis['num_leaves'] > 0:
        for leaf in leaf_analysis['leaves']:
            x, y, bw, bh = leaf['bbox']
            is_stressed = leaf['is_stressed']
            color = (0, 0, 255) if is_stressed else (0, 200, 0)  # BGR: red / green
            cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 2)

            # Only show a percentage when the leaf is flagged stressed -
            # "OK 0%" reads like a measurement, so just show "OK" for healthy.
            label = f"STRESS {leaf['stress_level'] * 100:.0f}%" if is_stressed else "OK"
            label_y = y - 8 if y - 8 > 12 else y + bh + 16
            cv2.putText(output, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        summary = (f"Leaves: {leaf_analysis['num_leaves']}  |  "
                   f"Stressed: {leaf_analysis['num_stressed']} "
                   f"({leaf_analysis['stressed_percent']:.0f}%)")
        cv2.rectangle(output, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(output, summary, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(output, diagnosis, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    else:
        cv2.rectangle(output, (20, 20), (w - 20, h - 20), (0, 165, 255), 2)
        cv2.putText(output, "SCANNING: No Chili Leaves Found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    return output

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
        global frame_counter, last_leaf_analysis, last_diagnosis, last_disease_confidence, last_has_disease

        soil_raw = request.headers.get('X-Soil-Raw', type=int, default=2500)
        soil_percent = request.headers.get('X-Soil-Percent', type=float, default=50.0)

        image_data = request.data
        if not image_data:
            return jsonify({"error": "No image received"}), 400

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # ========== FRAME THROTTLE DECISION ==========
        frame_counter += 1
        run_full_analysis = (last_leaf_analysis is None) or (frame_counter % ANALYZE_EVERY_N_FRAMES == 0)

        if run_full_analysis:
            leaf_analysis = analyze_leaf_health(img)

            if not leaf_analysis['is_plant_detected']:
                diagnosis = "No Chili Plant Detected"
                disease_confidence = 1.0
                has_disease = False
            elif leaf_analysis['stressed_percent'] > STRESSED_LEAF_PERCENT_THRESHOLD:
                diagnosis = "Chili Plant Stressed"
                disease_confidence = leaf_analysis['stress_level']
                has_disease = True
            else:
                diagnosis = "Chili Plant Healthy"
                disease_confidence = 1.0 - leaf_analysis['stress_level']
                has_disease = False

            last_leaf_analysis = leaf_analysis
            last_diagnosis = diagnosis
            last_disease_confidence = disease_confidence
            last_has_disease = has_disease
        else:
            leaf_analysis = last_leaf_analysis
            diagnosis = last_diagnosis
            disease_confidence = last_disease_confidence
            has_disease = last_has_disease

        # ========== SMART DECISION LOGIC ==========
        action_fertilize = False
        alert_disease = False

        soil_is_dry = soil_percent < MOISTURE_THRESHOLD_DRY
        leaf_is_stressed = leaf_analysis['stressed_percent'] > STRESSED_LEAF_PERCENT_THRESHOLD
        needs_fertilizer = leaf_is_stressed or has_disease

        action_water, decision_reason = handle_water_logic(soil_percent)

        if leaf_analysis['is_plant_detected']:
            if needs_fertilizer:
                if soil_is_dry:
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
                if can_fertilize_now():
                    action_fertilize = True
                    pending_fertilize = False
                    record_fertilize_given()
                    trigger_fertilize_pump()
                    decision_reason += f" | Soil now wet enough -> firing previously queued fertilizer dose for {FERTILIZE_PUMP_DURATION_SECONDS}s (7-day cooldown starts now)"
                else:
                    f_info = get_fertilize_tracker_info()
                    decision_reason += f" | Queued fertilizer dose still on cooldown, {f_info['days_remaining']} day(s) left"

            decision_reason += f" | Leaves detected: {leaf_analysis['num_leaves']}, stressed: {leaf_analysis['num_stressed']} ({leaf_analysis['stressed_percent']:.0f}%)"

            if has_disease:
                alert_disease = True
                decision_reason += f" | STRESS ALERT: {diagnosis}"

            if not action_water and not action_fertilize and not alert_disease:
                decision_reason = "All parameters normal - Chili plant is healthy!"
        else:
            diagnosis = "No Chili Plant Detected"
            disease_confidence = 1.0
            has_disease = False
            if not action_water:
                decision_reason = "No chili plant detected in the video frame."
            else:
                decision_reason += " | (No plant detected in frame, but water was still given because the soil is dry)"

        processed_img = draw_detection_box(img, leaf_analysis, diagnosis)

        _, buffer = cv2.imencode('.jpg', processed_img)
        latest_image_base64 = base64.b64encode(buffer).decode('utf-8')
        latest_image_timestamp = get_malaysia_time().isoformat()

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

        record = {
            "timestamp": latest_image_timestamp,
            "soil_percent": soil_percent,
            "soil_raw": soil_raw,
            "diagnosis": diagnosis,
            "plant_needs": plant_needs,
            "disease_confidence": disease_confidence,
            "leaf_stress": leaf_analysis['stress_level'],
            "leaf_color_normal": not leaf_analysis['color_abnormal'],
            "num_leaves": leaf_analysis['num_leaves'],
            "num_stressed_leaves": leaf_analysis['num_stressed'],
            "stressed_leaf_percent": leaf_analysis['stressed_percent'],
            "analyzed_this_frame": run_full_analysis,
            "action_water": action_water,
            "action_fertilize": action_fertilize,
            "alert_disease": alert_disease,
            "decision_reason": decision_reason
        }
        sensor_history.append(record)
        if len(sensor_history) > max_history:
            sensor_history.pop(0)

        return jsonify({
            "success": True,
            "message": "Analysis successful" if run_full_analysis else "Image updated (analysis reused from last cycle)",
            "analyzed_this_frame": run_full_analysis
        }), 200

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/soil', methods=['POST'])
def receive_soil_data():
    """
    Receives soil moisture data directly from the plain ESP32 sensor node.
    IMPORTANT: this endpoint only controls action_water. It must NEVER set
    action_fertilize - that decision belongs solely to /detect (ESP32-CAM).
    """
    try:
        data = request.json
        soil_raw = data.get("soil_raw", 2500)
        soil_percent = data.get("soil_percent", 50.0)

        last_diagnosis = sensor_history[-1]['diagnosis'] if sensor_history else "Waiting for camera"
        last_confidence = sensor_history[-1]['disease_confidence'] if sensor_history else 0.0
        last_stress = sensor_history[-1]['leaf_stress'] if sensor_history else 0.0

        action_water, water_reason = handle_water_logic(soil_percent)
        current_pump_status["action_water"] = action_water
        current_pump_status["decision_reason"] = f"Soil data: {water_reason}"

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
    """Endpoint read by the ESP32 Actuator Node for relay control + the Dashboard."""
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)