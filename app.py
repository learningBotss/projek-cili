"""
CHILI STRESS DETECTION BACKEND - PYTHON FLASK
UiTM ITT569 IoT Final Project - CDCS259
*Version: Multi-leaf detection fixes - watershed separation for touching
leaves, shape filtering to reject non-leaf green blobs, hide 0%
percentage labels, AND fixed YOLO class index mismatch (match by class
NAME instead of assuming index 1 = Stressed)*
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLO
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

MY_TIMEZONE = pytz.timezone('Asia/Kuala_Lumpur')

def get_malaysia_time():
    return datetime.now(MY_TIMEZONE)

MODEL_PATH = "best.pt"

# ---- NEW: figure out which class index actually means "stressed" by
# reading the model's own name mapping, instead of hardcoding index==1.
# YOLO classification models assign indices based on the ALPHABETICAL
# ORDER of your training folder names, which may not match what you
# expect (e.g. "Healthy_Leaf"/"Stressed_Leaf" vs "healthy"/"stressed").
STRESSED_CLASS_IDX = None
HEALTHY_CLASS_IDX = None

try:
    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
        import torch
        device = 0 if torch.cuda.is_available() else 'cpu'
        logger.info(f"YOLO (.pt) classification model loaded successfully! Using device: {device}")
        logger.info(f"Model class names/mapping: {yolo_model.names}")

        # Auto-detect which index is "stressed" vs "healthy" by name,
        # instead of assuming 0 = Healthy, 1 = Stressed.
        for idx, name in yolo_model.names.items():
            name_lower = str(name).lower()
            if "stress" in name_lower or "disease" in name_lower or "unhealthy" in name_lower:
                STRESSED_CLASS_IDX = idx
            elif "healthy" in name_lower or "normal" in name_lower:
                HEALTHY_CLASS_IDX = idx

        if STRESSED_CLASS_IDX is None:
            logger.warning(
                "Could not auto-detect 'stressed' class by name from "
                f"{yolo_model.names} - falling back to index 1 as stressed. "
                "Check /debug/model-info and rename your training folders "
                "to include 'healthy'/'stressed' if this looks wrong."
            )
            STRESSED_CLASS_IDX = 1

        logger.info(f"Resolved STRESSED_CLASS_IDX = {STRESSED_CLASS_IDX} "
                    f"({yolo_model.names.get(STRESSED_CLASS_IDX)})")

        model_available = True
    else:
        logger.warning(f"{MODEL_PATH} not found. Using OpenCV fallback (yellow-ratio heuristic).")
        model_available = False
        device = 'cpu'
except Exception as e:
    logger.warning(f"Failed to load YOLO model: {str(e)}. Using OpenCV fallback.")
    model_available = False
    device = 'cpu'

MOISTURE_THRESHOLD_DRY = 30.0
WATER_DAILY_LIMIT = 2

water_tracker = {"date": None, "count": 0, "history_times": []}

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

watering_session_active = False

def handle_water_logic(soil_percent):
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

FERTILIZE_COOLDOWN_DAYS = 7
FERTILIZE_PUMP_DURATION_SECONDS = 10

fertilize_tracker = {"last_given": None}
pending_fertilize = False
fertilize_on_until = None

def trigger_fertilize_pump():
    global fertilize_on_until
    fertilize_on_until = get_malaysia_time() + timedelta(seconds=FERTILIZE_PUMP_DURATION_SECONDS)

def fertilize_pump_is_on():
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
            "last_given": None, "days_since": None,
            "cooldown_days": FERTILIZE_COOLDOWN_DAYS, "can_fertilize": True,
            "days_remaining": 0.0, "pending": pending_fertilize
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

latest_image_base64 = None
latest_image_timestamp = None
sensor_history = []
max_history = 100

ANALYZE_EVERY_N_FRAMES = 3
frame_counter = 0
last_leaf_analysis = None
last_diagnosis = "Waiting for first analysis..."
last_disease_confidence = 0.0
last_has_disease = False

current_pump_status = {
    "action_water": False,
    "action_fertilize": False,
    "alert_disease": False,
    "decision_reason": "Waiting for initial data..."
}

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

# ---- NEW: debug endpoint so you can check the model's real class
# mapping straight from the browser, no need to dig through Render logs.
@app.route('/debug/model-info', methods=['GET'])
def debug_model_info():
    if not model_available:
        return jsonify({"model_loaded": False, "message": "Model not loaded, using OpenCV fallback"}), 200
    return jsonify({
        "model_loaded": True,
        "class_names": yolo_model.names,
        "resolved_stressed_class_idx": STRESSED_CLASS_IDX,
        "resolved_stressed_class_name": yolo_model.names.get(STRESSED_CLASS_IDX),
        "resolved_healthy_class_idx": HEALTHY_CLASS_IDX,
        "resolved_healthy_class_name": yolo_model.names.get(HEALTHY_CLASS_IDX) if HEALTHY_CLASS_IDX is not None else None,
        "device": str(device)
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

        action_water = False
        action_fertilize = False
        alert_disease = False
        decision_reason = ""

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

MIN_LEAF_AREA = 800  # tuned for full VGA (640x480) resolution

# ---- shape filters to reject non-leaf green blobs ----
# Leaves are roughly oval/elongated blobs with a fairly "filled-in" outline.
# Random green background clutter (walls, tarps, reflections) tends to be
# either very jagged (low solidity) or a shape/aspect ratio that doesn't
# look leaf-like at all. Tune these if real leaves get rejected.
MIN_SOLIDITY = 0.55        # area / convex_hull_area - rejects jagged/irregular blobs
MIN_ASPECT_RATIO = 0.15    # width/height - rejects super thin slivers
MAX_ASPECT_RATIO = 5.0     # width/height - rejects super wide/flat strips

# ---- watershed tuning ----
# Fraction of the max distance-transform value used to mark "sure foreground"
# peaks (one peak per leaf). Lower = more/smaller separate leaves detected
# (good when leaves overlap a lot); higher = fewer, larger merges.
WATERSHED_PEAK_RATIO = 0.35

STRESSED_LEAF_PERCENT_THRESHOLD = 40.0

def _is_leaf_shaped(contour, area):
    """Reject blobs that don't look like a leaf (background clutter, shadows,
    reflections, edges of pots/wires etc that happen to be green)."""
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False
    solidity = area / hull_area
    if solidity < MIN_SOLIDITY:
        return False

    x, y, bw, bh = cv2.boundingRect(contour)
    if bh == 0:
        return False
    aspect_ratio = bw / float(bh)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False

    return True

def _segment_leaf_blobs(mask_clean):
    """Separates touching/overlapping leaves into individual blobs using
    watershed (distance transform), instead of relying on plain external
    contours which merge every touching leaf into ONE giant contour/box.

    Returns a list of individual leaf masks (each a single-blob uint8 mask
    the same size as mask_clean), one per separated leaf region.
    """
    dist = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    _, sure_fg = cv2.threshold(dist, WATERSHED_PEAK_RATIO * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 1:
        return []  # nothing distinct found

    unknown = cv2.subtract(mask_clean, sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # watershed needs a 3-channel image
    color_for_watershed = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_for_watershed, markers)

    leaf_masks = []
    for label in range(2, num_markers + 1):  # label 1 = background
        leaf_mask = np.uint8(markers == label) * 255
        if cv2.countNonZero(leaf_mask) > 0:
            leaf_masks.append(leaf_mask)

    return leaf_masks

def analyze_leaf_health(img):
    """
    Segments the frame into green (leaf) regions, SEPARATES touching leaves
    with watershed so each gets its own box, FILTERS OUT non-leaf-shaped
    green blobs, then crops + classifies each real leaf with the YOLO model.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 35, 35])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    total_pixels = img.shape[0] * img.shape[1]
    green_pixels = cv2.countNonZero(mask_clean)
    green_ratio = green_pixels / total_pixels

    # ---- Try to separate touching leaves first (watershed) ----
    leaf_masks = _segment_leaf_blobs(mask_clean)

    # Build the list of candidate contours to evaluate: prefer the
    # watershed-separated blobs (many small leaves); fall back to plain
    # external contours only if watershed found nothing usable (e.g. a
    # single isolated leaf with no touching neighbours).
    candidate_contours = []
    if leaf_masks:
        for lm in leaf_masks:
            cnts, _ = cv2.findContours(lm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidate_contours.extend(cnts)
    else:
        candidate_contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaves = []
    for c in candidate_contours:
        area = cv2.contourArea(c)
        if area < MIN_LEAF_AREA:
            continue  # too small - likely noise, not a real leaf

        if not _is_leaf_shaped(c, area):
            continue  # doesn't look like a leaf - skip (don't box it)

        x, y, bw, bh = cv2.boundingRect(c)
        leaf_crop = img[y:y + bh, x:x + bw]

        leaf_stress_level = 0.0
        is_stressed = False

        if model_available and leaf_crop.size > 0:
            try:
                result = yolo_model(leaf_crop, verbose=False, device=device)
                # IMPORTANT: best.pt is a CLASSIFICATION model, not a
                # detection model - ultralytics returns classification
                # results in result[0].probs (top1 / top1conf), NOT
                # result[0].boxes. .boxes is always empty for a -cls model.
                #
                # FIXED: previously this hardcoded "class index 1 =
                # Stressed", but YOLO assigns indices based on the
                # ALPHABETICAL ORDER of your training folder names - if
                # your folders weren't literally "0_healthy"/"1_stressed"
                # or similar, index 1 might actually BE healthy, silently
                # forcing every leaf to read as healthy (0% stress) no
                # matter what the model saw. Now we match by the class
                # NAME resolved once at startup (STRESSED_CLASS_IDX),
                # which is safe regardless of alphabetical ordering.
                probs = result[0].probs
                if probs is not None:
                    pred_class_idx = int(probs.top1)
                    pred_conf = float(probs.top1conf)
                    is_stressed = (pred_class_idx == STRESSED_CLASS_IDX)
                    leaf_stress_level = pred_conf if is_stressed else (1.0 - pred_conf)
                else:
                    is_stressed = False
                    leaf_stress_level = 0.0
            except Exception as e:
                logger.warning(f"YOLO predict failed on leaf crop: {str(e)}")
        else:
            leaf_stress_level = 0.0
            is_stressed = False

        leaves.append({
            "bbox": (x, y, bw, bh),
            "area": float(area),
            "stress_level": float(leaf_stress_level),
            "is_stressed": is_stressed
        })

    leaves.sort(key=lambda l: -l["area"])

    num_leaves = len(leaves)
    num_stressed = sum(1 for l in leaves if l["is_stressed"])
    stressed_percent = (num_stressed / num_leaves * 100.0) if num_leaves > 0 else 0.0
    avg_stress_level = (sum(l["stress_level"] for l in leaves) / num_leaves) if num_leaves > 0 else 0.0

    is_plant_detected = num_leaves > 0 or green_ratio > 0.05

    return {
        "leaves": leaves,
        "num_leaves": num_leaves,
        "num_stressed": num_stressed,
        "stressed_percent": float(stressed_percent),
        "stress_level": float(avg_stress_level),
        "color_abnormal": stressed_percent > STRESSED_LEAF_PERCENT_THRESHOLD,
        "green_ratio": float(green_ratio),
        "is_plant_detected": bool(is_plant_detected),
        "mask_green": mask_clean
    }

def draw_detection_box(img, leaf_analysis, diagnosis):
    """Draws ONE box PER detected leaf (red = stressed, green = healthy).
    Percentage is hidden when stress_level is ~0% so the label just reads
    "OK" instead of "OK 0%"."""
    output = img.copy()
    h, w, _ = output.shape

    if leaf_analysis['is_plant_detected'] and leaf_analysis['num_leaves'] > 0:
        for leaf in leaf_analysis['leaves']:
            x, y, bw, bh = leaf['bbox']
            is_stressed = leaf['is_stressed']
            color = (0, 0, 255) if is_stressed else (0, 200, 0)  # BGR: red / green
            cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 2)

            pct = leaf['stress_level'] * 100
            if pct < 1.0:
                # Don't show a misleading "0%" - just show the state.
                label = "STRESS" if is_stressed else "OK"
            else:
                label = f"{'STRESS' if is_stressed else 'OK'} {pct:.0f}%"

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)