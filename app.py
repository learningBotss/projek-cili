"""
CHILI STRESS DETECTION BACKEND - PYTHON FLASK (SPLIT DATA VERSION)
UiTM ITT569 IoT Final Project - CDCS259
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import tensorflow as tf
import json
from datetime import datetime
import os
import logging
import base64

# ========== CONFIGURATION ==========
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "plant_disease_model.h5"
CLASS_NAMES = ["Chili Bell Bacterial Spot", "Chili Bell Healthy"]

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model TensorFlow (.h5) berjaya dikesan!")
        model_available = True
    else:
        logger.warning("Fail model tiada. Menggunakan OpenCV Fallback.")
        model_available = False
except Exception as e:
    logger.warning(f"Gagal load model: {str(e)}. Menggunakan OpenCV fallback.")
    model_available = False

MOISTURE_THRESHOLD_DRY = 70.0

# Simpanan state global terkini
latest_soil_percent = 50.0
latest_soil_raw = 2500
latest_image = None
latest_image_base64 = None
latest_image_timestamp = None

# Cache keputusan analisa terakhir dari kamera
latest_diagnosis = "Chili Bell Healthy"
latest_confidence = 0.90
latest_has_disease = False
latest_leaf_analysis = {"stress_level": 0.0, "color_abnormal": False, "green_ratio": 0.5, "health_score": 100.0}

sensor_history = []
max_history = 100

def run_smart_logic():
    """Fungsi pusat untuk proses matriks keputusan fertigasi pintar"""
    global sensor_history
    
    action_water = False
    action_fertilize = False
    alert_disease = False
    decision_reason = ""
    
    soil_is_dry = latest_soil_percent > MOISTURE_THRESHOLD_DRY
    leaf_is_stressed = latest_leaf_analysis['stress_level'] > 0.5
    leaf_color_changed = latest_leaf_analysis['color_abnormal']
    
    hari_sekarang = datetime.now().weekday()
    hari_baja = [0, 3] # Isnin & Khamis (2 kali seminggu)
    
    if soil_is_dry:
        if (hari_sekarang in hari_baja) and (leaf_is_stressed or latest_diagnosis == "Chili Bell Bacterial Spot"):
            action_fertilize = True
            decision_reason = "Tanah Kering + Daun Stres/Sakit + [HARI BAJA] -> Pam Baja Hidup"
        else:
            action_water = True
            decision_reason = "Tanah Kering -> Pam Air Hidup untuk selamatkan pokok"
            
    elif not soil_is_dry and (leaf_is_stressed or latest_diagnosis == "Chili Bell Bacterial Spot"):
        if hari_sekarang in hari_baja:
            action_fertilize = True
            decision_reason = "Tanah OK + Daun Stres/Sakit + [HARI BAJA] -> Pam Baja Hidup"
        else:
            decision_reason = "Tanah OK + Daun Stres/Sakit TAPI [BUKAN HARI BAJA] -> Pam Ditahan"
    
    if leaf_color_changed or latest_has_disease or latest_diagnosis == "Chili Bell Bacterial Spot":
        alert_disease = True
        decision_reason += f" | AMARAN PENYAKIT: {latest_diagnosis}"
    
    if not action_water and not action_fertilize and not alert_disease:
        decision_reason = "Semua parameter normal - Pokok Cili Sihat!"
        
    record = {
        "timestamp": datetime.now().isoformat(),
        "soil_percent": latest_soil_percent,
        "soil_raw": latest_soil_raw,
        "diagnosis": latest_diagnosis,
        "disease_confidence": latest_confidence,
        "leaf_stress": latest_leaf_analysis['stress_level'],
        "leaf_color_normal": not latest_leaf_analysis['color_abnormal'],
        "action_water": action_water,
        "action_fertilize": action_fertilize,
        "alert_disease": alert_disease,
        "decision_reason": decision_reason
    }
    sensor_history.append(record)
    if len(sensor_history) > max_history:
        sensor_history.pop(0)

# ========== ROUTES ==========

@app.route('/', methods=['GET'])
def index():
    return render_template('dashboard.html')

@app.route('/soil', methods=['POST'])
def receive_soil():
    """Endpoint baru untuk terima data tanah dari ESP32 Biasa"""
    global latest_soil_percent, latest_soil_raw
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    
    latest_soil_raw = data.get('soil_raw', 2500)
    latest_soil_percent = data.get('soil_percent', 50.0)
    
    # Setiap kali data tanah baru masuk, kemaskini logik keputusan
    run_smart_logic()
    return jsonify({"success": True}), 200

@app.route('/pump-status', methods=['GET'])
def get_pump_status():
    """Endpoint untuk ESP32 biasa check status pam secara berkala"""
    if not sensor_history:
        return jsonify({"action_water": False, "action_fertilize": False}), 200
    latest_record = sensor_history[-1]
    return jsonify({
        "action_water": latest_record.get("action_water", False),
        "action_fertilize": latest_record.get("action_fertilize", False)
    }), 200

@app.route('/detect', methods=['POST'])
def detect_plant():
    """Endpoint menerima gambar dari ESP32-CAM"""
    try:
        global latest_image, latest_image_base64, latest_image_timestamp
        global latest_leaf_analysis, latest_diagnosis, latest_confidence, latest_has_disease
        
        image_data = request.data
        if not image_data:
            return jsonify({"error": "Tiada imej diterima"}), 400
        
        latest_image = image_data
        latest_image_base64 = base64.b64encode(image_data).decode('utf-8')
        latest_image_timestamp = datetime.now().isoformat()
        
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Gagal decode imej"}), 400
        
        # Analisis Imej & Simpan ke State Global
        latest_leaf_analysis = analyze_leaf_health(img)
        
        if model_available:
            disease_result = detect_disease_with_model(img)
        else:
            disease_result = detect_disease_simple(img, latest_leaf_analysis)
        
        latest_diagnosis = disease_result['diagnosis']
        latest_confidence = disease_result['confidence']
        latest_has_disease = disease_result['has_disease']
        
        # Selepas imej selesai diproses, kemaskini keputusan pintar
        run_smart_logic()
        
        return jsonify({"success": True, "message": "Image analyzed successfully"}), 200
        
    except Exception as e:
        logger.error(f"Error dalam fungsi detect_plant: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/latest-image', methods=['GET'])
def get_latest_image():
    if latest_image is None:
        return jsonify({"error": "No image captured yet"}), 404
    return send_file(BytesIO(latest_image), mimetype='image/jpeg')

@app.route('/latest-image-data', methods=['GET'])
def get_latest_image_data():
    if latest_image_base64 is None:
        return jsonify({"error": "No image captured yet"}), 404
    return jsonify({"image_base64": latest_image_base64, "timestamp": latest_image_timestamp, "status": "OK"})

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({"count": len(sensor_history), "data": sensor_history[-20:]})

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
    })

# ========== OPENCV & AI FUNCTIONS ==========
def analyze_leaf_health(img):
    img_small = cv2.resize(img, (200, 200))
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask_green)
    total_pixels = img_small.shape[0] * img_small.shape[1]
    green_ratio = green_pixels / total_pixels
    
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    
    stress_level = (yellow_pixels / total_pixels) * 2
    stress_level = min(stress_level, 1.0)
    color_abnormal = green_ratio < 0.35
    health_score = green_ratio * 100
    
    return {
        "stress_level": float(stress_level),
        "color_abnormal": bool(color_abnormal),
        "green_ratio": float(green_ratio),
        "health_score": float(health_score)
    }

def detect_disease_with_model(img):
    try:
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        predictions = model.predict(img_batch, verbose=0)
        class_idx = np.argmax(predictions[0])
        return {"diagnosis": CLASS_NAMES[class_idx], "confidence": float(predictions[0][class_idx]), "has_disease": class_idx == 0}
    except:
        return {"diagnosis": "Model Error", "confidence": 0.0, "has_disease": False}

def detect_disease_simple(img, leaf_analysis):
    green_ratio = leaf_analysis['green_ratio']
    stress_level = leaf_analysis['stress_level']
    if stress_level > 0.6 and green_ratio < 0.25:
        return {"diagnosis": "Chili Bell Bacterial Spot (Estimated)", "confidence": 0.75, "has_disease": True}
    elif stress_level > 0.4 and green_ratio < 0.4:
        return {"diagnosis": "Leaf Stress / Nutrient Deficiency", "confidence": 0.60, "has_disease": True}
    return {"diagnosis": "Chili Bell Healthy", "confidence": 0.90, "has_disease": False}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
