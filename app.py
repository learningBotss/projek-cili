"""
CHILI STRESS DETECTION BACKEND - PYTHON FLASK
UiTM ITT569 IoT Final Project - CDCS259
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import tensorflow as tf
import json
from datetime import datetime
import os
import logging

# ========== CONFIGURATION ==========
def index():
    return render_template('dashboard.html')
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== LOAD PRE-TRAINED MODEL ==========
# Padan tepat dengan model yang di-train menggunakan folder Pepper__bell dari Kaggle PlantVillage
MODEL_PATH = "plant_disease_model.h5"
CLASS_NAMES = [
    "Chili Bell Bacterial Spot",  # Indeks 0 (Sakit - ikut turutan abjad folder dataset)
    "Chili Bell Healthy"          # Indeks 1 (Sihat)
]

# Cuba load model, kalau fail .h5 belum di-push, guna OpenCV fallback automatik
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model TensorFlow (.h5) berjaya dikesan dan di-load!")
        model_available = True
    else:
        logger.warning("Fail plant_disease_model.h5 tiada. Menggunakan OpenCV Fallback.")
        model_available = False
except Exception as e:
    logger.warning(f"Gagal load model: {str(e)}. Menggunakan OpenCV fallback.")
    model_available = False

# ========== SENSOR THRESHOLDS ==========
MOISTURE_THRESHOLD_DRY = 70.0  # % - Atas nilai ni = tanah kering

# ========== DATABASE SIMULATION (In-memory) ==========
sensor_history = []
max_history = 100

# ========== ROUTES ==========

@app.route('/', methods=['GET'])
def index():
    """Buka dashboard terus secara online melalui URL Render"""
    return render_template('dashboard.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint untuk Render monitor"""
    return jsonify({
        "status": "OK",
        "model_loaded": model_available,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/detect', methods=['POST'])
def detect_plant():
    try:
        # Ambil data sensor dari HTTP Headers (ESP32-CAM)
        soil_raw = request.headers.get('X-Soil-Raw', type=int, default=2500)
        soil_percent = request.headers.get('X-Soil-Percent', type=float, default=50.0)
        
        logger.info(f"Data Masuk -> Tanah: {soil_percent}% ({soil_raw})")
        
        # Ambil imej binari dari request body
        image_data = request.data
        if not image_data:
            return jsonify({"error": "Tiada imej diterima"}), 400
        
        # Decode imej format OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Gagal decode imej"}), 400
        
        # ========== STEP 1: ANALISIS WARNA DAUN (OPENCV) ==========
        leaf_analysis = analyze_leaf_health(img)
        
        # ========== STEP 2: DETECT PENYAKIT (AI MODEL / FALLBACK) ==========
        if model_available:
            disease_result = detect_disease_with_model(img)
        else:
            disease_result = detect_disease_simple(img, leaf_analysis)
        
        diagnosis = disease_result['diagnosis']
        disease_confidence = disease_result['confidence']
        has_disease = disease_result['has_disease']
        
        # ========== STEP 3: LOGIK KEPUTUSAN PINNTAR ==========
        action_water = False
        action_fertilize = False
        alert_disease = False
        decision_reason = ""
        
        soil_is_dry = soil_percent > MOISTURE_THRESHOLD_DRY
        leaf_is_stressed = leaf_analysis['stress_level'] > 0.5
        leaf_color_changed = leaf_analysis['color_abnormal']
        
        # Matriks keputusan:
        if soil_is_dry and not leaf_is_stressed:
            action_water = True
            decision_reason = "Tanah Kering + Daun Normal -> Pam Air Hidup"
            
        elif not soil_is_dry and (leaf_is_stressed or diagnosis == "Chili Bell Bacterial Spot"):
            action_fertilize = True
            decision_reason = "Tanah OK + Daun Sakit/Stres -> Pam Baja Hidup"
            
        if leaf_color_changed or has_disease or diagnosis == "Chili Bell Bacterial Spot":
            alert_disease = True
            decision_reason += f" | AMARAN PENYAKIT: {diagnosis}"
        
        if not action_water and not action_fertilize and not alert_disease:
            decision_reason = "Semua parameter normal - Pokok Cili Sihat!"
        
        # Simpan rekod dalam memori
        record = {
            "timestamp": datetime.now().isoformat(),
            "soil_percent": soil_percent,
            "soil_raw": soil_raw,
            "diagnosis": diagnosis,
            "disease_confidence": disease_confidence,
            "leaf_stress": leaf_analysis['stress_level'],
            "leaf_color_normal": not leaf_analysis['color_abnormal'],
            "action_water": action_water,
            "action_fertilize": action_fertilize,
            "alert_disease": alert_disease
        }
        sensor_history.append(record)
        if len(sensor_history) > max_history:
            sensor_history.pop(0)
            
        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "sensor": {
                "soil_raw": soil_raw,
                "soil_percent": soil_percent,
                "soil_status": "DRY" if soil_is_dry else "MOIST"
            },
            "leaf_analysis": leaf_analysis,
            "diagnosis": diagnosis,
            "confidence": disease_confidence,
            "has_disease": has_disease,
            "action_water": action_water,
            "action_fertilize": action_fertilize,
            "alert_disease": alert_disease,
            "decision_reason": decision_reason
        }), 200
        
    except Exception as e:
        logger.error(f"Error dalam fungsi detect_plant: {str(e)}")
        return jsonify({"error": str(e)}), 500

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

# ========== FUNCTIONS ==========

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
        confidence = float(predictions[0][class_idx])
        
        diagnosis = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "Unknown"
        has_disease = class_idx == 0  # 0 = Bacterial Spot (Sakit), 1 = Healthy
        
        return {"diagnosis": diagnosis, "confidence": confidence, "has_disease": has_disease}
    except:
        return {"diagnosis": "Model Error", "confidence": 0.0, "has_disease": False}

def detect_disease_simple(img, leaf_analysis):
    green_ratio = leaf_analysis['green_ratio']
    stress_level = leaf_analysis['stress_level']
    
    if stress_level > 0.6 and green_ratio < 0.25:
        return {"diagnosis": "Chili Bell Bacterial Spot (Estimated)", "confidence": 0.75, "has_disease": True}
    elif stress_level > 0.4 and green_ratio < 0.4:
        return {"diagnosis": "Leaf Stress / Nutrient Deficiency", "confidence": 0.60, "has_disease": True}
    else:
        return {"diagnosis": "Chili Bell Healthy", "confidence": 0.90, "has_disease": False}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
