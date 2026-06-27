"""
CHILI STRESS DETECTION BACKEND - PYTHON FLASK
UiTM ITT569 IoT Final Project - CDCS259
*Edisi Pembetulan Logik Amaran Tanah Kering*
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
import pytz  # Untuk selesaikan masalah waktu Malaysia (MYT)

# ========== CONFIGURATION ==========
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tetapkan Zon Masa Malaysia
MY_TIMEZONE = pytz.timezone('Asia/Kuala_Lumpur')

def get_malaysia_time():
    return datetime.now(MY_TIMEZONE)

# ========== LOAD PRE-TRAINED MODEL ==========
MODEL_PATH = "plant_disease_model.h5"
CLASS_NAMES = [
    "Chili Bell Bacterial Spot",  # Indeks 0 (Sakit)
    "Chili Bell Healthy"          # Indeks 1 (Sihat)
]

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model TensorFlow (.h5) berjaya dikesan!")
        model_available = True
    else:
        logger.warning("Fail plant_disease_model.h5 tiada. Guna OpenCV Fallback.")
        model_available = False
except Exception as e:
    logger.warning(f"Gagal load model: {str(e)}. Guna OpenCV fallback.")
    model_available = False

# ========== SENSOR THRESHOLDS ==========
# PEMBETULAN: Mengikut tetapan map baru ESP32: Peratusan rendah (< 30%) bermaksud tanah kering
MOISTURE_THRESHOLD_DRY = 30.0  

# ========== DATA STORAGE ==========
latest_image_base64 = None  
latest_image_timestamp = None
sensor_history = []
max_history = 100

# Global state untuk dikongsi dengan ESP32 Node (Pam)
current_pump_status = {
    "action_water": False,
    "action_fertilize": False,
    "alert_disease": False,
    "decision_reason": "Menunggu data awal dikesan..."
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
        
        # Ambil data sensor dari HTTP Headers (ESP32-CAM)
        soil_raw = request.headers.get('X-Soil-Raw', type=int, default=2500)
        soil_percent = request.headers.get('X-Soil-Percent', type=float, default=50.0)
        
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
        if model_available and leaf_analysis['is_plant_detected']:
            disease_result = detect_disease_with_model(img)
        else:
            disease_result = detect_disease_simple(img, leaf_analysis)
        
        diagnosis = disease_result['diagnosis']
        disease_confidence = disease_result['confidence']
        has_disease = disease_result['has_disease']
        
        # ========== STEP 3: LOGIK KEPUTUSAN PINTAR ==========
        action_water = False
        action_fertilize = False
        alert_disease = False
        decision_reason = ""
        
        # PEMBETULAN: soil_percent di bawah threshold baru dikira kering lencun
        soil_is_dry = soil_percent < MOISTURE_THRESHOLD_DRY
        leaf_is_stressed = leaf_analysis['stress_level'] > 0.5
        
        # Hanya jalankan logik jika tumbuhan sahih dikesan
        if leaf_analysis['is_plant_detected']:
            if soil_is_dry and not leaf_is_stressed:
                action_water = True
                decision_reason = "Tanah Kering + Daun Normal -> Pam Air Hidup"
            elif not soil_is_dry and (leaf_is_stressed or has_disease):
                action_fertilize = True
                decision_reason = "Tanah OK + Pokok Stres/Sakit -> Pam Baja Hidup"
            
            if has_disease:
                alert_disease = True
                decision_reason += f" | AMARAN PENYAKIT: {diagnosis}"
            
            if not action_water and not action_fertilize and not alert_disease:
                decision_reason = "Semua parameter normal - Pokok Cili Sihat!"
        else:
            # Jika tiada pokok dikesan (Contohnya nampak lantai/dinding sahaja)
            diagnosis = "No Chili Plant Detected"
            disease_confidence = 1.0
            has_disease = False
            decision_reason = "Tiada imej pokok cili dikesan dalam kawasan video."
        
        # ========== STEP 4: LUKIS KOTAK MERAH (BOUNDING BOX) ==========
        processed_img = draw_detection_box(img, leaf_analysis, diagnosis)
        
        # Tukar imej siap diproses ke Base64 string
        _, buffer = cv2.imencode('.jpg', processed_img)
        latest_image_base64 = base64.b64encode(buffer).decode('utf-8')
        latest_image_timestamp = get_malaysia_time().isoformat()
        
        # Simpan status untuk dikongsi dengan ESP32 Node
        current_pump_status = {
            "action_water": action_water,
            "action_fertilize": action_fertilize,
            "alert_disease": alert_disease,
            "decision_reason": decision_reason
        }
        
        # Tentukan keperluan pokok berdasarkan status pam yang aktif
        if action_water and action_fertilize:
            plant_needs = "Need Water & Fertilizer"
        elif action_water:
            plant_needs = "Need Water"
        elif action_fertilize:
            plant_needs = "Need Fertilizer"
        else:
            plant_needs = "Optimal (No Action)"

        # Simpan rekod dalam memori sejarah
        record = {
            "timestamp": latest_image_timestamp,
            "soil_percent": soil_percent,
            "soil_raw": soil_raw,
            "diagnosis": plant_needs,  # <--- Kita hantar status keperluan pokok ke frontend
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
            
        return jsonify({"success": True, "message": "Analisis Berjaya"}), 200
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/soil', methods=['POST'])
def receive_soil_data():
    """Menerima data kelembapan tanah terus dari peranti ESP32 Node"""
    try:
        data = request.json
        soil_raw = data.get("soil_raw", 2500)
        soil_percent = data.get("soil_percent", 50.0)
        
        # Ambil status diagnosis terakhir jika ada
        last_diagnosis = sensor_history[-1]['diagnosis'] if sensor_history else "Menunggu Kamera"
        last_confidence = sensor_history[-1]['disease_confidence'] if sensor_history else 0.0
        last_stress = sensor_history[-1]['leaf_stress'] if sensor_history else 0.0
        
        # PEMBETULAN LOGIK: soil_percent di bawah threshold baru dikira kering lencun
        soil_is_dry = soil_percent < MOISTURE_THRESHOLD_DRY
        
        if last_diagnosis == "Chili Bell Healthy":
            if soil_is_dry:
                current_pump_status["action_water"] = True
                current_pump_status["action_fertilize"] = False
                current_pump_status["decision_reason"] = "Data Tanah: Kering -> Pam Air Hidup"
            else:
                current_pump_status["action_water"] = False
                current_pump_status["action_fertilize"] = False
                current_pump_status["decision_reason"] = "Semua parameter normal - Pokok Cili Sihat!"
                
        elif "Bacterial Spot" in last_diagnosis or last_stress > 0.5:
            if not soil_is_dry:
                current_pump_status["action_water"] = False
                current_pump_status["action_fertilize"] = True
                current_pump_status["decision_reason"] = "Data Tanah: OK + Pokok Stres -> Pam Baja Hidup"
        
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

@app.route('/pump-status', methods=['GET'])
def get_pump_status():
    """Endpoint untuk dibaca oleh ESP32 Actuator Node kontrol relay"""
    return jsonify(current_pump_status), 200

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

# ========== PROCESS FUNCTIONS ==========

def analyze_leaf_health(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Range Warna Hijau Daun Cili
    lower_green = np.array([35, 35, 35])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask_green)
    total_pixels = img.shape[0] * img.shape[1]
    green_ratio = green_pixels / total_pixels
    
    # Range Warna Kuning (Tanda Stres)
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    
    stress_level = (yellow_pixels / total_pixels) * 3
    stress_level = min(stress_level, 1.0)
    
    # Sahkan sama ada objek tersebut pokok cili atau lantai kosong
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
        # Cari kontur luaran pokok untuk lukis kotak pengesanan merah
        contours, _ = cv2.findContours(leaf_analysis['mask_green'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Ambil kontur terbesar
            c = max(contours, key=cv2.contourArea)
            x, y, box_w, box_h = cv2.boundingRect(c)
            
            # Lukis kotak merah utama sekeliling kawasan daun terkesan
            cv2.rectangle(output, (x, y), (x + box_w, y + box_h), (0, 0, 255), 3)
            
            # Labelkan tag di atas kotak merah
            label = f"{diagnosis}"
            cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        # Jika tiada pokok dikesan, lukis kotak sempadan amaran di seluruh skrin feed
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
