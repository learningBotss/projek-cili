"""
CHILI STRESS DETECTION BACKEND - PYTHON FLASK
Deploy on Render.com or Railway.app (both free)

Receives image + sensor data from ESP32-CAM
Performs disease detection using TensorFlow
Applies decision logic
Returns JSON response with actions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import tensorflow as tf
import json
from datetime import datetime
import os
from pathlib import Path
import logging
from flask import render_template 

def index():
    return render_template('dashboard.html')
# ========== CONFIGURATION ==========
app = Flask(__name__)
CORS(app)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== LOAD PRE-TRAINED MODEL ==========
# Using a simple plant disease detection model
# For production, use: https://www.kaggle.com/datasets/emmarex/plantdisease

MODEL_PATH = "plant_disease_model.h5"
CLASS_NAMES = [
    "Healthy",
    "Powdery Mildew",
    "Leaf Spot",
    "Blight",
    "Rust",
    "Wilt",
    "Mosaic",
    "Unknown Disease"
]

# Try to load model, use fallback if not available
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
    model_available = True
except:
    logger.warning("Model not found - using simple color-based detection fallback")
    model_available = False

# ========== SENSOR THRESHOLDS ==========
SOIL_WET = 1500
SOIL_DRY = 3500

# Decision logic thresholds
MOISTURE_THRESHOLD_DRY = 70  # % - above this = dry, trigger watering
STRESS_CONFIDENCE_THRESHOLD = 0.6  # confidence needed to trigger fertilizer
COLOR_CHANGE_THRESHOLD = 0.5

# ========== DATABASE SIMULATION (In-memory) ==========
# In production, use PostgreSQL/MongoDB
sensor_history = []
max_history = 100

# ========== ROUTES ==========
@app.route('/', methods=['GET'])
# Setup logging
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "OK",
        "model_loaded": model_available,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/detect', methods=['POST'])
def detect_plant():
    """
    Main endpoint - receives image + sensor data
    
    Expected headers:
    - X-Soil-Raw: raw analog value
    - X-Soil-Percent: moisture percentage
    - Content-Type: application/octet-stream
    
    Returns JSON with detection results and actions
    """
    
    try:
        # Get sensor data from headers
        soil_raw = request.headers.get('X-Soil-Raw', type=int, default=2500)
        soil_percent = request.headers.get('X-Soil-Percent', type=float, default=50.0)
        
        logger.info(f"Request received - Soil: {soil_percent}% ({soil_raw})")
        
        # Get image from request body
        image_data = request.data
        if not image_data:
            return jsonify({"error": "No image data received"}), 400
        
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        logger.info(f"Image received - Size: {img.shape}")
        
        # ========== STEP 1: ANALYZE IMAGE ==========
        leaf_analysis = analyze_leaf_health(img)
        
        # ========== STEP 2: DISEASE DETECTION ==========
        if model_available:
            disease_result = detect_disease_with_model(img)
        else:
            disease_result = detect_disease_simple(img, leaf_analysis)
        
        diagnosis = disease_result['diagnosis']
        disease_confidence = disease_result['confidence']
        has_disease = disease_result['has_disease']
        
        # ========== STEP 3: APPLY DECISION LOGIC ==========
        # Your logic from the document:
        # 1. Soil dry + leaf normal → water
        # 2. Soil OK + leaf stress → fertilize
        # 3. Leaf color change → alert
        
        action_water = False
        action_fertilize = False
        alert_disease = False
        decision_reason = ""
        
        soil_is_dry = soil_percent > MOISTURE_THRESHOLD_DRY
        leaf_is_stressed = leaf_analysis['stress_level'] > 0.5
        leaf_color_changed = leaf_analysis['color_abnormal']
        
        # Decision matrix
        if soil_is_dry and not leaf_is_stressed:
            # Soil dry but leaf normal = watering issue
            action_water = True
            decision_reason = "Soil dry + Leaf normal → Activate water pump"
            
        elif not soil_is_dry and leaf_is_stressed:
            # Soil OK but leaf stressed = nutrient/disease issue
            action_fertilize = True
            decision_reason = "Soil OK + Leaf stressed → Activate fertilizer pump"
            
        if leaf_color_changed or has_disease:
            # Abnormal color or disease detected
            alert_disease = True
            if has_disease:
                decision_reason += f" | DISEASE ALERT: {diagnosis}"
            else:
                decision_reason += " | Color abnormality detected"
        
        # If everything normal
        if not action_water and not action_fertilize and not alert_disease:
            decision_reason = "All parameters normal - no action needed"
        
        # ========== SAVE TO HISTORY ==========
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
        
        logger.info(f"Detection complete: {decision_reason}")
        
        # ========== PREPARE RESPONSE ==========
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            
            # Sensor data echo
            "sensor": {
                "soil_raw": soil_raw,
                "soil_percent": soil_percent,
                "soil_status": "DRY" if soil_is_dry else "MOIST"
            },
            
            # Leaf analysis
            "leaf_analysis": {
                "stress_level": leaf_analysis['stress_level'],
                "color_abnormal": leaf_analysis['color_abnormal'],
                "green_ratio": leaf_analysis['green_ratio'],
                "health_score": leaf_analysis['health_score']
            },
            
            # Disease detection
            "diagnosis": diagnosis,
            "confidence": disease_confidence,
            "has_disease": has_disease,
            
            # Actions
            "action_water": action_water,
            "action_fertilize": action_fertilize,
            "alert_disease": alert_disease,
            
            # Decision reasoning
            "decision_reason": decision_reason,
            "next_check_minutes": 30
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in detect_plant: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get sensor history"""
    return jsonify({
        "count": len(sensor_history),
        "data": sensor_history[-20:]  # Last 20 records
    }), 200

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    if not sensor_history:
        return jsonify({"message": "No data yet"}), 200
    
    avg_soil = np.mean([r['soil_percent'] for r in sensor_history])
    avg_stress = np.mean([r['leaf_stress'] for r in sensor_history])
    disease_count = sum(1 for r in sensor_history if r['alert_disease'])
    
    return jsonify({
        "total_checks": len(sensor_history),
        "avg_soil_moisture": avg_soil,
        "avg_leaf_stress": avg_stress,
        "disease_alerts_count": disease_count,
        "last_check": sensor_history[-1]['timestamp'] if sensor_history else None
    }), 200

# ========== IMAGE ANALYSIS FUNCTIONS ==========

def analyze_leaf_health(img):
    """
    Analyze leaf health from image
    Returns: stress level, color abnormality, green ratio, health score
    """
    
    # Resize for faster processing
    img_small = cv2.resize(img, (200, 200))
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    
    # Define green color range (healthy leaves)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green pixels
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask_green)
    total_pixels = img_small.shape[0] * img_small.shape[1]
    green_ratio = green_pixels / total_pixels
    
    # Detect yellow/brown (signs of stress)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    
    # Calculate stress level (more yellow = more stress)
    stress_level = (yellow_pixels / total_pixels) * 2  # Scale to 0-1
    stress_level = min(stress_level, 1.0)
    
    # Color abnormality (if green ratio is low)
    color_abnormal = green_ratio < 0.3
    
    # Health score (0-100)
    health_score = green_ratio * 100
    
    return {
        "stress_level": float(stress_level),
        "color_abnormal": bool(color_abnormal),
        "green_ratio": float(green_ratio),
        "health_score": float(health_score)
    }

def detect_disease_with_model(img):
    """
    Use TensorFlow model for disease detection
    Requires trained model (download from Kaggle dataset)
    """
    
    try:
        # Preprocess image
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        predictions = model.predict(img_batch, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        diagnosis = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "Unknown"
        has_disease = class_idx > 0  # 0 = Healthy
        
        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "has_disease": has_disease
        }
        
    except Exception as e:
        logger.error(f"Model prediction error: {str(e)}")
        return {
            "diagnosis": "Model Error",
            "confidence": 0.0,
            "has_disease": False
        }

def detect_disease_simple(img, leaf_analysis):
    """
    Fallback simple disease detection using color analysis
    Works without ML model
    """
    
    green_ratio = leaf_analysis['green_ratio']
    stress_level = leaf_analysis['stress_level']
    
    # Simple logic:
    # - High stress + abnormal color = likely disease
    # - Just yellow/brown = nutrient deficiency
    
    if stress_level > 0.7 and green_ratio < 0.2:
        diagnosis = "Severe Leaf Damage / Disease"
        confidence = 0.8
        has_disease = True
        
    elif stress_level > 0.5 and green_ratio < 0.4:
        diagnosis = "Leaf Stress / Possible Nutrient Deficiency"
        confidence = 0.6
        has_disease = True
        
    elif green_ratio < 0.5:
        diagnosis = "Color Abnormality Detected"
        confidence = 0.5
        has_disease = True
        
    else:
        diagnosis = "Healthy"
        confidence = 0.95
        has_disease = False
    
    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "has_disease": has_disease
    }

# ========== START SERVER ==========

if __name__ == '__main__':
    # Get port from environment (Render/Railway provide this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    # For production, use gunicorn:
    # gunicorn --workers 2 --threads 2 -w 1 -b 0.0.0.0:$PORT app:app
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )
