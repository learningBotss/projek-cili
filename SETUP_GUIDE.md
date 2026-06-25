# CHILI STRESS DETECTION SYSTEM - COMPLETE SETUP GUIDE
## UiTM ITT569 IoT Final Project

---

## 📋 OVERVIEW
Full stack setup untuk:
1. **ESP32-CAM** → capture image + read soil sensor
2. **Python Backend** → process image + apply decision logic
3. **Web Dashboard** → monitor + control system

Total time: ~1-2 hours

---

## ⚙️ PART 1: PREPARE YOUR HARDWARE

### 1.1 Verify Your Components
```
☑️ ESP32-CAM module + USB cable
☑️ Soil moisture sensor (analog)
☑️ Water pump + relay module
☑️ Fertilizer pump + relay module
☑️ Power supply (5V)
☑️ Laptop/PC for programming
```

### 1.2 Document Your GPIO Pins
Test your circuit first. Write down which GPIO each component uses:

```
Soil Moisture Sensor → GPIO _____ (ADC pin)
Water Pump Relay → GPIO _____
Fertilizer Pump Relay → GPIO _____
```

Default in code:
- Soil: GPIO 34 (ADC1_CH6)
- Water Pump: GPIO 18
- Fertilizer Pump: GPIO 19

If different, update in ESP32 code BEFORE uploading.

---

## 📱 PART 2: SETUP ESP32-CAM (Arduino IDE)

### 2.1 Install Arduino IDE
Download: https://www.arduino.cc/en/software

### 2.2 Add ESP32 Support to Arduino IDE

**For Windows/Mac/Linux:**

1. Open Arduino IDE
2. Go to **File → Preferences**
3. In "Additional Board Manager URLs", paste:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Click OK
5. Go to **Tools → Board → Boards Manager**
6. Search for "ESP32"
7. Install "esp32" by Espressif Systems (latest version)

### 2.3 Configure Board Settings

1. **Tools → Board:** Select "ESP32 Wrover Module"
2. **Tools → Port:** Select your COM port (e.g., COM3)
3. **Tools → Upload Speed:** Set to 115200

### 2.4 Test Camera Connection
Before loading chili code, test if camera works:

1. **File → Examples → ESP32 → Camera → CameraWebServer**
2. Edit code:
   - Change WiFi SSID & password
   - Select camera model: **CAMERA_MODEL_AI_THINKER**
3. Upload to ESP32
4. Open Serial Monitor (Tools → Serial Monitor, 115200 baud)
5. When it shows IP address, go to http://[IP_ADDRESS] in browser
6. You should see live camera feed

**If camera doesn't work:**
- Check camera ribbon cable connection
- Try different USB cable
- Try different USB port

### 2.5 Upload Chili Stress Detection Code

1. **Copy** the full `ESP32_CAM_Chili_Detection.ino` file
2. **Paste** into Arduino IDE
3. **CRITICAL - UPDATE THESE FIRST:**

   ```cpp
   // Line ~37-39: WiFi credentials
   const char* ssid = "YOUR_WIFI_NAME";           // ← CHANGE THIS
   const char* password = "YOUR_WIFI_PASSWORD";   // ← CHANGE THIS
   
   // Line ~40: Backend server URL
   const char* serverUrl = "http://YOUR_BACKEND_SERVER/detect"; 
   // ← For local testing: http://192.168.1.XX:5000/detect
   // ← For Render: https://your-app.render.com/detect
   
   // Line ~46-49: GPIO Pins (update if different from your circuit)
   #define SOIL_SENSOR_PIN 34
   #define RELAY_WATER_PIN 18
   #define RELAY_FERT_PIN 19
   ```

4. **Verify** → **Upload** to ESP32

5. Open Serial Monitor (115200 baud)

   Expected output:
   ```
   ========================================
   CHILI STRESS DETECTION SYSTEM - STARTING
   ========================================
   [INIT] Pumps set to OFF
   [CAMERA] Camera initialized successfully!
   [WiFi] Connected!
   [WiFi] IP Address: 192.168.1.XX
   [INIT] System ready! First detection starting...
   ```

6. **If errors:**
   - Check WiFi credentials (typo?)
   - Check backend URL reachable (ping first)
   - Check Serial Monitor for specific error messages

---

## 🖥️ PART 3: SETUP PYTHON BACKEND

### Option A: Local Testing (Laptop)

**3A.1 Install Python**
- Download Python 3.9+ from python.org
- During install: ☑️ "Add Python to PATH"

**3A.2 Create Backend Folder**
```bash
mkdir chili-backend
cd chili-backend
```

**3A.3 Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**3A.4 Install Dependencies**
```bash
pip install -r requirements.txt
```

(If installation slow, use: `pip install --no-cache-dir -r requirements.txt`)

**3A.5 Run Backend**
```bash
python app.py
```

Expected output:
```
 * Running on http://0.0.0.0:5000
WARNING: This is a development server. Do not use it in production.
```

✓ Backend is now running on http://localhost:5000

### Option B: Deploy to Render.com (Production)

**3B.1 Create Render Account**
- Go to https://render.com
- Sign up (free)

**3B.2 Create GitHub Repository** (if you don't have one)
- Go to https://github.com
- Create new public repo called `chili-detection`
- Upload these files:
  - `app.py`
  - `requirements.txt`

**3B.3 Deploy to Render**
1. In Render dashboard, click **New +** → **Web Service**
2. Connect GitHub account
3. Select your `chili-detection` repo
4. Configure:
   - **Name:** chili-detection
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --workers 2 --threads 2 -w 1 -b 0.0.0.0:$PORT app:app`
5. Click **Create Web Service**
6. Wait 2-3 minutes for deployment
7. Copy the URL: `https://chili-detection-xxxxx.onrender.com`

**Note:** Free Render tier may go to sleep after 15 min inactivity. Upgrade to paid ($7/month) for always-on.

---

## 📊 PART 4: SETUP DASHBOARD

### 4.1 Update Backend URL in Dashboard

Open `dashboard.html`:

```javascript
// Line ~263: CHANGE THIS
const BACKEND_URL = "http://localhost:5000"; // Local testing
// OR
const BACKEND_URL = "https://chili-detection-xxxxx.onrender.com"; // Production
```

### 4.2 Open Dashboard

**Local Testing:**
1. Save `dashboard.html` on your desktop
2. Double-click to open in browser
3. You should see the dashboard

**Production:**
- Upload `dashboard.html` to:
  - Netlify (free)
  - GitHub Pages (free)
  - Or any web hosting

---

## 🚀 PART 5: FULL SYSTEM TEST

### 5.1 Check All Components

**Terminal 1 - ESP32 Serial Monitor:**
```
[INIT] System ready!
[TIMER] 30-minute interval reached - Starting detection...
[STEP 1] Reading soil moisture sensor...
[STEP 2] Capturing image from camera...
[STEP 3] Sending to backend server...
```

**Terminal 2 - Python Backend:**
```
127.0.0.1 - - [25/Jun/2026 10:30:45] "POST /detect HTTP/1.1" 200 -
```

**Browser - Dashboard:**
```
✓ Soil Moisture: XX%
✓ Leaf Health: XX%
✓ Diagnosis: Healthy
```

### 5.2 Test Water Pump Manually

Edit ESP32 code, in `void loop()` add:
```cpp
digitalWrite(RELAY_WATER_PIN, HIGH);
delay(5000);
digitalWrite(RELAY_WATER_PIN, LOW);
```

Upload and check if pump activates.

### 5.3 Test System Decision Logic

Scenarios to test:

**Scenario 1: Dry Soil (Trigger Water)**
- Remove soil moisture sensor into dry area
- Expected: Water pump activates

**Scenario 2: Wet Soil (No watering)**
- Keep soil sensor in wet soil
- Expected: Pump stays off

**Scenario 3: Abnormal Leaf Color (Alert)**
- Point camera at discolored leaf or paper
- Expected: Dashboard shows diagnosis alert

---

## 🔧 TROUBLESHOOTING

### "ESP32 not showing in COM ports"
- Try different USB cable
- Try different USB port on computer
- Install CH340 driver: https://www.wemos.cc/en/latest/ch340_driver.html
- Restart Arduino IDE

### "Camera returns NULL error"
```
[ERROR] Camera init failed: 0x101
```
- Check camera ribbon cable (tight connection?)
- Check correct camera model selected
- Try with new SD card in camera (if it has one)

### "Backend connection refused"
```
[ERROR] HTTP POST failed with code: -1
```
- Check ESP32 WiFi connected (Serial Monitor shows IP)
- Check backend URL correct in ESP32 code
- Test backend URL in browser: http://192.168.1.XX:5000/health
- Should return: `{"status": "OK"}`

### "Model not found" warning
```
Model not found - using simple color-based detection fallback
```
This is OK! System will use color analysis instead of ML model.

To use ML model:
1. Download pre-trained model from Kaggle
2. Save as `plant_disease_model.h5` in backend folder
3. Restart backend

Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease

### "Dashboard shows 'Cannot connect to backend'"
- Check backend running: `python app.py`
- Check BACKEND_URL in dashboard.html correct
- Check browser console (F12) for CORS errors
- If Render: wait 2-3 minutes after deployment

---

## 📈 SYSTEM FLOW DIAGRAM

```
┌─────────────┐
│  ESP32-CAM  │
├─────────────┤
│ • Camera    │
│ • WiFi      │
│ • Sensors   │
└──────┬──────┘
       │ POST /detect
       │ (image + sensor data)
       ▼
┌──────────────────────┐
│  Python Flask Srv    │
├──────────────────────┤
│ • Image Analysis     │
│ • Disease Detection  │
│ • Decision Logic     │
│ • JSON Response      │
└──────┬───────────────┘
       │ {"action_water": true, ...}
       ▼
┌─────────────────┐
│  ESP32-CAM      │
├─────────────────┤
│ • Relay Control │
│ • Pump ON/OFF   │
└─────────────────┘
       │
       ├──→ 💧 Water Pump
       └──→ 🌱 Fertilizer Pump

┌──────────────────────┐
│  Web Dashboard       │
├──────────────────────┤
│ • Fetch /history     │
│ • Display data       │
│ • Manual controls    │
│ • Alerts             │
└──────────────────────┘
```

---

## 🎯 DECISION LOGIC (Your Specification)

```cpp
if (Soil Dry & Leaf Normal) {
    → Activate WATER PUMP
    → "Likely just needs watering"
}

else if (Soil OK & Leaf Stressed) {
    → Activate FERTILIZER PUMP
    → "Nutrient deficiency likely"
}

else if (Leaf Color Change) {
    → ALERT USER
    → "Check for disease"
}

else {
    → ALL NORMAL
    → "No action needed"
}
```

---

## 📝 USEFUL COMMANDS

**ESP32 Serial Monitor Baud Rates to Try if garbled:**
```
115200 ← Most common (try this first)
74880 ← Sometimes default
921600 ← High speed
```

**Python Debugging:**
```bash
# Test backend without image
curl http://localhost:5000/health

# View recent records
curl http://localhost:5000/history

# View stats
curl http://localhost:5000/stats
```

**Stop Python Server:**
Press `Ctrl + C` in terminal

**Check WiFi Connectivity from ESP32:**
Serial Monitor should show:
```
[WiFi] Connected!
[WiFi] IP Address: 192.168.1.XX
```

---

## 📞 NEXT STEPS

After basic system works:

1. **Calibrate Soil Sensor**
   - Test dry and wet threshold values
   - Adjust `SOIL_WET` and `SOIL_DRY` in ESP32 code

2. **Train ML Model** (optional)
   - Download Kaggle plant disease dataset
   - Train TensorFlow model
   - Replace placeholder in `app.py`

3. **Add Mobile App**
   - Use Blynk app for notifications
   - Or write custom Flutter app

4. **Database Integration**
   - Switch from in-memory to PostgreSQL
   - Store historical data long-term

5. **Deploy Dashboard**
   - Host on Netlify/GitHub Pages
   - Add mobile responsive design

---

## 📚 RESOURCES

**ESP32 Documentation:**
https://docs.espressif.com/projects/esp-idf/en/latest/

**Arduino Libraries Used:**
- esp_camera.h (built-in)
- WiFi.h (built-in)
- HTTPClient.h (built-in)
- ArduinoJson.h (install via Library Manager)

**Flask Documentation:**
https://flask.palletsprojects.com/

**TensorFlow/OpenCV:**
https://docs.opencv.org/
https://www.tensorflow.org/

---

## ✅ SUCCESS CHECKLIST

- [ ] ESP32-CAM connects to WiFi
- [ ] Camera captures images (test via web server)
- [ ] Soil sensor reads correct values
- [ ] Backend server running (health check passes)
- [ ] Dashboard loads and connects to backend
- [ ] First detection completes (Serial Monitor shows complete routine)
- [ ] Pump relay activates when commanded
- [ ] Backend logs show received image
- [ ] Dashboard displays latest sensor data
- [ ] Decision logic works (actions triggered based on conditions)

---

**Good luck with your FYP! If stuck, debug step by step and check Serial Monitor.**

كُل التوفيق! (All the best!)
