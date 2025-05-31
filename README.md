# 🌿 Plant Disease Detection & Smart Monitoring System

This project is a complete IoT + AI solution that enables **real-time monitoring of environmental conditions** and **detection of plant diseases** using a Raspberry Pi (Flask server), ESP32, and deep learning models (ResNet50 + CNN ensemble with YOLOv5).

---

## 🚀 Features

- 🌡️ Live sensor readings: **Temperature**, **Humidity**, **Soil Moisture** via ESP32
- 📸 **Leaf image capture** using Raspberry Pi Camera
- 🧠 **Plant disease detection** using pretrained CNN+YOLO models
- 🛰️ **ESP32** sends sensor data to Flask via HTTP POST (`/upload-data`)
- 📬 **Telegram Bot** alerts for abnormal readings and disease detection
- 📊 Web Dashboard to view latest readings and predictions

---
---

## 🛠️ Components Used

- ESP32 Dev Board
- DHT22 Sensor (Temperature & Humidity)
- Soil Moisture Sensor (Analog)
- Raspberry Pi 4 (with Picamera2)
- Flask (Python backend)
- TensorFlow, OpenCV, YOLOv5 (for image classification)
- Telegram Bot API
- HTML (for dashboard)

--

---

## 🔌 How to Run

### 🧠 1. Model Training / Inference
- Train your CNN and YOLO models separately or use pre-trained weights.
- Save models to `/model/` and load in `app.py`.

### 🖥️ 2. Run Flask Server (on Raspberry Pi)

```bash
cd flask-server/
pip install -r requirements.txt
python app.py
