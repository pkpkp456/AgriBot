# === Your existing imports and code ===
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from libcamera import controls # Import controls for AWB modes
import time
import cv2
from ultralytics import YOLO
from datetime import datetime
import subprocess
import numpy as np
import serial
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, GlobalMaxPooling2D,
                                     BatchNormalization, Dense, Dropout, Concatenate, Input)
import pandas as pd
from tensorflow.keras.models import load_model
import os
import requests
import shutil


# --- NEW IMPORTS FOR FLASK ---
from flask import Flask, request, jsonify, render_template
import threading

# Replace these with your real bot token and chat ID
bot_token = '7677625616:AAEg84pnjLFVEi9_obND3-nVFAucfLSzs9w'
chat_id = '5073577271'

# -- Globals for sensor and prediction data to be shown on dashboard --
latest_sensor_data = {
    "temperature": None,
    "humidity": None,
    "soil_moisture": None,
}
latest_prediction_data = {
    "image_path": None,
    "predicted_class_label": None,
    "confidence": None,
    "timestamp": None,
}

def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    try:
        response = requests.post(url, data=payload)
        print("Telegram message status:", response.status_code)
        print("Telegram response:", response.text)
    except Exception as e:
        print("Error sending Telegram message:", e)

def send_telegram_photo(image_path, caption=""):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with open(image_path, 'rb') as photo:
        files = {'photo': photo}
        data = {'chat_id': chat_id, 'caption': caption}
        try:
            response = requests.post(url, files=files, data=data)
            print("ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‚Â¤ Telegram photo status:", response.status_code)
            print("ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‚Â¤ Telegram response:", response.text)
        except Exception as e:
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ Error sending Telegram photo:", e)

#..........ML setup...........##   
batch_size = 32
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
input_layer = Input(shape=img_shape)
resnet_weights = "/home/pi5/crop_grow/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

try:
    resnet_base = tf.keras.applications.ResNet50(include_top=False, weights=resnet_weights, input_shape=img_shape)
    print("Loaded ResNet50 weights from local path.")
except Exception as e:
    print(f"Failed to load ResNet50 weights locally. Error: {e}")
    resnet_base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=img_shape)

resnet_features = resnet_base(input_layer)
resnet_pool = GlobalMaxPooling2D()(resnet_features)

def build_cnn_model(input_layer):
    x = Conv2D(64, (3,3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    return x

cnn_features = build_cnn_model(input_layer)
merged = Concatenate()([resnet_pool, cnn_features])
class_count = 7
output_layer = Dense(class_count, activation='softmax')(merged)

model = load_model('/home/pi5/crop_grow/models/leaf_disease_ep10.keras')

def create_df(filepaths, labels):
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df

augmented_folder = "/home/pi5/crop_grow/LeafWorld"
filepaths = []
labels = []

# Modify predict_and_display to also return confidence and update globals
def predict_and_display(image_path, model, class_labels):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    confidence_score = float(prediction[0][predicted_class_index])
    predicted_class_label = class_labels[predicted_class_index]

    print(f"Predicted: {predicted_class_label} with confidence {confidence_score:.4f}")

    timestamp = datetime.now().strftime('%d-%b-%Y %H:%M:%S')
    caption = f"ÃƒÂ°Ã…Â¸Ã…â€™Ã‚Â± Leaf health result: {predicted_class_label}\n"
    caption += f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Confidence: {confidence_score:.2%}\n"
    caption += f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Â¢Ã¢â‚¬â„¢ {timestamp}\n"
    # Include sensor data if available
    if all(latest_sensor_data.values()):
        caption += (f"ÃƒÂ°Ã…Â¸Ã…â€™Ã‚Â¡ Temp: {latest_sensor_data['temperature']} Ãƒâ€šÃ‚Â°C\n"
                    f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â§ Humidity: {latest_sensor_data['humidity']} %\n"
                    f"ÃƒÂ°Ã…Â¸Ã…â€™Ã‚Â± Soil Moisture: {latest_sensor_data['soil_moisture']} %")

    send_telegram_photo(image_path, caption=caption)
    time.sleep(5)

    # Update latest prediction data for dashboard
    latest_prediction_data.update({
        "image_path": image_path,
        "predicted_class_label": predicted_class_label,
        "confidence": confidence_score,
        "timestamp": timestamp,
    })

    print("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Scan complete. Sending SCAN_DONE to Arduino.\n")
    ser.write(b"SCAN_DONE\n")

def leaf_check(img):
    # Gather filepaths and labels for training generator
    filepaths.clear()
    labels.clear()
    for disease in os.listdir(augmented_folder):
        disease_folder = os.path.join(augmented_folder, disease)
        if not os.path.isdir(disease_folder):
            continue
        for img_file in os.listdir(disease_folder):
            img_path = os.path.join(disease_folder, img_file)
            filepaths.append(img_path)
            labels.append(disease)
    df = create_df(filepaths, labels)
    train_df, _ = train_test_split(df, train_size=0.7, shuffle=True, random_state=123)

    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(preprocessing_function=scalar,
                               rotation_range=40,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               brightness_range=[0.4,0.6],
                               zoom_range=0.3,
                               horizontal_flip=True,
                               vertical_flip=True)

    train_gen = tr_gen.flow_from_dataframe(train_df,
                                           x_col='filepaths',
                                           y_col='labels',
                                           target_size=img_size,
                                           class_mode='categorical',
                                           color_mode='rgb',
                                           shuffle=True,
                                           batch_size=batch_size)

    class_labels = list(train_gen.class_indices.keys())

    print(img)
    predict_and_display(img, model, class_labels)
    return 0

# ---- Servo Setup ----
PAN_PIN = 17
TILT_PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan = GPIO.PWM(PAN_PIN, 50)
tilt = GPIO.PWM(TILT_PIN, 50)
pan.start(7.5)
tilt.start(7.5)
time.sleep(1)

def angle_to_duty(angle):
    return 2.5 + (angle / 18.0)

def smooth_move_servo(servo, current_angle, target_angle, step=2, delay=0.02):
    if current_angle == target_angle:
        return target_angle
    step = abs(step) if target_angle > current_angle else -abs(step)
    angle = current_angle
    while (angle < target_angle and step > 0) or (angle > target_angle and step < 0):
        servo.ChangeDutyCycle(angle_to_duty(angle))
        time.sleep(delay)
        angle += step
    servo.ChangeDutyCycle(angle_to_duty(target_angle))
    return target_angle

def move_pan_to(angle):
    global pan_angle
    pan_angle = smooth_move_servo(pan, pan_angle, angle)

def move_tilt_to(angle):
    global tilt_angle
    tilt_angle = smooth_move_servo(tilt, tilt_angle, angle)

def pan_full_scan():
    global pan_angle, tilt_angle
    for angle in range(0, 181, 30):
        move_pan_to(angle)
        time.sleep(0.5)
    for angle in range(180, -1, -30):
        move_pan_to(angle)
        time.sleep(0.5)

def capture_image(picamera, path="/home/pi5/crop_grow/leaf.jpg"):
    picamera.capture_file(path)
    print(f"Captured image saved to {path}")
    return path


# Initialize pan and tilt angles
pan_angle = 90
tilt_angle = 90

# Initialize picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)}))
picam2.start()

def plant_scan():
    global pan_angle, tilt_angle
    print("Setting pan and tilt to fixed position")
    move_pan_to(60)
    move_tilt_to(65)
    time.sleep(1)  # small delay to settle
    image_path = capture_image(picam2)
    shutil.copy(image_path, os.path.join('static', 'leaf.jpg'))
    return image_path

# Serial communication setup
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

# Main loop variables
running = True

def main_loop():
    global running
    while running:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            print("Received from Arduino:", line)
            if line == "OBSTACLE":
                print("Obstacle detected. Scanning plant...")
                image_path = plant_scan()
                leaf_check(image_path)
        time.sleep(0.1)

# ======== NEW FLASK APP CODE =======

app = Flask(__name__)

# Thread-safe lock for updating globals
data_lock = threading.Lock()

@app.route('/upload-data', methods=['POST'])
def upload_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        soil_moisture = data.get('soil_moisture')

        with data_lock:
            latest_sensor_data['temperature'] = temperature
            latest_sensor_data['humidity'] = humidity
            latest_sensor_data['soil_moisture'] = soil_moisture

        print(f"Updated sensor data: Temp={temperature}, Humidity={humidity}, Soil Moisture={soil_moisture}")

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print("Error processing /upload-data:", e)
        return jsonify({"error": "Failed to process data"}), 500

@app.route('/')
def dashboard():
    with data_lock:
        temp = latest_sensor_data.get('temperature', 'N/A')
        humidity = latest_sensor_data.get('humidity', 'N/A')
        soil_moisture = latest_sensor_data.get('soil_moisture', 'N/A')

        pred_label = latest_prediction_data.get('predicted_class_label', 'No data')
        confidence = latest_prediction_data.get('confidence')
        confidence_pct = f"{confidence:.2%}" if confidence else 'N/A'
        timestamp = latest_prediction_data.get('timestamp', 'N/A')
        image_path = latest_prediction_data.get('image_path')

    # Pass the image file path relative to a static folder or absolute path (adjust accordingly)
    image_url = None
    if image_path and os.path.exists(image_path):
        # For Flask to serve image, we need it in static folder or use send_file.
        # Here we just pass the file path; user can modify accordingly.
        image_url = '/static/leaf.jpg'  # Suggest you copy leaf.jpg to static folder after capture

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Plant Disease Detection Dashboard</title>
</head>
<body>
  <h1>Plant Disease Detection Dashboard</h1>

  <h3>Sensor Data</h3>
  <ul>
    <li>Temperature: {temp}</li>
    <li>Humidity: {humidity} %</li>
    <li>Soil Moisture: {soil_moisture} %</li>
  </ul>

  <h3>Last Prediction</h3>
  <p>Disease: {pred_label}</p>
  <p>Confidence: {confidence_pct}</p>
  <p>Timestamp: {timestamp}</p>

  <h3>Last Captured Image</h3>
  {image_url
    ? `<img src="{image_url}" alt="Last Leaf Image" style="max-width:100%;height:auto;">`
    : `<p>No image available</p>`
  }
</body>
</html>
"""

def flask_thread():
    app.run(host='0.0.0.0', port=5000)

# ==== Start Flask server in a thread ====
flask_app_thread = threading.Thread(target=flask_thread, daemon=True)
flask_app_thread.start()

# ==== Start main loop ====
try:
    main_loop()
except KeyboardInterrupt:
    print("Exiting program...")
finally:
    running = False
    try:
        pan.stop()
        tilt.stop()
    except Exception as e:
        print("Error stopping PWM:", e)
    GPIO.cleanup()
    pan = None
    tilt = None
    print("Cleaned up GPIO and exiting.")
    # --- Start Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
