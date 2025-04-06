import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import pygame
from flask import Flask, render_template_string, Response, jsonify
import threading
import time
import requests
from geopy.geocoders import Nominatim
import geocoder

app = Flask(__name__)

# Audio initialization with error handling
pygame.mixer.init()
try:
    pygame.mixer.music.load("alarm.wav")
    alarm_sound_loaded = True
except:
    print("Could not load alarm sound, using system beep")
    alarm_sound_loaded = False

# Drowsiness detection parameters
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
EAR_CONSEC_FRAMES = 30  # Number of consecutive frames below threshold to trigger alert
NO_FACE_THRESHOLD = 60  # Frames without face detection to trigger alert

# MediaPipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

# Landmark indices for left and right eyes
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Global variables with thread safety
frame = None
frame_count = 0
no_face_count = 0
status = "Monitoring Active"
alert_active = False
alert_frame = None
system_running = True
data_lock = threading.Lock()

def eye_aspect_ratio(eye_landmarks):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def play_alert():
    if alarm_sound_loaded:
        try:
            pygame.mixer.music.play(-1)  # Loop indefinitely
        except:
            print("Error playing alarm sound")

def stop_alert():
    if alarm_sound_loaded:
        try:
            pygame.mixer.music.stop()
        except:
            print("Error stopping alarm sound")

def drowsiness_detection():
    global frame, frame_count, no_face_count, status, alert_active, alert_frame
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while system_running:
        ret, frame = cap.read()
        if not ret:
            print("Camera error - reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(0)
            continue
        
        # Convert to RGB and process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        with data_lock:
            if results.multi_face_landmarks:
                no_face_count = 0  # Reset no face counter
                
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Get eye landmarks
                left_eye = [(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE_INDICES]
                right_eye = [(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE_INDICES]
                
                # Calculate EAR for both eyes
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Check for drowsiness
                if avg_ear < EAR_THRESHOLD:
                    frame_count += 1
                    if frame_count >= EAR_CONSEC_FRAMES and not alert_active:
                        status = "DROWSINESS DETECTED!"
                        alert_active = True
                        play_alert()
                else:
                    if frame_count > 0:
                        frame_count -= 1  # Slowly decay the counter for more robust detection

                    if frame_count == 0 and alert_active:
                        alert_active = False
                        stop_alert()
                        status = "Monitoring Active"

                
                # Draw eye landmarks and EAR value
                for eye in [left_eye, right_eye]:
                    for point in eye:
                        x = int(point[0] * frame.shape[1])
                        y = int(point[1] * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                no_face_count += 1
                if no_face_count >= NO_FACE_THRESHOLD:
                    status = "FACE NOT DETECTED!"
                    if not alert_active:
                        alert_active = True
                        play_alert()
                else:
                    status = "Searching for face..."
            
            # Add status text to frame
            color = (0, 0, 255) if alert_active else (0, 255, 0)
            cv2.putText(frame, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add red border if alert is active
            if alert_active:
                cv2.rectangle(frame, (0, 0), 
                              (frame.shape[1]-1, frame.shape[0]-1),
                              (0, 0, 255), 10)
            
            alert_frame = frame.copy()
        
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()

@app.route('/')
def dashboard():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Driver Monitoring System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyAb6G89VLpIblzAeuDsOHSp9Z9QLd3XXpc"></script>
        <style>
            body { 
                background-color: #f8f9fa; 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                padding-top: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .card {
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                border: none;
            }
            .status {
                font-size: 20px;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .alert-danger {
                animation: blink 1s infinite;
                background-color: #ff4444;
                color: white;
            }
            .alert-success {
                background-color: #00C851;
                color: white;
            }
            .alert-warning {
                background-color: #ffbb33;
                color: white;
            }
            @keyframes blink {
                50% { opacity: 0.7; }
            }
            #map {
                height: 300px;
                border-radius: 10px;
                margin-top: 10px;
            }
            .video-container {
                position: relative;
                margin-bottom: 20px;
            }
            .video-overlay {
                position: absolute;
                top: 10px;
                left: 10px;
                background-color: rgba(0,0,0,0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            .stat-card {
                text-align: center;
                padding: 15px;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .stat-label {
                font-size: 14px;
                color: #6c757d;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row mb-4">
                <div class="col-12">
                    <h2 class="text-center">🚗 Driver Monitoring Dashboard</h2>
                </div>
            </div>

            <div class="row">
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Live Camera Feed</h5>
                            <div class="video-container">
                                <img src="{{ url_for('video_feed') }}" class="img-fluid rounded">
                                <div class="video-overlay" id="videoStatus">Loading...</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Driver Status</h5>
                            <div class="status alert" id="statusText">Loading...</div>
                            
                            <div class="row mt-3">
                                <div class="col-6">
                                    <div class="stat-card">
                                        <div class="stat-value" id="earValue">0.00</div>
                                        <div class="stat-label">Eye Aspect Ratio</div>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="stat-card">
                                        <div class="stat-value" id="speedValue">0</div>
                                        <div class="stat-label">km/h</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row mt-3">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Location Information</h5>
                            <p><strong>Address:</strong> <span id="locationText">Acquiring...</span></p>
                            <p><strong>Coordinates:</strong> <span id="coordinatesText">0, 0</span></p>
                            <p><strong>IP Address:</strong> <span id="ipText">127.0.0.1</span></p>
                            <div id="map"></div>
                        </div>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">System Information</h5>
                            <div class="alert alert-info">
                                <strong>Detection Thresholds:</strong>
                                <ul class="mt-2">
                                    <li>EAR Threshold: {{EAR_THRESHOLD}}</li>
                                    <li>Closed Eye Frames: {{CLOSED_EYE_FRAME_THRESHOLD}}</li>
                                    <li>Face Missing Frames: {{FACE_MISSING_THRESHOLD}}</li>
                                </ul>
                            </div>
                            <div class="alert alert-secondary">
                                <strong>Instructions:</strong>
                                <ul class="mt-2">
                                    <li>Keep your face visible to the camera</li>
                                    <li>System will alert if drowsiness detected</li>
                                    <li>Alerts will stop when you become alert</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let map, marker;
            let alertAudio = new Audio('alarm.wav');
            let isAlertPlaying = false;

            function initMap(lat = 12.9716, lng = 77.5946) {
                map = new google.maps.Map(document.getElementById("map"), {
                    center: { lat, lng },
                    zoom: 15,
                });
                marker = new google.maps.Marker({ position: { lat, lng }, map: map });
            }

            function playAlertSound() {
                if (!isAlertPlaying) {
                    alertAudio.loop = true;
                    alertAudio.play().then(() => {
                        isAlertPlaying = true;
                    }).catch(e => console.error("Audio play failed:", e));
                }
            }

            function stopAlertSound() {
                if (isAlertPlaying) {
                    alertAudio.pause();
                    alertAudio.currentTime = 0;
                    isAlertPlaying = false;
                }
            }

            function updateData() {
                fetch('/api/status').then(res => res.json()).then(data => {
                    const status = document.getElementById("statusText");
                    status.textContent = data.status;
                    
                    // Update status color based on alert state
                    if (data.drowsy || data.face_missing) {
                        status.className = "status alert alert-danger";
                        playAlertSound();
                    } else {
                        status.className = "status alert alert-success";
                        stopAlertSound();
                    }
                    
                    // Update video overlay
                    document.getElementById("videoStatus").textContent = data.status;
                });

                fetch('/api/location').then(res => res.json()).then(data => {
                    document.getElementById("locationText").textContent = data.address;
                    document.getElementById("coordinatesText").textContent = 
                        `${data.latitude.toFixed(6)}, ${data.longitude.toFixed(6)}`;
                    document.getElementById("ipText").textContent = data.ip_address;
                    document.getElementById("speedValue").textContent = data.speed.toFixed(1);
                    
                    if (map && marker) {
                        let newPos = new google.maps.LatLng(data.latitude, data.longitude);
                        map.setCenter(newPos);
                        marker.setPosition(newPos);
                    }
                });

                // Update EAR value (simulated in this example)
                fetch('/api/status').then(res => res.json()).then(data => {
                    // In a real implementation, you would get the actual EAR value from the API
                    document.getElementById("earValue").textContent = data.drowsy ? "0.20" : "0.30";
                });
            }

            function requestLocationPermission() {
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(
                        (position) => {
                            const { latitude, longitude } = position.coords;
                            initMap(latitude, longitude);
                        },
                        (error) => {
                            console.error("Error getting location:", error);
                            initMap();
                        }
                    );
                } else {
                    console.error("Geolocation is not supported by this browser.");
                    initMap();
                }
            }

            window.onload = () => {
                requestLocationPermission();
                setInterval(updateData, 1000); // Update every second
            };
        </script>
    </body>
    </html>
    ''', ear_threshold=EAR_THRESHOLD,
        ear_frames=EAR_CONSEC_FRAMES,
        no_face_threshold=NO_FACE_THRESHOLD)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with data_lock:
                if alert_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', alert_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with data_lock:
        return jsonify({
            'status': status,
            'alert_active': alert_active,
            'drowsy': alert_active and status == "DROWSINESS DETECTED!",
            'face_missing': alert_active and status == "FACE NOT DETECTED!",
            'ear': 0.20 if alert_active else 0.30  # For now, you can use this placeholder
        })

@app.route('/api/location')
def api_location():
    try:
        g = geocoder.ip('me')
        lat, lng = g.latlng

        geolocator = Nominatim(user_agent="dms")
        location = geolocator.reverse((lat, lng), timeout=10)

        return jsonify({
            'latitude': lat,
            'longitude': lng,
            'address': location.address if location else "Unknown",
            'ip_address': g.ip,
            'speed': 0  # Placeholder for GPS speed if needed
        })
    except Exception as e:
        print("Location error:", e)
        return jsonify({
            'latitude': 0,
            'longitude': 0,
            'address': "Unable to fetch",
            'ip_address': "127.0.0.1",
            'speed': 0
        })


if __name__ == '__main__':
    # Make sure the alarm.wav file is accessible via Flask
    @app.route('/alarm.wav')
    def serve_alarm():
        return app.send_static_file('alarm.wav')
    
    detection_thread = threading.Thread(target=drowsiness_detection)
    detection_thread.daemon = True
    detection_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)