Driver Drowsiness Detection System 🚗👁️
Python
OpenCV
MediaPipe
Flask

A real-time driver drowsiness detection system that uses computer vision and facial landmarks to monitor alertness and prevent accidents caused by fatigue.

Prerequisites ⚠️
Important: This project requires Python 3.10 as MediaPipe has compatibility requirements. Other versions may not work properly.

Features ✨
👁 Eye Aspect Ratio (EAR) calculation for precise drowsiness detection

🚨 Audio-visual alerts when drowsiness is detected

🌍 Location tracking with GPS coordinates and address

📊 Web dashboard with live camera feed and status monitoring

⚡ Real-time processing at ~30 FPS using MediaPipe

🔔 Configurable thresholds for sensitivity adjustment

Installation ⚙️
Ensure you have Python 3.10 installed:

bash
Copy
python --version
If you need to install Python 3.10:

Windows/Mac

Linux (Ubuntu/Debian):

bash
Copy
sudo apt update
sudo apt install python3.10 python3.10-venv
Clone the repository:

bash
Copy
git clone https://github.com/achyuthkumarsunkara/SleepDetectionAlarm/tree/main
cd driver-drowsiness-detection
Create and activate a virtual environment (recommended):

bash
Copy
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
Install dependencies:

bash
Copy
pip install -r requirements.txt
Download the alarm sound file (optional):

bash
Copy
wget -O alarm.wav
Or place your own alarm.wav file in the project directory

Usage 🚀
Run the application:

bash
Copy
sleep_detection_vehicle.py
Open the web interface in your browser:

Copy
http://localhost:5000
Python Version Compatibility
This project specifically requires Python 3.10 because:

MediaPipe has known compatibility issues with other Python versions

Python 3.10 provides the optimal balance of stability and features needed

Some dependencies may not work correctly with newer Python versions

If you encounter installation issues:

Verify your Python version is exactly 3.10.x

Create a fresh virtual environment with Python 3.10

Reinstall all dependencies

Troubleshooting
If you get MediaPipe installation errors:

bash
Copy
# Try specifying the exact version
pip install mediapipe==0.10.0


