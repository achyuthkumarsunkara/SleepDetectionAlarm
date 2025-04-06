import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import pygame

# Load the alarm sound
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")  # Add your alarm sound file

# Eye Aspect Ratio (EAR) calculation function
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for sleep detection
EAR_THRESHOLD = 0.25  # Threshold for eye closure
FRAME_THRESHOLD = 75  # Number of consecutive frames with closed eyes before alarm (5 seconds)

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)  # Start video capture
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        cv2.putText(frame, "FACE NOT DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frame_count = 0
        pygame.mixer.music.stop()  # Stop the alarm if no face is detected
    else:
        for face_landmarks in results.multi_face_landmarks:
            cv2.putText(frame, "FACE DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Extract eye landmarks
            left_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [362, 385, 387, 263, 373, 380]]

            left_eye = np.array([(int(x * frame.shape[1]), int(y * frame.shape[0])) for x, y in left_eye])
            right_eye = np.array([(int(x * frame.shape[1]), int(y * frame.shape[0])) for x, y in right_eye])

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Draw eye contours
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

            if avg_ear < EAR_THRESHOLD:
                frame_count += 1
                if frame_count >= FRAME_THRESHOLD:
                    cv2.putText(frame, "SLEEPING! WAKE UP!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
            else:
                frame_count = 0
                pygame.mixer.music.stop()  # Stop the alarm if the driver is awake

    cv2.imshow("Sleep Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
