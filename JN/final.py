import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
from PIL import Image
import third

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Exercise detection functions here (e.g., detect_bicep_curl, detect_pushup, detect_tree_pose, etc.)
# You can copy all detection functions from the initial code here.

# Initialize Streamlit interface
st.title("Exercise Detection System")
st.write("Select an exercise from the sidebar and start the webcam to detect and track your exercise form.")

# Sidebar selection
exercise_options = [
    "Bicep Curls", "Pushups", "Tree Pose", "Jumping/Skipping", 
    "Squats", "Plank", "Lunges", "Burpees", "Sit-ups", "Side Plank"
]
exercise_name = st.sidebar.selectbox("Choose an exercise:", exercise_options)

# Initialize variables for exercise tracking
counter = 0
stage = "down"
elapsed_time = 0
start_time = time.time()

# Start webcam feed
run = st.checkbox("Start Webcam")
cap = None 

if run:
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])  # Placeholder for Streamlit image display

    # Exercise mapping to detection function
    exercise_functions = {
        'Bicep Curls': third.detect_bicep_curl,
        'Pushups': third.detect_pushup,
        'Tree Pose (Yoga)': third.detect_tree_pose,
        'Jumping/Skipping':third.detect_jumping,
        'Squats': third.detect_squat,
        'Plank': third.detect_plank,
        'Lunges':third.detect_lunge,
        'Burpees':third.detect_burpee,
        'sit-ups':third.detect_sit_up,
        'side_plank': third.detect_side_plank,
    }
    
    # Define the selected exercise function
    detection_function = exercise_functions[exercise_name]

    # MediaPipe pose setup
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame.")
                break

            # Flip and process the frame
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                if exercise_name == "Tree Pose" or exercise_name == "Plank" or exercise_name == "Side Plank":
                    angle, elapsed_time, stage = detection_function(landmarks, elapsed_time, stage)
                elif exercise_name == "Burpees" or exercise_name == "Sit-ups":
                    counter, stage, _ = detection_function(landmarks, counter, stage)
                else:
                    angle, counter, stage, vis_point = detection_function(landmarks, counter, stage)
                    cv2.putText(image, f'{angle:.1f}', 
                                tuple(np.multiply(vis_point, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Draw landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Convert image color back to BGR for display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            FRAME_WINDOW.image(image)

cap.release()
st.write("Webcam closed.")
