import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import third

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Define exercise functions here (detect_bicep_curl, detect_pushup, etc.)

def main():
    st.title("Exercise Detection System")
    st.write("Select an exercise to detect:")
    
    exercise_options = {
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
        # Add other exercises as needed
    }
    exercise_name = st.selectbox("Choose an exercise", list(exercise_options.keys()))
    detection_function = exercise_options[exercise_name]

    # Webcam access
    run_detection = st.checkbox("Start Detection")
    FRAME_WINDOW = st.image([])

    if run_detection:
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        counter = 0
        stage = "down"
        elapsed_time = 0
        global start_time
        start_time = time.time()

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to capture video")
                    break
                
                # Process frame for pose detection
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    if exercise_name == 'Tree Pose (Yoga)':
                        angle, elapsed_time, stage = detection_function(landmarks, elapsed_time, stage)
                    else:
                        angle, counter, stage, vis_point = detection_function(landmarks, counter, stage)

                    # Visual feedback on angle or counter
                    if exercise_name != 'Tree Pose (Yoga)':
                        cv2.putText(image, f'{angle:.1f}', tuple(np.multiply(vis_point, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Render pose landmarks on frame
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Update the Streamlit image container with the frame
                FRAME_WINDOW.image(image)

                # Exit the loop if user stops detection
                if not run_detection:
                    break

        cap.release()

if __name__ == "__main__":
    main()
