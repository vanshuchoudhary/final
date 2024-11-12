import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def detect_bicep_curl(landmarks, counter, stage):
    """Detect bicep curls and count reps"""
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    
    angle = calculate_angle(shoulder, elbow, wrist)
    
    if angle > 160:
        stage = "down"
    elif angle < 30 and stage == 'down':
        stage = "up"
        counter += 1
        
    return angle, counter, stage, elbow

def detect_pushup(landmarks, counter, stage):
    """Detect pushups and count reps"""
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
    angle = calculate_angle(shoulder, elbow, wrist)
    
    if angle > 160:
        stage = "up"
    elif angle < 90 and stage == 'up':
        stage = "down"
        counter += 1
        
    return angle, counter, stage, elbow

def detect_tree_pose(landmarks, elapsed_time, stage):
    """Detect Tree Pose (Vrksasana)"""
    global start_time
    current_time = time.time()  # Get current time

    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    angle = calculate_angle(hip, knee, ankle)
    
    if angle < 80:  # Foot should be placed on inner thigh
        if stage != "detected":
            stage = "detected"
            start_time = current_time  # Reset start time when pose is detected
        elapsed_time = min(elapsed_time + (current_time - start_time), 720)  # Update elapsed time
    else:
        stage = "not detected"
        start_time = current_time  # Only reset the start time when pose is not detected

    return angle, elapsed_time, stage

def detect_jumping(landmarks, counter, stage):
    """Detect jumping/skipping motion"""
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    
    if hip < 0.7:  # Person is in the air
        if stage == "down":
            counter += 1
            stage = "up"
    elif hip > 0.7:  # Person is on ground
        stage = "down"
        
    return hip, counter, stage, [0.5, hip]  # Return hip position for visualization


def detect_squat(landmarks, counter, stage):
    """Detect squats and count reps"""
    # Points for calculating the squat angle
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    # Calculate the angle at the knee
    angle = calculate_angle(hip, knee, ankle)
    
    # Check the stage and count a rep when squatting
    if angle > 160:
        stage = "up"
    elif angle < 90 and stage == 'up':
        stage = "down"
        counter += 1
        
    return angle, counter, stage, knee  # Returning knee for visualization

# def detect_plank(landmarks, elapsed_time, stage):
#     """Detect plank hold and measure time held"""
#     global start_time
#     current_time = time.time()
    
#     shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
#     hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
#     ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    
#     # Check if the plank is being held correctly
#     if abs(shoulder - hip) < 0.1 and abs(hip - ankle) < 0.1:
#         # Start the timer if it's not running
#         if stage != "holding":
#             stage = "holding"
#             start_time = current_time  # Set the start time when the correct form is first detected
#         # Increment elapsed time
#         elapsed_time += current_time - start_time
#         start_time = current_time  # Update start time to current for continuous counting
#     else:
#         stage = "not holding"
#         start_time = current_time  # Reset start time when form is broken
    
#     return elapsed_time, stage




def main():
    print("\nWelcome to Exercise Detection System!")
    print("Choose an exercise to detect:")
    print("1) Bicep Curls")
    print("2) Pushups")
    print("3) Tree Pose (Yoga)")
    print("4) Jumping/Skipping")
    print("5) squats")
    # print("6) plank hold")
    
    choice = input("\nEnter your choice (1-6): ")
    
    exercise_functions = {
        '1': ('Bicep Curls', detect_bicep_curl),
        '2': ('Pushups', detect_pushup),
        '3': ('Tree Pose', detect_tree_pose),
        '4': ('Jumping/Skipping', detect_jumping),
        '5': ('Squats', detect_squat),
        # '6': ('Plank Hold', detect_plank)
    }
    
    if choice not in exercise_functions:
        print("Invalid choice!")
        return
        
    exercise_name, detection_function = exercise_functions[choice]
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize counter variables
    counter = 0 
    stage = "down"
    
    elapsed_time = 0  # Initialize elapsed time
    global start_time  # Initialize start time for tree pose detection
    start_time = time.time()

    # Setup MediaPipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Detect exercise and update counter or timer
                if choice == '3' :  # For Tree Pose
                    angle, elapsed_time, stage = detection_function(landmarks, elapsed_time, stage)
                    elapsed_time = min(elapsed_time, 720)  # Cap the timer at 30 seconds
                else:
                    angle, counter, stage, vis_point = detection_function(landmarks, counter, stage)
                
                # Visualize angle for exercises other than Tree Pose
                if choice != '3':
                    cv2.putText(image, f'{angle:.1f}', 
                                tuple(np.multiply(vis_point, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
            except:
                pass
            
            # Draw status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Display exercise name
            cv2.putText(image, exercise_name, (10,25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
            
            # Display counter or timer
            if choice == '3':  # For Tree Pose
                cv2.putText(image, f'Time: {int(elapsed_time)} frames', 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                if elapsed_time >= 720:
                    cv2.putText(image, '30 Seconds Completed', 
                                (10,100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            else:  # For other exercises
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
            
        
                
            
                

                
            # Display stage
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render pose detection
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Display
            cv2.imshow('Exercise Detection', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
