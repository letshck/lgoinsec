import cv2
import numpy as np
import dlib
from scipy.spatial import distance

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect blinks for liveness detection
def detect_blinks(frame, face_landmarks):
    # Get the indexes of the facial landmarks for the left and right eye
    (lStart, lEnd) = (42, 48)  # Left eye landmarks
    (rStart, rEnd) = (36, 42)  # Right eye landmarks
    
    # Extract the left and right eye coordinates
    leftEye = face_landmarks[lStart:lEnd]
    rightEye = face_landmarks[rStart:rEnd]
    
    # Calculate the eye aspect ratios
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    
    # Average the eye aspect ratio for both eyes
    ear = (leftEAR + rightEAR) / 2.0
    
    # Check if the eye aspect ratio is below the blink threshold
    if ear < 0.2:  # Threshold for blink detection
        return True
    return False

# Main liveness detection function
def check_liveness(frame, num_frames=50, required_blinks=2):
    # Initialize dlib's face detector and facial landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    blink_counter = 0
    frames_counter = 0
    
    print("Liveness detection started. Please blink naturally...")
    
    # Capture video feed for liveness detection
    video_capture = cv2.VideoCapture(0)
    
    # Process frames for liveness detection
    while frames_counter < num_frames:
        # Read a frame
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = detector(gray, 0)
        
        # Process each face
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)
            landmarks_points = []
            
            # Convert landmarks to a numpy array
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))
            
            # Check for blinks
            if detect_blinks(frame, landmarks_points):
                blink_counter += 1
                print(f"Blink detected! Count: {blink_counter}")
        
        # Display instructions and progress
        text = f"Blinks: {blink_counter}/{required_blinks} | Frame: {frames_counter}/{num_frames}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Liveness Detection', frame)
        
        # Increment frame counter
        frames_counter += 1
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()
    
    # Check if enough blinks were detected
    if blink_counter >= required_blinks:
        print("Liveness verification successful!")
        return True
    else:
        print("Liveness verification failed. Please try again.")
        return False

# Additional movement-based liveness detection
def detect_head_movement(frame, previous_landmarks, current_landmarks, threshold=10):
    if previous_landmarks is None or current_landmarks is None:
        return False
    
    # Calculate the average movement of all landmarks
    total_movement = 0
    for i in range(len(previous_landmarks)):
        movement = distance.euclidean(previous_landmarks[i], current_landmarks[i])
        total_movement += movement
    
    avg_movement = total_movement / len(previous_landmarks)
    
    # Check if the movement exceeds the threshold
    if avg_movement > threshold:
        return True
    return False