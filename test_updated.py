import cv2
import face_recognition
import pickle
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime
import time
import numpy as np

# Import the simplified liveness detection
from facial_liveness import check_liveness

# Step 1: Capture and process the image
def capture_image():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open camera.")
        return None
    
    # Give the camera a moment to adjust
    time.sleep(2)
    
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        video_capture.release()
        return None
    
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"logs/captured_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)
    
    video_capture.release()
    return frame, image_path

# Step 2: Check for unauthorized access
def check_user(frame):
    # Load known face encodings
    try:
        with open("image_data.pkl", "rb") as f:
            data_dict = pickle.load(f)
            
            # Check if data is a dictionary with the expected keys
            if not isinstance(data_dict, dict) or "encodings" not in data_dict or "names" not in data_dict:
                print("Invalid data format in image_data.pkl")
                return True, "Database error"
                
            known_face_encodings = data_dict["encodings"]
            known_names = data_dict["names"]
            
            print(f"Loaded {len(known_names)} known faces")
    except Exception as e:
        print(f"Error loading face encodings: {e}")
        return True, f"Database error: {str(e)}"
    
    # Find faces in the captured image
    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        print("No face detected in the captured image")
        return True, "No face detected"  # Consider unauthorized if no face
    
    # Get encodings for faces in captured image
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    # Check each detected face against known faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]
            print(f"Authorized user detected: {name}")
            return False, name


            # If no matches found, it's an unauthorized access
        print("Unauthorized person detected!")
        return True, "Unauthorized person"  # Unauthorized - send alert

# Step 3: Send an email alert
def send_email(image_path, person_info):
    sender_email = "ayushsingh20112004@gmail.com"
    receiver_email = "ayushsinghnov20112004@gmail.com"
    # Use app password instead of regular password
    app_password = "uemz qknt orim emgd"  # Generate this in your Google Account settings
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Security Alert: Unauthorized Access Detected"
    
    # Email body
    body = f"""
    Security Alert: {person_info}
    
    Date and Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    This is an automated security notification. 
    An unauthorized person was detected by your security system.
    Please see the attached image for details.
    """
    message.attach(MIMEText(body, "plain"))
    
    # Attach image
    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(image_path)}")
        message.attach(part)
    
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Security alert email sent successfully")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

#def main():
#    print("Starting facial recognition security system...")
#    
#    # First, perform liveness detection
#    # Something like:
#    frame = capture_image()  # or however you get your frame
#    liveness_result = check_liveness(frame)
#    
#
#    
#    if not liveness_result:
#        print("Liveness check failed. Access denied.")
#        # Optionally take a picture of the spoof attempt and send an alert
#        result = capture_image()
#        if result is not None:
#            frame, image_path = result
#            send_email(image_path, "Possible spoofing attempt detected")
#        return
#    
#    print("Liveness verified. Proceeding with facial recognition...")
#    
#    # Capture image for facial recognition
#    result = capture_image()
#    if result is None:
#        print("Failed to capture image.")
#        return
#    
#    frame, image_path = result
#    
#    # Check if unauthorized person
#    try:
#        is_unauthorized, person_info = check_user(frame)
#    except TypeError as e:
#        print(f"Error checking user: {e}")
#        is_unauthorized, person_info = True, "Error in facial recognition"
#    
#    # Take action based on authorization status
#    if is_unauthorized:
#        # If unauthorized, send email with the captured image
#        send_email(image_path, person_info)
#    else:
#        # If authorized, just log access
#        print(f"Authorized access by {person_info} - No alert sent")
#        # Optionally delete the image of authorized user
#        try:
#            os.remove(image_path)
#            print(f"Deleted image of authorized user")
#        except Exception as e:
#            print(f"Could not delete image: {e}")

def main():
    print("Starting facial recognition security system...")

    result = capture_image()
    if result is None:
        print("Failed to capture image.")
        return

    frame, image_path = result
    liveness_result = check_liveness(frame)

    if not liveness_result:
        print("Liveness check failed. Access denied.")
        send_email(image_path, "Possible spoofing attempt detected")
        return

    print("Liveness verified. Proceeding with facial recognition...")

    is_unauthorized, person_info = check_user(frame)

    if is_unauthorized:
        send_email(image_path, person_info)
    else:
        print(f"Authorized access by {person_info} - No alert sent")
        os.remove(image_path)


if __name__ == "__main__":
    main()