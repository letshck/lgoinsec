import cv2
import face_recognition
import pickle
import smtplib
import ssl
import os
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
# Step 1: Capture and process the image
def capture_image():
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Press 's' to capture image and save it
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite("captured_image.jpg", frame)
            print("Image saved as captured_image.jpg")
            break  # Exit after capturing

        # Press 'q' to exit without capturing
        elif key == ord('q'):
            print("Exiting without capturing.")
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Step 2: Check for unauthorized access
def check_user(image):
    # Load known face encodings
    try:
        with open("image_data.pkl", "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_names = data["names"]
    except FileNotFoundError:
        print("No face encodings file found. Please run the encoding generator script first.")
        return False
    
    # Find faces in the captured image
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        print("No face detected in the captured image")
        return False
    
    # Get encodings for faces in captured image
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Check each detected face against known faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        if any(matches):
            # Find name of matched face
            match_index = matches.index(True)
            name = known_names[match_index]
            print(f"Authorized user detected: {name}")
            return False  # Not unauthorized
    
    # If no matches found, it's an unauthorized access
    print("Unauthorized person detected!")
    return True  # Unauthorized - send alert
# Step 3: Send an email alert
def send_email(image_path):
    sender_email = "mahik4854@gmail.com"
    receiver_email = "ayushsingh20112004@gmail.com"
    password = "12345678"
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Unauthorized Access Detected"
    
    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename= {image_path}")
        message.attach(part)
    
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
# Main function
if __name__ == "__main__":
    frame=capture_image()
    if check_user(frame):
        send_email("captured_image.jpg")
    input("Press Enter to exit...")
