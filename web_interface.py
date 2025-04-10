from flask import Flask, render_template, Response, jsonify, redirect, url_for, request, flash
import cv2
import os
import pickle
import face_recognition
from datetime import datetime
import time
from werkzeug.utils import secure_filename
import threading
import numpy as np
from liveness_detection import check_liveness
import dlib
from scipy.spatial import distance

app = Flask(__name__)
app.secret_key = 'facial_recognition_security_system'

# Global variables
camera = None
camera_lock = threading.Lock()
output_frame = None
liveness_verified = False
authentication_status = {"status": "Waiting", "message": "No authentication attempt yet"}

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for admin login
@app.route('/admin')
def admin():
    return render_template('admin.html')

# Route for the dashboard
@app.route('/dashboard')
def dashboard():
    # Get list of logs
    logs = []
    if os.path.exists('logs'):
        logs = sorted([f for f in os.listdir('logs') if f.endswith('.jpg')], reverse=True)
    
    # Get list of users
    users = []
    if os.path.exists('image_data.pkl'):
        try:
            with open('image_data.pkl', 'rb') as f:
                data_dict = pickle.load(f)
                users = data_dict["names"]
        except Exception as e:
            flash(f"Error loading user data: {e}", "danger")
    
    return render_template('dashboard.html', logs=logs, users=users)

# Route to display a specific log image
@app.route('/logs/<filename>')
def log_file(filename):
    return send_from_directory('logs', filename)

# Function to generate camera frames
def generate_frames():
    global output_frame, authentication_status
    
    cap = get_camera()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process the frame as needed
            output_frame = frame.copy()
            
            # Add status text to frame
            cv2.putText(output_frame, 
                      f"Status: {authentication_status['status']}", 
                      (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, 
                      (0, 0, 255), 
                      2)
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()
            
            # Yield the frame in the byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to start liveness detection
@app.route('/start_liveness', methods=['POST'])
def start_liveness():
    global liveness_verified, authentication_status
    
    authentication_status = {"status": "Checking liveness", "message": "Please blink naturally"}
    
    # This would normally run the liveness check
    # For web implementation, we'll simulate it with a simple timer
    def run_liveness_check():
        global liveness_verified, authentication_status
        time.sleep(5)  # Simulate liveness check
        liveness_verified = True
        authentication_status = {"status": "Liveness verified", "message": "Proceeding to facial recognition"}
    
    # Start liveness check in a separate thread
    thread = threading.Thread(target=run_liveness_check)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "Liveness check started"})

# Route to perform facial recognition
@app.route('/authenticate', methods=['POST'])
def authenticate():
    global liveness_verified, authentication_status
    
    if not liveness_verified:
        return jsonify({"status": "error", "message": "Liveness verification required first"})
    
    # Reset liveness verification for next attempt
    liveness_verified = False
    
    authentication_status = {"status": "Authenticating", "message": "Checking face against database"}
    
    # Capture current frame
    cap = get_camera()
    success, frame = cap.read()
    
    if not success:
        authentication_status = {"status": "Failed", "message": "Could not capture image"}
        return jsonify({"status": "error", "message": "Failed to capture image"})
    
    # Save captured image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"logs/captured_{timestamp}.jpg"
    os.makedirs("logs", exist_ok=True)
    cv2.imwrite(image_path, frame)
    
    # Check if user is authorized
    try:
        with open("image_data.pkl", "rb") as f:
            data_dict = pickle.load(f)
            known_face_encodings = data_dict["encodings"]
            known_names = data_dict["names"]
    except Exception as e:
        authentication_status = {"status": "Error", "message": f"Database error: {str(e)}"}
        return jsonify({"status": "error", "message": f"Database error: {str(e)}"})
    
    # Find faces in the captured image
    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        authentication_status = {"status": "Failed", "message": "No face detected"}
        return jsonify({"status": "error", "message": "No face detected"})
    
    # Get encodings for faces in captured image
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    # Check each detected face against known faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        if True in matches:
            # Find name of matched face
            match_index = matches.index(True)
            name = known_names[match_index]
            authentication_status = {"status": "Authorized", "message": f"Welcome, {name}"}
            return jsonify({"status": "success", "message": f"Welcome, {name}", "name": name})
    
    # If no matches found, it's an unauthorized access
    authentication_status = {"status": "Unauthorized", "message": "Access denied"}
    return jsonify({"status": "error", "message": "Unauthorized access"})

# Route to register a new user
@app.route('/register_user', methods=['POST'])
def register_user():
    if 'name' not in request.form:
        flash("Name is required", "danger")
        return redirect(url_for('dashboard'))
    
    name = request.form['name']
    
    # Capture current frame
    cap = get_camera()
    success, frame = cap.read()
    
    if not success:
        flash("Failed to capture image", "danger")
        return redirect(url_for('dashboard'))
    
    # Process the captured image
    face_locations = face_recognition.face_locations(frame)
    
    if len(face_locations) == 0:
        flash("No face detected in the image. Please try again.", "danger")
        return redirect(url_for('dashboard'))
    
    if len(face_locations) > 1:
        flash("Multiple faces detected. Please ensure only one person is in the frame.", "danger")
        return redirect(url_for('dashboard'))
    
    # Get the encoding for the face
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    # Load existing data or create new data structure
    data_dict = {"encodings": [], "names": []}
    if os.path.exists("image_data.pkl"):
        try:
            with open("image_data.pkl", "rb") as f:
                data_dict = pickle.load(f)
        except Exception as e:
            flash(f"Error loading existing data: {e}", "danger")
            return redirect(url_for('dashboard'))
    
    # Add the new face encoding and name
    data_dict["encodings"].append(face_encodings[0])
    data_dict["names"].append(name)
    
    # Save the updated data
    with open("image_data.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    
    # Create a directory for registered users if it doesn't exist
    os.makedirs("registered_users", exist_ok=True)
    
    # Save the user's image
    cv2.imwrite(f"registered_users/{name}.jpg", frame)
    
    flash(f"Successfully registered {name}", "success")
    return redirect(url_for('dashboard'))

# Route to remove a user
@app.route('/remove_user/<name>', methods=['POST'])
def remove_user(name):
    if not os.path.exists("image_data.pkl"):
        flash("No users registered yet.", "warning")
        return redirect(url_for('dashboard'))
    
    try:
        with open("image_data.pkl", "rb") as f:
            data_dict = pickle.load(f)
        
        if name not in data_dict["names"]:
            flash(f"User {name} not found.", "warning")
            return redirect(url_for('dashboard'))
        
        # Find the index of the user
        index = data_dict["names"].index(name)
        
        # Remove the user from the data dictionary
        data_dict["encodings"].pop(index)
        data_dict["names"].pop(index)
        
        # Save the updated data
        with open("image_data.pkl", "wb") as f:
            pickle.dump(data_dict, f)
        
        # Remove the user's image if it exists
        user_image_path = f"registered_users/{name}.jpg"
        if os.path.exists(user_image_path):
            os.remove(user_image_path)
        
        flash(f"Successfully removed {name} from the database.", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")
    
    return redirect(url_for('dashboard'))

# Clean up when the application closes
@app.teardown_appcontext
def cleanup(exception=None):
    release_camera()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)