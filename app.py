from flask import Flask, render_template, request, jsonify
import face_recognition
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load a known face (you can replace this with a database later)
known_image = face_recognition.load_image_file("known.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    data_url = request.json['image']
    header, encoded = data_url.split(",", 1)
    image_data = base64.b64decode(encoded)

    # Decode image to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process with face_recognition
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    for face_encoding in face_encodings:
        match = face_recognition.compare_faces([known_encoding], face_encoding)
        if match[0]:
            return jsonify({"match": True})

    return jsonify({"match": False})

if __name__ == '__main__':
    app.run(debug=True)
