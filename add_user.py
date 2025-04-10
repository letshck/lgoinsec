import cv2
import face_recognition
import pickle
import os
import argparse

# Create parser for command-line arguments
parser = argparse.ArgumentParser(description="Add a new user's face to the database")
parser.add_argument('--name', type=str, required=True, help='Full name of the user')
args = parser.parse_args()

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible")
        return None

    print("Please look at the camera. Capturing image in 3 seconds...")
    cv2.waitKey(3000)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image")
        return None

    return frame

def encode_face(frame):
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) != 1:
        print(f"Expected one face, but found {len(face_locations)}")
        return None

    encodings = face_recognition.face_encodings(frame, face_locations)
    return encodings[0]

def save_encoding(name, encoding):
    file_path = "image_data.pkl"

    # If the file exists, load it
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    else:
        data = {"encodings": [], "names": []}

    data["encodings"].append(encoding)
    data["names"].append(name)

    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    print(f"User '{name}' added successfully!")

def main():
    frame = capture_image()
    if frame is None:
        return

    encoding = encode_face(frame)
    if encoding is None:
        return

    save_encoding(args.name, encoding)

if __name__ == "__main__":
    main()
