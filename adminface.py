import cv2
import face_recognition
import pickle
import os

def create_encodings():
    # Create directory for admin images if it doesn't exist
    if not os.path.exists("admin_images"):
        os.makedirs("admin_images")
        print("Please place clear photos of authorized users in the 'admin_images' folder")
        print("Then run this script again")
        return
    
    # Get all images from the admin_images folder
    image_files = [f for f in os.listdir("admin_images") if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No images found in admin_images folder. Please add images and try again.")
        return
    
    # Process each image and create encodings
    known_encodings = []
    known_names = []
    
    for image_file in image_files:
        # Get person's name from filename (remove extension)
        name = os.path.splitext(image_file)[0]
        
        # Load image and find face encodings
        image_path = os.path.join("admin_images", image_file)
        print(f"Processing image: {image_path}")
        
        try:
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                print(f"No face found in {image_file}. Skipping.")
                continue
            
            # Get encodings
            encodings = face_recognition.face_encodings(image, face_locations)
            if encodings:
                known_encodings.append(encodings[0])  # Use first face found
                known_names.append(name)
                print(f"Encoded {name}'s face")
            else:
                print(f"Could not encode face in {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    # Save encodings to file
    if known_encodings:
        # Create a dictionary with the data
        data_dict = {"encodings": known_encodings, "names": known_names}
        
        # Save using pickle
        with open("image_data.pkl", "wb") as f:
            pickle.dump(data_dict, f)
        
        print(f"Successfully encoded {len(known_names)} faces and saved to image_data.pkl")
    else:
        print("No faces could be encoded. Please check your images.")

if __name__ == "__main__":
    create_encodings()