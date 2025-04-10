import cv2
import face_recognition
import pickle
import os

def register_new_user():
    """
    Function to register a new user in the face database
    """
    name = input("Enter the name of the person to register: ")
    
    print(f"Registering {name}. Please look at the camera.")
    print("Press 'c' to capture the image when ready.")
    
    # Initialize camera
    video_capture = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        if not ret:
            print("Failed to grab frame")
            break
            
        # Display the frame
        cv2.imshow("Register New User", frame)
        
        # Wait for 'c' key to capture
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the image temporarily
            temp_image_path = f"temp_{name}.jpg"
            cv2.imwrite(temp_image_path, frame)
            print(f"Image captured for {name}")
            break
        elif key == ord('q'):
            print("Registration canceled")
            video_capture.release()
            cv2.destroyAllWindows()
            return
    
    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()
    
    # Process the captured image
    image = face_recognition.load_image_file(temp_image_path)
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) == 0:
        print("No face detected in the image. Please try again.")
        os.remove(temp_image_path)
        return
    
    if len(face_locations) > 1:
        print("Multiple faces detected. Please ensure only one person is in the frame.")
        os.remove(temp_image_path)
        return
    
    # Get the encoding for the face
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Load existing data or create new data structure
    data_dict = {"encodings": [], "names": []}
    if os.path.exists("image_data.pkl"):
        with open("image_data.pkl", "rb") as f:
            try:
                data_dict = pickle.load(f)
            except Exception as e:
                print(f"Error loading existing data: {e}")
                print("Creating new database.")
    
    # Add the new face encoding and name
    data_dict["encodings"].append(face_encodings[0])
    data_dict["names"].append(name)
    
    # Save the updated data
    with open("image_data.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    
    print(f"Successfully registered {name}.")
    print(f"Total users in database: {len(data_dict['names'])}")
    
    # Create a directory for registered users if it doesn't exist
    if not os.path.exists("registered_users"):
        os.makedirs("registered_users")
    
    # Move the temporary image to the registered users directory
    os.rename(temp_image_path, f"registered_users/{name}.jpg")

def list_users():
    """
    Function to list all registered users
    """
    if not os.path.exists("image_data.pkl"):
        print("No users registered yet.")
        return
    
    try:
        with open("image_data.pkl", "rb") as f:
            data_dict = pickle.load(f)
            
            print("\nRegistered Users:")
            print("----------------")
            for i, name in enumerate(data_dict["names"]):
                print(f"{i+1}. {name}")
            print(f"\nTotal users: {len(data_dict['names'])}")
    except Exception as e:
        print(f"Error loading user data: {e}")

def remove_user():
    """
    Function to remove a user from the database
    """
    if not os.path.exists("image_data.pkl"):
        print("No users registered yet.")
        return
    
    try:
        with open("image_data.pkl", "rb") as f:
            data_dict = pickle.load(f)
            
        if len(data_dict["names"]) == 0:
            print("No users registered yet.")
            return
            
        print("\nSelect a user to remove:")
        for i, name in enumerate(data_dict["names"]):
            print(f"{i+1}. {name}")
            
        selection = input("\nEnter the number of the user to remove (or 'q' to cancel): ")
        
        if selection.lower() == 'q':
            print("Operation canceled.")
            return
            
        try:
            index = int(selection) - 1
            if index < 0 or index >= len(data_dict["names"]):
                print("Invalid selection.")
                return
                
            user_name = data_dict["names"][index]
            
            # Remove the user from the data dictionary
            data_dict["encodings"].pop(index)
            data_dict["names"].pop(index)
            
            # Save the updated data
            with open("image_data.pkl", "wb") as f:
                pickle.dump(data_dict, f)
                
            # Remove the user's image if it exists
            user_image_path = f"registered_users/{user_name}.jpg"
            if os.path.exists(user_image_path):
                os.remove(user_image_path)
                
            print(f"Successfully removed {user_name} from the database.")
            print(f"Total users remaining: {len(data_dict['names'])}")
            
        except ValueError:
            print("Please enter a valid number.")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    """
    Main function for the face database utility
    """
    while True:
        print("\nFace Recognition Database Utility")
        print("--------------------------------")
        print("1. Register new user")
        print("2. List all users")
        print("3. Remove a user")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            register_new_user()
        elif choice == '2':
            list_users()
        elif choice == '3':
            remove_user()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()