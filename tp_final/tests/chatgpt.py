import cv2
import os

# Load the pre-trained Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to count faces in an image
def count_faces(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces)

# Provide the path to the directory containing images
directory_path = '../faces.jpg'

# Iterate through images in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(directory_path, filename)
        num_faces = count_faces(image_path)
        print(f"Image: {filename}, Number of faces: {num_faces}")
