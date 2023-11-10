import cv2
import dlib

# Load the image
imagePath = '../faces.jpg'
image = cv2.imread(imagePath)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Detect the faces
faces = detector(gray)

# Draw rectangles on each face
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Count the faces
print("[INFO] Found {0} Faces.".format(len(faces)))

# Display the image with rectangles
cv2.imshow('Press ESC to exit', image)
while True:
    k = cv2.waitKey(30)
    if k == 27: # ESC to quit
        break
cv2.destroyAllWindows()
