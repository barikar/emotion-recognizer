import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load model architecture and weights
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.keras")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Preprocess image before feeding to model
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # reshape to match input dimensions
    return feature / 255.0  # normalize the image

# Start webcam
webcam = cv2.VideoCapture(0)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Read image from webcam
    _, im = webcam.read()
    
    # Check if frame is read correctly
    if not _:
        print("Failed to grab frame.")
        break
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    if len(faces) > 0:  # If faces are detected
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]  # Crop the face region
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)  # Draw rectangle around face

            # Resize and preprocess image
            image_resized = cv2.resize(image, (48, 48))  # Resize to 48x48
            img = extract_features(image_resized)  # Extract features

            # Predict emotion
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]  # Get the emotion label with the highest probability
            print("Predicted Output:", prediction_label)

            # Display predicted emotion on the image
            cv2.putText(im, prediction_label, (p, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

    # Show the image with the predicted emotion
    cv2.imshow("Output", im)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
