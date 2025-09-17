import cv2
import numpy as np
from keras.models import load_model
# Load the trained model
model = load_model("facialemotionmodel.h5")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion labels (adjust if your dataset classes are different)
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to preprocess face for the model
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))        # resize to 48x48
    face_img = face_img.reshape(1, 48, 48, 1)        # add batch + channel dims
    face_img = face_img.astype("float32") / 255.0    # normalize
    return face_img

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces[:1]:
        face = gray[y:y+h, x:x+w]
        face_processed = preprocess_face(face)

        prediction = model.predict(face_processed, verbose=0)
        emotion = labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Emotion Detection", frame)

    # Close if 'q' is pressed OR window is closed
    if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.getWindowProperty("Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
        break
