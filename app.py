from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cvlib as cv

app = Flask(__name__)

# Load trained gender detection model
model = load_model('gender_detection.h5')

# Open webcam
webcam = cv2.VideoCapture(0)

# Gender labels
classes = ['Man', 'Woman']

def generate_frames():
    while True:
        success, frame = webcam.read()
        if not success:
            break

        # Apply face detection
        faces, confidences = cv.detect_face(frame)

        # Loop through detected faces
        for idx, f in enumerate(faces):
            (startX, startY, endX, endY) = f

            # Draw rectangle around face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Crop face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue  # Skip if face crop is too small

            # Preprocessing for model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Predict gender
            conf = model.predict(face_crop)[0]
            idx = np.argmax(conf)
            label = f"{classes[idx]}: {conf[idx] * 100:.2f}%"

            # Adjust label position above face
            label_y = startY - 10 if startY - 10 > 10 else startY + 10

            # Display label (GREEN, thinner font)
            cv2.putText(frame, label, (startX + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
