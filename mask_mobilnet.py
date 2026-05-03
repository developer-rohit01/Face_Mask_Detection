import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load MobileNet model
model = load_model("mask_mobilenet.h5")

# Load DNN face detector
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Convert frame to blob for DNN
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Fix boundaries
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            # Resize
            face = cv2.resize(face, (224, 224))

            # Convert to array
            face = np.array(face, dtype="float32")

            # MobileNet preprocessing (IMPORTANT)
            face = preprocess_input(face)

            # Reshape for model
            face = np.expand_dims(face, axis=0)

            # Prediction
            pred = model.predict(face, verbose=0)[0][0]

            # Label logic
            label = "Mask" if pred < 0.5 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw label
            cv2.putText(frame, f"{label} ({pred:.2f})",
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

            # Draw box
            cv2.rectangle(frame,
                          (startX, startY),
                          (endX, endY),
                          color, 2)

    cv2.imshow("MobileNet Mask Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()