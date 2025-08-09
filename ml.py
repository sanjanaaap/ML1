from keras.models import load_model
import cv2
import numpy as np
import os
import time

# Force CPU mode if needed
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Camera initialization with DSHOW backend
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change to CAP_MSMF if on Windows
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("ERROR: Cannot open camera")
    exit()

# Warm-up camera
for _ in range(5):
    camera.read()
time.sleep(2)  # Allow camera to initialize

while True:
    # Read frame with retry
    for attempt in range(3):
        ret, frame = camera.read()
        if ret:
            break
        time.sleep(0.1)
    else:
        print("Failed to capture frame after 3 attempts")
        continue

    try:
        # Process frame
        resized = cv2.resize(frame, (224, 224))
        cv2.imshow("Webcam", resized)
        
        # Prediction
        image_array = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
        normalized = (image_array / 127.5) - 1
        prediction = model.predict(normalized, verbose=0)
        
        # Display results
        index = np.argmax(prediction)
        print(f"Class: {class_names[index].strip()} ({np.max(prediction)*100:.2f}%)")

    except Exception as e:
        print(f"Processing error: {e}")
        continue

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()