import cv2
import torch
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

print("Starting script...")

# Load YOLOv5 model
print("Loading YOLOv5 model...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
yolo_model.classes = [0]  # Only detect persons
yolo_model.conf = 0.25  # Lower confidence threshold for faster detection
yolo_model.eval()  # Set the model to evaluation mode
print("YOLOv5 model loaded successfully.")

# Load gender classification model
print("Loading gender classification model...")
gender_model = load_model('gender_classification_OG.h5')
print("Gender classification model loaded successfully.")

# Open video file (replace with camera feed for real-time usage)
print("Opening video file...")
video = cv2.VideoCapture('footage.mp4')
if not video.isOpened():
    raise Exception("Error opening video file")
print("Video file opened successfully.")

# Initialize variables
frame_count = 0
last_classification_time = time.time()
classification_interval = 1.0  # seconds
person_genders = {}

# Pre-allocate numpy array for faster processing
person_img = np.zeros((1, 224, 224, 3), dtype=np.float32)

print("Starting video processing...")

while True:
    start_time = time.time()
    
    ret, frame = video.read()
    if not ret:
        print("End of video reached.")
        break

    frame_count += 1
    current_time = time.time()

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # YOLOv5 detection
    results = yolo_model(frame)
    detections = results.xyxy[0].cpu().numpy()

    male_count = 0
    female_count = 0

    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        person_id = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"

        if current_time - last_classification_time >= classification_interval or person_id not in person_genders:
            person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            person_crop = cv2.resize(person_crop, (224, 224))
            np.copyto(person_img[0], person_crop)
            person_img_preprocessed = person_img / 255.0

            gender_pred = gender_model.predict(person_img_preprocessed, verbose=0)[0][0]
            gender = "Male" if gender_pred > 0.5 else "Female"
            person_genders[person_id] = gender

        gender = person_genders.get(person_id, "Unknown")

        if gender == "Male":
            male_count += 1
            color = (255, 0, 0)
        elif gender == "Female":
            female_count += 1
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, gender, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if current_time - last_classification_time >= classification_interval:
        last_classification_time = current_time
        person_genders = {k: v for k, v in person_genders.items() if k in [f"{int(d[0])}_{int(d[1])}_{int(d[2])}_{int(d[3])}" for d in detections]}

    cv2.putText(frame, f"Male: {male_count} Female: {female_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Gender Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User interrupted the process.")
        break

    # Calculate and print FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps:.2f}")

print("Video processing completed.")
video.release()
cv2.destroyAllWindows()

print("Script execution finished.")