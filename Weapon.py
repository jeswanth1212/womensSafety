import cv2
import torch
from yolov5 import YOLOv5

# Load YOLOv5 model (change 'yolov5s' to your model if needed)
model = YOLOv5('yolov5s.pt', device='cuda' if torch.cuda.is_available() else 'cpu')

# Load video
video_path = 'viole2.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model.predict(frame)

    # Process results
    for result in results.pandas().xyxy[0].itertuples():
        class_name = result.name
        if class_name == 'weapon':  # Ensure 'weapon' matches your model's label
            # Draw the "Weapon Detected" message
            cv2.putText(frame, 'Weapon Detected', (frame.shape[1] - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Weapon Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
