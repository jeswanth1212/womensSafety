import cv2
import tempfile
import os
from inference_sdk import InferenceHTTPClient

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="kLvuP8kL8bWktjLADNte"
)

def detect_violence(frame):
    # Save the frame to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file_path = temp_file.name
        cv2.imwrite(temp_file_path, frame)

    # Perform inference
    result = CLIENT.infer(temp_file_path, model_id="violence-detection-s9acq/1")

    # Delete the temporary file
    os.remove(temp_file_path)

    return result

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
    frame_interval = int(fps * 0.5)  # Number of frames to skip (0.5 seconds)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Check for violence every 0.5 seconds
        if frame_count % frame_interval == 0:
            result = detect_violence(frame)
            violence_detected = any(pred['class'] == 'Violence' and pred['confidence'] > 0.5 for pred in result.get('predictions', []))

            if violence_detected:
                print("Violence detected!")

        # Display the frame
        cv2.imshow('Video', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'viole.mp4'  # Replace with your video path
    main(video_path)