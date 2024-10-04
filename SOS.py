import cv2
import mediapipe as mp
import time

# Initialize MediaPipe hands detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Function to check if fingers are open or closed and if they are close together
def check_finger_state(hand_landmarks, hand_label):
    finger_tips = [8, 12, 16, 20]  # Landmarks for fingertips
    finger_bases = [5, 9, 13, 17]  # Landmarks for finger bases
    
    fingers_open = []
    fingers_close_together = True
    
    for i, (tip, base) in enumerate(zip(finger_tips, finger_bases)):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            fingers_open.append(True)
        else:
            fingers_open.append(False)
        
        # Check if fingers are close to each other by comparing the x-coordinates of neighboring fingertips
        if i > 0:  # Skip the first finger (index finger)
            previous_tip = finger_tips[i-1]
            if abs(hand_landmarks.landmark[tip].x - hand_landmarks.landmark[previous_tip].x) > 0.05:  # Adjust threshold as necessary
                fingers_close_together = False
    
    # Check if thumb is visible (different for left and right hands)
    if hand_label == "Right":
        thumb_visible = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
    else:  # Left hand
        thumb_visible = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
    
    return fingers_open, fingers_close_together, thumb_visible

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Variables for gesture tracking
last_gesture_time = time.time()
gesture_state = 0  # 0: waiting for thumb tucked in, 1: waiting for fingers closed, 2: waiting for thumb tucked in again, 3: waiting for fingers closed again
current_message = "Waiting for hand..."
sos_detected = False
thumb_tucked_in = False
fingers_closed_count = 0
fingers_open_count = 0

while cap.isOpened() and not sos_detected:
    success, image = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    # Flip the image horizontally and convert to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image
    results = hands.process(image_rgb)

    # Get number of hands detected
    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    if num_hands == 1:  # Detect SOS gesture only if exactly one hand is visible
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_label = results.multi_handedness[0].classification[0].label  # "Left" or "Right"
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Check finger state and proximity
        fingers_open, fingers_close_together, thumb_visible = check_finger_state(hand_landmarks, hand_label)

        # Determine the current gesture
        if not thumb_visible:
            thumb_tucked_in = True
            if all(not finger for finger in fingers_open) and fingers_close_together:
                if gesture_state == 0:
                    gesture_state = 1
                    fingers_closed_count += 1
                elif gesture_state == 2:
                    gesture_state = 3
                    fingers_closed_count += 1
            else:
                fingers_closed_count = 0
                gesture_state = 0
        else:
            thumb_tucked_in = False
            if gesture_state == 1:
                gesture_state = 2
            else:
                fingers_closed_count = 0
                gesture_state = 0

        # Update the current message
        if thumb_tucked_in:
            if gesture_state == 0:
                current_message = "Thumb tucked in"
            elif gesture_state == 2:
                current_message = "Thumb tucked in again"
        elif gesture_state == 1:
            current_message = "Fingers closed once"
        elif gesture_state == 3:
            current_message = "Fingers closed twice"
            sos_detected = True
            print("SOS GESTURE DETECTED !")
            break

        # Check if fingers are open
        if all(finger for finger in fingers_open) and fingers_close_together:
            fingers_open_count += 1
            if fingers_open_count > 5:  # Adjust threshold as necessary
                fingers_closed_count = 0
                gesture_state = 0
                current_message = "Fingers open"
        else:
            fingers_open_count = 0

    else:
        if current_message != "Waiting for hand...":
            current_message = "Waiting for hand..."
            print(current_message)
        thumb_tucked_in = False
        fingers_closed_count = 0
        gesture_state = 0

    # Display the current message on the image
    cv2.putText(image, current_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the image
    cv2.imshow('SOS Gesture Detection ', image)

    # Exit on key press
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
print("Program exited due to SOS detection or user intervention.")