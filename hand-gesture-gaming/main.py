import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
import numpy as np

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Keyboard Controller
keyboard = Controller()

# Start Webcam
cap = cv2.VideoCapture(0)

# Store previous hand position
prev_x, prev_y = None, None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame for natural movement
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    results = hands.process(rgb_frame)
    frame_height, frame_width, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the wrist (landmark 0)
            x, y = int(hand_landmarks.landmark[0].x * frame_width), int(hand_landmarks.landmark[0].y * frame_height)

            if prev_x is not None and prev_y is not None:
                dx, dy = x - prev_x, y - prev_y  # Calculate movement

                # Detect swipe direction
                if abs(dx) > abs(dy):  # Horizontal movement
                    if dx > 50:
                        print("Swipe Right → Move Right")
                        keyboard.press(Key.right)
                        keyboard.release(Key.right)
                    elif dx < -50:
                        print("Swipe Left → Move Left")
                        keyboard.press(Key.left)
                        keyboard.release(Key.left)
                else:  # Vertical movement
                    if dy < -50:
                        print("Swipe Up → Jump")
                        keyboard.press(Key.up)
                        keyboard.release(Key.up)
                    elif dy > 50:
                        print("Swipe Down → Roll")
                        keyboard.press(Key.down)
                        keyboard.release(Key.down)

            prev_x, prev_y = x, y  # Update previous position

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()