import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# Screen size
screen_w, screen_h = pyautogui.size()

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Smoothing variables
prev_x, prev_y = 0, 0
smoothening = 5

# Click debounce
click_delay = 0.4
last_click_time = 0

# Active region (reduce jitter)
frame_margin = 100

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape

    # Draw active region
    cv2.rectangle(frame, 
                  (frame_margin, frame_margin), 
                  (w - frame_margin, h - frame_margin), 
                  (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Index fingertip (8)
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            # Map coordinates
            mapped_x = np.interp(x, 
                                 (frame_margin, w - frame_margin), 
                                 (0, screen_w))
            mapped_y = np.interp(y, 
                                 (frame_margin, h - frame_margin), 
                                 (0, screen_h))

            # Smoothing
            curr_x = prev_x + (mapped_x - prev_x) / smoothening
            curr_y = prev_y + (mapped_y - prev_y) / smoothening

            pyautogui.moveTo(screen_w - curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            # Pinch detection
            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            distance = math.hypot(x - thumb_x, y - thumb_y)

            if distance < 30:
                current_time = time.time()
                if current_time - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = current_time
                    cv2.circle(frame, (x, y), 15, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Air Cursor System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
