import cv2
import mediapipe as mp
import numpy as np

# ---------------------- CONFIG ----------------------
CAM_WIDTH = 1280
CAM_HEIGHT = 720

BRUSH_THICKNESS = 8
ERASER_THICKNESS = 40

# ----------------------------------------------------

# Initialize Camera
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Canvas
canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)

# Drawing state
draw_color = (255, 0, 255)
prev_x, prev_y = 0, 0


# ---------------------- FUNCTIONS ----------------------

def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = []

    for tip in tips:
        fingers.append(
            hand_landmarks.landmark[tip].y <
            hand_landmarks.landmark[tip - 2].y
        )

    return fingers


def draw_palette(frame):
    colors = [
        (255, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255)
    ]

    for i, color in enumerate(colors):
        cv2.rectangle(frame,
                      (i * 200, 0),
                      ((i + 1) * 200, 100),
                      color,
                      -1)


# ---------------------- MAIN LOOP ----------------------

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    draw_palette(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame,
                                   hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            x1 = int(lm[8].x * CAM_WIDTH)
            y1 = int(lm[8].y * CAM_HEIGHT)

            fingers = fingers_up(hand_landmarks)
            finger_count = fingers.count(True)

            # ---------------- CLEAR MODE ----------------
            if finger_count == 4:
                canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)
                prev_x, prev_y = 0, 0
                cv2.putText(frame, "Canvas Cleared",
                            (450, 400),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 4)

            # ---------------- SELECTION MODE ----------------
            elif fingers[0] and fingers[1] and not fingers[2]:
                prev_x, prev_y = 0, 0

                if y1 < 100:
                    if 0 < x1 < 200:
                        draw_color = (255, 0, 255)
                    elif 200 < x1 < 400:
                        draw_color = (255, 0, 0)
                    elif 400 < x1 < 600:
                        draw_color = (0, 255, 0)
                    elif 600 < x1 < 800:
                        draw_color = (0, 0, 255)

                cv2.rectangle(frame,
                              (x1 - 20, y1 - 20),
                              (x1 + 20, y1 + 20),
                              draw_color,
                              cv2.FILLED)

            # ---------------- ERASER MODE ----------------
            elif fingers[0] and fingers[1] and fingers[2] and not fingers[3]:
                cv2.circle(frame, (x1, y1), 20, (0, 0, 0), cv2.FILLED)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1

                cv2.line(canvas,
                         (prev_x, prev_y),
                         (x1, y1),
                         (0, 0, 0),
                         ERASER_THICKNESS)

                prev_x, prev_y = x1, y1

            # ---------------- DRAW MODE ----------------
            elif fingers[0] and not fingers[1]:
                cv2.circle(frame, (x1, y1), 10, draw_color, cv2.FILLED)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1

                cv2.line(canvas,
                         (prev_x, prev_y),
                         (x1, y1),
                         draw_color,
                         BRUSH_THICKNESS)

                prev_x, prev_y = x1, y1

            # ---------------- RESET ----------------
            else:
                prev_x, prev_y = 0, 0

    # Merge Canvas
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Air Drawing System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
