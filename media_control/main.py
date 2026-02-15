import cv2
import mediapipe as mp
import time
import numpy as np
from pynput.keyboard import Key, Controller
from collections import deque

# Compute angle between three joints (for finger straightness)
def angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

class GestureController:
    def __init__(self):
        # Keyboard control
        self.kb = Controller()
        self.last_time = 0

        # Buffer for temporal smoothing
        self.buffer = deque(maxlen=7)

        # MediaPipe hand tracker
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.draw = mp.solutions.drawing_utils

    # Detect which fingers are extended using joint angles
    def fingers_extended(self, lm):
        joints = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
        return [angle(lm[m], lm[p], lm[t]) > 160 for m, p, t in joints]

    # Map finger states to gestures
    def classify(self, lm):
        f = self.fingers_extended(lm)
        c = sum(f)

        if c >= 4:
            return "PlayPause"
        if c == 0:
            return "Stop"
        if f[0] and not any(f[1:]):
            return "VolumeUp"
        if f[0] and f[1]:
            return "VolumeDown"
        return None

    # Execute media action with cooldown
    def act(self, gesture):
        if time.time() - self.last_time < 0.4:
            return

        keys = {
            "PlayPause": Key.media_play_pause,
            "VolumeUp": Key.media_volume_up,
            "VolumeDown": Key.media_volume_down
        }

        if gesture in keys:
            self.kb.press(keys[gesture])
            self.kb.release(keys[gesture])
            self.last_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Hand landmark detection
            res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                self.draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

                # Convert landmarks to pixel coordinates
                h, w, _ = frame.shape
                lm = np.array([[p.x * w, p.y * h, p.z] for p in hand.landmark])

                # Classify and smooth gesture
                g = self.classify(lm)
                self.buffer.append(g)

                if g and self.buffer.count(g) >= 5:
                    self.act(g)
                    cv2.putText(frame, g, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    GestureController().run()
