# metrics: latency, gesture_actions, gesture_usage_ratio

import cv2
import mediapipe as mp
import time
import numpy as np
from pynput.keyboard import Key, Controller
from collections import deque

# ---------------- Utility ----------------
def angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

# ---------------- Gesture Controller ----------------
class GestureController:
    def __init__(self):
        self.kb = Controller()
        self.last_action_time = 0

        # Temporal smoothing buffer
        self.buffer = deque(maxlen=7)

        # Metrics
        self.latencies = []
        self.gesture_actions = 0
        self.total_actions = 0

        # MediaPipe Hands
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.draw = mp.solutions.drawing_utils

    # ---------------- Finger State ----------------
    def fingers_extended(self, lm):
        joints = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
        return [angle(lm[m], lm[p], lm[t]) > 160 for m, p, t in joints]

    # ---------------- Gesture Classification ----------------
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

    # ---------------- Action Execution ----------------
    def act(self, gesture):
        if time.time() - self.last_action_time < 0.4:
            return

        keys = {
            "PlayPause": Key.media_play_pause,
            "VolumeUp": Key.media_volume_up,
            "VolumeDown": Key.media_volume_down
        }

        if gesture in keys:
            self.kb.press(keys[gesture])
            self.kb.release(keys[gesture])

            self.gesture_actions += 1
            self.total_actions += 1
            self.last_action_time = time.time()

    # ---------------- Main Loop ----------------
    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            start_time = time.perf_counter()

            res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                self.draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                lm = np.array([[p.x * w, p.y * h, p.z] for p in hand.landmark])

                gesture = self.classify(lm)
                self.buffer.append(gesture)

                if gesture and self.buffer.count(gesture) >= 5:
                    self.act(gesture)
                    cv2.putText(frame, gesture, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Latency measurement
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latencies.append(latency_ms)

            avg_latency = np.mean(self.latencies[-30:])
            cv2.putText(frame, f"Latency: {avg_latency:.1f} ms",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        self.report_metrics()

    # ---------------- Metrics Report ----------------
    def report_metrics(self):
        avg_latency = np.mean(self.latencies)
        gesture_ratio = (
            self.gesture_actions / self.total_actions
            if self.total_actions > 0 else 0
        )

        print("\n--- Performance Metrics ---")
        print(f"Average Latency: {avg_latency:.2f} ms")
        print(f"Gesture-Controlled Actions: {self.gesture_actions}")
        print(f"Gesture Usage Ratio: {gesture_ratio * 100:.1f}%")

# ---------------- Run ----------------
if __name__ == "__main__":
    GestureController().run()
