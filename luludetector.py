"""
Hand Motion Detector

Detects repeated up/down hand movement and plays:
    wa-na-nag-lulu-na.mp3

Controls:
    q - quit
"""

from collections import deque
from dataclasses import dataclass
import statistics
import time

import cv2
import mediapipe as mp
import pygame


# ------------------------------ Tunable Settings ------------------------------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

WINDOW_SECONDS = 1.2
MIN_SAMPLES = 18

SMOOTH_ALPHA = 0.35
VELOCITY_DEADZONE_NORM = 0.020
MIN_RANGE_NORM = 0.90
MIN_SPEED_NORM = 0.022
MIN_SIGN_CHANGES = 7

DETECTION_HOLD_SEC = 0.45
COOLDOWN_SECONDS = 5.0

HANDS_DETECT_CONF = 0.60
HANDS_TRACK_CONF = 0.60
HANDS_MODEL_COMPLEXITY = 1
# -----------------------------------------------------------------------------


@dataclass
class Sample:
    t: float
    y_norm: float
    hand_scale: float


class MotionDetector:
    def __init__(self):
        self.history = deque()
        self.smooth_y = None
        self.last_detect_t = -1.0
        self.last_play_t = -1.0

    def reset(self):
        self.history.clear()
        self.smooth_y = None

    def add(self, t_now: float, y_norm: float, hand_scale: float):
        if self.smooth_y is None:
            self.smooth_y = y_norm
        else:
            self.smooth_y = (SMOOTH_ALPHA * y_norm) + ((1.0 - SMOOTH_ALPHA) * self.smooth_y)

        self.history.append(Sample(t=t_now, y_norm=self.smooth_y, hand_scale=max(hand_scale, 1e-6)))

        while self.history and (t_now - self.history[0].t) > WINDOW_SECONDS:
            self.history.popleft()

    def detect(self, t_now: float) -> bool:
        if len(self.history) < MIN_SAMPLES:
            return (self.last_detect_t >= 0.0) and ((t_now - self.last_detect_t) <= DETECTION_HOLD_SEC)

        ys = [s.y_norm for s in self.history]
        scales = [s.hand_scale for s in self.history]
        dt = max(self.history[-1].t - self.history[0].t, 1e-6)

        ref_scale = max(statistics.median(scales), 1e-6)
        vertical_range_norm = (max(ys) - min(ys)) / ref_scale

        diffs_norm = []
        signs = []
        for i in range(len(self.history) - 1):
            dy = self.history[i + 1].y_norm - self.history[i].y_norm
            dtn = max(self.history[i + 1].t - self.history[i].t, 1e-6)
            vel_norm = (dy / dtn) / ref_scale
            diffs_norm.append(abs(vel_norm))
            if vel_norm > VELOCITY_DEADZONE_NORM:
                signs.append(1)
            elif vel_norm < -VELOCITY_DEADZONE_NORM:
                signs.append(-1)
            else:
                signs.append(0)

        sign_changes = 0
        prev = 0
        for s in signs:
            if s == 0:
                continue
            if prev != 0 and s != prev:
                sign_changes += 1
            prev = s

        avg_speed_norm = (sum(diffs_norm) / max(len(diffs_norm), 1))
        sustained_rhythm = sign_changes >= MIN_SIGN_CHANGES
        enough_range = vertical_range_norm >= MIN_RANGE_NORM
        enough_speed = avg_speed_norm >= MIN_SPEED_NORM

        detected = sustained_rhythm and enough_range and enough_speed and (dt >= 0.5)
        if detected:
            self.last_detect_t = t_now
            return True

        return (self.last_detect_t >= 0.0) and ((t_now - self.last_detect_t) <= DETECTION_HOLD_SEC)

    def can_play(self, t_now: float) -> bool:
        return self.last_play_t < 0.0 or (t_now - self.last_play_t) >= COOLDOWN_SECONDS

    def mark_played(self, t_now: float):
        self.last_play_t = t_now


def lm_to_px(lm, w, h):
    return (lm.x * w, lm.y * h)


def compute_hand_scale(hand_landmarks, w, h):
    wrist = lm_to_px(hand_landmarks.landmark[0], w, h)
    index_mcp = lm_to_px(hand_landmarks.landmark[5], w, h)
    pinky_mcp = lm_to_px(hand_landmarks.landmark[17], w, h)
    dx = index_mcp[0] - pinky_mcp[0]
    dy = index_mcp[1] - pinky_mcp[1]
    palm_width = (dx * dx + dy * dy) ** 0.5
    return max(palm_width / max(h, 1), 0.01)


def get_primary_hand(results):
    if not results.multi_hand_landmarks:
        return None

    if results.multi_handedness and len(results.multi_handedness) == len(results.multi_hand_landmarks):
        best_i = 0
        best_score = -1.0
        for i, handed in enumerate(results.multi_handedness):
            score = handed.classification[0].score if handed.classification else 0.0
            if score > best_score:
                best_score = score
                best_i = i
        return results.multi_hand_landmarks[best_i]

    return results.multi_hand_landmarks[0]


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=HANDS_MODEL_COMPLEXITY,
        min_detection_confidence=HANDS_DETECT_CONF,
        min_tracking_confidence=HANDS_TRACK_CONF,
    )

    pygame.mixer.init()
    try:
        sound = pygame.mixer.Sound("wa-na-nag-lulu-na.mp3")
    except Exception as exc:
        print(f"Could not load sound file: {exc}")
        print("Place wa-na-nag-lulu-na.mp3 in this folder.")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    detector = MotionDetector()
    print("Detector started. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Cannot access webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        t_now = time.perf_counter()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        detected = False
        primary = get_primary_hand(results)

        if primary is not None:
            mp_draw.draw_landmarks(
                frame,
                primary,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 180, 0), thickness=2),
            )

            wrist = primary.landmark[0]
            middle_mcp = primary.landmark[9]
            center_y_norm = (wrist.y + middle_mcp.y) * 0.5
            hand_scale = compute_hand_scale(primary, w, h)
            detector.add(t_now, center_y_norm, hand_scale)
            detected = detector.detect(t_now)
        else:
            detector.reset()

        if detected and detector.can_play(t_now):
            sound.play()
            detector.mark_played(t_now)

        status = "DETECTED" if detected else "WAITING"
        cv2.putText(frame, f"Motion: {status}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"Cooldown: {max(0.0, COOLDOWN_SECONDS - max(0.0, t_now - detector.last_play_t)):.1f}s",
            (10, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
        )

        cv2.imshow("Lulu Detector - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    hands.close()


if __name__ == "__main__":
    main()
