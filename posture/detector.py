"""
posture/detector.py
-------------------
MediaPipe pose detection + all posture math.

Posture metrics tracked:
  1. Forward Head Angle  — ear vs shoulder vs vertical line
     Good: < 15°   Warning: 15-25°   Bad: > 25°

  2. Shoulder Tilt       — height difference left vs right shoulder
     Good: < 0.03  Warning: 0.03-0.06  Bad: > 0.06  (normalised units)

  3. Spine Lean          — shoulder midpoint X vs hip midpoint X
     Detects leaning forward over desk
     Good: < 0.04  Warning: 0.04-0.08  Bad: > 0.08

  4. Neck Compression    — vertical distance nose to shoulder midpoint
     Detects head drooping / chin-to-chest
     Good: > 0.20  Warning: 0.15-0.20  Bad: < 0.15

Overall score 0-100 = weighted average of the four metrics.
"""

import math
import time
import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


# ── Thresholds ────────────────────────────────────────────────────────────

THRESHOLDS = {
    "forward_head": {"good": 15.0, "warn": 25.0},   # degrees
    "shoulder_tilt": {"good": 0.03, "warn": 0.06},   # normalised
    "spine_lean":   {"good": 0.04, "warn": 0.08},    # normalised
    "neck_compress": {"good": 0.20, "warn": 0.15},   # normalised (inverted)
}

WEIGHTS = {
    "forward_head": 0.40,
    "shoulder_tilt": 0.20,
    "spine_lean":    0.25,
    "neck_compress": 0.15,
}


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class PostureMetrics:
    forward_head_angle: float   = 0.0   # degrees
    shoulder_tilt:      float   = 0.0   # normalised
    spine_lean:         float   = 0.0   # normalised
    neck_compress:      float   = 0.0   # normalised
    score:              float   = 100.0 # 0-100
    status:             str     = "good"  # good / warn / bad
    feedback:           list    = field(default_factory=list)
    timestamp:          float   = field(default_factory=time.time)
    detected:           bool    = True


@dataclass
class FrameResult:
    annotated_image: np.ndarray
    metrics:         Optional[PostureMetrics]
    detected:        bool


# ── Maths helpers ─────────────────────────────────────────────────────────

def _angle_3pts(ax, ay, bx, by, cx, cy) -> float:
    """Angle at point B, in degrees."""
    ba = np.array([ax - bx, ay - by])
    bc = np.array([cx - bx, cy - by])
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(float(np.clip(cos_a, -1, 1))))


def _metric_score(value: float, good: float, warn: float, inverted=False) -> float:
    """
    Map a metric value to a 0-100 score component.
    inverted=True means lower value is worse (e.g. neck_compress).
    """
    if not inverted:
        # Lower is better
        if value <= good:
            return 100.0
        elif value <= warn:
            return 100.0 - 50.0 * (value - good) / (warn - good)
        else:
            return max(0.0, 50.0 - 50.0 * (value - warn) / (warn + 1e-8))
    else:
        # Higher is better
        if value >= good:
            return 100.0
        elif value >= warn:
            return 100.0 - 50.0 * (good - value) / (good - warn + 1e-8)
        else:
            return max(0.0, 50.0 - 50.0 * (warn - value) / (warn + 1e-8))


# ── Main detector class ───────────────────────────────────────────────────

class PostureDetector:

    def __init__(self, model_complexity: int = 1):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.L = mp_pose.PoseLandmark

    def close(self):
        self.pose.close()

    # ── Core processing ──────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, draw: bool = True) -> FrameResult:
        """
        Process a single BGR frame from OpenCV.
        Returns FrameResult with annotated image + metrics.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        annotated = frame.copy()

        if not results.pose_landmarks:
            cv2.putText(annotated, "No person detected — move into frame",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
            m = PostureMetrics(detected=False, status="unknown", score=0)
            return FrameResult(annotated, m, False)

        lm = results.pose_landmarks.landmark

        if draw:
            self._draw_skeleton(annotated, results)

        metrics = self._compute_metrics(lm)

        if draw:
            self._draw_hud(annotated, metrics)

        return FrameResult(annotated, metrics, True)

    # ── Posture math ─────────────────────────────────────────────────────

    def _compute_metrics(self, lm) -> PostureMetrics:
        L = self.L

        # Landmarks we need
        nose      = lm[L.NOSE]
        l_ear     = lm[L.LEFT_EAR]
        r_ear     = lm[L.RIGHT_EAR]
        l_sho     = lm[L.LEFT_SHOULDER]
        r_sho     = lm[L.RIGHT_SHOULDER]
        l_hip     = lm[L.LEFT_HIP]
        r_hip     = lm[L.RIGHT_HIP]

        # Use the ear with higher visibility
        ear = l_ear if l_ear.visibility > r_ear.visibility else r_ear
        sho = l_sho if ear == l_ear else r_sho

        # ── 1. Forward head angle ────────────────────────────────────────
        # Angle at shoulder between ear and a point directly above shoulder
        # (vertical reference = same X as shoulder, Y = shoulder Y - 1.0)
        fha = _angle_3pts(
            ear.x, ear.y,
            sho.x, sho.y,
            sho.x, sho.y - 1.0   # point directly above shoulder
        )
        # This gives ~10° for good posture, 30°+ for forward head

        # ── 2. Shoulder tilt ─────────────────────────────────────────────
        sho_tilt = abs(l_sho.y - r_sho.y)

        # ── 3. Spine lean ────────────────────────────────────────────────
        sho_mid_x = (l_sho.x + r_sho.x) / 2
        hip_mid_x = (l_hip.x + r_hip.x) / 2
        spine_lean = abs(sho_mid_x - hip_mid_x)

        # ── 4. Neck compression ──────────────────────────────────────────
        sho_mid_y = (l_sho.y + r_sho.y) / 2
        neck_compress = sho_mid_y - nose.y   # positive = nose above shoulders

        # ── Score each metric ────────────────────────────────────────────
        t = THRESHOLDS
        s_fha  = _metric_score(fha,        t["forward_head"]["good"],  t["forward_head"]["warn"])
        s_tilt = _metric_score(sho_tilt,   t["shoulder_tilt"]["good"], t["shoulder_tilt"]["warn"])
        s_lean = _metric_score(spine_lean, t["spine_lean"]["good"],    t["spine_lean"]["warn"])
        s_neck = _metric_score(neck_compress, t["neck_compress"]["good"], t["neck_compress"]["warn"], inverted=True)

        score = (
            s_fha  * WEIGHTS["forward_head"] +
            s_tilt * WEIGHTS["shoulder_tilt"] +
            s_lean * WEIGHTS["spine_lean"] +
            s_neck * WEIGHTS["neck_compress"]
        )

        # ── Status + human-readable feedback ────────────────────────────
        status = "good" if score >= 75 else ("warn" if score >= 50 else "bad")
        feedback = []

        if fha > t["forward_head"]["warn"]:
            feedback.append(f"Head too far forward ({fha:.0f}°) — tuck your chin")
        elif fha > t["forward_head"]["good"]:
            feedback.append(f"Slight forward head tilt ({fha:.0f}°)")

        if sho_tilt > t["shoulder_tilt"]["warn"]:
            feedback.append("Shoulders uneven — relax both sides equally")
        elif sho_tilt > t["shoulder_tilt"]["good"]:
            feedback.append("Slight shoulder asymmetry")

        if spine_lean > t["spine_lean"]["warn"]:
            feedback.append("Leaning sideways — sit up straight")
        elif spine_lean > t["spine_lean"]["good"]:
            feedback.append("Slight lean detected")

        if neck_compress < t["neck_compress"]["warn"]:
            feedback.append("Head drooping down — lift your gaze")
        elif neck_compress < t["neck_compress"]["good"]:
            feedback.append("Head slightly low")

        if not feedback:
            feedback.append("Great posture! Keep it up.")

        return PostureMetrics(
            forward_head_angle = round(fha, 1),
            shoulder_tilt      = round(sho_tilt, 3),
            spine_lean         = round(spine_lean, 3),
            neck_compress      = round(neck_compress, 3),
            score              = round(score, 1),
            status             = status,
            feedback           = feedback,
            timestamp          = time.time(),
            detected           = True,
        )

    # ── Drawing ──────────────────────────────────────────────────────────

    def _draw_skeleton(self, img, results):
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 200, 255), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 180), thickness=2),
        )

    def _draw_hud(self, img, m: PostureMetrics):
        h, w = img.shape[:2]
        colour = {
            "good":    (0, 220, 100),
            "warn":    (0, 200, 255),
            "bad":     (0, 80, 255),
            "unknown": (150, 150, 150),
        }[m.status]

        # Score bar background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (15, 15, 25), -1)
        img[:] = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)

        # Score bar fill
        bar_w = int((m.score / 100) * (w - 20))
        cv2.rectangle(img, (10, 8), (10 + bar_w, 22), colour, -1)
        cv2.rectangle(img, (10, 8), (w - 10, 22), (80, 80, 80), 1)

        # Text
        cv2.putText(img, f"Posture Score: {m.score:.0f}/100  |  {m.status.upper()}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

        # Feedback line (first item only — don't clutter)
        if m.feedback:
            cv2.putText(img, m.feedback[0],
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (200, 200, 200), 1, cv2.LINE_AA)
