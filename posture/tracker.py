"""
posture/tracker.py
------------------
Tracks posture state over time.
- Rolling average score (smooths out momentary dips)
- Bad posture timer: how long continuously below threshold
- Alert trigger: fires after BAD_POSTURE_ALERT_MINUTES of continuous bad posture
- Session stats: per-session good/warn/bad time breakdown
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional
from .detector import PostureMetrics


BAD_POSTURE_ALERT_MINUTES = 10   # alert after this many continuous bad minutes
BAD_SCORE_THRESHOLD       = 60   # score below this = "bad posture"
SMOOTHING_WINDOW          = 30   # frames for rolling average (~1 sec at 30fps)


@dataclass
class SessionStats:
    started_at:     float = field(default_factory=time.time)
    good_seconds:   float = 0.0
    warn_seconds:   float = 0.0
    bad_seconds:    float = 0.0
    total_frames:   int   = 0
    avg_score:      float = 0.0
    alerts_fired:   int   = 0

    @property
    def total_seconds(self):
        return self.good_seconds + self.warn_seconds + self.bad_seconds

    @property
    def good_pct(self):
        t = self.total_seconds
        return round(100 * self.good_seconds / t, 1) if t > 0 else 0

    @property
    def bad_pct(self):
        t = self.total_seconds
        return round(100 * self.bad_seconds / t, 1) if t > 0 else 0


class PostureTracker:

    def __init__(self,
                 on_alert: Optional[Callable[[float], None]] = None,
                 alert_minutes: float = BAD_POSTURE_ALERT_MINUTES):
        """
        on_alert: called when bad posture streak reaches alert_minutes.
                  Receives the number of bad minutes as argument.
        """
        self._on_alert     = on_alert
        self._alert_secs   = alert_minutes * 60

        # Rolling score buffer
        self._scores: deque = deque(maxlen=SMOOTHING_WINDOW)

        # Bad posture streak
        self._bad_since:      Optional[float] = None   # timestamp when streak started
        self._alert_fired_at: Optional[float] = None   # when we last fired an alert

        # Per-frame timing
        self._last_tick: Optional[float] = None

        # Session stats
        self.session = SessionStats()

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, metrics: PostureMetrics):
        """
        Feed a new PostureMetrics sample.
        Call this every frame.
        """
        now = time.time()
        dt  = (now - self._last_tick) if self._last_tick else 0.033
        self._last_tick = now

        if not metrics.detected:
            # No person in frame — pause streak but don't reset
            return

        # Rolling average
        self._scores.append(metrics.score)
        smooth = self.smooth_score

        # Update session stats
        self.session.total_frames += 1
        if metrics.status == "good":
            self.session.good_seconds += dt
        elif metrics.status == "warn":
            self.session.warn_seconds += dt
        else:
            self.session.bad_seconds += dt

        # Update running average score
        n = self.session.total_frames
        self.session.avg_score = (
            (self.session.avg_score * (n - 1) + metrics.score) / n
        )

        # Bad posture streak tracking
        if smooth < BAD_SCORE_THRESHOLD:
            if self._bad_since is None:
                self._bad_since = now
            streak = now - self._bad_since

            # Fire alert if streak exceeds threshold
            # and we haven't already alerted in the last alert_secs window
            if streak >= self._alert_secs:
                last = self._alert_fired_at
                if last is None or (now - last) >= self._alert_secs:
                    self._alert_fired_at = now
                    self.session.alerts_fired += 1
                    if self._on_alert:
                        self._on_alert(streak / 60)
        else:
            # Good posture — reset streak
            self._bad_since = None

    def reset_session(self):
        """Start a fresh session."""
        self.session = SessionStats()
        self._scores.clear()
        self._bad_since      = None
        self._alert_fired_at = None
        self._last_tick      = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def smooth_score(self) -> float:
        """Rolling average score over last ~1 second of frames."""
        if not self._scores:
            return 100.0
        return round(sum(self._scores) / len(self._scores), 1)

    @property
    def bad_streak_minutes(self) -> float:
        """How many minutes of continuous bad posture right now."""
        if self._bad_since is None:
            return 0.0
        return round((time.time() - self._bad_since) / 60, 1)

    @property
    def alert_in_minutes(self) -> Optional[float]:
        """Minutes until next alert fires (None if posture is good)."""
        if self._bad_since is None:
            return None
        remaining = self._alert_secs - (time.time() - self._bad_since)
        last = self._alert_fired_at
        if last:
            remaining = min(remaining, self._alert_secs - (time.time() - last))
        return max(0.0, round(remaining / 60, 1))

    def status_line(self) -> str:
        """One-line status for terminal display."""
        score = self.smooth_score
        streak = self.bad_streak_minutes
        parts = [f"Score: {score:.0f}/100"]
        if streak > 0:
            parts.append(f"| Bad posture: {streak:.1f} min")
            alert_in = self.alert_in_minutes
            if alert_in is not None and alert_in > 0:
                parts.append(f"| Alert in: {alert_in:.1f} min")
        parts.append(f"| Session: {self.session.good_pct}% good")
        return "  ".join(parts)
