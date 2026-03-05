"""
posture/alerts.py
-----------------
Cross-platform alerts: system notification + beep sound.

Works on Windows, macOS, Linux.
Uses plyer for notifications (falls back to print if not installed).
Uses winsound / afplay / aplay for beep.
"""

import sys
import time
import threading


def _beep():
    """Non-blocking cross-platform beep."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["afplay", "/System/Library/Sounds/Basso.aiff"])
        else:
            # Linux — try paplay, then aplay, then bell char
            import subprocess
            try:
                subprocess.Popen(["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                try:
                    subprocess.Popen(["aplay", "-q", "/usr/share/sounds/alsa/Front_Left.wav"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


def _notify(title: str, message: str):
    """Non-blocking cross-platform system notification."""
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="Posture Coach",
            timeout=8,
        )
    except Exception:
        # plyer not installed or notification failed — just print
        print(f"\n{'='*50}")
        print(f"  ⚠  {title}")
        print(f"     {message}")
        print(f"{'='*50}\n")


class AlertManager:

    def __init__(self):
        self._last_alert = 0.0
        self._cooldown   = 60.0   # minimum seconds between alerts

    def fire(self, bad_minutes: float):
        """
        Called by PostureTracker when bad posture streak is reached.
        Fires notification + beep in a background thread (non-blocking).
        """
        now = time.time()
        if now - self._last_alert < self._cooldown:
            return
        self._last_alert = now

        minutes = round(bad_minutes, 1)
        title   = "Posture Alert 🪑"
        message = (
            f"You've had poor posture for {minutes} minutes.\n"
            "Sit up straight, relax your shoulders, and tuck your chin."
        )

        # Fire in background so it doesn't block the main loop
        t = threading.Thread(target=self._fire_async, args=(title, message), daemon=True)
        t.start()

    def _fire_async(self, title: str, message: str):
        _beep()
        time.sleep(0.3)
        _notify(title, message)
