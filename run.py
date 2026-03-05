"""
run.py
------
Entry point. Run with: python run.py

What happens:
  1. Opens webcam
  2. Starts Flask dashboard on localhost:5000
  3. Opens browser to dashboard automatically
  4. Main loop: reads frames → detects posture → updates tracker → saves DB
  5. Every 10 min of bad posture → OS notification + beep
  6. Press Q in the webcam window to quit

Options:
  --no-window     don't show the OpenCV webcam window (headless mode)
  --no-browser    don't auto-open browser
  --cam N         use camera index N (default 0)
  --alert N       bad posture alert after N minutes (default 10)
  --port N        dashboard port (default 5000)
"""

import argparse
import os
import sys
import time
import webbrowser
import threading
import cv2

# ── Argument parsing ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Posture Coach — local posture monitor")
parser.add_argument("--no-window",  action="store_true", help="Run headless (no webcam window)")
parser.add_argument("--no-browser", action="store_true", help="Don't auto-open dashboard")
parser.add_argument("--cam",        type=int, default=0,    help="Camera index (default 0)")
parser.add_argument("--alert",      type=float, default=10, help="Alert after N bad minutes (default 10)")
parser.add_argument("--port",       type=int, default=5000, help="Dashboard port (default 5000)")
args = parser.parse_args()

# ── Imports (after argparse so --help works without mediapipe) ────────────
from posture.detector  import PostureDetector
from posture.tracker   import PostureTracker
from posture.database  import Database
from posture.alerts    import AlertManager
from dashboard.server  import run_server, update_state, set_db

# ── Initialise components ─────────────────────────────────────────────────
print("Initialising Posture Coach...")

db      = Database()
alerts  = AlertManager()

def on_alert(bad_minutes: float):
    alerts.fire(bad_minutes)
    print(f"\n⚠  POSTURE ALERT — {bad_minutes:.1f} minutes of poor posture!\n")

tracker  = PostureTracker(on_alert=on_alert, alert_minutes=args.alert)
detector = PostureDetector(model_complexity=1)

set_db(db)

# ── Start dashboard ───────────────────────────────────────────────────────
print(f"Starting dashboard on http://127.0.0.1:{args.port}")
run_server(port=args.port)

if not args.no_browser:
    def _open():
        time.sleep(1.5)
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    threading.Thread(target=_open, daemon=True).start()

# ── Open webcam ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    print(f"ERROR: Cannot open camera {args.cam}.")
    print("Try a different index with --cam 1")
    sys.exit(1)

# Set reasonable resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# ── Start session ─────────────────────────────────────────────────────────
session_id = db.start_session()
tracker.reset_session()

print(f"Camera opened. Session #{session_id} started.")
print(f"Dashboard: http://127.0.0.1:{args.port}")
print("Press Q in the webcam window (or Ctrl+C) to quit.\n")

# ── Main loop ─────────────────────────────────────────────────────────────
SHOW_WINDOW  = not args.no_window
SAVE_EVERY   = 30    # save DB row every N frames check (actual save throttled to 1/min)
PRINT_EVERY  = 30    # print status to terminal every N frames

frame_num = 0
last_metrics = None

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed — retrying...")
            time.sleep(0.1)
            continue

        frame_num += 1

        # ── Detect ───────────────────────────────────────────────────────
        result = detector.process_frame(frame, draw=SHOW_WINDOW)
        metrics = result.metrics

        # ── Track ────────────────────────────────────────────────────────
        if metrics:
            tracker.update(metrics)
            last_metrics = metrics

            # Push to dashboard
            update_state(metrics, tracker)

            # Persist to DB (throttled to once per minute)
            if frame_num % SAVE_EVERY == 0:
                db.maybe_save_score(metrics, tracker.smooth_score)

        # ── Terminal status ───────────────────────────────────────────────
        if frame_num % PRINT_EVERY == 0 and last_metrics:
            print(f"\r{tracker.status_line()}", end="", flush=True)

        # ── Show window ───────────────────────────────────────────────────
        if SHOW_WINDOW:
            cv2.imshow("Posture Coach  (press Q to quit)", result.annotated_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        else:
            # Small sleep so we don't hammer the CPU at 100%
            time.sleep(0.033)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    print("\nSaving session...")
    db.end_session(tracker.session)
    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    detector.close()
    print(f"Session saved. Goodbye!")
    print(f"\nFinal stats:")
    print(f"  Avg score:   {tracker.session.avg_score:.1f}/100")
    print(f"  Good time:   {tracker.session.good_pct}%")
    print(f"  Bad time:    {tracker.session.bad_pct}%")
    print(f"  Alerts:      {tracker.session.alerts_fired}")
    print(f"  Duration:    {tracker.session.total_seconds/60:.1f} min")
