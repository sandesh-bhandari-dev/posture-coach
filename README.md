# 🪑 Posture Coach

Real-time posture monitoring using your webcam + MediaPipe.

## Setup

### 1. Install dependencies
```bash
pip install opencv-python mediapipe flask plyer numpy
```
> `plyer` is optional (OS notifications) — works without it.

### 2. Run
```bash
python run.py
```

Opens webcam + dashboard at http://localhost:5000

## Options
```
--no-window     headless mode (no webcam popup)
--no-browser    don't auto-open browser
--cam N         use camera index N (default 0)
--alert N       alert after N bad minutes (default 10)
--port N        dashboard port (default 5000)
```

## Troubleshooting
- **Camera won't open** → try `--cam 1` or `--cam 2`
- **mediapipe fails** → requires Python 3.8–3.11 (not 3.12+)
- **Dashboard broken** → make sure `index.html` is in `dashboard/templates/`

## What it tracks
- Forward head angle
- Shoulder tilt
- Spine lean
- Neck compression

Score 0–100. Alerts after N minutes of continuous bad posture.
