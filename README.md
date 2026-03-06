# Posture Coach

![Python](https://img.shields.io/badge/python-3.11-blue) ![Flask](https://img.shields.io/badge/flask-3.1-lightgrey) ![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.14-orange)

This project is a real-time posture monitor that uses your webcam to track how you are sitting and gives you a live score. It watches your head position, shoulder alignment, and spine lean using MediaPipe pose detection. If you have been sitting with bad posture for too long it will send you an alert notification.

## Demo

![Posture Coach Dashboard](demo.png)

## What it tracks

Each frame your webcam captures is run through MediaPipe to find your body landmarks. Four measurements are taken from those landmarks and combined into a score from 0 to 100.

- Forward head angle measures how far your head is jutting forward compared to your shoulders
- Shoulder tilt measures the height difference between your left and right shoulder
- Spine lean measures whether your upper body is shifted sideways over your hips
- Neck compression measures whether your head is drooping down toward your chest

The score is a weighted average of all four. If your score stays below 60 for too long you get a desktop notification and a sound alert.

## Files

```
run.py                      entry point, opens webcam and starts everything
posture/detector.py         MediaPipe pose detection and all the posture math
posture/tracker.py          rolling averages, streak timing, alert logic
posture/database.py         saves scores and sessions to a local SQLite database
posture/alerts.py           cross-platform desktop notifications and beep sounds
dashboard/server.py         Flask server that streams live data to the browser
dashboard/templates/        the dashboard UI
```

## How to Run

This project requires Python 3.11. MediaPipe does not support Python 3.12 or newer yet.

First clone the repo and go into the folder.

```
git clone https://github.com/sandesh-bhandari-dev/posture-coach.git
cd posture-coach
```

Then install the required libraries. Make sure you are using Python 3.11 to do this.

```
python3.11 -m pip install opencv-python mediapipe==0.10.14 flask plyer numpy
```

On Windows if you have multiple Python versions installed run it like this instead.

```
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe -m pip install opencv-python mediapipe==0.10.14 flask plyer numpy
```

Then run it.

```
python3.11 run.py
```

On Windows with multiple Python versions.

```
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe run.py
```

The webcam window will open and the dashboard will launch automatically at http://localhost:5000. Press Q in the webcam window or Ctrl+C to stop.

## Options

```
--no-window     run without the webcam popup
--no-browser    do not auto open the browser
--cam N         use a different camera, default is 0
--alert N       send alert after N minutes of bad posture, default is 10
--port N        run the dashboard on a different port, default is 5000
```

## What to Expect

When you first run it MediaPipe needs a second to detect your pose. Once detected your score will update in real time. The heatmap and history chart build up over multiple sessions. Your session data is stored locally at ~/.posture_coach/history.db so nothing leaves your machine.
