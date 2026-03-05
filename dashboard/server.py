"""
dashboard/server.py
-------------------
Tiny Flask server that:
  - Serves the dashboard HTML
  - Exposes JSON endpoints for live score + historical data
  - Uses Server-Sent Events (SSE) to push live updates to browser
    (no WebSocket dependency needed)
"""

import json
import time
import threading
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Shared state — written by main loop, read by Flask endpoints
_state = {
    "score":       100.0,
    "status":      "good",
    "feedback":    ["Starting up..."],
    "streak_min":  0.0,
    "session":     {},
    "detected":    False,
}
_state_lock = threading.Lock()

# Module-level db reference (set by run.py)
_db = None


def set_db(db):
    global _db
    _db = db


def update_state(metrics, tracker):
    """Called from main loop every frame."""
    with _state_lock:
        _state["score"]      = metrics.score if metrics.detected else 0
        _state["status"]     = metrics.status
        _state["feedback"]   = metrics.feedback
        _state["streak_min"] = tracker.bad_streak_minutes
        _state["detected"]   = metrics.detected
        _state["session"]    = {
            "avg_score":    round(tracker.session.avg_score, 1),
            "good_pct":     tracker.session.good_pct,
            "bad_pct":      tracker.session.bad_pct,
            "alerts":       tracker.session.alerts_fired,
            "duration_min": round(tracker.session.total_seconds / 60, 1),
        }


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/live")
def live():
    """Snapshot of current posture state."""
    with _state_lock:
        return jsonify(dict(_state))


@app.route("/api/history")
def history():
    """Last 7 days of scores for the line chart."""
    if _db is None:
        return jsonify([])
    data = _db.get_scores_last_days(7)
    # Convert unix timestamps to ISO strings for JS
    for row in data:
        row["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M", time.localtime(row["ts"]))
    return jsonify(data)


@app.route("/api/heatmap")
def heatmap():
    """Weekly heatmap data: day × hour → avg_score."""
    if _db is None:
        return jsonify([])
    return jsonify(_db.get_hourly_heatmap(7))


@app.route("/api/daily")
def daily():
    """Daily averages for the last 7 days."""
    if _db is None:
        return jsonify([])
    return jsonify(_db.get_daily_averages(7))


@app.route("/api/sessions")
def sessions():
    """Recent sessions list."""
    if _db is None:
        return jsonify([])
    data = _db.get_sessions(20)
    for s in data:
        s["started_fmt"] = time.strftime("%b %d %H:%M", time.localtime(s["started_at"]))
        s["duration_fmt"] = f"{round(s['duration_s']/60, 1)} min" if s.get("duration_s") else "—"
    return jsonify(data)


@app.route("/stream")
def stream():
    """
    Server-Sent Events endpoint.
    Browser connects once; we push a JSON update every second.
    """
    def generate():
        while True:
            with _state_lock:
                data = json.dumps(dict(_state))
            yield f"data: {data}\n\n"
            time.sleep(1.0)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


def run_server(host="127.0.0.1", port=5000, debug=False):
    """Run Flask in a daemon thread so it doesn't block the main loop."""
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=debug,
                               use_reloader=False, threaded=True),
        daemon=True,
    )
    t.start()
    return t
