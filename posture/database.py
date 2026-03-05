"""
posture/database.py
-------------------
SQLite persistence.

Tables:
  sessions    — one row per work session (start, end, avg_score, stats)
  scores      — one row per minute (for heatmap + history chart)
"""

import sqlite3
import time
import os
from contextlib import contextmanager
from typing import List, Dict, Optional

DB_PATH = os.path.join(os.path.expanduser("~"), ".posture_coach", "history.db")


def _ensure_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@contextmanager
def _conn():
    _ensure_dir()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    """Create tables if they don't exist."""
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at  REAL NOT NULL,
                ended_at    REAL,
                avg_score   REAL,
                good_pct    REAL,
                warn_pct    REAL,
                bad_pct     REAL,
                alerts      INTEGER DEFAULT 0,
                duration_s  REAL
            );

            CREATE TABLE IF NOT EXISTS scores (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL,
                ts          REAL NOT NULL,
                score       REAL NOT NULL,
                status      TEXT NOT NULL,
                fwd_head    REAL,
                sho_tilt    REAL,
                spine_lean  REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_scores_ts ON scores(ts);
            CREATE INDEX IF NOT EXISTS idx_scores_session ON scores(session_id);
        """)


class Database:

    def __init__(self):
        init_db()
        self._session_id: Optional[int] = None
        self._last_score_save = 0.0
        self.SCORE_SAVE_INTERVAL = 60.0  # save one row per minute

    # ── Session lifecycle ─────────────────────────────────────────────────

    def start_session(self) -> int:
        with _conn() as con:
            cur = con.execute(
                "INSERT INTO sessions (started_at) VALUES (?)",
                (time.time(),)
            )
            self._session_id = cur.lastrowid
        return self._session_id

    def end_session(self, stats) -> None:
        if self._session_id is None:
            return
        with _conn() as con:
            con.execute("""
                UPDATE sessions
                SET ended_at=?, avg_score=?, good_pct=?, warn_pct=?,
                    bad_pct=?, alerts=?, duration_s=?
                WHERE id=?
            """, (
                time.time(),
                round(stats.avg_score, 1),
                stats.good_pct,
                round(100 * stats.warn_seconds / max(stats.total_seconds, 1), 1),
                stats.bad_pct,
                stats.alerts_fired,
                round(stats.total_seconds, 1),
                self._session_id,
            ))
        self._session_id = None

    # ── Per-minute score recording ────────────────────────────────────────

    def maybe_save_score(self, metrics, smooth_score: float) -> bool:
        """
        Save a score row at most once per minute.
        Returns True if a row was saved.
        """
        if self._session_id is None:
            return False
        now = time.time()
        if now - self._last_score_save < self.SCORE_SAVE_INTERVAL:
            return False
        self._last_score_save = now
        with _conn() as con:
            con.execute("""
                INSERT INTO scores
                    (session_id, ts, score, status, fwd_head, sho_tilt, spine_lean)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                self._session_id,
                now,
                round(smooth_score, 1),
                metrics.status,
                metrics.forward_head_angle,
                metrics.shoulder_tilt,
                metrics.spine_lean,
            ))
        return True

    # ── Read APIs for dashboard ───────────────────────────────────────────

    def get_sessions(self, limit: int = 30) -> List[Dict]:
        with _conn() as con:
            rows = con.execute("""
                SELECT * FROM sessions
                WHERE ended_at IS NOT NULL
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_scores_last_days(self, days: int = 7) -> List[Dict]:
        since = time.time() - days * 86400
        with _conn() as con:
            rows = con.execute("""
                SELECT ts, score, status, fwd_head, sho_tilt, spine_lean
                FROM scores
                WHERE ts >= ?
                ORDER BY ts ASC
            """, (since,)).fetchall()
        return [dict(r) for r in rows]

    def get_daily_averages(self, days: int = 7) -> List[Dict]:
        """One row per day: date string + avg score."""
        since = time.time() - days * 86400
        with _conn() as con:
            rows = con.execute("""
                SELECT
                    date(ts, 'unixepoch', 'localtime') as day,
                    ROUND(AVG(score), 1)               as avg_score,
                    COUNT(*)                           as samples
                FROM scores
                WHERE ts >= ?
                GROUP BY day
                ORDER BY day ASC
            """, (since,)).fetchall()
        return [dict(r) for r in rows]

    def get_hourly_heatmap(self, days: int = 7) -> List[Dict]:
        """
        Returns rows with: day_of_week (0=Mon), hour (0-23), avg_score.
        Used to build the weekly heatmap.
        """
        since = time.time() - days * 86400
        with _conn() as con:
            rows = con.execute("""
                SELECT
                    CAST(strftime('%w', ts, 'unixepoch', 'localtime') AS INTEGER) as dow,
                    CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INTEGER) as hour,
                    ROUND(AVG(score), 1) as avg_score,
                    COUNT(*) as samples
                FROM scores
                WHERE ts >= ?
                GROUP BY dow, hour
                ORDER BY dow, hour
            """, (since,)).fetchall()
        return [dict(r) for r in rows]

    def get_today_stats(self) -> Dict:
        since = time.time() - 86400
        with _conn() as con:
            row = con.execute("""
                SELECT
                    ROUND(AVG(score), 1) as avg_score,
                    COUNT(*) as samples,
                    SUM(CASE WHEN status='good' THEN 1 ELSE 0 END) as good_count,
                    SUM(CASE WHEN status='bad'  THEN 1 ELSE 0 END) as bad_count
                FROM scores WHERE ts >= ?
            """, (since,)).fetchone()
        return dict(row) if row else {}
