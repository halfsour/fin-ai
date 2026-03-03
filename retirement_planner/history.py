"""Simple file-based conversation history for the Retirement Planner.

Saves session data (profile, assessments, conversation exchanges) as JSON
files in ~/.retirement_planner/sessions/.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_SESSIONS_DIR = Path.home() / ".retirement_planner" / "sessions"


def _ensure_dir() -> Path:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSIONS_DIR


def new_session_id() -> str:
    """Generate a session ID based on the current timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_session(session_id: str, data: dict) -> Path:
    """Save or update a session file.

    Args:
        session_id: The session identifier.
        data: The session data dict to persist.

    Returns:
        Path to the saved session file.
    """
    _ensure_dir()
    path = _SESSIONS_DIR / f"{session_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_session(session_id: str) -> dict:
    """Load a session file by ID.

    Args:
        session_id: The session identifier.

    Returns:
        The session data dict.

    Raises:
        FileNotFoundError: If the session file doesn't exist.
    """
    path = _SESSIONS_DIR / f"{session_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> list[dict]:
    """List all saved sessions, most recent first.

    Returns:
        List of dicts with session_id, created_at, and summary info.
    """
    _ensure_dir()
    sessions = []
    for f in sorted(_SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            sessions.append({
                "session_id": f.stem,
                "created_at": data.get("created_at", "unknown"),
                "can_retire": data.get("assessment", {}).get("can_retire"),
                "net_worth": data.get("assessment", {}).get("net_worth"),
                "exchanges": len(data.get("conversation", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return sessions


def get_latest_session_id() -> str | None:
    """Return the most recent session ID, or None if no sessions exist."""
    _ensure_dir()
    files = sorted(_SESSIONS_DIR.glob("*.json"), reverse=True)
    return files[0].stem if files else None
