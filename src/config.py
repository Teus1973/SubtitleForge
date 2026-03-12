"""
SubtitleForge AI -- configuration constants and workspace cleanup.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Whisper model
# ---------------------------------------------------------------------------
# Runtime device selection:
# - "auto": try CUDA first, fall back to CPU if unavailable
# - "cuda": force GPU
# - "cpu": force CPU
WHISPER_DEVICE = os.environ.get("SUBTITLEFORGE_DEVICE", "auto").strip().lower()
WHISPER_COMPUTE_TYPE = os.environ.get("SUBTITLEFORGE_COMPUTE_TYPE", "").strip().lower() or None

# Ordered from fastest/smallest to slowest/most accurate.
# Shown in the sidebar selectbox; the UI default is "small".
WHISPER_MODEL_OPTIONS: list[str] = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
]
WHISPER_MODEL_DEFAULT = "small"

# ---------------------------------------------------------------------------
# Language maps
# ---------------------------------------------------------------------------
SOURCE_LANGUAGES: dict[str, str | None] = {
    "Auto-detect": None,
    "English": "en",
    "Hebrew": "he",
    "Italian": "it",
    "Spanish": "es",
    "Korean": "ko",
    "Japanese": "ja",
    "French": "fr",
}

TARGET_LANGUAGES: dict[str, str] = {
    "English": "en",
    "Hebrew": "he",
}

RTL_LANGUAGE_CODES: set[str] = {"he", "ar"}

# ---------------------------------------------------------------------------
# Channel-count labels (used by AudioExtractor display_label)
# ---------------------------------------------------------------------------
CHANNEL_LABELS: dict[int, str] = {
    1: "mono",
    2: "stereo",
    6: "5.1",
    8: "7.1",
}


def channels_to_label(n: int) -> str:
    return CHANNEL_LABELS.get(n, f"{n}ch")


# ---------------------------------------------------------------------------
# TempManager -- session-scoped workspace cleanup
# ---------------------------------------------------------------------------
class TempManager:
    """Tracks temporary files created during a pipeline run and deletes them
    on demand.  Integrates with Streamlit ``session_state`` when available,
    falling back to an internal set otherwise."""

    _SESSION_KEY = "_subtitleforge_temp_files"

    def __init__(self) -> None:
        self._fallback: set[str] = set()

    # -- internal helpers ---------------------------------------------------

    def _get_store(self) -> set[str]:
        """Return the set that holds registered paths, preferring
        ``st.session_state`` so tracked files survive Streamlit reruns."""
        try:
            import streamlit as st

            if self._SESSION_KEY not in st.session_state:
                st.session_state[self._SESSION_KEY] = set()
            return st.session_state[self._SESSION_KEY]
        except Exception:
            return self._fallback

    # -- public API ---------------------------------------------------------

    def register(self, path: Path) -> None:
        """Add *path* to the tracked set so it will be deleted on cleanup."""
        self._get_store().add(str(path))

    def cleanup(self) -> None:
        """Delete every registered temp file and clear the tracked set."""
        store = self._get_store()
        for p in list(store):
            try:
                fp = Path(p)
                if fp.is_file():
                    fp.unlink()
                    log.debug("TempManager: deleted %s", fp)
            except OSError as exc:
                log.warning("TempManager: failed to delete %s: %s", p, exc)
        store.clear()

    def cleanup_stale(self, max_age_hours: int = 24) -> None:
        """Scan *TEMP_DIR* for files older than *max_age_hours* and remove
        them.  Intended as a startup safety-net for orphaned files."""
        cutoff = time.time() - max_age_hours * 3600
        if not TEMP_DIR.is_dir():
            return
        for entry in TEMP_DIR.iterdir():
            if entry.name == ".gitignore":
                continue
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    entry.unlink()
                    log.info("TempManager: stale cleanup removed %s", entry)
            except OSError as exc:
                log.warning("TempManager: stale cleanup error for %s: %s", entry, exc)


temp_manager = TempManager()
