"""
SubtitleForge AI -- SRT formatting and RTL text correction.

Responsibilities
----------------
* Convert a list of :class:`~src.transcriber.TranscriptSegment` objects into
  a valid SRT string.
* Apply Unicode bidirectional marks so that Hebrew (and other RTL) subtitles
  render correctly in players that don't auto-detect text direction:
    - U+200F RIGHT-TO-LEFT MARK  (RLM) prepended to each line
    - U+200E LEFT-TO-RIGHT MARK  (LRM) appended after terminal punctuation
      that would otherwise be pulled to the wrong side by the Bidi algorithm.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

from .config import RTL_LANGUAGE_CODES
from .transcriber import TranscriptSegment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RLM = "\u200f"   # RIGHT-TO-LEFT MARK
LRM = "\u200e"   # LEFT-TO-RIGHT MARK

# Punctuation characters that the Unicode Bidi algorithm can misplace when
# they appear at the end of an RTL line.
_RTL_TRAILING_PUNCT_RE = re.compile(r"([.!?,;:\u2026])(\s*)$")

# Maximum characters per subtitle line before wrapping.
_MAX_LINE_CHARS = 42


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def _seconds_to_srt_time(seconds: float) -> str:
    """Convert a float number of seconds to SRT timestamp ``HH:MM:SS,mmm``."""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# RTL helpers
# ---------------------------------------------------------------------------

def apply_rtl_marks(text: str) -> str:
    """Wrap each line of *text* with Unicode bidi marks for RTL rendering.

    * Prepend RLM so the player sets paragraph direction to RTL.
    * Append LRM after trailing punctuation so it stays visually at the
      right edge of the line (its logical end) rather than floating left.
    """
    fixed_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Insert LRM before trailing punctuation so Bidi keeps it on the right.
        line = _RTL_TRAILING_PUNCT_RE.sub(r"\1" + LRM + r"\2", line)
        # Prepend RLM to force RTL paragraph direction.
        fixed_lines.append(RLM + line)
    return "\n".join(fixed_lines)


def is_rtl_language(language_code: str | None) -> bool:
    """Return ``True`` when *language_code* belongs to an RTL language."""
    if not language_code:
        return False
    normalized = language_code.lower().replace("_", "-").split("-", 1)[0]
    return normalized in RTL_LANGUAGE_CODES


# ---------------------------------------------------------------------------
# Segment → SRT block
# ---------------------------------------------------------------------------

def _wrap_text(text: str, max_chars: int = _MAX_LINE_CHARS) -> str:
    """Soft-wrap *text* at word boundaries to at most *max_chars* per line."""
    return textwrap.fill(text, width=max_chars)


def segment_to_srt_block(
    index: int,
    seg: TranscriptSegment,
    is_rtl: bool = False,
) -> str:
    """Return a single SRT block string (index + timecode + text)."""
    start_ts = _seconds_to_srt_time(seg.start)
    end_ts = _seconds_to_srt_time(seg.end)

    text = _wrap_text(seg.text)
    if is_rtl:
        text = apply_rtl_marks(text)

    return f"{index}\n{start_ts} --> {end_ts}\n{text}\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segments_to_srt(
    segments: list[TranscriptSegment],
    language_code: str | None = None,
) -> str:
    """Convert *segments* to a complete SRT file string.

    Parameters
    ----------
    segments:
        Ordered list of transcript segments (from :class:`~src.transcriber.Transcriber`).
    language_code:
        BCP-47 code of the subtitle language (e.g. ``"he"``).  When this
        matches a known RTL language, Unicode bidi marks are inserted.
    """
    is_rtl = is_rtl_language(language_code)

    blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(segment_to_srt_block(i, seg, is_rtl=is_rtl))

    return "\n".join(blocks)


def write_srt(
    segments: list[TranscriptSegment],
    output_path: Path,
    language_code: str | None = None,
) -> Path:
    """Write *segments* to *output_path* as a UTF-8 SRT file.

    Returns the path that was written.
    """
    srt_content = segments_to_srt(segments, language_code=language_code)
    output_path.write_text(srt_content, encoding="utf-8")
    return output_path
