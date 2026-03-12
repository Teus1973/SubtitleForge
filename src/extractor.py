"""
SubtitleForge AI -- FFprobe track probing + FFmpeg audio extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import ffmpeg

from .config import TEMP_DIR, channels_to_label, temp_manager

log = logging.getLogger(__name__)


@dataclass
class AudioTrack:
    """Metadata for a single audio stream inside a media container."""

    index: int
    codec: str
    language: str
    title: str
    channels: int
    display_label: str = field(init=False)

    def __post_init__(self) -> None:
        ch_label = channels_to_label(self.channels)
        lang_part = self.language if self.language else "Unknown"
        base = f"Track {self.index}: {lang_part} ({self.codec}, {ch_label})"
        self.display_label = f"{base} — {self.title}" if self.title else base


class AudioExtractor:
    """Probes video files for audio streams and extracts a selected track
    as 16 kHz mono WAV suitable for speech recognition."""

    @staticmethod
    def probe_video(video_path: Path) -> list[AudioTrack]:
        """Return metadata for every audio stream in *video_path*.

        Raises ``ffmpeg.Error`` if ffprobe fails (e.g. file not found,
        corrupt container).
        """
        info = ffmpeg.probe(str(video_path), select_streams="a")
        streams = info.get("streams", [])

        tracks: list[AudioTrack] = []
        audio_idx = 0
        for stream in streams:
            if stream.get("codec_type") != "audio":
                continue
            tags = stream.get("tags", {})
            tracks.append(
                AudioTrack(
                    index=audio_idx,
                    codec=stream.get("codec_name", "unknown"),
                    language=tags.get("language", ""),
                    title=tags.get("title", ""),
                    channels=stream.get("channels", 0),
                )
            )
            audio_idx += 1

        if not tracks:
            log.warning("No audio streams found in %s", video_path)

        return tracks

    @staticmethod
    def extract(
        video_path: Path,
        output_path: Path | None = None,
        track_index: int = 0,
    ) -> Path:
        """Extract audio track *track_index* from *video_path* as 16 kHz
        mono WAV.

        If *output_path* is ``None`` a file is created inside ``TEMP_DIR``.
        The resulting path is registered with :class:`TempManager` for
        automatic cleanup.
        """
        if output_path is None:
            output_path = TEMP_DIR / f"{video_path.stem}_track{track_index}.wav"

        log.info(
            "Extracting track %d from %s -> %s",
            track_index,
            video_path,
            output_path,
        )

        (
            ffmpeg.input(str(video_path))[f"a:{track_index}"]
            .output(
                str(output_path),
                ac=1,
                ar=16000,
                format="wav",
            )
            .overwrite_output()
            .run(quiet=True)
        )

        temp_manager.register(output_path)
        return output_path
