"""
SubtitleForge AI -- ffsubsync wrapper for subtitle alignment.

ffsubsync analyses the speech fingerprint of the reference video/audio and
shifts (and optionally rescales) the input SRT so that dialogue lines land
on the correct frames.

We drive it programmatically by temporarily patching ``sys.argv`` and calling
its ``main()`` entry-point, which is the same code path used by the CLI.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SubtitleSyncer:
    """Aligns an existing SRT file to a video/audio reference using ffsubsync.

    Usage::

        syncer = SubtitleSyncer()
        synced_path = syncer.sync(
            reference_video=Path("movie.mkv"),
            input_srt=Path("movie.srt"),
            output_srt=Path("movie_synced.srt"),
            reference_stream_index=0,   # audio track index
        )
    """

    @staticmethod
    def sync(
        reference_video: Path,
        input_srt: Path,
        output_srt: Path,
        reference_stream_index: int = 0,
        max_offset_seconds: float = 60.0,
        vad: str = "auditok",
        use_gss: bool = False,
    ) -> Path:
        """Synchronise *input_srt* to *reference_video* and write *output_srt*.

        Parameters
        ----------
        reference_video:
            The video (or audio) file whose speech timing is used as ground
            truth.
        input_srt:
            The misaligned subtitle file to fix.
        output_srt:
            Destination path for the corrected SRT.
        reference_stream_index:
            Zero-based audio track index inside *reference_video* (maps to
            ffsubsync's ``--reference-stream a:<n>`` flag).
        max_offset_seconds:
            Maximum shift ffsubsync is allowed to apply.  Increase if the
            subtitles are very far off.
        vad:
            Voice-activity detector backend.  ``"auditok"`` is pure-Python
            and works without extra system libs; ``"webrtc"`` is faster on
            long files.
        use_gss:
            Enable golden-section search for framerate-ratio correction.
            Useful when the SRT was authored for a different frame rate.
        """
        log.info(
            "Syncing '%s' against '%s' (audio track %d)",
            input_srt.name,
            reference_video.name,
            reference_stream_index,
        )

        argv: list[str] = [
            str(reference_video),
            "-i", str(input_srt),
            "-o", str(output_srt),
            "--reference-stream", f"a:{reference_stream_index}",
            "--max-offset-seconds", str(max_offset_seconds),
            "--vad", vad,
            "--output-encoding", "utf-8",
        ]
        if use_gss:
            argv.append("--gss")

        log.debug("ffsubsync argv: %s", argv)

        from ffsubsync.ffsubsync import make_parser, run

        output_srt.parent.mkdir(parents=True, exist_ok=True)
        parser = make_parser()
        args = parser.parse_args(argv)
        result = run(args)
        exit_code = int(result.get("retval", 1))

        if exit_code != 0:
            raise RuntimeError(
                f"ffsubsync exited with code {exit_code}. "
                "The subtitle file may be too far out of sync or in an "
                "unsupported format."
            )

        if not output_srt.is_file():
            raise FileNotFoundError(
                f"ffsubsync did not produce an output file at: {output_srt}"
            )

        log.info("Sync complete -> %s", output_srt)
        return output_srt
