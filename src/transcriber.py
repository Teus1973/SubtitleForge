"""
SubtitleForge AI -- Whisper transcription via faster-whisper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from .config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_DEFAULT,
    WHISPER_MODEL_OPTIONS,
)

log = logging.getLogger(__name__)

@dataclass
class TranscriptWord:
    """Word-level timing returned by Whisper."""

    start: float
    end: float
    word: str
    probability: float | None = None


@dataclass
class TranscriptSegment:
    """A single timed subtitle segment produced by Whisper."""

    start: float   # seconds
    end: float     # seconds
    text: str      # transcript text (stripped)
    words: list[TranscriptWord] = field(default_factory=list)

    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptResult:
    """A full transcription result including detected language metadata."""

    segments: list[TranscriptSegment]
    detected_language: str | None
    language_probability: float | None
    model_size: str


class Transcriber:
    """Thin wrapper around :class:`faster_whisper.WhisperModel`.

    The model is loaded once on construction and reused across calls.
    Use :meth:`transcribe` to get a list of :class:`TranscriptSegment`.
    """

    def __init__(self, model_size: str = WHISPER_MODEL_DEFAULT) -> None:
        if model_size not in WHISPER_MODEL_OPTIONS:
            raise ValueError(
                f"Unknown model '{model_size}'. "
                f"Choose one of: {WHISPER_MODEL_OPTIONS}"
            )
        self.model_size = model_size
        self.device, self.compute_type = self._resolve_runtime()
        log.info(
            "Loading Whisper model '%s' on %s (%s)",
            model_size,
            self.device,
            self.compute_type,
        )
        self._model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _resolve_runtime(self) -> tuple[str, str]:
        if WHISPER_DEVICE == "cpu":
            return "cpu", WHISPER_COMPUTE_TYPE or "int8"
        if WHISPER_DEVICE == "cuda":
            return "cuda", WHISPER_COMPUTE_TYPE or "float16"
        if WHISPER_DEVICE != "auto":
            raise ValueError(
                f"Invalid SUBTITLEFORGE_DEVICE '{WHISPER_DEVICE}'. "
                "Expected one of: auto, cpu, cuda."
            )

        # Auto mode: prefer CUDA, but fall back cleanly on machines without it.
        try:
            import ctranslate2

            if ctranslate2.get_supported_device() == "cuda":
                return "cuda", WHISPER_COMPUTE_TYPE or "float16"
        except Exception:
            log.debug("CUDA capability probe failed; falling back to CPU", exc_info=True)
        return "cpu", WHISPER_COMPUTE_TYPE or "int8"

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        initial_prompt: str | None = None,
        word_timestamps: bool = True,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptResult:
        """Transcribe *audio_path* and return timed segments plus metadata.

        Parameters
        ----------
        audio_path:
            Path to a WAV/MP3/FLAC file (16 kHz mono WAV recommended).
        language:
            BCP-47 language code (e.g. ``"en"``, ``"he"``).  Pass ``None``
            to let Whisper auto-detect.
        initial_prompt:
            Optional text prepended to the first window to guide the model
            toward domain-specific vocabulary or script direction.
        word_timestamps:
            When ``True``, Whisper produces per-word timing that
            :mod:`src.srt_utils` can use for fine-grained subtitle splitting.
        beam_size:
            Beam search width.  Higher = more accurate but slower.
        vad_filter:
            Use Silero VAD to skip silent regions (faster, fewer hallucinations).
        """
        log.info(
            "Transcribing %s | lang=%s | model=%s | vad=%s",
            audio_path.name,
            language or "auto",
            self.model_size,
            vad_filter,
        )

        segments_iter: Iterator[Segment]
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        log.info(
            "Detected language: %s (probability %.2f)",
            info.language,
            info.language_probability,
        )

        results: list[TranscriptSegment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            words = [
                TranscriptWord(
                    start=word.start,
                    end=word.end,
                    word=word.word,
                    probability=getattr(word, "probability", None),
                )
                for word in (seg.words or [])
                if getattr(word, "word", "").strip()
            ]
            results.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                    words=words,
                )
            )

        log.info("Transcription complete: %d segments", len(results))
        return TranscriptResult(
            segments=results,
            detected_language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            model_size=self.model_size,
        )
