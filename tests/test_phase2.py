"""
Phase 2 tests — transcription metadata, SRT formatting, and subtitle sync.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.srt_utils import LRM, RLM, apply_rtl_marks, is_rtl_language, segments_to_srt
from src.syncer import SubtitleSyncer
from src.transcriber import TranscriptSegment, Transcriber


class TestSrtUtils:
    def test_is_rtl_language_handles_regional_code(self):
        assert is_rtl_language("he-IL") is True
        assert is_rtl_language("ar_EG") is True
        assert is_rtl_language("en-US") is False

    def test_apply_rtl_marks_keeps_terminal_punctuation(self):
        text = "שלום עולם!"
        fixed = apply_rtl_marks(text)
        assert fixed.startswith(RLM)
        assert f"!{LRM}" in fixed

    def test_segments_to_srt_applies_rtl_for_detected_hebrew(self):
        segments = [TranscriptSegment(start=0.0, end=1.5, text="שלום עולם!")]
        srt = segments_to_srt(segments, language_code="he-IL")
        assert "00:00:00,000 --> 00:00:01,500" in srt
        assert RLM in srt
        assert f"!{LRM}" in srt


class TestTranscriber:
    def test_auto_mode_falls_back_to_cpu_when_cuda_unavailable(self, monkeypatch):
        import src.transcriber as transcriber_mod

        monkeypatch.setattr(transcriber_mod, "WHISPER_DEVICE", "auto")
        monkeypatch.setattr(transcriber_mod, "WHISPER_COMPUTE_TYPE", None)
        monkeypatch.setitem(
            sys.modules,
            "ctranslate2",
            SimpleNamespace(get_supported_device=lambda: "cpu"),
        )

        captured: dict[str, str] = {}

        class FakeWhisperModel:
            def __init__(self, model_size: str, device: str, compute_type: str):
                captured["model_size"] = model_size
                captured["device"] = device
                captured["compute_type"] = compute_type

        monkeypatch.setattr(transcriber_mod, "WhisperModel", FakeWhisperModel)

        transcriber = Transcriber(model_size="small")
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"
        assert captured["device"] == "cpu"
        assert captured["compute_type"] == "int8"

    def test_transcribe_preserves_word_timestamps(self, monkeypatch, tmp_path):
        import src.transcriber as transcriber_mod

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake")

        fake_info = SimpleNamespace(language="he", language_probability=0.98)
        fake_segments = [
            SimpleNamespace(
                start=0.0,
                end=1.2,
                text=" שלום עולם ",
                words=[
                    SimpleNamespace(start=0.0, end=0.4, word="שלום", probability=0.91),
                    SimpleNamespace(start=0.45, end=1.0, word="עולם", probability=0.88),
                ],
            )
        ]

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, *args, **kwargs):
                return iter(fake_segments), fake_info

        monkeypatch.setattr(transcriber_mod, "WHISPER_DEVICE", "cpu")
        monkeypatch.setattr(transcriber_mod, "WHISPER_COMPUTE_TYPE", "int8")
        monkeypatch.setattr(transcriber_mod, "WhisperModel", FakeWhisperModel)

        result = Transcriber(model_size="small").transcribe(audio_path=audio_path, language=None)

        assert result.detected_language == "he"
        assert result.language_probability == pytest.approx(0.98)
        assert len(result.segments) == 1
        assert result.segments[0].text == "שלום עולם"
        assert len(result.segments[0].words) == 2
        assert result.segments[0].words[0].word == "שלום"
        assert result.segments[0].words[1].end == pytest.approx(1.0)


class TestSyncer:
    def test_sync_uses_ffsubsync_run_without_touching_sys_argv(self, monkeypatch, tmp_path):
        import ffsubsync.ffsubsync as ffsubsync_module

        reference_video = tmp_path / "movie.mkv"
        input_srt = tmp_path / "input.srt"
        output_srt = tmp_path / "output.srt"
        reference_video.write_bytes(b"video")
        input_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

        original_argv = sys.argv[:]
        captured: dict[str, object] = {}

        class FakeParser:
            def parse_args(self, argv):
                captured["argv"] = argv
                return SimpleNamespace(srtout=str(output_srt))

        def fake_run(args):
            Path(args.srtout).write_text("synced", encoding="utf-8")
            return {"retval": 0}

        monkeypatch.setattr(ffsubsync_module, "make_parser", lambda: FakeParser())
        monkeypatch.setattr(ffsubsync_module, "run", fake_run)

        result = SubtitleSyncer.sync(
            reference_video=reference_video,
            input_srt=input_srt,
            output_srt=output_srt,
            reference_stream_index=1,
        )

        assert result == output_srt
        assert output_srt.read_text(encoding="utf-8") == "synced"
        assert sys.argv == original_argv
        assert captured["argv"] == [
            str(reference_video),
            "-i",
            str(input_srt),
            "-o",
            str(output_srt),
            "--reference-stream",
            "a:1",
            "--max-offset-seconds",
            "60.0",
            "--vad",
            "auditok",
            "--output-encoding",
            "utf-8",
        ]

    def test_sync_raises_on_nonzero_exit(self, monkeypatch, tmp_path):
        import ffsubsync.ffsubsync as ffsubsync_module

        reference_video = tmp_path / "movie.mkv"
        input_srt = tmp_path / "input.srt"
        output_srt = tmp_path / "output.srt"
        reference_video.write_bytes(b"video")
        input_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

        class FakeParser:
            def parse_args(self, argv):
                return SimpleNamespace(srtout=str(output_srt))

        monkeypatch.setattr(ffsubsync_module, "make_parser", lambda: FakeParser())
        monkeypatch.setattr(ffsubsync_module, "run", lambda args: {"retval": 1})

        with pytest.raises(RuntimeError):
            SubtitleSyncer.sync(reference_video, input_srt, output_srt)
