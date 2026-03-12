"""
Phase 1 tests — AudioExtractor (probe + extract) and config helpers.

Run with:
    E:\\work\\SubtitleForge\\venv\\Scripts\\python.exe -m pytest tests/test_phase1.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `src` is importable without
# installing the package.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TEMP_DIR,
    channels_to_label,
    TempManager,
)
from src.extractor import AudioExtractor, AudioTrack

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_VIDEO = PROJECT_ROOT / "temp" / "test_multitrack.mkv"


@pytest.fixture(scope="session")
def test_video() -> Path:
    if not TEST_VIDEO.is_file():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")
    return TEST_VIDEO


# ---------------------------------------------------------------------------
# config.py tests
# ---------------------------------------------------------------------------


class TestChannelsToLabel:
    def test_mono(self):
        assert channels_to_label(1) == "mono"

    def test_stereo(self):
        assert channels_to_label(2) == "stereo"

    def test_surround_51(self):
        assert channels_to_label(6) == "5.1"

    def test_surround_71(self):
        assert channels_to_label(8) == "7.1"

    def test_unknown_channel_count(self):
        assert channels_to_label(3) == "3ch"
        assert channels_to_label(12) == "12ch"


class TestTempDir:
    def test_temp_dir_exists(self):
        assert TEMP_DIR.is_dir(), f"TEMP_DIR does not exist: {TEMP_DIR}"


class TestTempManager:
    def test_register_and_cleanup(self, tmp_path):
        mgr = TempManager()
        dummy = tmp_path / "dummy.wav"
        dummy.write_bytes(b"test")
        assert dummy.is_file()

        mgr.register(dummy)
        mgr.cleanup()

        assert not dummy.is_file(), "TempManager.cleanup() should delete registered files"

    def test_cleanup_missing_file_does_not_raise(self, tmp_path):
        mgr = TempManager()
        ghost = tmp_path / "ghost.wav"
        mgr.register(ghost)
        mgr.cleanup()  # should not raise even though file never existed


# ---------------------------------------------------------------------------
# AudioTrack dataclass tests
# ---------------------------------------------------------------------------


class TestAudioTrack:
    def test_display_label_with_title(self):
        t = AudioTrack(index=0, codec="aac", language="eng", title="English Dialogue", channels=2)
        assert "Track 0" in t.display_label
        assert "eng" in t.display_label
        assert "aac" in t.display_label
        assert "stereo" in t.display_label
        assert "English Dialogue" in t.display_label

    def test_display_label_without_title(self):
        t = AudioTrack(index=1, codec="ac3", language="heb", title="", channels=6)
        assert "Track 1" in t.display_label
        assert "heb" in t.display_label
        assert "5.1" in t.display_label
        assert "—" not in t.display_label

    def test_display_label_unknown_language(self):
        t = AudioTrack(index=0, codec="mp3", language="", title="", channels=2)
        assert "Unknown" in t.display_label


# ---------------------------------------------------------------------------
# AudioExtractor.probe_video tests
# ---------------------------------------------------------------------------


class TestProbeVideo:
    def test_returns_list_of_audio_tracks(self, test_video):
        tracks = AudioExtractor.probe_video(test_video)
        assert isinstance(tracks, list)
        assert len(tracks) > 0, "Expected at least one audio track"

    def test_track_count_matches_file(self, test_video):
        tracks = AudioExtractor.probe_video(test_video)
        assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"

    def test_track_zero_is_english(self, test_video):
        tracks = AudioExtractor.probe_video(test_video)
        assert tracks[0].language == "eng"
        assert tracks[0].title == "English Dialogue"

    def test_track_one_is_hebrew(self, test_video):
        tracks = AudioExtractor.probe_video(test_video)
        assert tracks[1].language == "heb"
        assert tracks[1].title == "Hebrew Dub"

    def test_all_tracks_have_codec(self, test_video):
        tracks = AudioExtractor.probe_video(test_video)
        for t in tracks:
            assert t.codec, f"Track {t.index} has no codec"

    def test_track_indices_are_sequential(self, test_video):
        tracks = AudioExtractor.probe_video(test_video)
        for i, t in enumerate(tracks):
            assert t.index == i

    def test_probe_nonexistent_file_raises(self):
        with pytest.raises(Exception):
            AudioExtractor.probe_video(Path("nonexistent_file.mkv"))


# ---------------------------------------------------------------------------
# AudioExtractor.extract tests
# ---------------------------------------------------------------------------


class TestExtractAudio:
    def test_extract_track_0_produces_wav(self, test_video, tmp_path):
        out = tmp_path / "out_track0.wav"
        result = AudioExtractor.extract(test_video, output_path=out, track_index=0)
        assert result == out
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_extract_track_1_produces_wav(self, test_video, tmp_path):
        out = tmp_path / "out_track1.wav"
        result = AudioExtractor.extract(test_video, output_path=out, track_index=1)
        assert result.is_file()
        assert result.stat().st_size > 0

    def test_extract_default_output_path(self, test_video):
        """When output_path is None, a file should be created in TEMP_DIR."""
        result = AudioExtractor.extract(test_video, output_path=None, track_index=0)
        try:
            assert result.is_file()
            assert result.parent == TEMP_DIR
            assert result.suffix == ".wav"
        finally:
            if result.is_file():
                result.unlink()

    def test_extracted_wav_is_16khz_mono(self, test_video, tmp_path):
        """Verify the output WAV has the correct sample rate and channel count."""
        import ffmpeg as _ffmpeg

        out = tmp_path / "check.wav"
        AudioExtractor.extract(test_video, output_path=out, track_index=0)
        info = _ffmpeg.probe(str(out))
        audio_streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
        assert len(audio_streams) == 1
        stream = audio_streams[0]
        assert int(stream["sample_rate"]) == 16000, "Expected 16 kHz sample rate"
        assert int(stream["channels"]) == 1, "Expected mono (1 channel)"
