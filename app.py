"""
SubtitleForge AI -- Streamlit entry point.

Run with:
    E:\\work\\SubtitleForge\\venv\\Scripts\\python.exe -m streamlit run app.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st

from src.config import (
    SOURCE_LANGUAGES,
    TEMP_DIR,
    WHISPER_MODEL_DEFAULT,
    WHISPER_MODEL_OPTIONS,
    temp_manager,
)
from src.extractor import AudioExtractor, AudioTrack
from src.srt_utils import write_srt
from src.syncer import SubtitleSyncer
from src.transcriber import Transcriber

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _get_transcriber(model_size: str) -> Transcriber:
    """Reuse loaded Whisper models across Streamlit reruns."""
    return Transcriber(model_size=model_size)

# ---------------------------------------------------------------------------
# Startup: sweep orphaned temp files older than 24 h
# ---------------------------------------------------------------------------
temp_manager.cleanup_stale(max_age_hours=24)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SubtitleForge AI",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar -- language + model configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Language Settings")

    source_lang_label = st.selectbox(
        "Source Language",
        options=list(SOURCE_LANGUAGES.keys()),
        index=0,
        help="Language spoken in the video. 'Auto-detect' lets Whisper decide.",
    )
    source_lang_code: str | None = SOURCE_LANGUAGES[source_lang_label]

    st.header("Model Settings")

    model_size = st.selectbox(
        "Whisper Model",
        options=WHISPER_MODEL_OPTIONS,
        index=WHISPER_MODEL_OPTIONS.index(WHISPER_MODEL_DEFAULT),
        help=(
            "Larger models are more accurate but slower and require more RAM/VRAM. "
            "'small' is a good starting point."
        ),
    )

    initial_prompt = st.text_area(
        "Initial Prompt (optional)",
        placeholder="e.g. 'Technical interview in Hebrew about software engineering.'",
        help=(
            "Seed text to guide Whisper's vocabulary and style. "
            "Useful for domain-specific terms or to reinforce script direction."
        ),
        height=80,
    )

    st.divider()
    st.caption(
        "**SubtitleForge AI** — local subtitle generation & sync.  "
        "No paid APIs. Runs entirely on your hardware."
    )
    st.caption(
        "Phase 2 transcribes the spoken language only. Translation is not "
        "implemented yet, so subtitle direction follows the source/detected language."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("SubtitleForge AI")
st.markdown(
    "Generate new subtitles from video or synchronize an existing misaligned "
    "SRT file — all locally, using open-source AI."
)

# -- Video input (local path only — no browser upload for large files) -----
st.subheader("1. Select Video")

st.caption(
    "Enter the full path to your video file on this machine. "
    "The file is read directly from disk — nothing is uploaded through the browser."
)

col_path, col_btn = st.columns([5, 1])

with col_path:
    local_video_path_str = st.text_input(
        "Full path to video file",
        placeholder=r"C:\Videos\movie.mkv  or  E:\Shows\episode.mkv",
        key="video_path_input",
        label_visibility="collapsed",
    )

with col_btn:
    load_clicked = st.button("Load Video", use_container_width=True)

# Persist a validated path in session state so it survives reruns
# without the user having to click "Load" again.
if load_clicked:
    raw = local_video_path_str.strip()
    if not raw:
        st.error("Please enter a file path first.")
        st.session_state["video_path_validated"] = None
    elif not os.path.exists(raw):
        st.error(f"File not found: `{raw}`")
        st.session_state["video_path_validated"] = None
    elif not os.path.isfile(raw):
        st.error(f"Path exists but is not a file: `{raw}`")
        st.session_state["video_path_validated"] = None
    else:
        st.session_state["video_path_validated"] = raw
        st.success(f"Loaded: `{os.path.basename(raw)}`")

video_path: Path | None = None
validated = st.session_state.get("video_path_validated")
if validated:
    video_path = Path(validated)

# -- Audio track probing & selection ----------------------------------------
st.subheader("2. Select Audio Track")

audio_tracks: list[AudioTrack] = []
selected_track_index: int = 0

if video_path is not None:
    try:
        with st.spinner("Probing audio tracks…"):
            audio_tracks = AudioExtractor.probe_video(video_path)
    except Exception as exc:
        st.error(f"Failed to probe video: {exc}")
        audio_tracks = []

if audio_tracks:
    track_labels = [t.display_label for t in audio_tracks]

    if len(audio_tracks) == 1:
        st.info(f"Single audio track detected: **{track_labels[0]}**")
        selected_track_index = audio_tracks[0].index
    else:
        chosen_label = st.selectbox(
            "Choose audio track for transcription",
            options=track_labels,
        )
        selected_track_index = audio_tracks[track_labels.index(chosen_label)].index
elif video_path is not None:
    st.warning("No audio tracks found in this file.")

# -- Optional SRT upload ---------------------------------------------------
st.subheader("3. Optional — Upload Existing SRT to Sync")
uploaded_srt = st.file_uploader(
    "Upload an SRT file to sync (a few KB — safe to upload via browser). "
    "Leave empty to generate new subtitles from scratch.",
    type=["srt"],
    key="srt_uploader",
)

# -- Process button ---------------------------------------------------------
st.subheader("4. Process")

can_process = video_path is not None and len(audio_tracks) > 0

if st.button("🚀 Start Processing", disabled=not can_process, use_container_width=True):
    output_srt_path = TEMP_DIR / f"{video_path.stem}_subtitles.srt"  # type: ignore[union-attr]

    # ---- Mode: Sync existing SRT ----------------------------------------
    if uploaded_srt is not None:
        input_srt_path = TEMP_DIR / uploaded_srt.name
        input_srt_path.write_bytes(uploaded_srt.getbuffer())
        temp_manager.register(input_srt_path)

        with st.spinner(f"Syncing '{uploaded_srt.name}' to audio track {selected_track_index}…"):
            try:
                syncer = SubtitleSyncer()
                syncer.sync(
                    reference_video=video_path,  # type: ignore[arg-type]
                    input_srt=input_srt_path,
                    output_srt=output_srt_path,
                    reference_stream_index=selected_track_index,
                )
                st.success("Sync complete!")
            except Exception as exc:
                st.error(f"Sync failed: {exc}")
                log.exception("Sync error")
                st.stop()

    # ---- Mode: Generate new subtitles ------------------------------------
    else:
        # Step 1: extract audio
        with st.spinner(f"Extracting audio track {selected_track_index}…"):
            try:
                audio_path = AudioExtractor.extract(
                    video_path=video_path,  # type: ignore[arg-type]
                    track_index=selected_track_index,
                )
            except Exception as exc:
                st.error(f"Audio extraction failed: {exc}")
                log.exception("Extraction error")
                st.stop()

        # Step 2: transcribe
        with st.spinner(
            f"Transcribing with Whisper '{model_size}' "
            f"(language: {source_lang_label})…  This may take a few minutes."
        ):
            try:
                transcriber = _get_transcriber(model_size)
                transcript = transcriber.transcribe(
                    audio_path=audio_path,
                    language=source_lang_code,
                    initial_prompt=initial_prompt.strip() or None,
                    word_timestamps=True,
                    vad_filter=True,
                )
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
                log.exception("Transcription error")
                st.stop()

        if not transcript.segments:
            st.warning("No speech detected in the selected audio track.")
            st.stop()

        subtitle_language_code = source_lang_code or transcript.detected_language

        # Step 3: write SRT
        with st.spinner("Writing SRT file…"):
            try:
                write_srt(
                    segments=transcript.segments,
                    output_path=output_srt_path,
                    language_code=subtitle_language_code,
                )
            except Exception as exc:
                st.error(f"Failed to write SRT: {exc}")
                log.exception("SRT write error")
                st.stop()

        detected_label = transcript.detected_language or "unknown"
        st.success(
            f"Transcription complete! {len(transcript.segments)} subtitle segments "
            f"generated. Detected language: {detected_label}."
        )

    # ---- Download button (both modes) ------------------------------------
    if output_srt_path.is_file():
        srt_bytes = output_srt_path.read_bytes()
        st.download_button(
            label="⬇️ Download SRT",
            data=srt_bytes,
            file_name=output_srt_path.name,
            mime="text/plain",
            use_container_width=True,
        )

        # Preview first 10 blocks
        with st.expander("Preview subtitles (first 10 blocks)", expanded=True):
            preview_lines = output_srt_path.read_text(encoding="utf-8").split("\n\n")
            st.code("\n\n".join(preview_lines[:10]), language=None)
