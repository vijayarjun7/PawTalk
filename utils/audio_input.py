"""
utils/audio_input.py
--------------------
Centralised audio-input widget for PawTalk.

Each tab calls `get_audio_input(tab_key, cache_clear_fn, labels)` and
receives either `(audio_bytes, cache_key)` or `(None, None)`.

The function renders a mode selector ("Upload" / "Record") and the
appropriate input widget.  All session-state bookkeeping lives here so
app.py stays thin.

streamlit-mic-recorder is an optional dependency.  When it is absent the
mode selector is hidden and only the file uploader is shown, with a
gentle note explaining why recording is unavailable.
"""

from __future__ import annotations

import hashlib
import io

import streamlit as st

# ── optional import ────────────────────────────────────────────────────────────
try:
    from streamlit_mic_recorder import mic_recorder  # type: ignore
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bytes_key(data: bytes, prefix: str) -> str:
    """Return a stable cache key for raw audio bytes."""
    digest = hashlib.blake2b(data[:65_536], digest_size=16).hexdigest()
    return f"{prefix}{digest}"


def _file_key(f, prefix: str) -> str:
    """Return a stable cache key for a Streamlit UploadedFile object."""
    f.seek(0)
    header = f.read(65_536)
    f.seek(0)
    return f"{prefix}{hashlib.blake2b(header, digest_size=16).hexdigest()}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_audio_input(
    *,
    tab_key: str,
    cache_prefix: str,
    cache_clear_fn,
    upload_label: str = "Upload audio",
    upload_help: str = "WAV recommended. 2–10 seconds works best.",
    empty_prompt_icon: str = "🎙️",
    empty_prompt_text: str = "Waiting for audio…",
    empty_prompt_sub: str = "Upload a WAV, MP3, OGG, or FLAC file above.",
    record_button_label: str = "🎙️ Start recording",
    record_stop_label: str = "⏹️ Stop",
) -> tuple[bytes | None, str | None]:
    """
    Render an audio-input section (upload or record) and return
    ``(audio_bytes, cache_key)`` when audio is ready, or ``(None, None)``.

    Parameters
    ----------
    tab_key         Unique prefix for all widget keys in this tab.
    cache_prefix    Prefix for the session-state cache key, e.g. "bark_result_".
    cache_clear_fn  Zero-arg callable that clears old cached results for this tab.
    upload_label    Label shown above the file uploader.
    upload_help     Tooltip text for the file uploader.
    empty_prompt_*  Strings for the "nothing uploaded yet" placeholder.
    record_*        Strings for the recording button labels.

    Returns
    -------
    (audio_bytes, cache_key)
        audio_bytes  Raw bytes ready for ``audio_features.load_audio_from_bytes``.
        cache_key    Stable string key for ``st.session_state``.
    Both are ``None`` when no audio has been provided yet.
    """

    mode_key   = f"{tab_key}_input_mode"
    rec_key    = f"{tab_key}_last_recording"   # stores bytes of most recent recording

    # ── Mode selector (only when mic is available) ────────────────────────────
    if MIC_AVAILABLE:
        mode = st.radio(
            "Input method",
            options=["Upload a file", "Record in browser"],
            key=mode_key,
            horizontal=True,
            label_visibility="collapsed",
        )
    else:
        mode = "Upload a file"

    # ── Branch on mode ────────────────────────────────────────────────────────
    if mode == "Upload a file":
        return _upload_mode(
            tab_key=tab_key,
            cache_prefix=cache_prefix,
            cache_clear_fn=cache_clear_fn,
            upload_label=upload_label,
            upload_help=upload_help,
            empty_prompt_icon=empty_prompt_icon,
            empty_prompt_text=empty_prompt_text,
            empty_prompt_sub=empty_prompt_sub,
        )
    else:
        return _record_mode(
            tab_key=tab_key,
            cache_prefix=cache_prefix,
            cache_clear_fn=cache_clear_fn,
            rec_key=rec_key,
            record_button_label=record_button_label,
            record_stop_label=record_stop_label,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Upload mode
# ─────────────────────────────────────────────────────────────────────────────

def _upload_mode(
    *,
    tab_key: str,
    cache_prefix: str,
    cache_clear_fn,
    upload_label: str,
    upload_help: str,
    empty_prompt_icon: str,
    empty_prompt_text: str,
    empty_prompt_sub: str,
) -> tuple[bytes | None, str | None]:

    uploaded = st.file_uploader(
        upload_label,
        type=["wav", "mp3", "ogg", "flac"],
        help=upload_help,
        key=f"{tab_key}_uploader",
        on_change=cache_clear_fn,
    )

    if uploaded is None:
        _empty_prompt(empty_prompt_icon, empty_prompt_text, empty_prompt_sub)
        return None, None

    uploaded.seek(0)
    audio_bytes = uploaded.read()
    uploaded.seek(0)

    st.audio(uploaded)
    st.caption(f"📁 {uploaded.name}  ·  {len(audio_bytes) / 1024:.1f} KB")

    cache_key = _bytes_key(audio_bytes, cache_prefix)
    return audio_bytes, cache_key


# ─────────────────────────────────────────────────────────────────────────────
# Record mode
# ─────────────────────────────────────────────────────────────────────────────

def _record_mode(
    *,
    tab_key: str,
    cache_prefix: str,
    cache_clear_fn,
    rec_key: str,
    record_button_label: str,
    record_stop_label: str,
) -> tuple[bytes | None, str | None]:
    """Render the mic-recorder widget and handle state management."""

    st.markdown(
        "<p style='font-size:0.85rem; color:#666; margin-bottom:0.3rem;'>"
        "Press <strong>Start</strong>, speak into your mic, then press <strong>Stop</strong>."
        " Your browser may ask for microphone permission — please allow it."
        "</p>",
        unsafe_allow_html=True,
    )

    # mic_recorder returns a dict with keys 'bytes', 'id', 'sample_rate', etc.
    # or None if no recording has been made yet.
    # format="wav" is required — the default "webm" produces WebM/Opus bytes that
    # soundfile cannot decode (and librosa/audioread can only decode with ffmpeg).
    try:
        recording = mic_recorder(
            start_prompt=record_button_label,
            stop_prompt=record_stop_label,
            just_once=False,          # allow multiple recordings
            use_container_width=True,
            key=f"{tab_key}_mic",
            format="wav",             # force WAV output; default "webm" is unsupported by soundfile
        )
    except Exception as exc:
        st.error(f"🎙️ Microphone recorder failed to load: {exc}")
        st.info(
            "💡 Try the **Upload a file** option instead, or check that your "
            "browser allows microphone access on this page."
        )
        return None, None

    if recording is None:
        _empty_prompt(
            "🎙️",
            "No recording yet…",
            "Press the button above to start. Allow microphone access if prompted.",
        )
        return None, None

    raw_bytes: bytes = recording.get("bytes", b"")

    if not raw_bytes or len(raw_bytes) < 100:
        st.warning(
            "🐾 The recording appears to be empty or too short. "
            "Try again — speak clearly for at least 1–2 seconds."
        )
        return None, None

    # Detect if the previous recording changed (new 'id') so we can clear cache
    prev_id_key = f"{tab_key}_prev_rec_id"
    current_id  = recording.get("id", 0)
    if st.session_state.get(prev_id_key) != current_id:
        st.session_state[prev_id_key] = current_id
        cache_clear_fn()            # clear old results for this tab

    # Show playback and size info.
    # Use the format reported by the recorder (always "wav" when format="wav" is set above).
    rec_format = recording.get("format", "wav")
    st.audio(io.BytesIO(raw_bytes), format=f"audio/{rec_format}")
    st.caption(f"🎙️ Recording ready  ·  {len(raw_bytes) / 1024:.1f} KB")

    cache_key = _bytes_key(raw_bytes, cache_prefix)
    return raw_bytes, cache_key


# ─────────────────────────────────────────────────────────────────────────────
# Shared placeholder
# ─────────────────────────────────────────────────────────────────────────────

def _empty_prompt(icon: str, text: str, sub: str) -> None:
    st.markdown(
        f"""
        <div style="text-align:center; padding:2.5rem 1rem; color:#aaa;">
            <div style="font-size:3rem;">{icon}</div>
            <div style="font-size:1rem; margin-top:0.5rem;">{text}</div>
            <div style="font-size:0.82rem; margin-top:0.3rem;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
