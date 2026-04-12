"""
app.py
------
PawTalk — Dog Bark Translator & Voice Command Coach

Run with:
    streamlit run app.py

Architecture
------------
- Sidebar holds global settings (dog name, translation style).
- Two tabs: "Dog → Human" (bark analysis) and "Human → Dog" (voice coaching).
- Each tab has an explicit upload → analyze button → results flow so the user
  controls when processing happens.
- Audio processing results are cached in st.session_state keyed to file
  identity so re-renders (style changes, name edits) never re-run librosa.
- Every error path shows a friendly fallback message and a suggestion.
"""

import streamlit as st

from utils import (
    audio_features, audio_input, bark_classifier, voice_analyzer,
    translator, ui_helpers, hf_audio, ai_bark_classifier,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PawTalk",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS tweaks
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
        /* Slightly wider centered layout */
        .block-container { max-width: 840px; padding-top: 1.2rem; }

        /* Soften the default file-uploader border */
        [data-testid="stFileUploader"] { border-radius: 10px; }

        /* Remove top padding from sidebar */
        section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

        /* Tighten tab label spacing */
        .stTabs [data-baseweb="tab"] { padding: 0.5rem 1.1rem; font-size: 0.95rem; }

        /* Make st.metric labels slightly smaller */
        [data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — global settings
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> tuple[str | None, str]:
    """
    Render the sidebar settings panel.
    Returns (dog_name, style) — used by the bark tab.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding-bottom:0.8rem;">
                <span style="font-size:2.2rem;">🐾</span>
                <div style="font-weight:800; font-size:1.25rem; color:#FF6B35;">PawTalk</div>
                <div style="font-size:0.75rem; color:#888; margin-top:0.1rem;">
                    Settings
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # Dog name
        st.markdown("**Your dog's name** *(optional)*")
        dog_name = st.text_input(
            label="dog_name_label",
            placeholder="e.g. Biscuit, Luna, Sir Barks-a-Lot…",
            key="sidebar_dog_name",
            label_visibility="collapsed",
        ).strip() or None

        if dog_name:
            st.success(f"🐕 Translating for **{dog_name}**!")

        st.divider()

        # Translation style
        st.markdown("**Translation style**")
        style_options  = ["funny", "cute", "emotional"]
        style_captions = [
            "Comedic and irreverent",
            "Warm and wholesome",
            "Heartfelt and sincere",
        ]
        style_icons = {"funny": "😄", "cute": "🌸", "emotional": "💛"}
        style = st.radio(
            label="style_label",
            options=style_options,
            captions=style_captions,
            key="sidebar_style",
            label_visibility="collapsed",
        )
        st.caption(f"Currently: {style_icons[style]} **{style.capitalize()}**")

        st.divider()

        # About / disclaimer
        with st.expander("ℹ️ About PawTalk"):
            st.markdown(
                """
                PawTalk is a **gift app**, not a scientific instrument.
                It uses audio signal analysis (pitch, energy, rhythm) to make
                playful guesses about what your dog might be feeling.

                Results are for fun and inspiration — not veterinary or
                behavioural diagnosis.

                **Tab 1 — Dog → Human**
                Upload a bark clip and get a mood estimate + playful translation.

                **Tab 2 — Human → Dog**
                Upload yourself giving a command and get delivery coaching.

                Made with 🐾 and librosa.
                """
            )

    return dog_name, style


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _clear_bark_cache():
    """Remove all cached bark results so the next upload starts fresh."""
    for k in list(st.session_state.keys()):
        if k.startswith("bark_result_"):
            del st.session_state[k]


def _clear_voice_cache():
    for k in list(st.session_state.keys()):
        if k.startswith("voice_result_"):
            del st.session_state[k]


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Dog → Human (bark translation)
# ─────────────────────────────────────────────────────────────────────────────

def _bark_tab(dog_name: str | None, style: str) -> None:

    # ── Description ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#FFF8F0; border-radius:10px; padding:0.9rem 1.2rem;
                    border-left:4px solid #FF6B35; margin-bottom:1.2rem;">
            <strong>How it works:</strong> Upload a recording of your dog barking.
            PawTalk will listen to the audio's energy, pitch, and rhythm to estimate
            their mood — then translate it into something human-readable.
            <span style="color:#888; font-size:0.85rem;">
              (This is a fun estimate, not a scientific translation.)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Audio input (upload or in-browser recording) ──────────────────────────
    audio_bytes, cache_key = audio_input.get_audio_input(
        tab_key="bark",
        cache_prefix="bark_result_",
        cache_clear_fn=_clear_bark_cache,
        upload_label="Upload a bark recording",
        upload_help="Best results: 2–10 seconds of clear barking with minimal background noise. WAV recommended.",
        empty_prompt_icon="🎙️",
        empty_prompt_text="Waiting for your pup's vocal debut…",
        empty_prompt_sub="Upload a WAV, MP3, OGG, or FLAC file above.",
        record_button_label="🎙️ Start recording bark",
        record_stop_label="⏹️ Stop",
    )

    if audio_bytes is None:
        return

    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        analyze = st.button(
            "🔍 Analyze Bark",
            key="bark_analyze_btn",
            type="primary",
            use_container_width=True,
        )
    with col_note:
        if cache_key in st.session_state:
            st.caption("✅ Analysis complete — change style or name in the sidebar instantly.")
        else:
            st.caption("Press **Analyze** to process the audio. May take a few seconds.")

    # ── Processing ────────────────────────────────────────────────────────────
    if analyze and cache_key not in st.session_state:
        with st.spinner("Sniffing the audio… 🐽"):
            try:
                y, sr = audio_features.load_audio_from_bytes(audio_bytes)
                feats = audio_features.extract_features(y, sr)
            except audio_features.AudioTooShortError as e:
                st.error(f"🐾 Clip too short — {e}")
                st.info("💡 Try a recording that's at least 1–2 seconds of clear barking.")
                return
            except audio_features.AudioLoadError as e:
                st.error(f"🐾 Couldn't read the file — {e}")
                st.info("💡 Try saving the file as a WAV and uploading again.")
                return
            except Exception as e:
                st.error(f"🐾 Something went wrong during analysis: {e}")
                st.info(
                    "💡 Check that the file is a real audio recording (not renamed). "
                    "WAV files are most reliable."
                )
                return

            # ── Rule-based classification (always runs — no downloads) ─────────
            rule_result = bark_classifier.classify_bark_mood(feats)

            # ── AI supervised classifier (primary when checkpoint exists) ──────
            # Priority: AI confident → AI moderate → rule engine fallback
            # The AI path is entirely optional; missing checkpoint = rule only.
            if ai_bark_classifier.AI_AVAILABLE:
                if ai_bark_classifier.checkpoint_exists() and not ai_bark_classifier.AI_MODEL_LOADED:
                    st.caption("🤖 Loading AI bark model for the first time…")
                ai_result = ai_bark_classifier.classify_bark_ai(y, sr)
                result = ai_bark_classifier.combine_ai_and_rule(ai_result, rule_result)

                # Surface source to the user
                source = result.get("source", "rule_fallback")
                if source == "ai_confident":
                    st.caption("✅ AI classifier — high confidence")
                elif source == "ai_moderate":
                    st.caption("🔶 AI classifier — moderate confidence; rule engine consulted")
                else:
                    st.caption("ℹ️ Rule engine — AI model unavailable or uncertain")
            else:
                # torch/transformers not installed — rule engine only
                ai_result = {"available": False}
                result = {
                    **rule_result,
                    "ai":     ai_result,
                    "source": "rule_fallback",
                }

            # ── Optional HF AudioSet secondary signal ─────────────────────────
            # Runs on top of whichever result was chosen above.
            hf_loaded = hf_audio._pipeline_cache_is_ready()
            if hf_audio.HF_AVAILABLE and not hf_loaded:
                st.caption("🤖 Loading AudioSet model for the first time…")
            hf_result = hf_audio.classify_audio(y, sr)
            if hf_audio.HF_AVAILABLE and not hf_result.get("success"):
                st.caption(f"ℹ️ AudioSet model unavailable ({hf_result.get('error', '')})")

            # Attach HF insight without changing the mood (combine_results is
            # additive on confidence only, rule_result is already baked into
            # result — pass it again just for the HF merge step).
            result = hf_audio.combine_results(result, hf_result)

            # Store bytes (not numpy array) so waveform can be reconstructed
            # on re-render without holding a large array in session state.
            st.session_state[cache_key] = {
                "audio_bytes": audio_bytes,
                "sr":          sr,
                "result":      result,
            }

    if cache_key not in st.session_state:
        # Button hasn't been pressed yet — show a gentle prompt
        st.markdown(
            "<p style='color:#aaa; font-size:0.85rem; margin-top:0.3rem;'>"
            "👆 Hit Analyze when you're ready."
            "</p>",
            unsafe_allow_html=True,
        )
        return

    # ── Results ───────────────────────────────────────────────────────────────
    cached = st.session_state[cache_key]
    sr     = cached["sr"]
    result = cached["result"]

    mood        = result["mood"]
    confidence  = result["confidence"]
    explanation = result["explanation"]

    # Style and dog_name are free to change without re-analyzing
    translation = translator.get_bark_translation(mood, confidence, style=style, dog_name=dog_name)

    st.divider()

    # Reconstruct waveform from the stored bytes for display only.
    # This avoids keeping the numpy array in session state long-term.
    try:
        y, _ = audio_features.load_audio_from_bytes(cached["audio_bytes"])
        ui_helpers.render_waveform_plot(y, sr, title="Bark Waveform")
    except Exception:
        pass  # waveform is cosmetic — skip silently if reconstruction fails

    # ── Mood + confidence metrics row ────────────────────────────────────────
    mood_color  = ui_helpers.MOOD_COLORS.get(mood, ui_helpers.NEUTRAL_COLOR)
    mood_emoji  = translation["emoji"]
    mood_display = mood.replace("_", " ").title() if mood != "unknown" else "Unknown"

    source = result.get("source", "rule_fallback")
    source_labels = {
        "ai_confident": ("🤖 AI", "#06D6A0"),
        "ai_moderate":  ("🤖 AI+Rule", "#74B9FF"),
        "rule_fallback": ("📐 Rule", "#B2BEC3"),
    }
    source_text, source_color = source_labels.get(source, ("📐 Rule", "#B2BEC3"))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div style="text-align:center; padding:0.6rem;">
                <div style="font-size:0.7rem; color:#888; text-transform:uppercase;
                            letter-spacing:0.07em; font-weight:600;">Detected Mood</div>
                <div style="font-size:1.7rem; margin:0.2rem 0;">{mood_emoji}</div>
                <div style="font-size:1.1rem; font-weight:800; color:{mood_color};">
                    {mood_display}
                </div>
                <div style="font-size:0.72rem; color:{source_color}; font-weight:600;
                            margin-top:0.15rem;">{source_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        # Confidence delta label — note HF agreement adjustment when present
        hf_delta     = result.get("confidence_delta", 0)
        agreement    = result.get("agreement")
        if hf_delta > 0 and agreement is True:
            conf_suffix = f" (model agrees +{hf_delta})"
        elif hf_delta < 0 and agreement is False:
            conf_suffix = f" (model differs {hf_delta})"
        elif confidence >= 60:
            conf_suffix = " · strong signal"
        elif confidence >= 35:
            conf_suffix = " · moderate"
        else:
            conf_suffix = " · low signal"
        st.metric(label="Confidence", value=f"{confidence} / 100", delta=conf_suffix)
        st.progress(confidence / 100)
    with m3:
        st.metric(
            label="Style",
            value=style.capitalize(),
            delta=dog_name if dog_name else "no name set",
        )

    # ── Full translation card ────────────────────────────────────────────────
    ui_helpers.render_bark_result(
        translation=translation,
        mood=mood,
        confidence=confidence,
        explanation=explanation,
        feature_summary=result["feature_summary"],
    )

    # ── HF model secondary insight (collapsed by default) ────────────────────
    ui_helpers.render_model_insight(result)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Human → Dog (voice coaching)
# ─────────────────────────────────────────────────────────────────────────────

def _voice_tab() -> None:

    # ── Description ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#F0F8FF; border-radius:10px; padding:0.9rem 1.2rem;
                    border-left:4px solid #4ECDC4; margin-bottom:1.2rem;">
            <strong>How it works:</strong> Upload yourself saying a dog training command
            (e.g. <em>"Sit"</em>, <em>"Stay"</em>, <em>"Come"</em>).
            PawTalk will analyse your tone, volume, duration, and pace — then suggest
            how to sound more like a dog trainer.
            <span style="color:#888; font-size:0.85rem;">
              (Based on signal analysis, not magic.)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tips before upload ───────────────────────────────────────────────────
    with st.expander("📖 What makes a good dog command? (quick guide)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                **✅ Do**
                - Use one clear word: *Sit. Stay. Come.*
                - Keep it 0.5 – 1.5 seconds long
                - Stay consistent — same word, same tone, every time
                - Use an upbeat tone for recall and praise
                - Use a calm, firm tone for settle commands
                """
            )
        with c2:
            st.markdown(
                """
                **❌ Avoid**
                - Long sentences (*"Can you please sit down for me?"*)
                - Shouting — confidence beats volume
                - Whispering — dogs need clarity, not silence
                - Erratic or frightened-sounding delivery
                - Repeating the command over and over
                """
            )

    # ── Audio input (upload or in-browser recording) ──────────────────────────
    audio_bytes, cache_key = audio_input.get_audio_input(
        tab_key="voice",
        cache_prefix="voice_result_",
        cache_clear_fn=_clear_voice_cache,
        upload_label="Upload your voice command",
        upload_help="Aim for 0.5–3 seconds. Say one command word clearly. WAV recommended.",
        empty_prompt_icon="🗣️",
        empty_prompt_text="Waiting for your command, human…",
        empty_prompt_sub='Record yourself saying "Sit" or "Stay", then upload it here.',
        record_button_label="🎙️ Start recording command",
        record_stop_label="⏹️ Stop",
    )

    if audio_bytes is None:
        return

    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        analyze = st.button(
            "🔍 Analyze Voice",
            key="voice_analyze_btn",
            type="primary",
            use_container_width=True,
        )
    with col_note:
        if cache_key in st.session_state:
            st.caption("✅ Analysis complete.")
        else:
            st.caption("Press **Analyze** to process the audio.")

    # ── Processing ────────────────────────────────────────────────────────────
    if analyze and cache_key not in st.session_state:
        with st.spinner("Listening carefully… 👂"):
            try:
                y, sr      = audio_features.load_audio_from_bytes(audio_bytes)
                feats      = audio_features.extract_features(y, sr)
                assessment = voice_analyzer.analyze_voice_command(feats)
                st.session_state[cache_key] = {
                    "audio_bytes": audio_bytes,
                    "sr":          sr,
                    "assessment":  assessment,
                }
            except audio_features.AudioTooShortError as e:
                st.error(f"🐾 Clip too short — {e}")
                st.info("💡 Record at least 0.5 seconds. Try just saying 'Sit!' clearly.")
                return
            except audio_features.AudioLoadError as e:
                st.error(f"🐾 Couldn't read the file — {e}")
                st.info("💡 Try saving the recording as a WAV file and uploading again.")
                return
            except Exception as e:
                st.error(f"🐾 Something went wrong: {e}")
                st.info(
                    "💡 Make sure the file contains actual speech. "
                    "A silent or very noisy recording won't produce useful results."
                )
                return

    if cache_key not in st.session_state:
        st.markdown(
            "<p style='color:#aaa; font-size:0.85rem; margin-top:0.3rem;'>"
            "👆 Hit Analyze when you're ready."
            "</p>",
            unsafe_allow_html=True,
        )
        return

    # ── Results ───────────────────────────────────────────────────────────────
    cached     = st.session_state[cache_key]
    sr         = cached["sr"]
    assessment = cached["assessment"]

    tips          = translator.get_voice_tips(assessment)
    grade_summary = translator.get_grade_summary(assessment["overall_grade"])

    st.divider()

    try:
        y, _ = audio_features.load_audio_from_bytes(cached["audio_bytes"])
        ui_helpers.render_waveform_plot(y, sr, title="Your Voice Command")
    except Exception:
        pass

    # ── Quick metrics row ────────────────────────────────────────────────────
    tone_label    = assessment.get("tone_label", "unclear")
    overall_grade = assessment.get("overall_grade", "unclear")
    dur_label     = assessment.get("duration_label", "—")
    raw           = assessment.get("raw", {})
    duration_sec  = raw.get("duration_sec", 0.0)

    # Map grades to numeric for progress bar
    grade_pct = {"excellent": 1.0, "good": 0.72, "needs_work": 0.42, "unclear": 0.18}
    grade_color_map = {
        "excellent": "#06D6A0",
        "good":      "#74B9FF",
        "needs_work":"#F7DC6F",
        "unclear":   "#B2BEC3",
    }

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Tone",
            value=tone_label.capitalize(),
            delta="✓ good" if tone_label in ("calm", "firm", "upbeat") else "needs work",
        )
    with m2:
        st.metric(
            label="Duration",
            value=f"{duration_sec:.2f}s",
            delta=dur_label.replace("_", " "),
        )
    with m3:
        pace_label = assessment.get("pace_label", "unclear")
        st.metric(
            label="Pace",
            value=pace_label.capitalize(),
            delta="✓ good" if pace_label == "moderate" else "adjust",
        )
    with m4:
        st.metric(
            label="Grade",
            value=overall_grade.replace("_", " ").title(),
        )
        grade_val = grade_pct.get(overall_grade, 0.2)
        bar_color = grade_color_map.get(overall_grade, "#B2BEC3")
        # Streamlit progress bar doesn't support custom colours;
        # use a thin HTML bar for colour fidelity
        st.markdown(
            f"""<div style="background:#E0E0E0; border-radius:4px; height:5px; margin-top:-0.3rem;">
                <div style="background:{bar_color}; border-radius:4px;
                            height:5px; width:{int(grade_val * 100)}%;"></div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Full voice result ────────────────────────────────────────────────────
    ui_helpers.render_voice_result(
        assessment=assessment,
        tips=tips,
        grade_summary=grade_summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Sidebar — returns settings used by the bark tab
    dog_name, style = _render_sidebar()

    # Header
    ui_helpers.render_page_header()

    # Short description below the header
    st.markdown(
        """
        <p style="text-align:center; color:#666; font-size:0.95rem; margin-top:-0.5rem;">
            A fun audio toy for dog lovers.
            Upload a bark to get a playful translation, or upload your own voice
            to get dog-training delivery tips.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    bark_tab, voice_tab = ui_helpers.render_feature_tabs()

    with bark_tab:
        _bark_tab(dog_name=dog_name, style=style)

    with voice_tab:
        _voice_tab()


if __name__ == "__main__":
    main()
