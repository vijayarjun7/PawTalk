"""
ui_helpers.py
-------------
All Streamlit rendering logic for PawTalk.
Keeps app.py clean — each function renders a complete UI section.
"""

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
PRIMARY_COLOR   = "#FF6B35"   # warm orange
SECONDARY_COLOR = "#4ECDC4"   # teal
SUCCESS_COLOR   = "#06D6A0"   # mint green
WARNING_COLOR   = "#F7DC6F"   # warm yellow
DANGER_COLOR    = "#E74C3C"   # red
NEUTRAL_COLOR   = "#B2BEC3"   # grey

MOOD_COLORS = {
    "excited": "#FF6B35",
    "playful": "#FFD166",
    "alert":   "#FF4757",
    "anxious": "#A29BFE",
    "warning": "#E17055",
    "unknown": "#B2BEC3",
}

STATUS_COLORS = {
    "good":    SUCCESS_COLOR,
    "warning": WARNING_COLOR,
    "bad":     DANGER_COLOR,
}

STATUS_BG = {
    "good":    "#E8FBF5",
    "warning": "#FEFAED",
    "bad":     "#FDEDEC",
}


# ---------------------------------------------------------------------------
# Page-level components
# ---------------------------------------------------------------------------

def render_page_header() -> None:
    """Render the PawTalk app title, subtitle, and divider."""
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
            <div style="font-size: 3.5rem; line-height: 1.1;">🐾</div>
            <h1 style="font-size: 2.6rem; font-weight: 800; color: {primary}; margin: 0.2rem 0;">
                PawTalk
            </h1>
            <p style="font-size: 1.05rem; color: #666; margin-top: 0.3rem;">
                Dog Bark Translator &amp; Voice Command Coach
            </p>
            <hr style="border: none; border-top: 2px solid {primary}; margin: 1rem auto; width: 60%;">
        </div>
        """.format(primary=PRIMARY_COLOR),
        unsafe_allow_html=True,
    )


def render_feature_tabs():
    """
    Create the two main feature tabs.
    Returns (bark_tab, voice_tab) for use as context managers in app.py.
    """
    return st.tabs(["🐕  Bark Analyzer", "🗣️  Voice Command Coach"])


# ---------------------------------------------------------------------------
# Shared input components
# ---------------------------------------------------------------------------

def render_audio_uploader(key: str, label: str, help_text: str):
    """
    Render a styled file uploader accepting common audio formats.
    Returns the uploaded file object or None.
    """
    return st.file_uploader(
        label,
        type=["wav", "mp3", "ogg", "flac"],
        help=help_text,
        key=key,
    )


def render_waveform_plot(y: np.ndarray, sr: int, title: str = "Audio Waveform") -> None:
    """
    Render a styled matplotlib waveform plot using librosa.display if available,
    otherwise falling back to a plain numpy plot.
    """
    try:
        import librosa.display
        fig, ax = plt.subplots(figsize=(8, 2.2))
        fig.patch.set_facecolor("#1E1E2E")
        ax.set_facecolor("#1E1E2E")
        librosa.display.waveshow(y, sr=sr, ax=ax, color=PRIMARY_COLOR, alpha=0.85)
        ax.axhline(0, color="#555", linewidth=0.5)
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.tick_params(colors="#888")
        ax.spines[:].set_color("#444")
        ax.set_xlabel("Time (s)", color="#888", fontsize=8)
        ax.set_ylabel("Amplitude", color="#888", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        # Minimal fallback if librosa.display is unavailable
        fig, ax = plt.subplots(figsize=(8, 2))
        time_axis = np.linspace(0, len(y) / sr, num=len(y))
        ax.plot(time_axis, y, color=PRIMARY_COLOR, linewidth=0.6)
        ax.set_title(title)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Bark result rendering
# ---------------------------------------------------------------------------

def render_bark_result(
    translation: dict,
    mood: str,
    confidence: int,
    explanation: str,
    feature_summary: dict,
) -> None:
    """
    Render the full bark analysis result:
    - Colored mood card (headline + message + emoji)
    - Fun fact callout
    - Confidence meter + explanation
    - Expandable feature details chart

    Parameters
    ----------
    translation    : dict from translator.get_bark_translation()
    mood           : str  mood label
    confidence     : int  0–100 score from bark_classifier
    explanation    : str  plain-English reason from bark_classifier
    feature_summary: dict normalised feature values for the chart
    """
    mood_color = MOOD_COLORS.get(mood, NEUTRAL_COLOR)

    # --- Mood card ---
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {mood_color}22, {mood_color}11);
            border-left: 5px solid {mood_color};
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin: 1rem 0;
        ">
            <div style="font-size: 2.8rem; line-height: 1;">{translation['emoji']}</div>
            <h2 style="color: {mood_color}; margin: 0.4rem 0 0.6rem 0; font-size: 1.5rem;">
                {translation['headline']}
            </h2>
            <p style="color: #333; font-size: 0.98rem; line-height: 1.6; margin: 0;">
                {translation['message']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Fun fact ---
    st.markdown(
        f"""
        <div style="
            background: #EEF9FF;
            border-radius: 8px;
            padding: 0.8rem 1.2rem;
            margin: 0.5rem 0 0.8rem 0;
            font-size: 0.88rem;
            color: #444;
            border-left: 4px solid {SECONDARY_COLOR};
        ">
            <strong>🧠 Fun Fact:</strong> {translation['fun_fact']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Confidence meter ---
    # Color shifts green → yellow → red as confidence drops
    if confidence >= 60:
        conf_color = SUCCESS_COLOR
        conf_label = "High confidence"
    elif confidence >= 35:
        conf_color = WARNING_COLOR
        conf_label = "Medium confidence"
    else:
        conf_color = DANGER_COLOR
        conf_label = "Low confidence"

    # Progress-bar style meter using a thin colored div
    st.markdown(
        f"""
        <div style="margin-bottom: 0.4rem;">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;">
                <span style="
                    background:{conf_color}22; color:{conf_color};
                    border:1px solid {conf_color}; border-radius:20px;
                    padding:0.12rem 0.65rem; font-size:0.80rem; font-weight:600;
                ">{conf_label}: {confidence}/100</span>
            </div>
            <div style="
                background:#E0E0E0; border-radius:4px; height:6px; width:100%;
            ">
                <div style="
                    background:{conf_color}; border-radius:4px;
                    height:6px; width:{confidence}%;
                    transition: width 0.4s ease;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Explanation (why this mood was chosen) ---
    st.markdown(
        f"""
        <p style="
            font-size: 0.82rem; color: #666;
            margin: 0.4rem 0 1rem 0; font-style: italic;
        ">🔍 {explanation}</p>
        """,
        unsafe_allow_html=True,
    )

    # --- Alternate style cards ---
    alternates = translation.get("alternates", [])
    if alternates:
        style_icons = {"cute": "🌸", "funny": "😄", "emotional": "💛"}
        with st.expander("💬 See other translations"):
            for alt in alternates:
                alt_style = alt.get("style", "")
                icon = style_icons.get(alt_style, "💬")
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #E0E0E0;
                        border-radius: 10px;
                        padding: 0.8rem 1.1rem;
                        margin-bottom: 0.6rem;
                        background: #FAFAFA;
                    ">
                        <div style="
                            font-size: 0.72rem; text-transform: uppercase;
                            letter-spacing: 0.07em; color: #888;
                            font-weight: 600; margin-bottom: 0.3rem;
                        ">{icon} {alt_style.capitalize()}</div>
                        <div style="font-weight: 700; color: #333; font-size: 0.95rem;">
                            {alt['headline']}
                        </div>
                        <div style="color: #555; font-size: 0.88rem; margin-top: 0.3rem;
                                    line-height: 1.5;">
                            {alt['message']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # --- Feature detail expander ---
    with st.expander("📊 Audio Feature Details"):
        st.caption(
            "Qualitative summary of the audio signals PawTalk detected."
        )
        render_feature_chart(feature_summary)


def render_feature_chart(feature_summary: dict) -> None:
    """
    Render the feature summary as a simple two-column label table.
    feature_summary values are qualitative strings (e.g. 'loud', 'regular'),
    so we display them as styled badges rather than a numeric bar chart.
    """
    if not feature_summary:
        st.caption("No feature data available.")
        return

    # Map qualitative level words to badge colors
    _badge = {
        # energy / loudness
        "quiet":       ("#74B9FF", "#EBF5FF"),
        "moderate":    (WARNING_COLOR, "#FEFAED"),
        "loud":        (PRIMARY_COLOR, "#FFF0EA"),
        # variation
        "steady":      (SUCCESS_COLOR, "#E8FBF5"),
        "variable":    (WARNING_COLOR, "#FEFAED"),
        "spiky":       (DANGER_COLOR,  "#FDEDEC"),
        # tone
        "tonal":       (SUCCESS_COLOR, "#E8FBF5"),
        "mixed":       (WARNING_COLOR, "#FEFAED"),
        "noisy":       ("#A29BFE",     "#F0EEFF"),
        # rhythm
        "regular":     (SUCCESS_COLOR, "#E8FBF5"),
        "loose":       (WARNING_COLOR, "#FEFAED"),
        "chaotic":     (DANGER_COLOR,  "#FDEDEC"),
        # brightness
        "dark":        ("#74B9FF",     "#EBF5FF"),
        "mid":         (WARNING_COLOR, "#FEFAED"),
        "bright":      (PRIMARY_COLOR, "#FFF0EA"),
    }

    rows_html = ""
    for label, value in feature_summary.items():
        color, bg = _badge.get(str(value).lower(), (NEUTRAL_COLOR, "#F5F5F5"))
        rows_html += (
            f'<tr>'
            f'<td style="padding:0.3rem 0.6rem; font-size:0.82rem; color:#555;">{label}</td>'
            f'<td style="padding:0.3rem 0.6rem;">'
            f'<span style="background:{bg}; color:{color}; border:1px solid {color}; '
            f'border-radius:12px; padding:0.1rem 0.55rem; font-size:0.78rem; font-weight:600;">'
            f'{value}</span></td>'
            f'</tr>'
        )

    st.markdown(
        f'<table style="border-collapse:collapse; width:100%;">{rows_html}</table>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Voice result rendering
# ---------------------------------------------------------------------------

_TONE_COLORS = {
    "calm":    ("#74B9FF", "#EBF5FF"),   # blue
    "upbeat":  ("#FFD166", "#FFFAED"),   # yellow
    "firm":    ("#06D6A0", "#E8FBF5"),   # green
    "intense": ("#E74C3C", "#FDEDEC"),   # red
    "unclear": ("#B2BEC3", "#F5F5F5"),   # grey
}

_TONE_ICONS = {
    "calm":    "🌊",
    "upbeat":  "⭐",
    "firm":    "✅",
    "intense": "⚡",
    "unclear": "❓",
}

_PACE_COLORS = {
    "slow":     ("#74B9FF", "#EBF5FF"),
    "moderate": ("#06D6A0", "#E8FBF5"),
    "rapid":    ("#E74C3C", "#FDEDEC"),
    "unclear":  ("#B2BEC3", "#F5F5F5"),
}


def render_voice_result(
    assessment: dict,
    tips: list,
    grade_summary: dict,
) -> None:
    """
    Render the voice analysis result:
    - Grade banner
    - Tone label (primary finding) with description
    - 4-column metric row: Tone | Volume | Duration | Pace
    - Dog recommendation paragraph
    - Example command callout
    - Tip cards
    """
    grade_color = grade_summary.get("color", NEUTRAL_COLOR)
    grade_label = grade_summary.get("label", "—")
    grade_msg   = grade_summary.get("message", "")

    # --- Grade banner ---
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {grade_color}33, {grade_color}11);
            border-left: 5px solid {grade_color};
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        ">
            <div style="font-size: 0.75rem; color: #888; text-transform: uppercase;
                        letter-spacing: 0.08em; font-weight: 600;">Overall Grade</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: {grade_color};">
                {grade_label}
            </div>
            <div style="font-size: 0.92rem; color: #444; margin-top: 0.2rem;">
                {grade_msg}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Tone highlight (primary new output) ---
    tone_label = assessment.get("tone_label", "unclear")
    tone_desc  = assessment.get("tone_description", "")
    t_color, t_bg = _TONE_COLORS.get(tone_label, _TONE_COLORS["unclear"])
    t_icon = _TONE_ICONS.get(tone_label, "❓")

    st.markdown(
        f"""
        <div style="
            background: {t_bg}; border: 1.5px solid {t_color};
            border-radius: 10px; padding: 0.75rem 1.1rem; margin-bottom: 0.8rem;
        ">
            <span style="font-size:1.3rem;">{t_icon}</span>
            <span style="
                font-weight: 700; color: {t_color}; font-size: 1.05rem;
                margin-left: 0.4rem; text-transform: capitalize;
            ">{tone_label} tone</span>
            <p style="margin: 0.25rem 0 0 0; color: #555; font-size: 0.88rem;">
                {tone_desc}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 4 metric columns: Tone · Volume · Duration · Pace ---
    pitch_a    = assessment.get("pitch_assessment", {})
    loudness_a = assessment.get("loudness_assessment", {})
    duration_a = assessment.get("duration_assessment", {})
    pace_label = assessment.get("pace_label", "unclear")

    raw        = assessment.get("raw", {})
    bps        = raw.get("bursts_per_sec", 0.0)
    p_color, _ = _PACE_COLORS.get(pace_label, _PACE_COLORS["unclear"])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _render_metric_card(
            label="Tone",
            value=tone_label.capitalize(),
            icon=t_icon,
            is_good=tone_label in ("calm", "firm", "upbeat"),
            detail=f"F0 ~{pitch_a.get('f0_mean_hz', 0):.0f} Hz",
        )
    with col2:
        _render_metric_card(
            label="Volume",
            value=loudness_a.get("energy_level", "—").replace("_", " ").title(),
            icon="🔊",
            is_good=loudness_a.get("is_good", False),
            detail=loudness_a.get("consistency", "").capitalize(),
        )
    with col3:
        _render_metric_card(
            label="Duration",
            value=duration_a.get("label", "—").replace("_", " ").title(),
            icon="⏱️",
            is_good=duration_a.get("is_good", False),
            detail=f"{duration_a.get('duration_sec', 0):.2f}s",
        )
    with col4:
        _render_metric_card(
            label="Pace",
            value=pace_label.capitalize(),
            icon="🗣️",
            is_good=pace_label == "moderate",
            detail=f"~{bps:.1f}/s" if bps > 0 else "—",
        )

    st.markdown("")

    # --- Dog recommendation ---
    recommendation = assessment.get("dog_recommendation", "")
    if recommendation:
        st.markdown(
            f"""
            <div style="
                background: #F8F4FF; border-left: 4px solid #A29BFE;
                border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 0.7rem;
            ">
                <div style="font-weight:700; color:#6C63FF; font-size:0.85rem;
                            margin-bottom:0.3rem;">🐕 Dog-Friendly Advice</div>
                <p style="margin:0; color:#333; font-size:0.9rem; line-height:1.6;">
                    {recommendation}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Example command ---
    example = assessment.get("example_command", "")
    if example:
        st.markdown(
            f"""
            <div style="
                background: #F0FBF7; border-left: 4px solid {SUCCESS_COLOR};
                border-radius: 8px; padding: 0.75rem 1.1rem; margin-bottom: 1rem;
            ">
                <div style="font-weight:700; color:{SUCCESS_COLOR}; font-size:0.85rem;
                            margin-bottom:0.3rem;">✨ Try This Instead</div>
                <p style="margin:0; color:#333; font-size:0.92rem; font-style:italic;">
                    {example}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Tip cards ---
    if tips:
        st.markdown("##### Detailed Feedback")
        for tip in tips:
            render_tip_card(tip)

    # Footer
    st.markdown(
        "<p style='color:#aaa; font-size:0.78rem; margin-top:1rem;'>"
        "Consistency is everything in dog training — same word, same tone, every time. "
        "Always end with praise. 🐾"
        "</p>",
        unsafe_allow_html=True,
    )


def _render_metric_card(
    label: str,
    value: str,
    icon: str,
    is_good: bool,
    detail: str = "",
) -> None:
    """Render a single metric card inside a column."""
    border_color = SUCCESS_COLOR if is_good else WARNING_COLOR
    st.markdown(
        f"""
        <div style="
            border: 1.5px solid {border_color};
            border-radius: 10px;
            padding: 0.8rem 1rem;
            text-align: center;
            background: {'#F0FBF7' if is_good else '#FEFAED'};
        ">
            <div style="font-size: 1.5rem;">{icon}</div>
            <div style="font-size: 0.7rem; color: #888; text-transform: uppercase;
                        letter-spacing: 0.07em; font-weight: 600; margin-top: 0.2rem;">
                {label}
            </div>
            <div style="font-size: 1rem; font-weight: 700; color: #333; margin-top: 0.2rem;">
                {value}
            </div>
            <div style="font-size: 0.72rem; color: #888; margin-top: 0.1rem;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tip_card(tip: dict) -> None:
    """
    Render a single tip card.
    tip dict: {'category', 'icon', 'tip', 'status'}
    """
    status  = tip.get("status", "good")
    border  = STATUS_COLORS.get(status, NEUTRAL_COLOR)
    bg      = STATUS_BG.get(status, "#F9F9F9")
    icon    = tip.get("icon", "💡")
    cat     = tip.get("category", "")
    text    = tip.get("tip", "")

    st.markdown(
        f"""
        <div style="
            background: {bg};
            border-left: 4px solid {border};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
        ">
            <span style="font-size: 1.1rem;">{icon}</span>
            <strong style="color: {border}; margin-left: 0.4rem; font-size: 0.85rem;">
                {cat}
            </strong>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem; color: #333; line-height: 1.5;">
                {text}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Model insight panel (Hugging Face secondary signal)
# ---------------------------------------------------------------------------

def render_model_insight(combined: dict) -> None:
    """
    Render the HF model insight expander below the main bark result.

    Shows:
    - Whether the rule engine and model agreed
    - Top AudioSet labels + scores as a mini bar chart
    - A plain-English note about what the model heard

    Parameters
    ----------
    combined : dict — output of hf_audio.combine_results()
                      must contain keys: hf, agreement, confidence_delta
    """
    hf        = combined.get("hf", {})
    agreement = combined.get("agreement")
    delta     = combined.get("confidence_delta", 0)

    # Don't render anything if the model wasn't available at all
    if not hf.get("available", False):
        return

    with st.expander("🤖 Audio Model Insight", expanded=False):
        if not hf.get("success", False):
            st.caption(
                f"The audio model ran into an issue: {hf.get('error', 'unknown error')}. "
                "The rule-based result above is unaffected."
            )
            return

        # ── Agreement banner ──────────────────────────────────────────────────
        top_mood = hf.get("top_mood")
        if agreement is True:
            st.markdown(
                f"""
                <div style="
                    background:#E8FBF5; border-left:4px solid {SUCCESS_COLOR};
                    border-radius:8px; padding:0.6rem 1rem; margin-bottom:0.8rem;
                ">
                    <span style="font-weight:700; color:{SUCCESS_COLOR};">
                        ✅ Rule engine and model agree
                    </span>
                    <span style="color:#555; font-size:0.88rem; margin-left:0.4rem;">
                        — confidence nudged {'+' if delta >= 0 else ''}{delta} pts
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif agreement is False:
            rule_mood = combined.get("mood", "—")
            st.markdown(
                f"""
                <div style="
                    background:#FFF8F0; border-left:4px solid {WARNING_COLOR};
                    border-radius:8px; padding:0.6rem 1rem; margin-bottom:0.8rem;
                ">
                    <span style="font-weight:700; color:#B7791F;">
                        ⚠️ Signals differ
                    </span>
                    <span style="color:#555; font-size:0.88rem; margin-left:0.4rem;">
                        — rule engine says <strong>{rule_mood}</strong>,
                        model leans <strong>{top_mood}</strong>.
                        Rule-based result shown above; model view below.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # No mappable mood from model — neutral note
            st.caption(
                "The model didn't recognise a dog-bark mood in the top labels. "
                "Rule-based result is unaffected."
            )

        # ── Top-K labels bar chart ────────────────────────────────────────────
        top_labels = hf.get("top_labels", [])
        if top_labels:
            st.markdown(
                "<div style='font-size:0.78rem; color:#888; "
                "text-transform:uppercase; letter-spacing:0.07em; "
                "font-weight:600; margin-bottom:0.4rem;'>"
                f"Top {len(top_labels)} AudioSet labels"
                "</div>",
                unsafe_allow_html=True,
            )
            bars_html = ""
            for entry in top_labels:
                label    = entry["label"]
                score    = entry["score"]
                mood_tag = entry.get("mood")
                pct      = int(score * 100)
                bar_col  = MOOD_COLORS.get(mood_tag, NEUTRAL_COLOR) if mood_tag else "#C8D6E5"
                mood_badge = (
                    f'<span style="background:{bar_col}22; color:{bar_col}; '
                    f'border:1px solid {bar_col}; border-radius:10px; '
                    f'padding:0.05rem 0.45rem; font-size:0.72rem; '
                    f'font-weight:600; margin-left:0.4rem;">{mood_tag}</span>'
                    if mood_tag else ""
                )
                bars_html += f"""
                <div style="margin-bottom:0.45rem;">
                    <div style="display:flex; align-items:center;
                                justify-content:space-between; margin-bottom:0.15rem;">
                        <span style="font-size:0.82rem; color:#444;">
                            {label}{mood_badge}
                        </span>
                        <span style="font-size:0.78rem; color:#888; font-weight:600;">
                            {pct}%
                        </span>
                    </div>
                    <div style="background:#E8ECF0; border-radius:3px; height:5px;">
                        <div style="background:{bar_col}; border-radius:3px;
                                    height:5px; width:{pct}%;"></div>
                    </div>
                </div>
                """
            st.markdown(bars_html, unsafe_allow_html=True)

        # ── Model attribution note ────────────────────────────────────────────
        model_id = hf.get("model_id", "")
        st.caption(
            f"Model: `{model_id}` · Trained on AudioSet (527-class audio classification). "
            "Labels are general audio categories, not dog-mood labels — "
            "the mapping above is approximate and for fun only."
        )


# ---------------------------------------------------------------------------
# Error & utility helpers
# ---------------------------------------------------------------------------

def render_error(message: str) -> None:
    """Render a friendly error box with a paw print prefix."""
    st.error(f"🐾 Oops! {message}")


def render_loading_spinner(label: str = "Sniffing the audio..."):
    """Return a st.spinner() context manager. Centralizes label text."""
    return st.spinner(label)
