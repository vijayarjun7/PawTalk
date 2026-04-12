"""
voice_analyzer.py
-----------------
Analyzes a human voice audio clip and returns dog-friendly communication advice.

This module does NOT translate dog sounds — it works in the other direction:
you speak, we tell you how a dog probably heard it and how to do better.

Design
------
Four independent assessments run on the extracted audio features:

  1. Tone     — derived from pitch (F0 mean & variation) and spectral brightness.
               Labels: calm | upbeat | firm | intense | unclear
               Dogs respond differently to each; the label drives the recommendation.

  2. Loudness — RMS energy mapped to: too_soft | just_right | too_loud

  3. Duration — clip length mapped to: too_short | ideal | slightly_long | too_long

  4. Pace     — estimated from burst_count relative to duration.
               Labels: slow | moderate | rapid | unclear
               Rapid-fire commands confuse dogs; single clear bursts work best.

These four feed into:
  - A composite tone_label (the primary output the UI highlights)
  - An overall_grade: excellent | good | needs_work | unclear
  - A dog_recommendation: one plain-English paragraph of advice
  - An example_command: a short corrected example the user can try immediately

All thresholds are named constants at the top of the file — change numbers
there without touching the logic below.
"""

# =============================================================================
# Thresholds — change these to tune sensitivity, not the logic below
# =============================================================================

# --- Loudness (RMS, float32 audio, approx 0.0–1.0) ---
RMS_TOO_SOFT      = 0.018   # below → too quiet; dog may not register it
RMS_IDEAL_LOW     = 0.035
RMS_IDEAL_HIGH    = 0.185
RMS_TOO_LOUD      = 0.215   # above → may startle or stress the dog

# Volume consistency: rms_std / rms_mean coefficient of variation
RMS_CV_CONSISTENT = 0.40    # below → steady volume throughout
RMS_CV_VARIABLE   = 0.85    # above → very uneven (trailing off or bursting)

# --- Pitch (F0, Hz) ---
F0_UNDETECTED     = 10.0    # f0_mean below this → no reliable pitch found
F0_LOW_CUTOFF     = 100.0   # below → very deep / low-pitched voice
F0_HIGH_CUTOFF    = 350.0   # above → high / bright pitch

F0_STD_STEADY     = 25.0    # Hz — very flat pitch (calm / firm delivery)
F0_STD_EXPRESSIVE = 75.0    # Hz — animated but controlled (upbeat / praise)
# above F0_STD_EXPRESSIVE → erratic (intense or unclear)

# --- Spectral brightness (centroid, Hz) ---
CENTROID_DARK     = 1200.0  # below → dark/rumbling quality
CENTROID_BRIGHT   = 2800.0  # above → sharp/bright quality

# --- Duration (seconds) ---
DURATION_TOO_SHORT    = 0.30
DURATION_IDEAL_MAX    = 1.50
DURATION_SLIGHTLY_MAX = 3.00
# above DURATION_SLIGHTLY_MAX → too long

# --- Pace (bursts per second) ---
PACE_SLOW_MAX    = 0.5    # bursts/s — fewer than this → slow / deliberate
PACE_RAPID_MIN   = 2.0    # bursts/s — more than this → rapid fire

# --- Grading weights ---
# Each dimension contributes these points when "good"; max = sum of all
GRADE_WEIGHTS = {
    "tone":     2,   # tone matters most — it affects how the dog interprets intent
    "loudness": 2,
    "duration": 2,
    "pace":     1,   # pace is a bonus signal, not penalised as heavily
}
GRADE_MAX = sum(GRADE_WEIGHTS.values())   # 7


# =============================================================================
# Public API
# =============================================================================

def analyze_voice_command(features: dict) -> dict:
    """
    Analyze a human voice clip and return dog-training advice.

    Parameters
    ----------
    features : dict — output of audio_features.extract_features()

    Returns
    -------
    dict with keys:
        tone_label          : str  — 'calm' | 'upbeat' | 'firm' | 'intense' | 'unclear'
        tone_description    : str  — one sentence explaining what the tone label means
        loudness_label      : str  — 'too_soft' | 'just_right' | 'too_loud'
        duration_label      : str  — 'too_short' | 'ideal' | 'slightly_long' | 'too_long'
        pace_label          : str  — 'slow' | 'moderate' | 'rapid' | 'unclear'
        overall_grade       : str  — 'excellent' | 'good' | 'needs_work' | 'unclear'
        dog_recommendation  : str  — plain-English paragraph of targeted advice
        example_command     : str  — short example the user can try immediately
        raw                 : dict — numeric values used (for UI display / debugging)
    """
    # Pull features with safe defaults
    f0_mean   = float(features.get("f0_mean",                0.0))
    f0_std    = float(features.get("f0_std",                 0.0))
    rms_mean  = float(features.get("rms_mean",               0.0))
    rms_std   = float(features.get("rms_std",                0.0))
    duration  = float(features.get("duration_sec",           0.0))
    centroid  = float(features.get("spectral_centroid_mean", 0.0))
    burst_n   = int(features.get("burst_count",              0))

    tone     = _assess_tone(f0_mean, f0_std, centroid)
    loudness = _assess_loudness(rms_mean, rms_std)
    dur      = _assess_duration(duration)
    pace     = _assess_pace(burst_n, duration)
    grade    = _compute_grade(tone, loudness, dur, pace)

    recommendation = _build_recommendation(tone, loudness, dur, pace)
    example        = _suggest_example(tone, loudness, dur)

    return {
        "tone_label":         tone["label"],
        "tone_description":   tone["description"],
        "loudness_label":     loudness["label"],
        "duration_label":     dur["label"],
        "pace_label":         pace["label"],
        "overall_grade":      grade,
        "dog_recommendation": recommendation,
        "example_command":    example,
        # Keep the detailed sub-dicts for the UI metric cards and translator tips
        "pitch_assessment":    _legacy_pitch_assessment(tone, f0_mean, f0_std, f0_std),
        "loudness_assessment": _legacy_loudness_assessment(loudness, rms_mean),
        "duration_assessment": _legacy_duration_assessment(dur, duration),
        "energy_level":        loudness["label"],
        "raw": {
            "f0_mean_hz":     round(f0_mean, 1),
            "f0_std_hz":      round(f0_std, 1),
            "rms_mean":       round(rms_mean, 4),
            "duration_sec":   round(duration, 2),
            "centroid_hz":    round(centroid, 0),
            "burst_count":    burst_n,
            "bursts_per_sec": round(burst_n / max(duration, 0.01), 2),
        },
    }


# =============================================================================
# Assessment functions — each returns a small dict with label + is_good + tip
# =============================================================================

def _assess_tone(f0_mean: float, f0_std: float, centroid: float) -> dict:
    """
    Classify the overall vocal tone using pitch and spectral brightness.

    Tone label meanings for dogs:
      calm     — low flat pitch, dark spectrum; good for 'stay', 'down', 'wait'
      upbeat   — moderate-high pitch with expressive variation; good for 'come!', praise
      firm     — moderate pitch, very steady; good for 'sit', 'leave it', corrections
      intense  — high pitch OR very erratic OR very bright; may over-excite or confuse
      unclear  — pitch undetectable; can't give confident advice
    """
    # Can't determine tone without pitch
    if f0_mean < F0_UNDETECTED:
        return {
            "label":       "unclear",
            "description": "Pitch was too faint to detect reliably.",
            "is_good":     False,
            "tip": (
                "Your voice was too quiet or breathy to read clearly. "
                "Try recording closer to the mic with a more projected voice."
            ),
        }

    pitch_high   = f0_mean > F0_HIGH_CUTOFF
    pitch_low    = f0_mean < F0_LOW_CUTOFF
    bright       = centroid > CENTROID_BRIGHT
    dark         = centroid < CENTROID_DARK
    very_steady  = f0_std < F0_STD_STEADY
    expressive   = F0_STD_STEADY <= f0_std < F0_STD_EXPRESSIVE
    erratic      = f0_std >= F0_STD_EXPRESSIVE

    # Classify in priority order
    if erratic or (pitch_high and bright):
        label = "intense"
        description = "Your delivery sounds intense — energetic and urgent."
        is_good = False
        tip = (
            "Intense or erratic delivery can over-excite dogs or make them anxious. "
            "Try slowing down and using a steady, deliberate tone. "
            "Think 'firm teacher', not 'panicked shouting'."
        )

    elif pitch_high or bright:
        label = "upbeat"
        description = "Your tone is bright and upbeat — enthusiastic and encouraging."
        is_good = True
        tip = (
            "Upbeat tones are great for praise, recall, and reward markers. "
            "For calm-down commands ('stay', 'wait'), try dropping your pitch slightly."
        )

    elif (pitch_low or dark) and very_steady:
        label = "calm"
        description = "Your tone is calm and low — measured and reassuring."
        is_good = True
        tip = (
            "Calm tones work brilliantly for 'stay', 'wait', and settling commands. "
            "For commands that need energy like 'come!', try lifting your pitch a little."
        )

    elif very_steady or expressive:
        # Middle ground: moderate pitch, steady or mildly expressive
        label = "firm"
        description = "Your tone is firm and clear — confident without being harsh."
        is_good = True
        tip = (
            "Firm, steady tones are ideal for training commands. "
            "Dogs read consistency as leadership. "
            "Keep using this tone for 'sit', 'down', and 'leave it'."
        )

    else:
        # Residual: low pitch but somewhat erratic
        label = "unclear"
        description = "Your tone was hard to characterise consistently."
        is_good = False
        tip = (
            "Your pitch was inconsistent — try saying the command the same way each time. "
            "Pick one tone (calm, firm, or upbeat) and commit to it. "
            "Consistency is more important to dogs than any particular sound."
        )

    return {"label": label, "description": description, "is_good": is_good, "tip": tip}


def _assess_loudness(rms_mean: float, rms_std: float) -> dict:
    """
    Evaluate volume level and delivery consistency.
    Returns label, is_good, tip, and an rms_cv (coefficient of variation).
    """
    # Energy level
    if rms_mean < RMS_TOO_SOFT:
        label   = "too_soft"
        is_good = False
        tip = (
            "Your command was too quiet. Dogs don't need volume, but they do need "
            "clarity — speak at a comfortable conversational level, not a whisper."
        )
    elif rms_mean > RMS_TOO_LOUD:
        label   = "too_loud"
        is_good = False
        tip = (
            "You were quite loud. Shouting can put dogs on edge and trigger "
            "a fear response rather than a trained one. "
            "Firm and calm always beats loud."
        )
    else:
        label   = "just_right"
        is_good = True
        tip     = "Good volume — clear and confident without being overwhelming."

    # Append consistency note if volume is wildly variable
    rms_cv = (rms_std / rms_mean) if rms_mean > 1e-6 else 1.0
    if rms_cv > RMS_CV_VARIABLE and is_good:
        tip += (
            " However, your volume trailed off toward the end — "
            "try to keep consistent energy from start to finish of the command."
        )

    consistency = (
        "consistent" if rms_cv < RMS_CV_CONSISTENT else
        "variable"   if rms_cv < RMS_CV_VARIABLE   else
        "uneven"
    )

    return {
        "label":       label,
        "is_good":     is_good,
        "tip":         tip,
        "consistency": consistency,
        "rms_cv":      round(rms_cv, 3),
        # keep energy_level as alias for backward compat with translator
        "energy_level": label,
    }


def _assess_duration(duration_sec: float) -> dict:
    """
    Evaluate command length against dog-training best practices.
    Single short words are best; long phrases lose the dog's attention.
    """
    if duration_sec < DURATION_TOO_SHORT:
        label   = "too_short"
        is_good = False
        tip = (
            "That was very brief — under 0.3 seconds. "
            "Aim for a clean half-second to one-second command: long enough to be clear, "
            "short enough to stay crisp."
        )
    elif duration_sec <= DURATION_IDEAL_MAX:
        label   = "ideal"
        is_good = True
        tip = (
            "Ideal command length. Short, clear, and complete — "
            "exactly what dogs process best."
        )
    elif duration_sec <= DURATION_SLIGHTLY_MAX:
        label   = "slightly_long"
        is_good = False
        tip = (
            f"At {duration_sec:.1f}s this is a little long. "
            "Dogs respond to the sound of a command, not its meaning. "
            "Trim to just the core word: 'Sit.', 'Stay.', 'Come!'"
        )
    else:
        label   = "too_long"
        is_good = False
        tip = (
            f"At {duration_sec:.1f}s this is quite long for a command. "
            "By the time you finish, your dog has likely stopped listening. "
            "The most effective commands are one or two syllables, delivered once."
        )

    return {
        "label":        label,
        "is_good":      is_good,
        "tip":          tip,
        "duration_sec": round(duration_sec, 2),
    }


def _assess_pace(burst_count: int, duration_sec: float) -> dict:
    """
    Estimate speaking pace from burst_count (energy bursts per second).
    A 'burst' approximates a syllable or word — so bursts/sec ≈ speaking rate.

    slow     — deliberate, one clear word; ideal for stay/wait/down
    moderate — natural conversational pace; works for most commands
    rapid    — too many syllables too quickly; dogs can't lock onto the command word
    unclear  — not enough data to judge
    """
    if duration_sec < 0.1 or burst_count == 0:
        return {
            "label":        "unclear",
            "is_good":      False,
            "bursts_per_sec": 0.0,
            "tip": (
                "Not enough audio to estimate pace. "
                "Try a slightly longer clip — at least 0.5 seconds of clear speech."
            ),
        }

    bps     = burst_count / duration_sec
    is_good = True

    if bps < PACE_SLOW_MAX:
        label = "slow"
        tip = (
            "Very deliberate pace — this works well for settling commands "
            "like 'stay' or 'down'. For high-energy cues like 'come!' "
            "a slightly brisker tone can help convey urgency."
        )
    elif bps <= PACE_RAPID_MIN:
        label = "moderate"
        tip = (
            "Good pace — natural and clear. "
            "Dogs have enough time to pick out the command word."
        )
    else:
        label   = "rapid"
        is_good = False
        tip = (
            f"You're delivering about {bps:.1f} bursts per second — quite fast. "
            "Dogs respond to a single clear command word, not a stream of syllables. "
            "Try saying just one word, with a brief pause before and after."
        )

    return {
        "label":          label,
        "is_good":        is_good,
        "bursts_per_sec": round(bps, 2),
        "tip":            tip,
    }


# =============================================================================
# Grading
# =============================================================================

def _compute_grade(tone: dict, loudness: dict, dur: dict, pace: dict) -> str:
    """
    Weighted score across four dimensions → overall grade.

    Weighting rationale:
      - Tone and loudness have the most direct impact on whether a dog
        associates the command with safety/reward vs fear/confusion.
      - Duration is highly trainable; pace is a bonus signal.
    """
    score = (
        GRADE_WEIGHTS["tone"]     * (1 if tone["is_good"]     else 0)
        + GRADE_WEIGHTS["loudness"] * (1 if loudness["is_good"] else 0)
        + GRADE_WEIGHTS["duration"] * (1 if dur["is_good"]      else 0)
        + GRADE_WEIGHTS["pace"]     * (1 if pace["is_good"]     else 0)
    )

    ratio = score / GRADE_MAX

    if ratio >= 0.85:
        return "excellent"
    elif ratio >= 0.57:
        return "good"
    elif ratio >= 0.28:
        return "needs_work"
    else:
        return "unclear"


# =============================================================================
# Recommendation builder
# =============================================================================

# Dog-training principle mapped to (tone, scenario) pairs
# Indexed as (tone_label, loudness_label, duration_label) → advice paragraph
# Falls back to generic advice when no exact match.

def _build_recommendation(
    tone: dict, loudness: dict, dur: dict, pace: dict
) -> str:
    """
    Compose a single targeted paragraph of dog-friendly advice by combining
    the most important finding from each dimension.

    Priority: the worst-scoring dimension leads the advice.
    """
    findings = []

    # Tone is the lead signal
    if not tone["is_good"]:
        findings.append(tone["tip"])

    # Loudness second
    if not loudness["is_good"]:
        findings.append(loudness["tip"])

    # Duration third
    if not dur["is_good"]:
        findings.append(dur["tip"])

    # Pace last
    if not pace["is_good"]:
        findings.append(pace["tip"])

    if not findings:
        # Everything is good — give a reinforcement message
        return (
            "Great delivery overall. Your tone was {tone}, your volume was just right, "
            "and your command length was ideal. Keep being this consistent — "
            "dogs thrive on repetition and predictability. "
            "The more you deliver commands the same way, the faster they learn."
        ).format(tone=tone["label"])

    # Lead with the most important issue, add a positive closer
    main = findings[0]
    secondary = (" Also: " + findings[1]) if len(findings) > 1 else ""
    closer = _positive_closer(tone["label"], loudness["label"], dur["label"])

    return f"{main}{secondary} {closer}"


def _positive_closer(tone_label: str, loudness_label: str, dur_label: str) -> str:
    """Return a brief encouraging sentence to end the recommendation."""
    if dur_label == "ideal" and loudness_label == "just_right":
        return "Your length and volume are already solid — just work on the tone."
    if tone_label in ("calm", "firm", "upbeat"):
        return "Your tone foundation is good — small adjustments will make a big difference."
    return (
        "Remember: dogs don't understand words, they learn sounds. "
        "The clearer and more consistent you are, the faster they'll respond."
    )


# =============================================================================
# Example command suggester
# =============================================================================

# Maps tone label → example scenario → suggested command string
_EXAMPLES = {
    "calm": {
        "default":  '"Stay." — low, slow, one word. Hold eye contact.',
        "too_loud": '"Waaait." — drawn out, quiet, almost a murmur.',
        "too_long": '"Stay." — just that one word. Nothing before or after.',
    },
    "upbeat": {
        "default":   '"Come! Yes!" — bright, short, then immediate praise.',
        "too_soft":  '"Come!" — project it like you\'re calling across the park.',
        "too_long":  '"Come!" — one word, enthusiastic. That\'s the whole command.',
    },
    "firm": {
        "default":   '"Sit." — firm, neutral, once. Then wait.',
        "too_loud":  '"Sit." — firm but quiet. Like you\'re telling a fact, not a demand.',
        "too_long":  '"Sit." — one word. Don\'t explain yourself to the dog.',
    },
    "intense": {
        "default":   '"Sit." — slow it down. One word, steady breath before you say it.',
        "too_loud":  '"Sit." — half your current volume. Dogs hear everything.',
        "too_short": '"Siiit." — give it half a second, let the sound land.',
    },
    "unclear": {
        "default":   '"Sit." — record yourself saying just this one word clearly.',
        "too_soft":  '"Sit!" — project from your chest, not your throat.',
        "too_long":  '"Come!" — one sharp syllable. No sentences.',
    },
}


def _suggest_example(tone: dict, loudness: dict, dur: dict) -> str:
    """
    Return a short example command string the user can try immediately.
    Selects based on tone label + the worst-scoring secondary dimension.
    """
    tone_label     = tone["label"]
    loudness_label = loudness["label"]
    dur_label      = dur["label"]

    bank = _EXAMPLES.get(tone_label, _EXAMPLES["unclear"])

    # Pick the most specific key that matches a known problem
    if loudness_label in ("too_soft", "too_loud") and loudness_label in bank:
        return bank[loudness_label]
    if dur_label in ("too_short", "too_long") and dur_label in bank:
        return bank[dur_label]

    return bank.get("default", '"Sit." — firm, short, once.')


# =============================================================================
# Legacy compatibility shims
# (keep the sub-dict keys that translator.py and ui_helpers.py expect)
# =============================================================================

def _legacy_pitch_assessment(tone: dict, f0_mean: float, f0_std: float, f0_range: float) -> dict:
    """Produce the pitch_assessment dict shape that translator/ui_helpers expect."""
    # Map tone labels back to the pitch label vocabulary the UI uses
    label_map = {
        "calm":    "steady",
        "firm":    "steady",
        "upbeat":  "expressive",
        "intense": "inconsistent",
        "unclear": "undetectable",
    }
    return {
        "label":       label_map.get(tone["label"], "undetectable"),
        "f0_mean_hz":  round(f0_mean, 1),
        "f0_std_hz":   round(f0_std, 1),
        "f0_range_hz": round(f0_range, 1),
        "tip":         tone["tip"],
        "is_good":     tone["is_good"],
    }


def _legacy_loudness_assessment(loudness: dict, rms_mean: float) -> dict:
    """Produce the loudness_assessment dict shape that translator/ui_helpers expect."""
    return {
        "energy_level": loudness["label"],
        "consistency":  loudness["consistency"],
        "tip":          loudness["tip"],
        "is_good":      loudness["is_good"],
        "rms_mean":     round(rms_mean, 4),
    }


def _legacy_duration_assessment(dur: dict, duration_sec: float) -> dict:
    """Produce the duration_assessment dict shape that translator/ui_helpers expect."""
    return {
        "label":        dur["label"],
        "duration_sec": round(duration_sec, 2),
        "tip":          dur["tip"],
        "is_good":      dur["is_good"],
    }
