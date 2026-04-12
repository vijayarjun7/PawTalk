"""
bark_classifier.py
------------------
Rule-based dog bark mood classifier for PawTalk.

Design goals
------------
- Deterministic: same features always produce the same result.
- Readable: every mood's logic is in its own clearly-labelled block.
- Tweakable: all thresholds are named constants at the top of the file.
- Observable: the returned dict explains *why* a mood was chosen.

How it works
------------
1. Raw librosa feature values are normalised to [0, 1] so thresholds
   are dimensionless and easy to reason about.
2. Each mood is scored by a small weighted formula (0.0 – 1.0).
3. A secondary *veto* check can hard-block a mood when a required
   condition is clearly absent (e.g. "warning" requires high energy).
4. The winning mood is the highest-scoring non-vetoed mood.
5. Confidence (0–100) is computed from the margin between the winner
   and the runner-up, scaled by how strongly the winner scored.
6. An explanation string names the top-3 features that pushed the
   decision and their qualitative values.

Mood definitions
----------------
excited  – high energy, dense rapid barks, bright tone
playful  – moderate energy, rhythmic, moderate pitch
alert    – sudden energy bursts, sharp rises, high ZCR
anxious  – irregular energy, chaotic rhythm, frequent pauses
warning  – sustained loud energy, low-pitched, minimal pauses
"""

import numpy as np

# =============================================================================
# Tuneable thresholds
# Change these to adjust classifier sensitivity — do not change the formulas.
# =============================================================================

# --- Normalisation ranges (raw feature → 0..1) ---
RMS_LO,       RMS_HI       = 0.010,   0.30    # overall loudness
RMS_STD_LO,   RMS_STD_HI   = 0.000,   0.12    # loudness burstiness
ZCR_LO,       ZCR_HI       = 0.020,   0.35    # spectral noisiness
ZCR_STD_LO,   ZCR_STD_HI   = 0.000,   0.08    # ZCR variability
CENTROID_LO,  CENTROID_HI  = 400.0,  4500.0   # spectral brightness (Hz)
ROLLOFF_LO,   ROLLOFF_HI   = 800.0,  8000.0   # rolloff frequency (Hz)
TEMPO_LO,     TEMPO_HI     = 50.0,   220.0    # BPM
BEAT_REG_LO,  BEAT_REG_HI  = 0.00,    0.55    # IBI std dev; inf = fully irregular
F0_LO,        F0_HI        = 60.0,  2500.0    # fundamental frequency (Hz)
BURST_LO,     BURST_HI     = 0,       15      # burst count per clip
PAUSE_LO,     PAUSE_HI     = 0,       10      # pause count per clip

# --- Veto thresholds (normalised, 0–1 scale) ---
# A mood is vetoed when a *required* feature falls outside its expected range.
VETO_EXCITED_MIN_RMS    = 0.40   # excited needs real loudness
VETO_WARNING_MIN_RMS    = 0.50   # warning bark must be substantial
VETO_WARNING_MAX_F0     = 0.55   # warning bark should be low-pitched
VETO_PLAYFUL_MAX_RMS    = 0.85   # playful shouldn't be a full-on roar
VETO_ANXIOUS_MIN_VAR    = 0.20   # anxious needs some energy variation

# --- Confidence calibration ---
# confidence = base_score_weight * winner_score + margin_weight * margin
CONF_SCORE_WEIGHT  = 55    # how much the winner's raw score contributes (0–55)
CONF_MARGIN_WEIGHT = 45    # how much the gap over runner-up contributes (0–45)

# =============================================================================
# Public API
# =============================================================================

MOODS = ("excited", "playful", "alert", "anxious", "warning")


def classify_bark_mood(features: dict) -> dict:
    """
    Classify the mood of a dog bark.

    Parameters
    ----------
    features : dict
        Output of audio_features.extract_features().

    Returns
    -------
    dict with keys:
        mood        : str   — one of MOODS, or 'unknown'
        confidence  : int   — 0–100
        explanation : str   — plain-English reason for the prediction
        scores      : dict  — raw 0.0–1.0 score per mood (for debugging / charts)
        feature_summary : dict — human-readable normalised feature values
    """
    nf     = _normalise(features)
    scores = _score_all(nf)

    mood, confidence, explanation = _pick_winner(nf, scores, features)

    feature_summary = {
        "Energy":            _qual(nf["rms"],       ("quiet", "moderate", "loud")),
        "Energy variation":  _qual(nf["rms_std"],   ("steady", "variable", "spiky")),
        "Tone noisiness":    _qual(nf["zcr"],        ("tonal", "mixed", "noisy")),
        "Rhythm chaos":      _qual(nf["beat_reg"],  ("regular", "loose", "chaotic")),
        "Brightness":        _qual(nf["centroid"],  ("dark", "mid", "bright")),
        "Burst count":       str(features.get("burst_count", 0)),
        "Pause count":       str(features.get("pause_count", 0)),
    }

    return {
        "mood":            mood,
        "confidence":      confidence,
        "explanation":     explanation,
        "scores":          scores,
        "feature_summary": feature_summary,
    }


# =============================================================================
# Normalisation
# =============================================================================

def _normalise(features: dict) -> dict:
    """
    Map raw feature values to [0, 1] using the reference ranges above.
    beat_regularity = inf (no beats found) maps to 1.0 (maximally irregular).
    """
    def n(val, lo, hi):
        return float(np.clip((val - lo) / (hi - lo + 1e-9), 0.0, 1.0))

    beat_raw = features.get("beat_regularity", float("inf"))
    beat_norm = 1.0 if not np.isfinite(beat_raw) else n(beat_raw, BEAT_REG_LO, BEAT_REG_HI)

    return {
        "rms":      n(features.get("rms_mean",               0.0), RMS_LO,      RMS_HI),
        "rms_std":  n(features.get("rms_std",                0.0), RMS_STD_LO,  RMS_STD_HI),
        "zcr":      n(features.get("zcr_mean",               0.0), ZCR_LO,      ZCR_HI),
        "zcr_std":  n(features.get("zcr_std",                0.0), ZCR_STD_LO,  ZCR_STD_HI),
        "centroid": n(features.get("spectral_centroid_mean", 0.0), CENTROID_LO, CENTROID_HI),
        "rolloff":  n(features.get("spectral_rolloff_mean",  0.0), ROLLOFF_LO,  ROLLOFF_HI),
        "tempo":    n(features.get("tempo",                  0.0), TEMPO_LO,    TEMPO_HI),
        "beat_reg": beat_norm,
        "f0":       n(features.get("f0_mean",                0.0), F0_LO,       F0_HI),
        "burst":    n(features.get("burst_count",            0),   BURST_LO,    BURST_HI),
        "pause":    n(features.get("pause_count",            0),   PAUSE_LO,    PAUSE_HI),
    }


# =============================================================================
# Scoring
# Each mood formula is a weighted sum of soft-step activations.
# _ramp(v, t) → 0..1 as v rises above threshold t  (needs high value)
# _ramp(v, t, flip=True) → 0..1 as v falls below t (needs low value)
# The weights in each formula must sum to 1.0 for scores to stay in [0, 1].
# =============================================================================

def _ramp(value: float, threshold: float, flip: bool = False, span: float = 0.25) -> float:
    """
    Soft linear ramp from 0 to 1.

    flip=False (default): score rises as value climbs above threshold.
    flip=True           : score rises as value falls below threshold.
    span                : width of the transition zone (smaller = sharper).
    """
    if flip:
        return float(np.clip((threshold - value) / span, 0.0, 1.0))
    return float(np.clip((value - threshold) / span, 0.0, 1.0))


def _score_all(nf: dict) -> dict:
    """Return a 0.0–1.0 score for every mood."""
    return {mood: round(_SCORERS[mood](nf), 4) for mood in MOODS}


def _score_excited(nf: dict) -> float:
    """
    Excited bark: very loud, dense rapid bursts, bright and noisy tone.
    Typical: a dog at the door when owner arrives, or before a walk.

    Key features: high rms, many bursts, high zcr, bright centroid, fast tempo.
    """
    return (
        0.30 * _ramp(nf["rms"],     0.60)            # must be loud
        + 0.25 * _ramp(nf["burst"],  0.50)            # frequent short bursts
        + 0.20 * _ramp(nf["zcr"],    0.55)            # noisy / harmonic-rich
        + 0.15 * _ramp(nf["centroid"], 0.50)          # bright, high-frequency content
        + 0.10 * _ramp(nf["tempo"],  0.45)            # fast pace bonus
    )


def _score_playful(nf: dict) -> float:
    """
    Playful bark: moderate energy with a bouncy, regular rhythm.
    Typical: invite-to-play bark, fetch excitement, tug-of-war.

    Key features: moderate rms, regular beat pattern, moderate zcr, some bursts.
    """
    regularity = _ramp(nf["beat_reg"], 0.45, flip=True)   # low beat_reg = regular
    return (
        0.30 * regularity                                  # rhythmic is most important
        + 0.25 * _ramp(nf["rms"],    0.30)                # needs some energy
        + 0.20 * _ramp(nf["burst"],  0.25)                # a few bursts
        + 0.15 * _ramp(nf["zcr"],    0.35)                # moderate noisiness
        + 0.10 * _ramp(nf["tempo"],  0.30)                # moderate tempo
    )


def _score_alert(nf: dict) -> float:
    """
    Alert bark: sharp sudden bursts with quick silences between them,
    high energy variation, elevated ZCR.
    Typical: stranger approaching, unexpected noise, territorial response.

    Key features: high rms_std (spiky energy), many bursts, many pauses,
                  high zcr, bright centroid.
    """
    return (
        0.30 * _ramp(nf["rms_std"],  0.50)           # spiky — not sustained
        + 0.25 * _ramp(nf["burst"],   0.40)           # lots of short bursts
        + 0.20 * _ramp(nf["zcr"],     0.50)           # sharp, high-frequency
        + 0.15 * _ramp(nf["centroid"], 0.55)          # bright tone
        + 0.10 * _ramp(nf["pause"],   0.30)           # silences between barks
    )


def _score_anxious(nf: dict) -> float:
    """
    Anxious bark: chaotic irregular pattern, variable energy, many pauses
    mixed with whimper-like tonal fragments, unstable pitch.
    Typical: separation anxiety, thunderstorm fear, vet visit.

    Key features: high beat_reg (irregular), high rms_std, many pauses,
                  high zcr_std (tone keeps changing).
    """
    return (
        0.30 * _ramp(nf["beat_reg"], 0.55)           # irregular rhythm
        + 0.25 * _ramp(nf["rms_std"], 0.45)          # variable energy
        + 0.25 * _ramp(nf["pause"],   0.40)           # lots of gaps / hesitations
        + 0.20 * _ramp(nf["zcr_std"], 0.50)          # chaotic tone
    )


def _score_warning(nf: dict) -> float:
    """
    Warning / threat bark: sustained loud energy, low-pitched, deliberate
    and slow — more of a "woof" than rapid-fire barking.
    Typical: perceived intruder, territorial boundary warning.

    Key features: high rms (sustained), low f0 (low-pitched), low beat_reg
                  (regular slow pattern), low centroid (dark tone), few pauses.
    """
    low_pitch  = _ramp(nf["f0"],      0.40, flip=True)   # low fundamental freq
    low_bright = _ramp(nf["centroid"], 0.45, flip=True)   # dark tone
    return (
        0.30 * _ramp(nf["rms"],   0.55)              # sustained loudness
        + 0.25 * low_pitch                            # low fundamental frequency
        + 0.20 * low_bright                           # darker spectral content
        + 0.15 * _ramp(nf["beat_reg"], 0.45, flip=True)  # deliberate, regular
        + 0.10 * _ramp(nf["pause"],    0.30, flip=True)  # few gaps — sustained
    )


# Dispatch table — keeps _score_all() clean and makes adding moods trivial
_SCORERS = {
    "excited": _score_excited,
    "playful": _score_playful,
    "alert":   _score_alert,
    "anxious": _score_anxious,
    "warning": _score_warning,
}


# =============================================================================
# Winner selection, veto, confidence, explanation
# =============================================================================

def _pick_winner(nf: dict, scores: dict, raw: dict) -> tuple:
    """
    Choose the winning mood, apply veto rules, compute confidence, build explanation.

    Returns (mood, confidence_int, explanation_str).
    """
    # Sort moods from highest to lowest score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner_mood, winner_score = ranked[0]
    runner_mood, runner_score = ranked[1]

    # --- Veto checks: block a mood if a hard required condition is not met ---
    vetoed = _check_veto(winner_mood, nf)
    if vetoed:
        # Demote winner; try runner-up (apply veto to that too, just in case)
        if not _check_veto(runner_mood, nf):
            winner_mood, winner_score = runner_mood, runner_score
            # Third-place score for margin calculation (safe even with < 3 moods)
            runner_score = ranked[2][1] if len(ranked) > 2 else 0.0
        else:
            winner_mood, winner_score = "unknown", 0.0
            runner_score = 0.0

    # --- Confidence: blend of absolute score + margin over runner-up ---
    margin = winner_score - runner_score
    confidence = int(
        np.clip(
            winner_score * CONF_SCORE_WEIGHT + margin * CONF_MARGIN_WEIGHT * 2,
            0, 100
        )
    )

    # Suppress if score is too low to trust
    if winner_score < 0.20:
        winner_mood = "unknown"
        confidence  = max(confidence, 5)   # at least something, not zero

    explanation = _build_explanation(winner_mood, nf, scores, raw)

    return winner_mood, confidence, explanation


def _check_veto(mood: str, nf: dict) -> bool:
    """
    Return True if the mood should be vetoed based on hard conditions.
    These are intentionally narrow — only block when clearly wrong.
    """
    if mood == "excited" and nf["rms"] < VETO_EXCITED_MIN_RMS:
        return True   # excited bark must be genuinely loud

    if mood == "warning":
        if nf["rms"] < VETO_WARNING_MIN_RMS:
            return True   # warning bark must have real volume
        if nf["f0"] > VETO_WARNING_MAX_F0:
            return True   # warning bark must be relatively low-pitched

    if mood == "playful" and nf["rms"] > VETO_PLAYFUL_MAX_RMS:
        return True   # a near-maximal roar is not playful

    if mood == "anxious" and nf["rms_std"] < VETO_ANXIOUS_MIN_VAR:
        return True   # anxious bark must have some energy variation

    return False


# =============================================================================
# Explanation builder
# =============================================================================

# Qualitative descriptors used in explanations — indexed by normalised value
_LEVELS = {
    "rms":      [(0.35, "quiet"),     (0.65, "moderate"),   (1.0, "loud")],
    "rms_std":  [(0.35, "steady"),    (0.65, "variable"),   (1.0, "spiky")],
    "zcr":      [(0.35, "tonal"),     (0.65, "mixed-tone"), (1.0, "noisy")],
    "zcr_std":  [(0.35, "consistent"),(0.65, "shifting"),   (1.0, "chaotic")],
    "centroid": [(0.35, "dark"),      (0.65, "mid-range"),  (1.0, "bright")],
    "beat_reg": [(0.35, "regular"),   (0.65, "loose"),      (1.0, "irregular")],
    "tempo":    [(0.35, "slow"),      (0.65, "moderate"),   (1.0, "fast")],
    "f0":       [(0.35, "low-pitch"), (0.65, "mid-pitch"),  (1.0, "high-pitch")],
    "burst":    [(0.35, "few bursts"),(0.65, "some bursts"),(1.0, "many bursts")],
    "pause":    [(0.35, "few pauses"),(0.65, "some pauses"),(1.0, "many pauses")],
}

# Which features matter most for each mood, in priority order
_MOOD_FEATURES = {
    "excited": ["rms",     "burst",   "zcr",     "centroid"],
    "playful": ["beat_reg","rms",     "burst",   "zcr"],
    "alert":   ["rms_std", "burst",   "zcr",     "centroid"],
    "anxious": ["beat_reg","rms_std", "pause",   "zcr_std"],
    "warning": ["rms",     "f0",      "centroid","beat_reg"],
    "unknown": ["rms",     "rms_std", "zcr"],
}

# Short phrases describing what each feature's level means for that mood
_MOOD_FEATURE_PHRASES = {
    "excited": {
        "rms":      {"quiet": "low energy for an excited bark",
                     "moderate": "decent energy level",
                     "loud": "strong loud energy"},
        "burst":    {"few bursts": "bark pattern is sparse",
                     "some bursts": "moderate burst activity",
                     "many bursts": "rapid-fire burst pattern"},
        "zcr":      {"tonal": "relatively smooth tone",
                     "mixed-tone": "mixed harmonic content",
                     "noisy": "dense noisy texture"},
        "centroid": {"dark": "darker tonal quality",
                     "mid-range": "mid-range brightness",
                     "bright": "bright high-frequency content"},
    },
    "playful": {
        "beat_reg": {"regular": "nicely rhythmic pattern",
                     "loose": "loosely rhythmic",
                     "irregular": "quite irregular rhythm"},
        "rms":      {"quiet": "gentle energy",
                     "moderate": "energetic but not intense",
                     "loud": "high energy"},
        "burst":    {"few bursts": "few energy spikes",
                     "some bursts": "playful burst pattern",
                     "many bursts": "lots of activity"},
        "zcr":      {"tonal": "pure, clear tone",
                     "mixed-tone": "typical bark texture",
                     "noisy": "rough textured bark"},
    },
    "alert": {
        "rms_std":  {"steady": "energy is even — not spiky",
                     "variable": "noticeable energy variation",
                     "spiky": "sharp energy spikes"},
        "burst":    {"few bursts": "not many sudden bursts",
                     "some bursts": "some sharp bursts",
                     "many bursts": "rapid sharp burst pattern"},
        "zcr":      {"tonal": "relatively smooth",
                     "mixed-tone": "mixed tone content",
                     "noisy": "sharp high-frequency content"},
        "centroid": {"dark": "darker than typical alert bark",
                     "mid-range": "moderate brightness",
                     "bright": "sharp bright tone"},
    },
    "anxious": {
        "beat_reg": {"regular": "surprisingly regular for anxious",
                     "loose": "somewhat irregular rhythm",
                     "irregular": "very chaotic rhythm"},
        "rms_std":  {"steady": "unusually steady energy",
                     "variable": "noticeable energy swings",
                     "spiky": "very erratic energy"},
        "pause":    {"few pauses": "few hesitations",
                     "some pauses": "gaps and hesitations present",
                     "many pauses": "lots of stops and starts"},
        "zcr_std":  {"consistent": "stable tone",
                     "shifting": "shifting tone character",
                     "chaotic": "highly unstable tone"},
    },
    "warning": {
        "rms":      {"quiet": "quieter than a typical warning",
                     "moderate": "solid sustained volume",
                     "loud": "powerful sustained volume"},
        "f0":       {"low-pitch": "deep low-pitched bark",
                     "mid-pitch": "moderate pitch",
                     "high-pitch": "higher-pitched than typical warning"},
        "centroid": {"dark": "dark low-frequency tone",
                     "mid-range": "moderate tonal quality",
                     "bright": "brighter than typical warning"},
        "beat_reg": {"regular": "slow deliberate pattern",
                     "loose": "mostly deliberate",
                     "irregular": "more chaotic than typical warning"},
    },
    "unknown": {
        "rms":      {"quiet": "very low energy", "moderate": "moderate energy",
                     "loud": "high energy"},
        "rms_std":  {"steady": "steady", "variable": "variable", "spiky": "erratic"},
        "zcr":      {"tonal": "tonal", "mixed-tone": "mixed", "noisy": "noisy"},
    },
}


def _build_explanation(mood: str, nf: dict, scores: dict, raw: dict) -> str:
    """
    Produce a plain-English sentence explaining the top contributing features.
    """
    if mood == "unknown":
        return (
            "The audio features don't strongly match any known bark pattern — "
            f"top scores were {_top_scores_str(scores)}. "
            "Try a cleaner recording with less background noise."
        )

    feature_keys = _MOOD_FEATURES.get(mood, list(nf.keys())[:3])
    phrases      = _MOOD_FEATURE_PHRASES.get(mood, {})

    parts = []
    for key in feature_keys[:3]:           # use top 3 features in the explanation
        level = _level_label(key, nf.get(key, 0.0))
        phrase = phrases.get(key, {}).get(level, f"{key} is {level}")
        parts.append(phrase)

    # Extra context for burst/pause counts
    burst_n = raw.get("burst_count", 0)
    pause_n = raw.get("pause_count", 0)
    rhythm_note = ""
    if mood in ("alert", "excited") and burst_n > 0:
        rhythm_note = f" ({burst_n} burst{'s' if burst_n != 1 else ''} detected)"
    elif mood == "anxious" and pause_n > 0:
        rhythm_note = f" ({pause_n} pause{'s' if pause_n != 1 else ''} detected)"

    runner_up = _runner_up(mood, scores)
    runner_note = f" Closest alternative was '{runner_up}'." if runner_up else ""

    return (
        f"Classified as {mood}: {'; '.join(parts)}{rhythm_note}.{runner_note}"
    )


def _runner_up(winner: str, scores: dict) -> str:
    """Return the mood with the second-highest score."""
    others = {m: s for m, s in scores.items() if m != winner}
    if not others:
        return ""
    return max(others, key=others.get)


def _top_scores_str(scores: dict) -> str:
    """Format top 2 scores for display in the unknown explanation."""
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
    return ", ".join(f"{m} ({s:.2f})" for m, s in top)


# =============================================================================
# Utility helpers
# =============================================================================

def _level_label(feature_key: str, norm_value: float) -> str:
    """Map a normalised feature value to its qualitative label."""
    levels = _LEVELS.get(feature_key, [(0.5, "low"), (1.0, "high")])
    for threshold, label in levels:
        if norm_value <= threshold:
            return label
    return levels[-1][1]


def _qual(norm_value: float, labels: tuple) -> str:
    """
    Map a 0–1 normalised value to one of three qualitative labels.
    Used for the feature_summary dict shown in the UI.
    """
    lo, mid, hi = labels
    if norm_value < 0.35:
        return lo
    if norm_value < 0.65:
        return mid
    return hi
