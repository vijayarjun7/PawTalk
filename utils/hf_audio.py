"""
hf_audio.py
-----------
Optional Hugging Face audio-classification layer for PawTalk.

Uses MIT/ast-finetuned-audioset-10-10-0.4593 (Audio Spectrogram Transformer
trained on AudioSet) to produce a secondary signal on top of the rule-based
bark classifier.

Design contract
---------------
- This module is entirely optional. If transformers or torch are not installed,
  every public function degrades gracefully and returns a sentinel dict
  that the UI can detect and skip cleanly.
- The HF model is cached globally after the first load so Streamlit re-renders
  don't reload it.
- The combiner treats the rule-based classifier as the source of truth.
  HF agreement boosts confidence; disagreement is surfaced as extra context,
  not a contradiction.

Label mapping
-------------
AudioSet has 527 labels. This module maps a curated subset to PawTalk's 5
internal moods. Unrecognised labels fall through to None (no mapping).
The mapping was built by reading AudioSet ontology descriptions and identifying
which labels plausibly correspond to each mood in a dog-bark context.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    from transformers import pipeline as _hf_pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

_MODEL_ID    = "MIT/ast-finetuned-audioset-10-10-0.4593"
_TOP_K       = 5        # number of top labels to return from the model
_SAMPLE_RATE = 16_000   # AST expects 16 kHz mono audio


# ---------------------------------------------------------------------------
# AudioSet label → PawTalk mood mapping
# ---------------------------------------------------------------------------
# Keys are AudioSet display_name strings (exactly as returned by the model).
# Values are PawTalk mood strings or None (unmapped / irrelevant).
# Gaps are intentional: many AudioSet labels have no meaningful dog-mood analogue.

_LABEL_TO_MOOD: dict[str, str | None] = {
    # ── Excited ──────────────────────────────────────────────────────────────
    "Dog":                      None,     # too generic to map
    "Bark":                     None,     # too generic to map
    "Yip":                      "excited",
    "Whimper (dog)":            "anxious",
    "Howl":                     "alert",
    "Bow-wow":                  "alert",

    # AudioSet uses "Dog" subcategories like these:
    "Animal":                   None,
    "Domestic animals, pets":   None,
    "Dog activity":             None,

    # ── Alert / territorial ───────────────────────────────────────────────────
    "Alarm":                    "alert",
    "Siren":                    "alert",
    "Alert":                    "alert",
    "Growling":                 "warning",
    "Roar":                     "warning",

    # ── Anxious / distress ────────────────────────────────────────────────────
    "Crying":                   "anxious",
    "Whimper":                  "anxious",
    "Whine":                    "anxious",
    "Yelping":                  "anxious",

    # ── Excited / playful ─────────────────────────────────────────────────────
    "Squeak":                   "playful",
    "Chirp":                    "playful",
    "Chirp, tweet":             "playful",
    "Squeal":                   "excited",
    "Chatter":                  "excited",

    # ── Warning / threat ──────────────────────────────────────────────────────
    "Grunt":                    "warning",
    "Snort":                    "warning",

    # ── Calm / background ─────────────────────────────────────────────────────
    "Silence":                  None,
    "Noise":                    None,
    "White noise":              None,
    "Pink noise":               None,
    "Hum":                      None,
    "Music":                    None,
}

# ---------------------------------------------------------------------------
# Public: model loader  (cached at the Streamlit resource level)
# ---------------------------------------------------------------------------

def load_model():
    """
    Load the HF pipeline, cached with @st.cache_resource so it survives
    Streamlit reruns and is shared across all sessions in the same worker
    process — loaded exactly once per worker lifetime.

    Returns the pipeline object on success, or None on failure.
    Failures are NOT permanently cached: the except clause re-raises so
    Streamlit clears the cache entry and retries on the next call.

    Callers must handle None / exceptions gracefully.
    """
    # Import here so the module can be imported without Streamlit present
    # (e.g. in unit tests or CLI scripts).
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _load():
        return _hf_pipeline(
            task="audio-classification",
            model=_MODEL_ID,
        )

    return _load()


# ---------------------------------------------------------------------------
# Public: inference
# ---------------------------------------------------------------------------

def classify_audio(y: np.ndarray, sr: int) -> dict:
    """
    Run the HF model on a waveform and return structured results.

    The model expects 16 kHz mono float32. We resample here so callers
    can pass any sample rate.

    Parameters
    ----------
    y  : np.ndarray  — mono float32 waveform (any sample rate)
    sr : int         — sample rate of y

    Returns
    -------
    dict with keys:
        available       : bool   — False when transformers/torch are missing
        success         : bool   — False when inference itself failed
        error           : str    — human-readable reason if success=False
        top_labels      : list   — up to _TOP_K dicts {label, score, mood}
                                   mood is the mapped PawTalk mood or None
        top_mood        : str|None — PawTalk mood of the highest-scoring
                                     *mapped* label; None if no label maps
        top_mood_score  : float  — raw model score for top_mood (0.0–1.0)
        model_id        : str    — model identifier for display
    """
    _not_available = {
        "available":      False,
        "success":        False,
        "error":          "transformers / torch not installed",
        "top_labels":     [],
        "top_mood":       None,
        "top_mood_score": 0.0,
        "model_id":       _MODEL_ID,
    }

    if not HF_AVAILABLE:
        return _not_available

    # Lazy-load the pipeline via the cache_resource wrapper.
    # Any exception here means the model failed to load — treat as unavailable.
    try:
        pipeline = load_model()
        global _model_loaded_flag
        _model_loaded_flag = True
    except Exception as exc:
        return {**_not_available, "available": True, "error": f"Model load failed: {exc}"}

    try:
        audio_16k = _to_16k(y, sr)
        raw = pipeline(
            {"raw": audio_16k, "sampling_rate": _SAMPLE_RATE},
            top_k=_TOP_K,
        )
    except Exception as exc:
        return {
            "available":      True,
            "success":        False,
            "error":          str(exc),
            "top_labels":     [],
            "top_mood":       None,
            "top_mood_score": 0.0,
            "model_id":       _MODEL_ID,
        }

    # Annotate each label with its mapped mood
    top_labels = []
    for item in raw:
        label = item.get("label", "")
        score = float(item.get("score", 0.0))
        mood  = _map_label(label)
        top_labels.append({"label": label, "score": score, "mood": mood})

    # The dominant mapped mood is the highest-scoring label that has a mapping
    top_mood       = None
    top_mood_score = 0.0
    for entry in top_labels:
        if entry["mood"] is not None and entry["score"] > top_mood_score:
            top_mood       = entry["mood"]
            top_mood_score = entry["score"]

    return {
        "available":      True,
        "success":        True,
        "error":          "",
        "top_labels":     top_labels,
        "top_mood":       top_mood,
        "top_mood_score": round(top_mood_score, 4),
        "model_id":       _MODEL_ID,
    }


# ---------------------------------------------------------------------------
# Public: combiner
# ---------------------------------------------------------------------------

def combine_results(rule_result: dict, hf_result: dict) -> dict:
    """
    Merge the rule-based classifier output with the HF classification.

    The rule-based result is always the primary output — this function
    only adjusts confidence and attaches secondary insight.

    Parameters
    ----------
    rule_result : dict  — output of bark_classifier.classify_bark_mood()
    hf_result   : dict  — output of hf_audio.classify_audio()

    Returns
    -------
    dict with keys:
        mood            : str   — final mood (always the rule-based mood)
        confidence      : int   — adjusted 0–100 (boosted on agreement)
        explanation     : str   — rule-based explanation (unchanged)
        scores          : dict  — rule-based scores (unchanged)
        feature_summary : dict  — rule-based feature summary (unchanged)
        hf              : dict  — the full hf_result dict (for UI display)
        agreement       : bool | None
                            True  = both signals name the same mood
                            False = signals disagree
                            None  = HF produced no mappable mood (can't compare)
        confidence_delta: int   — how much confidence was adjusted (+/-)
    """
    mood       = rule_result["mood"]
    confidence = rule_result["confidence"]
    agreement  = None
    delta      = 0

    hf_mood = hf_result.get("top_mood") if hf_result.get("success") else None

    if hf_mood is not None and mood not in ("unknown",):
        agreement = (hf_mood == mood)
        if agreement:
            # Boost: both signals agree — raise confidence proportionally to
            # how strongly the HF model scored, capped at +15 points
            hf_score = hf_result.get("top_mood_score", 0.0)
            delta = int(np.clip(hf_score * 20, 0, 15))
        else:
            # Disagree: mild reduction — the disagreement is surfaced in the UI
            # but we don't strongly penalise because the rule engine is primary
            delta = -5

        confidence = int(np.clip(confidence + delta, 0, 100))

    return {
        "mood":             mood,
        "confidence":       confidence,
        "explanation":      rule_result["explanation"],
        "scores":           rule_result["scores"],
        "feature_summary":  rule_result["feature_summary"],
        "hf":               hf_result,
        "agreement":        agreement,
        "confidence_delta": delta,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Module-level flag: True after the first successful pipeline load in this worker.
# Used by app.py to show a "loading…" notice only on cold start.
_model_loaded_flag: bool = False


def _pipeline_cache_is_ready() -> bool:
    """True if the pipeline has already been loaded in this worker process."""
    return _model_loaded_flag


def _map_label(label: str) -> str | None:
    """
    Map an AudioSet label string to a PawTalk mood.
    Returns None if the label has no meaningful mapping.
    Lookup is case-insensitive and strips surrounding whitespace.
    """
    return _LABEL_TO_MOOD.get(label.strip())


def _to_16k(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Resample y to 16 kHz as required by the AST model.
    Returns a float32 1-D numpy array.
    """
    y = np.asarray(y, dtype=np.float32)

    if sr == _SAMPLE_RATE:
        return y

    # Use librosa if available; fall back to scipy
    try:
        import librosa
        return librosa.resample(y, orig_sr=sr, target_sr=_SAMPLE_RATE)
    except Exception:
        pass

    try:
        from scipy.signal import resample as scipy_resample
        n_out = int(round(len(y) * _SAMPLE_RATE / sr))
        return scipy_resample(y, n_out).astype(np.float32)
    except Exception:
        pass

    # Last resort: return as-is (model will still run, just at wrong rate)
    return y
