"""
utils/ai_bark_classifier.py
-----------------------------
Supervised AI inference module for PawTalk's bark classifier.

Loads the fine-tuned Wav2Vec2 checkpoint (checkpoints/best_model.pt) and
runs inference on a raw waveform.  Returns a result dict with the same
shape as bark_classifier.classify_bark_mood() so app.py can treat both
interchangeably.

Design contract
---------------
- Entirely optional: if torch / transformers are missing, or if no checkpoint
  exists, AI_AVAILABLE is False and every function degrades gracefully.
- The checkpoint is loaded once per worker (via @st.cache_resource) and
  reused across Streamlit reruns.
- Confidence is calibrated from the softmax top-1 probability:
    ≥ 0.70  → strong signal  (passes to app as primary result)
    0.45–0.70 → moderate    (passes, but UI shows caveat)
    < 0.45  → uncertain     (caller should fall back to rule engine)

Public API
----------
AI_AVAILABLE           : bool
AI_MODEL_LOADED        : bool  — True after first successful checkpoint load
load_ai_model()        : returns the loaded model or None
classify_bark_ai(y, sr): main inference entry point
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from transformers import Wav2Vec2Model  # noqa: F401 — import check only
    _TRANSFORMERS = True
except Exception:
    _TRANSFORMERS = False

AI_AVAILABLE: bool = _TORCH and _TRANSFORMERS

# ---------------------------------------------------------------------------
# Checkpoint location — relative to the PawTalk project root.
# The training script writes to checkpoints/best_model.pt by default.
# Override by calling set_checkpoint_path() before the first inference call.
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "checkpoints" / "best_model.pt"
_checkpoint_path: Path = _DEFAULT_CHECKPOINT

# Module-level flag: True once the checkpoint has been loaded successfully.
AI_MODEL_LOADED: bool = False

# Confidence thresholds
CONFIDENT_THRESHOLD  = 0.70   # above this → AI is primary
UNCERTAIN_THRESHOLD  = 0.45   # below this → fall back to rule engine

# Label list (must match training order)
_DEFAULT_LABELS = ["excited", "playful", "alert", "anxious", "warning"]

# Target sample rate expected by Wav2Vec2
_TARGET_SR = 16_000


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def set_checkpoint_path(path: str | Path) -> None:
    """Override the default checkpoint path before the first load."""
    global _checkpoint_path
    _checkpoint_path = Path(path)


def checkpoint_exists() -> bool:
    """True if the checkpoint file is present on disk."""
    return _checkpoint_path.exists()


# ---------------------------------------------------------------------------
# Model loader (cached at Streamlit resource level when running in app)
# ---------------------------------------------------------------------------

def load_ai_model():
    """
    Load the checkpoint and return the model + metadata, or None on failure.
    Uses @st.cache_resource when Streamlit is available so the model is only
    loaded once per worker process.
    Falls back to a plain module-level cache otherwise (CLI / test use).

    Returns
    -------
    dict with keys: model, label_names, config
    or None on any failure.
    """
    if not AI_AVAILABLE:
        return None
    if not checkpoint_exists():
        return None

    try:
        import streamlit as st
        return _load_with_streamlit_cache(st)
    except ImportError:
        return _load_plain()


# --- Streamlit-cached path ---

def _load_with_streamlit_cache(st):
    @st.cache_resource(show_spinner=False)
    def _inner():
        return _load_checkpoint()

    return _inner()


# --- Plain (non-Streamlit) path ---

_plain_cache = None


def _load_plain():
    global _plain_cache
    if _plain_cache is None:
        _plain_cache = _load_checkpoint()
    return _plain_cache


# --- Actual load logic ---

def _load_checkpoint():
    """
    Deserialise best_model.pt and rebuild the BarkClassifier.
    Returns a dict or raises on failure.
    """
    # Import here to keep the module importable without torch at the top level.
    from utils.ai_bark_classifier_model import BarkClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(str(_checkpoint_path), map_location=device, weights_only=False)

    label_map   = ckpt["label_map"]
    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    num_labels  = len(label_names)
    config      = ckpt.get("config", {})

    model = BarkClassifier(num_labels=num_labels, freeze_encoder=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return {"model": model, "label_names": label_names, "config": config, "device": device}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_bark_ai(y: np.ndarray, sr: int) -> dict:
    """
    Run AI inference on a mono waveform and return a structured result.

    Parameters
    ----------
    y  : np.ndarray  — mono float32 waveform (any sample rate)
    sr : int         — sample rate of y

    Returns
    -------
    dict with keys:
        available       : bool   — False when torch/transformers/checkpoint missing
        success         : bool   — False if inference itself failed
        error           : str    — human-readable reason (empty on success)
        mood            : str    — top-1 predicted mood (or 'unknown')
        confidence      : int    — 0–100
        uncertain       : bool   — True when confidence < UNCERTAIN_THRESHOLD * 100
        top2            : list[dict]  — [{"mood": str, "score": float}, ...]
        scores          : dict   — {mood: probability} for all labels
        explanation     : str    — brief plain-English explanation
        feature_summary : dict   — empty (AI path doesn't use librosa features)
        model_id        : str
    """
    global AI_MODEL_LOADED

    _unavailable = {
        "available":       False,
        "success":         False,
        "error":           "AI model unavailable (torch/transformers missing or no checkpoint)",
        "mood":            "unknown",
        "confidence":      0,
        "uncertain":       True,
        "top2":            [],
        "scores":          {},
        "explanation":     "",
        "feature_summary": {},
        "model_id":        str(_checkpoint_path),
    }

    if not AI_AVAILABLE:
        return _unavailable
    if not checkpoint_exists():
        return {**_unavailable, "error": f"Checkpoint not found: {_checkpoint_path}"}

    bundle = load_ai_model()
    if bundle is None:
        return {**_unavailable, "error": "Model failed to load"}

    AI_MODEL_LOADED = True
    model       = bundle["model"]
    label_names = bundle["label_names"]
    device      = bundle["device"]

    try:
        # Resample to 16 kHz
        y16 = _to_16k(y, sr)

        with torch.no_grad():
            input_tensor = torch.tensor(y16, dtype=torch.float32).unsqueeze(0).to(device)
            mask         = (input_tensor.abs() > 1e-8).long()
            logits       = model(input_tensor, attention_mask=mask)  # (1, num_labels)
            probs        = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    except Exception as exc:
        return {**_unavailable,
                "available": True,
                "error": f"Inference failed: {exc}"}

    # Build scores dict
    scores = {name: float(probs[i]) for i, name in enumerate(label_names)}

    # Top-2
    sorted_idx = np.argsort(probs)[::-1]
    top2 = [
        {"mood": label_names[idx], "score": float(probs[idx])}
        for idx in sorted_idx[:2]
    ]

    top1_idx   = int(sorted_idx[0])
    top1_mood  = label_names[top1_idx]
    top1_prob  = float(probs[top1_idx])
    confidence = int(round(top1_prob * 100))
    uncertain  = top1_prob < UNCERTAIN_THRESHOLD

    if uncertain:
        mood = "unknown"
    else:
        mood = top1_mood

    explanation = _build_explanation(top1_mood, top2, top1_prob, uncertain)

    return {
        "available":       True,
        "success":         True,
        "error":           "",
        "mood":            mood,
        "confidence":      confidence,
        "uncertain":       uncertain,
        "top2":            top2,
        "scores":          scores,
        "explanation":     explanation,
        "feature_summary": {},
        "model_id":        str(_checkpoint_path),
    }


# ---------------------------------------------------------------------------
# Combiner: merge AI result with rule-based fallback
# ---------------------------------------------------------------------------

def combine_ai_and_rule(ai_result: dict, rule_result: dict) -> dict:
    """
    Decide which result to use as the primary output and return a unified dict
    with the same shape as hf_audio.combine_results().

    Decision logic:
      - AI model unavailable or failed → use rule result unchanged
      - AI confident (prob ≥ CONFIDENT_THRESHOLD) → use AI as primary
      - AI moderate (UNCERTAIN ≤ prob < CONFIDENT) → blend (AI mood, rule confidence)
      - AI uncertain (prob < UNCERTAIN_THRESHOLD) → use rule result; attach AI as context

    The returned dict always has the same keys as bark_classifier output plus:
        ai              : dict  — the full ai_result
        source          : str   — "ai_confident" | "ai_moderate" | "rule_fallback"
    """
    ai_available = ai_result.get("available") and ai_result.get("success")

    if not ai_available or ai_result.get("uncertain"):
        # Rule engine wins; attach AI context for transparency
        return {
            **rule_result,
            "ai":     ai_result,
            "source": "rule_fallback",
        }

    ai_prob = ai_result["confidence"] / 100.0

    if ai_prob >= CONFIDENT_THRESHOLD:
        # AI is confident — use AI mood + confidence, but keep rule feature_summary
        return {
            "mood":            ai_result["mood"],
            "confidence":      ai_result["confidence"],
            "explanation":     ai_result["explanation"],
            "scores":          ai_result["scores"],
            "feature_summary": rule_result.get("feature_summary", {}),
            "ai":              ai_result,
            "source":          "ai_confident",
        }
    else:
        # AI moderate — use AI mood but blend confidence toward rule
        blended_conf = int(
            round(0.6 * ai_result["confidence"] + 0.4 * rule_result["confidence"])
        )
        # If both agree, nudge confidence up a bit
        if ai_result["mood"] == rule_result["mood"]:
            blended_conf = min(100, blended_conf + 5)

        return {
            "mood":            ai_result["mood"],
            "confidence":      blended_conf,
            "explanation":     ai_result["explanation"]
                               + f" Rule engine said: {rule_result['mood']}.",
            "scores":          ai_result["scores"],
            "feature_summary": rule_result.get("feature_summary", {}),
            "ai":              ai_result,
            "source":          "ai_moderate",
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_16k(y: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz. Uses librosa if available, scipy otherwise."""
    y = np.asarray(y, dtype=np.float32)
    if sr == _TARGET_SR:
        return y
    try:
        import librosa
        return librosa.resample(y, orig_sr=sr, target_sr=_TARGET_SR)
    except Exception:
        pass
    try:
        from scipy.signal import resample as sp_resample
        n_out = int(round(len(y) * _TARGET_SR / sr))
        return sp_resample(y, n_out).astype(np.float32)
    except Exception:
        pass
    return y  # last resort: return as-is


def _build_explanation(
    top_mood: str,
    top2: list[dict],
    top_prob: float,
    uncertain: bool,
) -> str:
    if uncertain:
        runner = top2[1]["mood"] if len(top2) > 1 else "—"
        return (
            f"The AI model is uncertain (top probability {top_prob:.0%}). "
            f"Leading guesses: {top2[0]['mood']} ({top2[0]['score']:.0%}) "
            f"and {runner} ({top2[1]['score']:.0%} if available). "
            "Falling back to the rule engine."
        )

    runner_str = ""
    if len(top2) > 1:
        runner_str = f" Runner-up: {top2[1]['mood']} ({top2[1]['score']:.0%})."
    return (
        f"AI classifier predicted '{top_mood}' with {top_prob:.0%} confidence.{runner_str}"
    )
