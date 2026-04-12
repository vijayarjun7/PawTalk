"""
audio_features.py
-----------------
Foundation layer for all audio feature extraction in PawTalk.
Uses librosa where available; falls back to pure numpy/scipy.

Both bark_classifier and voice_analyzer import from this module —
it is the single source of truth for all numeric audio features.

Public API
----------
load_audio_from_uploaded_file(uploaded_file, sr) -> (y, sr)
load_audio_from_bytes(audio_bytes, sr)           -> (y, sr)
extract_features(y, sr)                          -> dict
"""

import io
import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except Exception:           # catches ImportError *and* numba/llvm init failures
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants — tune here rather than scattered through the code
# ---------------------------------------------------------------------------

SR_DEFAULT  = 22050   # target sample rate after loading
HOP_LENGTH  = 512     # frames hop — shared across all librosa calls for consistency
N_FFT       = 2048    # FFT window size

MIN_DURATION_SEC = 0.3    # clips shorter than this are rejected

# Pause/burst detection
ENERGY_SILENCE_PERCENTILE = 20    # RMS frames below this percentile → silence candidate
ENERGY_BURST_PERCENTILE   = 75    # RMS frames above this percentile → burst candidate
MIN_PAUSE_FRAMES  = 3             # consecutive silence frames required to count as a pause
MIN_BURST_FRAMES  = 2             # consecutive burst frames required to count as a burst

# Pitch detection frequency bounds (covers dogs ~200–4000 Hz and humans ~85–300 Hz)
F0_FMIN_HZ = 60.0
F0_FMAX_HZ = 4200.0


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class AudioLoadError(Exception):
    """Raised when an audio file cannot be decoded."""

class AudioTooShortError(Exception):
    """Raised when the clip is too short for reliable feature extraction."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_audio_from_uploaded_file(uploaded_file, sr: int = SR_DEFAULT):
    """
    Load audio directly from a Streamlit UploadedFile object.

    Rewinds the file pointer before reading so the same object can be
    passed to st.audio() beforehand without exhausting it.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
    sr            : target sample rate (default 22050)

    Returns
    -------
    y  : np.ndarray  — mono float32 waveform
    sr : int         — sample rate (== the requested sr)

    Raises
    ------
    AudioLoadError      — file is empty or undecodable
    """
    if uploaded_file is None:
        raise AudioLoadError("No file provided.")

    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if not raw:
        raise AudioLoadError("The uploaded file appears to be empty.")

    return load_audio_from_bytes(raw, sr=sr)


def load_audio_from_bytes(audio_bytes: bytes, sr: int = SR_DEFAULT):
    """
    Load audio from an in-memory bytes object.

    Strategy:
      1. Try soundfile — fast, handles WAV / FLAC / OGG natively.
      2. Fall back to librosa.load — handles MP3 via audioread/ffmpeg.
      Both paths convert stereo → mono and resample to the target sr.

    Returns
    -------
    y  : np.ndarray  — mono float32 waveform
    sr : int         — sample rate (== the requested sr)

    Raises
    ------
    AudioLoadError
    """
    if not audio_bytes:
        raise AudioLoadError("Empty audio data.")

    buf = io.BytesIO(audio_bytes)

    # ---- Attempt 1: soundfile ------------------------------------------------
    if SOUNDFILE_AVAILABLE:
        try:
            buf.seek(0)
            y, native_sr = sf.read(buf, dtype="float32", always_2d=True)
            y = _to_mono(y)                          # (samples, channels) → (samples,)
            y = _resample(y, native_sr, sr)
            return y.astype(np.float32), sr
        except Exception:
            pass  # fall through

    # ---- Attempt 2: librosa (needs audioread/ffmpeg for MP3) -----------------
    if LIBROSA_AVAILABLE:
        try:
            buf.seek(0)
            y, native_sr = librosa.load(buf, sr=None, mono=False)
            # librosa returns (channels, samples) for multi-channel
            if y.ndim == 2:
                y = _to_mono(y.T)                    # transpose to (samples, channels)
            y = _resample(y, native_sr, sr)
            return y.astype(np.float32), sr
        except Exception as exc:
            raise AudioLoadError(f"Could not decode audio file: {exc}") from exc

    raise AudioLoadError(
        "Neither soundfile nor librosa is installed. "
        "Run: pip install soundfile librosa"
    )


# ---------------------------------------------------------------------------
# Feature extraction — public entry point
# ---------------------------------------------------------------------------

def extract_features(y: np.ndarray, sr: int) -> dict:
    """
    Compute all audio features used by bark_classifier and voice_analyzer.

    Each feature group is extracted independently; a failure in one group
    does not prevent the others from running — the failed keys fall back to
    their zero values so downstream code always receives a complete dict.

    Parameters
    ----------
    y  : np.ndarray  — mono float32/float64 waveform
    sr : int         — sample rate

    Returns
    -------
    dict with keys:
        duration_sec             float
        rms_mean                 float   — mean RMS energy (overall loudness proxy)
        rms_std                  float   — RMS std dev (energy burstiness)
        zcr_mean                 float   — mean zero-crossing rate
        zcr_std                  float   — ZCR std dev
        spectral_centroid_mean   float   — mean spectral centroid in Hz (brightness)
        spectral_centroid_std    float   — centroid variation
        spectral_rolloff_mean    float   — freq below which 85% of energy falls
        spectral_bandwidth_mean  float   — spectral spread
        tempo                    float   — estimated BPM (0.0 if undetectable)
        beat_regularity          float   — IBI std dev; inf = no beats found
        f0_mean                  float   — mean voiced F0 in Hz (0.0 if undetectable)
        f0_std                   float   — F0 std dev
        f0_range                 float   — F0 max − min
        pause_count              int     — number of detected silent gaps
        burst_count              int     — number of detected energy bursts
        mfcc_mean                ndarray — shape (13,) mean MFCCs

    Raises
    ------
    AudioTooShortError  — if duration < MIN_DURATION_SEC
    """
    y = np.asarray(y, dtype=np.float64)
    duration = len(y) / sr

    if duration < MIN_DURATION_SEC:
        raise AudioTooShortError(
            f"Audio is only {duration:.2f}s — please upload at least "
            f"{MIN_DURATION_SEC}s of audio for a meaningful reading."
        )

    base = _zero_feature_dict(duration)   # safe defaults for every key

    if LIBROSA_AVAILABLE:
        _fill_librosa(base, y, sr)
    else:
        _fill_scipy(base, y, sr)

    return base


# ---------------------------------------------------------------------------
# Private: librosa-backed extraction
# Each block is try/except'd individually so one bad call can't kill the dict
# ---------------------------------------------------------------------------

def _fill_librosa(d: dict, y: np.ndarray, sr: int) -> None:
    """Populate feature dict using librosa. Modifies d in place."""

    # --- RMS energy ---
    try:
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        d["rms_mean"] = float(np.mean(rms))
        d["rms_std"]  = float(np.std(rms))
    except Exception:
        rms = None

    # --- Zero-crossing rate ---
    try:
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
        d["zcr_mean"] = float(np.mean(zcr))
        d["zcr_std"]  = float(np.std(zcr))
    except Exception:
        pass

    # --- Spectral features ---
    try:
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        d["spectral_centroid_mean"] = float(np.mean(centroid))
        d["spectral_centroid_std"]  = float(np.std(centroid))
    except Exception:
        pass

    try:
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        d["spectral_rolloff_mean"] = float(np.mean(rolloff))
    except Exception:
        pass

    try:
        bw = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        d["spectral_bandwidth_mean"] = float(np.mean(bw))
    except Exception:
        pass

    # --- Tempo & beat regularity ---
    try:
        tempo_val, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=HOP_LENGTH
        )
        d["tempo"] = float(tempo_val) if np.isscalar(tempo_val) else float(tempo_val[0])
        d["beat_regularity"] = _beat_regularity(beat_frames, sr)
    except Exception:
        pass

    # --- Pitch (F0) via pyin, falling back to yin ---
    try:
        f0_mean, f0_std, f0_range = _extract_f0_librosa(y, sr)
        d["f0_mean"]  = f0_mean
        d["f0_std"]   = f0_std
        d["f0_range"] = f0_range
    except Exception:
        pass

    # --- Pause & burst detection ---
    try:
        rms_signal = rms if rms is not None else librosa.feature.rms(
            y=y, hop_length=HOP_LENGTH
        )[0]
        pauses, bursts = _detect_pauses_and_bursts(rms_signal)
        d["pause_count"] = pauses
        d["burst_count"] = bursts
    except Exception:
        pass

    # --- MFCCs ---
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
        d["mfcc_mean"] = np.mean(mfcc, axis=1).astype(np.float32)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Private: scipy-only fallback
# ---------------------------------------------------------------------------

def _fill_scipy(d: dict, y: np.ndarray, sr: int) -> None:
    """Populate feature dict using only numpy/scipy. Modifies d in place."""
    from scipy.signal import welch

    frame_len = HOP_LENGTH
    n_frames  = len(y) // frame_len
    if n_frames == 0:
        return

    frames = y[: n_frames * frame_len].reshape(n_frames, frame_len)

    # --- RMS ---
    try:
        rms_frames = np.sqrt(np.mean(frames ** 2, axis=1))
        d["rms_mean"] = float(np.mean(rms_frames))
        d["rms_std"]  = float(np.std(rms_frames))
    except Exception:
        rms_frames = None

    # --- ZCR ---
    try:
        sign_changes = np.abs(np.diff(np.sign(frames), axis=1))
        zcr_frames = sign_changes.mean(axis=1) / 2.0
        d["zcr_mean"] = float(np.mean(zcr_frames))
        d["zcr_std"]  = float(np.std(zcr_frames))
    except Exception:
        pass

    # --- Spectral features via Welch PSD ---
    try:
        freqs, psd = welch(y, fs=sr, nperseg=N_FFT)
        total_power = np.sum(psd) + 1e-10

        centroid = float(np.sum(freqs * psd) / total_power)
        d["spectral_centroid_mean"] = centroid

        cumulative = np.cumsum(psd)
        rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
        d["spectral_rolloff_mean"] = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        d["spectral_bandwidth_mean"] = float(
            np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / total_power)
        )
    except Exception:
        pass

    # --- Pitch: autocorrelation-based F0 estimate ---
    try:
        f0_mean, f0_std, f0_range = _extract_f0_autocorr(y, sr)
        d["f0_mean"]  = f0_mean
        d["f0_std"]   = f0_std
        d["f0_range"] = f0_range
    except Exception:
        pass

    # --- Pause & burst detection ---
    try:
        if rms_frames is not None:
            pauses, bursts = _detect_pauses_and_bursts(rms_frames)
            d["pause_count"] = pauses
            d["burst_count"] = bursts
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Private: pitch helpers
# ---------------------------------------------------------------------------

def _extract_f0_librosa(y: np.ndarray, sr: int):
    """
    Estimate F0 using probabilistic YIN (pyin).
    pyin returns a voiced_flag array that cleanly excludes unvoiced frames —
    much more reliable than thresholding raw YIN output.

    Falls back to plain YIN if pyin is unavailable (older librosa).

    Returns (f0_mean, f0_std, f0_range) — all 0.0 if no voiced frames found.
    """
    fmin = librosa.note_to_hz("C2")   # ~65 Hz
    fmax = librosa.note_to_hz("C7")   # ~2093 Hz

    # Clamp to what's meaningful given the clip length
    frame_length = 2048
    min_frames = 4
    if len(y) < frame_length * min_frames:
        return 0.0, 0.0, 0.0

    # --- Try pyin first (librosa >= 0.8) ---
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=HOP_LENGTH,
            frame_length=frame_length,
        )
        voiced = f0[voiced_flag.astype(bool)]
        if len(voiced) > 0:
            return float(np.nanmean(voiced)), float(np.nanstd(voiced)), float(np.nanmax(voiced) - np.nanmin(voiced))
    except Exception:
        pass

    # --- Fall back to yin ---
    try:
        f0 = librosa.yin(
            y, fmin=fmin, fmax=fmax, sr=sr, hop_length=HOP_LENGTH
        )
        # yin doesn't give confidence — filter by clamping to plausible range
        voiced = f0[(f0 >= F0_FMIN_HZ) & (f0 <= F0_FMAX_HZ)]
        if len(voiced) > 0:
            return float(np.mean(voiced)), float(np.std(voiced)), float(np.max(voiced) - np.min(voiced))
    except Exception:
        pass

    return 0.0, 0.0, 0.0


def _extract_f0_autocorr(y: np.ndarray, sr: int):
    """
    Simple autocorrelation-based F0 estimate used in the scipy-only fallback.
    Operates on overlapping frames; picks the peak lag in the expected pitch range.

    Returns (f0_mean, f0_std, f0_range).
    """
    frame_len = 2048
    hop       = HOP_LENGTH
    lag_min   = int(sr / F0_FMAX_HZ)
    lag_max   = int(sr / max(F0_FMIN_HZ, 1.0))

    n_frames = max(0, (len(y) - frame_len) // hop)
    if n_frames < 2 or lag_max <= lag_min:
        return 0.0, 0.0, 0.0

    f0_list = []
    for i in range(n_frames):
        frame = y[i * hop : i * hop + frame_len]
        # Normalized autocorrelation
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[frame_len - 1:]          # keep lags >= 0
        corr = corr / (corr[0] + 1e-10)     # normalize

        region = corr[lag_min : lag_max + 1]
        if len(region) == 0:
            continue
        peak_lag = int(np.argmax(region)) + lag_min
        # Only keep frames where the autocorrelation peak is strong (voiced)
        if corr[peak_lag] > 0.5:
            f0_list.append(sr / peak_lag)

    if not f0_list:
        return 0.0, 0.0, 0.0

    arr = np.array(f0_list)
    return float(np.mean(arr)), float(np.std(arr)), float(arr.max() - arr.min())


# ---------------------------------------------------------------------------
# Private: pause & burst detection
# ---------------------------------------------------------------------------

def _detect_pauses_and_bursts(rms_frames: np.ndarray):
    """
    Count silent pauses and energy bursts from a per-frame RMS array.

    A pause  = run of >= MIN_PAUSE_FRAMES consecutive frames below the
               ENERGY_SILENCE_PERCENTILE of the clip's RMS distribution.
    A burst  = run of >= MIN_BURST_FRAMES consecutive frames above the
               ENERGY_BURST_PERCENTILE.

    These two counts together characterize rhythm and delivery style:
    - Many bursts + few pauses → rapid, intense barking (excited/alert)
    - Alternating bursts/pauses → rhythmic attention-seeking
    - Few bursts, many pauses → calm / sleepy

    Returns
    -------
    (pause_count, burst_count) : (int, int)
    """
    if len(rms_frames) == 0:
        return 0, 0

    silence_thresh = float(np.percentile(rms_frames, ENERGY_SILENCE_PERCENTILE))
    burst_thresh   = float(np.percentile(rms_frames, ENERGY_BURST_PERCENTILE))

    # Clamp edge case where the whole clip is near-silent
    if burst_thresh <= silence_thresh:
        return 0, 0

    silence_mask = rms_frames <= silence_thresh
    burst_mask   = rms_frames >= burst_thresh

    pause_count = _count_runs(silence_mask, MIN_PAUSE_FRAMES)
    burst_count = _count_runs(burst_mask,   MIN_BURST_FRAMES)

    return pause_count, burst_count


def _count_runs(mask: np.ndarray, min_len: int) -> int:
    """Count contiguous True runs of length >= min_len in a boolean array."""
    count = 0
    run = 0
    for val in mask:
        if val:
            run += 1
        else:
            if run >= min_len:
                count += 1
            run = 0
    if run >= min_len:
        count += 1
    return count


# ---------------------------------------------------------------------------
# Private: beat regularity
# ---------------------------------------------------------------------------

def _beat_regularity(beat_frames: np.ndarray, sr: int) -> float:
    """
    Std dev of inter-beat intervals in seconds.
    Lower = more rhythmic/regular. Returns inf if fewer than 2 beats detected.
    """
    if len(beat_frames) < 2:
        return float("inf")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    return float(np.std(np.diff(beat_times)))


# ---------------------------------------------------------------------------
# Private: conversion helpers
# ---------------------------------------------------------------------------

def _to_mono(y: np.ndarray) -> np.ndarray:
    """
    Convert (samples, channels) array to mono.
    Uses a perceptually-weighted average for stereo (ITU-R BS.775 approximation),
    and a plain mean for > 2 channels.
    """
    if y.ndim == 1:
        return y
    n_ch = y.shape[1]
    if n_ch == 1:
        return y[:, 0]
    if n_ch == 2:
        # Weighted: left 0.5, right 0.5 (equal power for typical stereo)
        return 0.5 * y[:, 0] + 0.5 * y[:, 1]
    return y.mean(axis=1)


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample y from orig_sr to target_sr. No-op if rates already match."""
    if orig_sr == target_sr:
        return y
    if LIBROSA_AVAILABLE:
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    # scipy fallback
    from scipy.signal import resample as scipy_resample
    n_out = int(round(len(y) * target_sr / orig_sr))
    return scipy_resample(y, n_out).astype(np.float32)


# ---------------------------------------------------------------------------
# Private: zero-valued default dict (safe starting point for all features)
# ---------------------------------------------------------------------------

def _zero_feature_dict(duration: float) -> dict:
    return {
        "duration_sec":            duration,
        "rms_mean":                0.0,
        "rms_std":                 0.0,
        "zcr_mean":                0.0,
        "zcr_std":                 0.0,
        "spectral_centroid_mean":  0.0,
        "spectral_centroid_std":   0.0,
        "spectral_rolloff_mean":   0.0,
        "spectral_bandwidth_mean": 0.0,
        "tempo":                   0.0,
        "beat_regularity":         float("inf"),
        "f0_mean":                 0.0,
        "f0_std":                  0.0,
        "f0_range":                0.0,
        "pause_count":             0,
        "burst_count":             0,
        "mfcc_mean":               np.zeros(13, dtype=np.float32),
    }
