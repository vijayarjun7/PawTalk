"""
training/prepare_dataset.py
----------------------------
Prepare a labeled dog bark audio dataset for training PawTalk's AI classifier.

Usage
-----
    python training/prepare_dataset.py \\
        --data_dir  data/raw            \\
        --out_dir   data/processed      \\
        --split     0.7 0.15 0.15       \\
        --target_sr 16000               \\
        --clip_sec  3.0

Expected input layout
---------------------
    data/raw/
        excited/   clip1.wav  clip2.mp3 …
        playful/   …
        alert/     …
        anxious/   …
        warning/   …

Sub-directories must be named exactly as the LABELS list below (case-insensitive).
Any audio format decodable by soundfile or librosa is accepted.

Output layout
-------------
    data/processed/
        train/
            excited/   0001.wav  0002.wav …
            playful/   …
        val/
            …
        test/
            …
        label_map.json      {"excited": 0, "playful": 1, …}
        dataset_stats.json  clip counts per split per label
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Label definitions live in label_config.json (same directory as this script)
# so that training and inference always use the same canonical set.
_LABEL_CONFIG_PATH = Path(__file__).parent / "label_config.json"

def _load_label_config() -> tuple[list[str], dict[str, int]]:
    if not _LABEL_CONFIG_PATH.exists():
        # Hard-coded fallback — should not happen if the repo is intact
        labels = ["excited", "playful", "alert", "anxious", "warning"]
        return labels, {l: i for i, l in enumerate(labels)}
    cfg = json.loads(_LABEL_CONFIG_PATH.read_text())
    labels   = cfg["labels"]
    lmap     = cfg["label_map"]
    return labels, lmap

LABELS, LABEL_MAP = _load_label_config()

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff"}

TARGET_SR      = 16_000   # Wav2Vec2 / AST both expect 16 kHz
CLIP_SEC       = 3.0      # fixed clip length in seconds
MIN_CLIP_SEC   = 0.3      # clips shorter than this are skipped

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import soundfile as sf
    _SF = True
except ImportError:
    _SF = False

try:
    import librosa
    _LIBROSA = True
except Exception:
    _LIBROSA = False

if not _SF and not _LIBROSA:
    print("ERROR: install soundfile or librosa before running this script.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def load_clip(path: Path, target_sr: int) -> np.ndarray | None:
    """
    Load any audio file to a mono float32 array at target_sr.
    Returns None if the file cannot be decoded.
    """
    # Attempt 1: soundfile (WAV / FLAC / OGG)
    if _SF:
        try:
            y, sr = sf.read(str(path), dtype="float32", always_2d=True)
            y = _to_mono(y)
            y = _resample(y, sr, target_sr)
            return y
        except Exception:
            pass

    # Attempt 2: librosa (MP3 + everything else via audioread)
    if _LIBROSA:
        try:
            y, sr = librosa.load(str(path), sr=None, mono=True)
            y = _resample(y, sr, target_sr)
            return y.astype(np.float32)
        except Exception:
            pass

    return None


def normalize_clip(y: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad (repeat-pad) or centre-crop the waveform to exactly target_len samples.
    Normalize peak amplitude to 0.9.
    """
    # Trim or pad
    if len(y) > target_len:
        # Centre crop
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
    elif len(y) < target_len:
        # Repeat-pad (safer than zero-pad for short clips)
        repeats = int(np.ceil(target_len / len(y)))
        y = np.tile(y, repeats)[:target_len]

    # Peak normalize
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y * (0.9 / peak)

    return y.astype(np.float32)


def _to_mono(y: np.ndarray) -> np.ndarray:
    """(samples, channels) → mono."""
    if y.ndim == 1:
        return y
    return y.mean(axis=1)


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    if _LIBROSA:
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    from scipy.signal import resample as scipy_resample
    n_out = int(round(len(y) * target_sr / orig_sr))
    return scipy_resample(y, n_out).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def collect_files(data_dir: Path) -> dict[str, list[Path]]:
    """
    Walk data_dir and collect all audio files grouped by label.
    Sub-directory name must match a label (case-insensitive).
    """
    files: dict[str, list[Path]] = {label: [] for label in LABELS}

    for sub in data_dir.iterdir():
        if not sub.is_dir():
            continue
        label = sub.name.lower()
        if label not in files:
            print(f"  [skip] Unknown label directory: {sub.name}")
            continue
        for fp in sorted(sub.iterdir()):
            if fp.suffix.lower() in AUDIO_EXTENSIONS:
                files[label].append(fp)

    return files


def split_files(
    files: dict[str, list[Path]],
    train_frac: float,
    val_frac: float,
    seed: int = 42,
) -> dict[str, dict[str, list[Path]]]:
    """
    Stratified train / val / test split.
    Returns {"train": {"excited": [...], ...}, "val": {...}, "test": {...}}.
    """
    rng = random.Random(seed)
    splits: dict[str, dict[str, list[Path]]] = {
        "train": {}, "val": {}, "test": {}
    }

    for label, paths in files.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n     = len(shuffled)
        n_tr  = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))

        splits["train"][label] = shuffled[:n_tr]
        splits["val"][label]   = shuffled[n_tr : n_tr + n_val]
        splits["test"][label]  = shuffled[n_tr + n_val :]

    return splits


def process_and_save(
    splits: dict,
    out_dir: Path,
    target_sr: int,
    clip_sec: float,
) -> dict:
    """
    Decode, normalize, and write WAV files to out_dir/<split>/<label>/<n>.wav.
    Returns a stats dict: {split: {label: count}}.
    """
    target_len = int(target_sr * clip_sec)
    stats: dict = {}

    for split_name, label_map in splits.items():
        stats[split_name] = {}
        for label, paths in label_map.items():
            out_label_dir = out_dir / split_name / label
            out_label_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for i, src_path in enumerate(paths):
                y = load_clip(src_path, target_sr)
                if y is None:
                    print(f"  [warn] Could not decode: {src_path}")
                    continue
                dur = len(y) / target_sr
                if dur < MIN_CLIP_SEC:
                    print(f"  [skip] Too short ({dur:.2f}s): {src_path.name}")
                    continue

                y_norm = normalize_clip(y, target_len)

                out_path = out_label_dir / f"{i:05d}.wav"
                if _SF:
                    sf.write(str(out_path), y_norm, target_sr, subtype="PCM_16")
                else:
                    # scipy fallback — write 16-bit PCM WAV manually
                    import scipy.io.wavfile as wf
                    pcm = (y_norm * 32767).astype(np.int16)
                    wf.write(str(out_path), target_sr, pcm)

                saved += 1

            stats[split_name][label] = saved
            print(f"  {split_name}/{label}: {saved} clips")

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare dog bark dataset for PawTalk AI classifier training."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root directory containing per-label sub-directories of audio files.",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed",
        help="Where to write the processed dataset (default: data/processed).",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train / val / test fractions (must sum to ~1.0). Default: 0.70 0.15 0.15",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=TARGET_SR,
        help=f"Target sample rate in Hz (default: {TARGET_SR}).",
    )
    parser.add_argument(
        "--clip_sec",
        type=float,
        default=CLIP_SEC,
        help=f"Fixed clip length in seconds — pads/crops to this (default: {CLIP_SEC}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    if not data_dir.is_dir():
        print(f"ERROR: --data_dir '{data_dir}' does not exist.")
        sys.exit(1)

    train_frac, val_frac, test_frac = args.split
    if abs(train_frac + val_frac + test_frac - 1.0) > 0.01:
        print(f"WARNING: split fractions sum to {train_frac + val_frac + test_frac:.3f}, not 1.0")

    print(f"\nScanning: {data_dir}")
    files = collect_files(data_dir)

    total = sum(len(v) for v in files.values())
    if total == 0:
        print("No audio files found. Check --data_dir and subdirectory names.")
        sys.exit(1)

    print(f"Found {total} files across {sum(1 for v in files.values() if v)} labels")
    for label, paths in files.items():
        print(f"  {label}: {len(paths)} files")

    splits = split_files(files, train_frac, val_frac, seed=args.seed)

    print(f"\nProcessing → {out_dir}")
    stats = process_and_save(splits, out_dir, args.target_sr, args.clip_sec)

    # Save label map and stats alongside the data
    label_map_path = out_dir / "label_map.json"
    label_map_path.write_text(json.dumps(LABEL_MAP, indent=2))
    print(f"\nLabel map  → {label_map_path}")

    stats_path = out_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Stats      → {stats_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
