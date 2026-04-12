---
title: PawTalk
emoji: 🐾
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# PawTalk 🐾

**Dog Bark Mood Analyzer & Voice Command Coach**

A local Streamlit app that listens to dog bark audio and makes a playful guess about what your dog might be feeling — and coaches you on how to deliver clearer training commands. No cloud APIs, no subscriptions, no account required.

> **Honest caveat:** PawTalk is a gift app built on audio signal analysis. It is not a dog-language translator. Mood estimates are acoustic guesses, not behavioral science. Results are for fun and inspiration only.

**Live app → [pawtalk-buddy.streamlit.app](https://pawtalk-buddy.streamlit.app)**
&nbsp;&nbsp;|&nbsp;&nbsp;
**HF Space → [huggingface.co/spaces/Vijayarv07/PawTalk](https://huggingface.co/spaces/Vijayarv07/PawTalk)**
&nbsp;&nbsp;|&nbsp;&nbsp;
**GitHub → [vijayarjun7/PawTalk](https://github.com/vijayarjun7/PawTalk)**

---

## 🎁 Why This Exists

This project was built as a personalized gift — combining audio analysis, a rule-based classifier, a Hugging Face secondary signal, and deliberately playful UX.

The goal is not accuracy. The goal is experience.

---

## 🚀 Quick Demo

1. Upload a dog bark audio file (WAV recommended, 2–10 seconds)
2. Get a mood prediction — excited, alert, anxious, playful, or warning
3. Read the playful "translation" in your chosen style
4. Optionally upload your own voice to get dog-friendly command feedback

No account. No cloud. Runs locally.

---

## Features

### 🐕 Dog → Human: Bark Analyzer

Upload a bark recording and PawTalk estimates one of five moods:

| Mood | What it sounds like |
|---|---|
| **Excited** | High energy, rapid bursts, bright tone |
| **Playful** | Moderate energy, bouncy rhythm |
| **Alert** | Sharp spikes, quick silences, elevated pitch |
| **Anxious** | Irregular energy, many pauses, chaotic rhythm |
| **Warning** | Sustained loud energy, low-pitched, deliberate |

Each result includes:
- A playful translated message in your chosen style (funny / cute / emotional)
- A fun dog behavior fact
- A confidence meter explaining how strong the signal was
- An audio feature summary (energy, rhythm, brightness, burst count)
- An optional **Audio Model Insight** panel showing secondary Hugging Face labels

### 🗣️ Human → Dog: Voice Command Coach

Upload a clip of yourself saying a command (e.g. "Sit", "Stay", "Come"). PawTalk evaluates four dimensions of your delivery:

| Dimension | What it measures |
|---|---|
| **Tone** | Calm / upbeat / firm / intense / unclear |
| **Volume** | Too soft / just right / too loud |
| **Duration** | Too short / ideal / slightly long / too long |
| **Pace** | Slow / moderate / rapid-fire |

Returns an overall grade (Excellent → Unclear), a targeted recommendation, a suggested example command, and specific tips per dimension.

---

## How It Works

### 1. Audio feature extraction (`utils/audio_features.py`)

Every uploaded clip passes through librosa to extract a fixed set of numeric features:

- **RMS energy** — overall loudness and its variation
- **Zero-crossing rate** — spectral noisiness / tonality
- **Spectral centroid & rolloff** — brightness and harmonic content
- **Tempo & beat regularity** — rhythm and delivery pace
- **F0 (pitch)** — fundamental frequency via probabilistic YIN
- **Pause count & burst count** — detected silence gaps and energy spikes

If librosa is unavailable (e.g. numba wheel missing on Python 3.14), a scipy-only fallback computes the subset of features that don't require FFT mel processing.

### 2. Rule-based bark classifier (`utils/bark_classifier.py`)

This is the primary source of truth. Features are normalised to [0, 1] and each of the five moods is scored by a small weighted formula. A secondary veto layer hard-blocks physically impossible predictions (e.g. "warning" requires genuinely sustained loud energy). Confidence (0–100) is computed from the winner's raw score plus the margin over the runner-up.

This classifier runs entirely locally with no model downloads.

### 3. Hugging Face enhancement layer (`utils/hf_audio.py`)

An optional secondary signal using [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) — an Audio Spectrogram Transformer trained on AudioSet's 527 audio categories.

How it interacts with the rule engine:

- The rule-based mood is **always the final answer**. The HF model never overrides it.
- If the HF top label maps to the same mood → confidence is nudged up (max +15 pts).
- If the HF top label maps to a different mood → confidence is nudged down (-5 pts) and both signals are shown.
- If no HF label maps to any PawTalk mood (common for generic clips) → confidence is unchanged.

The HF layer is entirely optional. If `transformers` or `torch` are not installed, the app runs normally and the model insight panel is simply absent.

### 4. Translation and message layer (`utils/translator.py`)

Pure text. No audio logic lives here. Each mood has three style banks (funny / cute / emotional), each with multiple entries. The selection is deterministic — same file always picks the same message. Swap the sidebar style any time without re-analyzing.

---

## Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| Audio loading | soundfile (WAV/FLAC/OGG), librosa (MP3 fallback) |
| Feature extraction | librosa, numpy, scipy |
| Rule classifier | Pure Python / numpy |
| Waveform display | matplotlib, librosa.display |
| HF model | transformers, torch |

---

## Installation

**Python 3.9–3.12 recommended.** Python 3.13+ may lack numba wheels; PawTalk falls back to scipy-only mode automatically in that case.

### 1. Install core dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Hugging Face dependencies (optional)

The app works without these. Install them to enable the Audio Model Insight panel:

```bash
pip install transformers torch
```

On first use, the AST model weights (~90 MB) download automatically from the Hugging Face Hub. Subsequent runs use the local cache.

### 3. Install ffmpeg (optional, for MP3 support)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

WAV, FLAC, and OGG files work without ffmpeg.

---

## Running Locally

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

The sidebar lets you set an optional dog name and choose a translation style. Both settings update the display without re-running audio analysis.

---

## Required Dependencies

```
streamlit>=1.32.0
librosa>=0.10.1
numpy>=1.24.0
soundfile>=0.12.1
scipy>=1.11.0
matplotlib>=3.7.0
audioread>=3.0.0

# Optional — enables the Audio Model Insight panel
transformers>=4.35.0
torch>=2.0.0
```

---

## Project Structure

```
PawTalk/
├── app.py                 # Streamlit entry point — thin controller only
├── requirements.txt
├── README.md
├── assets/
│   ├── sample_audio/      # Drop test WAV files here (not committed to git)
│   └── screenshots/       # bark.png, voice.png for this README
└── utils/
    ├── __init__.py
    ├── audio_features.py  # librosa feature extraction — shared by both tabs
    ├── bark_classifier.py # Rule-based mood classifier (primary source of truth)
    ├── hf_audio.py        # Hugging Face secondary signal + combiner
    ├── voice_analyzer.py  # Human voice command quality assessor
    ├── translator.py      # All playful message content — edit this to change personality
    └── ui_helpers.py      # All Streamlit rendering — edit this to change layout
```

---

## 📸 Screenshots

| Bark Analyzer | Voice Command Coach |
|---|---|
| ![Bark analysis result showing mood card and confidence meter](assets/screenshots/bark.png) | ![Voice coach result showing grade and tip cards](assets/screenshots/voice.png) |

> Screenshots not yet captured — run the app locally and save them to `assets/screenshots/` to populate this table.

---

## 🧪 Testing

For a structured testing checklist and scenario walkthrough, see [TESTING.md](./TESTING.md).

---

## Performance & Caching Behaviour

### Rule-based mode — always available

The librosa feature extractor and rule-based classifier run entirely locally with no network calls. They are available immediately on every cold start, even with no internet connection.

### Hugging Face model — lazy, cached after first load

The HF pipeline is **not loaded at app startup**. It loads on the first bark analysis that requests it. On a hosted environment (Streamlit Cloud, Railway, etc.) that first request may be slower than usual — the model weights (~90 MB) download once and are then cached locally.

| Scenario | Behaviour |
|---|---|
| transformers / torch not installed | App runs normally; Audio Model Insight panel is hidden |
| First analysis after cold start | HF model loads once; a "loading model…" notice is shown |
| Subsequent analyses in same session | Pipeline is reused from `st.cache_resource` — no reload |
| Model download fails (no internet) | App continues with rule-only result; friendly notice shown |
| Worker restart on hosted platform | `st.cache_resource` reloads on the next request |

### Audio results — cached per file

Analysis results are stored in `st.session_state` keyed to a content hash of the uploaded file. Re-uploading the same file reuses the cached result instantly. Changing the dog name or translation style updates the display without re-running any audio processing.

---

## How to Test Without a Dog

You don't need a dog to try the app. Here are some practical options:

**Free sample audio datasets**
- [ESC-50](https://github.com/karolpiczak/ESC-50) — Environmental Sound Classification dataset includes dog bark samples (WAV format, ready to use).
- [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) — includes a "dog_bark" class.
- [Freesound.org](https://freesound.org) — search "dog bark" and download CC-licensed WAV clips.

**YouTube or online clips**
- Use publicly available dog bark clips for personal local testing only. Do not redistribute audio you do not own.
- Look for clips that isolate a single type of bark (excited greeting bark, alert bark, whining) to see how mood prediction varies.

**Voice Command Coach**
- Record yourself saying "Sit", "Stay", or "Come" using your phone's voice memo app, export as WAV or M4A, and upload.
- Try variations: whisper vs. projected voice, one word vs. a full sentence, steady tone vs. rising pitch — and compare the grades.

**Edge cases worth testing**
- Silence or near-silence — should fail gracefully with a "too short" or low-confidence result
- Background music or TV audio — the classifier will likely return "unknown" or a low-confidence mood
- Non-dog animal sounds — cat meowing, bird chirping — the rule engine may misfire; this is expected
- A very short clip (under 0.3 s) — triggers an explicit "too short" error with a clear suggestion

---

## Limitations

**No real dog-language translation.** PawTalk does not decode dog communication. It applies acoustic heuristics — the same techniques used in general audio analysis — to make a plausible guess about emotional state. The "translations" are pre-written creative text matched to that guess.

**Mood inference is approximate.** The rule-based classifier uses thresholds tuned on general acoustic intuition, not a dataset of labeled dog vocalizations. Two different dogs barking "excitedly" may produce different acoustic profiles; only one may match.

**Results depend on audio quality.** Background noise, room echo, microphone distance, and recording clipping all degrade feature extraction. A clip recorded on a phone in a quiet room will give more meaningful results than one captured in a noisy park.

**The Hugging Face model is a general audio classifier.** The AST model was trained on AudioSet — a large dataset of general environmental sounds, not dog emotions. Most dog bark clips will score highly on broad labels like "Dog" or "Animal" rather than specific emotion-related labels. When specific labels do appear (e.g. "Growling", "Whimper"), they're used as supporting evidence only — never as the deciding signal.

**The Voice Command Coach is not a dog trainer.** Delivery tips are based on general principles from dog training literature (short commands, consistent tone, clear energy). They are not a substitute for working with a qualified trainer, especially for behavioral issues.

---

## Possible Future Improvements

- **Recorded audio input** — allow recording directly in the browser rather than requiring a file upload
- **Comparison mode** — upload two bark clips side by side to compare mood profiles
- **Custom mood vocabulary** — let users label their own dog's sounds to build a personalized history
- **Fine-tuned HF model** — replace the general AudioSet model with one fine-tuned specifically on dog vocalizations if a suitable labeled dataset becomes available
- **Batch analysis** — analyze a folder of clips at once for shelter or research use

---

Made with 🐾, librosa, and a healthy respect for the limits of audio analysis.
