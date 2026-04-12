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

# PawTalk — AI-Powered Dog Bark Analyzer & Voice Command Coach

Analyze your dog's vocalizations with a hybrid AI + rule-based system, and get feedback on how clearly you're delivering training commands.

**Live app → [pawtalk-buddy.streamlit.app](https://pawtalk-buddy.streamlit.app)**
&nbsp;&nbsp;|&nbsp;&nbsp;
**HF Space → [huggingface.co/spaces/Vijayarv07/PawTalk](https://huggingface.co/spaces/Vijayarv07/PawTalk)**
&nbsp;&nbsp;|&nbsp;&nbsp;
**GitHub → [vijayarjun7/PawTalk](https://github.com/vijayarjun7/PawTalk)**

---

## Quick Demo

1. Upload a bark recording or record directly in the browser
2. Get a predicted behavioral state — excited, alert, anxious, playful, or warning
3. Read the playful AI-assisted interpretation
4. Optionally record yourself giving a command and get delivery coaching

No account. No subscription. Runs locally.

---

## Features

### Bark → Human: Behavioral State Analyzer

Upload or record a dog bark. PawTalk estimates one of five states:

| State | Signal characteristics |
|---|---|
| **Excited** | High energy, rapid bursts, bright tone |
| **Playful** | Moderate energy, bouncy rhythm |
| **Alert** | Sharp spikes, quick silences, elevated pitch |
| **Anxious** | Irregular energy, many pauses, chaotic rhythm |
| **Warning** | Sustained loud energy, low-pitched, deliberate |

Each result includes:
- Predicted state with confidence score (0–100)
- Playful interpretation in your chosen style (funny / cute / emotional)
- Source badge — AI classifier, AI+Rule blend, or rule engine
- Audio feature summary (energy, rhythm, brightness, burst count)
- Optional AudioSet model insight panel

### Human → Dog: Voice Command Coach

Record or upload yourself saying a command ("Sit", "Stay", "Come"). PawTalk evaluates four dimensions:

| Dimension | What it measures |
|---|---|
| **Tone** | Calm / upbeat / firm / intense / unclear |
| **Volume** | Too soft / just right / too loud |
| **Duration** | Too short / ideal / slightly long / too long |
| **Pace** | Slow / moderate / rapid-fire |

Returns an overall grade (Excellent → Unclear) with specific tips per dimension.

### Other Features

- **In-browser recording** via microphone (requires `streamlit-mic-recorder`)
- **File upload** for WAV, MP3, FLAC, OGG
- **Session caching** — re-uploading the same file reuses the cached result instantly
- **Style switching** — change translation style without re-running analysis
- **Fallback-safe** — app works with no AI model, no internet, and no GPU

---

## How It Works

```
Audio input (upload or mic recording)
    │
    ▼
1. Load + normalize → mono 16 kHz float32
    │
    ▼
2. Feature extraction (librosa)
   RMS energy, ZCR, spectral centroid, pitch (pyin/yin),
   tempo, beat regularity, pause/burst counts, MFCCs
    │
    ├──► 3a. Rule-based classifier
    │         Normalised features → weighted mood scores → veto layer
    │         Always runs. No downloads. No network.
    │
    ├──► 3b. AI classifier (when checkpoint exists)
    │         Wav2Vec2-base encoder → mean-pool → linear head
    │         Pretrained on speech, fine-tuned on labeled bark clips
    │         Returns top-1 mood + top-2 + per-class probabilities
    │
    ▼
4. Combiner: AI result + rule fallback
   AI confident (≥70%)  → AI is primary
   AI moderate (45–70%) → AI mood, blended confidence
   AI uncertain (<45%)  → rule engine result
    │
    ▼
5. Optional: HF AudioSet model (MIT/ast-finetuned-audioset-10-10-0.4593)
   Agreement boosts confidence (+15 max)
   Disagreement surfaced as context, never overrides
    │
    ▼
6. Translation + UI rendering
```

---

## AI Model Status

PawTalk includes a complete supervised training pipeline built on [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base).

**What is implemented:**
- Full training pipeline (`prepare_dataset.py`, `train_classifier.py`, `evaluate_classifier.py`)
- Inference module with confidence thresholds and rule-engine fallback
- Evaluation: accuracy, macro-F1, per-class P/R/F1, confusion matrix

**What depends on your dataset:**
- Model quality is entirely a function of the labeled bark clips you supply
- No labeled dog bark dataset is included in this repository
- No accuracy numbers are claimed — they would only be meaningful against a real dataset

**Current demo behavior:**
- If `checkpoints/best_model.pt` is present → AI classifier runs as primary
- If no checkpoint → rule-based classifier runs transparently
- The UI always shows which system produced the result

See [training/DATA_SOURCES.md](training/DATA_SOURCES.md) for where to obtain real labeled bark data.

---

## Fallback Logic

The app never breaks regardless of what is installed or available.

```
torch + transformers installed AND checkpoint exists
    → AI classifier (primary)

AI model uncertain OR confidence too low
    → Rule engine (fallback)

torch/transformers not installed OR no checkpoint
    → Rule engine (always available)

HF AudioSet model unavailable
    → Confidence unchanged, insight panel hidden
```

The rule-based classifier runs entirely locally — no network, no model download, no GPU required.

---

## Training Pipeline

Requires: `torch`, `transformers`, `soundfile` or `librosa`.

**Step 1 — Prepare labeled data**

Organize your bark clips into per-label directories:

```
data/raw/
    excited/   clip1.wav  clip2.mp3  …
    playful/   …
    alert/     …
    anxious/   …
    warning/   …
```

Then run:

```bash
python training/prepare_dataset.py \
    --data_dir data/raw \
    --out_dir  data/processed \
    --split    0.70 0.15 0.15 \
    --clip_sec 3.0
```

Outputs normalized 16kHz WAV clips in `data/processed/train|val|test/<label>/`, plus `label_map.json`.

**Step 2 — Train**

```bash
python training/train_classifier.py \
    --data_dir      data/processed \
    --out_dir       checkpoints \
    --epochs        30 \
    --freeze_encoder              \
    --unfreeze_at   20
```

`--freeze_encoder` trains only the linear head for the first 20 epochs, then unfreezes the full encoder for fine-tuning. Saves `checkpoints/best_model.pt` on every val-accuracy improvement.

**Step 3 — Evaluate**

```bash
python training/evaluate_classifier.py \
    --checkpoint            checkpoints/best_model.pt \
    --confusion_matrix_png  checkpoints/confusion.png
```

Prints per-class precision, recall, F1, overall accuracy, macro-F1, top-2 accuracy, and saves a confusion matrix image.

**Step 4 — Use in the app**

Once `checkpoints/best_model.pt` exists, the Streamlit app picks it up automatically on the next run. No code changes needed.

---

## Project Structure

```
PawTalk/
├── app.py                          # Streamlit entry point — thin controller
├── requirements.txt
├── Dockerfile                      # HF Spaces / Docker deployment
│
├── utils/
│   ├── audio_features.py           # librosa feature extraction (shared)
│   ├── bark_classifier.py          # Rule-based mood classifier
│   ├── ai_bark_classifier.py       # AI inference module + combiner
│   ├── ai_bark_classifier_model.py # BarkClassifier nn.Module definition
│   ├── hf_audio.py                 # HF AudioSet secondary signal
│   ├── audio_input.py              # Upload / mic-record widget
│   ├── voice_analyzer.py           # Voice command quality assessor
│   ├── translator.py               # Playful message content
│   └── ui_helpers.py               # Streamlit rendering helpers
│
├── training/
│   ├── label_config.json           # Canonical label set (source of truth)
│   ├── prepare_dataset.py          # Normalize + split raw audio
│   ├── train_classifier.py         # Fine-tune Wav2Vec2 + head
│   ├── evaluate_classifier.py      # Metrics + confusion matrix
│   └── DATA_SOURCES.md             # Where to get labeled bark data
│
├── checkpoints/                    # Saved model weights (not committed)
│   └── best_model.pt
│
└── data/                           # Training data (not committed)
    ├── raw/
    └── processed/
```

---

## Screenshots

| Bark Analyzer | Voice Command Coach |
|---|---|
| ![Bark analysis result](assets/screenshots/bark.png) | ![Voice coach result](assets/screenshots/voice.png) |

*Run the app locally and save screenshots to `assets/screenshots/` to populate this table.*

---

## Why This Project Exists

PawTalk was built as a personalized gift — an AI-powered toy that combines real audio analysis with playful UX. The goal was never accuracy. The goal was experience, honesty, and a system that is technically interesting without overclaiming what it can do.

The supervised training pipeline was added to show how a production-grade bark classifier would be built — using a pretrained speech encoder, fine-tuned classification head, proper train/val/test splits, and a principled fallback strategy.

---

## Limitations

**This is not a dog-language translator.** PawTalk applies acoustic signal analysis to make a plausible guess about a dog's behavioral state. The "translations" are pre-written creative text matched to that guess.

**Audio quality matters.** Background noise, room echo, and microphone distance all degrade feature extraction. A clean recording in a quiet room will give more interpretable results.

**The AI model is only as good as its training data.** Wav2Vec2 was pretrained on human speech, not dog vocalizations. Fine-tuning on a small or poorly-labeled dataset will produce unreliable predictions. The rule engine is a meaningful fallback precisely because of this.

**The HF AudioSet model is a general audio classifier.** It was trained on 527 broad audio categories, not bark emotions. It provides supporting evidence only — never the deciding signal.

**The Voice Command Coach is not a dog trainer.** Tips are based on general principles from dog training literature and acoustic signal properties. They are not a substitute for working with a qualified trainer.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Audio loading | soundfile, librosa |
| Feature extraction | librosa, numpy, scipy |
| AI encoder | Hugging Face Transformers — `facebook/wav2vec2-base` |
| Training | PyTorch, AdamW, CosineAnnealingLR |
| Rule classifier | Pure Python / numpy |
| Secondary signal | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Waveform display | matplotlib |
| In-browser recording | streamlit-mic-recorder |
| Deployment | Docker (HF Spaces), Streamlit Cloud |

---

## Getting Started

**Python 3.9–3.12 recommended.** Python 3.13+ may lack numba wheels; PawTalk falls back to scipy-only mode automatically.

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Optional: enable AI bark classifier**

```bash
# Already included in requirements.txt — install torch + transformers if not present
pip install torch transformers

# Then train your model (requires labeled data — see training/DATA_SOURCES.md)
python training/train_classifier.py --data_dir data/processed --out_dir checkpoints
```

**Optional: enable in-browser recording**

```bash
pip install streamlit-mic-recorder
```

**Optional: MP3 support**

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

WAV, FLAC, and OGG work without ffmpeg.

---

Made with librosa, PyTorch, and a healthy respect for the limits of audio analysis.
