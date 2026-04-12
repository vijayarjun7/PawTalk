# Dataset Sources for PawTalk AI Classifier

The training pipeline requires real, labeled dog bark audio clips.
**No dataset is included in this repository.** You must supply one before running any training script.

---

## Required directory layout

```
data/
└── raw/
    ├── excited/    ← WAV/MP3/FLAC/OGG files of excited barks
    ├── playful/    ← playful barks
    ├── alert/      ← alert/territorial barks
    ├── anxious/    ← anxious/fearful vocalizations
    └── warning/    ← warning/growling/threat barks
```

Sub-directory names must match the labels in `label_config.json` exactly (case-insensitive).
Any audio format that soundfile or librosa can decode is accepted.
Clip length can vary — `prepare_dataset.py` pads or crops everything to a fixed length.

---

## Recommended sources

### Free, labeled, bark-specific

| Dataset | Labels available | Format | Notes |
|---|---|---|---|
| [BarkBase](https://github.com/kfcb/barkbase) | ~6 dog behavior classes | WAV | Small; may need manual re-labeling to match PawTalk's 5 classes |
| [Dog Bark Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/dog-bark-dataset) | generic dog bark | WAV | No emotion labels — useful for pre-training |
| [DCASE 2023 Animal Sound](https://dcase.community/) | animal sounds inc. dog | WAV | Broad classes, not bark-specific |

### Larger general audio datasets (require filtering for dog barks)

| Dataset | Notes |
|---|---|
| [ESC-50](https://github.com/karolpiczak/ESC-50) | Class "Dog" — 40 clips, no mood labels |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | Class "dog_bark" — ~1000 clips, no mood labels |
| [FreeSound](https://freesound.org) | Search for specific bark types; CC-licensed clips available |
| [AudioSet](https://research.google.com/audioset/) | 527 classes inc. "Dog", "Bark", "Growling", "Whimper" — large, weakly labeled |

### Hugging Face datasets

The training pipeline does **not** yet have a built-in HF dataset loader,
but the data preparation step (converting any audio source to
`data/raw/<label>/` files) is straightforward for any HF dataset with
audio + label columns. Example:

```python
from datasets import load_dataset
ds = load_dataset("your-org/bark-dataset", split="train")
for row in ds:
    label = row["label"]          # map to one of: excited/playful/alert/anxious/warning
    audio = row["audio"]["array"] # numpy float32
    sr    = row["audio"]["sampling_rate"]
    # save to data/raw/<label>/<idx>.wav
```

---

## Label mapping

The five PawTalk labels and their definitions are in [`label_config.json`](label_config.json).

If your dataset uses different class names, you will need to manually map them
before running `prepare_dataset.py`. There is no automated mapping — bark
behavior labeling conventions vary significantly between datasets.

Minimum recommended clips per label: **50 for a toy run, 200+ for useful results**.
Practically, Wav2Vec2 fine-tuning with frozen encoder can produce reasonable
results with ~100 clips per class on a CPU in under an hour.

---

## What NOT to do

- Do not fabricate bark clips with TTS or sound effects.
- Do not use human speech clips as "bark" training data.
- Do not mix clips from multiple microphone conditions without normalizing.
- Do not train on clips shorter than 0.3 seconds — the model can't extract
  meaningful features from them and `prepare_dataset.py` will skip them.
