"""
training/train_classifier.py
-----------------------------
Fine-tune a Wav2Vec2-base encoder with a linear classification head on
the prepared dog bark dataset.

Architecture
------------
facebook/wav2vec2-base
  → mean-pool the last hidden state   (768-d)
  → dropout 0.25
  → Linear(768, num_labels)
  → CrossEntropyLoss

Why Wav2Vec2-base:
  - Pretrained on 960h of speech → strong low-level acoustic representations
  - 94 MB — feasible on laptop GPU or CPU (slower)
  - Can be swapped for wav2vec2-large or ast-patch-400 by changing MODEL_ID

Usage
-----
    python training/train_classifier.py \\
        --data_dir   data/processed     \\
        --out_dir    checkpoints/       \\
        --epochs     30                 \\
        --batch_size 16                 \\
        --lr         3e-4               \\
        --freeze_encoder

    # Resume from a checkpoint:
    python training/train_classifier.py \\
        --data_dir data/processed --out_dir checkpoints/ \\
        --resume   checkpoints/best_model.pt

Output
------
    checkpoints/best_model.pt     — state dict of the best val-accuracy model
    checkpoints/training_log.json — per-epoch metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    from transformers import Wav2Vec2Model  # noqa — availability check
except ImportError:
    print("ERROR: transformers is required. Install with: pip install transformers")
    sys.exit(1)

# Shared model definition (also used by inference)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ai_bark_classifier_model import BarkClassifier  # noqa: E402

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID      = "facebook/wav2vec2-base"
TARGET_SR     = 16_000
NUM_LABELS    = 5
LABEL_NAMES   = ["excited", "playful", "alert", "anxious", "warning"]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BarkDataset(Dataset):
    """
    Reads pre-processed WAV clips from data/processed/<split>/<label>/*.wav.
    Returns (waveform_tensor, label_int).
    """

    def __init__(self, root: Path, split: str, label_map: dict[str, int]):
        self.samples: list[tuple[Path, int]] = []
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Run prepare_dataset.py first."
            )
        for label, idx in label_map.items():
            label_dir = split_dir / label
            if not label_dir.is_dir():
                continue
            for fp in sorted(label_dir.iterdir()):
                if fp.suffix == ".wav":
                    self.samples.append((fp, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        y = _load_wav(path)
        return torch.tensor(y, dtype=torch.float32), label


def _load_wav(path: Path) -> np.ndarray:
    if _SF:
        try:
            y, _ = sf.read(str(path), dtype="float32", always_2d=False)
            return y
        except Exception:
            pass
    if _LIBROSA:
        y, _ = librosa.load(str(path), sr=None, mono=True)
        return y.astype(np.float32)
    raise RuntimeError(f"Cannot load {path} — install soundfile or librosa.")


def collate_fn(batch):
    """Pad waveforms in a batch to the same length."""
    waveforms, labels = zip(*batch)
    max_len = max(w.shape[0] for w in waveforms)
    padded  = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded[i, : w.shape[0]] = w
    return padded, torch.tensor(labels, dtype=torch.long)



def build_attention_mask(input_values: torch.Tensor) -> torch.Tensor:
    """
    Build a padding mask: 1 for real signal frames, 0 for zero-padded frames.
    A frame is considered padding if all samples in that frame are exactly 0.
    This is a heuristic — works because our padding is literal zeros.
    """
    # Mark non-zero positions
    return (input_values.abs() > 1e-8).long()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: BarkClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels    = labels.to(device)
        mask      = build_attention_mask(waveforms).to(device)

        optimizer.zero_grad()
        logits = model(waveforms, attention_mask=mask)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(
    model: BarkClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels    = labels.to(device)
        mask      = build_attention_mask(waveforms).to(device)

        logits = model(waveforms, attention_mask=mask)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)

    return total_loss / max(n, 1), correct / max(n, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: BarkClassifier,
    out_dir: Path,
    epoch: int,
    val_acc: float,
    label_map: dict,
    config: dict,
    filename: str = "best_model.pt",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":        epoch,
            "val_acc":      val_acc,
            "model_state":  model.state_dict(),
            "label_map":    label_map,
            "config":       config,
        },
        out_dir / filename,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PawTalk AI bark classifier (Wav2Vec2 + linear head)."
    )
    parser.add_argument("--data_dir",       default="data/processed",
                        help="Processed dataset root (output of prepare_dataset.py).")
    parser.add_argument("--out_dir",        default="checkpoints",
                        help="Where to save checkpoints and training log.")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze Wav2Vec2 encoder weights (train head only).")
    parser.add_argument("--unfreeze_at",    type=int,   default=None,
                        help="Epoch at which to unfreeze the encoder (for staged training).")
    parser.add_argument("--resume",         default=None,
                        help="Path to a checkpoint to resume from.")
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    # Load label map from dataset
    label_map_path = data_dir / "label_map.json"
    if not label_map_path.exists():
        print(f"ERROR: {label_map_path} not found. Run prepare_dataset.py first.")
        sys.exit(1)
    label_map: dict[str, int] = json.loads(label_map_path.read_text())
    num_labels = len(label_map)
    print(f"Labels ({num_labels}): {list(label_map.keys())}")
    print(f"Device: {device}")

    # Datasets & loaders
    train_ds = BarkDataset(data_dir, "train", label_map)
    val_ds   = BarkDataset(data_dir, "val",   label_map)
    print(f"Train: {len(train_ds)} clips  |  Val: {len(val_ds)} clips")

    if len(train_ds) == 0:
        print("ERROR: no training clips found.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Model
    print(f"Loading encoder: {MODEL_ID}  (freeze={args.freeze_encoder})")
    model = BarkClassifier(
        num_labels=num_labels,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch   = 0
    best_val_acc  = 0.0
    training_log  = []

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch - 1}  (best val_acc={best_val_acc:.4f})")

    config = {
        "model_id":       MODEL_ID,
        "num_labels":     num_labels,
        "freeze_encoder": args.freeze_encoder,
        "lr":             args.lr,
        "batch_size":     args.batch_size,
        "target_sr":      TARGET_SR,
    }

    # Training loop
    for epoch in range(start_epoch, args.epochs):

        # Staged unfreeze
        if args.unfreeze_at is not None and epoch == args.unfreeze_at:
            print(f"Epoch {epoch}: unfreezing encoder weights.")
            for p in model.encoder.parameters():
                p.requires_grad = True
            # Reinitialise optimizer to include newly unfrozen params
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch
            )

        tr_loss, tr_acc   = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch":    epoch,
            "tr_loss":  round(tr_loss,  4),
            "tr_acc":   round(tr_acc,   4),
            "val_loss": round(val_loss, 4),
            "val_acc":  round(val_acc,  4),
        }
        training_log.append(row)
        print(
            f"Epoch {epoch:3d}  "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            + (" ← best" if val_acc > best_val_acc else "")
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, out_dir, epoch, val_acc, label_map, config)

    # Save training log
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.json"
    log_path.write_text(json.dumps(training_log, indent=2))
    print(f"\nBest val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint  : {out_dir / 'best_model.pt'}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
