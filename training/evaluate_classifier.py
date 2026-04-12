"""
training/evaluate_classifier.py
---------------------------------
Evaluate the trained PawTalk bark classifier on the held-out test split.

Produces:
  - Per-class precision / recall / F1
  - Overall accuracy and macro-F1
  - Confusion matrix (terminal + optional PNG)
  - Top-2 prediction accuracy (correct label in top-2)

Usage
-----
    python training/evaluate_classifier.py \\
        --checkpoint  checkpoints/best_model.pt  \\
        --data_dir    data/processed             \\
        --split       test                       \\
        --confusion_matrix_png  results/confusion.png
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
    from torch.utils.data import DataLoader
except ImportError:
    print("ERROR: PyTorch not installed.")
    sys.exit(1)

try:
    from transformers import Wav2Vec2Model
except ImportError:
    print("ERROR: transformers not installed.")
    sys.exit(1)

# BarkClassifier lives in the shared utils module so both training and
# inference use the identical definition.  BarkDataset and helpers come
# from train_classifier (they are training-only concerns).
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(Path(__file__).parent))

from utils.ai_bark_classifier_model import BarkClassifier          # noqa: E402
from train_classifier import BarkDataset, collate_fn, build_attention_mask  # noqa: E402


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    all_labels: list[int],
    all_preds: list[int],
    all_logits: np.ndarray,
    label_names: list[str],
) -> dict:
    """
    Compute accuracy, macro-F1, per-class P/R/F1, top-2 accuracy,
    and confusion matrix.
    """
    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    n      = len(labels)
    num_c  = len(label_names)

    accuracy = float((labels == preds).mean())

    # Per-class metrics
    per_class = {}
    for c, name in enumerate(label_names):
        tp = int(((labels == c) & (preds == c)).sum())
        fp = int(((labels != c) & (preds == c)).sum())
        fn = int(((labels == c) & (preds != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[name] = {"precision": prec, "recall": rec, "f1": f1,
                           "support": int((labels == c).sum())}

    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    # Top-2 accuracy
    if all_logits is not None:
        top2_preds = np.argsort(all_logits, axis=1)[:, -2:]   # top 2 indices
        top2_acc   = float(
            np.array([labels[i] in top2_preds[i] for i in range(n)]).mean()
        )
    else:
        top2_acc = accuracy

    # Confusion matrix
    cm = np.zeros((num_c, num_c), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1

    return {
        "accuracy":  accuracy,
        "macro_f1":  macro_f1,
        "top2_acc":  top2_acc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def print_report(metrics: dict, label_names: list[str]) -> None:
    cm    = np.array(metrics["confusion_matrix"])
    pc    = metrics["per_class"]
    width = max(len(n) for n in label_names) + 2

    print(f"\nAccuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"Top-2 acc: {metrics['top2_acc']:.4f}")

    print(f"\n{'Label':<{width}} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print("-" * (width + 27))
    for name in label_names:
        v = pc[name]
        print(
            f"{name:<{width}} {v['precision']:>6.3f} {v['recall']:>6.3f} "
            f"{v['f1']:>6.3f} {v['support']:>5}"
        )

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = " " * (width + 1) + "  ".join(f"{n[:4]:>4}" for n in label_names)
    print(header)
    for i, name in enumerate(label_names):
        row = "  ".join(f"{cm[i, j]:>4}" for j in range(len(label_names)))
        print(f"{name:<{width}} {row}")


def save_confusion_matrix_png(
    cm: np.ndarray,
    label_names: list[str],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping confusion matrix PNG.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(label_names)),
        yticks=range(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm.max() / 2.0
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained PawTalk bark classifier."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt from train_classifier.py.")
    parser.add_argument("--data_dir",   default="data/processed",
                        help="Processed dataset root (contains label_map.json).")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate (default: test).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--confusion_matrix_png", default=None,
                        help="Optional path to save confusion matrix PNG.")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load checkpoint
    ckpt      = torch.load(str(ckpt_path), map_location=device)
    label_map = ckpt["label_map"]
    config    = ckpt.get("config", {})
    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    print(f"Checkpoint : {ckpt_path}")
    print(f"Epoch      : {ckpt.get('epoch', '?')}")
    print(f"Val acc    : {ckpt.get('val_acc', '?'):.4f}")
    print(f"Labels     : {label_names}")
    print(f"Device     : {device}")

    # Rebuild model
    model = BarkClassifier(num_labels=len(label_map), freeze_encoder=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Dataset
    ds = BarkDataset(data_dir, args.split, label_map)
    if len(ds) == 0:
        print(f"No clips found in split '{args.split}'. Exiting.")
        sys.exit(1)
    print(f"\nEvaluating on {args.split}: {len(ds)} clips")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    all_labels, all_preds, all_logits_list = [], [], []

    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            mask      = build_attention_mask(waveforms).to(device)
            logits    = model(waveforms, attention_mask=mask)

            all_labels.extend(labels.tolist())
            all_preds.extend(logits.argmax(1).tolist())
            all_logits_list.append(logits.cpu().numpy())

    all_logits = np.concatenate(all_logits_list, axis=0)

    metrics = compute_metrics(all_labels, all_preds, all_logits, label_names)
    print_report(metrics, label_names)

    if args.confusion_matrix_png:
        save_confusion_matrix_png(
            np.array(metrics["confusion_matrix"]),
            label_names,
            Path(args.confusion_matrix_png),
        )

    # Save metrics JSON
    metrics_path = ckpt_path.parent / f"eval_{args.split}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
