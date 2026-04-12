"""
utils/ai_bark_classifier_model.py
-----------------------------------
BarkClassifier model definition shared between training and inference.

Keeping the class here (instead of inside train_classifier.py) means
ai_bark_classifier.py can import it without pulling in the full training
script or its CLI argument parsing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import transformers
from transformers import Wav2Vec2Model

MODEL_ID = "facebook/wav2vec2-base"


class BarkClassifier(nn.Module):
    """
    Wav2Vec2 encoder → mean-pool → dropout → linear classification head.

    Parameters
    ----------
    num_labels      : number of output classes
    freeze_encoder  : if True, the Wav2Vec2 weights are frozen (train head only)
    model_id        : Hugging Face model hub ID (can override for larger models)
    """

    def __init__(
        self,
        num_labels: int,
        freeze_encoder: bool = False,
        model_id: str = MODEL_ID,
    ):
        super().__init__()
        # Temporarily raise transformers verbosity to ERROR during from_pretrained
        # to suppress the LOAD REPORT and "not sharded" INFO lines that appear
        # when loading Wav2Vec2Model (which omits quantizer/projection weights
        # that exist in the full Wav2Vec2ForPreTraining checkpoint).
        # These messages are expected and harmless — they would just pollute logs.
        _prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.encoder = Wav2Vec2Model.from_pretrained(model_id)
        transformers.logging.set_verbosity(_prev_verbosity)

        self.dropout    = nn.Dropout(0.25)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_values   : (batch, time) float32 waveform tensor at 16 kHz
        attention_mask : (batch, time) int64 mask in *sample* space —
                         1 for real signal, 0 for zero-padded samples.
                         Pass None when all clips in the batch have the same
                         length (e.g. after prepare_dataset.py normalisation).

        Returns
        -------
        logits : (batch, num_labels) float32

        Note on mask dimensions
        -----------------------
        Wav2Vec2's convolutional feature extractor downsamples the waveform
        from sample-space to frame-space (ratio ≈ 320x for wav2vec2-base).
        A 3-second clip at 16 kHz = 48000 samples → ~149 frames.

        We must convert the sample-space attention_mask to a frame-space mask
        before using it for mean-pooling over last_hidden_state.
        We use the encoder's own _get_feat_extract_output_lengths() for this
        conversion so that the ratio is always consistent with the model config.
        """
        # Pass the sample-space mask to the encoder so it can ignore padding
        # in its own internal computations.
        out    = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state    # (B, T_frames, H)

        if attention_mask is not None:
            # Convert sample-space mask to frame-space mask.
            # Sum along the time axis to get the number of real samples per item,
            # convert to frame counts, then build a boolean frame-space mask.
            sample_lengths = attention_mask.sum(dim=1)          # (B,)
            frame_lengths  = self.encoder._get_feat_extract_output_lengths(
                sample_lengths
            )                                                    # (B,)
            T = hidden.shape[1]
            frame_mask = (
                torch.arange(T, device=hidden.device).unsqueeze(0)
                < frame_lengths.unsqueeze(1)
            ).unsqueeze(-1).float()                              # (B, T, 1)
            pooled = (hidden * frame_mask).sum(1) / frame_mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)                          # (B, H)

        return self.classifier(self.dropout(pooled))
