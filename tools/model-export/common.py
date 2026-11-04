"""Shared bits for the Tracer model-export scripts.

The model class mirrors the reference code on the desklib model card
(https://huggingface.co/desklib/ai-text-detector-v1.01): DeBERTa-v3-large,
attention-masked mean pooling over the last hidden state, then a single
Linear(hidden, 1) logit. P(AI) = sigmoid(logit).
"""
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"            # gitignored: downloads + exported models
HF_DIR = WORK / "hf"
OUT_DIR = WORK / "onnx"

REPO_ID = "desklib/ai-text-detector-v1.01"
# Pin the exact commit so the SHA-256s we ship in the app stay valid.
REVISION = "5fdea974cd4287c61674951ec78803aa274e2fb7"


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts
        return self.classifier(pooled)  # [batch, 1] logit


def load_torch_model():
    model = DesklibAIDetectionModel.from_pretrained(str(HF_DIR))
    model.eval()
    return model
