"""Model and model laoder for a multi layer perceptron assignment model."""

import torch
from torch import nn

from chirpdetector.config import Config

model = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)


def load_trained_mlp(cfg: Config) -> nn.Module:
    """Load a trained multi layer perceptron assignment model."""
    model.load_state_dict(torch.load(cfg.hyper.modelpath+"/assignment_mlp.pt"))
    model.eval()
    return model

