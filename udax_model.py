"""UDAX-ByteNet-v1 — a small byte-level neural predictor.

Given the previous `context` bytes, the network outputs a probability
distribution over the 256 possible values of the next byte. A range
encoder (see udax_codec.py) turns those predictions into a compressed
bitstream; the decoder re-runs the same model to recover the bytes.

Architecture (prototype, per Experiment 001 spec):
    Input      : previous 16 bytes
    Embedding  : 256 symbols -> 16 dims
    Hidden     : 256 units, GELU
    Output     : 256 logits (one per byte value)
    Loss       : cross-entropy  ==  -sum log2 P(x_t | x_<t)  in bits
"""
import numpy as np
import torch
import torch.nn as nn

# Floor mixed into every prediction so no byte ever gets (near-)zero
# probability: guarantees the entropy model stays representable in
# constriction's fixed-point arithmetic on BOTH sides, identically.
PROB_FLOOR_EPS = 0.002


class UDAXByteNet(nn.Module):
    def __init__(self, context=16, embedding=16, hidden=256):
        super().__init__()
        self.context = context
        self.embedding = nn.Embedding(256, embedding)
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(context * embedding, hidden),
            nn.GELU(),
            nn.Linear(hidden, 256),
        )

    def forward(self, x):
        x = self.embedding(x)
        return self.network(x)


def predict_probabilities(model, context):
    """Probability of each possible next byte, as a float64 numpy array.

    `context` must contain exactly `model.context` previous bytes.
    A small uniform floor is mixed in and the result renormalized;
    encoder and decoder both call this function, so both see bit-wise
    identical distributions (same weights, CPU inference, 1 thread).
    """
    x = torch.tensor(list(context), dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        logits = model(x)[0]
        probabilities = torch.softmax(logits, dim=-1)
    probs = probabilities.to(torch.float64).numpy()
    probs = (1.0 - PROB_FLOOR_EPS) * probs + PROB_FLOOR_EPS / 256.0
    probs /= probs.sum()
    return probs


def load_model(checkpoint_path, device="cpu"):
    """Load a checkpoint saved by train.py and return an eval-mode model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UDAXByteNet(context=checkpoint["context"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def model_hash(checkpoint_path):
    """SHA-256 of the checkpoint file — stored in every chunk header."""
    import hashlib
    h = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
