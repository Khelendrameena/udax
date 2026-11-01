"""Train UDAX-ByteNet-v1 on a folder of MP4 files.

Usage:
    python train.py [--steps 20000] [--batch 512] [--lr 1e-3]
                    [--out udax_bytenet_v1.pt]

Trains on SEPARATE videos from the experiment's test video (spec rule:
never train and test on the same file). Progress prints the loss in
nats; divide by ln(2) for bits/byte — below 8.0 bits/byte means the
entropy coder can, in principle, compress.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from udax_model import UDAXByteNet

CONTEXT = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="udax_bytenet_v1.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_paths = sorted(Path("training_videos").glob("*.mp4"))
    if not video_paths:
        raise RuntimeError("No MP4 training files found in training_videos/")
    videos = [np.fromfile(p, dtype=np.uint8) for p in video_paths]
    videos = [v for v in videos if len(v) > CONTEXT]
    if not videos:
        raise RuntimeError("Training videos are too small")
    total_mb = sum(len(v) for v in videos) / 1e6
    print(f"Training data: {len(videos)} files, {total_mb:.1f} MB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = UDAXByteNet(context=CONTEXT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for step in range(args.steps):
        xs = np.empty((args.batch, CONTEXT), dtype=np.int64)
        ys = np.empty(args.batch, dtype=np.int64)
        for j in range(args.batch):
            data = random.choice(videos)
            pos = random.randrange(CONTEXT, len(data))
            xs[j] = data[pos - CONTEXT:pos]
            ys[j] = data[pos]
        x = torch.from_numpy(xs).to(device)
        y = torch.from_numpy(ys).to(device)

        model.train()
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}: loss={loss.item():.4f} nats "
                  f"({loss.item() / np.log(2):.3f} bits/byte)")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "context": CONTEXT,
            "architecture": "UDAX-ByteNet-v1",
            "steps": args.steps,
        },
        args.out,
    )
    print(f"Saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
