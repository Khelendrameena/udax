# UDAX — Neural Lossless Compression (Experiment 001)

UDAX compresses files **losslessly** with a neural predictor + range encoder:
the AI predicts the probability of each next byte, and an arithmetic/range
coder stores the actual byte in ~`−log2 P(byte)` bits. The decoder re-runs the
same frozen model, regenerates the same probabilities, and recovers the
original bytes exactly — verified by SHA-256.

**Model:** `UDAX-ByteNet-v1` — 16-byte context → embedding (16) → MLP (256,
GELU) → 256-way softmax. Trained once on a corpus of MP4s, frozen, and used
identically at both endpoints.

## Experiment 001 results (150 KB unseen H.264/AAC MP4, all SHA-256 verified)

| Method | Ratio |
|---|---:|
| Zstandard level 3 | 0.972 |
| **UDAX (any chunk size)** | **0.962–0.963** |
| Zstandard level 10 | 0.956 |
| Zstandard level 19 | 0.952 |

The neural codec beats zstd's default level on ratio and needs no raw-fallback
chunks; zstd-19 still wins on ratio and is ~10⁵× faster (the Python prototype
runs at ~2.4 KB/s — a correctness prototype, per the experiment spec).
Full analysis: [EXPERIMENT-001-REPORT.md](EXPERIMENT-001-REPORT.md)

## Layout
```
udax/
├── training_videos/     # training MP4s go here (not committed)
├── udax_model.py        # UDAX-ByteNet-v1 + probability prediction
├── train.py             # training loop -> udax_bytenet_v1.pt
├── udax_codec.py        # range encode/decode + UDAX chunk file format
├── test_udax.py         # milestone: small-sample SHA-256 round-trip
└── experiment.py        # Experiment 001: zstd vs chunk sizes A–G
```

## Run it
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch numpy constriction zstandard    # use the CPU wheel index for torch if no GPU

# 1) put training MP4s in training_videos/ (keep the test video OUT)
python train.py --steps 20000

# 2) milestone: 4 KB round-trip, SHA-256 must match
python test_udax.py sample.mp4 4096

# 3) full experiment table
python experiment.py test.mp4
```

## Chunk format
File header: magic `UDAXF1`, chunk size, total size, model SHA-256.
Per chunk: version, chunk id, original size, payload size, mode
(0 = raw fallback, 1 = neural), model hash, chunk SHA-256 — then the
payload (16-byte literal seed + range-coded stream). Chunks decode
independently, so the format streams and parallelizes.

## Design rules that keep it correct
- Same frozen weights + same CPU inference + 1 thread on both sides;
  deterministic algorithms enforced (`udax_codec.init_determinism`).
- A small uniform floor is mixed into every prediction so no symbol's
  probability can underflow the entropy coder's fixed-point model.
- If a chunk's neural payload isn't smaller than the raw chunk, the
  chunk is stored raw — the format can never expand a file by more
  than its headers.
