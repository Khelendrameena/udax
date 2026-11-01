# UDAX Experiment 001 — Lossless MP4 Compression: Results

**Date:** 2026-11-01 · **Status:** complete (prototype scale) · **All SHA-256 checks passed**

## Setup
- **Codec:** UDAX-ByteNet-v1 — byte-level neural predictor (16-byte context, embedding 16, hidden 256, GELU, 256-way softmax) + constriction range encoder. A uniform probability floor (ε = 0.002) is mixed into every prediction so both sides build identical fixed-point entropy models.
- **Training:** 4 ffmpeg-generated H.264/AAC MP4s (4.5 MB total), 6,000 steps, batch 512, AdamW. Final train loss ≈ **7.52–7.59 bits/byte** (started at 8.03).
- **Test file:** `test.mp4` — 152,908 bytes, H.264/AAC, **never used in training**. SHA-256 `4ba3e37ea3371e76…`
- **Determinism:** single-threaded CPU inference + deterministic algorithms on both sides; every chunk also carries its own SHA-256 checksum, and the whole-file hash is verified after decode.

## Results

| Test | Method | Size (bytes) | Ratio | Encode | Decode | SHA-256 |
|---|---|---:|---:|---:|---:|:--:|
| A | Original (no compression) | 152,908 | 1.000 | — | — | OK |
| B | Zstandard whole-file, level 3 | 148,596 | 0.972 | 0.01 s | <0.01 s | OK |
| B | Zstandard whole-file, level 10 | 146,160 | 0.956 | 0.01 s | <0.01 s | OK |
| B | Zstandard whole-file, level 19 | 145,503 | **0.952** | 0.04 s | <0.01 s | OK |
| C | UDAX, 64 KB chunks (3 chunks, all neural) | 147,182 | 0.963 | 61.3 s | 63.5 s | OK |
| D | UDAX, 256 KB chunks (1 chunk) | 147,078 | 0.962 | 63.6 s | 62.9 s | OK |
| E | UDAX, 1 MB chunks (1 chunk) | 147,078 | 0.962 | 63.6 s | 62.8 s | OK |
| F | UDAX, 4 MB chunks (1 chunk) | 147,078 | 0.962 | 63.6 s | 63.0 s | OK |
| G | UDAX, 16 MB chunks (1 chunk) | 147,078 | 0.962 | 63.6 s | 62.0 s | OK |

(E/F/G are single-chunk at this file size, so their encodings are identical by construction; the per-chunk cost is visible in C: three chunks cost only **+104 bytes** over one chunk — 48 bytes of header + 16 seed bytes per chunk, roughly.)

Milestone check (spec §8): the first 4,096 bytes round-tripped **4096 → 3808 bytes (0.930)** with identical SHA-256 — the file head is mostly MP4 box structure, which the model predicts best.

## What the results actually say
1. **The neural codec works and genuinely compresses unseen MP4 data.** Implied rate ≈ 7.70 bits/byte on the test file (vs 7.55 train) — a small, honest generalization gap. It **beats Zstandard level 3** (0.963 vs 0.972) and needs **no raw-fallback chunks** on this file.
2. **Zstandard at high levels still wins on ratio** (0.952 at level 19) and is not remotely challenged on speed: UDAX prototype ≈ 2.4 KB/s vs zstd ≈ 4–15 MB/s here. At this speed a 100 MB file would take **~11–12 hours** to encode — the spec's Problem B (speed) is the binding constraint, exactly as predicted.
3. **Chunking is nearly free.** 64 KB chunks add ~0.07% overhead — the chunk format is viable for streaming/parallel decode; the ratio does not depend on giant chunks.
4. **Problem C confirmed in miniature:** most MP4 bytes are already entropy-coded by H.264/AAC, so total headroom is small (~4–5% for any lossless method here). A better predictor (longer context, recurrent/attention state) is the only route past zstd-19 — the 16-byte MLP has already harvested the easy structure (headers, boxes, sync patterns).

## Honest limitations of this run
- Prototype scale: 150 KB test file, 4 training files, 6,000 steps, CPU-only. The spec's 100 MB run requires the optimized (batched/Rust) codec — v0.6 in the roadmap — before it is practical.
- One test file, one codec family (H.264/AAC). Results will vary with resolution, encoder settings and content.
- Training loss and test ratio come from ffmpeg-generated content, not camera footage.

## Next steps (roadmap)
- v0.3 ✅ SHA-verified round-trip · v0.4 ✅ independent chunks · v0.5 ✅ automatic raw fallback
- Longer training (20k+ steps) + more diverse corpus; GRU predictor comparison (carries state, likely better bits/byte at similar cost).
- Batch/vectorized inference + Rust range coding for speed; then the full 100 MB run.
