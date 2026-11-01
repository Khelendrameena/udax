"""Milestone test (spec section 8): encode + decode a small MP4 sample
and verify SHA-256 equality.

Usage:
    python test_udax.py [sample_path] [num_bytes]
Default: first 4096 bytes of sample.mp4.
"""
import hashlib
import sys
import time

from udax_codec import decode_chunk, encode_chunk, init_determinism
from udax_model import load_model


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    sample_path = sys.argv[1] if len(sys.argv) > 1 else "sample.mp4"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

    init_determinism()
    model, _ckpt = load_model("udax_bytenet_v1.pt")

    with open(sample_path, "rb") as f:
        original = f.read(n)
    print(f"Original: {len(original)} bytes")

    t0 = time.perf_counter()
    payload = encode_chunk(original, model)
    t_enc = time.perf_counter() - t0

    if len(payload) >= len(original):
        mode = "raw"
        payload = original
    else:
        mode = "neural"
    print(f"Encoded:  {len(payload)} bytes (mode: {mode}, {t_enc:.2f}s)")

    t0 = time.perf_counter()
    if mode == "neural":
        reconstructed = decode_chunk(payload, len(original), model)
    else:
        reconstructed = payload
    t_dec = time.perf_counter() - t0
    print(f"Decoded in {t_dec:.2f}s")

    print(f"Original SHA256:      {sha256(original)}")
    print(f"Reconstructed SHA256: {sha256(reconstructed)}")
    if reconstructed == original:
        print("SUCCESS: Perfect lossless reconstruction!")
    else:
        print("FAILURE: bytes differ")
        sys.exit(1)


if __name__ == "__main__":
    main()
