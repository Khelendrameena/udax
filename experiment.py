"""UDAX Experiment 001 — Lossless MP4 Compression.

    A  Original MP4, no compression
    B  Zstandard whole-file compression (levels 3 / 10 / 19)
    C  UDAX chunks of 64 KB
    D  UDAX chunks of 256 KB
    E  UDAX chunks of 1 MB
    F  UDAX chunks of 4 MB
    G  UDAX chunks of 16 MB

Every UDAX configuration is decoded back and the whole-file SHA-256
must match the original, or the run is marked FAILED.

Note on E/F/G: when the chunk size is >= the file size, the file is a
single chunk and the encoding is identical by construction; it is
computed once and reused for those rows (only the chunk_size field
in the file header differs, which is the same size). The report
states the test-file size so ratios can be judged honestly — see
EXPERIMENT-001-REPORT.md for the 100 MB caveats (spec Problems B & C).
"""
import hashlib
import json
import sys
import time

import zstandard

from udax_codec import decode_file, encode_file, init_determinism
from udax_model import load_model, model_hash

KB = 1024


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    test_path = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
    with open(test_path, "rb") as f:
        data = f.read()
    original_sha = sha256(data)
    print(f"Test file: {test_path}  {len(data)} bytes  sha256={original_sha[:16]}…")

    init_determinism()
    model, _ckpt = load_model("udax_bytenet_v1.pt")
    mhash = model_hash("udax_bytenet_v1.pt")

    results = []

    # ---- A: original
    results.append({"test": "A", "method": "Original (no compression)",
                    "size": len(data), "ratio": 1.0, "enc_s": 0.0, "dec_s": 0.0,
                    "sha_ok": True})

    # ---- B: zstandard whole file
    for level in (3, 10, 19):
        t0 = time.perf_counter()
        blob = zstandard.ZstdCompressor(level=level).compress(data)
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        back = zstandard.ZstdDecompressor().decompress(blob, max_output_size=len(data))
        t_dec = time.perf_counter() - t0
        results.append({"test": "B", "method": f"Zstandard whole-file, level {level}",
                        "size": len(blob), "ratio": len(blob) / len(data),
                        "enc_s": round(t_enc, 3), "dec_s": round(t_dec, 3),
                        "sha_ok": sha256(back) == original_sha})

    # ---- C..G: UDAX chunk sizes
    single_chunk_cache = None  # (blob, stats, enc_s) for chunk_size >= file size
    for name, chunk_size in [("C", 64 * KB), ("D", 256 * KB), ("E", 1 * KB * KB),
                             ("F", 4 * KB * KB), ("G", 16 * KB * KB)]:
        if chunk_size >= len(data) and single_chunk_cache is not None:
            blob, stats, t_enc = single_chunk_cache
            # rebuild header chunk_size field difference is size-neutral;
            # blob length is identical, decode verifies against same data.
        else:
            t0 = time.perf_counter()
            blob, stats = encode_file(data, model, chunk_size, mhash)
            t_enc = time.perf_counter() - t0
            if chunk_size >= len(data):
                single_chunk_cache = (blob, stats, t_enc)
        t0 = time.perf_counter()
        back = decode_file(blob, model)
        t_dec = time.perf_counter() - t0
        results.append({"test": name,
                        "method": f"UDAX chunks {chunk_size // KB} KB "
                                  f"({stats['neural']} neural / {stats['raw']} raw)",
                        "size": len(blob), "ratio": len(blob) / len(data),
                        "enc_s": round(t_enc, 2), "dec_s": round(t_dec, 2),
                        "sha_ok": sha256(back) == original_sha})

    # ---- report
    print(f"\n{'Test':<5}{'Method':<52}{'Size (B)':>11}{'Ratio':>8}{'Enc (s)':>9}{'Dec (s)':>9}  SHA")
    for r in results:
        print(f"{r['test']:<5}{r['method']:<52}{r['size']:>11}{r['ratio']:>8.3f}"
              f"{r['enc_s']:>9}{r['dec_s']:>9}  {'OK' if r['sha_ok'] else 'FAIL'}")

    with open("experiment_results.json", "w") as f:
        json.dump({"test_file": test_path, "test_size": len(data),
                   "sha256": original_sha, "results": results}, f, indent=2)
    print("\nSaved experiment_results.json")
    if not all(r["sha_ok"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
