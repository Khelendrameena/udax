"""UDAX codec — neural predictor + range encoder/decoder + chunk format.

Per chunk (spec section 9):
    UDAX_CHUNK_HEADER   version, chunk_id, original_size, payload_size,
                        compression_mode, model_hash, checksum
    UDAX_CHUNK_PAYLOAD  seed bytes + range-coded stream   (neural mode)
                        or the raw chunk bytes             (raw mode)

The first `context` bytes of every chunk are stored literally as the
prediction seed; the decoder resets its context at each chunk, so any
chunk can be decoded independently (streaming / parallel friendly).

Raw fallback: if the neural payload is not smaller than the raw chunk,
the chunk is stored raw (mode 0). Most MP4 payload bytes are already
entropy-coded by the video codec, so this fallback is expected to
trigger often — that is a legitimate experimental result, not a bug.
"""
import hashlib
import struct
from collections import deque

import constriction
import numpy as np
import torch

from udax_model import predict_probabilities

CONTEXT = 16

MODE_RAW = 0
MODE_NEURAL = 1

_FILE_MAGIC = b"UDAXF1"
_CHUNK_MAGIC = b"UC"
_CHUNK_VERSION = 1
# magic(2) version(1) mode(1) chunk_id(4) original_size(4) payload_size(4) checksum(32)
_CHUNK_HEADER = struct.Struct("<2sBBIII32s")
# magic(6) chunk_size(4) total_size(8) model_hash(32)
_FILE_HEADER = struct.Struct("<6sIQ32s")


def init_determinism():
    """Encoder and decoder MUST produce identical probabilities.
    Single-threaded CPU inference + deterministic algorithms."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------- chunk core

def encode_chunk(raw: bytes, model) -> bytes:
    """Neural-encode one chunk. Returns seed + range-coded payload."""
    if len(raw) <= CONTEXT:
        return bytes(raw)  # too small to code; caller treats as raw
    seed = raw[:CONTEXT]
    context = deque(seed, maxlen=CONTEXT)
    coder = constriction.stream.queue.RangeEncoder()
    for actual_byte in raw[CONTEXT:]:
        probs = predict_probabilities(model, context)
        entropy_model = constriction.stream.model.Categorical(probs, perfect=False)
        coder.encode(int(actual_byte), entropy_model)
        context.append(actual_byte)
    coded = coder.get_compressed().astype("<u4").tobytes()
    return bytes(seed) + coded


def decode_chunk(payload: bytes, original_length: int, model) -> bytes:
    """Reverse encode_chunk. `payload` is seed + range-coded words."""
    if original_length <= CONTEXT:
        return bytes(payload[:original_length])
    seed = payload[:CONTEXT]
    words = np.frombuffer(payload[CONTEXT:], dtype="<u4").astype(np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(words)
    context = deque(seed, maxlen=CONTEXT)
    output = bytearray(seed)
    for _ in range(original_length - CONTEXT):
        probs = predict_probabilities(model, context)
        entropy_model = constriction.stream.model.Categorical(probs, perfect=False)
        byte = int(decoder.decode(entropy_model))
        output.append(byte)
        context.append(byte)
    return bytes(output)


# ---------------------------------------------------------------- file format

def encode_file(data: bytes, model, chunk_size: int, model_hash_hex: str):
    """Encode a whole file chunk-by-chunk. Returns (blob, stats dict)."""
    model_hash = bytes.fromhex(model_hash_hex)
    out = bytearray()
    out += _FILE_HEADER.pack(_FILE_MAGIC, chunk_size, len(data), model_hash)
    stats = {"chunks": 0, "neural": 0, "raw": 0, "neural_saved_bytes": 0}
    for chunk_id, start in enumerate(range(0, len(data), chunk_size)):
        chunk = data[start:start + chunk_size]
        checksum = hashlib.sha256(chunk).digest()
        payload = encode_chunk(chunk, model) if len(chunk) > CONTEXT else bytes(chunk)
        if len(payload) < len(chunk):
            mode = MODE_NEURAL
            stats["neural"] += 1
            stats["neural_saved_bytes"] += len(chunk) - len(payload)
        else:
            mode = MODE_RAW
            payload = bytes(chunk)
            stats["raw"] += 1
        out += _CHUNK_HEADER.pack(
            _CHUNK_MAGIC, _CHUNK_VERSION, mode, chunk_id,
            len(chunk), len(payload), checksum)
        out += payload
        stats["chunks"] += 1
    return bytes(out), stats


def decode_file(blob: bytes, model) -> bytes:
    """Decode a UDAX file blob, verifying every chunk's SHA-256."""
    magic, _chunk_size, total_size, _mhash = _FILE_HEADER.unpack_from(blob, 0)
    if magic != _FILE_MAGIC:
        raise ValueError("Not a UDAX file")
    offset = _FILE_HEADER.size
    out = bytearray()
    while offset < len(blob):
        cmagic, _ver, mode, _cid, orig_size, payload_size, checksum = \
            _CHUNK_HEADER.unpack_from(blob, offset)
        if cmagic != _CHUNK_MAGIC:
            raise ValueError("Corrupt chunk header")
        offset += _CHUNK_HEADER.size
        payload = blob[offset:offset + payload_size]
        offset += payload_size
        if mode == MODE_NEURAL:
            chunk = decode_chunk(payload, orig_size, model)
        else:
            chunk = bytes(payload[:orig_size])
        if hashlib.sha256(chunk).digest() != checksum:
            raise ValueError(f"Chunk checksum mismatch (chunk id {_cid})")
        out += chunk
    if len(out) != total_size:
        raise ValueError("Size mismatch after decode")
    return bytes(out)
