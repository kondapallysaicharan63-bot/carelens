"""Encrypt event clips at rest (AES-256-GCM). Wellness monitor, not a medical device.

File format: b"CLENC1" | 8-byte key id (SHA-256(key)[:8]) | 12-byte nonce | ciphertext+tag.
The associated data is the encrypted file's basename, so a ciphertext renamed or swapped
with another clip fails authentication. Whole files are handled in memory (clips are short);
files over MAX_BYTES are refused.

The key is never written to disk by this module: `genkey` prints it for the operator to
put in .env as CARELENS_CLIP_KEY. Plaintext briefly exists before encryption and unlink is
not a secure erase on flash storage, so also use full-disk encryption (LUKS).
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import os
import sys
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"CLENC1"
SUFFIX = ".enc"
KEY_ENV = "CARELENS_CLIP_KEY"
MAX_BYTES = 200 * 1024 * 1024
_ID, _NONCE = 8, 12


class ClipLockError(ValueError):
    """Any clip-encryption problem; messages never contain key material."""


def generate_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode()


def load_key(value: str) -> bytes:
    try:
        key = base64.urlsafe_b64decode((value or "").strip().encode())
    except (binascii.Error, ValueError, UnicodeEncodeError):
        key = b""
    if len(key) != 32:
        raise ClipLockError("clip key is missing or invalid (expected urlsafe base64 of 32 bytes; "
                            "make one with: python -m carelens.cliplock genkey)")
    return key


def _key_id(key: bytes) -> bytes:
    return hashlib.sha256(key).digest()[:_ID]


def encrypt_file(plain_path, key: bytes, out_dir=None) -> Path:
    """Write `<name>.enc` (in `out_dir` or beside the input) atomically, then delete the plaintext."""
    src = Path(plain_path)
    if src.stat().st_size > MAX_BYTES:
        raise ClipLockError(f"clip too large to encrypt (> {MAX_BYTES // 2**20} MB)")
    dest = Path(out_dir or src.parent) / (src.name + SUFFIX)
    nonce = os.urandom(_NONCE)
    blob = MAGIC + _key_id(key) + nonce + AESGCM(key).encrypt(
        nonce, src.read_bytes(), dest.name.encode())
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    src.unlink()
    return dest


def decrypt_to_bytes(enc_path, key: bytes) -> bytes:
    p = Path(enc_path)
    if p.stat().st_size > MAX_BYTES + 64:
        raise ClipLockError("encrypted clip too large")
    blob = p.read_bytes()
    head = len(MAGIC) + _ID + _NONCE
    if len(blob) < head + 16 or not blob.startswith(MAGIC):
        raise ClipLockError("not an encrypted CareLens clip (bad magic)")
    if blob[len(MAGIC):len(MAGIC) + _ID] != _key_id(key):
        raise ClipLockError("clip was encrypted with a different key")
    nonce = blob[len(MAGIC) + _ID:head]
    try:
        return AESGCM(key).decrypt(nonce, blob[head:], p.name.encode())
    except InvalidTag:
        raise ClipLockError("clip failed authentication (tampered, corrupted or renamed)") from None


def main(argv=None):
    ap = argparse.ArgumentParser(description="CareLens clip encryption helper")
    ap.add_argument("cmd", choices=["genkey"])
    ap.parse_args(argv)
    print(generate_key())
    print(f"# put the line above in .env as {KEY_ENV}=<key> (this tool never writes it to disk; "
          "back it up - clips are unrecoverable without it)", file=sys.stderr)


if __name__ == "__main__":
    main()
