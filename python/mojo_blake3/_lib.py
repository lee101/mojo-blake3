"""ctypes bridge to the compiled Mojo BLAKE3 kernel."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_BLAKE3_LIB") or os.path.join(
    ROOT, "dist", "libmojo-blake3.so"
)

I = ctypes.c_int64
U64 = ctypes.c_uint64


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "blake3.mojo")
    if os.environ.get("MOJO_BLAKE3_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not os.path.exists(source):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(
            "compiled library not found; run `pixi run build` or set "
            "MOJO_BLAKE3_LIB=/path/to/libmojo-blake3.so"
        )
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= os.path.getmtime(source):
            return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo not found; run inside the Pixi environment")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    process = subprocess.run(
        [
            mojo,
            "build",
            "--emit",
            "shared-lib",
            source,
            "-o",
            LIB,
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if process.returncode or not os.path.exists(LIB):
        raise BuildError((process.stderr or process.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def library() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        function = _library.mojo_blake3_hash
        function.argtypes = [I, I, I, I, U64, I, I, I]
        function.restype = I
    return _library


def _buffer(value: bytes | bytearray | memoryview):
    view = memoryview(value).cast("B")
    if not view:
        storage = (ctypes.c_ubyte * 1)()
    elif view.readonly:
        storage = (ctypes.c_ubyte * len(view)).from_buffer_copy(view)
    else:
        storage = (ctypes.c_ubyte * len(view)).from_buffer(view)
    return storage, len(view)


def hash_bytes(
    data: bytes | bytearray | memoryview,
    key: bytes,
    flags: int,
    length: int,
    seek: int,
    max_threads: int = 1,
) -> bytes:
    source, size = _buffer(data)
    key_buffer, key_size = _buffer(key)
    if key_size != 32:
        raise ValueError("BLAKE3 keys must be exactly 32 bytes")
    if size > sys.maxsize:
        raise OverflowError("input is too large for the Mojo ABI")
    if length < 0 or length > sys.maxsize:
        raise OverflowError("output length is too large for the Mojo ABI")
    if flags not in (0, 16, 32, 64):
        raise ValueError("invalid BLAKE3 mode flags")
    if max_threads < -(1 << 63) or max_threads > (1 << 63) - 1:
        raise OverflowError("max_threads does not fit in the Mojo ABI")
    destination = (ctypes.c_ubyte * max(length, 1))()
    status = library().mojo_blake3_hash(
        ctypes.addressof(source),
        size,
        ctypes.addressof(key_buffer),
        flags,
        seek,
        ctypes.addressof(destination),
        length,
        max_threads,
    )
    if status:
        raise RuntimeError(f"Mojo BLAKE3 kernel failed with status {status}")
    if length == 0:
        return b""
    return bytes(destination)
