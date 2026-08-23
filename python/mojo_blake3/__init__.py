"""Python-compatible BLAKE3 hashing backed by Mojo."""

from __future__ import annotations

import mmap
import os
import struct
from typing import Any

from ._lib import hash_bytes

__all__ = ["blake3"]
__version__ = "0.1.0"

_IV = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)
_IV_BYTES = struct.pack("<8I", *_IV)
_KEYED_HASH = 16
_DERIVE_KEY_CONTEXT = 32
_DERIVE_KEY_MATERIAL = 64


def _byteslike(value: Any) -> memoryview:
    try:
        return memoryview(value).cast("B")
    except (TypeError, ValueError) as error:
        raise TypeError("a bytes-like object is required") from error


class blake3:
    """Incremental BLAKE3 hasher with keyed, derive-key, and XOF modes."""

    AUTO = -1
    digest_size = 32
    block_size = 64
    key_size = 32
    name = "blake3"

    def __init__(
        self,
        data: Any = None,
        /,
        *,
        key: Any = None,
        derive_key_context: str | None = None,
        max_threads: int = 1,
        usedforsecurity: bool = True,
    ):
        if key is not None and derive_key_context is not None:
            raise ValueError("cannot use key and derive_key_context at the same time")
        if not isinstance(max_threads, int):
            raise TypeError("max_threads must be an integer")
        if max_threads != self.AUTO and max_threads < 1:
            raise ValueError("max_threads must be positive or blake3.AUTO")
        if not isinstance(usedforsecurity, bool):
            raise TypeError("usedforsecurity must be a bool")

        if key is not None:
            key_view = _byteslike(key)
            if len(key_view) != 32:
                raise ValueError("expected a 32-byte key")
            self._key = key_view.tobytes()
            self._flags = _KEYED_HASH
        elif derive_key_context is not None:
            if not isinstance(derive_key_context, str):
                raise TypeError("derive_key_context must be a string")
            context = derive_key_context.encode("utf-8")
            self._key = hash_bytes(
                context, _IV_BYTES, _DERIVE_KEY_CONTEXT, 32, 0
            )
            self._flags = _DERIVE_KEY_MATERIAL
        else:
            self._key = _IV_BYTES
            self._flags = 0

        self._data: bytes | bytearray = b""
        self._max_threads = max_threads
        self._usedforsecurity = usedforsecurity
        if data is not None:
            self.update(data)

    def update(self, data: Any) -> "blake3":
        view = _byteslike(data)
        if not self._data and isinstance(data, bytes):
            self._data = data
            return self
        if isinstance(self._data, bytes):
            self._data = bytearray(self._data)
        self._data.extend(view)
        return self

    def update_mmap(
        self, path: os.PathLike[str] | str | bytes
    ) -> "blake3":
        with open(path, "rb") as file:
            if os.fstat(file.fileno()).st_size == 0:
                return self
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                self.update(mapping)
        return self

    @staticmethod
    def _output_args(length: int, seek: int) -> tuple[int, int]:
        if not isinstance(length, int):
            raise TypeError("length must be an integer")
        if not isinstance(seek, int):
            raise TypeError("seek must be an integer")
        if length < 0:
            raise ValueError("length must be non-negative")
        if seek < 0 or seek > (1 << 64) - 1:
            raise ValueError("seek must fit in an unsigned 64-bit integer")
        return length, seek

    def digest(self, length: int = 32, *, seek: int = 0) -> bytes:
        length, seek = self._output_args(length, seek)
        return hash_bytes(
            self._data,
            self._key,
            self._flags,
            length,
            seek,
            self._max_threads,
        )

    def hexdigest(self, length: int = 32, *, seek: int = 0) -> str:
        return self.digest(length, seek=seek).hex()

    def copy(self) -> "blake3":
        duplicate = object.__new__(type(self))
        duplicate._key = self._key
        duplicate._flags = self._flags
        if isinstance(self._data, bytes):
            duplicate._data = self._data
        else:
            duplicate._data = self._data.copy()
        duplicate._max_threads = self._max_threads
        duplicate._usedforsecurity = self._usedforsecurity
        return duplicate

    def reset(self) -> None:
        self._data = b""
