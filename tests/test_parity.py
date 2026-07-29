from __future__ import annotations

import ctypes
import os

import blake3 as upstream
import numpy as np
import pytest

import mojo_blake3


def payload(size: int) -> bytes:
    return bytes((index * 251 + index // 7 + 19) & 255 for index in range(size))


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            b"",
            "af1349b9f5f9a1a6a0404dea36dcc949"
            "9bcb25c9adc112b7cc9a93cae41f3262",
        ),
        (
            b"abc",
            "6437b3ac38465133ffb63b75273a8db5"
            "48c558465d79db03fd359c6cd5bd9d85",
        ),
    ],
)
def test_published_vectors(data, expected):
    assert mojo_blake3.blake3(data).hexdigest() == expected


@pytest.mark.parametrize(
    "size",
    [
        0,
        1,
        3,
        4,
        63,
        64,
        65,
        127,
        128,
        1023,
        1024,
        1025,
        2048,
        2049,
        3072,
        4096,
        8193,
        16384,
    ],
)
def test_unkeyed_boundary_parity(size):
    data = payload(size)
    assert mojo_blake3.blake3(data).digest() == upstream.blake3(data).digest()


@pytest.mark.parametrize("chunks", [3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33])
def test_tree_shapes_match_upstream(chunks):
    data = payload(chunks * 1024 + 17)
    assert mojo_blake3.blake3(data).digest() == upstream.blake3(data).digest()


def test_large_tree_matches_upstream():
    data = payload(2_000_123)
    assert mojo_blake3.blake3(data).digest() == upstream.blake3(data).digest()


@pytest.mark.parametrize("size", [15, 16, 17, 31, 47, 63, 65, 1023, 1025])
def test_simd_block_load_tails_match_upstream(size):
    data = payload(size)
    assert mojo_blake3.blake3(data).digest() == upstream.blake3(data).digest()


@pytest.mark.parametrize(
    "size",
    [4 * 1024 * 1024 - 1, 4 * 1024 * 1024, 4 * 1024 * 1024 + 17],
)
def test_parallel_threshold_and_remainder_match_upstream(size):
    pattern = bytes(range(256))
    data = (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]
    assert mojo_blake3.blake3(
        data, max_threads=mojo_blake3.blake3.AUTO
    ).digest() == upstream.blake3(data).digest()


@pytest.mark.parametrize("size", [0, 31, 64, 1024, 1025, 9001])
def test_keyed_hashing_matches_upstream(size):
    key = bytes(range(32))
    data = payload(size)
    assert mojo_blake3.blake3(data, key=key).digest(97) == upstream.blake3(
        data, key=key
    ).digest(97)


@pytest.mark.parametrize("size", [0, 64, 1024, 4097])
def test_derive_key_matches_upstream(size):
    context = "org.example.mojo-blake3 2026-07-29 purpose"
    data = payload(size)
    assert mojo_blake3.blake3(
        data, derive_key_context=context
    ).digest(131) == upstream.blake3(
        data, derive_key_context=context
    ).digest(
        131
    )


def test_unicode_derive_key_context_matches_upstream():
    context = "mojo-blake3 key context \N{SNOWMAN}"
    data = b"key material"
    assert mojo_blake3.blake3(
        data, derive_key_context=context
    ).digest() == upstream.blake3(
        data, derive_key_context=context
    ).digest()


@pytest.mark.parametrize("length", [0, 1, 31, 32, 33, 63, 64, 65, 257])
def test_xof_lengths_match_upstream(length):
    data = payload(5000)
    assert mojo_blake3.blake3(data).digest(length) == upstream.blake3(
        data
    ).digest(length)


@pytest.mark.parametrize("seek", [0, 1, 31, 63, 64, 65, 127, 1024, 12345])
def test_seekable_xof_matches_upstream(seek):
    data = payload(3333)
    assert mojo_blake3.blake3(data).digest(150, seek=seek) == upstream.blake3(
        data
    ).digest(150, seek=seek)


def test_hexdigest_length_and_seek_match_upstream():
    data = payload(2001)
    assert mojo_blake3.blake3(data).hexdigest(
        91, seek=73
    ) == upstream.blake3(data).hexdigest(91, seek=73)


def test_incremental_updates_and_chaining():
    parts = [payload(7), payload(1024), payload(13), payload(5000)]
    mojo = mojo_blake3.blake3()
    reference = upstream.blake3()
    for part in parts:
        assert mojo.update(part) is mojo
        assert reference.update(part) is reference
    assert mojo.digest(100) == reference.digest(100)


def test_bytearray_and_memoryview_inputs():
    data = bytearray(payload(3000))
    mojo = mojo_blake3.blake3(memoryview(data)[:1000])
    mojo.update(memoryview(data)[1000:])
    assert mojo.digest() == upstream.blake3(data).digest()


def test_numpy_buffer_input_matches_upstream_and_is_snapshotted():
    data = np.arange(4097, dtype=np.uint32)
    expected = upstream.blake3(memoryview(data).cast("B")).digest()
    mojo = mojo_blake3.blake3(data)
    data[:] = 0
    assert mojo.digest() == expected


def test_noncontiguous_buffer_is_rejected_without_crossing_ffi():
    data = np.arange(32, dtype=np.uint8)[::2]
    with pytest.raises(TypeError, match="bytes-like"):
        mojo_blake3.blake3(data)


def test_multibyte_numpy_input_hashes_raw_bytes_without_narrowing():
    data = np.array([0x0102, 0xABCD, 0xFFFF], dtype="<u2")
    raw = memoryview(data).cast("B")
    assert mojo_blake3.blake3(data).digest() == upstream.blake3(raw).digest()


def test_copy_is_independent_and_preserves_mode():
    key = bytes(reversed(range(32)))
    original = mojo_blake3.blake3(b"prefix", key=key)
    copied = original.copy()
    original.update(b"-left")
    copied.update(b"-right")
    assert original.digest() == upstream.blake3(
        b"prefix-left", key=key
    ).digest()
    assert copied.digest() == upstream.blake3(
        b"prefix-right", key=key
    ).digest()


def test_reset_preserves_keyed_mode():
    key = bytes(range(32))
    hasher = mojo_blake3.blake3(b"discard", key=key)
    assert hasher.reset() is None
    hasher.update(b"replacement")
    assert hasher.digest() == upstream.blake3(
        b"replacement", key=key
    ).digest()


@pytest.mark.parametrize("data", [b"", payload(12345)])
def test_update_mmap_matches_upstream(tmp_path, data):
    path = tmp_path / "input.bin"
    path.write_bytes(data)
    mojo = mojo_blake3.blake3(b"prefix")
    reference = upstream.blake3(b"prefix")
    assert mojo.update_mmap(path) is mojo
    assert reference.update_mmap(path) is reference
    assert mojo.digest() == reference.digest()


def test_standard_attributes_match_upstream():
    mojo = mojo_blake3.blake3()
    reference = upstream.blake3()
    for name in ("name", "digest_size", "block_size", "key_size", "AUTO"):
        assert getattr(mojo, name) == getattr(reference, name)


def test_usedforsecurity_is_accepted_and_does_not_change_hash():
    data = payload(1234)
    assert mojo_blake3.blake3(
        data, usedforsecurity=False
    ).digest() == upstream.blake3(data, usedforsecurity=False).digest()


@pytest.mark.parametrize("size", [0, 1, 31, 33])
def test_invalid_key_length_raises(size):
    with pytest.raises(ValueError):
        mojo_blake3.blake3(key=b"x" * size)


def test_key_and_context_are_mutually_exclusive():
    with pytest.raises(ValueError):
        mojo_blake3.blake3(
            key=b"x" * 32, derive_key_context="org.example.context"
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"derive_key_context": b"bytes"}, TypeError),
        ({"max_threads": 0}, ValueError),
        ({"max_threads": 1.5}, TypeError),
        ({"usedforsecurity": 1}, TypeError),
    ],
)
def test_constructor_validation(kwargs, error):
    with pytest.raises(error):
        mojo_blake3.blake3(**kwargs)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda h: h.digest(-1), ValueError),
        (lambda h: h.digest(1.5), TypeError),
        (lambda h: h.digest(seek=-1), ValueError),
        (lambda h: h.digest(seek=1.5), TypeError),
        (lambda h: h.digest(seek=1 << 64), ValueError),
    ],
)
def test_output_validation(call, error):
    with pytest.raises(error):
        call(mojo_blake3.blake3())


def test_auto_thread_constant_is_accepted():
    data = os.urandom(2048)
    assert mojo_blake3.blake3(
        data, max_threads=mojo_blake3.blake3.AUTO
    ).digest() == upstream.blake3(
        data, max_threads=upstream.blake3.AUTO
    ).digest()


def test_ffi_rejects_invalid_addresses_and_values():
    from mojo_blake3._lib import library

    function = library().mojo_blake3_hash
    byte = ctypes.c_ubyte(0)
    address = ctypes.addressof(byte)
    assert function(0, 1, address, 0, 0, address, 1, 1) == -2
    assert function(address, 0, 0, 0, 0, address, 1, 1) == -2
    assert function(address, 0, address, 0, 0, 0, 1, 1) == -2
    assert function(address, 0, address, 1, 0, address, 1, 1) == -3
    assert function(address, 0, address, 0, 0, address, 1, 0) == -4


def test_ffi_accepts_null_for_unused_empty_buffers():
    from mojo_blake3._lib import library

    function = library().mojo_blake3_hash
    key = (ctypes.c_ubyte * 32)()
    assert function(0, 0, ctypes.addressof(key), 0, 0, 0, 0, 1) == 0
