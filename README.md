# mojo-blake3

BLAKE3 tree hashing implemented in Mojo, with a Python API shaped like the
[`blake3`](https://pypi.org/project/blake3/) package.

This is a complete implementation of the BLAKE3 hash construction rather than a call
through to a system crypto library. Chunk compression, binary tree reduction, keyed
hashing, derive-key mode, and extensible output all execute in the compiled Mojo shared
library.

The Python package is named `mojo_blake3`, so existing code can use the familiar API with
one import change:

```python
import mojo_blake3 as blake3

hasher = blake3.blake3(b"hello ")
hasher.update(b"world")
print(hasher.hexdigest())

key = bytes(range(32))
tag = blake3.blake3(b"authenticated data", key=key).digest()
xof = blake3.blake3(b"seed").digest(length=100, seek=64)
```

## Scope and coverage

The target is the public API of the upstream Python `blake3` package, not every
component in the upstream BLAKE3 repository. The following behavior is covered by
parity tests against upstream `blake3` 1.0.9:

- `blake3(data=None, *, key=None, derive_key_context=None, max_threads=1,
  usedforsecurity=True)`
- incremental `update`, `copy`, and `reset`
- `digest(length=32, *, seek=0)` and `hexdigest(...)`
- `update_mmap(path)`
- unkeyed, 32-byte keyed, and derive-key hashing
- arbitrary-length seekable XOF output
- the standard `name`, `digest_size`, `block_size`, `key_size`, and `AUTO` attributes

The tests cover published vectors, block and chunk boundaries, SIMD load tails, uneven
tree shapes, the parallel threshold and remainder, multi-megabyte inputs, keyed and
derive-key modes, XOF lengths and seeks, incremental updates, copying, resetting, mmap
input, buffer validation, and direct C ABI validation.

Not covered:

- the upstream repository's Rust, C, command-line, and architecture-specific APIs
- a persistent streaming state in Mojo; Python buffers incremental and mmap input until
  output is requested
- more than 16 Mojo workers; `AUTO` and larger values are capped at 16
- non-Linux platforms or a prebuilt wheel; this source release builds a Linux shared
  library with the pinned Mojo toolchain

The package deliberately does not replace the upstream `blake3` import name, so both
implementations can be installed for parity testing.

## Install and run

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-blake3.so`. From the repository checkout, the example
above runs with:

```bash
pixi run python -c \
  'import mojo_blake3 as b; print(b.blake3(b"abc").hexdigest())'
```

Expected output:

```text
6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85
```

## Benchmarks

Measured by running `pixi run bench` on this machine on 2026-08-23: Intel Xeon E5-2697
v4 at 2.30 GHz, Linux 6.8.0-136-generic, Python 3.13.14. Times are the best per call from three timed batches
and include construction of the Python hasher. Both implementations use their default
single worker thread except for the explicitly labeled 16-thread row. The relative
column is upstream time divided by Mojo time, so a value below 1 means Mojo is slower.

| workload | mojo-blake3 | upstream blake3 | relative | Mojo throughput |
| --- | ---: | ---: | ---: | ---: |
| unkeyed 64 KiB | 0.106 ms | 0.031 ms | 0.29x | 618.0 MB/s |
| unkeyed 1 MiB | 1.171 ms | 0.373 ms | 0.32x | 895.6 MB/s |
| unkeyed 16 MiB | 20.813 ms | 6.191 ms | 0.30x | 806.1 MB/s |
| unkeyed 16 MiB, 16 threads | 8.252 ms | 1.358 ms | 0.16x | 2033.2 MB/s |
| keyed 16 MiB | 20.231 ms | 6.688 ms | 0.33x | 829.3 MB/s |
| 1 MiB input + 1 MiB XOF | 3.110 ms | 1.769 ms | 0.57x | 337.1 MB/s |

These results describe this run only. The Mojo implementation remains slower than
upstream in every measured workload.

No GPU path is shipped. BLAKE3 compression is integer-heavy, but this package hashes a
single contiguous message per Python call, so a device path pays host/device transfers
on every digest. A light implementation probe also found that the pinned Mojo 1.1
toolchain does not expose the `std.gpu` host launch API used by earlier nightlies, even
with the matching `max` runtime installed. The probe was removed rather than retaining
an unlaunchable or unmeasured path; CPU remains the default and only device.

## How it works

Python owns input, key, and output buffers. Their addresses cross the C ABI as 64-bit
integers through `ctypes`; the exported Mojo function reconstructs
`UnsafePointer[UInt8, AnyOrigin[mut=True]]` values. No Mojo allocation crosses the ABI,
and all multibyte words are loaded and emitted in BLAKE3's little-endian order.

The kernel splits input into 1024-byte chunks and compresses each chunk as 64-byte
blocks. Compression maps the four independent G functions in each half-round onto SIMD
lanes, with compile-time-unrolled message schedules. Full chunks are processed in
AVX2-width pairs, with a scalar chunk remainder; XOF blocks use the same paired-counter
strategy with scalar seek and length tails. Block loading uses unaligned SIMD loads plus
a scalar tail and skips zeroing when all 64 bytes are overwritten. For sufficiently
large inputs, independent chunk pairs are computed in parallel and then merged with a
54-level fixed stack according to the BLAKE3 binary tree rules. The final output object
is retained so the root flag can be applied correctly. Keyed and derive-key modes use
the same tree with the mode flags and key words specified by BLAKE3.

Immutable Python `bytes` inputs are retained and passed through `ctypes` without a
constructor or FFI copy. Mutable buffers are snapshotted once to preserve upstream
hasher semantics, and that contiguous snapshot crosses the FFI boundary zero-copy.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task holds `/tmp/mojo-bench.lock` so concurrent jobs on the same machine
do not overlap.

MIT licensed.
