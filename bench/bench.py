"""Benchmark mojo-blake3 against the upstream Rust-backed Python package."""

from __future__ import annotations

import os
import platform
import sys
import time

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import blake3 as upstream  # noqa: E402
import mojo_blake3  # noqa: E402


def measure(function, rounds: int = 3) -> float:
    function()
    start = time.perf_counter()
    function()
    first = time.perf_counter() - start
    repetitions = max(1, min(200, int(0.15 / max(first, 1e-9))))
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(repetitions):
            function()
        best = min(best, (time.perf_counter() - start) / repetitions)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def benchmark(
    name: str,
    mojo_function,
    upstream_function,
    input_bytes: int,
) -> tuple[str, float, float, float, float]:
    assert mojo_function() == upstream_function()
    mojo_seconds = measure(mojo_function)
    upstream_seconds = measure(upstream_function)
    speed = upstream_seconds / mojo_seconds
    throughput = input_bytes / mojo_seconds / 1_000_000
    return name, mojo_seconds, upstream_seconds, speed, throughput


def main() -> None:
    mib = bytes(range(256)) * 4096
    small = mib[: 64 * 1024]
    medium = mib
    large = mib * 16
    key = bytes(range(32))

    cases = [
        (
            "unkeyed 64 KiB",
            lambda: mojo_blake3.blake3(small).digest(),
            lambda: upstream.blake3(small).digest(),
            len(small),
        ),
        (
            "unkeyed 1 MiB",
            lambda: mojo_blake3.blake3(medium).digest(),
            lambda: upstream.blake3(medium).digest(),
            len(medium),
        ),
        (
            "unkeyed 16 MiB",
            lambda: mojo_blake3.blake3(large).digest(),
            lambda: upstream.blake3(large).digest(),
            len(large),
        ),
        (
            "unkeyed 16 MiB, 16 threads",
            lambda: mojo_blake3.blake3(large, max_threads=16).digest(),
            lambda: upstream.blake3(large, max_threads=16).digest(),
            len(large),
        ),
        (
            "keyed 16 MiB",
            lambda: mojo_blake3.blake3(large, key=key).digest(),
            lambda: upstream.blake3(large, key=key).digest(),
            len(large),
        ),
        (
            "1 MiB input + 1 MiB XOF",
            lambda: mojo_blake3.blake3(medium).digest(1024 * 1024),
            lambda: upstream.blake3(medium).digest(1024 * 1024),
            len(medium),
        ),
    ]

    print(f"Machine: {cpu_name()}")
    print(f"Platform: {platform.platform()}; Python {platform.python_version()}")
    print()
    print("| workload | mojo-blake3 | upstream blake3 | relative | Mojo throughput |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for case in cases:
        name, mojo_s, upstream_s, speed, throughput = benchmark(*case)
        print(
            f"| {name} | {mojo_s * 1e3:.3f} ms | "
            f"{upstream_s * 1e3:.3f} ms | {speed:.2f}x | "
            f"{throughput:.1f} MB/s |"
        )


if __name__ == "__main__":
    main()
