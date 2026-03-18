#!/usr/bin/env python3
"""Micro-benchmarks for hot paths in local_read_mcp.server.app."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from local_read_mcp.server import app as server_app


@dataclass
class BenchResult:
    name: str
    mean_seconds: float
    min_seconds: float
    max_seconds: float
    per_op_us: float


def run_benchmark(name: str, fn, iterations: int, repeats: int) -> BenchResult:
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        timings.append(time.perf_counter() - start)

    mean_seconds = sum(timings) / len(timings)
    return BenchResult(
        name=name,
        mean_seconds=mean_seconds,
        min_seconds=min(timings),
        max_seconds=max(timings),
        per_op_us=(mean_seconds / iterations) * 1_000_000,
    )


def format_result(result: BenchResult) -> str:
    return (
        f"{result.name}: "
        f"mean={result.mean_seconds:.6f}s "
        f"min={result.min_seconds:.6f}s "
        f"max={result.max_seconds:.6f}s "
        f"per_op={result.per_op_us:.3f}us"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark server hot paths")
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    # Case A: Pre-optimization baseline pattern (build wrapper every call)
    def create_wrapper_each_time():
        server_app.create_simple_converter_wrapper(server_app.TextConverter, "text")

    # Case B: Optimized pattern (cached wrapper lookup)
    server_app._SIMPLE_CONVERTER_CACHE.clear()
    server_app.get_simple_converter_wrapper("text")  # warm cache

    def cached_wrapper_lookup():
        server_app.get_simple_converter_wrapper("text")

    baseline = run_benchmark(
        "create_wrapper_each_time",
        create_wrapper_each_time,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    optimized = run_benchmark(
        "cached_wrapper_lookup",
        cached_wrapper_lookup,
        iterations=args.iterations,
        repeats=args.repeats,
    )

    print(format_result(baseline))
    print(format_result(optimized))
    speedup = baseline.mean_seconds / optimized.mean_seconds if optimized.mean_seconds else float("inf")
    print(f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()

