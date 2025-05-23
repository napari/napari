import argparse
import importlib
from typing import NamedTuple

from .utils import run_benchmark_from_module


class BenchmarkIdentifier(NamedTuple):
    module: str
    klass: str
    method: str


def split_identifier(value: str) -> BenchmarkIdentifier:
    """Split a string into a module and class identifier."""
    parts = value.split('.')
    if len(parts) != 3:
        raise argparse.ArgumentError(
            "Benchmark identifier should be in the form 'module.class.benchmark'"
        )
    return BenchmarkIdentifier(*parts)


def main():
    parser = argparse.ArgumentParser(
        description='Run selected napari benchmarks for debugging.'
    )
    parser.add_argument(
        'benchmark', type=split_identifier, help='Benchmark to run.'
    )
    args = parser.parse_args()
    module = importlib.import_module(
        f'.{args.benchmark.module}', package='napari.benchmarks'
    )
    run_benchmark_from_module(
        module, args.benchmark.klass, args.benchmark.method
    )


if __name__ == '__main__':
    main()
