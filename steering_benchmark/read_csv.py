"""
Steering Benchmark: Result Aggregation

This script reads evaluation CSV files and computes aggregate steering success rates.

Usage:
    python -m steering_benchmark.read_csv --model_set falcon --model_version 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from steering_benchmark.parse_results import JUDGE
from steering_benchmark.model_loading import resolve_model_args


def readfile(fname):
    """Read a CSV file and count steered successes."""
    total = 0
    steered = 0
    with open(fname) as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            t = int(line.strip().split(',')[-1])
            steered += t
            total += 1
    print(f"Steered: {steered} out of {total}")
    return steered, total


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate steering evaluation results")
    parser.add_argument("--model_set", type=str, default='phi')
    parser.add_argument("--model_version", type=str, default=None)
    parser.add_argument("--model_size", type=str, default=None)
    args = parser.parse_args()

    methods = ['rfm']
    concepts = ['fears', 'personalities', 'moods', 'places', 'personas']

    # Configure model parameters
    MODEL_SET = args.model_set
    MODEL_VERSION, MODEL_SIZE = resolve_model_args(MODEL_SET, args.model_version, args.model_size)

    VERSIONS = [1, 2, 3, 4, 5]

    for VERSION in VERSIONS:
        VERSION_LABEL = f'_v{VERSION}' if VERSION >= 2 else ''

        results = {}
        for method in methods:
            results[method] = []
            for concept in concepts:
                fname = f'csvs/{method}_{concept}_{JUDGE}_outputs_500_concepts_{MODEL_SET}_{MODEL_VERSION}_{MODEL_SIZE}_english_only{VERSION_LABEL}.csv'
                try:
                    steered, total = readfile(fname)
                    results[method].append((concept, steered, total))
                except FileNotFoundError:
                    print(f"File not found: {fname}")

        for method in results:
            total = 0
            steered = 0
            for c, s, t in results[method]:
                total += t
                steered += s
            if total > 0:
                print(f"Version {VERSION} - Method: {method}, Steered: {steered}/{total} ({steered/total:.2%})")


if __name__ == "__main__":
    main()
