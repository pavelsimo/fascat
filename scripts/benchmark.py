from __future__ import annotations

import argparse
import json
from pathlib import Path

from fascat.benchmark import BenchmarkOptions, run_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fascat conversion throughput.")
    parser.add_argument("inputs", nargs="+", type=Path, help="STEP, IGES, or BREP input files.")
    parser.add_argument("--output-dir", type=Path, default=Path("dist/benchmarks"), help="Directory for outputs.")
    parser.add_argument("--output-suffix", default=".glb", help="Output suffix, for example .glb or .usdc.")
    parser.add_argument("--profile", default="realtime-desktop", help="Conversion profile.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of runs per input.")
    parser.add_argument("--validate-output", action="store_true", help="Run output validation inside each benchmark.")
    args = parser.parse_args()

    report = run_benchmarks(
        BenchmarkOptions(
            inputs=tuple(args.inputs),
            output_dir=args.output_dir,
            output_suffix=args.output_suffix,
            profile=args.profile,
            repeat=args.repeat,
            validate_output=args.validate_output,
        )
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
