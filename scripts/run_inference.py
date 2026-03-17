#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from engines import hf, nano_vllm, sglang


ENGINES = {
    "hf": hf,
    "sglang": sglang,
    "nano_vllm": nano_vllm,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinerU-Diffusion inference with a selected engine.")
    parser.add_argument("--engine", choices=sorted(ENGINES.keys()), default="hf")
    args, remaining = parser.parse_known_args()

    engine_parser = argparse.ArgumentParser(
        description=f"Run MinerU-Diffusion inference with the {args.engine} engine."
    )
    engine_parser.add_argument("--engine", choices=sorted(ENGINES.keys()), default=args.engine)
    ENGINES[args.engine].add_arguments(engine_parser)
    return engine_parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    ENGINES[args.engine].run(args)


if __name__ == "__main__":
    main()
