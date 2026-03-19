#!/usr/bin/env python3

import argparse
import importlib
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

ENGINES = {
    "hf": "engines.hf",
    "nano_dvlm": "engines.nano_dvlm",
    "sglang": "engines.sglang",
}


def _load_engine(name: str):
    return importlib.import_module(ENGINES[name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinerU-Diffusion inference with a selected engine.")
    parser.add_argument("--engine", choices=sorted(ENGINES.keys()), default="hf")
    args, remaining = parser.parse_known_args()

    engine_parser = argparse.ArgumentParser(
        description=f"Run MinerU-Diffusion inference with the {args.engine} engine."
    )
    engine_parser.add_argument("--engine", choices=sorted(ENGINES.keys()), default=args.engine)
    _load_engine(args.engine).add_arguments(engine_parser)
    return engine_parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    _load_engine(args.engine).run(args)


if __name__ == "__main__":
    main()
