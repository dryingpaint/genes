"""Find SOTA baselines for all evals and write to baselines.yaml.

Usage:
    python scripts/find_baselines.py
    python scripts/find_baselines.py --eval clinvar
"""

import argparse

from genbench.research.sota_finder import find_sota, update_baselines_yaml
from genbench.research.baselines_config import update_sota, load_baselines, BASELINES_PATH


def main():
    parser = argparse.ArgumentParser(description="Find SOTA baselines for evals")
    parser.add_argument("--eval", type=str, help="Specific eval to research")
    parser.add_argument("--no-search", action="store_true", help="Use known baselines only, skip web search")
    args = parser.parse_args()

    use_search = not args.no_search

    if args.eval:
        print(f"Finding SOTA for: {args.eval}")
        sota = find_sota(args.eval, use_search=use_search)
        update_sota(args.eval, sota)
        print(f"  Model: {sota.get('model', 'unknown')}")
        print(f"  Metric: {sota.get('metric', 'unknown')}")
        print(f"  Paper: {sota.get('paper', 'unknown')}")
        print(f"  Written to: {BASELINES_PATH}")
    else:
        print("Finding SOTA for all evals...")
        update_baselines_yaml(use_search=use_search)
        baselines = load_baselines()
        for name, info in sorted(baselines.items()):
            print(f"  {name}: {info.get('model', '?')} — {info.get('metric', '?')}")


if __name__ == "__main__":
    main()
