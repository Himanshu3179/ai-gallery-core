import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CURRENT_DIR = Path(__file__).resolve().parent
from bulk_processing.hybrid_search import HybridSearchEngine, print_search_results, interactive_mode


def main():
    parser = argparse.ArgumentParser(
        description="Unified Gallery Search: Open-Vocabulary Vector + Dense Caption Narrative + Calibrated Tags"
    )
    parser.add_argument("--query", "-q", type=str, help="Search query (e.g. 'basketball', 'netflix', 'umbrella', 'red car')")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return (default: 5)")
    args = parser.parse_args()

    engine = HybridSearchEngine()

    if args.query:
        results, elapsed_ms = engine.search(args.query, top_k=args.top_k)
        print_search_results(results, args.query, elapsed_ms)
    else:
        interactive_mode(engine)


if __name__ == "__main__":
    main()
