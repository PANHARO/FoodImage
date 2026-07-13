"""Command-line entry point for EfficientNet-B0 workflows."""

import argparse

try:
    from .test import run_evaluation
    from .train import run_training
except ImportError:  # Support running this file directly.
    from test import run_evaluation
    from train import run_training


def main():
    """Parse a single requested action."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train EfficientNet-B0")
    action.add_argument(
        "--evaluate", action="store_true", help="Evaluate best checkpoint"
    )
    args = parser.parse_args()
    if args.train:
        run_training()
    else:
        run_evaluation()


if __name__ == "__main__":
    main()
