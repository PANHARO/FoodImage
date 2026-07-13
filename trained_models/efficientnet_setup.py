"""Compatibility entry point for EfficientNet-B0 workflows.

Run from the project root:
    python trained_models/efficientnet_setup.py --train
    python trained_models/efficientnet_setup.py --evaluate
"""

try:
    from efficientnet.cli import main
except ImportError:
    from trained_models.efficientnet.cli import main


if __name__ == "__main__":
    main()
