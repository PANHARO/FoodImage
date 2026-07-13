"""Compatibility entry point for MobileNetV2 workflows.

Run from the project root:
    python trained_models/mobilenet.py --train
    python trained_models/mobilenet.py --evaluate
"""

try:
    from mobilenet.cli import main
except ImportError:
    from trained_models.mobilenet.cli import main


if __name__ == "__main__":
    main()
