"""Compatibility entry point for model training.

New code should import reusable helpers from ``training`` or
``training.train``. This file keeps the historical ``python train.py`` command
working.
"""

from training.train import *  # noqa: F401,F403
from training.train import main


if __name__ == "__main__":
    main()
