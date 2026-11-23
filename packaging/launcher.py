"""PyInstaller entry point. The frozen executable re-runs itself for the worker/server/download children."""

import multiprocessing
import sys

from noelle.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
