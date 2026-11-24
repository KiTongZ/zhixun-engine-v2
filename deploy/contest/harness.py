"""Copy of template/harness.py: the platform runs this file as describe/train/predict."""
import sys

from contest_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
