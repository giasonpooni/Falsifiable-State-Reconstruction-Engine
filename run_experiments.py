"""Thin CLI for the Phase 1 grid.

    python run_experiments.py      -> prints markdown tables, writes results/ next to this file

The grid itself lives in set_lcm.experiments.phase1 so tests import it from the package.
"""
from __future__ import annotations

import sys
from pathlib import Path

from set_lcm.experiments.phase1 import main

if __name__ == "__main__":
    sys.exit(main(Path(__file__).parent / "results"))
