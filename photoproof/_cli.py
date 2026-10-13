"""
photoproof/_cli.py
Thin entry point so `photoproof` console script launches `streamlit run app.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).parent.parent / "app.py"
    sys.exit(subprocess.call(["streamlit", "run", str(app_path)] + sys.argv[1:]))
