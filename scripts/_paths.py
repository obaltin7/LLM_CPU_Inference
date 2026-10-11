"""Make the src/ package importable when a script is run directly.

Importing this module from a script in scripts/ puts <repo>/src on sys.path, so the scripts work
from a plain checkout (``python scripts/run_campaign.py``) without ``pip install -e .``.
"""
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
