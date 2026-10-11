"""Shared measurement, analysis and configuration modules of the CPU inference energy study."""
import os

# Repository root (the parent of src/); every data path in the project is resolved against it,
# regardless of the working directory the scripts are started from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
