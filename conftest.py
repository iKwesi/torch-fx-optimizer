"""
Root conftest.py to configure pytest for the entire project.

This adds the project root to sys.path so that imports work correctly.
"""

import sys
from pathlib import Path

# Add src directory to sys.path so torch_fx_optimizer package can be imported
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))
