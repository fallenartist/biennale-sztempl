# python/sztempl/config.py

import yaml
from pathlib import Path

# Locate config.yaml in the parent directory of this package
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Load once at import time
with open(CONFIG_PATH, "r") as f:
	settings = yaml.safe_load(f)
