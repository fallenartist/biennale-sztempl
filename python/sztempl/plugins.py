# artstream/plugins.py
import importlib
from sztempl.config import settings

ENABLED = settings.get("modules", [])

def load_plugins(app_context):
	for name in ENABLED:
		module = importlib.import_module(f"sztempl.{name}")
		module.register(app_context)