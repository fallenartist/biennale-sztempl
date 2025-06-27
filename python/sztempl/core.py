import logging
from sztempl.config import settings
from sztempl.plugins import load_plugins

def setup_logging():
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
	)

class AppContext:
	def __init__(self):
		self.modules = []

	def register_module(self, module):
		self.modules.append(module)

	def run(self):
		for mod in self.modules:
			if hasattr(mod, "start"):
				mod.start()
		try:
			# keep the main thread alive
			while True:
				pass
		except KeyboardInterrupt:
			for mod in self.modules:
				if hasattr(mod, "stop"):
					mod.stop()

def main():
	"""Console entry point for the `sztempl` command."""
	setup_logging()
	ctx = AppContext()
	load_plugins(ctx)
	ctx.run()

if __name__ == "__main__":
	main()
